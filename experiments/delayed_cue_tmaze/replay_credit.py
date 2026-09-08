"""Offline frozen-weight credit replay of trusted local training captures."""
import argparse
import copy
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from ghost.ghost_terminal_core import Config, System
from experiments.delayed_cue_tmaze.audit_recurrent import unrolled_forward


def replay(path):
    # Only load captures you generated locally: these contain numpy/pickle data.
    data = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**data["config"])
    online = System(cfg, data["condition"], torch.device("cpu"), data["seed"])
    for name, weights in data["weights"].items():
        getattr(online, name).load_state_dict(weights)
    online.strategizer.core.restore(data["recurrent"]["strategizer"])
    online.strategy_memory = data["memory"].clone()
    reference = copy.deepcopy(online.strategizer)
    reference.core.forward = MethodType(unrolled_forward, reference.core)
    memory = data["memory"].clone()
    ref_score = torch.zeros(cfg.worlds)
    names = {id(p): n for n, p in online.strategizer.named_parameters()}
    selected = [names[id(p)] for p in online.strategy_eprop.parameters]
    named = dict(reference.named_parameters())
    results = []
    for t, row in enumerate(data["rows"]):
        latent = row["latent"]
        online.feedback = row["feedback"].clone()
        out, _, readout, desire, variance = online.strategy_and_action(latent, deterministic=True)
        ref = reference(latent, row["feedback"].clone(), deterministic=True, previous_strategy=memory)
        if cfg.learned_strategy_memory:
            memory = ref["strategy"]
        else:
            memory = cfg.strategy_retention*memory+(1-cfg.strategy_retention)*ref["strategy"]
        parity = float((readout-memory).abs().max().detach())
        def score(strategy):
            logits = online.actor(latent, strategy, desire.detach(), variance.detach(),
                                  deterministic=True, exploration=0.)["logits"]
            probs = (1-cfg.exploration_rate)*logits.softmax(-1)+cfg.exploration_rate/3
            return probs.gather(1, row["action"][:, None]).log().squeeze(1)
        logp = score(readout)
        ref_score = cfg.strategy_trace_decay*ref_score+score(memory)
        online.strategy_eprop.accumulate(out["proposal"], out["gate"],
            out["previous_strategy"], readout, logp, cfg.strategy_retention,
            learned_gate=cfg.learned_strategy_memory)
        gradients = torch.autograd.grad((ref_score*row["td"]).mean(),
            [named[n] for n in selected], retain_graph=True, allow_unused=True)
        approx, exact = [], []
        for p, trace, gradient in zip(online.strategy_eprop.parameters,
                                      online.strategy_eprop.reward_traces, gradients):
            view = (cfg.worlds,)+(1,)*(trace.ndim-1)
            approx.append((trace*row["td"].view(view)).mean(0).flatten())
            exact.append((torch.zeros_like(p) if gradient is None else gradient).flatten())
        a, b = torch.cat(approx).detach(), torch.cat(exact).detach()
        an, bn = float(a.norm()), float(b.norm())
        results.append(dict(step=t, stage=int(row["stage"])+1, forward_error=parity,
            cosine=float(F.cosine_similarity(a,b,dim=0)) if an and bn else None,
            norm_ratio=an/bn if bn else None,
            relative_error=float((a-b).norm())/bn if bn else None,
            # These deltas are from the actual actor optimizer step during training,
            # holding the readout and observation fixed, not from frozen replay.
            junction=[dict(world=w, cue=int(row["cue"][w]), td=float(row["td"][w]),
                           reward=float(row["reward"][w]), action=int(row["action"][w]),
                           logp_delta=float(row["actor_logp_delta"][w]))
                      for w in range(cfg.worlds) if row["y"][w] == 1]))
        live = (~row["done"]).float()
        online.reset(row["done"])
        memory = memory*live[:,None]
        ref_score = ref_score*live
        reference.core.mem = reference.core.mem*live[:,None]
        reference.core.spk = reference.core.spk*live[:,None]
    return dict(seed=data["seed"], start=data["start"], rows=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("*.pt"))
    if not paths:
        parser.error("no captures found")
    reports = []
    for path in paths:
        print(f"Replaying {path.name}", flush=True)
        reports.append(replay(path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(captures=reports, scope=
        "Frozen segment-start weights; recorded latents, feedback and TD treated as external constants. Eligibility starts at zero at segment boundary; recurrent state is restored. Not full-system BPTT or exact replay of changing training weights. Junction logp deltas measure the actual actor step at fixed inputs, not total strategy/critic updates."), indent=2, allow_nan=False))
    return int(any(r["forward_error"]>1e-5 for c in reports for r in c["rows"]))


if __name__ == "__main__":
    raise SystemExit(main())
