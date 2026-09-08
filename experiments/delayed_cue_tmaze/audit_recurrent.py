"""Fixed-input e-prop versus unrolled surrogate-gradient delay sweep."""
import argparse
import copy
import json
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from experiments.delayed_cue_tmaze.train import config_from_args, parser as train_parser
from ghost.ghost_terminal_core import System, SurrogateSpike


def unrolled_forward(self, value):
    # Same LIF equations and reset derivative as production, but retain the
    # graph across decisions. This override exists only on the audit copy.
    if self.mem is None:
        mem = value.new_zeros(value.shape[0], self.hidden_dim)
        spk = torch.zeros_like(mem)
    else:
        mem, spk = self.mem, self.spk
    mask = 1-torch.eye(self.hidden_dim, device=value.device)
    features = []
    for _ in range(self.cfg.snn_ticks):
        current = self.input(value)+F.linear(spk, self.recurrent.weight*mask)+self.bias
        mem = self.decay*mem+current-spk
        spk = SurrogateSpike.apply(mem-1, self.cfg.surrogate_scale)
        features.append(torch.cat((mem, spk), -1))
    self.mem, self.spk = mem, spk
    self.last_output = torch.stack(features).mean(0)
    return self.last_output


def compare(seed, delay):
    cfg = config_from_args(train_parser().parse_args([
        "--preset", "historical-reward-eprop", "--worlds", "2", "--seed", str(seed)]))
    online = System(cfg, "separated", torch.device("cpu"), seed)
    reference = copy.deepcopy(online.strategizer)
    reference.core.forward = MethodType(unrolled_forward, reference.core)
    memory = torch.zeros(2, cfg.strategy_dim)
    scores = []
    forward_error = 0.
    # Opposite synthetic latent cues at the first decision, identical hidden
    # inputs thereafter. Not an environment performance experiment.
    for t in range(delay+1):
        latent = torch.zeros(2, cfg.latent_dim)
        if t == 0:
            latent[:, 0] = torch.tensor([-1., 1.])
        out, _, readout, desire, variance = online.strategy_and_action(latent, deterministic=True)
        ref = reference(latent, torch.zeros_like(online.feedback), deterministic=True,
                        previous_strategy=memory)
        memory = ref["strategy"]
        forward_error = max(forward_error, float((readout-memory).abs().max().detach()))
        actions = torch.tensor([0, 1])
        # Freeze desirability/variance paths, matching strategy e-prop's
        # actor-to-memory learning signal rather than outcome-head training.
        def score(strategy):
            logits = online.actor(latent, strategy, desire.detach(), variance.detach(),
                                  deterministic=True, exploration=0.)["logits"]
            probs = (1-cfg.exploration_rate)*logits.softmax(-1)+cfg.exploration_rate/3
            return probs.gather(1, actions[:, None]).log().squeeze(1)
        logp = score(readout)
        scores.append(score(memory))
        online.strategy_eprop.accumulate(out["proposal"], out["gate"],
            out["previous_strategy"], readout, logp, cfg.strategy_retention,
            learned_gate=cfg.learned_strategy_memory)
    objective = sum(cfg.strategy_trace_decay**(delay-t)*s.mean()
                    for t, s in enumerate(scores))
    named = dict(reference.named_parameters())
    names = {id(p): name for name, p in online.strategizer.named_parameters()}
    selected = [names[id(p)] for p in online.strategy_eprop.parameters]
    gradients = torch.autograd.grad(objective, [named[n] for n in selected], allow_unused=True)
    groups = {}
    for group in ("core", "non_core", "all"):
        pairs = [(trace.mean(0), torch.zeros_like(p) if grad is None else grad)
                 for name, p, trace, grad in zip(selected, online.strategy_eprop.parameters,
                     online.strategy_eprop.reward_traces, gradients)
                 if group == "all" or name.startswith("core.") == (group == "core")]
        a = torch.cat([x.flatten() for x, _ in pairs]).detach()
        b = torch.cat([y.flatten() for _, y in pairs]).detach()
        an, bn = float(a.norm()), float(b.norm())
        groups[group] = dict(eprop_norm=an, reference_norm=bn,
            cosine=float(F.cosine_similarity(a, b, dim=0)) if an>0 and bn>0 else None,
            norm_ratio=an/bn if bn>0 else None,
            relative_error=float((a-b).norm())/bn if bn>0 else None)
    return dict(seed=seed, delay=delay, forward_max_error=forward_error, groups=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--seed-list", help="comma-separated seeds; overrides --seed")
    parser.add_argument("--delays", default="0,1,4,8,16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    delays = [int(v) for v in args.delays.split(",")]
    if any(d<0 for d in delays):
        parser.error("delays must be nonnegative")
    try:
        seeds = [int(v) for v in args.seed_list.split(",")] if args.seed_list else [args.seed]
    except ValueError:
        parser.error("seed-list must contain comma-separated integers")
    if len(set(seeds)) != len(seeds):
        parser.error("seed-list must not contain duplicates")
    rows = []
    for seed in seeds:
        for delay in delays:
            print(f"Auditing seed={seed} delay={delay}", flush=True)
            rows.append(compare(seed, delay))
    report = dict(results=rows, scope="Synthetic latent cue, fixed actions, zero predictor feedback, frozen weights; unit positive terminal TD multiplies decayed action-score sum. Not a maze rollout or full-system gradient.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    for row in rows:
        print(json.dumps(row))
    return int(any(r["forward_max_error"]>1e-5 for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
