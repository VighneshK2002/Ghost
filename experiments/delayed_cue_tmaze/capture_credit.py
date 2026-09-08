"""Opt-in real-trajectory capture for offline credit analysis."""
from dataclasses import asdict
from pathlib import Path
import torch
from experiments.delayed_cue_tmaze.critic_diagnostics import oracle_distances, successor


def direction_metrics(vectors, cues):
    """Zero vectors have undefined cosine, not perfect agreement."""
    norms = vectors.norm(dim=1)
    batch = vectors.mean(0)
    bn = batch.norm()
    valid = norms > 1e-12
    unit = vectors / norms.clamp_min(1e-12)[:, None]
    pair = unit @ unit.T
    cues = torch.as_tensor(cues)
    upper = torch.triu(torch.ones(len(cues), len(cues), dtype=torch.bool), 1)
    mask = upper & valid[:, None] & valid[None, :]
    def mean(mask):
        return float(pair[mask].mean()) if mask.any() else None
    return dict(world_norm=norms.tolist(), batch_norm=float(bn),
        cancellation_ratio=float(bn/norms.mean()) if norms.mean()>1e-12 else None,
        world_batch_cosine=[float(unit[i] @ (batch/bn)) if valid[i] and bn>1e-12 else None for i in range(len(cues))],
        same_cue_cosine=mean(mask & (cues[:, None] == cues[None, :])),
        opposite_cue_cosine=mean(mask & (cues[:, None] != cues[None, :])))


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clone(v) for v in value]
    return value


class CreditCapture:
    def __init__(self, cfg, seed, condition):
        self.cfg, self.seed, self.condition = cfg, seed, condition
        self.next_start = 0
        self.payload = None
        self.oracle = None
        self.updates = {}

    def before(self, system, env, latent, completed):
        if self.payload is None and completed >= self.next_start:
            self.payload = dict(config=asdict(self.cfg), seed=self.seed,
                condition=self.condition, start=completed,
                weights={n: clone(getattr(system, n).state_dict())
                         for n in ("encoder", "strategizer", "actor", "predictor")},
                recurrent={n: clone(getattr(system, n).core.snapshot())
                           for n in ("strategizer", "predictor")},
                memory=clone(system.strategy_memory), rows=[])
        if self.payload is not None:
            if self.oracle is None:
                self.oracle = [oracle_distances(env, cue) for cue in (0, 1)]
            labels = []
            for w in range(self.cfg.worlds):
                state = (int(env.x[w]), int(env.y[w]), int(env.direction[w]))
                distances = self.oracle[int(env.cue[w])]
                distance = distances.get(state)
                remaining = int(env.episode_time_limit[w])-int(env.age[w])
                labels.append([dict(excess=(1+distances[successor(env, state, a)]-distance)
                    if successor(env, state, a) in distances and distance is not None else None,
                    fatal=distances.get(successor(env, state, a), float("inf")) > remaining-1)
                    for a in range(3)])
            self.payload["rows"].append(dict(latent=clone(latent),
                oracle_actions=labels,
                feedback=clone(system.feedback), cue=env.cue.copy(),
                x=env.x.copy(), y=env.y.copy(), direction=env.direction.copy(),
                age=env.age.copy(), episode=env.episode.copy(),
                stage=env.curriculum_stage, episode_stage=env.episode_stage.copy(),
                hallway_length=env.hallway_length.copy(), episode_time_limit=env.episode_time_limit.copy()))

    def pre_update(self, name, learner, td):
        if self.payload is None:
            return
        traces = learner.traces if name == "actor" else learner.reward_traces
        vectors = torch.cat([clone(t).reshape(self.cfg.worlds, -1) for t in traces], 1)
        vectors *= clone(td)[:, None]
        self.payload["rows"][-1].setdefault("interference", {})[name] = direction_metrics(vectors, self.payload["rows"][-1]["cue"])
        self.updates[name] = (vectors, [clone(p) for p in learner.parameters])

    def post_update(self, name, learner):
        if self.payload is None:
            return
        vectors, before = self.updates.pop(name)
        delta = torch.cat([(clone(p)-b).flatten() for p, b in zip(learner.parameters, before)])
        metrics = self.payload["rows"][-1]["interference"][name]
        batch = vectors.mean(0)
        dn = delta.norm()
        metrics["optimizer_step_norm"] = float(dn)
        metrics["batch_step_cosine"] = float(torch.nn.functional.cosine_similarity(batch, delta, dim=0)) if batch.norm()>1e-12 and dn>1e-12 else None
        metrics["world_step_cosine"] = [float(torch.nn.functional.cosine_similarity(v, delta, dim=0)) if v.norm()>1e-12 and dn>1e-12 else None for v in vectors]

    def after(self, system, action, reward, done, td, latent, strategy,
              desire, variance, logp):
        if self.payload is None:
            return
        self.payload["rows"][-1].update(action=clone(action), reward=clone(reward),
                                       done=clone(done), td=clone(td))
        # A separate stateless actor copy avoids touching training activations.
        with torch.random.fork_rng(devices=[]):
            actor = type(system.actor)(self.cfg, persistent=False).to(system.device)
        actor.load_state_dict(system.actor.state_dict())
        with torch.no_grad():
            logits = actor(latent.detach(), strategy.detach(), desire.detach(),
                           variance.detach(), deterministic=True, exploration=0.)["logits"]
            probabilities = (1-self.cfg.exploration_rate)*logits.softmax(-1)+self.cfg.exploration_rate/3
            after = probabilities.gather(1, action[:, None]).log().squeeze(1)
        self.payload["rows"][-1]["actor_logp_delta"] = clone(after-logp.detach())
    def end_step(self):
        if self.payload is not None and len(self.payload["rows"]) >= self.cfg.credit_capture_length:
            self.flush()

    def flush(self):
        if self.payload is None:
            return
        directory = Path(self.cfg.credit_capture_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.condition}_seed{self.seed}_t{self.payload['start']}.pt"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite capture: {path}")
        torch.save(self.payload, path)
        self.next_start = self.payload["start"]+self.cfg.credit_capture_every
        self.payload = None
