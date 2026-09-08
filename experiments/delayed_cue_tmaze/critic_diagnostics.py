"""Read-only training diagnostics; oracle labels never enter learning."""
import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np


def successor(env, state, action):
    x, y, direction = state
    if action < 2:
        return x, y, (direction + (-1 if action == 0 else 1)) % 4
    dx, dy = env.DELTAS[direction]
    return (x+dx, y+dy, direction) if (x+dx, y+dy) in env.valid else state


def oracle_distances(env, cue):
    goal = env.left_goal if cue == 0 else env.right_goal
    reverse = {}
    for x, y in env.valid:
        if (x, y) in (env.left_goal, env.right_goal):
            continue
        for direction in range(4):
            state = (x, y, direction)
            for action in range(3):
                reverse.setdefault(successor(env, state, action), []).append(state)
    distances = {(*goal, direction): 0 for direction in range(4)}
    queue = deque(distances)
    while queue:
        node = queue.popleft()
        for previous in reverse.get(node, []):
            if previous not in distances:
                distances[previous] = distances[node]+1
                queue.append(previous)
    return distances


class CriticDiagnostics:
    def __init__(self, env, seed, condition):
        self.distances = [oracle_distances(env, cue) for cue in range(2)]
        self.rows = []
        self.pending = [[] for _ in range(env.cfg.worlds)]
        self.seed, self.condition = seed, condition

    def before(self, env, actions, values, logvars):
        for w, action in enumerate(actions):
            state = (int(env.x[w]), int(env.y[w]), int(env.direction[w]))
            cue = int(env.cue[w])
            distances = self.distances[cue]
            remaining = int(env.episode_time_limit[w])-int(env.age[w])
            after = successor(env, state, int(action))
            distance = distances.get(state)
            next_distance = distances.get(after)
            excess = None if next_distance is None or distance is None else 1+next_distance-distance
            row = dict(seed=self.seed, condition=self.condition, world=w,
                       episode=int(env.episode[w]), transition=env.completed_transitions+w+1,
                       stage=int(env.episode_stage[w])+1, cue=cue, age=int(env.age[w]),
                       hallway_length=int(env.hallway_length[w]),
                       x=state[0], y=state[1], direction=state[2], action=int(action),
                       remaining=remaining, oracle_distance=distance,
                       oracle_excess=excess, oracle_mistake=excess is None or excess > 0,
                       fatal_action=next_distance is None or next_distance > remaining-1,
                       value=float(values[w]), uncertainty=float(np.exp(logvars[w]/2)),
                       discounted_return=None, critic_error=None, censored=True)
            self.rows.append(row)
            self.pending[w].append(row)

    def after(self, rewards, done, gamma):
        for w, reward in enumerate(rewards):
            self.pending[w][-1]["reward"] = float(reward)
            if done[w]:
                total = 0.0
                for row in reversed(self.pending[w]):
                    total = row["reward"]+gamma*total
                    row.update(discounted_return=total, critic_error=row["value"]-total,
                               censored=False)
                self.pending[w].clear()

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.condition}_seed{self.seed}.json"
        path.write_text(json.dumps(self.rows, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input.read_text())
    complete = [r for r in rows if not r["censored"]]
    if not complete:
        parser.error("no completed episodes available")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    def correlation(x, y):
        return float(np.corrcoef(x, y)[0, 1]) if len(x)>1 and np.std(x)>0 and np.std(y)>0 else None
    report = {"completed_decisions": len(complete), "censored_decisions": len(rows)-len(complete), "groups": []}
    last_transition = max(r["transition"] for r in rows)
    for stage in sorted({r["stage"] for r in complete}):
        group = [r for r in complete if r["stage"] == stage]
        finite = [r for r in group if r["oracle_excess"] is not None]
        x = [r["oracle_excess"] for r in finite]
        y = [abs(r["critic_error"]) for r in finite]
        axes[0].scatter(x, y, s=5, alpha=.12, label=f"Stage {stage}")
        for phase in range(4):
            subset = [r for r in group if min(3, int(4*(r["transition"]-1)/last_transition)) == phase]
            if not subset:
                continue
            valid = [r for r in subset if r["oracle_excess"] is not None]
            report["groups"].append(dict(stage=stage, training_quarter=phase+1, n=len(subset),
                error_excess_r=correlation([r["oracle_excess"] for r in valid], [abs(r["critic_error"]) for r in valid]),
                uncertainty_mistake_r=correlation([r["uncertainty"] for r in subset], [r["oracle_mistake"] for r in subset])))
        chunks = np.array_split(group, min(20, len(group)))
        axes[1].plot([np.mean([r["transition"] for r in c]) for c in chunks],
                     [np.mean([abs(r["critic_error"]) for r in c]) for c in chunks], label=f"Stage {stage}")
        ordered = sorted(group, key=lambda r: r["uncertainty"])
        bins = np.array_split(ordered, min(8, len(ordered)))
        axes[2].plot([np.mean([r["uncertainty"] for r in c]) for c in bins],
                     [np.mean([r["oracle_mistake"] for r in c]) for c in bins], "o-", label=f"Stage {stage}")
    axes[0].set(xlabel="Excess shortest-path actions (fatal terminals excluded)", ylabel="Absolute realised-return prediction error")
    axes[1].set(xlabel="Training transitions", ylabel="Mean absolute critic error")
    axes[2].set(xlabel="Predicted return standard deviation", ylabel="Oracle action mistake rate")
    for ax in axes:
        ax.legend()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
