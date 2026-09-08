"""Locate suppressed oracle actions in trusted local interference captures."""
import argparse
import json
from pathlib import Path
import math
import torch


def classify(td, delta, batch_cosine, step_cosine, td_threshold=0.001,
             delta_threshold=1e-6, cosine_threshold=0.01):
    suppressed = delta < -delta_threshold
    near_zero = abs(td) <= td_threshold
    # These describe accumulated, TD-weighted eligibility, NOT the current
    # action score gradient, so they cannot attribute the probability change.
    batch_opposed = batch_cosine is not None and batch_cosine < -cosine_threshold
    step_opposed = step_cosine is not None and step_cosine < -cosine_threshold
    reversal = (batch_cosine is not None and batch_cosine > cosine_threshold
                and step_opposed)
    return dict(suppressed=suppressed, near_zero_td=near_zero,
                batch_opposed=batch_opposed, step_opposed=step_opposed,
                batch_aligned_step_opposed=reversal,
                positive_td_suppressed=td>td_threshold and suppressed)


def analyse(paths, min_start=0, td_threshold=0.001, delta_threshold=1e-6,
            cosine_threshold=0.01):
    events = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["start"] < min_start:
            continue
        for step, row in enumerate(data["rows"]):
            if "interference" not in row:
                raise ValueError(f"{path}: requires interference captures")
            for world, action in enumerate(row["action"].tolist()):
                label = row["oracle_actions"][world][action]
                kind = "fatal" if label["fatal"] else "optimal" if label["excess"] == 0 else "detour"
                td, delta = float(row["td"][world]), float(row["actor_logp_delta"][world])
                actor = row["interference"]["actor"]
                batch, actual = actor["world_batch_cosine"][world], actor["world_step_cosine"][world]
                flags = classify(td, delta, batch, actual, td_threshold, delta_threshold, cosine_threshold)
                events.append(dict(seed=data["seed"], start=data["start"], step=step,
                    world=world, episode=int(row["episode"][world]), cue=int(row["cue"][world]),
                    stage=int(row["stage"])+1, x=int(row["x"][world]), y=int(row["y"][world]),
                    direction=int(row["direction"][world]), action=action, action_class=kind,
                    td=td, logp_delta=delta, world_batch_cosine=batch, world_step_cosine=actual,
                    batch_step_cosine=actor["batch_step_cosine"],
                    world_direction_norm=actor["world_norm"][world], **flags))
    groups = {}
    for e in events:
        key = (e["seed"], e["start"], e["stage"], e["cue"], e["action_class"])
        groups.setdefault(key, []).append(e)
    summaries = []
    for key, values in sorted(groups.items()):
        suppressed = [e for e in values if e["suppressed"]]
        summaries.append(dict(zip(("seed", "start", "stage", "cue", "action_class"), key),
            n=len(values), suppressed_n=len(suppressed),
            mean_td=sum(e["td"] for e in values)/len(values),
            mean_logp_delta=sum(e["logp_delta"] for e in values)/len(values),
            suppressed_patterns={flag:sum(e[flag] for e in suppressed) for flag in
                ("near_zero_td", "batch_opposed", "step_opposed", "batch_aligned_step_opposed", "positive_td_suppressed")}))
    return dict(thresholds=dict(td=td_threshold, logp_delta=delta_threshold, cosine=cosine_threshold),
        groups=summaries, suppressed_optimal_events=[e for e in events if e["action_class"]=="optimal" and e["suppressed"]],
        limitations=["Pattern counts overlap; denominators are suppressed_n.",
            "Cosines use TD-weighted historical eligibility, not current-action gradients.",
            "An aligned batch but opposed step is consistent with optimizer transformation, not proof momentum caused suppression.",
            "Near-zero current TD does not imply negligible eligibility-weighted credit.",
            "No optimizer counterfactuals or full gradients are recoverable from these captures.",
            "Actions in already time-infeasible states are labelled fatal; optimal means route-efficient and time-feasible."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-start", type=int, default=0)
    parser.add_argument("--td-threshold", type=float, default=.001)
    parser.add_argument("--delta-threshold", type=float, default=1e-6)
    parser.add_argument("--cosine-threshold", type=float, default=.01)
    args = parser.parse_args()
    if any(not math.isfinite(v) or v < 0 for v in (args.td_threshold, args.delta_threshold, args.cosine_threshold)):
        parser.error("thresholds must be finite and nonnegative")
    paths = sorted(args.directory.glob("*.pt"))
    if not paths:
        parser.error("no captures found")
    report = analyse(paths, args.min_start, args.td_threshold, args.delta_threshold, args.cosine_threshold)
    if not report["groups"]:
        parser.error("no captures match min-start")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    for group in report["groups"]:
        if group["action_class"] == "optimal":
            print(json.dumps(group))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
