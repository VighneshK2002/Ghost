"""Summarise trusted local capture files; no model updates or replay."""
import argparse
import json
from pathlib import Path
import torch


def summarise(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    groups = {}
    for row in data["rows"]:
        if "interference" not in row:
            raise ValueError("Capture predates interference instrumentation; rerun with a fresh directory")
        for w, action in enumerate(row["action"].tolist()):
            label = row["oracle_actions"][w][action]
            kind = "fatal" if label["fatal"] else "optimal" if label["excess"] == 0 else "detour"
            key = (int(row["stage"])+1, int(row["cue"][w]), kind)
            entry = groups.setdefault(key, [])
            entry.append(dict(td=float(row["td"][w]), actor_logp_delta=float(row["actor_logp_delta"][w]),
                **{f"{name}_{metric}": row["interference"][name][metric][w]
                   for name in ("actor", "strategy") for metric in ("world_batch_cosine", "world_step_cosine")}))
    def average(values):
        values = [v for v in values if v is not None]
        return sum(values)/len(values) if values else None
    summaries = []
    for (stage, cue, kind), entries in sorted(groups.items()):
        summaries.append(dict(stage=stage, cue=cue, action_class=kind, n=len(entries),
            positive_td_fraction=sum(e["td"]>0 for e in entries)/len(entries),
            **{f"mean_{k}": average([e[k] for e in entries]) for k in entries[0]}))
    batch = {name: {metric: average([r["interference"][name][metric] for r in data["rows"]])
             for metric in ("cancellation_ratio", "same_cue_cosine", "opposite_cue_cosine", "batch_step_cosine")}
             for name in ("actor", "strategy")}
    return dict(seed=data["seed"], start=data["start"], batch=batch, oracle_groups=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("*.pt"))
    if not paths:
        parser.error("no captures found")
    results = sorted([summarise(p) for p in paths], key=lambda r:(r["seed"], r["start"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(captures=results, caveat="Oracle optimality is not advantage under the current policy. Positive TD on a detour is not automatically a critic bug. Eligibility vectors include history; pairwise cue labels describe current episodes. Cancellation and Adam alignment are descriptive, not causal evidence."), indent=2, allow_nan=False))
    for r in results:
        print(f"seed={r['seed']} start={r['start']} {r['batch']}")


if __name__ == "__main__":
    main()
