"""Train Ghost on the canonical non-vision delayed-cue T-maze."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ghost.ghost_terminal_core import CONDITIONS, Config, evaluate, run_condition, save


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seed", type=int, default=11, help="first training seed")
    result.add_argument("--seeds", type=int, default=1, help="number of consecutive seeds")
    result.add_argument("--condition", choices=CONDITIONS, default="separated")
    result.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    result.add_argument("--transitions", type=int, default=65_536)
    result.add_argument("--worlds", type=int, default=24)
    result.add_argument("--evaluation-episodes", type=int, default=192)
    result.add_argument("--report-every", type=int, default=4_096)
    result.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/delayed_cue_tmaze.pt"),
    )
    result.add_argument(
        "--summary",
        type=Path,
        default=Path("results/delayed_cue_tmaze/latest_run.json"),
    )
    result.add_argument("--skip-evaluation", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if min(args.seeds, args.worlds, args.transitions, args.evaluation_episodes) < 1:
        raise SystemExit("seeds, worlds, transitions, and evaluation episodes must be positive")
    device = torch.device(args.device)
    config = Config(
        seed=args.seed,
        device=args.device,
        worlds=args.worlds,
        transitions=args.transitions,
        evaluation_episodes=args.evaluation_episodes,
        report_every=args.report_every,
        checkpoint=str(args.checkpoint),
    )
    trained = []
    summaries = []
    for seed in range(args.seed, args.seed + args.seeds):
        run_config = replace(config, seed=seed)
        result = run_condition(run_config, args.condition, seed, device)
        row = {
            "condition": args.condition,
            "seed": seed,
            "training_transitions": args.transitions,
            "training_episodes": result["episodes"],
            "training_success_rate": result["rate"],
            "training_wrong_rate": result["wrong"],
            "training_timeout_rate": result["timeout"],
        }
        if not args.skip_evaluation:
            real = evaluate(result["system"], run_config, seed, "real")
            shuffled = evaluate(result["system"], run_config, seed, "shuffle")
            zero = evaluate(result["system"], run_config, seed, "zero")
            row.update(
                evaluation_episodes_minimum=args.evaluation_episodes,
                evaluation_success_rate=real[0],
                evaluation_wrong_rate=real[1],
                evaluation_timeout_rate=max(0.0, 1.0 - real[0] - real[1]),
                shuffled_strategy_success_rate=shuffled[0],
                zero_strategy_success_rate=zero[0],
            )
        trained.append(result)
        summaries.append(row)
        print(json.dumps(row, sort_keys=True))

    save(args.checkpoint, config, trained)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {"config": asdict(config), "checkpoint": str(args.checkpoint), "runs": summaries},
            indent=2,
        )
        + "\n"
    )
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
