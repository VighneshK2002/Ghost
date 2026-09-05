"""Evaluate a strictly compatible Ghost checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ghost.checkpoint import (
    CheckpointCompatibilityError,
    available_runs,
    load_system,
    read_checkpoint,
)
from ghost.ghost_terminal_core import CONDITIONS, evaluate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--seed", type=int, help="evaluate one seed; default: every matching run")
    result.add_argument("--condition", choices=CONDITIONS, default="separated")
    result.add_argument("--episodes", type=int, default=192)
    result.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    result.add_argument(
        "--intervention",
        choices=(
            "real",
            "shuffle",
            "zero",
            "predictor_shuffle",
            "predictor_zero",
        ),
        default="real",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("episodes must be positive")
    try:
        payload = read_checkpoint(args.checkpoint)
    except (CheckpointCompatibilityError, FileNotFoundError) as error:
        raise SystemExit(f"checkpoint error: {error}") from error
    seeds = [args.seed] if args.seed is not None else [
        seed for condition, seed in available_runs(payload) if condition == args.condition
    ]
    if not seeds:
        raise SystemExit(f"no {args.condition!r} runs found in {args.checkpoint}")
    for seed in seeds:
        try:
            system, config, _ = load_system(
                args.checkpoint,
                args.condition,
                seed,
                torch.device(args.device),
                evaluation_episodes=args.episodes,
            )
        except (CheckpointCompatibilityError, KeyError) as error:
            raise SystemExit(f"checkpoint error: {error}") from error
        success, wrong = evaluate(system, config, seed, args.intervention)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "seed": seed,
                    "checkpoint": str(args.checkpoint),
                    "evaluation_episodes_minimum": args.episodes,
                    "intervention": args.intervention,
                    "success_rate": success,
                    "wrong_rate": wrong,
                    "timeout_rate": max(0.0, 1.0 - success - wrong),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
