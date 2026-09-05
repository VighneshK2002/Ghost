"""Train Ghost on the canonical non-vision delayed-cue T-maze."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ghost.ghost_terminal_core import CONDITIONS, Config, evaluate, run_condition, save


PRESET_CURRENT = "current"
PRESET_HISTORICAL_REWARD_EPROP = "historical-reward-eprop"
PRESETS = (PRESET_CURRENT, PRESET_HISTORICAL_REWARD_EPROP)


def cue_schedule(value: str) -> tuple[tuple[int, float], ...]:
    try:
        schedule = tuple(
            (int(transition), float(probability))
            for item in value.split(",")
            for transition, probability in (item.split(":"),)
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cue schedule must look like 0:0.5,16384:0.8,32768:0.5"
        ) from error
    if not schedule or schedule[0][0] != 0:
        raise argparse.ArgumentTypeError("cue schedule must start at transition 0")
    if any(
        transition < 0 or not 0.0 <= probability <= 1.0
        for transition, probability in schedule
    ):
        raise argparse.ArgumentTypeError(
            "cue transitions must be non-negative and probabilities in [0, 1]"
        )
    if any(
        current[0] >= following[0]
        for current, following in zip(schedule, schedule[1:])
    ):
        raise argparse.ArgumentTypeError(
            "cue schedule transitions must be strictly increasing"
        )
    return schedule


def config_from_args(args: argparse.Namespace) -> Config:
    values = {
        "seed": args.seed,
        "device": args.device,
        "worlds": args.worlds,
        "transitions": args.transitions,
        "evaluation_episodes": args.evaluation_episodes,
        "report_every": args.report_every,
        "checkpoint": str(args.checkpoint),
        "persist_recurrent_state_across_episodes": args.persist_recurrent_state,
        "cue_probability_schedule": args.cue_schedule,
    }
    if args.preset == PRESET_HISTORICAL_REWARD_EPROP:
        values.update(
            encoder_learning_mode="reward_eprop",
            use_reward_adaln=False,
            use_actor_encoder_eprop=False,
            use_predictor_eprop=True,
            use_predictor_encoder_eprop=False,
            use_strategy_encoder_eprop=True,
            use_representation_critic=False,
            critic_encoder_weight=0.0,
            jepa_variance_weight=0.0,
            sigreg_weight=0.0,
            strategy_sigreg_weight=0.0,
        )
    return Config(**values)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--preset",
        choices=PRESETS,
        default=PRESET_CURRENT,
        help="model configuration preset; current preserves the existing CLI defaults",
    )
    result.add_argument("--seed", type=int, default=11, help="first training seed")
    result.add_argument("--seeds", type=int, default=1, help="number of consecutive seeds")
    result.add_argument("--condition", choices=CONDITIONS, default="separated")
    result.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    result.add_argument("--transitions", type=int, default=65_536)
    result.add_argument("--worlds", type=int, default=24)
    result.add_argument("--evaluation-episodes", type=int, default=192)
    result.add_argument("--report-every", type=int, default=4_096)
    result.add_argument(
        "--persist-recurrent-state",
        action="store_true",
        help=(
            "preserve recurrent activations and strategy/feedback memory across "
            "episode boundaries while resetting eligibility traces"
        ),
    )
    result.add_argument(
        "--cue-schedule",
        type=cue_schedule,
        default=(),
        metavar="TRANSITION:P_LEFT,...",
        help=(
            "deterministic training cue schedule, for example "
            "0:0.5,32768:0.8,49152:0.5; evaluation remains balanced"
        ),
    )
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
    config = config_from_args(args)
    print(
        f"preset={args.preset} "
        f"encoder_learning={config.encoder_learning_mode} "
        f"reward_adaln={config.use_reward_adaln} "
        f"actor_encoder_eprop={config.use_actor_encoder_eprop} "
        f"strategy_encoder_eprop={config.use_strategy_encoder_eprop} "
        f"recurrent_persistence="
        f"{config.persist_recurrent_state_across_episodes} "
        f"cue_schedule={config.cue_probability_schedule or 'legacy-random'}"
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
            predictor_shuffled = evaluate(
                result["system"], run_config, seed, "predictor_shuffle"
            )
            predictor_zero = evaluate(
                result["system"], run_config, seed, "predictor_zero"
            )
            row.update(
                evaluation_episodes_minimum=args.evaluation_episodes,
                evaluation_success_rate=real[0],
                evaluation_wrong_rate=real[1],
                evaluation_timeout_rate=max(0.0, 1.0 - real[0] - real[1]),
                shuffled_strategy_success_rate=shuffled[0],
                zero_strategy_success_rate=zero[0],
                shuffled_predictor_success_rate=predictor_shuffled[0],
                zero_predictor_success_rate=predictor_zero[0],
            )
        trained.append(result)
        summaries.append(row)
        print(json.dumps(row, sort_keys=True))

    save(args.checkpoint, config, trained)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(
            {
                "preset": args.preset,
                "config": asdict(config),
                "checkpoint": str(args.checkpoint),
                "runs": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"saved checkpoint: {args.checkpoint}")
    print(f"saved summary: {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
