"""Train Ghost on the canonical non-vision delayed-cue T-maze."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch

from ghost.ghost_terminal_core import (
    CONDITIONS,
    TRAINING_PREDICTOR_FEEDBACK_MODES,
    Config,
    evaluate,
    run_condition,
    save,
)


PRESET_CURRENT = "current"
PRESET_HISTORICAL_REWARD_EPROP = "historical-reward-eprop"
PRESETS = (PRESET_CURRENT, PRESET_HISTORICAL_REWARD_EPROP)


def seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seed list must be comma-separated integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("seed list must not be empty")
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seed list must not contain duplicates")
    return seeds


def timer_durations(value: str) -> tuple[int, ...]:
    try:
        durations = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "timer durations must be comma-separated positive integers"
        ) from error
    if not durations or any(duration < 1 for duration in durations):
        raise argparse.ArgumentTypeError(
            "timer durations must be comma-separated positive integers"
        )
    if len(set(durations)) != len(durations):
        raise argparse.ArgumentTypeError("timer durations must be unique")
    return durations


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
        "use_strategic_prediction_timer": args.strategic_prediction_timer,
        "prediction_timer_durations": args.prediction_timer_durations,
        "adaptive_exploration": args.adaptive_exploration,
        "adaptive_exploration_target": args.adaptive_exploration_target,
        "critic_diagnostics_dir": args.critic_diagnostics_dir,
        "credit_capture_dir": args.credit_capture_dir,
        "credit_capture_every": args.credit_capture_every,
        "credit_capture_length": args.credit_capture_length,
        "adaptive_exploration_min": args.adaptive_exploration_min,
        "adaptive_exploration_max": args.adaptive_exploration_max,
        "adaptive_exploration_ema_decay": args.adaptive_exploration_ema_decay,
        "adaptive_exploration_power": args.adaptive_exploration_power,
        "adaptive_timer_arbitration": args.adaptive_timer_arbitration,
        "adaptive_timer_min_influence": args.adaptive_timer_min_influence,
        "adaptive_timer_patience_episodes": args.adaptive_timer_patience_episodes,
        "adaptive_timer_min_improvement": args.adaptive_timer_min_improvement,
        "adaptive_timer_exploration_threshold": (
            args.adaptive_timer_exploration_threshold
        ),
        "adaptive_timer_gate_ema_decay": args.adaptive_timer_gate_ema_decay,
        "adaptive_timer_state_machine": args.adaptive_timer_state_machine,
        "timer_gate_scope": args.timer_gate_scope,
        "timer_controller_bootstrap_influence": (
            args.timer_controller_bootstrap_influence
        ),
        "timer_controller_episodes_per_cue": (
            args.timer_controller_episodes_per_cue
        ),
        "timer_controller_progress_threshold": (
            args.timer_controller_progress_threshold
        ),
        "timer_controller_success_threshold": (
            args.timer_controller_success_threshold
        ),
        "timer_controller_imbalance_threshold": (
            args.timer_controller_imbalance_threshold
        ),
        "timer_controller_timeout_threshold": (
            args.timer_controller_timeout_threshold
        ),
        "timer_controller_transition_hold_episodes": (
            args.timer_controller_transition_hold_episodes
        ),
        "training_predictor_feedback": args.training_predictor_feedback,
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
    result.add_argument(
        "--seed-list",
        type=seed_list,
        default=(),
        metavar="SEED,...",
        help="explicit training seeds; overrides --seed and --seeds",
    )
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
        "--strategic-prediction-timer",
        action="store_true",
        help=(
            "let the strategizer select when each shared-predictor latent "
            "forecast is supervised"
        ),
    )
    result.add_argument(
        "--prediction-timer-durations",
        type=timer_durations,
        default=Config().prediction_timer_durations,
        metavar="STEPS,...",
        help="allowed strategic prediction horizons (default: 1,2,4,8,16)",
    )
    result.add_argument(
        "--adaptive-exploration",
        action="store_true",
        help="adapt training exploration from the weaker cue's success EMA",
    )
    result.add_argument("--adaptive-exploration-min", type=float, default=0.02)
    result.add_argument("--adaptive-exploration-max", type=float, default=0.25)
    result.add_argument(
        "--adaptive-exploration-ema-decay", type=float, default=0.98
    )
    result.add_argument("--adaptive-exploration-power", type=float, default=2.0)
    result.add_argument(
        "--adaptive-timer-arbitration",
        action="store_true",
        help=(
            "reduce timer influence while action exploration is effective, "
            "then promote it when weaker-cue success stalls"
        ),
    )
    result.add_argument("--adaptive-timer-min-influence", type=float, default=0.10)
    result.add_argument(
        "--adaptive-timer-patience-episodes", type=int, default=256
    )
    result.add_argument(
        "--adaptive-timer-min-improvement", type=float, default=0.02
    )
    result.add_argument(
        "--adaptive-timer-exploration-threshold", type=float, default=0.50
    )
    result.add_argument(
        "--adaptive-timer-gate-ema-decay", type=float, default=0.98
    )
    result.add_argument(
        "--adaptive-timer-state-machine",
        action="store_true",
        help=(
            "select an exploration-led or early latched-timer path from "
            "balanced per-cue training evidence"
        ),
    )
    result.add_argument(
        "--credit-capture-dir", default="",
    )
    result.add_argument("--credit-capture-every", type=int, default=8192)
    result.add_argument("--credit-capture-length", type=int, default=16)
    result.add_argument(
        "--critic-diagnostics-dir", default="",
        help="save per-decision oracle and realised-return diagnostics per seed",
    )
    result.add_argument(
        "--adaptive-exploration-target", choices=("actor", "strategy"), default="actor",
        help="apply adaptive strength to actor mixing or episode-held strategy readout noise",
    )
    result.add_argument(
        "--timer-gate-scope", choices=("all", "feedback-only"), default="all",
        help="gate all three timer paths (legacy) or only predictor feedback",
    )
    result.add_argument(
        "--timer-controller-bootstrap-influence", type=float, default=0.50
    )
    result.add_argument(
        "--timer-controller-episodes-per-cue", type=int, default=32
    )
    result.add_argument(
        "--timer-controller-progress-threshold", type=float, default=0.05
    )
    result.add_argument(
        "--timer-controller-success-threshold", type=float, default=0.10
    )
    result.add_argument(
        "--timer-controller-imbalance-threshold", type=float, default=0.30
    )
    result.add_argument(
        "--timer-controller-timeout-threshold", type=float, default=0.70
    )
    result.add_argument(
        "--timer-controller-transition-hold-episodes", type=int, default=256
    )
    result.add_argument(
        "--training-predictor-feedback",
        choices=TRAINING_PREDICTOR_FEEDBACK_MODES,
        default="normal",
        help=(
            "intervene on predictor feedback before it reaches the "
            "strategizer during training"
        ),
    )
    result.add_argument(
        "--compare-training-predictor-feedback",
        action="store_true",
        help=(
            "run matched normal/shuffle/zero training-feedback arms and "
            "write one checkpoint per arm"
        ),
    )
    result.add_argument(
        "--factorial-exploration-timer",
        action="store_true",
        help=(
            "run fixed/adaptive exploration crossed with timer off/on; "
            "writes one checkpoint per arm"
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


def summarize_run(result, args: argparse.Namespace, run_config: Config,
                  seed: int) -> dict:
    row = {
        "condition": args.condition,
        "seed": seed,
        "training_transitions": args.transitions,
        "training_episodes": result["episodes"],
        "training_success_rate": result["rate"],
        "training_wrong_rate": result["wrong"],
        "training_timeout_rate": result["timeout"],
        "final_exploration_rate": result["final_exploration_rate"],
        "final_success_ema": result["final_success_ema"],
        "final_timer_influence": result["final_timer_influence"],
        "timer_arbitration_stalled": result["timer_arbitration_stalled"],
        "timer_rescue_latched": result["timer_rescue_latched"],
        "timer_controller_state": result["timer_controller_state"],
        "timer_controller_timeout_ema": result[
            "timer_controller_timeout_ema"
        ],
        "timer_arbitration_improvement": result[
            "timer_arbitration_improvement"
        ],
        "training_predictor_feedback": run_config.training_predictor_feedback,
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
            evaluation_timeout_rate=max(0.0, 1.0-real[0]-real[1]),
            shuffled_strategy_success_rate=shuffled[0],
            zero_strategy_success_rate=zero[0],
            shuffled_predictor_success_rate=predictor_shuffled[0],
            zero_predictor_success_rate=predictor_zero[0],
        )
    return row


def arm_checkpoint(path: Path, arm: str) -> Path:
    return path.with_name(f"{path.stem}_{arm}{path.suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if min(args.seeds, args.worlds, args.transitions, args.evaluation_episodes) < 1:
        raise SystemExit("seeds, worlds, transitions, and evaluation episodes must be positive")
    if args.adaptive_timer_arbitration and not (
        args.adaptive_exploration and args.strategic_prediction_timer
    ):
        raise SystemExit(
            "--adaptive-timer-arbitration requires --adaptive-exploration "
            "and --strategic-prediction-timer"
        )
    if args.adaptive_timer_state_machine and not (
        args.adaptive_exploration and args.strategic_prediction_timer
    ):
        raise SystemExit(
            "--adaptive-timer-state-machine requires --adaptive-exploration "
            "and --strategic-prediction-timer"
        )
    if args.adaptive_timer_state_machine and args.adaptive_timer_arbitration:
        raise SystemExit(
            "choose either --adaptive-timer-state-machine or "
            "--adaptive-timer-arbitration"
        )
    if args.factorial_exploration_timer and (
        args.strategic_prediction_timer or args.adaptive_exploration
    ):
        raise SystemExit(
            "--factorial-exploration-timer sets timer and exploration modes; "
            "do not combine it with their individual flags"
        )
    if args.factorial_exploration_timer and args.compare_training_predictor_feedback:
        raise SystemExit(
            "choose either --factorial-exploration-timer or "
            "--compare-training-predictor-feedback"
        )
    if (
        args.compare_training_predictor_feedback
        and args.training_predictor_feedback != "normal"
    ):
        raise SystemExit(
            "--compare-training-predictor-feedback sets all feedback modes; "
            "do not combine it with --training-predictor-feedback"
        )
    selected_seeds = args.seed_list or tuple(range(args.seed, args.seed + args.seeds))
    device = torch.device(args.device)
    config = replace(config_from_args(args), seed=selected_seeds[0])
    print(
        f"preset={args.preset} "
        f"encoder_learning={config.encoder_learning_mode} "
        f"reward_adaln={config.use_reward_adaln} "
        f"actor_encoder_eprop={config.use_actor_encoder_eprop} "
        f"strategy_encoder_eprop={config.use_strategy_encoder_eprop} "
        f"recurrent_persistence="
        f"{config.persist_recurrent_state_across_episodes} "
        f"strategic_prediction_timer="
        f"{config.use_strategic_prediction_timer} "
        f"timer_durations={config.prediction_timer_durations} "
        f"adaptive_exploration={config.adaptive_exploration} "
        f"adaptive_exploration_target={config.adaptive_exploration_target} "
        f"adaptive_timer_arbitration={config.adaptive_timer_arbitration} "
        f"adaptive_timer_state_machine={config.adaptive_timer_state_machine} "
        f"timer_gate_scope={config.timer_gate_scope} "
        f"training_predictor_feedback="
        f"{config.training_predictor_feedback} "
        f"seeds={selected_seeds} "
        f"cue_schedule={config.cue_probability_schedule or 'legacy-random'}"
    )

    if args.compare_training_predictor_feedback:
        summaries = []
        checkpoints = {}
        arm_configs = {}
        for feedback_mode in TRAINING_PREDICTOR_FEEDBACK_MODES:
            checkpoint = arm_checkpoint(args.checkpoint, feedback_mode)
            run_config = replace(
                config,
                training_predictor_feedback=feedback_mode,
                checkpoint=str(checkpoint),
            )
            print(f"training_predictor_feedback_arm={feedback_mode}")
            trained = []
            for seed in selected_seeds:
                seed_config = replace(run_config, seed=seed)
                result = run_condition(
                    seed_config, args.condition, seed, device
                )
                row = summarize_run(result, args, seed_config, seed)
                row["feedback_arm"] = feedback_mode
                trained.append(result)
                summaries.append(row)
                print(json.dumps(row, sort_keys=True))
            save(checkpoint, run_config, trained)
            checkpoints[feedback_mode] = str(checkpoint)
            arm_configs[feedback_mode] = asdict(run_config)
            print(f"saved checkpoint: {checkpoint}")

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                {
                    "experiment": "training_predictor_feedback_comparison",
                    "preset": args.preset,
                    "seeds": list(selected_seeds),
                    "checkpoints": checkpoints,
                    "arm_configs": arm_configs,
                    "runs": summaries,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved summary: {args.summary}")
        return 0

    if args.factorial_exploration_timer:
        arms = (
            ("fixed_no_timer", False, False),
            ("adaptive_no_timer", False, True),
            ("fixed_timer", True, False),
            ("adaptive_timer", True, True),
        )
        summaries = []
        checkpoints = {}
        arm_configs = {}
        for arm, timer_enabled, adaptive_enabled in arms:
            checkpoint = arm_checkpoint(args.checkpoint, arm)
            run_config = replace(
                config,
                use_strategic_prediction_timer=timer_enabled,
                adaptive_exploration=adaptive_enabled,
                checkpoint=str(checkpoint),
            )
            print(
                f"factorial_arm={arm} timer={timer_enabled} "
                f"adaptive_exploration={adaptive_enabled}"
            )
            trained = []
            for seed in selected_seeds:
                seed_config = replace(run_config, seed=seed)
                result = run_condition(
                    seed_config, args.condition, seed, device
                )
                row = summarize_run(result, args, seed_config, seed)
                row.update(
                    factorial_arm=arm,
                    strategic_prediction_timer=timer_enabled,
                    adaptive_exploration=adaptive_enabled,
                )
                trained.append(result)
                summaries.append(row)
                print(json.dumps(row, sort_keys=True))
            save(checkpoint, run_config, trained)
            checkpoints[arm] = str(checkpoint)
            arm_configs[arm] = asdict(run_config)
            print(f"saved checkpoint: {checkpoint}")

        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(
                {
                    "experiment": "exploration_timer_factorial",
                    "preset": args.preset,
                    "seeds": list(selected_seeds),
                    "checkpoints": checkpoints,
                    "arm_configs": arm_configs,
                    "runs": summaries,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"saved summary: {args.summary}")
        return 0

    trained = []
    summaries = []
    for seed in selected_seeds:
        run_config = replace(config, seed=seed)
        result = run_condition(run_config, args.condition, seed, device)
        row = summarize_run(result, args, run_config, seed)
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
