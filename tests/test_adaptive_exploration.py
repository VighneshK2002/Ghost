import json

import numpy as np
import pytest
import torch

from experiments.delayed_cue_tmaze.train import main, parser
from ghost.ghost_terminal_core import (
    AdaptiveExploration,
    AdaptiveTimerArbitration,
    AdaptiveTimerStateMachine,
    Config,
    System,
)


@pytest.mark.parametrize("scope, expected", [("all", 0.2), ("feedback-only", 1.0)])
def test_timer_gate_scope_preserves_outcome_context(scope, expected) -> None:
    config = Config(worlds=2, use_strategic_prediction_timer=True,
                    timer_gate_scope=scope)
    system = System(config, "separated", torch.device("cpu"), seed=12)
    system.timer_influence = 0.2
    assert system.timer_credit_influence == pytest.approx(expected)
    inputs = []
    handle = system.strategizer.outcome_head.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0].detach().clone()))
    system.strategy_and_action(torch.zeros(2, config.latent_dim))
    handle.remove()
    timer_context = inputs[0][:, -len(config.prediction_timer_durations):]
    assert torch.allclose(timer_context.sum(-1), torch.full((2,), expected))
    assert system.timer_influence == 0.2


def test_cli_timer_gate_scope_config() -> None:
    from experiments.delayed_cue_tmaze.train import config_from_args

    assert config_from_args(parser().parse_args([])).timer_gate_scope == "all"
    args = parser().parse_args(["--timer-gate-scope", "feedback-only"])
    assert config_from_args(args).timer_gate_scope == "feedback-only"


def test_strategy_exploration_is_episode_held_and_memory_stays_clean() -> None:
    config = Config(worlds=2, adaptive_exploration=True,
                    adaptive_exploration_target="strategy", learned_strategy_memory=True)
    system = System(config, "separated", torch.device("cpu"), seed=12)
    calls = []
    handle = system.actor.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(kwargs["exploration"]),
        with_kwargs=True)
    latent = torch.zeros(2, config.latent_dim)
    strategy, _, readout, _, _ = system.strategy_and_action(latent, exploration=0.25)
    noise = system.strategy_noise.clone()
    assert torch.equal(system.strategy_memory, strategy["strategy"].detach())
    assert torch.allclose(readout, strategy["strategy"]+0.25*noise)
    _, _, readout, _, _ = system.strategy_and_action(latent, exploration=0.1)
    assert torch.equal(system.strategy_noise, noise)
    assert torch.allclose(readout, system.strategy_memory+0.1*noise)
    system.reset(torch.tensor([True, False]))
    assert system.strategy_noise_valid.tolist() == [False, True]
    system.strategy_and_action(latent, exploration=0.1)
    assert torch.equal(system.strategy_noise[1], noise[1])
    assert not torch.equal(system.strategy_noise[0], noise[0])
    _, _, readout, _, _ = system.strategy_and_action(latent, deterministic=True)
    assert torch.equal(readout.detach(), system.strategy_memory)
    assert calls == [0.0]*4
    handle.remove()


def test_adaptive_exploration_tracks_weaker_cue_success() -> None:
    config = Config(
        adaptive_exploration=True,
        adaptive_exploration_min=0.02,
        adaptive_exploration_max=0.25,
        adaptive_exploration_ema_decay=0.0,
        adaptive_exploration_power=2.0,
    )
    scheduler = AdaptiveExploration(config)

    assert scheduler.rate == pytest.approx(0.25)

    scheduler.update(
        np.array([True, True]),
        np.array([True, False]),
        np.array([0, 1]),
    )
    assert scheduler.success_ema.tolist() == [1.0, 0.0]
    assert scheduler.rate == pytest.approx(0.25)

    scheduler.update(
        np.array([False, True]),
        np.array([False, True]),
        np.array([0, 1]),
    )
    assert scheduler.success_ema.tolist() == [1.0, 1.0]
    assert scheduler.rate == pytest.approx(0.02)


def test_cli_parses_adaptive_exploration_parameters() -> None:
    args = parser().parse_args(
        [
            "--adaptive-exploration",
            "--adaptive-exploration-min", "0.03",
            "--adaptive-exploration-max", "0.30",
            "--adaptive-exploration-ema-decay", "0.95",
            "--adaptive-exploration-power", "1.5",
        ]
    )

    assert args.adaptive_exploration is True
    assert args.adaptive_exploration_min == pytest.approx(0.03)
    assert args.adaptive_exploration_max == pytest.approx(0.30)
    assert args.adaptive_exploration_ema_decay == pytest.approx(0.95)
    assert args.adaptive_exploration_power == pytest.approx(1.5)


def test_timer_arbitration_escalates_when_exploration_stalls() -> None:
    config = Config(
        adaptive_exploration=True,
        adaptive_exploration_ema_decay=0.0,
        adaptive_timer_arbitration=True,
        use_strategic_prediction_timer=True,
        adaptive_timer_min_influence=0.1,
        adaptive_timer_patience_episodes=2,
        adaptive_timer_min_improvement=0.05,
        adaptive_timer_gate_ema_decay=0.0,
    )
    exploration = AdaptiveExploration(config)
    arbitration = AdaptiveTimerArbitration(config, exploration)

    done = np.array([True, True])
    failures = np.array([False, False])
    cues = np.array([0, 1])
    exploration.update(done, failures, cues)
    arbitration.update(done)

    assert arbitration.stalled is True
    assert arbitration.rescue_latched is True
    assert arbitration.gate == pytest.approx(1.0)

    successes = np.array([True, True])
    exploration.update(done, successes, cues)
    arbitration.update(done)

    assert arbitration.stalled is False
    assert arbitration.rescue_latched is True
    assert arbitration.gate == pytest.approx(1.0)


def test_cli_parses_timer_arbitration_parameters() -> None:
    args = parser().parse_args(
        [
            "--adaptive-exploration",
            "--strategic-prediction-timer",
            "--adaptive-timer-arbitration",
            "--adaptive-timer-min-influence", "0.15",
            "--adaptive-timer-patience-episodes", "128",
            "--adaptive-timer-min-improvement", "0.03",
            "--adaptive-timer-exploration-threshold", "0.6",
            "--adaptive-timer-gate-ema-decay", "0.97",
        ]
    )

    assert args.adaptive_timer_arbitration is True
    assert args.adaptive_timer_min_influence == pytest.approx(0.15)
    assert args.adaptive_timer_patience_episodes == 128
    assert args.adaptive_timer_min_improvement == pytest.approx(0.03)
    assert args.adaptive_timer_exploration_threshold == pytest.approx(0.6)
    assert args.adaptive_timer_gate_ema_decay == pytest.approx(0.97)


@pytest.mark.parametrize(
    ("successes", "expected_state", "expected_latch"),
    [
        (np.array([True, True]), "balanced_progress", False),
        (np.array([False, False]), "timer_rescue", True),
        (np.array([True, False]), "timer_rescue", True),
    ],
)
def test_timer_state_machine_selects_from_balanced_evidence(
    successes: np.ndarray,
    expected_state: str,
    expected_latch: bool,
) -> None:
    config = Config(
        adaptive_exploration=True,
        adaptive_exploration_ema_decay=0.0,
        use_strategic_prediction_timer=True,
        adaptive_timer_state_machine=True,
        timer_controller_episodes_per_cue=1,
        adaptive_timer_gate_ema_decay=0.0,
    )
    exploration = AdaptiveExploration(config)
    controller = AdaptiveTimerStateMachine(config, exploration)
    done = np.array([True, True])
    wrong = np.array([False, False])
    cues = np.array([0, 1])

    exploration.update(done, successes, cues)
    controller.update(done, successes, wrong, cues, curriculum_stage=0)

    assert controller.state == expected_state
    assert controller.rescue_latched is expected_latch


def test_cli_parses_timer_state_machine_parameters() -> None:
    args = parser().parse_args(
        [
            "--adaptive-exploration",
            "--strategic-prediction-timer",
            "--adaptive-timer-state-machine",
            "--timer-controller-bootstrap-influence", "0.4",
            "--timer-controller-episodes-per-cue", "24",
            "--timer-controller-progress-threshold", "0.04",
            "--timer-controller-success-threshold", "0.12",
            "--timer-controller-imbalance-threshold", "0.25",
            "--timer-controller-timeout-threshold", "0.65",
            "--timer-controller-transition-hold-episodes", "128",
        ]
    )

    assert args.adaptive_timer_state_machine is True
    assert args.timer_controller_bootstrap_influence == pytest.approx(0.4)
    assert args.timer_controller_episodes_per_cue == 24
    assert args.timer_controller_progress_threshold == pytest.approx(0.04)
    assert args.timer_controller_success_threshold == pytest.approx(0.12)
    assert args.timer_controller_imbalance_threshold == pytest.approx(0.25)
    assert args.timer_controller_timeout_threshold == pytest.approx(0.65)
    assert args.timer_controller_transition_hold_episodes == 128


def test_factorial_cli_writes_four_arms(tmp_path) -> None:
    checkpoint = tmp_path / "factorial.pt"
    summary = tmp_path / "factorial.json"

    status = main(
        [
            "--factorial-exploration-timer",
            "--seed-list", "11",
            "--worlds", "2",
            "--transitions", "2",
            "--evaluation-episodes", "1",
            "--report-every", "2",
            "--skip-evaluation",
            "--checkpoint", str(checkpoint),
            "--summary", str(summary),
        ]
    )

    assert status == 0
    payload = json.loads(summary.read_text())
    assert {row["factorial_arm"] for row in payload["runs"]} == {
        "fixed_no_timer",
        "adaptive_no_timer",
        "fixed_timer",
        "adaptive_timer",
    }
    assert all(tmp_path.joinpath(path).exists() for path in (
        "factorial_fixed_no_timer.pt",
        "factorial_adaptive_no_timer.pt",
        "factorial_fixed_timer.pt",
        "factorial_adaptive_timer.pt",
    ))
