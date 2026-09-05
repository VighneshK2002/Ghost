import torch
import pytest

from experiments.delayed_cue_tmaze.train import config_from_args, parser
from ghost.ghost_terminal_core import Config, System


def make_system(persistent: bool) -> System:
    config = Config(
        worlds=2,
        persist_recurrent_state_across_episodes=persistent,
        use_predictor_eprop=False,
        use_strategy_encoder_eprop=False,
    )
    system = System(config, "separated", torch.device("cpu"), seed=11)
    for core in (system.strategizer.core, system.predictor.core, system.actor.core):
        core.initial(config.worlds, system.device)
        core.mem.fill_(1.0)
        core.spk.fill_(1.0)
    system.feedback.fill_(1.0)
    system.strategy_memory.fill_(1.0)
    system.desirability_memory.fill_(1.0)
    system.outcome_logvar_memory.fill_(1.0)
    for trace in system.actor_eprop.traces:
        trace.fill_(1.0)
    return system


def test_cli_flag_enables_cross_episode_recurrent_persistence() -> None:
    args = parser().parse_args(["--persist-recurrent-state"])

    config = config_from_args(args)

    assert config.persist_recurrent_state_across_episodes is True


def test_cli_parses_transition_indexed_cue_schedule() -> None:
    args = parser().parse_args(
        ["--cue-schedule", "0:0.5,16384:0.8,32768:0.5"]
    )

    config = config_from_args(args)

    assert config.cue_probability_schedule == (
        (0, 0.5),
        (16384, 0.8),
        (32768, 0.5),
    )


@pytest.mark.parametrize(
    "value",
    ("100:0.5", "0:1.2", "0:0.5,100:0.8,50:0.2", "not-a-schedule"),
)
def test_cli_rejects_invalid_cue_schedule(value: str) -> None:
    with pytest.raises(SystemExit):
        parser().parse_args(["--cue-schedule", value])


def test_default_episode_reset_clears_recurrent_and_learning_state() -> None:
    system = make_system(persistent=False)
    done = torch.tensor([True, False])

    system.reset(done)

    assert torch.count_nonzero(system.strategizer.core.mem[0]) == 0
    assert torch.count_nonzero(system.predictor.core.mem[0]) == 0
    assert torch.count_nonzero(system.actor.core.mem[0]) == 0
    assert torch.count_nonzero(system.strategy_memory[0]) == 0
    assert torch.count_nonzero(system.feedback[0]) == 0
    assert all(
        torch.count_nonzero(trace[0]) == 0 for trace in system.actor_eprop.traces
    )
    assert torch.all(system.strategy_memory[1] == 1)


def test_persistent_episode_reset_keeps_recurrent_but_clears_eligibility() -> None:
    system = make_system(persistent=True)
    done = torch.tensor([True, False])

    system.reset(done)

    assert torch.all(system.strategizer.core.mem[0] == 1)
    assert torch.all(system.predictor.core.mem[0] == 1)
    assert torch.all(system.actor.core.mem[0] == 1)
    assert torch.all(system.strategy_memory[0] == 1)
    assert torch.all(system.feedback[0] == 1)
    assert all(
        torch.count_nonzero(trace[0]) == 0 for trace in system.actor_eprop.traces
    )
    assert all(torch.all(trace[1] == 1) for trace in system.actor_eprop.traces)
