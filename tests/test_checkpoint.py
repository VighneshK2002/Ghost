from pathlib import Path

import pytest
import torch

from ghost import Config, System
from ghost.checkpoint import CheckpointCompatibilityError, load_system
from ghost.ghost_terminal_core import evaluate, save


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    config = Config(worlds=2, transitions=2, evaluation_episodes=2)
    system = System(config, "separated", torch.device("cpu"), seed=11)
    checkpoint = tmp_path / "roundtrip.pt"
    save(
        checkpoint,
        config,
        [{"condition": "separated", "seed": 11, "system": system}],
    )

    loaded, loaded_config, _ = load_system(
        checkpoint, "separated", 11, torch.device("cpu")
    )
    assert loaded_config.worlds == 2
    for expected, actual in zip(system.actor.parameters(), loaded.actor.parameters()):
        torch.testing.assert_close(expected, actual)


def test_incompatible_checkpoint_is_rejected(tmp_path: Path) -> None:
    config = Config(worlds=1)
    system = System(config, "separated", torch.device("cpu"), seed=11)
    state = {
        name: getattr(system, name).state_dict()
        for name in ("encoder", "strategizer", "actor", "predictor")
    }
    state["actor"] = dict(state["actor"])
    state["actor"]["head.weight"] = torch.zeros(1, 1)
    checkpoint = tmp_path / "incompatible.pt"
    torch.save(
        {
            "config": vars(config),
            "conditions": {"separated_seed11": state},
        },
        checkpoint,
    )

    with pytest.raises(CheckpointCompatibilityError):
        load_system(checkpoint, "separated", 11, torch.device("cpu"))


@pytest.mark.parametrize("intervention", ("predictor_shuffle", "predictor_zero"))
def test_predictor_feedback_interventions_run(intervention: str) -> None:
    config = Config(worlds=2, evaluation_episodes=2)
    system = System(config, "separated", torch.device("cpu"), seed=11)

    success, wrong = evaluate(system, config, 11, intervention)

    assert 0.0 <= success <= 1.0
    assert 0.0 <= wrong <= 1.0
