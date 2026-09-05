import json
from pathlib import Path

from experiments.delayed_cue_tmaze.train import main
from ghost.checkpoint import available_runs, read_checkpoint


def test_short_training_loop_and_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "smoke.pt"
    summary = tmp_path / "smoke.json"
    exit_code = main(
        [
            "--seed",
            "11",
            "--transitions",
            "2",
            "--worlds",
            "2",
            "--evaluation-episodes",
            "2",
            "--report-every",
            "2",
            "--skip-evaluation",
            "--checkpoint",
            str(checkpoint),
            "--summary",
            str(summary),
        ]
    )
    assert exit_code == 0
    assert summary.exists()
    summary_payload = json.loads(summary.read_text())
    assert summary_payload["preset"] == "current"
    assert summary_payload["config"]["encoder_learning_mode"] == "cue_auxiliary"
    assert summary_payload["config"]["use_reward_adaln"] is True
    assert available_runs(read_checkpoint(checkpoint)) == [("separated", 11)]


def test_historical_reward_eprop_preset_is_saved(tmp_path: Path) -> None:
    checkpoint = tmp_path / "historical-smoke.pt"
    summary = tmp_path / "historical-smoke.json"
    exit_code = main(
        [
            "--preset",
            "historical-reward-eprop",
            "--seed",
            "11",
            "--transitions",
            "2",
            "--worlds",
            "2",
            "--evaluation-episodes",
            "2",
            "--report-every",
            "2",
            "--skip-evaluation",
            "--checkpoint",
            str(checkpoint),
            "--summary",
            str(summary),
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads(summary.read_text())
    checkpoint_payload = read_checkpoint(checkpoint)
    assert summary_payload["preset"] == "historical-reward-eprop"
    for payload in (summary_payload, checkpoint_payload):
        config = payload["config"]
        assert config["encoder_learning_mode"] == "reward_eprop"
        assert config["use_reward_adaln"] is False
        assert config["use_actor_encoder_eprop"] is False
        assert config["use_predictor_eprop"] is True
        assert config["use_strategy_encoder_eprop"] is True
        assert config["use_representation_critic"] is False
