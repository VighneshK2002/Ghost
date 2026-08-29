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
    assert available_runs(read_checkpoint(checkpoint)) == [("separated", 11)]
