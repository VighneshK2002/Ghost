import json

import torch

from experiments.delayed_cue_tmaze.train import main, parser
from ghost.ghost_terminal_core import apply_training_predictor_feedback


def test_training_feedback_interventions_preserve_active_world_alignment() -> None:
    feedback = torch.tensor(
        [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]
    )
    active = torch.tensor([True, False, True, True])

    assert apply_training_predictor_feedback(
        feedback, "normal", active
    ) is feedback
    torch.testing.assert_close(
        apply_training_predictor_feedback(feedback, "zero", active),
        torch.zeros_like(feedback),
    )
    torch.testing.assert_close(
        apply_training_predictor_feedback(feedback, "shuffle", active),
        torch.tensor([[4.0, 40.0], [0.0, 0.0], [1.0, 10.0], [3.0, 30.0]]),
    )


def test_cli_parses_training_predictor_feedback() -> None:
    args = parser().parse_args(
        ["--training-predictor-feedback", "shuffle"]
    )

    assert args.training_predictor_feedback == "shuffle"


def test_feedback_comparison_cli_writes_three_arms(tmp_path) -> None:
    checkpoint = tmp_path / "feedback.pt"
    summary = tmp_path / "feedback.json"

    status = main(
        [
            "--compare-training-predictor-feedback",
            "--strategic-prediction-timer",
            "--seed-list", "16",
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
    assert payload["experiment"] == "training_predictor_feedback_comparison"
    assert {row["feedback_arm"] for row in payload["runs"]} == {
        "normal", "shuffle", "zero",
    }
    assert all(tmp_path.joinpath(path).exists() for path in (
        "feedback_normal.pt",
        "feedback_shuffle.pt",
        "feedback_zero.pt",
    ))
