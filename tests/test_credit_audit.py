from experiments.delayed_cue_tmaze.audit_credit import audit
from experiments.delayed_cue_tmaze.audit_recurrent import compare


def test_capture_preserves_training_and_replays(tmp_path):
    from dataclasses import replace
    import torch
    from ghost.ghost_terminal_core import run_condition
    from experiments.delayed_cue_tmaze.train import parser, config_from_args
    from experiments.delayed_cue_tmaze.replay_credit import replay
    cfg = config_from_args(parser().parse_args([
        "--preset", "historical-reward-eprop", "--worlds", "2",
        "--transitions", "8", "--report-every", "1000"]))
    normal = run_condition(cfg, "separated", 12, torch.device("cpu"))["system"]
    recorded = run_condition(replace(cfg, credit_capture_dir=str(tmp_path),
        credit_capture_length=4), "separated", 12, torch.device("cpu"))["system"]
    for name in ("encoder", "actor", "strategizer", "predictor"):
        for key, value in getattr(normal, name).state_dict().items():
            assert torch.equal(value, getattr(recorded, name).state_dict()[key])
    report = replay(next(tmp_path.glob("*.pt")))
    from experiments.delayed_cue_tmaze.audit_interference import summarise
    summary = summarise(next(tmp_path.glob("*.pt")))
    assert summary["oracle_groups"]
    from experiments.delayed_cue_tmaze.investigate_updates import analyse
    investigation = analyse(list(tmp_path.glob("*.pt")))
    assert sum(g["n"] for g in investigation["groups"]) == 8
    assert len(report["rows"]) == 4
    assert max(row["forward_error"] for row in report["rows"]) < 1e-6


def test_interference_opposite_directions():
    import torch
    from experiments.delayed_cue_tmaze.capture_credit import direction_metrics
    metrics = direction_metrics(torch.tensor([[1., 0.], [-1., 0.]]), [0, 1])
    assert metrics["cancellation_ratio"] == 0
    assert metrics["opposite_cue_cosine"] == -1
    assert metrics["world_batch_cosine"] == [None, None]


def test_update_classification():
    from experiments.delayed_cue_tmaze.investigate_updates import classify
    flags = classify(.1, -.02, .5, -.3)
    assert flags["positive_td_suppressed"]
    assert flags["batch_aligned_step_opposed"]
    assert not flags["batch_opposed"]
    flags = classify(0., 0., None, None)
    assert flags["near_zero_td"]
    assert not flags["suppressed"]
    assert not flags["step_opposed"]


def test_recurrent_cli_seed_matrix(monkeypatch, tmp_path):
    import json
    from experiments.delayed_cue_tmaze import audit_recurrent
    path = tmp_path / "matrix.json"
    monkeypatch.setattr("sys.argv", ["audit", "--seed-list", "12,13,15,16",
                                  "--delays", "0,4", "--output", str(path)])
    monkeypatch.setattr(audit_recurrent, "compare", lambda seed, delay:
                        dict(seed=seed, delay=delay, forward_max_error=0.0))
    assert audit_recurrent.main() == 0
    rows = json.loads(path.read_text())["results"]
    assert [(r["seed"], r["delay"]) for r in rows] == [
        (s, d) for s in (12, 13, 15, 16) for d in (0, 4)]


def test_unrolled_recurrent_forward_parity():
    for delay in (0, 2):
        result = compare(12, delay)
        assert result["forward_max_error"] < 1e-6
        assert result["groups"]["core"]["reference_norm"] > 0
        assert result["groups"]["all"]["cosine"] is not None


def test_baseline_credit_audit():
    report = audit(12)
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    assert not report["config"]["use_strategic_prediction_timer"]
    assert not report["config"]["adaptive_exploration"]
