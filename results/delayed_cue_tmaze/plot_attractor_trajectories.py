"""Plot seed trajectories through the delayed-cue T-maze failure basins."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ghost-matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ghost-cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG = ROOT / "artifacts" / "historical_reward_eprop_reproduction.log"
DEFAULT_SUMMARY = Path(__file__).resolve().parent / "historical_reproduction.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "figures" / "attractor_trajectories.png"

RETENTION_RE = re.compile(
    r"strategy_cue_retention decode=\[cue:(?P<cue>[0-9.]+),"
    r"d1_4:(?P<d1>[0-9.]+),d5_8:(?P<d5>[0-9.]+),d9\+:(?P<d9>[0-9.]+)\]"
)
REPORT_RE = re.compile(
    r"condition=\S+\s+seed=(?P<seed>\d+) transitions=(?P<transitions>\d+)"
    r".*?curriculum=(?P<stage>\d+)/4"
    r".*?cue_success=\[(?P<left>[0-9.]+),(?P<right>[0-9.]+)\]"
    r".*?window_rate=(?P<window>[0-9.]+)"
)


@dataclass(frozen=True)
class Report:
    seed: int
    transitions: int
    stage: int
    left_success: float
    right_success: float
    window_success: float
    delayed_decode: float

    @property
    def balanced_success(self) -> float:
        return min(self.left_success, self.right_success)

    @property
    def cue_imbalance(self) -> float:
        return abs(self.left_success - self.right_success)


def parse_reports(path: Path) -> dict[int, list[Report]]:
    reports: dict[int, list[Report]] = {}
    pending_delayed_decode: float | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        retention = RETENTION_RE.search(line)
        if retention:
            pending_delayed_decode = float(retention.group("d9"))
            continue

        report = REPORT_RE.search(line)
        if not report:
            continue
        if pending_delayed_decode is None:
            raise ValueError(
                "Found a progress report without a preceding cue-retention report"
            )

        row = Report(
            seed=int(report.group("seed")),
            transitions=int(report.group("transitions")),
            stage=int(report.group("stage")),
            left_success=float(report.group("left")),
            right_success=float(report.group("right")),
            window_success=float(report.group("window")),
            delayed_decode=pending_delayed_decode,
        )
        reports.setdefault(row.seed, []).append(row)
        pending_delayed_decode = None

    if not reports:
        raise ValueError(f"No training progress reports found in {path}")
    return reports


def load_evaluation_rates(path: Path) -> dict[int, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(run["seed"]): float(run["evaluation_success_rate"])
        for run in payload["runs"]
    }


def outcome_kind(final: Report, evaluation_rate: float) -> str:
    if evaluation_rate >= 0.9 and final.balanced_success >= 0.75:
        return "escaped"
    if (
        final.balanced_success <= 0.1
        and max(final.left_success, final.right_success) >= 0.9
    ):
        return "one-cue attractor"
    if max(final.left_success, final.right_success) <= 0.1:
        return "collapsed"
    return "unstable / partial"


def build_figure(
    reports: dict[int, list[Report]], evaluation_rates: dict[int, float]
) -> plt.Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (phase_axis, gap_axis) = plt.subplots(1, 2, figsize=(12.6, 6.2))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.84, wspace=0.34)

    colors = {
        "escaped": "#16794b",
        "one-cue attractor": "#d97706",
        "collapsed": "#c2415d",
        "unstable / partial": "#2563eb",
    }
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]

    phase_axis.axvspan(0.75, 1.0, ymin=0.0, ymax=0.25, color=colors["escaped"], alpha=0.08)
    phase_axis.axvspan(0.0, 0.2, ymin=0.75, ymax=1.0, color=colors["one-cue attractor"], alpha=0.08)
    phase_axis.axvspan(0.0, 0.2, ymin=0.0, ymax=0.2, color=colors["collapsed"], alpha=0.06)
    gap_axis.axvspan(0.875, 1.0, ymin=0.75, ymax=1.0, color=colors["escaped"], alpha=0.08)
    gap_axis.axvspan(0.875, 1.0, ymin=0.0, ymax=0.2, color=colors["one-cue attractor"], alpha=0.08)

    for index, seed in enumerate(sorted(reports)):
        rows = reports[seed]
        evaluation_rate = evaluation_rates.get(seed, float("nan"))
        kind = outcome_kind(rows[-1], evaluation_rate)
        color = colors[kind]
        marker = markers[index % len(markers)]
        sizes = [14 + 22 * i / max(1, len(rows) - 1) for i in range(len(rows))]

        balanced = [row.balanced_success for row in rows]
        imbalance = [row.cue_imbalance for row in rows]
        delayed_decode = [row.delayed_decode for row in rows]

        phase_axis.plot(balanced, imbalance, color=color, linewidth=1.8, alpha=0.82)
        phase_axis.scatter(
            balanced,
            imbalance,
            s=sizes,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )
        gap_axis.plot(delayed_decode, balanced, color=color, linewidth=1.8, alpha=0.82)
        gap_axis.scatter(
            delayed_decode,
            balanced,
            s=sizes,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            zorder=3,
        )

        phase_offsets = {
            11: (-5, 8), 12: (9, 14), 13: (9, 3),
            14: (-5, -14), 15: (9, -14), 16: (8, 8),
        }
        gap_offsets = {
            11: (-5, 5), 12: (-5, 12), 13: (-5, 8),
            14: (-5, 10), 15: (-5, -14), 16: (6, 5),
        }
        for axis, x, y, offsets in (
            (phase_axis, balanced[-1], imbalance[-1], phase_offsets),
            (gap_axis, delayed_decode[-1], balanced[-1], gap_offsets),
        ):
            x_offset, y_offset = offsets[seed]
            alignment = "right" if x_offset < 0 else "left"
            axis.annotate(
                f"s{seed}",
                (x, y),
                xytext=(x_offset, y_offset),
                textcoords="offset points",
                ha=alignment,
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

    phase_axis.set(
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
        xlabel="Balanced cue success  min(left, right)",
        ylabel="Cue imbalance  |left − right|",
        title="Policy trajectory",
    )
    gap_axis.set(
        xlim=(0.45, 1.025),
        ylim=(-0.03, 1.03),
        xlabel="Delayed-cue decoding accuracy  (d9+)",
        ylabel="Balanced cue success  min(left, right)",
        title="Memory–control separation",
    )

    phase_axis.text(0.98, 0.08, "escaped basin", ha="right", color=colors["escaped"], fontsize=9)
    phase_axis.text(0.02, 0.91, "one-cue attractor", ha="left", color=colors["one-cue attractor"], fontsize=9)
    phase_axis.text(0.02, 0.06, "collapse", ha="left", color=colors["collapsed"], fontsize=9)
    gap_axis.text(0.995, 0.91, "memory + control", ha="right", color=colors["escaped"], fontsize=9)
    gap_axis.text(0.995, 0.06, "memory without control", ha="right", color=colors["one-cue attractor"], fontsize=9)

    for axis in (phase_axis, gap_axis):
        axis.set_aspect("equal", adjustable="box")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)

    legend_handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=2, label=label)
        for label, color in colors.items()
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        "Delayed-cue T-maze: seed-dependent learning trajectories",
        y=0.96,
        fontsize=15,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.018,
        "Measured every 4,104 transitions; marker size increases with training. "
        "Outcome labels use final evaluation and per-cue training success.",
        ha="center",
        fontsize=8.5,
        color="#475569",
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = parse_reports(args.log)
    evaluation_rates = load_evaluation_rates(args.summary)
    figure = build_figure(reports, evaluation_rates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()
