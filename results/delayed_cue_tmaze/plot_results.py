"""Plot final seed-level values supported by saved experiment output."""

from __future__ import annotations

import csv
import os
import tempfile
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

HERE = Path(__file__).resolve().parent


def main() -> None:
    with (HERE / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    seeds = [int(row["seed"]) for row in rows]
    values = [float(row["success_rate_reported"]) for row in rows]
    colors = ["#2563eb" if value >= 0.97 else "#dc2626" for value in values]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars = axis.bar([str(seed) for seed in seeds], values, color=colors, width=0.68)
    axis.axhline(0.97, color="#0f172a", linestyle="--", linewidth=1.2)
    axis.set_ylim(0, 1.08)
    axis.set_xlabel("Training seed")
    axis.set_ylabel("Reported evaluation success rate")
    figure.suptitle("Delayed-cue T-maze — saved final evaluations", fontsize=16)
    axis.set_title(
        "Historical saved output · configured minimum 192 episodes/seed · checkpoints missing",
        fontsize=9,
        color="#475569",
        pad=10,
    )
    axis.text(0.72, 0.91, "97% reference", transform=axis.transAxes,
              ha="left", va="bottom", fontsize=9, color="#0f172a")
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{100 * value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    output = HERE / "figures" / "seed_results.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
