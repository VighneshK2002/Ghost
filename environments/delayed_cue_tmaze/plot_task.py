"""Generate a schematic of the delayed-cue T-maze."""

from __future__ import annotations

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


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.set_xlim(-4.5, 4.5)
    axis.set_ylim(-4.6, 1.7)
    axis.set_aspect("equal")
    axis.axis("off")

    corridor = "#cbd5e1"
    axis.plot([0, 0], [-4, 0], color=corridor, linewidth=34, solid_capstyle="butt")
    axis.plot([-3.5, 3.5], [0, 0], color=corridor, linewidth=34, solid_capstyle="butt")
    axis.plot([0, 0], [-4, 0], color="#334155", linewidth=2)
    axis.plot([-3.5, 3.5], [0, 0], color="#334155", linewidth=2)

    axis.scatter([0], [-3.8], s=260, color="#2563eb", zorder=3)
    axis.text(0, -4.35, "start: LEFT / RIGHT cue visible", ha="center", fontsize=10)
    axis.annotate(
        "cue hidden during delay",
        xy=(0, -1.6),
        xytext=(1.0, -2.5),
        arrowprops={"arrowstyle": "->", "color": "#475569"},
        fontsize=10,
        color="#475569",
    )
    axis.scatter([0], [0], s=180, marker="D", color="#f59e0b", zorder=3)
    axis.text(0, 0.55, "identical junction observation", ha="center", fontsize=10)
    axis.scatter([-3.35, 3.35], [0, 0], s=250, marker="*", color="#16a34a", zorder=3)
    axis.text(-3.35, 0.55, "left goal", ha="center", fontsize=10)
    axis.text(3.35, 0.55, "right goal", ha="center", fontsize=10)
    axis.text(
        0,
        1.35,
        "Correct arm depends on the earlier cue, not the current observation",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    output = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "delayed_cue_tmaze"
        / "figures"
        / "task.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, transparent=False)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()
