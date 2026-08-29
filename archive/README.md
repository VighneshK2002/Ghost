# Historical research archive

These files preserve experimental history but are not the public API.

- `legacy/ghost.py` and `legacy/ghost_v1.py`–`ghost_v5.py` are successive
  monolithic Marimo iterations with differing environments, predictors,
  critics, and consolidation experiments.
- `vision/` contains the separate visual-observation T-maze branch. It has
  unique functionality and is retained for future work, but it does not
  support the current non-vision headline result.
- `demos/spiking_visualization.py` is a standalone interactive SNN demo, not an
  automated test.
- `marimo/ghost_terminal.py` is the interactive dashboard for the canonical
  engine; it is optional and archived to keep the package surface focused.

Use `ghost/`, `environments/delayed_cue_tmaze/`, and
`experiments/delayed_cue_tmaze/` for the canonical implementation. The archive
is intentionally preserved in the repository as research provenance; Git
history retains earlier root locations and artifacts.
