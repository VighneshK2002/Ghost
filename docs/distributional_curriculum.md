# Distributional delayed-cue T-maze curriculum

`BatchedTMaze` now draws a new hallway length independently for every resetting
world. The notebook contains the same canonical environment definition. No new
environment or observation inputs are introduced.

Length is the starting row and counts the corridor cells **including the junction**:
length 1 begins at the junction; length 7 requires six forward moves to reach it
in the default 9×9 maze. The cue remains independent of length. Position was
already observable and remains so; this change does not claim that the learned
policy necessarily uses memory.

Default probabilities in the 9×9 geometry (rounded for display):

| Stage (UI) | H=1 | H=2 | H=3 | H=4 | H=5 | H=6 | H=7 | Mean H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.800 | 0.150 | 0.050 | 0.000 | 0.000 | 0.000 | 0.000 | 1.250 |
| 2 | 0.583 | 0.200 | 0.133 | 0.083 | 0.000 | 0.000 | 0.000 | 1.717 |
| 3 | 0.367 | 0.250 | 0.217 | 0.167 | 0.000 | 0.000 | 0.000 | 2.183 |
| 4 | 0.150 | 0.300 | 0.300 | 0.250 | 0.000 | 0.000 | 0.000 | 2.650 |
| 5 | 0.110 | 0.223 | 0.250 | 0.250 | 0.083 | 0.083 | 0.000 | 3.223 |
| 6 | 0.070 | 0.147 | 0.200 | 0.250 | 0.167 | 0.167 | 0.000 | 3.797 |
| 7 | 0.030 | 0.070 | 0.150 | 0.250 | 0.250 | 0.250 | 0.000 | 4.370 |
| 8 | 0.027 | 0.057 | 0.117 | 0.200 | 0.233 | 0.250 | 0.117 | 4.773 |
| 9 | 0.023 | 0.043 | 0.083 | 0.150 | 0.217 | 0.250 | 0.233 | 5.177 |
| 10 | 0.020 | 0.030 | 0.050 | 0.100 | 0.200 | 0.250 | 0.350 | 5.580 |

The original four distributions are anchors at stages 1, 4, 7, and 10.
Two intermediate stages use convex blends at 1/3 and 2/3 of each anchor-to-anchor
shift. Each promotion therefore moves probability by one third of the former
shift. The first/final distributions and both-cue mastery requirements are
unchanged; additional stages can require more training episodes. Explicit custom
distributions are used as supplied, without automatic interpolation.

For smaller mazes, out-of-range default lengths merge into the farthest legal row.
For taller mazes, the final default length extends to the farthest row. Every stage
still varies; geometry must support at least two legal starting rows. Custom
`Config.curriculum_length_distributions` accepts a tuple of distributions, each
represented by `(length, mass)` pairs (or a dictionary). Positive masses are
normalized; invalid geometry, single-length stages, disjoint adjacent supports,
and shifts toward shorter lengths are rejected.

A separate seeded random stream draws lengths, preserving the original cue RNG
and scheduled cue quotas. The notebook's shuffled balanced cue pairs are unchanged.
Length sampling is independent per reset, rather than stratified or forced equal
across worlds.

Timeouts interpolate the existing four `curriculum_episode_limits` anchors by
sampled length, with a minimum sufficient for the shortest route plus slack.
`episode_time_limit`, `hallway_length`, and `episode_stage` are assigned per world
at reset. Promotion does not change these for worlds already running. Completed
old-stage episodes do not contribute to the new stage's mastery history.
Success still requires enough completed episodes and sufficient success for
**each cue**. Rewards and cue visibility rules are unchanged.

Evaluation (`curriculum=False`) now samples the final distribution without
progression, rather than always starting at the farthest row. Historical fixed-start
scores and new evaluation scores therefore measure different start distributions.

Notebook interval/final reports and canonical training progress report:

- `success_by_hallway_length`: counts, success, and left/right cue breakdowns;
- `mean_hallway_length`, `min_hallway_length`, `max_hallway_length`;
- `worst_cue_success` and `worst_length_success`;
- `hallway_distribution_by_stage`: configured probabilities, observed counts,
  and observed proportions for completed episodes in each reporting window.

Worst-length success includes only lengths with at least
`curriculum_min_episodes_per_length` samples (default 8), and is diagnostic only.
Unobserved groups and insufficient evidence remain `None`, not zero. Counts refer
to completed episodes, so a short reporting window can overrepresent short lengths.
No neural architecture, learning rule, or slowdown-controller behavior is changed.

## Gradual transitions between stages

After both-cue mastery, the next stage becomes the target. The actual sampling
probabilities interpolate linearly from the previous distribution to that target
over `curriculum_blend_episodes` completed episodes (default 64, summed across
worlds). At promotion the blend fraction is zero, so there is no immediate jump.
Set the notebook's blend-duration control to 0 for immediate switching.

Only new resets use the evolving probabilities. Running episodes retain their
lengths and timeouts. Episodes sampled during the blend do not count toward the
next mastery decision, even if they finish after blending ends. Evaluation starts
at the settled final distribution. No stage or blend signal enters observations.

The orange curriculum curve shows interpolation progress; the step curve shows
the target stage. Live reports include the blend fraction and actual sampling
probabilities. The per-stage probability table continues to describe target
(stage endpoint) distributions, rather than the intermediate blend.
