# Results still to reproduce

The saved Marimo output verifies that seeds 11–16 were evaluated in one run,
with success rates 0.979, 1.000, 1.000, 0.990, 0.000, and 0.985. It also shows
that checkpoint saving was disabled. The exact completed episode counts are not
logged; the configured minimum was 192 episodes per seed.

The tracked `online_delayed_cue_strategy_tmaze_gated_memory_v8.pt` is not a
substitute: it contains only seeds 11–13 and its state tensors are incompatible
with the available committed engine. Partial/non-strict loading would silently
change the model and is prohibited.

Before strengthening the headline claim:

1. run seeds 11–16 from a clean revision with the canonical 65,536-transition
   configuration;
2. save each compatible state plus exact package versions and Git commit;
3. log exact evaluation success, wrong, timeout, total episode, and per-cue
   counts (not only three-decimal rates);
4. repeat `real`, `shuffle`, and `zero` evaluation from identical episode seed
   streams;
5. publish the validated checkpoint bundle through a GitHub Release and place
   its URL and SHA-256 digest in the repository;
6. replace the historical-output CSV with freshly generated results and report
   mean, standard deviation, and every failed seed across a predeclared set.

Until then, the scientifically conservative statement is: **five saved runs
exceeded 97% evaluation success, and one saved run failed; exact checkpoint
reproduction is pending.**
