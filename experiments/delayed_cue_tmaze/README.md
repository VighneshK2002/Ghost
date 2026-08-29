# Delayed-cue T-maze

This is the canonical non-vision Ghost experiment.

## Task

The agent starts in a T-shaped maze and receives a left/right cue during its
first two decisions. The cue then becomes an explicit `hidden` observation.
At the later junction, the physical observation is the same for both cue
conditions, so the current observation alone cannot select the correct arm.
Correct and wrong goals give +1 and -1; a pure timeout gives -0.1.

Training uses a four-stage start-row curriculum. Evaluation disables the
curriculum and always starts at the maximally delayed row.

## Commands

```bash
python -m experiments.delayed_cue_tmaze.train
python -m experiments.delayed_cue_tmaze.evaluate \
  --checkpoint artifacts/delayed_cue_tmaze.pt --episodes 192 --seed 11
```

The full default run uses 24 parallel worlds, 65,536 transitions, one seed,
and at least 192 completed evaluation episodes. Use `--seeds 6` to train
consecutive seeds beginning at `--seed`; this is expensive.

For an execution-only smoke test:

```bash
python -m experiments.delayed_cue_tmaze.train \
  --transitions 8 --worlds 2 --evaluation-episodes 4 --report-every 8 \
  --checkpoint artifacts/smoke.pt
```

## Metrics

The CLI records seed, configured training transitions, observed training
episodes, success/wrong/timeout rates, and evaluation strategy interventions.
The current evaluator operates on parallel worlds and stops after crossing the
requested episode minimum, so the field is named
`evaluation_episodes_minimum`. A future result logger should store exact counts
and per-cue outcomes rather than only rates.

`shuffle` and `zero` interventions modify the learned strategy input at
evaluation time. The code also contains the existing `stateless_strategizer`
and `actor_only` training controls. These controls are not reported as
scientific ablations until their own seeded runs are completed and recorded.

## Checkpoints

New checkpoints save model states for each `condition_seedN` run and the full
configuration. The evaluator uses strict loading. Historical artifacts that
fail this check are intentionally rejected instead of being partially loaded.
