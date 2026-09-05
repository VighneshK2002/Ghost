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
# Reproduce the historical reward-eprop configuration (compute-intensive)
python -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --seed 11 --seeds 6 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/historical_reward_eprop_reproduction.pt \
  --summary results/delayed_cue_tmaze/historical_reproduction.json

# Run the current default configuration
python -m experiments.delayed_cue_tmaze.train

# Experimental cross-episode recurrent persistence
python -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --persist-recurrent-state \
  --seed 11 --seeds 6 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/recurrent_persistence.pt \
  --summary results/delayed_cue_tmaze/recurrent_persistence.json

# Controlled early left-cue bias, followed by balanced training
python -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --cue-schedule "0:0.8,16384:0.5" \
  --seed 11 --seeds 1 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/seed11_early_left.pt \
  --summary results/delayed_cue_tmaze/seed11_early_left.json

python -m experiments.delayed_cue_tmaze.evaluate \
  --checkpoint artifacts/delayed_cue_tmaze.pt --episodes 192 --seed 11
```

The full default run uses 24 parallel worlds, 65,536 transitions, one seed,
and at least 192 completed evaluation episodes. Use `--seeds 6` to train
consecutive seeds beginning at `--seed`; this is expensive.

The `historical-reward-eprop` preset selects the configuration used for the
saved six-seed result: reward-eprop encoder learning, reward-conditioned
AdaLN disabled, actor-to-encoder e-prop disabled, recurrent predictor e-prop
enabled, strategy-to-encoder e-prop enabled, and the representation critic
disabled. The resolved configuration is stored in both the summary and
checkpoint.

`--persist-recurrent-state` preserves the actor, strategizer, and predictor
recurrent activations plus strategy and feedback memory when an episode ends.
Eligibility traces remain episode-scoped. The option is disabled by default.

`--cue-schedule` accepts comma-separated `transition:probability_left` entries.
Scheduled training uses deterministic per-segment quotas, while cue order is
shuffled. The schedule must begin at transition zero. When omitted, the
historical random cue sampler is unchanged. Evaluation always uses its original
balanced random sampler.

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
evaluation time. `predictor_shuffle` and `predictor_zero` instead modify only
the predictor-derived feedback before it reaches the strategizer on the next
step. The code also contains the existing `stateless_strategizer` and
`actor_only` training controls. These controls are not reported as scientific
ablations until their own seeded runs are completed and recorded.

## Checkpoints

New checkpoints save model states for each `condition_seedN` run and the full
configuration. The evaluator uses strict loading. Historical artifacts that
fail this check are intentionally rejected instead of being partially loaded.
