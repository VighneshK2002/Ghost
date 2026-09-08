# Delayed-cue T-maze

## Real training segment credit capture

Investigate suppressed oracle-optimal actions using existing interference captures:

```bash
.venv/bin/python -m experiments.delayed_cue_tmaze.investigate_updates \
  results/delayed_cue_tmaze/interference_segments --min-start 32832 \
  --output results/delayed_cue_tmaze/update_investigation.json
```

Reports per-seed/segment/stage/cue/action-class counts and detailed suppressed
optimal events with position, episode, TD, probability change and alignment.
Optional `--td-threshold`, `--delta-threshold`, `--cosine-threshold` control
near-zero tolerances (defaults 0.001, 1e-6, 0.01). Pattern counts overlap and
refer to suppressed actions, not all actions. Omit `--min-start` for all captures.
This is a descriptive investigation: accumulated eligibility cosine is not the
current-action directional derivative. It cannot isolate momentum versus Adam
scaling or prove why the probability changed; that requires new counterfactual
instrumentation. No model is trained or modified.

New captures also contain per-action oracle excess/fatal labels and actor and
strategizer TD-weighted per-world direction summaries. `cancellation_ratio` is
the norm of the mean direction divided by mean individual norm (zero means full
cancellation). Same/opposite cue cosines omit zero vectors. World-to-batch and
world-to-actual-optimizer-step cosines distinguish aggregation from Adam effects.
These measure historical eligibility directions, not isolated current-action
gradients. Oracle route optimality is not policy advantage; a positive TD on a
detour does not itself establish a critic bug.

```bash
.venv/bin/python -m experiments.delayed_cue_tmaze.audit_interference PATH \
  --output results/delayed_cue_tmaze/interference_report.json
```

Use a fresh capture directory. Older captures lack the required measurements.
The summary groups decisions by seed, segment, curriculum stage, cue and oracle
action class; the raw files retain all per-decision values. Only load trusted
local captures.

Add `--credit-capture-dir PATH --credit-capture-every 8192
--credit-capture-length 16` to a separated, no-timer, non-adaptive baseline run.
Each seed records the first 16 decisions at the start and at approximately every
8192 transitions. Weights, recurrent state and clean strategy memory are saved
at segment start; latent inputs, feedback, actions, rewards, TD and position/cue
labels are recorded per decision. Capture refuses to overwrite existing files.

```bash
.venv/bin/python -m experiments.delayed_cue_tmaze.replay_credit PATH \
  --output results/delayed_cue_tmaze/real_credit_replay.json
```

Only load trusted local captures (PyTorch pickle). Replay freezes weights at
segment start, conditions on recorded latents/feedback/TD, and starts eligibility
at zero. It compares e-prop and unrolled surrogate gradients within this window;
it does not reproduce changing training weights or pre-window eligibility.
Forward parity is checked between replay branches, not against the original
changing-weight trajectory. Per-cue junction entries report the actual actor
optimizer step's change in selected-action log probability at fixed inputs;
they do not measure the combined strategy/critic update or oracle correctness.

## Baseline numerical credit audit

For the recurrent surrogate-gradient comparison:

Use `--seed-list 12,13,15,16` instead of `--seed 12` to audit all four
initializations at the same delays in one JSON report. Progress prints before
each comparison. No training or checkpoint changes occur.

```bash
.venv/bin/python -m experiments.delayed_cue_tmaze.audit_recurrent \
  --seed 12 --delays 0,1,4,8,16 \
  --output results/delayed_cue_tmaze/recurrent_credit_seed12.json
```

Uses a synthetic opposite latent cue followed by identical zero inputs, fixed
actions and zero predictor feedback at frozen initial weights. The reference
retains both LIF and strategy-memory graphs across decisions. Both sides use
the same decayed sum of action log probabilities, corresponding to a unit
positive terminal TD signal. Actor outcome conditioning is detached on both
sides. Reports forward parity, gradient cosine, norm ratio and relative error
for core/non-core/all strategizer parameters. Only forward mismatch fails the
CLI; approximate gradient disagreement is a measurement, not a pass/fail rule.
This is not a maze rollout, a trained-checkpoint audit, or a full-system gradient.

```bash
.venv/bin/python -m experiments.delayed_cue_tmaze.audit_credit \
  --seed 12 --output results/delayed_cue_tmaze/baseline_credit_audit.json
```

Uses `historical-reward-eprop`, two worlds, no timer and no adaptive exploration
(ordinary fixed exploration remains). Runs analytic score-trace, actor autograd,
signed actor-update, first-step strategy-head and reset checks, not training.
The JSON includes exact configuration and tolerances. A nonzero exit indicates
a failed check. The extended audit checks the production external-memory
recursion against a four-step unrolled synthetic reference, observes TD at forced
timeouts in the real training loop, checks blocked feedback at decision/bootstrap
calls, and compares the first predictor update with feedback enabled/blocked.
It does not validate full recurrent SNN derivatives against BPTT or separately
exercise goal terminals. Passing is not a robustness claim.

## Adaptive strategy exploration

For per-decision critic/oracle logging, add `--critic-diagnostics-dir PATH` to
training. Each seed writes `separated_seed12.json` (condition/seed-specific).
Plot with `python -m experiments.delayed_cue_tmaze.critic_diagnostics INPUT.json
--output plot.png` (on one command line). This also writes correlation summaries
to `plot.json`. Use a fresh directory for each experiment.

Oracle excess is `1 + distance_after - distance_before`, using shortest paths
over position and orientation and treating both goals as terminal. Wrong-goal
actions have undefined excess and are excluded from the numeric excess scatter;
they remain in the binary mistake analysis. Fatal/time-budget labels are saved
separately. Critic error is predicted value minus realised discounted return
under the evolving training policy, not TD error or optimal-return error.
Incomplete episodes are censored. Correlations by stage/training quarter are
descriptive, not independent-sample significance tests or proof of calibration.

Add `--adaptive-exploration --adaptive-exploration-target strategy` to move the
adaptive strength from uniform actor-action mixing to additive strategy-readout
noise. Each world samples a uniform vector in [-1, 1] at its first training
decision and holds it until episode reset; the current success-EMA strength
multiplies that vector. Defaults give a per-coordinate amplitude of 0.02–0.25,
not an action probability or a matched exploration budget. Readouts may extend
beyond [-1, 1]; there is no clipping, preserving the additive path's derivative.
The actor and predictor receive the same noisy readout, while stored strategy
memory remains clean. Actor policy sampling remains enabled, but extra uniform
action mixing is zero. Evaluation adds no strategy noise. The default target
`actor` preserves prior runs; use new checkpoint and summary paths for comparison.

This is the canonical non-vision Ghost experiment.

## Task

The agent starts in a T-shaped maze and receives a left/right cue during its
first two decisions. The cue then becomes an explicit `hidden` observation.
At the later junction, the physical observation is the same for both cue
conditions, so the current observation alone cannot select the correct arm.
Correct and wrong goals give +1 and -1; a pure timeout gives -0.1.

Training uses ten overlapping distributions over hallway lengths, shifting
probability toward longer delays while retaining shorter-length rehearsal. Each
resetting world samples independently of its cue and keeps a length-specific
timeout for that episode. Promotion still requires mastery of both cues. After promotion, sampling
probabilities blend into the next target over 64 completed episodes by default;
mastery evidence resumes for episodes sampled from the settled target.
Evaluation disables progression and samples the final distribution. See
[distributional curriculum details](../../docs/distributional_curriculum.md) for
probabilities, length semantics, diagnostics, and the change in evaluation scope.

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

# Experimental strategy-controlled prediction horizon
python -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --strategic-prediction-timer \
  --prediction-timer-durations 1,2,4,8,16 \
  --seed 11 --seeds 1 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/seed11_strategic_timer.pt \
  --summary results/delayed_cue_tmaze/seed11_strategic_timer.json

# Repeat the strategic-timer experiment on an explicit seed cohort
python -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --strategic-prediction-timer \
  --prediction-timer-durations 1,2,4,8,16 \
  --cue-schedule "0:0.5" \
  --seed-list 12,13,15,16 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/collapsed_seeds_strategic_timer.pt \
  --summary results/delayed_cue_tmaze/collapsed_seeds_strategic_timer.json

# Matched 2x2 comparison: fixed/adaptive exploration x timer off/on
python -u -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --factorial-exploration-timer \
  --prediction-timer-durations 1,2,4,8,16 \
  --cue-schedule "0:0.5" \
  --seed-list 12,13,15,16 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/exploration_timer_factorial.pt \
  --summary results/delayed_cue_tmaze/exploration_timer_factorial.json

# Does aligned predictor feedback matter during acquisition?
python -u -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --strategic-prediction-timer \
  --compare-training-predictor-feedback \
  --prediction-timer-durations 1,2,4,8,16 \
  --cue-schedule "0:0.5" \
  --seed-list 16 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/seed16_training_feedback.pt \
  --summary results/delayed_cue_tmaze/seed16_training_feedback.json

# Let adaptive exploration hand off to the timer when progress stalls
python -u -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --adaptive-exploration --strategic-prediction-timer \
  --adaptive-timer-arbitration \
  --prediction-timer-durations 1,2,4,8,16 \
  --cue-schedule "0:0.5" \
  --seed-list 12,16 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/seed12_16_timer_arbitration.pt \
  --summary results/delayed_cue_tmaze/seed12_16_timer_arbitration.json

# Per-seed state-machine controller over all four diagnostic seeds
python -u -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --adaptive-exploration --strategic-prediction-timer \
  --adaptive-timer-state-machine \
  --adaptive-timer-gate-ema-decay 0.95 \
  --prediction-timer-durations 1,2,4,8,16 \
  --cue-schedule "0:0.5" \
  --seed-list 12,13,15,16 --worlds 24 --transitions 65536 \
  --report-every 4096 --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/four_seed_timer_controller.pt \
  --summary results/delayed_cue_tmaze/four_seed_timer_controller.json

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

`--strategic-prediction-timer` adds a categorical timer to each strategizer
decision. The shared predictor makes one latent forecast conditioned on the
selected duration, and that stored forecast is supervised only when its timer
expires. The origin inputs and detached predictor state are replayed at expiry,
so no temporal autograd graph is retained. Timer log-probability joins the
strategizer's TD-modulated eligibility trace, while prediction error updates
only the predictor. Forecasts that have not expired are cancelled at episode
boundaries. Logs report the mean selected duration, timer entropy, and counts
for each configured duration.

`--adaptive-exploration` replaces the fixed training exploration mixture with
a bounded schedule driven by an episodic success EMA for each cue. The weaker
cue controls the rate: `min + (max - min) * (1 - success_ema) ** power`.
Defaults are a 0.02 minimum, 0.25 maximum, 0.98 EMA decay, and power 2.0.
Evaluation still applies no forced exploration. `--factorial-exploration-timer`
runs fixed/no-timer, adaptive/no-timer, fixed/timer, and adaptive/timer arms for
the same seeds. It writes four checkpoints by suffixing the supplied checkpoint
name and records all arms in one JSON summary.

`--adaptive-timer-arbitration` requires both adaptive exploration and the
strategic timer. It initially limits timer feedback, timer-conditioned outcome
input, and timer TD credit while action exploration is high. Timer influence
rises as exploration falls, or after the weaker cue's success EMA fails to
improve for 256 completed episodes. The timed predictor continues training
underneath the gate. Logs and summaries report the timer weight, stagnation
state, rescue-latch state, and most recent weaker-cue improvement. Once
stagnation triggers the rescue latch, timer influence ramps to full and remains
there through later success changes and curriculum transitions.

`--adaptive-timer-state-machine` is a separate trajectory-dependent controller.
It starts with 0.5 timer influence and waits for 32 completed episodes from
each cue. Balanced progress selects an exploration-led path whose timer weight
follows inverse exploration. Stagnation, excessive cue imbalance, or a
timeout-dominated trajectory selects an early timer rescue and permanently
latches it. Curriculum promotions hold the current support for 256 episodes
before the next classification. The controller never receives the seed ID.

`--training-predictor-feedback` selects `normal`, `shuffle`, or `zero` feedback
during learning without changing predictor targets or optimization. Shuffling
is deterministic, operates only among active worlds, and does not consume the
actor's random-number stream. `--compare-training-predictor-feedback` runs all
three training modes on the same seeds and writes suffixed checkpoints. Each
model is still evaluated with normal, shuffled, and zeroed predictor feedback,
so the summary distinguishes effects on acquisition from effects on execution.

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
# Feedback-only timer arbitration

`--timer-gate-scope feedback-only` applies the controller weight only to predictor
feedback entering the strategizer. Timer policy credit and timer conditioning of
the outcome head stay at full strength; predictor training is unchanged. The
default `--timer-gate-scope all` preserves the previous three-path gating.
The scope is saved in configuration and used during evaluation too.

Matched to the four-seed state-machine run (new output paths):

```bash
.venv/bin/python -u -m experiments.delayed_cue_tmaze.train \
  --preset historical-reward-eprop --condition separated \
  --adaptive-exploration --strategic-prediction-timer \
  --adaptive-timer-state-machine --timer-gate-scope feedback-only \
  --adaptive-timer-gate-ema-decay 0.95 \
  --prediction-timer-durations 1,2,4,8,16 \
  --cue-schedule "0:0.5" --seed-list 12,13,15,16 \
  --worlds 24 --transitions 65536 --report-every 4096 \
  --evaluation-episodes 192 --device cpu \
  --checkpoint artifacts/four_seed_timer_controller_feedback_only.pt \
  --summary results/delayed_cue_tmaze/four_seed_timer_controller_feedback_only.json
```
