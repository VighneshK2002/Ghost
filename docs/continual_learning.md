# Continual learning in Ghost

Ghost currently uses “continual learning” to mean that learning updates occur
inside the agent/environment interaction loop while neural and eligibility
state can persist from one decision to the next. It does not yet mean that the
system has passed standard continual-learning task-sequence or forgetting
benchmarks.

## Demonstrated in the implementation

- The environment loop updates enabled model parameters online after
  transitions rather than collecting a fixed offline dataset first.
- No replay buffer is used by the canonical delayed-cue experiment.
- Recurrent membrane/spike state and explicit strategy memory span decisions.
- Persistent recurrent state is detached between decisions, so the code does
  not run full episode-level backpropagation through time.
- Per-world eligibility and Jacobian traces carry selected temporal credit and
  are modulated by later prediction or TD signals.
- Episode completion resets neural, explicit-memory, and trace state for that
  world; learned weights and Adam optimizer state continue across episodes.
- A saved run demonstrates that several seeds can learn the single delayed-cue
  task to high evaluation success, while another saved seed fails.

The default `cue_auxiliary` encoder is updated with reconstruction and cue
classification on every transition. The predictor uses a per-transition
prediction objective plus explicit recurrent eligibility. Actor and strategy
parameters use reward/TD-modulated traces. The optional representation critic
and reward-conditioned JEPA paths have additional Adam/EMA updates when enabled.

## What is not demonstrated

- retention across a sequence of distinct tasks or changing data distributions;
- quantified catastrophic forgetting, forward transfer, or backward transfer;
- stable lifetime learning without episode resets;
- multi-context memory retrieval over long horizons;
- autonomous intent formation or causal world-model planning;
- a hardware-local learning rule for every conventional component;
- native neuromorphic execution or energy efficiency.

The current T-maze isolates temporal memory in one stationary task. It is a
useful mechanism test, not evidence of general lifelong intelligence.

## Learning and state boundaries

There are three relevant boundaries:

1. **Within an environment decision:** PyTorch autograd computes conventional
   gradients through the current encoder/predictor/heads.
2. **Across decisions in an episode:** numeric recurrent state and explicit
   traces persist, while the autograd graph is detached. Eligibility mechanisms
   supply selected credit paths.
3. **Across episodes:** learned weights and optimizer moments persist, while
   per-world membrane, spike, memory, feedback, and eligibility state reset.

Current checkpoints preserve module weights only. Because optimizer moments,
active traces, and neural state are absent, loading a checkpoint starts a new
execution state around previously learned weights; it does not resume an
uninterrupted “lifetime.”

## Required stronger evaluation

A credible next continual-learning study should predeclare task/context
sequences, evaluation intervals, seeds, and metrics; log exact transition and
episode counts; measure performance on old tasks after new learning; and save
complete compatible artifacts. Useful metrics include average accuracy,
forgetting, forward/backward transfer, recovery time after context shifts, and
memory-ablation sensitivity.
