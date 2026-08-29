# Architecture

This document describes the canonical non-vision implementation in
`ghost/ghost_terminal_core.py`. It distinguishes neural state, explicit memory,
learning state, and learned parameters because all four have different
lifetimes.

## Data flow

For each parallel world and environment decision:

1. `StatelessEncoder` converts the 15-value observation to a 16-value latent.
2. `Strategizer` consumes that latent, predictor feedback, and the previous
   strategy memory. Its recurrent LIF core proposes an updated strategy and a
   learned gate blends proposal and previous memory.
3. `Actor` consumes latent, strategy, desirability, and outcome uncertainty,
   then samples one of left-turn, right-turn, or forward.
4. The environment supplies sparse reward and a next observation.
5. `Predictor` receives latent, strategy/desirability, and action. Its recurrent
   LIF state predicts the next latent as a bounded residual.
6. Prediction change and prediction error form feedback for the next strategy
   decision. Reward/TD and prediction signals update the enabled online traces
   and parameters.

The actor is stateless in the canonical `separated` condition. Memory is
assigned to the predictor, strategizer, and explicit strategy vector. The
`actor_only` control instead makes the actor core persistent and suppresses the
strategy input.

## Recurrent spiking core

Each `RecurrentSNN` performs five internal ticks per environment decision. For
membrane vector `u`, previous spike vector `s`, input `x`, decay `d`, and a
diagonal-masked recurrent matrix, the implementation is:

```text
u_t = d u_(t-1) + W_in x + W_rec s_(t-1) + b - s_(t-1)
s_t = H(u_t - 1)
```

`H` is a hard threshold in the forward pass. Backpropagation uses the directly
implemented triangular surrogate derivative
`scale * clamp(1 - abs(u - 1), min=0)`, with default scale 0.30. The block
returns the mean across ticks of concatenated membrane and spike values.

Persistent blocks detach `mem` and `spk` between environment decisions. This
preserves numerical neural state but prevents ordinary autograd from building
an episode-length graph.

## Encoder

The encoder LIF block is constructed with `persistent=False`, so its membrane
and spike state start from zero for each decision. Its output passes through
reward-adaptive LayerNorm, a linear latent head, another conditioned norm, and
`tanh`. The default `cue_auxiliary` learning mode also uses a conventional
observation decoder and cue-classification head. It applies MSE reconstruction
plus weighted cue cross-entropy through an Adam step.

The code also contains a `reward_eprop`/JEPA path using a stop-gradient EMA
target encoder, reward-conditioned target latent, predictor loss, optional
variance/SIGReg terms, and optional encoder-directed eligibility. It is an
implemented experimental path, but it is not the verified non-vision headline
configuration represented by the saved six-seed output.

## Predictor

The predictor is a persistent recurrent LIF block. A conventional MLP encodes
strategy plus desirability. The core consumes latent, strategy context, and a
one-hot action. A linear head produces a bounded residual:

```text
z_hat_(t+1) = z_t + 0.5 tanh(delta_t)
```

The feedback carried to the next decision is the concatenation of predicted
change and detached next-latent prediction error. Predictor membrane/spike
state and its explicit eligibility tensors persist between decisions until
that world's episode ends.

## Strategy and outcome state

The canonical strategizer has four state sources:

- persistent recurrent LIF membrane and spikes;
- previous strategy vector;
- predictor feedback;
- learned weights.

Given proposal `p`, previous strategy `m`, and sigmoid gate `g`, explicit memory
is updated elementwise as:

```text
m' = (1 - g) m + g p
```

The strategy proposal and gate affect actor behavior. A separate linear outcome
head reads detached strategizer features and strategy to estimate signed
desirability and clipped log variance. That calibration head receives a
distributional TD-style objective; task credit to the strategy core is carried
through actor likelihood and recurrent strategy eligibility.

## Actor

The actor uses a conventional MLP to encode strategy, desirability, and outcome
log variance. It concatenates that context with the current latent and strategy
memory, processes the result with a non-persistent LIF core, and applies a
linear/softmax categorical head. Training mixes in 10% uniform exploration by
default. Evaluation removes that forced mixture but still samples the learned
categorical policy; it is not argmax evaluation.

## Neural state versus learning state

| State | Persists across decisions? | Reset at world episode end? | Persists across runs? |
|---|---|---|---|
| Predictor membrane/spikes | Yes | Yes | No |
| Strategizer membrane/spikes (`separated`) | Yes | Yes | No |
| Actor membrane/spikes (`separated`) | No | Effectively per decision | No |
| Explicit strategy/desirability/log-variance memory | Yes | Yes | No |
| Predictor feedback | Yes | Yes | No |
| Eligibility/Jacobian/reward traces | Yes | Yes | No |
| Adam moments | Yes during training | No | Not saved by current checkpoint |
| Learned module weights | Yes | No | Yes when checkpointed |

`System.reset(done_mask)` clears neural state, explicit memory, feedback, and
enabled traces only for worlds whose episode completed. Learned weights and
optimizer state continue across episodes. The current checkpoint writer saves
module weights, but not optimizer state or live neural/eligibility state; it is
therefore a policy artifact, not a full lifetime-resume snapshot.

## Temporal credit

Ordinary autograd is used within the current transition. It does not backprop
through the full episode because recurrent state is detached. Separate trace
objects record per-world local derivatives:

- `RewardEprop` accumulates decayed policy score gradients and applies a
  TD-modulated, maximize-mode Adam step;
- `PredictorEprop` tracks input/recurrent/bias eligibility for the persistent
  predictor and installs later prediction gradients;
- `RecurrentStrategyEprop` combines LIF eligibility with exact Jacobians of the
  explicit strategy-memory recursion and delayed actor-score credit;
- optional predictor-to-encoder and strategy-to-encoder traces target selected
  encoder projections.

These are implementation-specific e-prop-related mechanisms, not a claim that
every parameter update is neuron-local or directly hardware deployable.
