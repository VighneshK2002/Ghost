# Refactor audit

Audit date: 2026-08-29. Audited revision: `deccc33` plus the pre-existing
uncommitted move of the terminal implementation into `ghost/`.

## Canonical path

| Existing item | Finding | Public migration |
|---|---|---|
| `ghost_terminal_core.py` | Canonical non-vision delayed-cue engine. The pre-existing `ghost/ghost_terminal_core.py` move was byte-identical to `HEAD`. | Retained behavior-first in `ghost/ghost_terminal_core.py`; small conceptual modules provide the public import surface. |
| `ghost_terminal.py` | Marimo terminal/dashboard for the non-vision engine. | Preserved as optional `archive/marimo/ghost_terminal.py`; imports the package-qualified core. |
| `ghost.py`, `ghost_v1.py` … `ghost_v5.py` | Earlier monolithic Marimo research iterations. They include different point-control, predictor, Fisher/NAF, and trajectory experiments. | Preserve under `archive/legacy/`; not canonical. |
| `ghost_vision.py`, `ghost_vision_core.py` | Unique visual-observation T-maze branch, not the headline non-vision result. | Preserve under `archive/vision/`; explicitly experimental. |
| `test.py` | Standalone Marimo SNN visualization, not an automated test. | Preserve under `archive/demos/`. |

## Delayed-cue experiment and result evidence

- `BatchedTMaze` in the terminal core is the canonical vectorized non-vision
  environment. Its 15-value observation contains position, orientation,
  local walls, a three-way cue (`left`, `right`, `hidden`), and previous action.
- The cue is visible only while `age < cue_steps` (default: two decisions).
- The canonical training configuration uses 24 worlds, 65,536 transitions,
  seed 11 as the first seed, and a configured minimum of 192 evaluation
  episodes.
- `__marimo__/session/ghost_terminal.py.json` contains saved output for six
  runs, seeds 11–16. Five report 97.9%–100.0%; seed 15 reports 0.0%. The
  all-run mean shown by the saved notebook is 82.56%.
- The saved session says checkpoint saving was disabled. Consequently those
  exact six trained states are not present.
- `online_delayed_cue_strategy_tmaze_gated_memory_v8.pt` contains three states
  (`separated_seed11`–`13`) and the expected configuration, but strict loading
  fails because encoder, predictor, and feedback tensor shapes do not match
  any committed source revision. It is historical evidence, not a releasable
  reproduction artifact.
- No CSV learning curves or exact episode counts from the completed evaluation
  were found. Printed rates have three-decimal precision, and the evaluator
  can finish a batch after crossing the configured episode minimum.

## Implementation inventory

- **Spiking:** encoder, predictor, strategizer, and actor use recurrent LIF
  blocks with a triangular surrogate-spike derivative. Predictor and strategy
  cores can retain membrane/spike state across decisions.
- **Persistent state:** predictor LIF state, strategizer LIF state in the
  separated condition, explicit gated strategy memory, predictor feedback,
  and online eligibility traces. All are reset for completed worlds.
- **Conventional PyTorch:** linear layers, LayerNorm, GELU/Tanh MLPs, softmax
  categorical policy, reconstruction/cue heads, autograd within a decision,
  Adam optimizers, and optional EMA targets/representation critic.
- **Controls:** `stateless_strategizer` and `actor_only` are implemented; the
  evaluator also supports shuffle/zero interventions on the strategy signal.
- **Learning:** online updates occur in the transition loop. There is no replay
  buffer and recurrent state is detached between environment decisions, so
  ordinary BPTT is not used across the episode. Explicit eligibility traces
  carry selected temporal credit.

## Artifacts, generated files, and dependencies

- Tracked checkpoint audit:

  | Artifact | Bytes | Role / release decision |
  |---|---:|---|
  | `Ghost/ghost_checkpoint.pt` | 128,676 | Older 3-value point-control task, not delayed-cue evidence; archive via Git history. |
  | `Ghost/ghost_v3_checkpoint.pt` | 312,336 | Older recurrent-intent/outer-delta point-control experiment; not canonical. |
  | `online_delayed_cue_strategy_tmaze_gated_memory_v8.pt` | 387,810 | Non-vision seeds 11–13, 65,536 transitions; incompatible with committed source. Do not release as reproducible. |
  | `online_delayed_cue_tmaze_gated_memory_cue_auxiliary.pt` | 168,754 | One non-vision seed, only 12,536 configured transitions; not the headline run and incompatible with current shapes. |
  | `online_delayed_cue_tmaze_gated_memory_reward_eprop.pt` | 640,666 | Visual/reward-eprop branch despite the generic filename; not the non-vision headline artifact. |
  | `online_delayed_cue_lifetime_tmaze_gated_memory_reward_eprop.pt` | 642,850 | Visual lifetime/energy branch; separate unfinished research path. |

  None is both headline-ready and strictly compatible with its claimed source;
  publish a newly generated, validated checkpoint via a GitHub Release rather
  than normal Git.
- Tracked `__pycache__/*.pyc` and Marimo session JSON are generated. Session
  files reach roughly 1.5 MB and one contains a machine-specific absolute path.
- Minimal runtime dependencies are PyTorch and NumPy. Matplotlib/pytest are
  development extras; Marimo/AnyWidget/Traitlets support the optional UI.
- The clean validation environment used Python 3.13.1, PyTorch 2.13.0, and
  NumPy 2.5.2. The package declares Python 3.10+ because the source syntax and
  dependency floor support it; CI across that range remains to be added.

## Public-data scan

No credential assignment or private-key marker was detected in the current
text files. Generated Marimo state contains a machine-specific path. Git
history exposes a corporate-domain author email; the owner should decide
whether it is appropriate before publication. See `PUBLIC_RELEASE_CHECKLIST.md`.

## Conservative migration decision

The 150 KB engine remains intact to avoid silently changing eligibility-trace
or optimizer behavior. Public modules in `ghost/{neurons,encoder,predictor,
strategy,actor,learning}` expose the conceptual boundaries, while the clean
experiment CLI imports the same implementation. A deeper physical split should
follow only after a validated compatible checkpoint exists.
