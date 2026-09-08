"""Self-contained Ghost e-prop architecture with experimental prediction-bank additions."""
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from dataclasses import dataclass, asdict
    from collections import deque
    from typing import Dict, Iterable, List, Sequence, Tuple
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    return mo, np, torch, nn, F, dataclass, asdict, deque, Dict, Iterable, List, Sequence, Tuple, math, plt, ConnectionPatch


@app.cell
def _(dataclass, np, torch, nn, F, deque, Dict, Iterable, List, Sequence, Tuple, math):
    # Verbatim canonical definitions, tested against ghost_terminal_core.py.
    # Duplicated intentionally so this notebook has no runtime ghost dependency.
    ACTIONS = ("left", "right", "forward")
    ACTION_DIM = len(ACTIONS)
    @dataclass
    class Config:
        seed: int = 11
        device: str = "cpu"
        worlds: int = 24
        transitions: int = 65536
        report_every: int = 4096
        maze_width: int = 9
        maze_height: int = 9
        cue_steps: int = 2
        episode_limit: int = 48
        observation_dim: int = 15
        latent_dim: int = 16
        strategy_dim: int = 8
        hidden_dim: int = 40
        conditioning_dim: int = 24
        snn_ticks: int = 5
        membrane_decay: float = 0.90
        predictor_membrane_decay: float = 0.97
        strategy_membrane_decay: float = 0.98
        surrogate_scale: float = 0.30
        gamma: float = 0.99
        actor_trace_decay: float = 0.85
        strategy_trace_decay: float = 0.99
        encoder_trace_decay: float = 0.99
        predictor_trace_decay: float = 0.995
        use_strategic_prediction_timer: bool = False
        prediction_timer_durations: Tuple[int, ...] = (1, 2, 4, 8, 16)
        strategy_retention: float = 0.95
        learned_strategy_memory: bool = True
        persist_recurrent_state_across_episodes: bool = False
        td_clip: float = 3.0
        encoder_lr: float = 3e-4
        encoder_eprop_lr: float = 3e-5
        jepa_variance_weight: float = 0.0

        #SIGReg
        sigreg_weight: float = 0.0
        sigreg_projections: int = 128
        sigreg_frequency_samples: int = 8
        sigreg_max_frequency: float = 5.0
        sigreg_trace_decay: float = 0.99

        # Strategy SIGReg
        strategy_sigreg_weight: float = 0.0
        strategy_sigreg_projections: int = 128
        strategy_sigreg_frequency_samples: int = 8
        strategy_sigreg_max_frequency: float = 5.0
        strategy_sigreg_trace_decay: float = 0.99

        use_reward_adaln: bool = True
        reward_adaln_strength: float = 0.25
        use_actor_encoder_eprop: bool = False
        use_representation_critic: bool = False
        representation_critic_lr: float = 3e-4
        representation_critic_target_tau: float = 0.005
        critic_encoder_weight: float = 0.0
        predictor_lr: float = 3e-4
        use_predictor_eprop: bool = True
        use_predictor_encoder_eprop: bool = False

        use_strategy_encoder_eprop: bool = True
        strategy_encoder_trace_decay: float = 0.99
        strategy_encoder_eprop_lr: float = 1e-5
        strategy_encoder_eprop_clip: float = 1.0


        detach_predictor_from_encoder: bool = True

        predictor_reward_event_weight: float = 8.0
        predictor_eprop_clip: float = 1.0
        actor_eprop_lr: float = 3e-4
        strategy_eprop_lr: float = 3e-4
        critic_lr: float = 3e-4
        terminal_outcome_variance: float = 0.01
        timeout_penalty: float = -0.1
        curriculum_success_threshold: float = 0.65
        curriculum_min_episodes_per_cue: int = 32
        curriculum_history_per_cue: int = 64
        # Timeout anchors at start rows 1, 3, 5, and the farthest row.
        curriculum_episode_limits: Tuple[int, ...] = (16, 24, 36, 48)
        # Empty selects geometry-adjusted defaults; custom entries are (length, mass).
        curriculum_length_distributions: Tuple[Tuple[Tuple[int, float], ...], ...] = ()
        curriculum_min_episodes_per_length: int = 8
        curriculum_blend_episodes: int = 64
        cue_probability_schedule: Tuple[Tuple[int, float], ...] = ()
        encoder_learning_mode: str = "cue_auxiliary"
        cue_aux_weight: float = 2.0
        exploration_rate: float = 0.10
        adaptive_exploration: bool = False
        adaptive_exploration_min: float = 0.02
        adaptive_exploration_max: float = 0.25
        adaptive_exploration_ema_decay: float = 0.98
        adaptive_exploration_power: float = 2.0
        adaptive_timer_arbitration: bool = False
        adaptive_timer_min_influence: float = 0.10
        adaptive_timer_patience_episodes: int = 256
        adaptive_timer_min_improvement: float = 0.02
        adaptive_timer_exploration_threshold: float = 0.50
        adaptive_timer_gate_ema_decay: float = 0.98
        adaptive_timer_state_machine: bool = False
        timer_gate_scope: str = "all"
        adaptive_exploration_target: str = "actor"
        critic_diagnostics_dir: str = ""
        credit_capture_dir: str = ""
        credit_capture_every: int = 8192
        credit_capture_length: int = 16
        timer_controller_bootstrap_influence: float = 0.50
        timer_controller_episodes_per_cue: int = 32
        timer_controller_progress_threshold: float = 0.05
        timer_controller_success_threshold: float = 0.10
        timer_controller_imbalance_threshold: float = 0.30
        timer_controller_timeout_threshold: float = 0.70
        timer_controller_transition_hold_episodes: int = 256
        training_predictor_feedback: str = "normal"
        evaluation_episodes: int = 192
        checkpoint: str = "online_delayed_cue_strategy_tmaze_gated_memory_v8.pt"


    class BatchedTMaze:
        """Small vectorized delayed-cue T-maze with terminal sparse reward."""

        # Orientations: north, east, south, west.
        DELTAS = np.asarray(((0, -1), (1, 0), (0, 1), (-1, 0)), np.int64)

        def __init__(self, cfg: Config, seed: int,
                     curriculum: bool = True) -> None:
            self.cfg = cfg
            self.rng = np.random.default_rng(seed)
            self.center = cfg.maze_width//2
            self.left_goal = (1, 1)
            self.right_goal = (cfg.maze_width-2, 1)
            self.valid = {(self.center, y)
                          for y in range(1, cfg.maze_height-1)}
            self.valid.update((x, 1) for x in range(1, cfg.maze_width-1))
            full_start = cfg.maze_height-2
            if full_start < 2 or cfg.maze_width < 5:
                raise ValueError("Distributional T-maze requires height >= 4 and width >= 5")
            if len(cfg.curriculum_episode_limits)!=4 or any(v<1 for v in cfg.curriculum_episode_limits):
                raise ValueError("Provide four positive timeout anchors")
            if cfg.curriculum_min_episodes_per_length<1:
                raise ValueError("Length diagnostic sample minimum must be positive")
            # Length counts corridor cells INCLUDING the junction: start_y = length.
            # Distributional curricula avoid fixed-stage temporal shortcuts and retain
            # shorter-delay rehearsal instead of switching to one deterministic start.
            anchors = (
                ((1,.80),(2,.15),(3,.05)),
                ((1,.15),(2,.30),(3,.30),(4,.25)),
                ((1,.03),(2,.07),(3,.15),(4,.25),(5,.25),(6,.25)),
                ((1,.02),(2,.03),(3,.05),(4,.10),(5,.20),(6,.25),(7,.35)),
            )
            # Insert two convex blends per adjacent anchor pair. This makes each
            # promotion one third of the former distribution shift (10 stages total).
            # Explicit custom curricula are already stages and are not expanded.
            defaults=[]
            for left_entries,right_entries in zip(anchors,anchors[1:]):
                left,right=dict(left_entries),dict(right_entries)
                for alpha in (0.,1/3,2/3):
                    blended={h:(1-alpha)*left.get(h,0.)+alpha*right.get(h,0.)
                        for h in sorted(set(left)|set(right))}
                    defaults.append(tuple((h,p) for h,p in blended.items() if p>0))
            defaults.append(anchors[-1])
            distributions=[]
            for entries in cfg.curriculum_length_distributions or defaults:
                distribution={}
                for length, probability in (entries.items() if isinstance(entries,dict) else entries):
                    if int(length)!=length or length<1 or not np.isfinite(probability) or probability<=0:
                        raise ValueError("Hallway lengths must be positive integers with positive finite mass")
                    if cfg.curriculum_length_distributions:
                        if length>full_start:
                            raise ValueError("Hallway length exceeds maze geometry")
                        mapped=int(length)
                    else:
                        # Merge long lengths for small mazes; extend the longest
                        # default length to the farthest row for larger mazes.
                        mapped=min(int(length),full_start) if full_start<=7 else (full_start if length==7 else int(length))
                    distribution[mapped]=distribution.get(mapped,0.)+float(probability)
                if len(distribution)<2:
                    raise ValueError("Every curriculum stage needs multiple hallway lengths")
                total=sum(distribution.values())
                distribution={h:p/total for h,p in sorted(distribution.items())}
                if distributions:
                    previous=distributions[-1]
                    if not set(previous)&set(distribution):
                        raise ValueError("Adjacent hallway distributions must overlap")
                    support=sorted(set(previous)|set(distribution))
                    if any(sum(p for h,p in distribution.items() if h<=cut)>
                           sum(p for h,p in previous.items() if h<=cut)+1e-9 for cut in support):
                        raise ValueError("Later stages must shift probability toward longer hallways")
                distributions.append(distribution)
            self.curriculum_length_distributions=tuple(distributions)
            # Metadata only: maxima do NOT determine individual episode starts.
            self.start_rows=tuple(max(d) for d in distributions)
            self.episode_limits=tuple(self._limit_for_length(h) for h in self.start_rows)
            if cfg.curriculum_blend_episodes<0 or int(cfg.curriculum_blend_episodes)!=cfg.curriculum_blend_episodes:
                raise ValueError("curriculum_blend_episodes must be a nonnegative integer")
            self.curriculum_blend_completed=cfg.curriculum_blend_episodes
            self.curriculum_enabled=curriculum
            self.curriculum_stage=0 if curriculum else len(distributions)-1
            # Independent stream preserves cue quotas/RNG and avoids length-cue coupling.
            self.length_rng=np.random.default_rng(np.random.SeedSequence([seed, 71939]))
            self.curriculum_history = (
                deque(maxlen=cfg.curriculum_history_per_cue),
                deque(maxlen=cfg.curriculum_history_per_cue))
            self.cue_probability_schedule = tuple(
                (int(transition), float(probability))
                for transition, probability in cfg.cue_probability_schedule
            )
            self._validate_cue_probability_schedule()
            self.completed_transitions = 0
            self.cue_assignment_counts = np.zeros(2, np.int64)
            self.cue_assignment_probability_sum = 0.0
            self._cue_schedule_index = -1
            self._scheduled_assignments = 0
            self._scheduled_left_assignments = 0
            b = cfg.worlds
            self.x = np.zeros(b, np.int64)
            self.y = np.zeros(b, np.int64)
            self.direction = np.zeros(b, np.int64)
            self.cue = np.zeros(b, np.int64)
            self.age = np.zeros(b, np.int64)
            self.hallway_length = np.zeros(b, np.int64)
            self.episode_time_limit = np.zeros(b, np.int64)
            self.episode_stage = np.zeros(b, np.int64)
            self.episode_blend = np.ones(b, np.float64)
            self.previous_action = np.full(b, 2, np.int64)
            self.episode = np.zeros(b, np.int64)
            self.reset(np.ones(b, bool))

        @property
        def curriculum_blend_fraction(self):
            duration=self.cfg.curriculum_blend_episodes
            return min(1.,self.curriculum_blend_completed/duration) if duration else 1.

        @property
        def current_length_distribution(self):
            target=self.curriculum_length_distributions[self.curriculum_stage]
            alpha=self.curriculum_blend_fraction
            if self.curriculum_stage==0 or alpha>=1.:
                return dict(target)
            previous=self.curriculum_length_distributions[self.curriculum_stage-1]
            blended={h:(1-alpha)*previous.get(h,0.)+alpha*target.get(h,0.)
                for h in sorted(set(previous)|set(target))}
            return {h:p for h,p in blended.items() if p>0}

        def _limit_for_length(self, length: int) -> int:
            # Preserve historical timeout anchors, interpolating by sampled length.
            # The limit stays fixed for that episode even if another world promotes.
            anchors={}
            for row,limit in zip((1,3,5,self.cfg.maze_height-2),self.cfg.curriculum_episode_limits):
                row=min(row,self.cfg.maze_height-2)
                anchors[row]=max(anchors.get(row,0),int(limit))
            rows=sorted(anchors)
            interpolated=int(np.ceil(np.interp(length,rows,[anchors[r] for r in rows])))
            shortest_route=(length-1)+1+max(self.center-1,self.cfg.maze_width-2-self.center)
            return max(interpolated,shortest_route+2)

        @staticmethod
        def summarize_hallways(records, min_samples=8, distributions=()):
            """Completed-episode diagnostics; empty cue/length groups remain missing."""
            rows=[]
            lengths=sorted({r["hallway_length"] for r in records} |
                           {h for distribution in distributions for h in distribution})
            for length in lengths:
                group=[r for r in records if r["hallway_length"]==length]
                row=dict(hallway_length=length,samples=len(group),
                    success=float(np.mean([r["success"] for r in group])) if group else None)
                for cue,label in ((0,"left"),(1,"right")):
                    selected=[r["success"] for r in group if r["cue"]==cue]
                    row["samples_"+label]=len(selected)
                    row["success_"+label]=float(np.mean(selected)) if selected else None
                rows.append(row)
            enough=[r["success"] for r in rows if r["samples"]>=min_samples]
            sampled=[r["hallway_length"] for r in records]
            stage_rows=[]
            for stage,distribution in enumerate(distributions,1):
                group=[r for r in records if r["curriculum_stage"]==stage]
                for length,probability in distribution.items():
                    count=sum(r["hallway_length"]==length for r in group)
                    stage_rows.append(dict(curriculum_stage=stage,hallway_length=length,
                        probability=probability,samples=count,
                        observed_probability=count/len(group) if group else None))
            cue_rates=[float(np.mean([r["success"] for r in records if r["cue"]==cue]))
                if any(r["cue"]==cue for r in records) else None for cue in (0,1)]
            return dict(worst_cue_success=min(cue_rates) if all(v is not None for v in cue_rates) else None,
                success_by_hallway_length=rows,
                mean_hallway_length=float(np.mean(sampled)) if sampled else None,
                min_hallway_length=min(sampled) if sampled else None,
                max_hallway_length=max(sampled) if sampled else None,
                worst_length_success=min(enough) if enough else None,
                worst_length_min_samples=min_samples,
                hallway_distribution_by_stage=stage_rows)

        def _validate_cue_probability_schedule(self) -> None:
            schedule = self.cue_probability_schedule
            if not schedule:
                return
            if schedule[0][0] != 0:
                raise ValueError("cue probability schedule must start at transition 0")
            previous_transition = -1
            for transition, probability in schedule:
                if transition <= previous_transition:
                    raise ValueError(
                        "cue probability schedule transitions must be strictly increasing"
                    )
                if not 0.0 <= probability <= 1.0:
                    raise ValueError("cue probability must be in [0, 1]")
                previous_transition = transition

        def _scheduled_cues(self, count: int) -> np.ndarray:
            schedule_index = max(
                index
                for index, (transition, _) in enumerate(self.cue_probability_schedule)
                if transition <= self.completed_transitions
            )
            probability_left = self.cue_probability_schedule[schedule_index][1]
            if schedule_index != self._cue_schedule_index:
                self._cue_schedule_index = schedule_index
                self._scheduled_assignments = 0
                self._scheduled_left_assignments = 0

            new_total = self._scheduled_assignments + count
            target_left = int(np.floor(new_total * probability_left + 0.5))
            left_count = target_left - self._scheduled_left_assignments
            cues = np.ones(count, np.int64)
            cues[:left_count] = 0
            self.rng.shuffle(cues)
            self._scheduled_assignments = new_total
            self._scheduled_left_assignments = target_left
            return cues

        @property
        def current_cue_probability_left(self) -> float:
            if not self.curriculum_enabled or not self.cue_probability_schedule:
                return 0.5
            probability = self.cue_probability_schedule[0][1]
            for transition, candidate in self.cue_probability_schedule:
                if transition > self.completed_transitions:
                    break
                probability = candidate
            return probability

        def reset(self, mask: np.ndarray) -> None:
            count = int(mask.sum())
            if not count:
                return
            self.x[mask] = self.center
            distribution=self.current_length_distribution
            self.episode_blend[mask]=self.curriculum_blend_fraction
            sampled=self.length_rng.choice(list(distribution),size=count,p=list(distribution.values()))
            self.hallway_length[mask]=sampled
            self.episode_time_limit[mask]=[self._limit_for_length(int(h)) for h in sampled]
            self.episode_stage[mask]=self.curriculum_stage
            self.y[mask]=sampled
            self.direction[mask] = 0
            if self.curriculum_enabled and self.cue_probability_schedule:
                assignment_probability_left = self.current_cue_probability_left
                assigned_cues = self._scheduled_cues(count)
            else:
                # Preserve the historical RNG path when no schedule is requested.
                assignment_probability_left = 0.5
                assigned_cues = self.rng.integers(0, 2, count)
            self.cue[mask] = assigned_cues
            self.cue_assignment_counts += np.bincount(assigned_cues, minlength=2)
            self.cue_assignment_probability_sum += count * assignment_probability_left
            self.age[mask] = 0
            self.previous_action[mask] = 2
            self.episode[mask] += 1

        def _wall(self, world: int, relative_turn: int) -> float:
            direction = (int(self.direction[world])+relative_turn) % 4
            dx, dy = self.DELTAS[direction]
            target = (int(self.x[world]+dx), int(self.y[world]+dy))
            return float(target not in self.valid)

        def observation(self) -> np.ndarray:
            rows = []
            for world in range(self.cfg.worlds):
                position = (2*np.asarray((
                    self.x[world]/(self.cfg.maze_width-1),
                    self.y[world]/(self.cfg.maze_height-1)), np.float32)-1)
                orientation = np.eye(4, dtype=np.float32)[self.direction[world]]
                walls = np.asarray((self._wall(world, -1),
                                    self._wall(world, 0),
                                    self._wall(world, 1)), np.float32)
                cue = np.zeros(3, np.float32)
                cue[int(self.cue[world]) if self.age[world] < self.cfg.cue_steps
                    else 2] = 1
                previous = np.eye(ACTION_DIM, dtype=np.float32)[
                    self.previous_action[world]]
                rows.append(np.concatenate((position, orientation, walls,
                                            cue, previous)))
            result = np.stack(rows)
            if result.shape[1] != self.cfg.observation_dim:
                raise RuntimeError(f"observation dimension is {result.shape[1]}")
            return result

        def step(self, actions: np.ndarray):
            rewards = np.zeros(self.cfg.worlds, np.float32)
            successes = np.zeros(self.cfg.worlds, bool)
            wrong = np.zeros(self.cfg.worlds, bool)
            for world, action in enumerate(actions.tolist()):
                if action == 0:
                    self.direction[world] = (self.direction[world]-1) % 4
                elif action == 1:
                    self.direction[world] = (self.direction[world]+1) % 4
                else:
                    dx, dy = self.DELTAS[self.direction[world]]
                    target = (int(self.x[world]+dx), int(self.y[world]+dy))
                    if target in self.valid:
                        self.x[world], self.y[world] = target
                self.previous_action[world] = action
                self.age[world] += 1
                position = (int(self.x[world]), int(self.y[world]))
                at_left, at_right = (position == self.left_goal,
                                     position == self.right_goal)
                correct = ((self.cue[world] == 0 and at_left)
                           or (self.cue[world] == 1 and at_right))
                incorrect = ((self.cue[world] == 0 and at_right)
                             or (self.cue[world] == 1 and at_left))
                successes[world], wrong[world] = correct, incorrect
                # Correct and incorrect goals remain signed sparse outcomes.
                rewards[world] = 1.0 if correct else -1.0 if incorrect else 0.0
            timeout = self.age >= self.episode_time_limit
            pure_timeout = timeout & ~successes & ~wrong
            rewards[pure_timeout] = self.cfg.timeout_penalty
            done = successes | wrong | timeout
            # Preserve the actual post-action observation for world-model
            # learning.  The ordinary return value remains the automatically reset
            # observation consumed by the policy on the following decision.
            self.transition_observation = self.observation()
            cue = self.cue.copy()
            terminal_age = self.age.copy()
            # Preserve completed episode metadata BEFORE auto-reset/promotion.
            self.transition_hallway_length=self.hallway_length.copy()
            self.transition_episode_stage=self.episode_stage.copy()
            self.transition_episode_blend=self.episode_blend.copy()
            self.transition_episode_time_limit=self.episode_time_limit.copy()
            if self.curriculum_enabled:
                # Progress the blend by completed episodes across all worlds. Only
                # episodes sampled from the settled target count toward mastery.
                self.curriculum_blend_completed=min(self.cfg.curriculum_blend_episodes,
                    self.curriculum_blend_completed+int(done.sum()))
                for world in np.flatnonzero(done & (self.episode_stage==self.curriculum_stage)
                                           & (self.episode_blend>=1.)):
                    self.curriculum_history[int(cue[world])].append(
                        float(successes[world]))
                enough_evidence = all(
                    len(history) >= self.cfg.curriculum_min_episodes_per_cue
                    for history in self.curriculum_history)
                mastered = self.curriculum_blend_fraction>=1. and enough_evidence and all(
                    float(np.mean(history)) >=
                    self.cfg.curriculum_success_threshold
                    for history in self.curriculum_history)
                if mastered and self.curriculum_stage < len(self.start_rows)-1:
                    self.curriculum_stage += 1
                    self.curriculum_blend_completed=0
                    for history in self.curriculum_history:
                        history.clear()
            self.completed_transitions += self.cfg.worlds
            self.reset(done)
            return self.observation(), rewards, done, successes, wrong, cue, terminal_age

        @property
        def curriculum_rates(self) -> Tuple[float, float]:
            return tuple(float(np.mean(history)) if history else 0.0
                         for history in self.curriculum_history)


    class SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, voltage: torch.Tensor, scale: float):
            ctx.save_for_backward(voltage)
            ctx.scale = scale
            return (voltage >= 0).to(voltage.dtype)

        @staticmethod
        def backward(ctx, gradient: torch.Tensor):
            (voltage,) = ctx.saved_tensors
            return (gradient*ctx.scale
                    *torch.clamp(1-voltage.abs(), min=0)), None


    class RecurrentSNN(nn.Module):
        """Multi-tick recurrent LIF block with explicitly managed online state."""

        def __init__(self, input_dim: int, hidden_dim: int, cfg: Config,
                     persistent: bool, decay: float | None = None,
                     record_eligibility: bool = False) -> None:
            super().__init__()
            self.cfg, self.hidden_dim, self.persistent = cfg, hidden_dim, persistent
            self.decay = cfg.membrane_decay if decay is None else decay
            self.record_eligibility = record_eligibility
            self.input = nn.Linear(input_dim, hidden_dim)
            self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.orthogonal_(self.recurrent.weight, gain=0.35)
            self.bias = nn.Parameter(torch.full((hidden_dim,), 0.05))
            self.mem: torch.Tensor | None = None
            self.spk: torch.Tensor | None = None
            self.last_output: torch.Tensor | None = None
            self.last_eligibility_records = []

        def initial(self, batch: int, device: torch.device) -> None:
            self.mem = torch.zeros(batch, self.hidden_dim, device=device)
            self.spk = torch.zeros_like(self.mem)

        def snapshot(self):
            if self.mem is None:
                return None
            return self.mem.clone(), self.spk.clone()

        def restore(self, state) -> None:
            if state is None:
                self.mem = self.spk = None
            else:
                self.mem, self.spk = state[0].clone(), state[1].clone()

        def reset(self, mask: torch.Tensor | None = None) -> None:
            if self.mem is None:
                return
            if mask is None:
                self.mem.zero_(); self.spk.zero_()
            elif bool(mask.any()):
                self.mem[mask] = 0; self.spk[mask] = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if (not self.persistent or self.mem is None
                    or self.mem.shape[0] != value.shape[0]):
                mem = torch.zeros(value.shape[0], self.hidden_dim,
                                  device=value.device, dtype=value.dtype)
                spk = torch.zeros_like(mem)
            else:
                mem, spk = self.mem.detach(), self.spk.detach()
            features = []
            self.last_eligibility_records = []
            recurrent_mask = 1-torch.eye(
                self.hidden_dim, device=value.device, dtype=value.dtype)
            for _ in range(self.cfg.snn_ticks):
                presynaptic_spikes = spk
                current = (self.input(value)
                           +F.linear(spk, self.recurrent.weight*recurrent_mask)
                           +self.bias)
                mem = self.decay*mem+current-spk
                spk = SurrogateSpike.apply(
                    mem-1.0, self.cfg.surrogate_scale)
                if self.record_eligibility:
                    pseudo_derivative = (self.cfg.surrogate_scale
                        *torch.clamp(1-(mem.detach()-1.0).abs(), min=0))
                    self.last_eligibility_records.append((
                        value.detach(), presynaptic_spikes.detach(),
                        pseudo_derivative))
                features.append(torch.cat((mem, spk), -1))
            if self.persistent:
                self.mem, self.spk = mem.detach(), spk.detach()
            self.last_output = torch.stack(features).mean(0)
            return self.last_output


    def mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
                             nn.GELU(), nn.Linear(hidden, output_dim), nn.Tanh())


    class RewardAdaptiveLayerNorm(nn.Module):
        """LayerNorm whose affine transformation is causally reward-conditioned.

        A fixed signed component prevents the learned modulation from solving the
        JEPA objective by silently discarding reward.  The learned residual starts
        at zero and can reshape that affective signal as training progresses.
        """

        def __init__(self, feature_dim: int, hidden_dim: int,
                     strength: float) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(feature_dim)
            self.strength = strength
            self.conditioner = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2*feature_dim),
            )
            nn.init.zeros_(self.conditioner[-1].weight)
            nn.init.zeros_(self.conditioner[-1].bias)
            direction = torch.where(
                torch.arange(feature_dim) % 2 == 0,
                torch.ones(feature_dim),
                -torch.ones(feature_dim),
            )
            self.register_buffer("fixed_direction", direction)

        def forward(self, value: torch.Tensor,
                    reward: torch.Tensor | None = None) -> torch.Tensor:
            normalized = self.norm(value)
            if reward is None:
                reward = torch.zeros(
                    value.shape[0], device=value.device, dtype=value.dtype)
            reward = reward.to(dtype=value.dtype).reshape(-1, 1)
            gamma, beta = self.conditioner(reward).chunk(2, -1)
            fixed_beta = (
                self.strength*reward*self.fixed_direction.to(value.dtype))
            return normalized*(1+gamma)+beta+fixed_beta


    class StatelessEncoder(nn.Module):
        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg
            self.core = RecurrentSNN(
                cfg.observation_dim, cfg.hidden_dim, cfg, persistent=False)
            self.norm = RewardAdaptiveLayerNorm(
                2*cfg.hidden_dim, cfg.hidden_dim,
                cfg.reward_adaln_strength if cfg.use_reward_adaln else 0.0)
            self.latent_head = nn.Linear(2*cfg.hidden_dim, cfg.latent_dim)
            # A final conditioned normalization leaves no downstream encoder
            # projection that can erase the fixed reward modulation.
            self.latent_norm = RewardAdaptiveLayerNorm(
                cfg.latent_dim, cfg.hidden_dim,
                cfg.reward_adaln_strength if cfg.use_reward_adaln else 0.0)
            self.decoder = nn.Sequential(
                nn.Linear(cfg.latent_dim, cfg.hidden_dim), nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.observation_dim))
            self.cue_head = nn.Linear(cfg.latent_dim, 3)

        # def encode(self, observation: torch.Tensor,
        #            reward: torch.Tensor | None = None) -> torch.Tensor:
        #     hidden = self.norm(self.core(observation), reward)
        #     return torch.tanh(self.latent_norm(
        #         self.latent_head(hidden), reward))

        def encode_with_pre_tanh(
                self,
                observation: torch.Tensor,
                reward: torch.Tensor | None = None
            ) -> Tuple[torch.Tensor, torch.Tensor]:

            hidden = self.norm(
                    self.core(observation),
                    reward,
                )

            pre_tanh = self.latent_norm(
                self.latent_head(hidden),
                reward,
            )

            latent = torch.tanh(pre_tanh)

            return latent,pre_tanh

        def encode(
                self,
                observation: torch.Tensor,
                reward: torch.Tensor | None = None
            ) -> torch.Tensor:

            latent, _=self.encode_with_pre_tanh(
                observation,
                reward
            )

            return latent

        

        def forward(self, observation: torch.Tensor,
                    reward: torch.Tensor | None = None):
            latent = self.encode(observation, reward)
            return latent, self.decoder(latent), self.cue_head(latent)


    class Strategizer(nn.Module):
        def __init__(self, cfg: Config, persistent: bool) -> None:
            super().__init__(); self.cfg = cfg


            self.feedback_encoder = mlp(
                2 * cfg.latent_dim,
                cfg.hidden_dim,
                cfg.conditioning_dim,
            )


            core_input_dim = cfg.latent_dim+cfg.conditioning_dim
            if cfg.learned_strategy_memory:
                core_input_dim += cfg.strategy_dim
            self.core = RecurrentSNN(
                core_input_dim, cfg.hidden_dim, cfg,
                persistent=persistent, decay=cfg.strategy_membrane_decay,
                record_eligibility=True)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.strategy_head = nn.Linear(2*cfg.hidden_dim, cfg.strategy_dim)
            self.timer_head = (
                nn.Linear(2*cfg.hidden_dim, len(cfg.prediction_timer_durations))
                if cfg.use_strategic_prediction_timer else None
            )
            if self.timer_head is not None:
                nn.init.zeros_(self.timer_head.weight)
                nn.init.zeros_(self.timer_head.bias)
            self.gate_head = (nn.Linear(2*cfg.hidden_dim, cfg.strategy_dim)
                              if cfg.learned_strategy_memory else None)
            if self.gate_head is not None:
                nn.init.zeros_(self.gate_head.weight)
                nn.init.constant_(self.gate_head.bias,
                                  math.log(0.05/0.95))
            # This head evaluates the actual deterministic strategy proposal.  Its
            # two scalars are expected signed return (desirability) and uncertainty
            # about that outcome, not variance of a strategy-sampling policy.
            outcome_input_dim = 2*cfg.hidden_dim+cfg.strategy_dim
            if self.timer_head is not None:
                outcome_input_dim += len(cfg.prediction_timer_durations)
            self.outcome_head = nn.Linear(outcome_input_dim, 2)
            # With sparse reward an arbitrary initial critic would manufacture TD
            # credit before any outcome had occurred.  A zero critic makes early
            # no-reward transitions genuinely censored until terminal evidence.
            nn.init.zeros_(self.outcome_head.weight)
            nn.init.zeros_(self.outcome_head.bias)

        def forward(self, latent: torch.Tensor, feedback: torch.Tensor,
                    deterministic: bool = False,
                    timer_influence: float = 1.0,
                    previous_strategy: torch.Tensor | None = None):
            
            conditioning = self.feedback_encoder(feedback)
            if self.cfg.learned_strategy_memory:
                if previous_strategy is None:
                    previous_strategy = torch.zeros(
                        latent.shape[0], self.cfg.strategy_dim,
                        device=latent.device, dtype=latent.dtype)
                core_input = torch.cat(
                    (latent, conditioning, previous_strategy), -1)
            else:
                core_input = torch.cat((latent, conditioning), -1)
            feature = self.norm(self.core(core_input))


            proposal_pre_tanh = self.strategy_head(feature)
            proposal = torch.tanh(proposal_pre_tanh)


            if self.gate_head is not None:
                gate = torch.sigmoid(self.gate_head(feature))
                strategy = ((1-gate)*previous_strategy+gate*proposal)
            else:
                gate = torch.full_like(proposal, 1-self.cfg.strategy_retention)
                strategy = proposal
            if self.timer_head is not None:
                timer_logits = self.timer_head(feature)
                timer_distribution = torch.distributions.Categorical(
                    logits=timer_logits)
                timer_index = (
                    timer_logits.argmax(-1)
                    if deterministic else timer_distribution.sample()
                )
                timer_logp = timer_distribution.log_prob(timer_index)
                timer_entropy = timer_distribution.entropy()
                duration_values = torch.as_tensor(
                    self.cfg.prediction_timer_durations,
                    device=feature.device,
                    dtype=torch.long,
                )
                timer_duration = duration_values[timer_index]
                timer_context = F.one_hot(
                    timer_index, len(self.cfg.prediction_timer_durations)
                ).to(feature.dtype)
            else:
                timer_logits = None
                timer_index = torch.zeros(
                    latent.shape[0], device=latent.device, dtype=torch.long)
                timer_duration = torch.ones_like(timer_index)
                timer_logp = torch.zeros_like(latent[:, 0])
                timer_entropy = torch.zeros_like(timer_logp)
                timer_context = None
            # The outcome loss trains only this calibration head.  Task gradients
            # reach the strategy core through the actor likelihood/e-prop path,
            # preserving the intended strategizer -> actor division of labour.
            outcome_inputs = [feature.detach(), strategy.detach()]
            if timer_context is not None:
                outcome_inputs.append(timer_influence*timer_context.detach())
            outcome = self.outcome_head(torch.cat(outcome_inputs, -1))
            desirability = outcome[:, 0]
            outcome_logvar = outcome[:, 1].clamp(-5.0, 2.0)
            return {
                "feature": feature,
                "proposal_pre_tanh": proposal_pre_tanh,
                "proposal": proposal,
                "gate": gate,
                "strategy": strategy,
                "previous_strategy": previous_strategy,
                "desirability": desirability,
                "outcome_logvar": outcome_logvar,
                "timer_logits": timer_logits,
                "timer_index": timer_index,
                "timer_duration": timer_duration,
                "timer_logp": timer_logp,
                "timer_entropy": timer_entropy,
            }

        def snapshot(self):
            return self.core.snapshot()

        def restore(self, state) -> None:
            self.core.restore(state)

        def reset(self, mask: torch.Tensor) -> None:
            self.core.reset(mask)


    class Actor(nn.Module):
        def __init__(self, cfg: Config, persistent: bool) -> None:
            super().__init__(); self.cfg = cfg
            self.strategy_encoder = mlp(
                cfg.strategy_dim+2, cfg.hidden_dim, cfg.conditioning_dim)
            actor_input_dim = cfg.latent_dim+cfg.conditioning_dim
            if cfg.learned_strategy_memory:
                actor_input_dim += cfg.strategy_dim
            self.core = RecurrentSNN(
                actor_input_dim, cfg.hidden_dim, cfg,
                persistent=persistent)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.head = nn.Linear(2*cfg.hidden_dim, ACTION_DIM)

        def forward(self, latent: torch.Tensor, strategy: torch.Tensor,
                    desirability: torch.Tensor, outcome_logvar: torch.Tensor,
                    deterministic: bool = False,
                    exploration: float = 0.0):
            strategy_context = torch.cat((strategy, desirability[:, None],
                                          outcome_logvar[:, None]), -1)
            conditioning = self.strategy_encoder(strategy_context)
            actor_inputs = [latent, conditioning]
            if self.cfg.learned_strategy_memory:
                actor_inputs.append(strategy)
            feature = self.norm(self.core(torch.cat(actor_inputs, -1)))
            logits = self.head(feature)
            probabilities = logits.softmax(-1)
            if exploration:
                probabilities = ((1-exploration)*probabilities
                                 +exploration/ACTION_DIM)
            distribution = torch.distributions.Categorical(probs=probabilities)
            action = logits.argmax(-1) if deterministic else distribution.sample()
            return {"logits": logits, "action": action,
                    "logp": distribution.log_prob(action),
                    "entropy": distribution.entropy()}

        def reset(self, mask: torch.Tensor) -> None:
            self.core.reset(mask)


    class Predictor(nn.Module):

        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg


            #Strategy Encoder
            self.strategy_encoder = mlp(
                cfg.strategy_dim + 1,
                cfg.hidden_dim,
                cfg.conditioning_dim,
            )

            timer_input_dim = (
                len(cfg.prediction_timer_durations)
                if cfg.use_strategic_prediction_timer else 0
            )
            self.core = RecurrentSNN(
                cfg.latent_dim+cfg.conditioning_dim+ACTION_DIM+timer_input_dim,
                cfg.hidden_dim, cfg, persistent=True,
                decay=cfg.predictor_membrane_decay,
                record_eligibility=True)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.head = nn.Linear(
                2 * cfg.hidden_dim,
                cfg.latent_dim,
            )

        def forward(
                self,
                latent: torch.Tensor,
                strategy: torch.Tensor,
                desirability: torch.Tensor,
                action: torch.Tensor,
                timer_index: torch.Tensor | None = None,
            ):

            strategy_context = torch.cat(
                (
                    strategy,
                    desirability[:, None],
                ),
                dim=-1,
            )

            conditioning = self.strategy_encoder(
                strategy_context
            )

            action_code = F.one_hot(action, ACTION_DIM).float()
            predictor_inputs = [latent, conditioning, action_code]
            if self.cfg.use_strategic_prediction_timer:
                if timer_index is None:
                    raise ValueError(
                        "timer_index is required when strategic prediction timing is enabled"
                    )
                predictor_inputs.append(F.one_hot(
                    timer_index, len(self.cfg.prediction_timer_durations)
                ).to(latent.dtype))
            feature = self.norm(self.core(torch.cat(predictor_inputs, -1)))

            delta = self.head(feature)

            predicted_next_latent = (
                latent
                + 0.5 * torch.tanh(delta)
            )

            return predicted_next_latent

        def reset(self, mask: torch.Tensor) -> None:
            self.core.reset(mask)


    class RepresentationCritic(nn.Module):
        """Strategy-conditioned value head with a differentiable latent input."""

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(cfg.latent_dim+cfg.strategy_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, 2),
            )
            final = self.network[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        def forward(
                self, latent: torch.Tensor,
                strategy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            outcome = self.network(torch.cat((latent, strategy), -1))
            value = outcome[:, 0]
            logvar = outcome[:, 1].clamp(-5.0, 2.0)
            return value, logvar


    def module_parameters(modules: Iterable[nn.Module]) -> List[nn.Parameter]:
        result: List[nn.Parameter] = []
        seen = set()
        for module in modules:
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    seen.add(id(parameter)); result.append(parameter)
        return result


    class RewardEprop:
        """Per-world score eligibility followed by an unnormalised Adam step."""

        def __init__(self, parameters: Sequence[nn.Parameter], worlds: int,
                     decay: float, learning_rate: float) -> None:
            self.parameters = list(parameters)
            self.decay = decay
            self.traces = [torch.zeros(
                (worlds,)+tuple(parameter.shape), device=parameter.device)
                for parameter in self.parameters]
            self.optimizer = torch.optim.Adam(
                self.parameters, lr=learning_rate, maximize=True)

        def accumulate(self, objective: torch.Tensor) -> float:
            batch = objective.shape[0]
            basis = torch.eye(batch, device=objective.device,
                              dtype=objective.dtype)
            gradients = torch.autograd.grad(
                objective, self.parameters, grad_outputs=basis,
                is_grads_batched=True, retain_graph=True, allow_unused=True)
            with torch.no_grad():
                for trace, gradient in zip(self.traces, gradients):
                    trace.mul_(self.decay)
                    if gradient is not None:
                        trace.add_(gradient.detach())
            return float(torch.stack([trace.square().sum()
                                      for trace in self.traces]).sum().sqrt())

        def apply(self, td_error: torch.Tensor) -> Tuple[float, float]:
            self.optimizer.zero_grad(set_to_none=True)
            direction_square = torch.zeros((), device=td_error.device)
            for parameter, trace in zip(self.parameters, self.traces):
                view = (td_error.shape[0],)+(1,)*(trace.ndim-1)
                direction = (trace*td_error.view(view)).mean(0)
                parameter.grad = direction
                direction_square += direction.square().sum()
            before = [parameter.detach().clone() for parameter in self.parameters]
            self.optimizer.step()
            step_square = torch.stack([
                (parameter-old).square().sum()
                for parameter, old in zip(self.parameters, before)]).sum()
            return (float(direction_square.sqrt().detach()),
                    float(step_square.sqrt().detach()))

        def reset(self, mask: torch.Tensor) -> None:
            if bool(mask.any()):
                with torch.no_grad():
                    for trace in self.traces:
                        trace[mask] = 0


    class PredictorEprop:
        """Online recurrent eligibility for the supervised JEPA predictor.

        Predictor state is deliberately detached between environment decisions,
        so ordinary autograd only sees the current decision.  These neuron-local
        traces carry derivatives of earlier inputs and recurrent activity until a
        later prediction error supplies the learning signal.
        """

        def __init__(self, predictor: Predictor, worlds: int,
                     trace_decay: float, gradient_clip: float) -> None:
            self.predictor = predictor
            self.trace_decay = trace_decay
            self.gradient_clip = gradient_clip
            core = predictor.core
            h, d = core.hidden_dim, core.input.in_features
            device = next(predictor.parameters()).device
            self.epsilon_in = torch.zeros(worlds, h, d, device=device)
            self.epsilon_rec = torch.zeros(worlds, h, h, device=device)
            self.epsilon_bias = torch.zeros(worlds, h, device=device)
            self.last_eligibility_norm = 0.0
            self.last_gradient_norm = 0.0

        def gradients(self, loss: torch.Tensor) -> Dict[str, torch.Tensor]:
            core = self.predictor.core
            if core.last_output is None:
                raise RuntimeError("predictor core has no eligibility output")
            learning_signal = torch.autograd.grad(
                loss, core.last_output, retain_graph=True)[0].detach()
            h = core.hidden_dim
            gradient_in = torch.zeros_like(core.input.weight)
            gradient_rec = torch.zeros_like(core.recurrent.weight)
            gradient_bias = torch.zeros_like(core.bias)
            recurrent_mask = 1-torch.eye(
                h, device=learning_signal.device, dtype=learning_signal.dtype)
            ticks = max(len(core.last_eligibility_records), 1)

            with torch.no_grad():
                for tick, (value, presynaptic, pseudo_derivative) in enumerate(
                        core.last_eligibility_records):
                    carry = core.decay*(self.trace_decay if tick == 0 else 1.0)
                    self.epsilon_in.mul_(carry).add_(value[:, None, :])
                    self.epsilon_rec.mul_(carry).add_(
                        presynaptic[:, None, :]*recurrent_mask[None])
                    self.epsilon_bias.mul_(carry).add_(1)
                    coefficient = (
                        learning_signal[:, :h]
                        +learning_signal[:, h:]*pseudo_derivative)/ticks
                    gradient_in.add_(torch.einsum(
                        "bh,bhd->hd", coefficient, self.epsilon_in))
                    gradient_rec.add_(torch.einsum(
                        "bh,bhk->hk", coefficient, self.epsilon_rec))
                    gradient_bias.add_(
                        (coefficient*self.epsilon_bias).sum(0))

                gradient_rec.mul_(recurrent_mask)
                eligibility_square = (
                    self.epsilon_in.square().sum()
                    +self.epsilon_rec.square().sum()
                    +self.epsilon_bias.square().sum())
                gradient_square = (
                    gradient_in.square().sum()
                    +gradient_rec.square().sum()
                    +2*gradient_bias.square().sum())
                self.last_eligibility_norm = float(eligibility_square.sqrt())
                raw_gradient_norm = gradient_square.sqrt()
                if self.gradient_clip > 0:
                    scale = torch.clamp(
                        self.gradient_clip/raw_gradient_norm.clamp_min(1e-12),
                        max=1.0)
                    gradient_in.mul_(scale)
                    gradient_rec.mul_(scale)
                    gradient_bias.mul_(scale)
                self.last_gradient_norm = float(torch.minimum(
                    raw_gradient_norm,
                    torch.as_tensor(
                        self.gradient_clip, device=raw_gradient_norm.device)
                    if self.gradient_clip > 0 else raw_gradient_norm))

            return {
                "input_weight": gradient_in,
                "input_bias": gradient_bias,
                "recurrent_weight": gradient_rec,
                "bias": gradient_bias.clone(),
            }

        def install(self, gradients: Dict[str, torch.Tensor]) -> None:
            core = self.predictor.core
            core.input.weight.grad = gradients["input_weight"]
            core.input.bias.grad = gradients["input_bias"]
            core.recurrent.weight.grad = gradients["recurrent_weight"]
            core.bias.grad = gradients["bias"]

        def reset(self, mask: torch.Tensor) -> None:
            if not bool(mask.any()):
                return
            with torch.no_grad():
                self.epsilon_in[mask] = 0
                self.epsilon_rec[mask] = 0
                self.epsilon_bias[mask] = 0


    class RecurrentStrategyEprop:
        """Bellec-style LIF eligibility plus exact leaky-memory eligibility.

        The recurrent SNN carries neuron-local membrane eligibility across every
        SNN tick and environment decision.  The resulting derivative of each
        strategy write is then carried through the external strategy memory:

            dm_t/dtheta = keep*dm_{t-1}/dtheta + write*ds_t/dtheta.

        The actor's current d log pi / dm contracts with this Jacobian before the
        ordinary reward eligibility trace is advanced.  No temporal autograd
        graph or BPTT is retained.
        """

        def __init__(self, strategizer: Strategizer,
                     parameters: Sequence[nn.Parameter], worlds: int,
                     decay: float, learning_rate: float,
                     persistent: bool) -> None:
            self.strategizer = strategizer
            self.parameters = list(parameters)
            self.decay = decay
            self.persistent = persistent
            cfg = strategizer.cfg
            b, h = worlds, cfg.hidden_dim
            d, k = strategizer.core.input.in_features, cfg.strategy_dim
            device = next(strategizer.parameters()).device
            self.epsilon_in = torch.zeros(b, h, d, device=device)
            self.epsilon_rec = torch.zeros(b, h, h, device=device)
            self.epsilon_bias = torch.zeros(b, h, device=device)
            self.memory_jacobians = [torch.zeros(
                (b, k)+tuple(parameter.shape), device=device)
                for parameter in self.parameters]
            self.reward_traces = [torch.zeros(
                (b,)+tuple(parameter.shape), device=device)
                for parameter in self.parameters]
            self.optimizer = torch.optim.Adam(
                self.parameters, lr=learning_rate, maximize=True)
            core = strategizer.core
            self.core_parameter_kind = {
                id(core.input.weight): "input_weight",
                id(core.input.bias): "input_bias",
                id(core.recurrent.weight): "recurrent_weight",
                id(core.bias): "bias",
            }
            self.other_indices = [index for index, parameter in
                                  enumerate(self.parameters)
                                  if id(parameter) not in self.core_parameter_kind]
            self.last_recurrent_norm = 0.0
            self.last_memory_norm = 0.0
            self.last_score_norm = 0.0

        def _current_output_jacobians(
                self, output: torch.Tensor) -> List[torch.Tensor]:
            cfg, core = self.strategizer.cfg, self.strategizer.core
            b, output_dim, h = output.shape[0], output.shape[1], cfg.hidden_dim
            current = [torch.zeros(
                (b, output_dim)+tuple(parameter.shape), device=output.device)
                for parameter in self.parameters]
            if core.last_output is None:
                raise RuntimeError("strategizer core has no eligibility output")

            # Spatial learning signal from each strategy coordinate to the
            # strategizer's mean membrane/spike feature.  Batch elements are
            # independent, so differentiating the coordinate sum gives one local
            # derivative per world without mixing samples.
            feature_gradients = []
            for coordinate in range(output_dim):
                feature_gradients.append(torch.autograd.grad(
                    output[:, coordinate].sum(), core.last_output,
                    retain_graph=True)[0].detach())
            feature_gradient = torch.stack(feature_gradients, 1)

            if not self.persistent:
                self.epsilon_in.zero_()
                self.epsilon_rec.zero_()
                self.epsilon_bias.zero_()
            jacobian_in = torch.zeros(
                b, output_dim, h, self.epsilon_in.shape[-1], device=output.device)
            jacobian_rec = torch.zeros(
                b, output_dim, h, h, device=output.device)
            jacobian_bias = torch.zeros(b, output_dim, h, device=output.device)
            recurrent_mask = 1-torch.eye(h, device=output.device)
            ticks = max(len(core.last_eligibility_records), 1)
            with torch.no_grad():
                for value, presynaptic, pseudo_derivative in (
                        core.last_eligibility_records):
                    self.epsilon_in.mul_(core.decay).add_(value[:, None, :])
                    self.epsilon_rec.mul_(core.decay).add_(
                        presynaptic[:, None, :]*recurrent_mask[None])
                    self.epsilon_bias.mul_(core.decay).add_(1)
                    coefficient = (feature_gradient[:, :, :h]
                        +feature_gradient[:, :, h:]
                         *pseudo_derivative[:, None, :])/ticks
                    jacobian_in.add_(
                        coefficient[:, :, :, None]
                        *self.epsilon_in[:, None, :, :])
                    jacobian_rec.add_(
                        coefficient[:, :, :, None]
                        *self.epsilon_rec[:, None, :, :])
                    jacobian_bias.add_(
                        coefficient*self.epsilon_bias[:, None, :])

            # Feed-forward readout/conditioning parameters have an exact local
            # strategy-write Jacobian.  Its later consequences are still carried
            # exactly by the external memory recursion below.
            if self.other_indices:
                basis = torch.eye(b, device=output.device,
                                  dtype=output.dtype)
                other_parameters = [self.parameters[index]
                                    for index in self.other_indices]
                for coordinate in range(output_dim):
                    gradients = torch.autograd.grad(
                        output[:, coordinate], other_parameters,
                        grad_outputs=basis, is_grads_batched=True,
                        retain_graph=True, allow_unused=True)
                    for index, gradient in zip(self.other_indices, gradients):
                        if gradient is not None:
                            current[index][:, coordinate].copy_(gradient.detach())

            for index, parameter in enumerate(self.parameters):
                kind = self.core_parameter_kind.get(id(parameter))
                if kind == "input_weight":
                    current[index].copy_(jacobian_in)
                elif kind in ("input_bias", "bias"):
                    current[index].copy_(jacobian_bias)
                elif kind == "recurrent_weight":
                    current[index].copy_(jacobian_rec)
            self.last_recurrent_norm = float(torch.stack((
                jacobian_in.square().sum(), jacobian_rec.square().sum(),
                jacobian_bias.square().sum())).sum().sqrt())
            return current

        def accumulate(self, proposal: torch.Tensor, gate: torch.Tensor,
                       previous_strategy: torch.Tensor,
                       actor_strategy: torch.Tensor,
                       actor_logp: torch.Tensor, keep: float,
                       learned_gate: bool,
                       timer_logp: torch.Tensor | None = None) -> float:
            output_parts = [proposal]
            if learned_gate:
                output_parts.append(gate)
            if timer_logp is not None:
                output_parts.append(timer_logp[:, None])
            combined = torch.cat(output_parts, -1)
            combined_jacobians = self._current_output_jacobians(combined)
            k = proposal.shape[-1]
            proposal_jacobians = [value[:, :k]
                                  for value in combined_jacobians]
            offset = k
            if learned_gate:
                gate_jacobians = [value[:, offset:offset+k]
                                  for value in combined_jacobians]
                offset += k
                proposal_memory_jacobian = torch.stack([
                    torch.autograd.grad(
                        proposal[:, coordinate].sum(), previous_strategy,
                        retain_graph=True)[0].detach()
                    for coordinate in range(k)], 1)
                gate_memory_jacobian = torch.stack([
                    torch.autograd.grad(
                        gate[:, coordinate].sum(), previous_strategy,
                        retain_graph=True)[0].detach()
                    for coordinate in range(k)], 1)
                identity = torch.eye(k, device=proposal.device)[None]
                transition = ((1-gate)[:, :, None]*identity
                    +(proposal-previous_strategy)[:, :, None]
                     *gate_memory_jacobian
                    +gate[:, :, None]*proposal_memory_jacobian)
            else:
                gate_jacobians = []
            timer_jacobians = (
                [value[:, offset] for value in combined_jacobians]
                if timer_logp is not None else None
            )
            memory_signal = torch.autograd.grad(
                actor_logp.sum(), actor_strategy, retain_graph=True)[0].detach()
            memory_square = torch.zeros((), device=proposal.device)
            score_square = torch.zeros((), device=proposal.device)
            with torch.no_grad():
                for index, (memory, reward_trace, proposal_jacobian) in enumerate(
                        zip(self.memory_jacobians, self.reward_traces,
                            proposal_jacobians)):
                    if learned_gate:
                        shape = (proposal.shape[0], proposal.shape[1]) + (
                            1,)*(memory.ndim-2)
                        direct = (gate.view(shape)*proposal_jacobian
                            +(proposal-previous_strategy).view(shape)
                             *gate_jacobians[index])
                        propagated = torch.bmm(
                            transition, memory.reshape(
                                memory.shape[0], memory.shape[1], -1))
                        memory.copy_(propagated.reshape_as(memory)+direct)
                    else:
                        memory.mul_(keep).add_(
                            proposal_jacobian, alpha=1-keep)
                    view = (memory_signal.shape[0], memory_signal.shape[1]) + (
                        1,)*(memory.ndim-2)
                    score = (memory*memory_signal.view(view)).sum(1)
                    if timer_jacobians is not None:
                        score = score + timer_jacobians[index]
                    reward_trace.mul_(self.decay).add_(score)
                    memory_square += memory.square().sum()
                    score_square += score.square().sum()
            self.last_memory_norm = float(memory_square.sqrt())
            self.last_score_norm = float(score_square.sqrt())
            return float(torch.stack([trace.square().sum()
                                      for trace in self.reward_traces]).sum().sqrt())

        def apply(
                self,
                td_error: torch.Tensor,
                minimizing_gradients: Sequence[torch.Tensor | None] | None = None,
            ) -> Tuple[float, float]:

            self.optimizer.zero_grad(set_to_none=True)

            task_direction_square = torch.zeros(
                (),
                device=td_error.device,
            )

            for index, (parameter, trace) in enumerate(
                zip(self.parameters, self.reward_traces)
            ):
                view = (
                    td_error.shape[0],
                ) + (1,) * (trace.ndim - 1)

                # Reward-e-prop is a direction that should be maximized.
                task_direction = (
                    trace * td_error.view(view)
                ).mean(dim=0)

                combined_direction = task_direction

                # SIGReg is a loss that should be minimized. Because this
                # optimizer uses maximize=True, subtract its gradient.
                if minimizing_gradients is not None:
                    regularization_gradient = minimizing_gradients[index]

                    if regularization_gradient is not None:
                        combined_direction = (
                            combined_direction
                            - regularization_gradient
                        )

                parameter.grad = combined_direction

                # Preserve the existing task-gradient diagnostic.
                task_direction_square += task_direction.square().sum()

            before = [
                parameter.detach().clone()
                for parameter in self.parameters
            ]

            self.optimizer.step()

            step_square = torch.stack([
                (parameter - old).square().sum()
                for parameter, old in zip(self.parameters, before)
            ]).sum()

            return (
                float(task_direction_square.sqrt().detach()),
                float(step_square.sqrt().detach()),
            )

        def reset(self, mask: torch.Tensor) -> None:
            if not bool(mask.any()):
                return
            with torch.no_grad():
                self.epsilon_in[mask] = 0
                self.epsilon_rec[mask] = 0
                self.epsilon_bias[mask] = 0
                for memory, trace in zip(
                        self.memory_jacobians, self.reward_traces):
                    memory[mask] = 0
                    trace[mask] = 0


    class StrategyEncoderEprop:
        """Carries strategy-memory credit back to the encoder latent head.

        The trace follows:

            encoder parameters
                -> current latent
                -> strategy proposal/gate
                -> persistent strategy memory
                -> later actor log probability
                -> delayed TD error
        """

        def __init__(
                self,
                encoder: StatelessEncoder,
                worlds: int,
                strategy_dim: int,
                decay: float,
                learning_rate: float,
                gradient_clip: float,
            ) -> None:
            self.encoder = encoder
            self.decay = decay
            self.gradient_clip = gradient_clip

            # Start with only the final encoder projection.
            self.parameters = [
                encoder.latent_head.weight,
                encoder.latent_head.bias,
            ]

            device = encoder.latent_head.weight.device

            # dm/dtheta for every world and strategy-memory coordinate.
            self.memory_jacobians = [
                torch.zeros(
                    (worlds, strategy_dim) + tuple(parameter.shape),
                    device=device,
                )
                for parameter in self.parameters
            ]

            # Reward eligibility:
            # decay * old_trace + d(log pi)/dtheta
            self.reward_traces = [
                torch.zeros(
                    (worlds,) + tuple(parameter.shape),
                    device=device,
                )
                for parameter in self.parameters
            ]

            self.optimizer = torch.optim.Adam(
                self.parameters,
                lr=learning_rate,
                maximize=True,
            )

            self.last_memory_norm = 0.0
            self.last_score_norm = 0.0
            self.last_eligibility_norm = 0.0
            self.last_gradient_norm = 0.0

        def _output_jacobians(
                self,
                output: torch.Tensor,
            ) -> list[torch.Tensor]:

            if not output.requires_grad:
                raise RuntimeError(
                    "StrategyEncoderEprop requires an attached encoder graph"
                )

            batch = output.shape[0]
            output_dim = output.shape[1]

            basis = torch.eye(
                batch,
                device=output.device,
                dtype=output.dtype,
            )

            jacobians = [
                torch.zeros(
                    (batch, output_dim) + tuple(parameter.shape),
                    device=output.device,
                    dtype=output.dtype,
                )
                for parameter in self.parameters
            ]

            for coordinate in range(output_dim):
                gradients = torch.autograd.grad(
                    output[:, coordinate],
                    self.parameters,
                    grad_outputs=basis,
                    is_grads_batched=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                for jacobian, gradient in zip(jacobians, gradients):
                    if gradient is not None:
                        jacobian[:, coordinate].copy_(gradient.detach())

            return jacobians

        def accumulate(
                self,
                proposal: torch.Tensor,
                gate: torch.Tensor,
                previous_strategy: torch.Tensor,
                actor_strategy: torch.Tensor,
                actor_logp: torch.Tensor,
                learned_gate: bool,
                keep: float,
            ) -> float:
            """Advance strategizer-to-encoder eligibility by one environment step.

            Carries:

                encoder parameters
                    -> strategy proposal and write gate
                    -> persistent strategy memory
                    -> actor log probability
                    -> reward eligibility trace

            The TD error is applied later by ``apply()``.
            """

            batch = proposal.shape[0]
            strategy_dim = proposal.shape[1]

            # ---------------------------------------------------------------
            # 1. Current proposal/gate Jacobians with respect to the encoder.
            # ---------------------------------------------------------------

            if learned_gate:
                combined = torch.cat((proposal, gate), dim=-1)

                combined_jacobians = self._output_jacobians(combined)

                proposal_jacobians = [
                    jacobian[:, :strategy_dim]
                    for jacobian in combined_jacobians
                ]

                gate_jacobians = [
                    jacobian[:, strategy_dim:]
                    for jacobian in combined_jacobians
                ]

                # -----------------------------------------------------------
                # 2. Strategy-memory transition Jacobian:
                #
                #       dm_t / dm_{t-1}
                #
                # memory:
                #
                #   m_t = (1-g_t)m_{t-1} + g_t p_t
                #
                # proposal and gate may both depend on previous memory.
                # -----------------------------------------------------------

                proposal_memory_jacobian = torch.stack(
                    [
                        torch.autograd.grad(
                            proposal[:, coordinate].sum(),
                            previous_strategy,
                            retain_graph=True,
                        )[0].detach()
                        for coordinate in range(strategy_dim)
                    ],
                    dim=1,
                )

                gate_memory_jacobian = torch.stack(
                    [
                        torch.autograd.grad(
                            gate[:, coordinate].sum(),
                            previous_strategy,
                            retain_graph=True,
                        )[0].detach()
                        for coordinate in range(strategy_dim)
                    ],
                    dim=1,
                )

                identity = torch.eye(
                    strategy_dim,
                    device=proposal.device,
                    dtype=proposal.dtype,
                )[None]

                transition = (
                    (1.0 - gate)[:, :, None] * identity
                    + (proposal - previous_strategy)[:, :, None]
                    * gate_memory_jacobian
                    + gate[:, :, None]
                    * proposal_memory_jacobian
                )

            else:
                # In the fixed-memory case:
                #
                #   m_t = keep*m_{t-1} + (1-keep)*proposal
                #
                # so:
                #
                #   dm_t/dm_{t-1} = keep*I
                #   direct encoder contribution = (1-keep)*dp/dtheta
                proposal_jacobians = self._output_jacobians(proposal)
                gate_jacobians = []

                identity = torch.eye(
                    strategy_dim,
                    device=proposal.device,
                    dtype=proposal.dtype,
                )[None]

                transition = keep * identity

            # ---------------------------------------------------------------
            # 3. Actor learning signal with respect to strategy memory.
            #
            #       d log(pi_t) / d m_t
            #
            # This tells us which strategy-memory directions affected the
            # action selected by the actor.
            # ---------------------------------------------------------------

            memory_signal = torch.autograd.grad(
                actor_logp.sum(),
                actor_strategy,
                retain_graph=True,
            )[0].detach()

            if memory_signal.shape != (batch, strategy_dim):
                raise RuntimeError(
                    "Unexpected actor-strategy gradient shape: "
                    f"expected {(batch, strategy_dim)}, "
                    f"received {tuple(memory_signal.shape)}"
                )

            memory_square = torch.zeros(
                (),
                device=proposal.device,
                dtype=proposal.dtype,
            )

            score_square = torch.zeros(
                (),
                device=proposal.device,
                dtype=proposal.dtype,
            )

            # ---------------------------------------------------------------
            # 4. Advance dm/dtheta and contract it with dlog(pi)/dm.
            # ---------------------------------------------------------------

            with torch.no_grad():
                for index, (
                    memory_jacobian,
                    reward_trace,
                    proposal_jacobian,
                ) in enumerate(
                    zip(
                        self.memory_jacobians,
                        self.reward_traces,
                        proposal_jacobians,
                    )
                ):
                    # memory_jacobian has shape:
                    #
                    #   [batch, strategy_dim, *parameter_shape]
                    #
                    # For example, latent-head weights produce:
                    #
                    #   [batch, strategy_dim, latent_dim, hidden_dim]

                    parameter_dimensions = memory_jacobian.ndim - 2

                    # -------------------------------------------------------
                    # Direct current-time encoder contribution:
                    #
                    # learned gate:
                    #
                    #   g * dp/dtheta
                    #   + (p-m_previous) * dg/dtheta
                    #
                    # fixed retention:
                    #
                    #   (1-keep) * dp/dtheta
                    # -------------------------------------------------------

                    if learned_gate:
                        broadcast_shape = (
                            batch,
                            strategy_dim,
                        ) + (1,) * parameter_dimensions

                        direct = (
                            gate.view(broadcast_shape)
                            * proposal_jacobian
                            + (proposal - previous_strategy).view(
                                broadcast_shape
                            )
                            * gate_jacobians[index]
                        )
                    else:
                        direct = (
                            (1.0 - keep)
                            * proposal_jacobian
                        )

                    # -------------------------------------------------------
                    # Propagate earlier encoder influence through memory:
                    #
                    #   transition @ dm_previous/dtheta
                    #
                    # Flatten parameter axes temporarily so torch.bmm can
                    # apply the strategy-memory transition independently to
                    # every parameter derivative.
                    # -------------------------------------------------------

                    flattened_memory = memory_jacobian.reshape(
                        batch,
                        strategy_dim,
                        -1,
                    )

                    propagated = torch.bmm(
                        transition,
                        flattened_memory,
                    ).reshape_as(memory_jacobian)

                    memory_jacobian.copy_(
                        propagated + direct
                    )

                    # -------------------------------------------------------
                    # Policy score for each encoder parameter:
                    #
                    #   dlog(pi)/dtheta
                    #       =
                    #   sum_k [
                    #       dlog(pi)/dm_k
                    #       * dm_k/dtheta
                    #   ]
                    # -------------------------------------------------------

                    signal_shape = (
                        batch,
                        strategy_dim,
                    ) + (1,) * parameter_dimensions

                    score = (
                        memory_jacobian
                        * memory_signal.view(signal_shape)
                    ).sum(dim=1)

                    # -------------------------------------------------------
                    # Long-running reward eligibility:
                    #
                    #   E_t = decay*E_{t-1} + dlog(pi_t)/dtheta
                    #
                    # The delayed TD error will be applied by ``apply()``.
                    # -------------------------------------------------------

                    reward_trace.mul_(self.decay).add_(score)

                    memory_square += memory_jacobian.square().sum()
                    score_square += score.square().sum()

            # ---------------------------------------------------------------
            # 5. Diagnostics.
            # ---------------------------------------------------------------

            self.last_memory_norm = float(
                memory_square.sqrt().detach()
            )

            self.last_score_norm = float(
                score_square.sqrt().detach()
            )

            trace_square = torch.stack(
                [
                    trace.square().sum()
                    for trace in self.reward_traces
                ]
            ).sum()

            self.last_eligibility_norm = float(
                trace_square.sqrt().detach()
            )

            return self.last_eligibility_norm

        def apply(
                self,
                td_error: torch.Tensor,
            ) -> tuple[float, float]:

            self.optimizer.zero_grad(set_to_none=True)

            gradient_square = torch.zeros(
                (),
                device=td_error.device,
            )

            for parameter, trace in zip(
                self.parameters,
                self.reward_traces,
            ):
                view = (
                    td_error.shape[0],
                ) + (1,) * (trace.ndim - 1)

                gradient = (
                    trace * td_error.view(view)
                ).mean(dim=0)

                parameter.grad = gradient
                gradient_square += gradient.square().sum()

            raw_gradient_norm = gradient_square.sqrt()

            if self.gradient_clip > 0:
                scale = torch.clamp(
                    self.gradient_clip
                    / raw_gradient_norm.clamp_min(1e-12),
                    max=1.0,
                )

                for parameter in self.parameters:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)

            self.last_gradient_norm = float(
                torch.minimum(
                    raw_gradient_norm,
                    torch.as_tensor(
                        self.gradient_clip,
                        device=raw_gradient_norm.device,
                    ),
                )
                if self.gradient_clip > 0
                else raw_gradient_norm
            )

            before = [
                parameter.detach().clone()
                for parameter in self.parameters
            ]

            self.optimizer.step()

            step_square = torch.stack([
                (parameter - old).square().sum()
                for parameter, old in zip(self.parameters, before)
            ]).sum()

            return (
                self.last_gradient_norm,
                float(step_square.sqrt().detach()),
            )

        def reset(self, mask: torch.Tensor) -> None:
            if not bool(mask.any()):
                return

            with torch.no_grad():
                for memory_jacobian in self.memory_jacobians:
                    memory_jacobian[mask] = 0

                for reward_trace in self.reward_traces:
                    reward_trace[mask] = 0



    return Config, BatchedTMaze, SurrogateSpike, RecurrentSNN, mlp, RewardAdaptiveLayerNorm, StatelessEncoder, Strategizer, Actor, Predictor, RepresentationCritic, module_parameters, RewardEprop, PredictorEprop, RecurrentStrategyEprop, StrategyEncoderEprop


@app.cell
def _(dataclass, Config, BatchedTMaze, nn, torch, F, np, RecurrentSNN,
      StatelessEncoder, Strategizer, Actor, Predictor, RewardEprop, PredictorEprop,
      RecurrentStrategyEprop, StrategyEncoderEprop, module_parameters, mlp):
    import copy

    @dataclass
    class Settings:
        seed: int = 12
        slots: int = 4
        prediction_horizons: tuple = (1, 2, 4, 8, 16)
        max_prediction_age: int = 16
        strategy_min_std: float = .05
        strategy_entropy_weight: float = .001
        adaptive_strategy_exploration: bool = False
        adaptive_actor_exploration: bool = False
        exploration_window: int = 100
        exploration_warmup: int = 20
        exploration_target_success: float = .8
        strategy_exploration_max: float = 2.
        actor_temperature_max: float = 2.
        cue_success_bonus: bool = False
        cue_bonus_scale: float = .1
        cue_bonus_window: int = 100
        cue_bonus_min_samples: int = 10
        prediction_shaping_beta: float = 0.
        balanced_cue_batches: bool = True
        curriculum_blend_episodes: int = 64
        strategy_conditioning: bool = True
        stochastic_strategy: bool = True
        encoder_eprop: bool = True
        adaptive_strategy_lr: bool = False
        slowdown_trigger_source: str = "prediction"
        strategy_lr_min_scale: float = .1
        prediction_change_window: int = 20
        prediction_change_warmup: int = 8
        prediction_change_threshold: float = 3.
        prediction_change_min_increase: float = .15
        strategy_lr_hold_episodes: int = 5
        strategy_lr_recovery_episodes: int = 10
        prediction_stable_episodes: int = 3


    class BalancedMaze(BatchedTMaze):
        """Shuffled L/R pairs across resets, including single-world resets."""
        def __init__(self, cfg, seed, curriculum=True, balanced=True):
            self.balanced = balanced
            self.cue_deck = []
            super().__init__(cfg, seed, curriculum)

        def reset(self, mask):
            # Canonical placement, episode bookkeeping, and curriculum retained.
            previous_counts = getattr(self, "cue_assignment_counts", None)
            previous_counts = previous_counts.copy() if previous_counts is not None else None
            previous_probability = getattr(self, "cue_assignment_probability_sum", 0.)
            super().reset(mask)
            if self.balanced:
                assigned = []
                for _ in range(int(mask.sum())):
                    if not self.cue_deck:
                        self.cue_deck = self.rng.permutation(2).tolist()
                    assigned.append(self.cue_deck.pop())
                self.cue[mask] = assigned
                if previous_counts is not None:
                    self.cue_assignment_counts = previous_counts + np.bincount(assigned, minlength=2)
                    self.cue_assignment_probability_sum = previous_probability + .5*len(assigned)

    class ExternalValueCritic(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # z, clean memory, mean hidden/delta/intent, age, occupancy
            size = 2*cfg.latent_dim+2*cfg.strategy_dim+2*cfg.hidden_dim+2
            self.core = RecurrentSNN(size,cfg.hidden_dim,cfg,persistent=False)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.head = nn.Linear(2*cfg.hidden_dim,1)
            nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        def forward(self, state):
            return self.head(self.norm(self.core(state.detach()))).flatten()

    class PredictionManager(nn.Module):
        """Legacy initializer only: preserves historical RNG draws, never run."""
        def __init__(self,cfg):
            super().__init__()
            self.core = RecurrentSNN(2*cfg.hidden_dim+2*cfg.latent_dim+
                cfg.strategy_dim+1,cfg.hidden_dim,cfg,persistent=False)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.head = nn.Linear(2*cfg.hidden_dim,3)
        def forward(self,hidden,z,intent,prediction,age):
            context = torch.cat((hidden,z,intent,prediction,age),-1).detach()
            return self.head(self.norm(self.core(context)))

    class PredictiveActor(Actor):
        """Preserve Ghost actor SNN/readout; explicitly extend its conditioning."""
        def __init__(self,cfg):
            super().__init__(cfg,persistent=False)
            self.strategy_encoder = mlp(cfg.strategy_dim+2*cfg.latent_dim+3,
                cfg.hidden_dim,cfg.conditioning_dim)
        def forward(self,latent,strategy,context,deterministic=False,temperature=1.):
            latent,strategy,context = latent.detach(),strategy.detach(),context.detach()
            conditioning = self.strategy_encoder(torch.cat((strategy,context),-1))
            inputs = [latent,conditioning]
            if self.cfg.learned_strategy_memory:
                inputs.append(strategy)
            logits = self.head(self.norm(self.core(torch.cat(inputs,-1))))
            if not deterministic and temperature != 1.:
                logits = logits / temperature
            distribution = torch.distributions.Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else distribution.sample()
            return dict(action=action,logp=distribution.log_prob(action),
                logits=logits,entropy=distribution.entropy())

    class PredictiveAgent(nn.Module):
        def __init__(self,settings):
            super().__init__()
            self.settings = settings
            if (not np.isfinite(settings.cue_bonus_scale) or settings.cue_bonus_scale < 0
                    or settings.cue_bonus_min_samples < 1
                    or settings.cue_bonus_window < 2*settings.cue_bonus_min_samples):
                raise ValueError("Cue bonus requires a finite nonnegative scale and a window covering both cue minimums")
            self.cue_success_history = []
            if (not 1 <= settings.exploration_warmup <= settings.exploration_window
                    or not 0 < settings.exploration_target_success <= 1
                    or not 1 <= settings.strategy_exploration_max < float("inf")
                    or not 1 <= settings.actor_temperature_max < float("inf")):
                raise ValueError("Invalid adaptive exploration window, warmup, target, or maximum")
            self.exploration_success_history = []
            self.strategy_exploration_scale = 1.
            self.actor_temperature = 1.
            horizons = tuple(int(h) for h in settings.prediction_horizons)
            if not horizons or any(h<1 for h in horizons) or len(set(horizons))!=len(horizons):
                raise ValueError("prediction_horizons must be distinct positive integers")
            if settings.slots<1 or settings.max_prediction_age<1 or settings.strategy_min_std<=0:
                raise ValueError("slots, maximum age, and minimum strategy std must be positive")
            if settings.prediction_shaping_beta<0:
                raise ValueError("Invalid shaping coefficient")
            if not (0 < settings.strategy_lr_min_scale <= 1
                    and 3 <= settings.prediction_change_warmup <= settings.prediction_change_window
                    and 1 < settings.prediction_change_threshold < float("inf")
                    and 0 < settings.prediction_change_min_increase < float("inf")
                    and settings.strategy_lr_hold_episodes >= 1
                    and settings.strategy_lr_recovery_episodes >= 1
                    and settings.prediction_stable_episodes >= 1):
                raise ValueError("Invalid loss change detector settings")
            if settings.slowdown_trigger_source not in ("prediction","critic"):
                raise ValueError("slowdown_trigger_source must be prediction or critic")
            self.prediction_loss_history = ({h:[] for h in horizons}
                if settings.slowdown_trigger_source=="prediction" else {"critic":[]})
            self.prediction_episode_index = 0
            self.prediction_change_state = "armed"
            self.prediction_hold_remaining = 0
            self.prediction_stable_count = 0
            self.prediction_recovery_count = 0
            self.strategy_lr_scale = 1.
            self.cfg = Config(seed=settings.seed,worlds=1,encoder_learning_mode="reward_eprop",
                curriculum_blend_episodes=settings.curriculum_blend_episodes,
                use_reward_adaln=False)
            cfg = self.cfg
            self.encoder = StatelessEncoder(cfg)
            self.strategizer = Strategizer(cfg,persistent=True)
            self.actor = PredictiveActor(cfg)
            # Reuse Ghost's existing horizon input, assigned exogenously here.
            predictor_cfg = copy.deepcopy(cfg)
            predictor_cfg.use_strategic_prediction_timer = True
            predictor_cfg.prediction_timer_durations = horizons
            self.predictor = Predictor(predictor_cfg)
            # Consume precisely the historical initialization draws, then release
            # the legacy module before constructing the critic. Never run it.
            _legacy_manager = PredictionManager(cfg)
            del _legacy_manager
            self.critic = ExternalValueCritic(cfg)
            initial_excess = max(.20-settings.strategy_min_std,.01)
            self.strategy_logstd = nn.Parameter(torch.full((cfg.strategy_dim,),
                float(np.log(np.expm1(initial_excess)))))
            self.actor_credit = RewardEprop(list(self.actor.parameters()),1,
                cfg.actor_trace_decay,cfg.actor_eprop_lr)
            parameters = module_parameters([self.strategizer.core,self.strategizer.norm,
                self.strategizer.strategy_head,self.strategizer.gate_head,self.strategizer.feedback_encoder])
            self.strategy_credit = RecurrentStrategyEprop(self.strategizer,parameters,1,
                cfg.strategy_trace_decay,cfg.strategy_eprop_lr,persistent=True)
            self.encoder_credit = StrategyEncoderEprop(self.encoder,1,cfg.strategy_dim,
                cfg.strategy_encoder_trace_decay,cfg.strategy_encoder_eprop_lr,cfg.strategy_encoder_eprop_clip)
            self.std_credit = RewardEprop([self.strategy_logstd],1,cfg.strategy_trace_decay,cfg.strategy_eprop_lr)
            self.predictor_credit = PredictorEprop(self.predictor,1,cfg.predictor_trace_decay,0.)
            self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(),lr=cfg.predictor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=cfg.critic_lr)
            # This RNG is independent of policy draws.
            self.horizon_rng = np.random.default_rng(settings.seed+700001)
            self.reliability = {h:0. for h in horizons}
            self.reset_episode()

        def prepare_exploration(self, train):
            # Freeze controls for this episode; only past raw success is evidence.
            s = self.settings
            pressure = 0.
            if train and len(self.exploration_success_history) >= s.exploration_warmup:
                rate = sum(self.exploration_success_history)/len(self.exploration_success_history)
                pressure = max(0., (s.exploration_target_success-rate)/s.exploration_target_success)
            self.strategy_exploration_scale = (1.+(s.strategy_exploration_max-1.)*pressure
                if s.adaptive_strategy_exploration and s.stochastic_strategy else 1.)
            self.actor_temperature = (1.+(s.actor_temperature_max-1.)*pressure
                if s.adaptive_actor_exploration else 1.)

        def record_exploration_success(self, success):
            self.exploration_success_history.append(float(success))
            self.exploration_success_history = self.exploration_success_history[-self.settings.exploration_window:]

        def success_bonus(self, cue, success, train):
            if not train or not success or not self.settings.cue_success_bonus:
                return 0.
            groups = [[outcome for c,outcome in self.cue_success_history if c == side]
                      for side in (0, 1)]
            if min(map(len, groups)) < self.settings.cue_bonus_min_samples:
                return 0.
            rates = [sum(group)/len(group) for group in groups]
            return self.settings.cue_bonus_scale * max(0., rates[1-cue]-rates[cue])

        def record_cue_success(self, cue, success):
            # Episode reset and curriculum transitions must not clear this window.
            self.cue_success_history.append((int(cue), float(success)))
            self.cue_success_history = self.cue_success_history[-self.settings.cue_bonus_window:]

        def adapt_strategy_lr(self,horizon_accuracy,critic_loss=None):
            """Detect an episode loss shift without any environment/stage input.

            Fit preceding per-horizon prediction trends or one critic-loss trend
            before inserting this episode.
            Pool normalized deviations by median so horizon counts do not reweight
            the signal. Keep collecting history while latched; never clear it.
            Returned LR applies to the NEXT episode, after targets are observed.
            """
            s=self.settings
            self.prediction_episode_index+=1
            index=self.prediction_episode_index
            scores=[];details=[]
            evidence=horizon_accuracy
            if s.slowdown_trigger_source=="critic":
                # Episode mean of unclipped, pre-update squared TD error.
                evidence={} if critic_loss is None else {"critic":dict(count=1,mse_sum=critic_loss)}
            for horizon,entry in evidence.items():
                if not entry["count"]:
                    continue  # Censored/missing forecasts are not zero loss.
                loss=entry["mse_sum"]/entry["count"]
                history=self.prediction_loss_history[horizon]
                if len(history)>=s.prediction_change_warmup:
                    x=np.asarray([i-index for i,_ in history],dtype=float)
                    y=np.asarray([v for _,v in history],dtype=float)
                    slope,intercept=np.polyfit(x,y,1)
                    baseline=max(float(intercept),1e-8)
                    noise=float(np.sqrt(np.mean((y-(slope*x+intercept))**2)))
                    # Require both an unusual deviation and a meaningful rise.
                    scale=max(noise,baseline*s.prediction_change_min_increase/
                        s.prediction_change_threshold,1e-8)
                    score=(loss-baseline)/scale
                    scores.append(score)
                    details.append(dict(horizon=horizon,loss=loss,baseline=baseline,
                        trigger_loss=baseline+s.prediction_change_threshold*scale,score=score))
                history.append((index,loss))
                del history[:-s.prediction_change_window]
            score=float(np.median(scores)) if scores else None
            triggered=False
            if score is not None and s.adaptive_strategy_lr:
                stable=abs(score)<1.
                self.prediction_stable_count=self.prediction_stable_count+1 if stable else 0
                if self.prediction_change_state=="armed" and score>=s.prediction_change_threshold:
                    triggered=True
                    self.prediction_change_state="hold"
                    self.prediction_hold_remaining=s.strategy_lr_hold_episodes
                    self.prediction_stable_count=0
                    self.strategy_lr_scale=s.strategy_lr_min_scale
                elif self.prediction_change_state=="hold":
                    self.prediction_hold_remaining=max(0,self.prediction_hold_remaining-1)
                    if self.prediction_hold_remaining==0 and self.prediction_stable_count>=s.prediction_stable_episodes:
                        self.prediction_change_state="recover"
                        self.prediction_recovery_count=0
                elif self.prediction_change_state=="recover":
                    if score>=s.prediction_change_threshold:
                        # Same unresolved event: protect again without retriggering.
                        self.prediction_change_state="hold"
                        self.prediction_hold_remaining=s.strategy_lr_hold_episodes
                        self.strategy_lr_scale=s.strategy_lr_min_scale
                    else:
                        self.prediction_recovery_count+=1
                        fraction=min(1.,self.prediction_recovery_count/s.strategy_lr_recovery_episodes)
                        self.strategy_lr_scale=s.strategy_lr_min_scale+(1-s.strategy_lr_min_scale)*fraction
                        if fraction==1. and self.prediction_stable_count>=s.prediction_stable_episodes:
                            self.prediction_change_state="armed"
            if not s.adaptive_strategy_lr:
                self.strategy_lr_scale=1.
            # Scale Adam step sizes, preserving eligibility and optimizer moments.
            for credit,base_lr in ((self.strategy_credit,self.cfg.strategy_eprop_lr),
                    (self.encoder_credit,self.cfg.strategy_encoder_eprop_lr),
                    (self.std_credit,self.cfg.strategy_eprop_lr)):
                for group in credit.optimizer.param_groups:
                    group["lr"]=base_lr*self.strategy_lr_scale
            return dict(slowdown_trigger_source=s.slowdown_trigger_source,
                prediction_change_score=score,prediction_change_triggered=triggered,
                prediction_change_state=self.prediction_change_state if score is not None else "insufficient evidence",
                prediction_change_threshold=s.prediction_change_threshold,
                prediction_change_details=details,strategy_lr_next_scale=self.strategy_lr_scale)

        def reset_episode(self):
            cfg,k = self.cfg,self.settings.slots
            self.memory = torch.zeros(1,cfg.strategy_dim)
            self.active = torch.zeros(k,dtype=torch.bool)
            self.predictions = torch.zeros(k,cfg.latent_dim)
            self.hidden = torch.zeros(k,2*cfg.hidden_dim)
            self.intents = torch.zeros(k,cfg.strategy_dim)
            self.origins = torch.zeros(k,cfg.latent_dim)
            self.ages = torch.zeros(k,dtype=torch.long)
            self.slot_horizons = torch.full((k,),int(self.settings.prediction_horizons[0]),dtype=torch.long)
            self.slot_ids = torch.full((k,),-1,dtype=torch.long)
            self.next_slot = 0
            self.step_index = 0
            self.queue = []
            self.strategizer.core.initial(1,torch.device("cpu"))
            self.predictor.core.initial(1,torch.device("cpu"))
            self.actor.core.reset()
            for c in (self.actor_credit,self.strategy_credit,self.encoder_credit,self.std_credit,self.predictor_credit):
                c.reset(torch.ones(1,dtype=torch.bool))

        def sample_strategy(self,clean,train):
            # Clean gated memory stays bounded. Sample in pre-tanh coordinates.
            mean = torch.atanh(clean.clamp(-.999,.999))
            std = self.settings.strategy_min_std + F.softplus(self.strategy_logstd)
            if train and self.settings.adaptive_strategy_exploration and self.settings.stochastic_strategy:
                std = std * self.strategy_exploration_scale
            distribution = torch.distributions.Normal(mean,std)
            if train and self.settings.stochastic_strategy:
                raw = distribution.sample()  # detached action for score-function credit
                strategy = torch.tanh(raw)
                logp = (distribution.log_prob(raw)-
                    torch.log(1-strategy.square()+1e-6)).sum(-1)
            else:
                strategy = torch.tanh(mean)
                logp = mean.sum(-1)*0. + std.sum()*0.
            return strategy,logp,std,distribution.entropy().sum(-1)

        def meta_state(self,z):
            weight = self.active.float()[:,None]
            n = weight.sum().clamp_min(1)
            pooled = torch.cat(((self.hidden*weight).sum(0,keepdim=True)/n,
                ((self.predictions-z.detach())*weight).sum(0,keepdim=True)/n,
                (self.intents*weight).sum(0,keepdim=True)/n,
                (self.ages.float()[:,None]*weight).sum(0,keepdim=True)/n/self.settings.max_prediction_age,
                weight.mean().reshape(1,1)),-1)
            return torch.cat((z.detach(),self.memory.detach(),pooled),-1).detach()

        def manage(self,z,train):
            expired = self.active & (self.ages>=self.settings.max_prediction_age)
            self.active[expired] = False
            incoming = self.active.clone()
            choice = torch.ones_like(self.ages)
            probs = z.new_zeros((self.settings.slots, 3))
            probs[:, 1] = 1.  # Fixed USE rule for all eligible predictions.
            use = incoming
            count = use.sum().clamp_min(1)
            target = ((self.predictions*use[:,None]).sum(0,keepdim=True)/count).detach()
            age = (self.ages.float()*use).sum()/count/self.settings.max_prediction_age
            error = z.new_tensor([self.reliability[int(h)] for h in self.slot_horizons])
            reliability = (error*use).sum()/count
            delta = (target-z.detach()) if bool(use.any()) else torch.zeros_like(z)
            context = torch.cat((delta,-delta,age.reshape(1,1),
                use.float().mean().reshape(1,1),reliability.reshape(1,1)),-1).detach()
            return dict(target=target,use=use,context=context,incoming=incoming,
                choice=choice.detach(),probs=probs,entropy=0.,
                entropy_grad=None,discard=0,expired=int(expired.sum()))

        @staticmethod
        def reward_step(credit,advantage,entropy_grad=None,entropy_weight=0.):
            original = credit.traces
            try:
                if entropy_grad is not None:
                    # Add entropy ascent separately from reward-modulated eligibility.
                    credit.optimizer.zero_grad(set_to_none=True)
                    for p,t,g in zip(credit.parameters,original,entropy_grad):
                        view=(advantage.shape[0],)+(1,)*(t.ndim-1)
                        p.grad=(t*advantage.view(view)).mean(0)+entropy_weight*g.detach()
                    credit.optimizer.step()
                else:
                    credit.apply(advantage)
            finally:
                credit.traces=original

        def capture_prediction_jacobians(self,prediction):
            """Freeze creation derivatives including canonical recurrent e-prop.

            Each output coordinate uses the SAME pre-step epsilon state.
            Eligibility advances once, while no temporal autograd graph is saved.
            """
            credit=self.predictor_credit
            names=("epsilon_in","epsilon_rec","epsilon_bias")
            base=[getattr(credit,n).clone() for n in names]
            params=list(self.predictor.parameters())
            jac=[[] for _ in params]
            core_params={id(self.predictor.core.input.weight):"input_weight",
                id(self.predictor.core.input.bias):"input_bias",
                id(self.predictor.core.recurrent.weight):"recurrent_weight",
                id(self.predictor.core.bias):"bias"}
            for coordinate in range(self.cfg.latent_dim):
                for name,value in zip(names,base):
                    getattr(credit,name).copy_(value)
                scalar=prediction[0,coordinate]
                temporal=credit.gradients(scalar)
                local=torch.autograd.grad(scalar,params,retain_graph=True,allow_unused=True)
                for dst,p,g in zip(jac,params,local):
                    kind=core_params.get(id(p))
                    dst.append((temporal[kind] if kind else
                        g if g is not None else torch.zeros_like(p)).detach().clone())
            return [torch.stack(values) for values in jac]

        def create_prediction(self,z,strategy,action,train):
            horizon_index=int(self.horizon_rng.integers(len(self.settings.prediction_horizons)))
            horizon=int(self.settings.prediction_horizons[horizon_index])
            intent=strategy.detach() if self.settings.strategy_conditioning else torch.zeros_like(strategy)
            prediction=self.predictor(z.detach(),intent,torch.zeros(1),action.detach(),
                torch.tensor([horizon_index]))
            jacobians=self.capture_prediction_jacobians(prediction) if train else None
            # Creation is unconditional: bank fullness/management cannot censor data.
            self.queue.append(dict(prediction=prediction.detach().clone(),jacobians=jacobians,
                creation_step=self.step_index,target_step=self.step_index+horizon,horizon=horizon,
                origin=z.detach().clone(),intent=intent.clone(),action=action.detach().clone()))
            hidden=torch.cat((self.predictor.core.mem,self.predictor.core.spk),-1).detach()
            free=[(self.next_slot+i)%self.settings.slots for i in range(self.settings.slots)
                if not self.active[(self.next_slot+i)%self.settings.slots]]
            if free:
                i=free[0];self.next_slot=(i+1)%self.settings.slots
                self.active[i]=True;self.ages[i]=0;self.slot_ids[i]=self.step_index
                self.predictions[i]=prediction.detach()[0];self.hidden[i]=hidden[0]
                self.intents[i]=intent[0];self.origins[i]=z.detach()[0];self.slot_horizons[i]=horizon
            return bool(free)

        def supervise(self,observation,train):
            due=[r for r in self.queue if r["target_step"]==self.step_index]
            if any(r["target_step"]<self.step_index for r in self.queue):
                raise RuntimeError("Missed scheduled supervision")
            self.queue=[r for r in self.queue if r["target_step"]!=self.step_index]
            if not due:
                return []
            with torch.no_grad():
                target=self.encoder.encode(observation).detach()
            errors=[]
            gradients=[torch.zeros_like(p) for p in self.predictor.parameters()]
            for record in due:
                residual=record["prediction"]-target
                mse=float(residual.square().mean())
                errors.append(dict(horizon=record["horizon"],mse=mse,
                    creation_step=record["creation_step"],target_step=self.step_index))
                if train:
                    coefficient=2*residual[0]/self.cfg.latent_dim
                    for dst,jac in zip(gradients,record["jacobians"]):
                        dst.add_(torch.tensordot(coefficient,jac,dims=1)/len(due))
                    h=record["horizon"]
                    self.reliability[h]=.98*self.reliability[h]+.02*mse
            if train:
                self.predictor_optimizer.zero_grad(set_to_none=True)
                for p,g in zip(self.predictor.parameters(),gradients):
                    p.grad=g
                # Preserve canonical core gradient clipping after contraction.
                core=self.predictor.core
                core_params=[core.input.weight,core.input.bias,core.recurrent.weight,core.bias]
                torch.nn.utils.clip_grad_norm_(core_params,self.cfg.predictor_eprop_clip)
                self.predictor_optimizer.step()
            return errors

        @staticmethod
        def actor_reward(reward,gamma,beta,z,next_z,target,use,done,carried_phi):
            # Target is fixed across the physical transition. A boundary
            # correction accounts for switching targets since the last step.
            now=-((z-target).square().mean()) if use else z.new_zeros(())
            following=(-((next_z-target).square().mean()) if use and not done
                       else z.new_zeros(()))
            base=gamma*following-now
            switch_correction=now-carried_phi
            shaping=base+switch_correction
            return reward+beta*shaping,following,shaping,switch_correction

    return Settings, BalancedMaze, ExternalValueCritic, PredictionManager, PredictiveActor, PredictiveAgent

@app.cell
def _(torch,np,BatchedTMaze):
    def episode(agent,env,train=False):
        agent.train(train)
        agent.reset_episode()
        agent.prepare_exploration(train)
        cfg,s=agent.cfg,agent.settings
        cue=int(env.cue[0])
        episode_stage=int(env.episode_stage[0])+1
        episode_blend=float(env.episode_blend[0])
        sampled_distribution=env.current_length_distribution
        hallway_length=int(env.hallway_length[0])
        episode_time_limit=int(env.episode_time_limit[0])
        carried_phi=torch.zeros(())
        frames=[];validations=[];bank_creations=0;discarded=0;expired=0
        horizon_created={h:0 for h in s.prediction_horizons}
        for step in range(episode_time_limit):
            observation=env.observation().copy()
            z=agent.encoder.encode(torch.from_numpy(observation))
            bank_before=agent.predictions.numpy().copy()
            ages_before=agent.ages.numpy().copy()
            # Baseline observes clean pre-decision memory and bank only.
            value=agent.critic(agent.meta_state(z))
            managed=agent.manage(z,train)
            feedback=torch.cat((managed["context"][:,:cfg.latent_dim],
                managed["context"][:,cfg.latent_dim:2*cfg.latent_dim]),-1)
            previous=agent.memory.detach().clone().requires_grad_(train)
            out=agent.strategizer(z if s.encoder_eprop else z.detach(),feedback.detach(),
                previous_strategy=previous)
            clean=out["strategy"]
            strategy,score,std,entropy=agent.sample_strategy(clean,train)
            agent.memory=clean.detach().clone()
            actor=agent.actor(z,strategy,managed["context"],deterministic=not train,
                temperature=agent.actor_temperature)
            if train:
                agent.actor_credit.accumulate(actor["logp"])
                agent.strategy_credit.accumulate(out["proposal"],out["gate"],previous,
                    clean,score,cfg.strategy_retention,True)
                if s.encoder_eprop:
                    agent.encoder_credit.accumulate(out["proposal"],out["gate"],previous,
                        clean,score,True,cfg.strategy_retention)
                agent.std_credit.accumulate(score)
                std_entropy_grad=torch.autograd.grad(entropy.sum(),[agent.strategy_logstd],
                    retain_graph=True)
            pose=(int(env.x[0]),int(env.y[0]));direction=int(env.direction[0])
            visible=bool(env.age[0]<cfg.cue_steps)
            created=agent.create_prediction(z,strategy,actor["action"],train)
            bank_creations+=int(created)
            horizon_created[agent.queue[-1]["horizon"]]+=1
            discarded+=managed["discard"];expired+=managed["expired"]
            spikes={}
            for name,module in [("encoder",agent.encoder),("predictor",agent.predictor),
                    ("critic",agent.critic),
                    ("strategizer",agent.strategizer),("actor",agent.actor)]:
                if module is None:
                    continue
                spikes[name]=module.core.last_output[:,cfg.hidden_dim:].detach().numpy().copy()
            _,rewards,dones,successes,wrong,_,_=env.step(actor["action"].detach().numpy())
            reward=float(rewards[0]);done=bool(dones[0])
            cue_bonus=agent.success_bonus(cue, done and bool(successes[0]), train)
            learning_reward=reward+cue_bonus
            agent.step_index+=1;agent.ages[agent.active]+=1
            terminal_observation=torch.from_numpy(env.transition_observation.copy())
            with torch.no_grad():
                next_z=agent.encoder.encode(terminal_observation)
                next_value=(torch.zeros(1) if done else agent.critic(agent.meta_state(next_z)))
                external_target=learning_reward+cfg.gamma*next_value
                external_td=(external_target-value.detach()).clamp(-cfg.td_clip,cfg.td_clip)
                actor_signal,next_phi,shaping,correction=agent.actor_reward(learning_reward,cfg.gamma,
                    s.prediction_shaping_beta,z.detach(),next_z,managed["target"],
                    bool(managed["use"].any()),done,carried_phi)
            # Supervision is unconditional and observes the actual arrival.
            step_validations=agent.supervise(terminal_observation,train)
            validations.extend(step_validations)
            prediction_mse=(float(np.mean([r["mse"] for r in step_validations]))
                if step_validations else None)
            strategy_update_norm=encoder_update_norm=0.
            if train:
                agent.actor_credit.apply(actor_signal.reshape(1))
                _,strategy_update_norm=agent.strategy_credit.apply(external_td)
                if s.encoder_eprop:
                    _,encoder_update_norm=agent.encoder_credit.apply(external_td)
                if s.stochastic_strategy:
                    agent.reward_step(agent.std_credit,external_td,std_entropy_grad,s.strategy_entropy_weight)
                agent.critic_optimizer.zero_grad(set_to_none=True)
                (value-external_target).square().mean().backward()
                agent.critic_optimizer.step()
            frames.append(dict(manager_removed=True,observation=observation[0],bank_before=bank_before,
                ages_before=ages_before,context=managed["context"].numpy()[0].copy(),
                feedback=feedback.numpy()[0].copy(),
                z=z.detach().numpy()[0].copy(),
                predictions=agent.predictions.numpy().copy(),active=agent.active.numpy().copy(),
                managed_active=managed["incoming"].numpy().copy(),
                management=managed["choice"].numpy().copy(),probs=managed["probs"].numpy().copy(),
                used=managed["use"].numpy().copy(),
                strategy=strategy.detach().numpy()[0].copy(),
                strategy_latent=strategy.detach().numpy()[0].copy(),
                clean_strategy_latent=clean.detach().numpy()[0].copy(),
                actor=actor["logits"].softmax(-1).detach().numpy()[0].copy(),
                action=int(actor["action"][0]),pose=pose,direction=direction,cue=cue,visible=visible,
                value=float(value.detach()),reward=reward,cue_bonus=cue_bonus,
                learning_reward=learning_reward,external_td=float(external_td),
                actor_signal=float(actor_signal),shaping=float(shaping),
                switch_correction=float(correction),phi=float(carried_phi),phi_next=float(next_phi),
                alignment=float((next_z-managed["target"]).square().mean()) if bool(managed["use"].any()) else None,
                intent=agent.intents.numpy().copy(),ages=agent.ages.numpy().copy(),
                horizons=agent.slot_horizons.numpy().copy(),ids=agent.slot_ids.numpy().copy(),
                hidden=agent.hidden.numpy().copy(),spike_rates=spikes,
                strategy_std_mean=float(std.mean().detach()),strategy_std_min=float(std.min().detach()),
                strategy_entropy=float(entropy.mean().detach()),manager_entropy=managed["entropy"],
                critic_loss=float((value.detach()-external_target).square().mean()),
                prediction_validation_loss=prediction_mse,
                strategy_lr_scale=agent.strategy_lr_scale,
                strategy_update_norm=strategy_update_norm,encoder_update_norm=encoder_update_norm,
                queue_length=len(agent.queue)))
            carried_phi=next_phi.detach()
            if done:break
        else:raise RuntimeError("Maze did not terminate")
        censored={h:sum(r["horizon"]==h for r in agent.queue) for h in s.prediction_horizons}
        by_horizon={h:dict(count=sum(r["horizon"]==h for r in validations),
            mse_sum=sum(r["mse"] for r in validations if r["horizon"]==h),
            created=horizon_created[h],censored=censored[h]) for h in s.prediction_horizons}
        # Sufficient statistics enable conditional diagnostics without saving all training frames.
        diagnostics={}
        for phase in ("all","visible","hidden"):
            selected=[f for f in frames if phase=="all" or f["visible"]==(phase=="visible")]
            diagnostics[phase]={}
            for name in ("z","strategy_latent"):
                diagnostics[phase][name]=dict(count=len(selected),
                    total=np.sum([f[name] for f in selected],axis=0).tolist() if selected else [])
        age_stats={}
        for f in frames:
            for i in np.flatnonzero(f["active"]):
                age=int(f["ages"][i])
                entry=age_stats.setdefault(age,dict(count=0,
                    hidden_sum=np.zeros(2*cfg.hidden_dim),prediction_sum=np.zeros(cfg.latent_dim)))
                entry["count"]+=1;entry["hidden_sum"]+=f["hidden"][i];entry["prediction_sum"]+=f["predictions"][i]
        for entry in age_stats.values():
            entry["hidden_sum"]=entry["hidden_sum"].tolist();entry["prediction_sum"]=entry["prediction_sum"].tolist()
        junction=[f for f in frames if f["pose"]==(env.center,1) and f["direction"]==0]
        report=dict(strategy_exploration_scale=agent.strategy_exploration_scale,
            actor_temperature=agent.actor_temperature,success=float(successes[0]),cue=cue,episode_return=sum(f["reward"] for f in frames),
            cue_success_bonus=sum(f["cue_bonus"] for f in frames),
            learning_return=sum(f["learning_reward"] for f in frames),
            wrong_goal=float(wrong[0]),timeout=float(not successes[0] and not wrong[0]),
            steps=len(frames),curriculum_stage=episode_stage,
            hallway_length=hallway_length,episode_time_limit=episode_time_limit,
            curriculum_blend_fraction=episode_blend,
            sampled_length_distribution=sampled_distribution,
            curriculum_length_distributions=env.curriculum_length_distributions,
            length_diagnostic_min_samples=cfg.curriculum_min_episodes_per_length,
            next_curriculum_stage=env.curriculum_stage+1,
            prediction_creation_count=len(frames),behavioral_creation_count=bank_creations,
            prediction_validation_count=len(validations),
            prediction_validation_loss=float(np.mean([r["mse"] for r in validations])) if validations else None,
            prediction_censored_count=len(agent.queue),prediction_discard_count=discarded,
            prediction_expiration_count=expired,mean_active_traces=float(np.mean([f["active"].sum() for f in frames])),
            mean_prediction_age=float(np.mean([f["ages"][f["active"]].mean() if f["active"].any() else 0 for f in frames])),
            maximum_prediction_age=max(int(f["ages"][f["active"]].max()) if f["active"].any() else 0 for f in frames),
            strategy_std_mean=float(np.mean([f["strategy_std_mean"] for f in frames])),
            strategy_std_min=min(f["strategy_std_min"] for f in frames),
            strategy_entropy=float(np.mean([f["strategy_entropy"] for f in frames])),
            manager_entropy=float(np.mean([f["manager_entropy"] for f in frames])),
            critic_loss=float(np.mean([f["critic_loss"] for f in frames])),
            strategy_lr_scale=float(np.mean([f["strategy_lr_scale"] for f in frames])),
            strategy_lr_scale_min=min(f["strategy_lr_scale"] for f in frames),
            strategy_lr_training_steps=len(frames) if train else 0,
            strategy_lr_slowed_steps=sum(f["strategy_lr_scale"]<.99 for f in frames) if train else 0,
            strategy_lr_slowed_fraction=float(np.mean([f["strategy_lr_scale"]<.99 for f in frames])) if train else 0.,
            strategy_update_norm=float(np.mean([f["strategy_update_norm"] for f in frames])),
            encoder_update_norm=float(np.mean([f["encoder_update_norm"] for f in frames])),
            horizon_accuracy=by_horizon,diagnostics=diagnostics,predictor_age_stats=age_stats,
            junction_count=len(junction),
            junction_probability_sum=np.sum([f["actor"] for f in junction],axis=0).tolist() if junction else [0.,0.,0.],
            management_counts=[sum(int(((f["management"]==g)&f["managed_active"]).sum()) for f in frames) for g in range(3)])
        if train:
            agent.record_cue_success(cue, bool(successes[0]))
            agent.record_exploration_success(bool(successes[0]))
            report.update(agent.adapt_strategy_lr(by_horizon,critic_loss=report["critic_loss"])
                if s.slowdown_trigger_source=="critic" else agent.adapt_strategy_lr(by_horizon))
        else:
            report.update(slowdown_trigger_source=s.slowdown_trigger_source,
                prediction_change_score=None,prediction_change_triggered=False,
                prediction_change_state="evaluation",prediction_change_threshold=s.prediction_change_threshold,
                prediction_change_details=[],strategy_lr_next_scale=agent.strategy_lr_scale)
        agent.reset_episode()
        return report,frames

    def summarize(history):
        result=dict(episodes=len(history),success=float(np.mean([r["success"] for r in history])),
            wrong_goal_rate=float(np.mean([r["wrong_goal"] for r in history])),
            timeout_rate=float(np.mean([r["timeout"] for r in history])))
        groups=[[r for r in history if r["cue"]==c] for c in (0,1)]
        for c,label in enumerate(("left","right")):
            result["episodes_"+label]=len(groups[c])
            result["success_"+label]=float(np.mean([r["success"] for r in groups[c]])) if groups[c] else None
            n=sum(r["junction_count"] for r in groups[c])
            probs=np.sum([r["junction_probability_sum"] for r in groups[c]],axis=0)/n if n else [None]*3
            result["junction_p_left_given_"+label]=float(probs[0]) if n else None
            result["junction_p_right_given_"+label]=float(probs[1]) if n else None
            result["junction_samples_"+label]=n
        result["worst_cue_success"]=min(result["success_left"],result["success_right"]) if all(groups) else None
        for phase in ("all","visible","hidden"):
            for name in ("z","strategy_latent"):
                means=[]
                for group in groups:
                    entries=[r["diagnostics"][phase][name] for r in group]
                    n=sum(e["count"] for e in entries)
                    means.append(np.sum([e["total"] for e in entries if e["count"]],axis=0)/n if n else None)
                result[phase+"_"+name+"_separation"]=float(np.linalg.norm(means[0]-means[1])) if all(m is not None for m in means) else None
        ages=sorted({age for r in history for age in r["predictor_age_stats"]})
        age_rows=[]
        for age in ages:
            row={"age":age}
            for name in ("hidden_sum","prediction_sum"):
                means=[]
                for group in groups:
                    entries=[r["predictor_age_stats"][age] for r in group if age in r["predictor_age_stats"]]
                    n=sum(e["count"] for e in entries)
                    means.append(np.sum([e[name] for e in entries],axis=0)/n if n else None)
                row[name.replace("_sum","")+"_separation"]=float(np.linalg.norm(means[0]-means[1])) if all(m is not None for m in means) else None
            age_rows.append(row)
        horizon_rows=[]
        for h in sorted({h for r in history for h in r["horizon_accuracy"]}):
            entries=[r["horizon_accuracy"][h] for r in history if h in r["horizon_accuracy"]]
            n=sum(e["count"] for e in entries)
            horizon_rows.append(dict(horizon=h,validations=n,
                mse=sum(e["mse_sum"] for e in entries)/n if n else None,
                created=sum(e["created"] for e in entries),censored=sum(e["censored"] for e in entries)))
        result["accuracy_by_horizon"]=horizon_rows
        result["separation_by_age"]=age_rows
        for name in ("strategy_std_mean","strategy_std_min","strategy_entropy","manager_entropy",
                     "mean_active_traces","mean_prediction_age","maximum_prediction_age"):
            values=[r[name] for r in history]
            result[name]=(min(values) if name=="strategy_std_min" else max(values)
                if name=="maximum_prediction_age" else float(np.mean(values)))
        counts=np.sum([r["management_counts"] for r in history],axis=0)
        result["management_distribution"]=(counts/counts.sum()).tolist() if counts.sum() else [0.,0.,0.]
        transitions=sum(r["steps"] for r in history)
        result["behavioral_creation_rate"]=sum(r["behavioral_creation_count"] for r in history)/transitions
        result["discard_rate"]=sum(r["prediction_discard_count"] for r in history)/transitions
        for name in ("strategy_lr_scale","strategy_lr_scale_min",
                     "strategy_update_norm","encoder_update_norm"):
            result[name]=min(r[name] for r in history) if name.endswith("_min") else float(np.mean([r[name] for r in history]))
        result["slowdown_trigger_source"]=history[-1]["slowdown_trigger_source"]
        result["prediction_change_events"]=sum(r["prediction_change_triggered"] for r in history)
        result["prediction_change_state"]=history[-1]["prediction_change_state"]
        result["strategy_lr_slowed_steps"]=sum(r["strategy_lr_slowed_steps"] for r in history)
        training_steps=sum(r["strategy_lr_training_steps"] for r in history)
        result["strategy_lr_slowed_fraction"]=result["strategy_lr_slowed_steps"]/training_steps if training_steps else 0.
        result["curriculum_blend_fraction"]=history[-1]["curriculum_blend_fraction"]
        result["sampling_distribution"]=[dict(hallway_length=h,probability=p)
            for h,p in history[-1]["sampled_length_distribution"].items()]
        result.update(BatchedTMaze.summarize_hallways(history,
            history[0]["length_diagnostic_min_samples"],history[0]["curriculum_length_distributions"]))
        return result
    return episode,summarize

@app.cell
def _(np,plt):
    def critic_spike_density(history,threshold=.15,window=20):
        """Trailing episode exceedance frequency; no stage-dependent reset.

        Each above-threshold episode counts once, including consecutive episodes.
        Early windows divide by available episodes, not the requested window size.
        """
        if threshold<0 or window<1:
            raise ValueError("Spike threshold must be nonnegative and window positive")
        losses=np.asarray([r["critic_loss"] for r in history],dtype=float)
        spikes=losses>threshold
        counts=[];fractions=[];excess=[];magnitudes=[];combined=[]
        for i in range(len(losses)):
            start=max(0,i-window+1)
            counts.append(int(spikes[start:i+1].sum()))
            fractions.append(counts[-1]/(i-start+1))
            excess.append(float(np.maximum(losses[start:i+1]-threshold,0.).mean()))
            magnitudes.append(excess[-1]/fractions[-1] if counts[-1] else 0.)
            combined.append(fractions[-1]*magnitudes[-1])
        return dict(spikes=spikes,counts=counts,fractions=fractions,mean_excess=excess,
            spike_magnitude=magnitudes,density_times_magnitude=combined)

    def training_plot(history,window=100):
        """All rolling windows span curriculum changes; boundaries are annotations only."""
        x=np.arange(1,len(history)+1)
        series={key:[] for key in ("success","left","right","timeout","mse",
            "strategy_std_mean","critic_loss")}
        horizons=sorted({h for r in history for h in r["horizon_accuracy"]})
        horizon_series={h:[] for h in horizons}
        boundaries=[]
        for i,row in enumerate(history):
            if i and row["curriculum_stage"]!=history[i-1]["curriculum_stage"]:
                boundaries.append((i+1,row["curriculum_stage"]))
            block=history[max(0,i-window+1):i+1]
            for key in ("success","timeout","strategy_std_mean"):
                series[key].append(float(np.mean([r[key] for r in block])))
            for cue,label in ((0,"left"),(1,"right")):
                group=[r["success"] for r in block if r["cue"]==cue]
                series[label].append(float(np.mean(group)) if group else np.nan)
            loss_block=history[max(0,i-window+1):i+1]
            series["critic_loss"].append(float(np.mean([r["critic_loss"] for r in loss_block])))
            count=sum(r["prediction_validation_count"] for r in loss_block)
            series["mse"].append(sum((r["prediction_validation_loss"] or 0)*
                r["prediction_validation_count"] for r in loss_block)/count if count else np.nan)
            for h in horizons:
                entries=[r["horizon_accuracy"][h] for r in loss_block if h in r["horizon_accuracy"]]
                n=sum(e["count"] for e in entries)
                horizon_series[h].append(sum(e["mse_sum"] for e in entries)/n if n else np.nan)
        fig,axes=plt.subplots(3,2,figsize=(14,10),sharex=True,layout="constrained")
        panels=[("Episode outcomes (continuous rolling window)","Rate",[("success","Success"),("left","Left cue"),
                    ("right","Right cue"),("timeout","Timeout")]),
                ("Prediction loss (continuous rolling window)","Validated latent MSE",[("mse","All horizons")]),
                ("Strategy exploration spread","Mean Gaussian standard deviation",[("strategy_std_mean","Strategy std")]),
                ("Critic loss · raw episode means + continuous rolling average","Mean squared TD error",[("critic_loss","Rolling episode mean")])]
        for ax,(title,ylabel,lines) in zip(axes.flat,panels):
            for key,label in lines:ax.plot(x,series[key],label=label)
            ax.set(title=title,ylabel=ylabel);ax.legend(fontsize=8)
        axes[1,1].plot(x,[r["critic_loss"] for r in history],label="Raw episode mean (no smoothing)",
            color="tab:orange",marker=".",markersize=3,linewidth=.7,alpha=.7)
        axes[1,1].legend(fontsize=8)
        for h,values in horizon_series.items():
            axes[0,1].plot(x,values,label=f"Horizon {h}",alpha=.6,linewidth=1)
        axes[0,1].legend(fontsize=8)
        axes[0,0].set_ylim(-.03,1.03)
        axes[2,0].step(x,[r["curriculum_stage"] for r in history],where="post",label="Episode stage")
        axes[2,0].plot(x,[max(1.,r["curriculum_stage"]-1+r.get("curriculum_blend_fraction",1.))
            for r in history],label="Distribution blend",color="darkorange")
        axes[2,0].legend(fontsize=8)
        axes[2,0].set(title="Curriculum target and gradual blend",ylabel="Stage",
            yticks=sorted({r["curriculum_stage"] for r in history}))
        for key,label in (("strategy_update_norm","Strategizer"),("encoder_update_norm","Strategy → encoder")):
            axes[2,1].plot(x,[r[key] for r in history],label=label)
        axes[2,1].set(title="Actual parameter updates (episode mean)",ylabel="L2 step norm")
        axes[2,1].legend(fontsize=8)
        for ax in axes.flat:
            for boundary,stage in boundaries:ax.axvline(boundary,color="gray",ls=":",alpha=.6)
            ax.grid(alpha=.2);ax.set_xlabel("Completed training episode")
        fig.suptitle(f"Training · {len(history)} episodes · rolling {window} episodes; all windows continuous")
        plt.close(fig)
        return fig
    return training_plot,critic_spike_density


@app.cell
def _(mo):
    mo.md("""# Ghost · external-reward predictive control

    Optional adaptive exploration uses prior raw training success over a continuous
    rolling window. After warmup, success below the configured target increases
    strategy standard deviation and/or actor temperature linearly, up to their
    respective maxima at zero success. At or above the target both return to 1.
    Controls are fixed within each episode. Strategy adaptation requires stochastic
    strategy; evaluation stays deterministic. Per-episode multipliers are exported.

    Optional weaker-cue reward: successful training episodes receive
    `bonus_scale × max(0, other_cue_success_rate − current_cue_success_rate)`.
    Rates use only prior training episodes in a continuous rolling window, with
    a minimum sample count for each cue. The bonus enters actor reward and critic /
    strategy TD targets. Evaluation and plotted success rates / raw returns stay
    unbonused. The bonus defaults off.

    Strategy uses **external TD only**. The bank uses a fixed USE rule; there is no learned manager. The critic observes the
    pre-decision state; its value is not an actor input or a selection weight.
    The actor receives strategy, prediction delta/residual, age, active fraction,
    and historical error, and learns from external reward plus predictive shaping.
    Prediction confidence is an input only.

    Forecasts are created every environment step. Their horizons are sampled
    independently and supplied through Ghost's existing horizon-conditioning
    input. A detached forecast and its creation-time predictor e-prop Jacobians
    remain queued even if the behavioral slot is discarded or the bank is full.
    Scheduled targets use the online encoder. Forecasts never change after creation.
    Targets beyond episode termination are censored and counted explicitly.

    The clean gated strategy memory is preserved. Training samples a squashed
    Gaussian intent with positive standard-deviation floor. Gaussian score
    derivatives enter the original recurrent strategy e-prop, and the approved
    encoder-projection e-prop path. Entropy regularization is reported for the
    Gaussian before squashing. Deterministic evaluation uses the mean.
    Positive variance provides exploration, but does not guarantee useful actions.

    **Optional loss change detector:** select prediction loss (default) or critic
    loss. Critic mode uses the episode mean of unclipped squared TD error, measured
    before each critic update, and ignores prediction losses. Both modes use the
    same trend/variability detector, hold, recovery, and rearming rules.
    In prediction mode, each completed training episode
    is compared with a linear trend fitted to preceding episode losses separately
    for each forecast horizon. The median normalized upward deviation must exceed
    the threshold; a relative increase floor prevents triggering on tiny changes.
    Horizons with no validated forecasts are skipped. No curriculum state enters
    the detector, and prediction-loss plot windows never reset at stage changes.

    A detected shift lowers strategy-related learning rates starting with the
    **next episode**. Protection lasts at least the hold duration and until the
    score stays within ±1 for the configured stable-episode count. Learning rates
    then recover gradually; a new spike during recovery extends the same event.
    The detector rearms only after recovery and stabilization. Critic, predictor,
    and actor rates retain their configured values. History continues
    throughout protection, so the detector can settle around a new loss level.
    Missing evidence pauses protection/recovery; evaluation never updates it.
    This is an experimental change heuristic, not calibrated uncertainty.

    **Shaping correction:** a prediction is fixed across each physical transition.
    When the chosen prediction changes, a boundary correction carries the previous
    potential forward. Initial and terminal potentials are zero, so discounted
    shaping sums to zero over a complete episode, including target switches.

    The original spiking modules/e-prop equations are retained. The actor's input
    conditioning is extended. An EMA predictor is deferred; immutable behavioral
    forecasts already remain fixed until removal. This is a new experiment, not
    an established successful configuration. Fresh training only; older bank
    checkpoints use different learning rules and shapes.

    The curriculum samples overlapping hallway-length distributions independently
    of the cue at every reset. After mastery, probabilities blend linearly into
    the next target over 64 completed episodes by default (configurable; 0 restores
    immediate switching). Mastery evidence resumes only for episodes sampled from
    the settled target. The orange curriculum curve shows blend progress. Length counts cells including the junction (length
    1 begins at the junction); timeouts depend on the sampled length and stay fixed
    through the episode. Later stages favor longer delays while retaining shorter
    rehearsal. Evaluation samples the final distribution. Position observations
    are preserved; no length or stage input is added. These changes discourage
    fixed-stage timing shortcuts but do not establish that the model uses memory.

    One-world training uses shuffled L/R pairs across resets. The original
    worst-cue curriculum criterion is retained. Separation diagnostics are
    observational cue-group differences; they are not causal memory tests.
    """)
    setup=mo.ui.dictionary({
        "seed":mo.ui.number(value=12,start=0,label="Seed"),
        "slots":mo.ui.dropdown([4,8],value=4,label="Behavioral slots"),
        "episodes":mo.ui.number(value=4,start=1,stop=10000,label="Training episodes from scratch"),
        "plot_every":mo.ui.number(value=10,start=1,stop=1000,label="Update plots every N episodes"),
        "plot_window":mo.ui.number(value=100,start=1,stop=1000,label="Plot rolling window (episodes across curriculum changes)"),
        "report_every":mo.ui.number(value=100,start=1,stop=1000,label="Report every N episodes"),
        "horizons":mo.ui.dropdown(["1","1,2,4","1,2,4,8,16"],value="1,2,4,8,16",label="Exogenous horizons"),
        "max_prediction_age":mo.ui.number(value=16,start=1,stop=64,label="Maximum behavioral age"),
        "strategy_min_std":mo.ui.number(value=.05,start=.001,stop=1.,step=.01,label="Minimum strategy std"),
        "strategy_entropy_weight":mo.ui.number(value=.001,start=0.,stop=.1,step=.001,label="Strategy entropy weight"),
        "curriculum_blend_episodes":mo.ui.number(value=64,start=0,step=1,label="Episodes to blend into next curriculum (0 = immediate)"),
        "adaptive_strategy_exploration":mo.ui.checkbox(value=False,label="Adaptive strategy exploration (requires stochastic strategy)"),
        "adaptive_actor_exploration":mo.ui.checkbox(value=False,label="Adaptive actor exploration"),
        "exploration_window":mo.ui.number(value=100,start=1,step=1,label="Exploration success window (episodes)"),
        "exploration_warmup":mo.ui.number(value=20,start=1,step=1,label="Exploration warmup (≤ window)"),
        "exploration_target_success":mo.ui.number(value=.8,start=.01,stop=1.,step=.05,label="Success target for baseline exploration"),
        "strategy_exploration_max":mo.ui.number(value=2.,start=1.,step=.1,label="Maximum strategy std multiplier"),
        "actor_temperature_max":mo.ui.number(value=2.,start=1.,step=.1,label="Maximum actor temperature"),
        "cue_success_bonus":mo.ui.checkbox(value=False,label="Reward successful episodes more for the weaker cue"),
        "cue_bonus_scale":mo.ui.number(value=.1,start=0.,step=.05,label="Maximum weaker-cue success bonus"),
        "cue_bonus_window":mo.ui.number(value=100,start=2,step=1,label="Cue bonus history (completed training episodes)"),
        "cue_bonus_min_samples":mo.ui.number(value=10,start=1,step=1,label="Minimum episodes per cue before bonus (window ≥ 2× minimum)"),
        "balanced_cue_batches":mo.ui.checkbox(value=True,label="Balanced shuffled cue pairs"),
        "strategy_conditioning":mo.ui.checkbox(value=True,label="Condition predictor on strategy"),
        "stochastic_strategy":mo.ui.checkbox(value=True,label="Stochastic strategy (off = diagnostic ablation)"),
        "slowdown_trigger_source":mo.ui.dropdown({"Prediction loss":"prediction","Critic loss":"critic"},
            value="Prediction loss",label="Slowdown trigger source"),
        "strategy_lr_min_scale":mo.ui.number(value=.1,start=.01,stop=1.,step=.01,label="Minimum strategy LR multiplier"),
        "prediction_change_window":mo.ui.number(value=20,start=3,step=1,label="Loss trend window (episodes; per horizon for prediction)"),
        "prediction_change_warmup":mo.ui.number(value=8,start=3,step=1,label="Minimum prior loss episodes (≤ window)"),
        "prediction_change_threshold":mo.ui.number(value=3.,start=1.1,step=.1,label="Upward deviation threshold"),
        "prediction_change_min_increase":mo.ui.number(value=.15,start=.01,step=.01,label="Minimum relative loss increase (0.15 = 15%)"),
        "strategy_lr_hold_episodes":mo.ui.number(value=5,start=1,step=1,label="Minimum protected episodes"),
        "strategy_lr_recovery_episodes":mo.ui.number(value=10,start=1,step=1,label="LR recovery episodes"),
        "prediction_stable_episodes":mo.ui.number(value=3,start=1,step=1,label="Stable episodes before recovery / rearming"),
        "encoder_eprop":mo.ui.checkbox(value=True,label="External-credit encoder projection e-prop"),
    }).form(submit_button_label="Train and record evaluation")
    setup
    return (setup,)


@app.cell
def _(mo):
    get_testing_snapshot, set_testing_snapshot = mo.state(None)
    return get_testing_snapshot, set_testing_snapshot


@app.cell
def _(setup,mo,Settings,PredictiveAgent,BalancedMaze,torch,episode,summarize,training_plot,asdict):
    mo.stop(setup.value is None,mo.md("Submit to start. Four episodes is a smoke test."))
    _values={k:v for k,v in setup.value.items() if k not in ("episodes","report_every","horizons","plot_every","plot_window")}
    _values["prediction_horizons"]=tuple(int(h) for h in setup.value["horizons"].split(","))
    config=Settings(**_values)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model=PredictiveAgent(config)
        environment=BalancedMaze(model.cfg,config.seed,balanced=config.balanced_cue_batches)
        history=[];interval_reports=[]
        with mo.status.progress_bar(total=int(setup.value["episodes"]),title="Training episodes",
                show_eta=True,show_rate=True,completion_title="Training complete") as _bar:
            for _ep in range(int(setup.value["episodes"])):
                _report,_=episode(model,environment,train=True)
                history.append(_report)
                if (_ep+1)%int(setup.value["report_every"])==0 or _ep+1==int(setup.value["episodes"]):
                    _block=history[-min((_ep%int(setup.value["report_every"]))+1,len(history)):]
                    _summary=summarize(_block)
                    _summary["through_episode"]=_ep+1
                    interval_reports.append(_summary)
                    _bar.update(increment=1,subtitle=f"Episode {_ep+1}: worst-cue success {_summary['worst_cue_success']}")
                else:_bar.update()
                if _ep==0 or (_ep+1)%int(setup.value["plot_every"])==0 or _ep+1==int(setup.value["episodes"]):
                    mo.output.replace(mo.vstack([
                        mo.md(f"Training: **{_ep+1}/{int(setup.value['episodes'])} episodes**. "
                            "Dotted lines mark curriculum changes; all rolling windows continue across them. Orange critic points are unsmoothed episode means. "
                            "Prediction MSE uses only forecasts whose targets were reached; censored forecasts are excluded."),
                        training_plot(history,int(setup.value["plot_window"])),
                    ]))
        with torch.no_grad():
            evaluation,frames=episode(model,BalancedMaze(model.cfg,config.seed+10000,
                curriculum=False,balanced=config.balanced_cue_batches))
    summary=summarize(history)
    # Prepare an isolated snapshot after all episodes and evaluation.
    # The downstream publisher waits for final reporting to succeed too.
    # Plot refreshes above never publish weights to Testing.
    completed_training_snapshot = dict(architecture="ghost_external_predictive_control_v6",
        config=asdict(config),ghost_config=asdict(model.cfg),
        weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        reliability=dict(model.reliability))
    mo.vstack([training_plot(history,int(setup.value["plot_window"])),
        mo.md(f"Training worst-cue success: {summary['worst_cue_success']}. "
        f"Single deterministic final-distribution evaluation success: {bool(evaluation['success'])}."),
    ])
    return model,config,history,interval_reports,summary,evaluation,frames,completed_training_snapshot


@app.cell
def _(completed_training_snapshot, set_testing_snapshot):
    # This cell runs only after the training cell completes successfully.
    # Keep the previous completed snapshot while another run is in progress.
    set_testing_snapshot(completed_training_snapshot)
    return


@app.cell
def _(model,config,history,interval_reports,summary,evaluation,mo,torch,asdict):
    import io
    _buffer=io.BytesIO()
    torch.save(dict(architecture="ghost_external_predictive_control_v6",
        config=asdict(config),ghost_config=asdict(model.cfg),weights=model.state_dict(),
        history=history,interval_reports=interval_reports,summary=summary,
        evaluation=evaluation,reliability=dict(model.reliability)),_buffer)
    mo.download(_buffer.getvalue(),filename="ghost_external_predictive_control_v6.pt",
        label="Download weights and diagnostics (not training-resume state)")
    return


@app.cell
def _(mo):
    mo.md("""## Testing · live model dataflow

    Testing receives new weights only after the entire training run finishes.
    Plot updates do not reload the model; the previous completed snapshot stays selected.
    Train above or load a **v6 predictive-control checkpoint** below. Changing the
    cue replays a deterministic episode from fresh recurrent memory with the same
    evaluation seed and forecast-horizon draws. Scrub the timestep to inspect its
    actual forward-pass tensors. This evaluates the policy without weight updates.
    The cue is only visible during the configured initial cue window.
    """)
    testing_source = mo.ui.dropdown(["Latest training", "Checkpoint"],
        value="Latest training", label="Model source")
    testing_file = mo.ui.file(filetypes=[".pt", ".pth"], multiple=False,
        label="Load notebook checkpoint")
    testing_cue = mo.ui.dropdown({"Left": 0, "Right": 1}, value="Left", label="Cue")
    testing_seed = mo.ui.number(start=0, value=10012, step=1, label="Evaluation seed")
    mo.vstack([testing_source, testing_file, testing_seed])
    return testing_source, testing_file, testing_cue, testing_seed


@app.cell
def _(torch, Settings, PredictiveAgent):
    def load_testing_model(payload):
        if not isinstance(payload, dict) or payload.get("architecture") != "ghost_external_predictive_control_v6":
            raise ValueError("Expected a ghost_external_predictive_control_v6 checkpoint exported by this notebook.")
        if payload["config"].get("management", False):
            raise ValueError("This checkpoint used learned management; use the original notebook to evaluate it.")
        settings = Settings(**{k:v for k,v in payload["config"].items()
            if k not in ("management", "remove_manager", "manager_entropy_weight",
                          "use_target_encoder", "encoder_target_tau")})
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(settings.seed)
            loaded = PredictiveAgent(settings)
        # Construction currently derives architecture from Settings. Do not silently
        # load a checkpoint whose environment or forward-pass configuration differs.
        from dataclasses import asdict
        if {k:v for k,v in payload.get("ghost_config", {}).items()
                if k != "encoder_target_tau"} != asdict(loaded.cfg):
            raise ValueError("Checkpoint Ghost configuration differs from this notebook; use the matching notebook version.")
        loaded.load_state_dict({k:v for k,v in payload["weights"].items()
            if not k.startswith(("manager.", "target_encoder."))}, strict=True)
        loaded.reliability = {h: float(payload.get("reliability", {}).get(h, 0.))
                              for h in settings.prediction_horizons}
        loaded.eval()
        return loaded
    return (load_testing_model,)


@app.cell
def _(testing_source, testing_file, get_testing_snapshot, load_testing_model, torch, mo):
    import io as testing_io
    _payload = None
    _error = None
    try:
        if testing_source.value == "Checkpoint":
            if testing_file.value:
                _payload = torch.load(testing_io.BytesIO(testing_file.value[0].contents),
                                      map_location="cpu", weights_only=True)
        else:
            _payload = get_testing_snapshot()
        testing_model = load_testing_model(_payload) if _payload is not None else None
    except (ValueError, TypeError, KeyError, RuntimeError, EOFError, OSError, testing_io.UnsupportedOperation) as _exc:
        _error = str(_exc)
        testing_model = None
    except Exception as _exc:
        # Includes safe-loader rejection of incompatible checkpoint objects.
        _error = f"{type(_exc).__name__}: {_exc}"
        testing_model = None
    mo.stop(testing_model is None, mo.callout(
        _error or "Train above, or choose Checkpoint and upload weights to begin testing.",
        kind="warn" if _error else "info"))
    mo.md(f"Loaded v6 model · {testing_model.settings.slots} slots · "
          f"{testing_model.cfg.latent_dim} encoder latents. "
          + ("Legacy checkpoint: historical prediction errors default to zero."
             if "reliability" not in _payload else "Historical prediction errors restored."))
    return (testing_model,)


@app.cell
def _(testing_model, testing_cue, testing_seed, BalancedMaze, episode, torch, np):
    import copy as testing_copy
    # Clone the loaded model so UI reruns cannot carry over memory, RNG, or bank state.
    _agent = testing_copy.deepcopy(testing_model)
    _seed = int(testing_seed.value)
    _agent.horizon_rng = np.random.default_rng(_seed + 700001)
    _env = BalancedMaze(_agent.cfg, _seed, curriculum=False, balanced=False)
    _env.cue[0] = int(testing_cue.value)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(_seed)
        testing_report, testing_frames = episode(_agent, _env, train=False)
    testing_geometry = (_agent.cfg.maze_width, _agent.cfg.maze_height)
    return testing_report, testing_frames, testing_geometry


@app.cell
def _(mo, testing_frames):
    timestep = mo.ui.slider(0, max(1, len(testing_frames)-1), value=0, step=1,
        show_value=True, label="Evaluation timestep")
    timestep
    return (timestep,)


@app.cell
def _(testing_frames, timestep):
    testing_frame = testing_frames[min(timestep.value, len(testing_frames)-1)]
    return (testing_frame,)


@app.cell
def _(mo, plt, np):
    from wigglystuff import WidgetDAG

    def tensor_panel(title, values, mask=None, probabilities=False, labels=None):
        data = np.asarray(values)
        fig, ax = plt.subplots(figsize=(3.2, 2.0), layout="constrained")
        if probabilities:
            ax.bar(labels or list(range(data.size)), data.ravel(), color="#4f83cc")
            ax.set_ylim(0, 1)
            for i, value in enumerate(data.ravel()):
                ax.text(i, float(value)+.02, f"{value:.3f}", ha="center", fontsize=8)
        else:
            data = np.atleast_2d(data)
            if mask is not None:
                data = np.ma.array(data, mask=np.broadcast_to(~np.asarray(mask)[:, None], data.shape))
            limit = max(1., float(np.max(np.abs(np.asarray(values)))))
            picture = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
            fig.colorbar(picture, ax=ax, shrink=.65)
            ax.set_xlabel("Dimension")
            if data.shape[0] == 1:
                ax.set_yticks([])
            else:
                ax.set_ylabel("Slot")
                ax.set_yticks(range(data.shape[0]))
        ax.set_title(title, fontsize=10)
        plt.close(fig)
        return mo.as_html(fig)
    return WidgetDAG, tensor_panel


@app.cell
def _(testing_frame, testing_geometry, testing_cue, testing_report, tensor_panel, WidgetDAG, mo, plt, np):
    _f = testing_frame
    _width, _height = testing_geometry
    _maze = np.zeros((_height, _width))
    _maze[1:_height-1, _width//2] = 1
    _maze[1, 1:_width-1] = 1
    _fig, _ax = plt.subplots(figsize=(2.5, 2.5), layout="constrained")
    _ax.imshow(_maze, cmap="Greys_r", vmin=0, vmax=1)
    _ax.plot(*_f["pose"], "ro")
    _dx, _dy = [(0,-1),(1,0),(0,1),(-1,0)][_f["direction"]]
    _ax.arrow(*_f["pose"], .45*_dx, .45*_dy, color="red", head_width=.2)
    _ax.set_title("Cue " + ("visible" if _f["visible"] else "hidden"))
    plt.close(_fig)
    _nodes = {
        "Maze / cue": mo.vstack([mo.as_html(_fig), testing_cue]),
        "Encoder input": tensor_panel(f"Observation ({len(_f['observation'])} features)", _f["observation"]),
        "Encoder latent": tensor_panel("Encoder latent z", _f["z"]),
        "Incoming bank": tensor_panel("Bank before prediction creation", _f["bank_before"], _f["managed_active"]),
        "Prediction context": mo.vstack([
            tensor_panel("Fixed USE rule", _f["probs"], _f["managed_active"]),
            tensor_panel("Actor context: delta, residual, age, usage, error", _f["context"]),
            mo.ui.table([dict(slot=i, age=int(_f["ages_before"][i]),
                decision=["WAIT", "USE", "DISCARD"][_f["management"][i]]
                    if _f["managed_active"][i] else "inactive",
                used=bool(_f["used"][i])) for i in range(len(_f["used"]))], selection=None)]),
        "Strategy memory": tensor_panel("Strategy latent (evaluation mean)", _f["strategy"]),
        "Actor": tensor_panel("Action probabilities", _f["actor"], probabilities=True,
                              labels=["left", "right", "forward"]),
        "Predictor / updated bank": mo.vstack([
            tensor_panel("Bank after creation (feeds next timestep)", _f["predictions"], _f["active"]),
            mo.ui.table([dict(slot=i, active=bool(_f["active"][i]),
                horizon=int(_f["horizons"][i]), created_at=int(_f["ids"][i]))
                for i in range(len(_f["active"]))], selection=None)]),
    }
    _edges = [("Maze / cue", "Encoder input"), ("Encoder input", "Encoder latent"),
        ("Encoder latent", "Prediction context"), ("Incoming bank", "Prediction context"),
        ("Encoder latent", "Strategy memory"), ("Prediction context", "Strategy memory"),
        ("Encoder latent", "Actor"), ("Strategy memory", "Actor"), ("Prediction context", "Actor"),
        ("Encoder latent", "Predictor / updated bank"), ("Strategy memory", "Predictor / updated bank"),
        ("Actor", "Predictor / updated bank"), ("Incoming bank", "Predictor / updated bank")]
    mo.vstack([
        mo.md(f"Episode: **{testing_report['steps']} steps** · success **{bool(testing_report['success'])}** · "
              f"selected action **{['left','right','forward'][_f['action']]}**. "
              "Gray bank rows are inactive. Memory and incoming predictions come from earlier timesteps; "
              "the updated bank feeds the next decision."),
        WidgetDAG(nodes=_nodes, edges=_edges),
    ])
    return


@app.cell
def _(mo):
    spike_module=mo.ui.dropdown(["encoder","predictor","critic","strategizer","actor"],
        value="strategizer",label="Spike-rate module")
    spike_module
    return (spike_module,)


@app.cell
def _(testing_frame,spike_module,plt):
    _raster,_ax=plt.subplots(figsize=(12,2.5),layout="constrained")
    _ax.imshow(testing_frame["spike_rates"][spike_module.value],
        aspect="auto",cmap="binary",vmin=0,vmax=1)
    _ax.set(xlabel="Neuron",ylabel="World / slot",title="Mean spike rate over internal Ghost ticks")
    plt.close(_raster)
    _raster
    return


if __name__ == "__main__":
    app.run()
