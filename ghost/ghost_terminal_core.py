#!/usr/bin/env python3
"""Learning engine for the interactive delayed-cue T-maze notebook.

Task
----
Each world is a sparse-reward T-maze.  A left/right cue is visible only for
the first two decisions.  At the later junction the physical observation is
identical for both cues, so a reactive policy cannot select the correct arm
without information carried through time.  Correct and wrong terminals give
+1 and -1 respectively; timeouts give -0.1 so waiting is not risk-free.

Architecture and online learning
--------------------------------
* Stateless spiking online encoder: either the legacy reconstruction/cue
  baseline or JEPA with causal reward-conditioned Adaptive LayerNorm.
* Stateful spiking predictor: latent prediction of a stop-gradient EMA target
  encoder.  The default is the historical one-step objective; an experimental
  strategizer-controlled timer can instead supervise the same shared predictor
  at a selected future step.  Reward is revealed only to the target, so its
  recurrent state must retain earlier evidence to predict affective modulation.
* Stateful spiking strategizer: predictor-feedback conditioned, deterministic
  strategy readout plus task-outcome desirability and uncertainty readouts.
  Bellec-style recurrent LIF eligibility and exact leaky-strategy-memory
  Jacobians carry causal derivatives online; sparse TD error modulates their
  reward trace using Adam.  Outcome statistics use distributional TD.
* Optional strategy-conditioned representation critic: distributional TD on
  ``Q(online_latent, detached_strategy)`` with EMA targets and a scaled direct
  gradient into the online encoder.
* Stateless spiking actor: current latent input, MLP strategy conditioning,
  one immediate action.  Its reward-modulated score eligibility can also
  update the encoder directly.  Neural state resets each environment
  decision, while eligibility does not.

Controls are ``stateless_strategizer`` and ``actor_only``.  Final evaluation
also intervenes on the learned strategy by shuffling or zeroing it.
"""

from __future__ import annotations

import copy
import contextlib
import io
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import marimo
except ImportError:  # The experiment engine does not require the notebook UI.
    class _NoopApp:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def cell(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]
            return lambda function: function

    class _MarimoFallback:
        App = _NoopApp

    marimo = _MarimoFallback()


__generated_with = "0.23.15"
app = marimo.App(width="full")


ACTIONS = ("left", "right", "forward")
ACTION_DIM = len(ACTIONS)
CONDITIONS = ("separated", "stateless_strategizer", "actor_only")
TRAINING_PREDICTOR_FEEDBACK_MODES = ("normal", "shuffle", "zero")


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
    encoder_target_tau: float = 0.005
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


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


class AdaptiveExploration:
    """Bounded exploration controlled by the weaker cue's success EMA."""

    def __init__(self, cfg: Config) -> None:
        self.minimum = cfg.adaptive_exploration_min
        self.maximum = cfg.adaptive_exploration_max
        self.decay = cfg.adaptive_exploration_ema_decay
        self.power = cfg.adaptive_exploration_power
        self.success_ema = np.zeros(2, np.float64)

    @property
    def rate(self) -> float:
        weakest = float(self.success_ema.min())
        return self.minimum + (self.maximum-self.minimum)*(
            1.0-weakest)**self.power

    def update(self, done: np.ndarray, success: np.ndarray,
               cue: np.ndarray) -> None:
        for cue_value in (0, 1):
            mask = done & (cue == cue_value)
            count = int(mask.sum())
            if count:
                batch_success = float(success[mask].mean())
                retained = self.decay**count
                self.success_ema[cue_value] = (
                    retained*self.success_ema[cue_value]
                    +(1.0-retained)*batch_success)


class AdaptiveTimerArbitration:
    """Hand control from action exploration to the strategic timer."""

    def __init__(self, cfg: Config, exploration: AdaptiveExploration) -> None:
        self.exploration = exploration
        self.minimum = cfg.adaptive_timer_min_influence
        self.patience = cfg.adaptive_timer_patience_episodes
        self.min_improvement = cfg.adaptive_timer_min_improvement
        self.exploration_threshold = cfg.adaptive_timer_exploration_threshold
        self.decay = cfg.adaptive_timer_gate_ema_decay
        self.gate = self.minimum
        self.stalled = False
        self.rescue_latched = False
        self.episodes_since_review = 0
        self.review_success = 0.0
        self.last_improvement = 0.0

    @property
    def exploration_pressure(self) -> float:
        span = self.exploration.maximum-self.exploration.minimum
        if span <= 0.0:
            return 0.0
        return float(np.clip(
            (self.exploration.rate-self.exploration.minimum)/span,
            0.0,
            1.0,
        ))

    @property
    def target(self) -> float:
        inverse_exploration = 1.0-self.exploration_pressure
        rescue = 1.0 if self.rescue_latched else 0.0
        return max(self.minimum, inverse_exploration, rescue)

    def update(self, done: np.ndarray) -> None:
        completed = int(done.sum())
        if not completed:
            return
        self.episodes_since_review += completed
        weakest = float(self.exploration.success_ema.min())
        if self.episodes_since_review >= self.patience:
            self.last_improvement = weakest-self.review_success
            self.stalled = (
                self.exploration_pressure >= self.exploration_threshold
                and self.last_improvement < self.min_improvement
            )
            self.rescue_latched = self.rescue_latched or self.stalled
            self.review_success = weakest
            self.episodes_since_review = 0
        retained = self.decay**completed
        self.gate = retained*self.gate+(1.0-retained)*self.target


class AdaptiveTimerStateMachine:
    """Select an exploration-led or timer-led path from online evidence."""

    def __init__(self, cfg: Config, exploration: AdaptiveExploration) -> None:
        self.exploration = exploration
        self.minimum = cfg.adaptive_timer_min_influence
        self.bootstrap = cfg.timer_controller_bootstrap_influence
        self.episodes_per_cue = cfg.timer_controller_episodes_per_cue
        self.progress_threshold = cfg.timer_controller_progress_threshold
        self.maintenance_progress_threshold = (
            cfg.adaptive_timer_min_improvement
        )
        self.success_threshold = cfg.timer_controller_success_threshold
        self.imbalance_threshold = cfg.timer_controller_imbalance_threshold
        self.timeout_threshold = cfg.timer_controller_timeout_threshold
        self.transition_hold = cfg.timer_controller_transition_hold_episodes
        self.decay = cfg.adaptive_timer_gate_ema_decay
        self.outcome_decay = cfg.adaptive_exploration_ema_decay
        self.gate = self.bootstrap
        self.state = "bootstrap"
        self.stalled = False
        self.rescue_latched = False
        self.last_improvement = 0.0
        self.review_success = 0.0
        self.cue_episodes = np.zeros(2, np.int64)
        self.review_cue_episodes = np.zeros(2, np.int64)
        self.timeout_ema = 0.0
        self.wrong_ema = 0.0
        self.curriculum_stage = 0
        self.transition_episodes_remaining = 0

    @property
    def exploration_pressure(self) -> float:
        span = self.exploration.maximum-self.exploration.minimum
        if span <= 0.0:
            return 0.0
        return float(np.clip(
            (self.exploration.rate-self.exploration.minimum)/span,
            0.0,
            1.0,
        ))

    @property
    def target(self) -> float:
        if self.rescue_latched:
            return 1.0
        if self.state == "bootstrap":
            return self.bootstrap
        inverse_exploration = 1.0-self.exploration_pressure
        if self.state == "curriculum_transition":
            return max(self.gate, inverse_exploration)
        return max(self.minimum, inverse_exploration)

    def _update_outcome_ema(self, completed: int, batch_value: float,
                            current: float) -> float:
        retained = self.outcome_decay**completed
        return retained*current+(1.0-retained)*batch_value

    def _review(self) -> None:
        weaker = float(self.exploration.success_ema.min())
        imbalance = float(np.ptp(self.exploration.success_ema))
        self.last_improvement = weaker-self.review_success
        competent = weaker >= 0.65
        required_progress = (
            self.progress_threshold
            if self.state == "bootstrap"
            else self.maintenance_progress_threshold
        )
        balanced_progress = (
            weaker >= self.success_threshold
            and imbalance <= self.imbalance_threshold
            and self.timeout_ema <= self.timeout_threshold
            and (self.last_improvement >= required_progress or competent)
        )
        self.stalled = not balanced_progress
        if self.stalled:
            self.rescue_latched = True
            self.state = "timer_rescue"
        else:
            self.state = "balanced_progress"
        self.review_success = weaker
        self.review_cue_episodes = self.cue_episodes.copy()

    def update(self, done: np.ndarray, success: np.ndarray,
               wrong: np.ndarray, cue: np.ndarray,
               curriculum_stage: int) -> None:
        completed = int(done.sum())
        if not completed:
            return

        finished_cues = cue[done]
        self.cue_episodes += np.bincount(finished_cues, minlength=2)
        timeout = done & ~success & ~wrong
        self.timeout_ema = self._update_outcome_ema(
            completed, float(timeout[done].mean()), self.timeout_ema)
        self.wrong_ema = self._update_outcome_ema(
            completed, float(wrong[done].mean()), self.wrong_ema)

        if curriculum_stage != self.curriculum_stage:
            self.curriculum_stage = curriculum_stage
            self.transition_episodes_remaining = self.transition_hold
            self.review_success = float(self.exploration.success_ema.min())
            self.review_cue_episodes = self.cue_episodes.copy()
            if not self.rescue_latched:
                self.state = "curriculum_transition"

        if self.transition_episodes_remaining > 0:
            self.transition_episodes_remaining = max(
                0, self.transition_episodes_remaining-completed)

        evidence = self.cue_episodes-self.review_cue_episodes
        enough_evidence = bool((evidence >= self.episodes_per_cue).all())
        if enough_evidence and self.transition_episodes_remaining == 0:
            self._review()

        retained = self.decay**completed
        self.gate = retained*self.gate+(1.0-retained)*self.target


def apply_training_predictor_feedback(
        feedback: torch.Tensor,
        mode: str,
        active: torch.Tensor,
    ) -> torch.Tensor:
    """Intervene on feedback without changing predictor learning or RNG state."""
    if mode == "normal":
        return feedback
    if mode == "zero":
        return torch.zeros_like(feedback)
    if mode != "shuffle":
        raise ValueError(
            "training predictor feedback must be normal, shuffle, or zero"
        )

    # Terminal feedback is never consumed by the clean trajectory because the
    # corresponding recurrent state is reset.  Shuffle only among active
    # worlds so terminal prediction errors cannot leak into a new episode.
    shuffled = torch.zeros_like(feedback)
    indices = torch.nonzero(active, as_tuple=False).flatten()
    if indices.numel() > 1:
        sources = torch.roll(indices, 1)
        shuffled[indices] = feedback[sources]
    return shuffled


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


class PredictorEncoderEprop:
    """Carries predictor error back to the earlier latent-head activity, allowing temporal credit assignment within the world model.

    The trace follows:

        encoder parameters
        --> latent
        --> predictor input current
        --> persistent predictor state 
        --> later predictor error
    
    Only encoder.latent_head is targeted initially
    """

    def __init__(
        self,
        predictor: Predictor,
        encoder: StatelessEncoder,
        worlds: int,
        trace_decay: float,
        gradient_clip: float
        ) -> None:

        self.predictor = predictor
        self.encoder = encoder
        self.worlds = worlds
        self.trace_decay = trace_decay
        self.gradient_clip = gradient_clip

        self.weight = encoder.latent_head.weight
        self.bias = encoder.latent_head.bias

        predictor_hidden = predictor.core.hidden_dim
        device = self.weight.device

        self.weight_trace = torch.zeros(
            worlds,
            predictor_hidden,
            *self.weight.shape,
            device=device
        )

        self.bias_trace = torch.zeros(
            worlds,
            predictor_hidden,
            *self.bias.shape,
            device=device
        )

        self.last_eligibility_norm = 0.0
        self.last_gradient_norm = 0.0

    def latent_jacobians(
            self,
            latent: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        if not latent.requires_grad:
            raise RuntimeError(
                "PredictorEncoderEprop requires an attached latent graph"
            )
        
        batch = latent.shape[0]

        basis = torch.eye(
            batch,
            device=latent.device,
            dtype=latent.dtype
        )

        weight_jacobians = []

        bias_jacobians = []

        for latent_dimension in range(latent.shape[1]):
            weight_gradient, bias_gradient = torch.autograd.grad(
                latent[:, latent_dimension],
                (self.weight,self.bias),
                grad_outputs = basis,
                is_grads_batched=True,
                retain_graph=True
            )

            weight_jacobians.append(weight_gradient.detach())
            bias_jacobians.append(bias_gradient.detach())

        weight_jacobian=torch.stack(
            weight_jacobians, dim=1
        )

        bias_jacobian=torch.stack(
            bias_jacobians, dim=1
        )
        return weight_jacobian, bias_jacobian

    def gradients(
            self,
            loss: torch.Tensor,
            latent: torch.Tensor
        ) -> Dict[str, torch.Tensor]:
        core = self.predictor.core

        if core.last_output is None:
            raise RuntimeError(
                "predictor core has no eligibility output"
            )

        learning_signal = torch.autograd.grad(
            loss,
            core.last_output,
            retain_graph=True
        )[0].detach()

        weight_jacobian,bias_jacobian = (
            self.latent_jacobians(latent)
        )

        latent_input_weight = (
            core.input.weight[:, :latent.shape[1]].detach()
        )

        weight_injection = torch.einsum(
            "hd,bdof->bhof",
            latent_input_weight,
            weight_jacobian
        )

        bias_injection = torch.einsum(
            "hd, bdo->bho",
            latent_input_weight,
            bias_jacobian
        )
        gradient_weight = torch.zeros_like(self.weight)
        gradient_bias = torch.zeros_like(self.bias)

        past_weight_trace = self.weight_trace.clone()
        past_bias_trace = self.bias_trace.clone()

        hidden = core.hidden_dim
        ticks = max(len(core.last_eligibility_records), 1)

        with torch.no_grad():
            for tick, (
                    _value,
                    _presynaptic,
                    pseudo_derivative
            ) in enumerate(core.last_eligibility_records):

                carry = core.decay * (
                    self.trace_decay if tick == 0 else 1.0
                )

                past_weight_trace.mul_(carry)
                past_bias_trace.mul_(carry)

                coefficient = (
                    learning_signal[:, :hidden]
                    + learning_signal[:, hidden:]
                    * pseudo_derivative
                ) / ticks

                gradient_weight.add_(
                    torch.einsum(
                        "bh,bhof->of",
                        coefficient,
                        past_weight_trace,
                    )
                )

                gradient_bias.add_(
                    torch.einsum(
                        "bh,bho->o",
                        coefficient,
                        past_bias_trace,
                    )
                )

                self.weight_trace.mul_(carry).add_(
                    weight_injection)

                self.bias_trace.mul_(carry).add_(
                    bias_injection)

            eligibility_norm = torch.sqrt(
                self.weight_trace.square().sum()
                + self.bias_trace.square().sum()
            )

            raw_gradient_norm = torch.sqrt(
                gradient_weight.square().sum()
                + gradient_bias.square().sum()
            )

            if self.gradient_clip > 0:
                scale = torch.clamp(
                    self.gradient_clip
                    / raw_gradient_norm.clamp_min(1e-12),
                    max=1.0,
                )

                gradient_weight.mul_(scale)
                gradient_bias.mul_(scale)

                clipped_gradient_norm = torch.minimum(
                    raw_gradient_norm,
                    torch.as_tensor(
                        self.gradient_clip,
                        device=raw_gradient_norm.device,
                    ),
                )
            else:
                clipped_gradient_norm = raw_gradient_norm

            self.last_eligibility_norm = float(
                eligibility_norm)

            self.last_gradient_norm = float(
                clipped_gradient_norm)

        return {
            "weight": gradient_weight,
            "bias": gradient_bias,
        }

    def install_add(
            self,
            gradients: Dict[str, torch.Tensor]
        ) -> None:

        with torch.no_grad():
            if self.weight.grad is None:
                self.weight.grad = gradients["weight"].clone()
            else:
                self.weight.grad.add_(gradients["weight"])

            if self.bias.grad is None:
                self.bias.grad = gradients["bias"].clone()
            else:
                self.bias.grad.add_(gradients["bias"])

    def reset(self, mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return

        with torch.no_grad():
            self.weight_trace[mask] = 0
            self.bias_trace[mask] = 0



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


class ContinuousSIGReg:
    """Running characteristic-function match to an isotropic Gaussian."""

    def __init__(
        self,
        dimension: int,
        projections: int,
        frequency_samples: int,
        max_frequency: float,
        trace_decay: float,
        seed: int,
    ) -> None:
        if not 0.0 <= trace_decay < 1.0:
            raise ValueError(
                "SIGReg trace_decay must be in [0, 1)"
            )

        self.dimension = dimension
        self.projections = projections
        self.frequency_samples = frequency_samples
        self.max_frequency = max_frequency
        self.trace_decay = trace_decay

        # These directions must remain fixed because every ECF trace
        # entry must continue referring to the same projection.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        directions = torch.randn(
            projections,
            dimension,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )

        self.directions_cpu = (
            directions
            / torch.linalg.vector_norm(
                directions,
                dim=1,
                keepdim=True,
            ).clamp_min(1e-12)
        )

        # These are initialized from the first observed batch.
        self.running_mean = None
        self.running_real = None
        self.running_imaginary = None

    def loss(
        self,
        states: torch.Tensor,
    ) -> torch.Tensor:
        if states.ndim != 2:
            raise ValueError(
                "SIGReg expects [samples, dimensions]"
            )

        if states.shape[1] != self.dimension:
            raise ValueError(
                "SIGReg state dimension mismatch"
            )

        directions = self.directions_cpu.to(
            device=states.device,
            dtype=states.dtype,
        )

        frequencies = torch.linspace(
            self.max_frequency / self.frequency_samples,
            self.max_frequency,
            self.frequency_samples,
            device=states.device,
            dtype=states.dtype,
        )

        current_mean = states.mean(
            dim=0,
            keepdim=True,
        )

        # Use the historical mean as the centering reference. It is
        # detached, so SIGReg cannot erase the current gradient by
        # differentiating through the centering operation.
        if self.running_mean is None:
            center_reference = current_mean.detach()
        else:
            center_reference = self.running_mean.detach()

        centered = states - center_reference

        projected = centered @ directions.T

        phase = (
            projected[:, :, None]
            * frequencies[None, None, :]
        )

        current_real = torch.cos(
            phase
        ).mean(dim=0)

        current_imaginary = torch.sin(
            phase
        ).mean(dim=0)

        decay = self.trace_decay
        update = 1.0 - decay

        # Update only the numerical running statistics. Historical
        # computation graphs are never retained.
        with torch.no_grad():
            if self.running_mean is None:
                self.running_mean = (
                    current_mean.detach().clone()
                )

                self.running_real = (
                    current_real.detach().clone()
                )

                self.running_imaginary = (
                    current_imaginary.detach().clone()
                )

            else:
                self.running_mean.mul_(decay).add_(
                    current_mean.detach(),
                    alpha=update,
                )

                self.running_real.mul_(decay).add_(
                    current_real.detach(),
                    alpha=update,
                )

                self.running_imaginary.mul_(decay).add_(
                    current_imaginary.detach(),
                    alpha=update,
                )

        gaussian_real = torch.exp(
            -0.5 * frequencies.square()
        )[None, :]

        real_residual = (
            self.running_real - gaussian_real
        ).detach()

        imaginary_residual = (
            self.running_imaginary
        ).detach()

        mean_residual = self.running_mean.detach()

        # This scalar is constructed for its gradient. Its numerical
        # value is not the actual running SIGReg objective.
        ecf_surrogate = 2.0 * (
            real_residual * current_real
            + imaginary_residual * current_imaginary
        ).mean()

        mean_surrogate = 2.0 * (
            mean_residual * current_mean
        ).mean()

        surrogate_loss = (
            ecf_surrogate
            + mean_surrogate
        )

        # This is the actual running objective used for diagnostics.
        metric_loss = (
            real_residual.square()
            + imaginary_residual.square()
        ).mean() + mean_residual.square().mean()

        # Forward value = metric_loss
        # Backward gradient = surrogate_loss gradient
        return (
            surrogate_loss
            + metric_loss
            - surrogate_loss.detach()
        )

    


class System:
    @property
    def timer_credit_influence(self) -> float:
        """Feedback-only arbitration leaves timer credit and context intact."""
        return self.timer_influence if self.cfg.timer_gate_scope == "all" else 1.0

    def __init__(self, cfg: Config, condition: str,
                 device: torch.device, seed: int) -> None:
        if cfg.timer_gate_scope not in ("all", "feedback-only"):
            raise ValueError("timer_gate_scope must be all or feedback-only")
        if cfg.adaptive_exploration_target not in ("actor", "strategy"):
            raise ValueError("adaptive_exploration_target must be actor or strategy")
        if cfg.adaptive_exploration_target == "strategy" and (
                not cfg.adaptive_exploration or condition == "actor_only"):
            raise ValueError("strategy exploration requires adaptive exploration and a strategizer")
        if cfg.encoder_learning_mode not in (
                "cue_auxiliary", "reward_eprop", "hybrid"):
            raise ValueError(
                "encoder_learning_mode must be cue_auxiliary, "
                "reward_eprop, or hybrid")
        if not 0.0 < cfg.encoder_target_tau <= 1.0:
            raise ValueError("encoder_target_tau must be in (0, 1]")
        if not 0.0 < cfg.representation_critic_target_tau <= 1.0:
            raise ValueError(
                "representation_critic_target_tau must be in (0, 1]")
        if cfg.critic_encoder_weight < 0.0:
            raise ValueError("critic_encoder_weight must be non-negative")
        if cfg.reward_adaln_strength < 0.0:
            raise ValueError("reward_adaln_strength must be non-negative")
        if not 0.0 <= cfg.predictor_trace_decay <= 1.0:
            raise ValueError("predictor_trace_decay must be in [0, 1]")
        if cfg.predictor_reward_event_weight < 1.0:
            raise ValueError(
                "predictor_reward_event_weight must be at least 1")
        if cfg.predictor_eprop_clip < 0.0:
            raise ValueError("predictor_eprop_clip must be non-negative")
        if not cfg.prediction_timer_durations:
            raise ValueError("prediction_timer_durations must not be empty")
        if any(duration < 1 for duration in cfg.prediction_timer_durations):
            raise ValueError("prediction timer durations must be positive")
        if len(set(cfg.prediction_timer_durations)) != len(
                cfg.prediction_timer_durations):
            raise ValueError("prediction timer durations must be unique")
        if not 0.0 <= cfg.exploration_rate <= 1.0:
            raise ValueError("exploration_rate must be in [0, 1]")
        if not (
            0.0 <= cfg.adaptive_exploration_min
            <= cfg.adaptive_exploration_max <= 1.0
        ):
            raise ValueError(
                "adaptive exploration bounds must satisfy 0 <= min <= max <= 1"
            )
        if not 0.0 <= cfg.adaptive_exploration_ema_decay < 1.0:
            raise ValueError(
                "adaptive_exploration_ema_decay must be in [0, 1)"
            )
        if cfg.adaptive_exploration_power <= 0.0:
            raise ValueError("adaptive_exploration_power must be positive")
        if not 0.0 <= cfg.adaptive_timer_min_influence <= 1.0:
            raise ValueError("adaptive_timer_min_influence must be in [0, 1]")
        if cfg.adaptive_timer_patience_episodes < 1:
            raise ValueError("adaptive_timer_patience_episodes must be positive")
        if cfg.adaptive_timer_min_improvement < 0.0:
            raise ValueError("adaptive_timer_min_improvement must be non-negative")
        if not 0.0 <= cfg.adaptive_timer_exploration_threshold <= 1.0:
            raise ValueError(
                "adaptive_timer_exploration_threshold must be in [0, 1]"
            )
        if not 0.0 <= cfg.adaptive_timer_gate_ema_decay < 1.0:
            raise ValueError("adaptive_timer_gate_ema_decay must be in [0, 1)")
        if cfg.adaptive_timer_arbitration and not (
            cfg.adaptive_exploration and cfg.use_strategic_prediction_timer
        ):
            raise ValueError(
                "adaptive timer arbitration requires adaptive exploration "
                "and the strategic prediction timer"
            )
        if cfg.adaptive_timer_state_machine and not (
            cfg.adaptive_exploration and cfg.use_strategic_prediction_timer
        ):
            raise ValueError(
                "adaptive timer state machine requires adaptive exploration "
                "and the strategic prediction timer"
            )
        if cfg.adaptive_timer_state_machine and cfg.adaptive_timer_arbitration:
            raise ValueError(
                "choose either adaptive timer arbitration or its state machine"
            )
        if not 0.0 <= cfg.timer_controller_bootstrap_influence <= 1.0:
            raise ValueError(
                "timer_controller_bootstrap_influence must be in [0, 1]"
            )
        if cfg.timer_controller_episodes_per_cue < 1:
            raise ValueError("timer_controller_episodes_per_cue must be positive")
        if cfg.timer_controller_progress_threshold < 0.0:
            raise ValueError(
                "timer_controller_progress_threshold must be non-negative"
            )
        if not 0.0 <= cfg.timer_controller_success_threshold <= 1.0:
            raise ValueError(
                "timer_controller_success_threshold must be in [0, 1]"
            )
        if not 0.0 <= cfg.timer_controller_imbalance_threshold <= 1.0:
            raise ValueError(
                "timer_controller_imbalance_threshold must be in [0, 1]"
            )
        if not 0.0 <= cfg.timer_controller_timeout_threshold <= 1.0:
            raise ValueError(
                "timer_controller_timeout_threshold must be in [0, 1]"
            )
        if cfg.timer_controller_transition_hold_episodes < 0:
            raise ValueError(
                "timer_controller_transition_hold_episodes must be non-negative"
            )
        if cfg.training_predictor_feedback not in (
                TRAINING_PREDICTOR_FEEDBACK_MODES):
            raise ValueError(
                "training_predictor_feedback must be normal, shuffle, or zero"
            )
        torch.manual_seed(seed)
        self.cfg, self.condition, self.device = cfg, condition, device
        self.encoder = StatelessEncoder(cfg).to(device)
        self.use_jepa = cfg.encoder_learning_mode == "reward_eprop"
        self.target_encoder = (
            copy.deepcopy(self.encoder).to(device)
            if self.use_jepa or cfg.use_representation_critic else None)
        if self.target_encoder is not None:
            self.target_encoder.eval()
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
        
        self.strategizer = Strategizer(
            cfg, persistent=condition == "separated").to(device)
        
        self.actor = Actor(
            cfg, persistent=condition == "actor_only").to(device)
        
        self.predictor = Predictor(cfg).to(device)

        #Continuous SIGReg on the encoder latent
        self.sigreg = (
            ContinuousSIGReg(
                dimension=cfg.latent_dim,
                projections=cfg.sigreg_projections,
                frequency_samples=cfg.sigreg_frequency_samples,
                max_frequency=cfg.sigreg_max_frequency,
                trace_decay=cfg.sigreg_trace_decay,
                seed=30_000 + seed,
            )
            if self.use_jepa and cfg.sigreg_weight > 0.0
            else None
        )
        
        #Continuous SIGReg on the strategy latent
        self.strategy_sigreg = (
            ContinuousSIGReg(
                dimension=cfg.strategy_dim,
                projections=cfg.strategy_sigreg_projections,
                frequency_samples=cfg.strategy_sigreg_frequency_samples,
                max_frequency=cfg.strategy_sigreg_max_frequency,
                trace_decay=cfg.strategy_sigreg_trace_decay,
                seed=40_000 + seed,
            )
            if (
                condition == "separated"
                and cfg.strategy_sigreg_weight > 0.0
            )
            else None
        )

        #Strategy Encoder Eprop
        self.strategy_encoder_eprop = (
            StrategyEncoderEprop(
                encoder=self.encoder,
                worlds=cfg.worlds,
                strategy_dim=cfg.strategy_dim,
                decay=cfg.strategy_encoder_trace_decay,
                learning_rate=cfg.strategy_encoder_eprop_lr,
                gradient_clip=cfg.strategy_encoder_eprop_clip,
            )
            if (
                cfg.use_strategy_encoder_eprop
                and condition != "actor_only"
            )
            else None
        )        

        # Keep policy-sampling RNG identical in critic-on/off ablations.
        rng_state = torch.random.get_rng_state()
        self.representation_critic = None
        if cfg.use_representation_critic:
            self.representation_critic = RepresentationCritic(cfg).to(device)
        torch.random.set_rng_state(rng_state)
        self.target_representation_critic = (
            copy.deepcopy(self.representation_critic).to(device)
            if self.representation_critic is not None else None)
        if self.target_representation_critic is not None:
            self.target_representation_critic.eval()
            for parameter in self.target_representation_critic.parameters():
                parameter.requires_grad_(False)
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=cfg.encoder_lr)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=cfg.predictor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.strategizer.outcome_head.parameters(), lr=cfg.critic_lr)
        self.representation_critic_optimizer = (
            torch.optim.Adam(
                self.representation_critic.parameters(),
                lr=cfg.representation_critic_lr)
            if self.representation_critic is not None else None)
        self.actor_parameters = module_parameters((
            self.actor.core, self.actor.norm, self.actor.head,
            self.actor.strategy_encoder))
        self.encoder_parameters = module_parameters((
            self.encoder.core, self.encoder.norm, self.encoder.latent_head,
            self.encoder.latent_norm))
        self.strategy_parameters = module_parameters((
            self.strategizer.core, self.strategizer.norm,
            self.strategizer.strategy_head,
            *((self.strategizer.gate_head,)
              if self.strategizer.gate_head is not None else ()),
            *((self.strategizer.timer_head,)
              if self.strategizer.timer_head is not None else ()),
            self.strategizer.feedback_encoder))
        self.actor_eprop = RewardEprop(
            self.actor_parameters, cfg.worlds, cfg.actor_trace_decay,
            cfg.actor_eprop_lr)

        self.predictor_eprop = (PredictorEprop(
            self.predictor, cfg.worlds, cfg.predictor_trace_decay,
            cfg.predictor_eprop_clip)
            if cfg.use_predictor_eprop else None)

        self.predictor_encoder_eprop = (PredictorEncoderEprop(
                self.predictor,
                self.encoder,
                cfg.worlds,
                cfg.predictor_trace_decay,
                cfg.predictor_eprop_clip
            )
            if self.use_jepa and cfg.use_predictor_encoder_eprop
            else None
            )
        
            
        self.encoder_eprop = (
            RewardEprop(
                self.encoder_parameters,
                cfg.worlds,
                cfg.encoder_trace_decay,
                cfg.encoder_eprop_lr,
            )
            if cfg.use_actor_encoder_eprop
            else None
        )
        self.strategy_eprop = RecurrentStrategyEprop(
            self.strategizer, self.strategy_parameters, cfg.worlds,
            cfg.strategy_trace_decay, cfg.strategy_eprop_lr,
            persistent=condition == "separated")


        self.feedback = torch.zeros(
            cfg.worlds,
            2 * cfg.latent_dim,
            device=device,
        )
        self.timer_influence = 1.0
        self.strategy_noise = torch.zeros(cfg.worlds, cfg.strategy_dim, device=device)
        self.strategy_noise_valid = torch.zeros(cfg.worlds, dtype=torch.bool, device=device)


        self.strategy_memory = torch.zeros(
            cfg.worlds, cfg.strategy_dim, device=device)
        self.desirability_memory = torch.zeros(cfg.worlds, device=device)
        self.outcome_logvar_memory = torch.zeros(cfg.worlds, device=device)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_encoder is None:
            return
        tau = self.cfg.encoder_target_tau
        for target, online in zip(
                self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(1-tau).add_(online, alpha=tau)

    @torch.no_grad()
    def update_target_representation_critic(self) -> None:
        if self.target_representation_critic is None:
            return
        tau = self.cfg.representation_critic_target_tau
        for target, online in zip(
                self.target_representation_critic.parameters(),
                self.representation_critic.parameters()):
            target.mul_(1-tau).add_(online, alpha=tau)

    def reset(self, mask: torch.Tensor) -> None:
        self.strategy_noise_valid[mask] = False
        if not self.cfg.persist_recurrent_state_across_episodes:
            self.strategizer.reset(mask); self.predictor.reset(mask)
            self.actor.reset(mask)

            self.feedback[mask] = 0
            self.strategy_memory[mask] = 0
            self.desirability_memory[mask] = 0
            self.outcome_logvar_memory[mask] = 0

        # Eligibility remains episode-scoped even when recurrent activity is
        # persistent, preventing rewards in a later episode from assigning
        # credit to actions taken in an earlier episode.
        self.actor_eprop.reset(mask); self.strategy_eprop.reset(mask)

        if self.predictor_eprop is not None:
            self.predictor_eprop.reset(mask)
        
        if self.predictor_encoder_eprop is not None:
            self.predictor_encoder_eprop.reset(mask)

        if self.encoder_eprop is not None:
            self.encoder_eprop.reset(mask)

        if self.strategy_encoder_eprop is not None:
            self.strategy_encoder_eprop.reset(mask)

    def strategy_and_action(self, latent: torch.Tensor,
                            deterministic: bool = False,
                            exploration: float | None = None):
        strategy_latent = (
            latent
            if self.strategy_encoder_eprop is not None
            else latent.detach()
        )

        actor_latent = (
            latent
            if self.encoder_eprop is not None
            else latent.detach()
        )

        strategy = self.strategizer(
            strategy_latent, self.feedback.detach(), deterministic,
            timer_influence=self.timer_credit_influence,
            previous_strategy=(self.strategy_memory.detach().requires_grad_(True)
                               if self.cfg.learned_strategy_memory else None))
        
        if self.condition == "actor_only":
            actor_strategy = torch.zeros_like(strategy["strategy"])
            actor_desirability = torch.zeros_like(strategy["desirability"])
            actor_outcome_logvar = torch.zeros_like(
                strategy["outcome_logvar"])
        elif self.condition == "separated":
            if self.cfg.learned_strategy_memory:
                actor_strategy = strategy["strategy"]
                actor_desirability = strategy["desirability"]
                actor_outcome_logvar = strategy["outcome_logvar"]
            else:
                keep = self.cfg.strategy_retention
                actor_strategy = (keep*self.strategy_memory.detach()
                                  +(1-keep)*strategy["strategy"])
                actor_desirability = (keep*self.desirability_memory.detach()
                                      +(1-keep)*strategy["desirability"])
                actor_outcome_logvar = (
                    keep*self.outcome_logvar_memory.detach()
                    +(1-keep)*strategy["outcome_logvar"])
            self.strategy_memory = actor_strategy.detach()
            self.desirability_memory = actor_desirability.detach()
            self.outcome_logvar_memory = actor_outcome_logvar.detach()
        else:
            actor_strategy = strategy["strategy"]
            actor_desirability = strategy["desirability"]
            actor_outcome_logvar = strategy["outcome_logvar"]
        exploration_amount = (
            self.cfg.exploration_rate if exploration is None else exploration)
        if self.cfg.adaptive_exploration_target == "strategy":
            if not deterministic and exploration_amount > 0:
                missing = ~self.strategy_noise_valid
                self.strategy_noise[missing] = (
                    2*torch.rand_like(self.strategy_noise[missing])-1)
                self.strategy_noise_valid[missing] = True
                # Episode-held, bounded additive readout noise. Memory above
                # remains clean; identity derivative preserves policy credit.
                actor_strategy = actor_strategy + exploration_amount*self.strategy_noise
            exploration_amount = 0.0
        actor = self.actor(
            actor_latent, actor_strategy, actor_desirability,
            actor_outcome_logvar, deterministic,
            exploration=(
                0.0 if deterministic else
                exploration_amount
            ))
        return (strategy, actor, actor_strategy, actor_desirability,
                actor_outcome_logvar)


def latent_reconstruction_update(
        system: System, observation: torch.Tensor,
        reward_context: torch.Tensor | None = None,
        policy_graph: bool = False):
    cue_target = observation[:, 9:12].argmax(-1)
    keep_encoder_graph = policy_graph and (
        system.use_jepa
        or system.encoder_eprop is not None
        or system.predictor_encoder_eprop is not None
    )


    if system.use_jepa:
        # JEPA mode never reconstructs observations and never consumes cue
        # labels as a learning target.  Its encoder update happens in
        # predictor_update against the EMA target encoder.
        latent, sigreg_state = (
            system.encoder.encode_with_pre_tanh(
                observation,
                reward_context,
            )
        )
        cue_logits = system.encoder.cue_head(latent)
        loss = torch.zeros((), device=observation.device)
    else:
        latent, reconstruction, cue_logits = system.encoder(
            observation, reward_context)
        sigreg_state = latent.detach()
        cue_weights = torch.tensor(
            (32.0, 32.0, 1.0), device=observation.device)
        cue_loss = F.cross_entropy(
            cue_logits, cue_target, weight=cue_weights)
        loss = F.mse_loss(reconstruction, observation)
        if system.cfg.encoder_learning_mode in ("cue_auxiliary", "hybrid"):
            loss = loss+system.cfg.cue_aux_weight*cue_loss
        system.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        system.encoder_optimizer.step()
        if keep_encoder_graph:
            latent, _, cue_logits = system.encoder(
                observation, reward_context)
        else:
            with torch.no_grad():
                latent, _, cue_logits = system.encoder(
                    observation, reward_context)
    with torch.no_grad():
        cue_prediction = cue_logits.argmax(-1)
        cue_accuracy = (cue_prediction == cue_target).float().mean()
        visible = cue_target < 2
        visible_correct = float(
            ((cue_prediction == cue_target)&visible).float().sum())
        visible_count = float(visible.float().sum())

    returned_sigreg_state = (
        sigreg_state
        if keep_encoder_graph
        else sigreg_state.detach()
    )

    return (
        latent if keep_encoder_graph else latent.detach(),
        float(loss.detach()),
        float(cue_accuracy),
        visible_correct,
        visible_count,
        returned_sigreg_state,
    )


@dataclass
class PendingTimedPrediction:
    due_step: torch.Tensor
    active: torch.Tensor
    forecast: torch.Tensor
    latent: torch.Tensor
    strategy: torch.Tensor
    desirability: torch.Tensor
    action: torch.Tensor
    timer_index: torch.Tensor
    predictor_mem: torch.Tensor
    predictor_spikes: torch.Tensor


class TimedPredictionBuffer:
    """Resolve strategy-timed latent forecasts without retaining BPTT graphs."""

    def __init__(self, cfg: Config, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self.step = 0
        self.pending: List[PendingTimedPrediction] = []

    def forecast(
            self,
            system: System,
            latent: torch.Tensor,
            strategy: torch.Tensor,
            desirability: torch.Tensor,
            action: torch.Tensor,
            timer_index: torch.Tensor,
            timer_duration: torch.Tensor,
        ) -> torch.Tensor:
        state = system.predictor.core.snapshot()
        if state is None:
            system.predictor.core.initial(latent.shape[0], latent.device)
            state = system.predictor.core.snapshot()
        assert state is not None
        predicted = system.predictor(
            latent.detach(), strategy.detach(), desirability.detach(), action,
            timer_index,
        )
        self.pending.append(PendingTimedPrediction(
            due_step=(self.step+timer_duration.detach()).clone(),
            active=torch.ones_like(timer_duration, dtype=torch.bool),
            forecast=predicted.detach().clone(),
            latent=latent.detach().clone(),
            strategy=strategy.detach().clone(),
            desirability=desirability.detach().clone(),
            action=action.detach().clone(),
            timer_index=timer_index.detach().clone(),
            predictor_mem=state[0].detach().clone(),
            predictor_spikes=state[1].detach().clone(),
        ))
        return predicted

    def resolve(
            self,
            system: System,
            target_latent: torch.Tensor,
            reward: torch.Tensor,
            done: torch.Tensor,
            train: bool,
        ) -> Tuple[torch.Tensor, float, float, float, float, float]:
        current_step = self.step+1
        error_sum = torch.zeros_like(target_latent)
        error_count = torch.zeros(
            target_latent.shape[0], 1,
            device=target_latent.device, dtype=target_latent.dtype)
        replay_losses = []
        squared_error_sum = torch.zeros((), device=target_latent.device)
        scalar_count = 0
        joy_error_sum = 0.0
        joy_event_count = 0.0

        live_state = system.predictor.core.snapshot()
        live_output = system.predictor.core.last_output
        live_records = system.predictor.core.last_eligibility_records

        for pending in self.pending:
            due = pending.active & (pending.due_step == current_step)
            if bool(due.any()):
                stored_error = target_latent.detach()[due]-pending.forecast[due]
                error_sum[due] += stored_error
                error_count[due] += 1
                squared_error_sum += stored_error.square().sum()
                scalar_count += stored_error.numel()
                decisive = reward.detach()[due].abs() >= 0.5
                if bool(decisive.any()):
                    joy_error_sum += float(
                        stored_error[decisive].square().mean(-1).sum())
                    joy_event_count += float(decisive.sum())

                if train:
                    system.predictor.core.restore((
                        pending.predictor_mem[due],
                        pending.predictor_spikes[due],
                    ))
                    replay_prediction = system.predictor(
                        pending.latent[due],
                        pending.strategy[due],
                        pending.desirability[due],
                        pending.action[due],
                        pending.timer_index[due],
                    )
                    per_world_mse = (
                        target_latent.detach()[due]-replay_prediction
                    ).square().mean(-1)
                    reward_conditioned = (
                        system.use_jepa and system.cfg.use_reward_adaln)
                    event_strength = (
                        reward.detach()[due].abs().clamp(max=1)
                        if reward_conditioned
                        else torch.zeros_like(reward[due])
                    )
                    weights = 1+(
                        system.cfg.predictor_reward_event_weight-1
                    )*event_strength
                    replay_losses.append((per_world_mse*weights).sum())

                pending.active[due] = False

            # Never compare a forecast from one episode against the next one.
            pending.active[done] = False

        self.pending = [
            pending for pending in self.pending
            if bool(pending.active.any())
        ]
        self.step = current_step

        if live_state is not None:
            system.predictor.core.restore(live_state)
        system.predictor.core.last_output = live_output
        system.predictor.core.last_eligibility_records = live_records

        prediction_loss = 0.0
        predictor_gradient_norm = 0.0
        if train and replay_losses:
            # Normalize by resolved forecasts, not by the number of timer
            # records, because several worlds can expire simultaneously.
            resolved_worlds = max(error_count.sum().item(), 1.0)
            loss = torch.stack(replay_losses).sum()/resolved_worlds
            system.predictor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.sqrt(sum(
                parameter.grad.square().sum()
                for parameter in system.predictor.parameters()
                if parameter.grad is not None
            ))
            if system.cfg.predictor_eprop_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    system.predictor.parameters(),
                    system.cfg.predictor_eprop_clip,
                )
                gradient_norm = torch.minimum(
                    gradient_norm,
                    torch.as_tensor(
                        system.cfg.predictor_eprop_clip,
                        device=gradient_norm.device,
                    ),
                )
            system.predictor_optimizer.step()
            prediction_loss = float(loss.detach())
            predictor_gradient_norm = float(gradient_norm.detach())

        prediction_mse = (
            float(squared_error_sum/scalar_count)
            if scalar_count else 0.0
        )
        mean_error = error_sum/error_count.clamp_min(1)
        return (
            mean_error,
            prediction_loss,
            prediction_mse,
            joy_error_sum,
            joy_event_count,
            predictor_gradient_norm,
        )


def timed_predictor_update(
        system: System,
        buffer: TimedPredictionBuffer,
        latent: torch.Tensor,
        strategy: torch.Tensor,
        desirability: torch.Tensor,
        action: torch.Tensor,
        timer_index: torch.Tensor,
        timer_duration: torch.Tensor,
        target_latent: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        train: bool = True,
    ):
    forecast = buffer.forecast(
        system, latent, strategy, desirability, action,
        timer_index, timer_duration,
    )
    (expired_error, prediction_loss, prediction_mse,
     joy_error_sum, joy_event_count, predictor_gradient_norm) = buffer.resolve(
        system, target_latent, reward, done, train,
    )
    predicted_change = forecast.detach()-latent.detach()
    feedback = torch.cat((predicted_change, expired_error), -1)
    return (
        feedback,
        prediction_loss,
        prediction_mse,
        joy_error_sum,
        joy_event_count,
        0.0,
        predictor_gradient_norm,
        0.0,
        0.0,
        0.0,
    )


def predictor_update(
        system: System,
        latent: torch.Tensor,
        strategy: torch.Tensor,
        desirability: torch.Tensor,
        action: torch.Tensor,
        next_latent: torch.Tensor,
        valid: torch.Tensor,
        reward: torch.Tensor,
        train_encoder: bool = False,
        sigreg_state: torch.Tensor | None = None,
    ):
        
    predictor_latent = (
        latent.detach()
        if system.cfg.detach_predictor_from_encoder
        else latent
    )

    #Predictor Call
    predicted_next_latent = system.predictor(
        predictor_latent,
        strategy.detach(),
        desirability.detach(),
        action,
    )

    target_next_latent = next_latent.detach()

    error = (
        target_next_latent
        - predicted_next_latent
    )

    per_world_mse = error.square().mean(
        dim=-1
    )

    reward_conditioned = (
        system.use_jepa and system.cfg.use_reward_adaln)

    event_strength = (
        reward.detach().abs().clamp(max=1)
        if reward_conditioned else torch.zeros_like(reward))

    sample_weight = valid.float()*(
        1+(system.cfg.predictor_reward_event_weight-1)*event_strength)


    prediction_loss = (
        per_world_mse
        * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1.0)

    
    encoder_regularization_loss = torch.zeros(
        (),
        device = prediction_loss.device,
        dtype = prediction_loss.dtype
    )

    variance_loss = torch.zeros_like(
        encoder_regularization_loss
    )

    sigreg_loss = torch.zeros_like(
        encoder_regularization_loss
    )

    #regularization Block
    if train_encoder:
        # Use the attached encoder latent here—not predictor_latent,
        # because predictor_latent may be detached.
        latent_std = torch.sqrt(
            latent.var(dim=0, unbiased=False) + 1e-4
        )

        variance_loss = F.relu(
            1.0 - latent_std
        ).mean()

        encoder_regularization_loss = (
            encoder_regularization_loss
            + system.cfg.jepa_variance_weight
            * variance_loss
        )

        if system.sigreg is not None:
            if sigreg_state is None:
                raise RuntimeError(
                    "SIGReg requires the attached pre-tanh encoder state"
                )

            if not sigreg_state.requires_grad:
                raise RuntimeError(
                    "SIGReg pre-tanh state was unexpectedly detached"
                )

            sigreg_loss = system.sigreg.loss(
                sigreg_state
            )

            encoder_regularization_loss = (
                encoder_regularization_loss
                + system.cfg.sigreg_weight
                * sigreg_loss
            )

    total_loss = (
        prediction_loss
        + encoder_regularization_loss
    )

    #Optimizer Section
    system.predictor_optimizer.zero_grad(
        set_to_none=True
    )

    if train_encoder:
        system.encoder_optimizer.zero_grad(
            set_to_none=True
    )

    recurrent_gradients = (
        system.predictor_eprop.gradients(prediction_loss)
        if system.predictor_eprop is not None else None)

    temporal_encoder_gradients = (
        system.predictor_encoder_eprop.gradients(
            prediction_loss,
            latent,
        )
        if (
            train_encoder
            and not system.cfg.detach_predictor_from_encoder
            and system.predictor_encoder_eprop is not None
        )
        else None
    )


    total_loss.backward()


    if recurrent_gradients is not None:
        system.predictor_eprop.install(
            recurrent_gradients
        )

    if temporal_encoder_gradients is not None:
        system.predictor_encoder_eprop.install_add(
            temporal_encoder_gradients
        )

    
    system.predictor_optimizer.step()

    if train_encoder:
        system.encoder_optimizer.step()
    

    predicted_change = (
        predicted_next_latent.detach()
        - latent.detach()
    )

    feedback = torch.cat(
        (
            predicted_change,
            error.detach(),
        ),
        dim=-1,
    )


    detached_per_world_mse = (
        per_world_mse.detach()
    )


    decisive_event = reward.detach().abs() >= 0.5

    joy_error_sum = float(
        detached_per_world_mse[
            decisive_event
        ].sum()
    )

    joy_event_count = float(decisive_event.sum())

    eligibility_norm = (
        system.predictor_eprop.last_eligibility_norm
        if system.predictor_eprop is not None else 0.0)
    gradient_norm = (
        system.predictor_eprop.last_gradient_norm
        if system.predictor_eprop is not None else 0.0)

    encoder_eligibility_norm = (
        system.predictor_encoder_eprop.last_eligibility_norm
        if system.predictor_encoder_eprop is not None
        else 0.0
    )

    encoder_temporal_gradient_norm = (
        system.predictor_encoder_eprop.last_gradient_norm
        if system.predictor_encoder_eprop is not None
        else 0.0
    )

    return (
        feedback,
        float(prediction_loss.detach()),
        float(detached_per_world_mse.mean()),
        joy_error_sum,
        joy_event_count,
        eligibility_norm,
        gradient_norm,
        encoder_eligibility_norm,
        encoder_temporal_gradient_norm,
        float(sigreg_loss.detach())
    )


def representation_critic_update(
        system: System, observation: torch.Tensor,
        strategy: torch.Tensor, reward: torch.Tensor,
        done: torch.Tensor, next_observation: torch.Tensor,
        next_strategy: torch.Tensor):
    if system.representation_critic is None:
        return 0.0, 0.0, 0.0

    # Re-encode after the JEPA/legacy auxiliary update so this graph is fresh.
    # Strategy is deliberately read-only: critic gradients shape the latent
    # and critic head, but never the recurrent strategizer state.
    latent = system.encoder.encode(observation)
    value, logvar = system.representation_critic(
        latent, strategy.detach())
    with torch.no_grad():
        target_next_latent = system.target_encoder.encode(next_observation)
        next_value, next_logvar = system.target_representation_critic(
            target_next_latent, next_strategy.detach())
        target = reward+system.cfg.gamma*(~done).float()*next_value
        target_variance = torch.where(
            done,
            torch.full_like(
                target, system.cfg.terminal_outcome_variance),
            system.cfg.gamma**2*next_logvar.exp())

    residual = target-value
    loss = 0.5*(
        (residual.square()+target_variance)*torch.exp(-logvar)
        +logvar).mean()
    system.representation_critic_optimizer.zero_grad(set_to_none=True)
    system.encoder_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        before = [
            parameter.detach().clone()
            for parameter in system.encoder_parameters]
    system.representation_critic_optimizer.step()
    if system.cfg.critic_encoder_weight > 0.0:
        original_lrs = [
            group["lr"] for group in system.encoder_optimizer.param_groups]
        for group, learning_rate in zip(
                system.encoder_optimizer.param_groups, original_lrs):
            group["lr"] = (
                learning_rate*system.cfg.critic_encoder_weight)
        system.encoder_optimizer.step()
        for group, learning_rate in zip(
                system.encoder_optimizer.param_groups, original_lrs):
            group["lr"] = learning_rate
    with torch.no_grad():
        step = torch.stack([
            (parameter-old).square().sum()
            for parameter, old in zip(
                system.encoder_parameters, before)]).sum().sqrt()
    return (float(loss.detach()), float(residual.detach().abs().mean()),
            float(step))

#Diagnostics

def centroid_accuracy(values: List[torch.Tensor], labels: List[torch.Tensor]):
    if not values:
        return 0.5
    x, y = torch.cat(values), torch.cat(labels)
    if not bool((y == 0).any()) or not bool((y == 1).any()):
        return 0.5
    centers = torch.stack((x[y == 0].mean(0), x[y == 1].mean(0)))
    prediction = torch.cdist(x, centers).argmin(-1)
    return float((prediction == y).float().mean())


def cue_strength(
    values: List[torch.Tensor],
    labels: List[torch.Tensor]):
    if not values:
        return 0.0, 0.0
    x = torch.cat(values)
    y = torch.cat(labels)

    left = x[y == 0]
    right = x[y == 1]

    if left.shape[0] == 0 or right.shape[0] == 0:
        return 0.0, 0.0
    
    left_center = left.mean(dim=0)
    right_center = right.mean(dim=0)

    distance = torch.linalg.vector_norm(
        right_center - left_center)
    
    left_scatter = (
        (left - left_center).square().sum(dim=-1).mean()
    )

    right_scatter = (
        (right - right_center).square().sum(dim=-1).mean()
    )

    within_cue_rms = torch.sqrt(
        0.5 * (left_scatter +right_scatter) + 1e-8
    )

    snr = distance/within_cue_rms

    return float(distance), float(snr)


def latent_distribution_health(
        values: List[torch.Tensor]
    ) -> Dict[str, float]:
    #effective rank
    #standard deviation
    #saturation
    if not values:
        return {
            "effective_rank": 0.0,
            "std_mean": 0.0,
            "std_min": 0.0,
            "std_max": 0.0,
            "saturation": 0.0,
            "correlation_rms": 0.0,
            "correlation_max": 0.0,
            "top_eigen_fraction": 0.0,
        }
    
    states = torch.cat(values, dim=0).float()

    standard_deviation = states.std(
        dim=0,
        unbiased=False
    )

    centered = states - states.mean(
        dim=0,
        keepdim=True
    )

    covariance = (
        centered.T @ centered / max(centered.shape[0] - 1,1)
    )

    eigenvalues = torch.linalg.eigvalsh(
        covariance
    ).clamp_min(0.0)

    total_variance = eigenvalues.sum()

    # Standardize each dimension so correlation is not dominated
    # by differences in scale.
    normalized = (
        centered
        / standard_deviation.clamp_min(1e-6)
    )

    correlation = (
        normalized.T @ normalized
        / max(normalized.shape[0], 1)
    )

    dimension = correlation.shape[0]

    off_diagonal_mask = ~torch.eye(
        dimension,
        device=correlation.device,
        dtype=torch.bool,
    )

    off_diagonal_correlation = correlation[
        off_diagonal_mask
    ]

    correlation_rms = float(
        torch.sqrt(
            off_diagonal_correlation.square().mean()
        )
    )

    correlation_max = float(
        off_diagonal_correlation.abs().max()
    )

    top_eigen_fraction = float(
        eigenvalues.max()
        / total_variance.clamp_min(1e-12)
    )

    if float(total_variance) <= 1e-12:
        effective_rank = 0.0

    else:
        proportions = eigenvalues / total_variance

        entropy = -(
            proportions * torch.log(proportions.clamp_min(1e-12))
        ).sum()

        effective_rank = float(torch.exp(entropy))

    saturation = float(
        (states.abs() > 0.95).float().mean()
    )

    return {
        "effective_rank": effective_rank,
        "std_mean": float(standard_deviation.mean()),
        "std_min": float(standard_deviation.min()),
        "std_max": float(standard_deviation.max()),
        "saturation": saturation,
        "correlation_rms": correlation_rms,
        "correlation_max": correlation_max,
        "top_eigen_fraction": top_eigen_fraction,
    }

  
def run_condition(cfg: Config, condition: str, seed: int,
                  device: torch.device, progress_callback=None):
    env = BatchedTMaze(cfg, seed*1000+17)
    diagnostics = None
    if cfg.critic_diagnostics_dir:
        from experiments.delayed_cue_tmaze.critic_diagnostics import CriticDiagnostics
        diagnostics = CriticDiagnostics(env, seed, condition)
    system = System(cfg, condition, device, seed)
    capture = None
    if cfg.credit_capture_dir:
        if condition != "separated" or cfg.use_strategic_prediction_timer or cfg.adaptive_exploration or cfg.persist_recurrent_state_across_episodes:
            raise ValueError("credit capture requires separated baseline without timer, adaptive exploration or episode persistence")
        if cfg.credit_capture_every <= 0 or cfg.credit_capture_length <= 0:
            raise ValueError("credit capture interval and length must be positive")
        from experiments.delayed_cue_tmaze.capture_credit import CreditCapture
        capture = CreditCapture(cfg, seed, condition)
    exploration_scheduler = (
        AdaptiveExploration(cfg) if cfg.adaptive_exploration else None
    )
    if cfg.adaptive_timer_state_machine and exploration_scheduler is not None:
        timer_arbitration = AdaptiveTimerStateMachine(
            cfg, exploration_scheduler
        )
    elif cfg.adaptive_timer_arbitration and exploration_scheduler is not None:
        timer_arbitration = AdaptiveTimerArbitration(
            cfg, exploration_scheduler
        )
    else:
        timer_arbitration = None
    timed_predictions = (
        TimedPredictionBuffer(cfg, device)
        if cfg.use_strategic_prediction_timer else None
    )
    print(
        "predictor_encoder_eprop_enabled=",
        system.predictor_encoder_eprop is not None
    )
    observation = torch.tensor(env.observation(), device=device)
    completed = episodes = successes = wrong_total = timeout_total = 0
    window_steps = window_episodes = window_successes = window_wrong = 0
    window_hallways = []
    window_timeouts = 0
    sums: Dict[str, float] = {}
    strategy_values: List[torch.Tensor] = []
    encoder_values: List[torch.Tensor] = []
    predictor_values: List[torch.Tensor] = []
    strategy_labels: List[torch.Tensor] = []

    strategy_delay_values = {
        "cue": [],
        "delay_1_4": [],
        "delay_5_8": [],
        "delay_9_plus": [],
    }

    strategy_delay_labels = {
        "cue": [],
        "delay_1_4": [],
        "delay_5_8": [],
        "delay_9_plus": [],
    }

    visible_encoder_values: List[torch.Tensor] = []
    visible_encoder_labels: List[torch.Tensor] = []
    latent_distribution_values: List[torch.Tensor] = []
    previous_strategy = torch.zeros(
        cfg.worlds, cfg.strategy_dim, device=device)
    reward_context = torch.zeros(cfg.worlds, device=device)
    reported_cue_assignments = np.zeros(2, np.int64)
    reported_cue_probability_sum = 0.0
    timer_counts = np.zeros(len(cfg.prediction_timer_durations), np.int64)

    def add(name: str, value: float) -> None:
        sums[name] = sums.get(name, 0.0)+value

    while completed < cfg.transitions:
        decision_age_np = env.age.copy()
        decision_cue_np = env.cue.copy()


        (latent, reconstruction, cue_accuracy, cue_visible_correct, cue_visible_count, sigreg_state) = latent_reconstruction_update(system, observation, reward_context, policy_graph=True)
        
        latent_distribution_values.append(
            latent.detach().cpu()
        )
        visible_np = env.age < cfg.cue_steps
        if visible_np.any():
            visible_mask = torch.tensor(
                visible_np,
                device=device,
                dtype=torch.bool
            )

            visible_encoder_values.append(
                latent.detach()[visible_mask].cpu()
            )

            visible_encoder_labels.append(
                torch.tensor(
                    env.cue[visible_np],
                    dtype=torch.long
                )
            )
        exploration_rate = (
            exploration_scheduler.rate
            if exploration_scheduler is not None
            else cfg.exploration_rate
        )
        timer_influence = (
            timer_arbitration.gate
            if timer_arbitration is not None else 1.0
        )
        system.timer_influence = timer_influence
        if capture is not None:
            capture.before(system, env, latent, completed)
        (strategy, actor, actor_strategy, actor_desirability,
         actor_outcome_logvar) = system.strategy_and_action(
             latent, exploration=exploration_rate)
        if cfg.use_strategic_prediction_timer:
            timer_counts += np.bincount(
                strategy["timer_index"].detach().cpu().numpy(),
                minlength=len(cfg.prediction_timer_durations),
            )

        decision_age = torch.tensor(
            decision_age_np,
            device=device,
            dtype=torch.long,
        )

        decision_cue = torch.tensor(
            decision_cue_np,
            device=device,
            dtype=torch.long,
        )

        delay_masks = {
            "cue": (
                decision_age < cfg.cue_steps
            ),

            "delay_1_4": (
                (decision_age >= cfg.cue_steps)
                & (decision_age < cfg.cue_steps + 4)
            ),

            "delay_5_8": (
                (decision_age >= cfg.cue_steps + 4)
                & (decision_age < cfg.cue_steps + 8)
            ),

            "delay_9_plus": (
                decision_age >= cfg.cue_steps + 8
            ),
        }

        for delay_name, delay_mask in delay_masks.items():
            if bool(delay_mask.any()):
                strategy_delay_values[delay_name].append(
                    actor_strategy.detach()[
                        delay_mask
                    ].cpu()
                )

                strategy_delay_labels[delay_name].append(
                    decision_cue[
                        delay_mask
                    ].detach().cpu()
                )


        # Eligibility is accumulated before reward arrives.  The actor keeps
        # its score trace; the strategizer additionally carries recurrent
        # synaptic and external strategy-memory eligibility through time.
        actor_trace = system.actor_eprop.accumulate(actor["logp"])
        encoder_trace = (
            system.encoder_eprop.accumulate(actor["logp"])
            if system.encoder_eprop is not None else 0.0)
        if condition != "actor_only":
            keep = cfg.strategy_retention if condition == "separated" else 0.0
            strategy_trace = system.strategy_eprop.accumulate(
                strategy["proposal"], strategy["gate"],
                strategy["previous_strategy"], actor_strategy,
                actor["logp"], keep,
                learned_gate=cfg.learned_strategy_memory,
                timer_logp=(system.timer_credit_influence*strategy["timer_logp"]
                    if cfg.use_strategic_prediction_timer else None))

            strategy_encoder_trace = (
                system.strategy_encoder_eprop.accumulate(
                    proposal=strategy["proposal"],
                    gate=strategy["gate"],
                    previous_strategy=strategy["previous_strategy"],
                    actor_strategy=actor_strategy,
                    actor_logp=actor["logp"],
                    learned_gate=cfg.learned_strategy_memory,
                    keep=keep,
                )
                if system.strategy_encoder_eprop is not None
                else 0.0
            )            
        else:
            strategy_trace = 0.0
            strategy_encoder_trace = 0.0

        strategy_sigreg_loss = 0.0
        strategy_sigreg_gradient = 0.0
        strategy_sigreg_gradients = None

        if system.strategy_sigreg is not None:
            strategy_sigreg_objective = system.strategy_sigreg.loss(
                strategy["proposal_pre_tanh"]
            )

            weighted_strategy_sigreg = (
                cfg.strategy_sigreg_weight
                * strategy_sigreg_objective
            )

            raw_strategy_sigreg_gradients = torch.autograd.grad(
                weighted_strategy_sigreg,
                system.strategy_parameters,
                retain_graph=True,
                allow_unused=True,
            )

            strategy_sigreg_gradients = [
                (
                    gradient.detach()
                    if gradient is not None
                    else None
                )
                for gradient in raw_strategy_sigreg_gradients
            ]

            gradient_square = torch.zeros(
                (),
                device=device,
            )

            for gradient in strategy_sigreg_gradients:
                if gradient is not None:
                    gradient_square += gradient.square().sum()

            strategy_sigreg_loss = float(
                strategy_sigreg_objective.detach()
            )

            strategy_sigreg_gradient = float(
                gradient_square.sqrt()
            )

        with torch.no_grad():
            permutation = torch.roll(torch.arange(
                cfg.worlds, device=device), 1)
            real_prob = actor["logits"].softmax(-1)
            # Actor is stateless in the separated conditions, so these probes
            # cannot disturb a hidden action state.
            if condition != "actor_only":
                shuffled = system.actor(
                    latent, actor_strategy[permutation],
                    actor_desirability[permutation],
                    actor_outcome_logvar[permutation], deterministic=True)[
                        "logits"].softmax(-1)
                zeroed = system.actor(
                    latent, torch.zeros_like(actor_strategy),
                    torch.zeros_like(actor_desirability),
                    torch.zeros_like(actor_outcome_logvar),
                    deterministic=True)[
                        "logits"].softmax(-1)
                shuffle_tv = float(0.5*(real_prob-shuffled).abs().sum(-1).mean())
                zero_tv = float(0.5*(real_prob-zeroed).abs().sum(-1).mean())
            else:
                shuffle_tv = zero_tv = 0.0

        if diagnostics is not None:
            diagnostics.before(env, actor["action"].detach().cpu().numpy(),
                               strategy["desirability"].detach().cpu().numpy(),
                               strategy["outcome_logvar"].detach().cpu().numpy())
        next_np, reward_np, done_np, success_np, wrong_np, cue_np, age_np = (
            env.step(actor["action"].detach().cpu().numpy()))
        if diagnostics is not None:
            diagnostics.after(reward_np, done_np, cfg.gamma)
        if exploration_scheduler is not None:
            exploration_scheduler.update(done_np, success_np, cue_np)
        if timer_arbitration is not None:
            if isinstance(timer_arbitration, AdaptiveTimerStateMachine):
                timer_arbitration.update(
                    done_np,
                    success_np,
                    wrong_np,
                    cue_np,
                    env.curriculum_stage,
                )
            else:
                timer_arbitration.update(done_np)
        next_observation = torch.tensor(next_np, device=device)
        prediction_observation = torch.tensor(
            env.transition_observation, device=device)
        done = torch.tensor(done_np, device=device, dtype=torch.bool)
        reward = torch.tensor(reward_np, device=device)
        # Terminal outcomes are the only non-zero rewards in this task, so
        # terminal transitions must train the predictor.  Its target is the
        # true post-action observation captured before the environment reset.
        valid_prediction = torch.ones_like(done)
        next_reward_context = torch.where(
            done, torch.zeros_like(reward), reward)
        if system.use_jepa:
            (_, next_reconstruction, next_cue_accuracy,
             next_cue_visible_correct,
             next_cue_visible_count, _) = latent_reconstruction_update(
                 system, next_observation, next_reward_context)
            with torch.no_grad():
                target_next_latent = system.target_encoder.encode(
                    prediction_observation, reward)
                neutral_target = system.target_encoder.encode(
                    prediction_observation, torch.zeros_like(reward))
                reward_latent_shift = (
                    target_next_latent-neutral_target).square().mean(-1)
                rewarded = reward != 0
                reward_latent_shift = float(
                    reward_latent_shift[rewarded].mean()
                    if bool(rewarded.any()) else 0.0)

            if timed_predictions is not None:
                (new_feedback, predictor_loss, prediction_mse,
                 joy_prediction_error, joy_event_count,
                 predictor_eligibility, predictor_eprop_gradient,
                 predictor_encoder_eligibility,
                 predictor_encoder_eprop_gradient,
                 sigreg_loss) = timed_predictor_update(
                    system,
                    timed_predictions,
                    latent,
                    actor_strategy,
                    actor_desirability,
                    actor["action"],
                    strategy["timer_index"],
                    strategy["timer_duration"],
                    target_next_latent,
                    reward,
                    done,
                )
            else:
                (new_feedback, predictor_loss, prediction_mse,
                joy_prediction_error, joy_event_count,
                predictor_eligibility, predictor_eprop_gradient,
                predictor_encoder_eligibility,
                predictor_encoder_eprop_gradient,
                sigreg_loss) = predictor_update(
                    system,
                    latent,
                    actor_strategy,
                    actor_desirability,
                    actor["action"],
                    target_next_latent,
                    valid_prediction,
                    reward,
                    train_encoder=True,
                    sigreg_state=sigreg_state,
                )

            # The policy/value bootstrap must use the newly updated online
            # representation, not the slowly moving JEPA target.
            with torch.no_grad():
                next_latent = system.encoder.encode(
                    next_observation, next_reward_context)
        else:
            (next_latent, next_reconstruction, next_cue_accuracy,
             next_cue_visible_correct,
             next_cue_visible_count, _) = latent_reconstruction_update(
                 system, next_observation, next_reward_context)
            
            if timed_predictions is not None:
                with torch.no_grad():
                    timed_target_latent = system.encoder.encode(
                        prediction_observation, reward)
                (
                    new_feedback,
                    predictor_loss,
                    prediction_mse,
                    joy_prediction_error,
                    joy_event_count,
                    predictor_eligibility,
                    predictor_eprop_gradient,
                    predictor_encoder_eligibility,
                    predictor_encoder_eprop_gradient,
                    sigreg_loss,
                ) = timed_predictor_update(
                    system,
                    timed_predictions,
                    latent,
                    actor_strategy,
                    actor_desirability,
                    actor["action"],
                    strategy["timer_index"],
                    strategy["timer_duration"],
                    timed_target_latent,
                    reward,
                    done,
                )
            else:
                (
                    new_feedback,
                    predictor_loss,
                    prediction_mse,
                    joy_prediction_error,
                    joy_event_count,
                    predictor_eligibility,
                    predictor_eprop_gradient,
                    predictor_encoder_eligibility,
                    predictor_encoder_eprop_gradient,
                    sigreg_loss,
                ) = predictor_update(
                    system,
                    latent,
                    actor_strategy,
                    actor_desirability,
                    actor["action"],
                    next_latent,
                    valid_prediction,
                    reward,
                )
            reward_latent_shift = 0.0

        new_feedback = apply_training_predictor_feedback(
            new_feedback,
            cfg.training_predictor_feedback,
            ~done,
        )
        if timed_predictions is not None:
            new_feedback = timer_influence*new_feedback

        # Provisional next value advances from the current strategic state but
        # is immediately rolled back: no state or graph is consumed twice.
        snapshot = system.strategizer.snapshot()
        previous_feedback = system.feedback
        system.feedback = new_feedback.detach()
        with torch.no_grad():
            next_strategy = system.strategizer(
                next_latent, system.feedback, deterministic=True,
                timer_influence=system.timer_credit_influence,
                previous_strategy=(system.strategy_memory.detach()
                    if cfg.learned_strategy_memory else None))
            next_value = next_strategy["desirability"]
            next_outcome_logvar = next_strategy["outcome_logvar"]
        system.strategizer.restore(snapshot)
        if condition == "actor_only":
            next_actor_strategy = torch.zeros_like(
                next_strategy["strategy"])
        elif condition == "separated" and not cfg.learned_strategy_memory:
            keep = cfg.strategy_retention
            next_actor_strategy = (
                keep*system.strategy_memory.detach()
                +(1-keep)*next_strategy["strategy"])
        else:
            next_actor_strategy = next_strategy["strategy"]

        (representation_critic_loss,
         representation_critic_mae,
         critic_encoder_step) = representation_critic_update(
             system, observation, actor_strategy, reward, done,
             next_observation, next_actor_strategy)
        td = (reward+cfg.gamma*(~done).float()*next_value
              -strategy["desirability"].detach()).clamp(
                  -cfg.td_clip, cfg.td_clip)

        if capture is not None:
            capture.pre_update("actor", system.actor_eprop, td)
        actor_direction, actor_step = system.actor_eprop.apply(td)
        if capture is not None:
            capture.post_update("actor", system.actor_eprop)
            capture.after(system, actor["action"], reward, done, td, latent,
                          actor_strategy, actor_desirability, actor_outcome_logvar,
                          actor["logp"])
        if system.encoder_eprop is not None:
            encoder_direction, encoder_step = system.encoder_eprop.apply(td)
        else:
            encoder_direction = encoder_step = 0.0

        if system.strategy_encoder_eprop is not None:
            (
                strategy_encoder_direction,
                strategy_encoder_step,
            ) = system.strategy_encoder_eprop.apply(td)
        else:
            strategy_encoder_direction = 0.0
            strategy_encoder_step = 0.0

        system.update_target_encoder()
        system.update_target_representation_critic()
        if condition != "actor_only":
            if capture is not None:
                capture.pre_update("strategy", system.strategy_eprop, td)
            strategy_direction, strategy_step = system.strategy_eprop.apply(
                td,
                minimizing_gradients=strategy_sigreg_gradients,
            )
            if capture is not None:
                capture.post_update("strategy", system.strategy_eprop)
                capture.end_step()
        else:
            strategy_direction = strategy_step = 0.0
        target = reward+cfg.gamma*(~done).float()*next_value.detach()
        # Distributional Bellman target.  Propagating the next state's
        # variance prevents long runs of censored zero reward from teaching
        # false certainty before a terminal outcome has been observed.
        target_variance = torch.where(
            done,
            torch.full_like(target, cfg.terminal_outcome_variance),
            cfg.gamma**2*next_outcome_logvar.detach().exp())
        outcome_residual = target-strategy["desirability"]
        critic_loss = 0.5*(
            (outcome_residual.square()+target_variance)*torch.exp(
                -strategy["outcome_logvar"])
            +strategy["outcome_logvar"]).mean()
        system.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward(); system.critic_optimizer.step()

        system.feedback = new_feedback.detach()
        system.reset(done)
        observation = next_observation
        reward_context = next_reward_context
        batch = cfg.worlds
        completed += batch; window_steps += batch
        timed_out = int((done_np & ~success_np & ~wrong_np).sum())
        finished, won, lost = (int(done_np.sum()), int(success_np.sum()),
                               int(wrong_np.sum()))
        episodes += finished; successes += won; wrong_total += lost
        timeout_total += timed_out
        window_hallways.extend(dict(hallway_length=int(env.transition_hallway_length[w]),
            curriculum_stage=int(env.transition_episode_stage[w])+1,cue=int(cue_np[w]),
            success=float(success_np[w])) for w in np.flatnonzero(done_np))
        window_episodes += finished; window_successes += won
        window_wrong += lost; window_timeouts += timed_out
        hidden_mask = torch.tensor(
            (age_np >= cfg.cue_steps)&(~done_np), device=device)
        if bool(hidden_mask.any()):
            strategy_values.append(actor_strategy.detach()[hidden_mask].cpu())
            encoder_values.append(latent.detach()[hidden_mask].cpu())
            predictor_values.append(
                system.predictor.core.last_output.detach()[hidden_mask].cpu())
            hidden_labels = torch.tensor(
                cue_np[hidden_mask.cpu().numpy()], dtype=torch.long)
            strategy_labels.append(hidden_labels)
        stability = F.cosine_similarity(
            actor_strategy.detach(), previous_strategy, -1).mean()
        previous_strategy = actor_strategy.detach()
        success_mask = torch.tensor(success_np, device=device,
                                    dtype=torch.bool)
        wrong_mask = torch.tensor(wrong_np, device=device,
                                  dtype=torch.bool)
        timeout_mask = done & ~success_mask & ~wrong_mask
        for name, value in (
            ("reward", float(reward.mean())),
            ("exploration_rate", exploration_rate),
            ("timer_influence", timer_influence),
            ("entropy", float(actor["entropy"].detach().mean())),
            ("timer_entropy", float(
                strategy["timer_entropy"].detach().mean())),
            ("timer_duration", float(
                strategy["timer_duration"].detach().float().mean())),
            ("desirability", float(
                strategy["desirability"].detach().mean())),
            ("outcome_std", float(
                (0.5*strategy["outcome_logvar"].detach()).exp().mean())),
            ("outcome_residual", float(
                outcome_residual.detach().abs().mean())),
            ("outcome_nll", float(critic_loss.detach())),
            ("representation_critic_loss", representation_critic_loss),
            ("representation_critic_mae", representation_critic_mae),
            ("terminal_outcome_abs_error", float(
                (strategy["desirability"].detach()-reward).abs()[done].sum())),
            ("terminal_outcome_count", float(done.sum())),
            ("success_desirability", float(
                strategy["desirability"].detach()[success_mask].sum())),
            ("success_count", float(success_mask.sum())),
            ("wrong_desirability", float(
                strategy["desirability"].detach()[wrong_mask].sum())),
            ("wrong_count", float(wrong_mask.sum())),
            ("timeout_desirability", float(
                strategy["desirability"].detach()[timeout_mask].sum())),
            ("timeout_count", float(timeout_mask.sum())),
            ("reconstruction", 0.5*(reconstruction+next_reconstruction)),
            ("encoder_cue_accuracy", 0.5*(cue_accuracy+next_cue_accuracy)),
            ("visible_cue_correct", cue_visible_correct+next_cue_visible_correct),
            ("visible_cue_count", cue_visible_count+next_cue_visible_count),
            ("predictor_loss", predictor_loss), ("prediction_mse", prediction_mse),
            ("joy_prediction_error", joy_prediction_error),
            ("joy_event_count", joy_event_count),
            ("predictor_eligibility", predictor_eligibility),
            ("predictor_eprop_gradient", predictor_eprop_gradient),
            ("predictor_encoder_eligibility", predictor_encoder_eligibility),
            ("predictor_encoder_eprop_gradient", predictor_encoder_eprop_gradient),

            ("strategy_sigreg_loss", strategy_sigreg_loss),
            ("strategy_sigreg_gradient", strategy_sigreg_gradient),

            ("strategy_encoder_trace", strategy_encoder_trace),
            ("strategy_encoder_memory",
            system.strategy_encoder_eprop.last_memory_norm
            if system.strategy_encoder_eprop is not None else 0.0),
            ("strategy_encoder_score",
            system.strategy_encoder_eprop.last_score_norm
            if system.strategy_encoder_eprop is not None else 0.0),
            ("strategy_encoder_gradient", strategy_encoder_direction),
            ("strategy_encoder_step", strategy_encoder_step),

            ("sigreg_loss", sigreg_loss),
            ("reward_latent_shift", reward_latent_shift),
            ("td_abs", float(td.abs().mean())), ("actor_trace", actor_trace),
            ("encoder_trace", encoder_trace),
            ("strategy_trace", strategy_trace),
            ("strategy_recurrent_e",
             system.strategy_eprop.last_recurrent_norm),
            ("strategy_memory_e", system.strategy_eprop.last_memory_norm),
            ("strategy_score_e", system.strategy_eprop.last_score_norm),
            ("actor_direction", actor_direction),
            ("encoder_direction", encoder_direction),
            ("strategy_direction", strategy_direction),
            ("actor_step", actor_step), ("encoder_step", encoder_step),
            ("critic_encoder_step", critic_encoder_step),
            ("strategy_step", strategy_step),
            ("shuffle_tv", shuffle_tv), ("zero_tv", zero_tv),
            ("strategy_stability", float(stability))):
            add(name, value)
        add("strategy_gate", float(strategy["gate"].detach().mean()))
        add("strategy_gate_saturation", float(
            ((strategy["gate"].detach() < 0.01)
             |(strategy["gate"].detach() > 0.99)).float().mean()))

        if window_steps >= cfg.report_every:
            decisions = window_steps/cfg.worlds

            strategy_decode = centroid_accuracy(
                strategy_values, strategy_labels)

            strategy_delay_decode = {
                delay_name: centroid_accuracy(
                    strategy_delay_values[delay_name],
                    strategy_delay_labels[delay_name],
                )
                for delay_name in strategy_delay_values
            }

            strategy_delay_strength = {
                delay_name: cue_strength(
                    strategy_delay_values[delay_name],
                    strategy_delay_labels[delay_name],
                )
                for delay_name in strategy_delay_values
            }

            encoder_decode = centroid_accuracy(
                encoder_values, strategy_labels)

            predictor_decode = centroid_accuracy(
                predictor_values, strategy_labels)

            visible_encoder_decode = centroid_accuracy(
                visible_encoder_values,
                visible_encoder_labels
            )

            latent_health = latent_distribution_health(
                latent_distribution_values
            )

            debug_x = torch.cat(visible_encoder_values)
            debug_y = torch.cat(visible_encoder_labels)

            debug_left = debug_x[debug_y == 0]
            debug_right = debug_x[debug_y == 1]

            debug_distance = torch.linalg.vector_norm(
                debug_left.mean(dim=0)-debug_right.mean(dim=0)
            )

            print(
                "strategy_cue_retention",
                "decode=["
                f"cue:{strategy_delay_decode['cue']:.3f},"
                f"d1_4:{strategy_delay_decode['delay_1_4']:.3f},"
                f"d5_8:{strategy_delay_decode['delay_5_8']:.3f},"
                f"d9+:{strategy_delay_decode['delay_9_plus']:.3f}]",
                "snr=["
                f"cue:{strategy_delay_strength['cue'][1]:.3f},"
                f"d1_4:{strategy_delay_strength['delay_1_4'][1]:.3f},"
                f"d5_8:{strategy_delay_strength['delay_5_8'][1]:.3f},"
                f"d9+:{strategy_delay_strength['delay_9_plus'][1]:.3f}]",
            )

            print(
                "cue_strength_debug",
                "samples=", debug_x.shape[0],
                "left=", debug_left.shape[0],
                "right=",debug_right.shape[0],
                "distance=", f"{float(debug_distance):.8e}",
            )

            visible_encoder_distance, visible_encoder_snr = cue_strength(
                visible_encoder_values,
                visible_encoder_labels
            )

            curriculum_rates = env.curriculum_rates
            cue_assignments = env.cue_assignment_counts - reported_cue_assignments
            cue_assignment_count = int(cue_assignments.sum())
            cue_probability_sum = (
                env.cue_assignment_probability_sum
                - reported_cue_probability_sum
            )
            window_cue_probability_left = (
                cue_probability_sum / cue_assignment_count
                if cue_assignment_count
                else env.current_cue_probability_left
            )
            progress = {
                "condition": condition,
                "seed": seed,
                "transitions": completed,
                "fraction": min(completed/max(cfg.transitions, 1), 1.0),
                "episodes": episodes,
                "successes": successes,
                "curriculum_stage": env.curriculum_stage+1,
                "curriculum_stages": len(env.start_rows),
                "cue_assignments_left": int(cue_assignments[0]),
                "cue_assignments_right": int(cue_assignments[1]),
                "cue_probability_left": window_cue_probability_left,
                "cue_probability_left_current": (
                    env.current_cue_probability_left),
                "window_success": (
                    window_successes/max(window_episodes, 1)),
                "window_wrong": window_wrong/max(window_episodes, 1),
                "window_timeout": window_timeouts/max(window_episodes, 1),
                "reward": sums["reward"]/decisions,
                "exploration_rate": sums["exploration_rate"]/decisions,
                "success_ema": (
                    exploration_scheduler.success_ema.tolist()
                    if exploration_scheduler is not None else None),
                "timer_influence": sums["timer_influence"]/decisions,
                "timer_arbitration_stalled": (
                    timer_arbitration.stalled
                    if timer_arbitration is not None else None),
                "timer_rescue_latched": (
                    timer_arbitration.rescue_latched
                    if timer_arbitration is not None else None),
                "timer_controller_state": (
                    getattr(timer_arbitration, "state", None)
                    if timer_arbitration is not None else None),
                "timer_controller_timeout_ema": (
                    getattr(timer_arbitration, "timeout_ema", None)
                    if timer_arbitration is not None else None),
                "timer_arbitration_improvement": (
                    timer_arbitration.last_improvement
                    if timer_arbitration is not None else None),
                "entropy": sums["entropy"]/decisions,
                "timer_entropy": sums["timer_entropy"]/decisions,
                "timer_duration": sums["timer_duration"]/decisions,
                "timer_counts": timer_counts.tolist(),
                "td_abs": sums["td_abs"]/decisions,
                "desirability": sums["desirability"]/decisions,
                "outcome_std": sums["outcome_std"]/decisions,
                "representation_critic_mae": (
                    sums["representation_critic_mae"]/decisions),
                "prediction_mse": sums["prediction_mse"]/decisions,
                "joy_prediction_mse": (
                    sums["joy_prediction_error"]
                    /max(sums["joy_event_count"], 1)),
                "joy_event_count": sums["joy_event_count"],
                "predictor_eligibility": (
                    sums["predictor_eligibility"]/decisions),

                "predictor_eprop_gradient": (
                    sums["predictor_eprop_gradient"]/decisions),

                "predictor_encoder_eligibility": (
                    sums["predictor_encoder_eligibility"] / decisions),

                "predictor_encoder_eprop_gradient": (
                    sums["predictor_encoder_eprop_gradient"] / decisions),

                "reward_latent_shift": (
                    sums["reward_latent_shift"]/decisions),
                "cue_decode": strategy_decode,
                "encoder_cue_decode": encoder_decode,
                "predictor_cue_decode": predictor_decode,
                "visible_encoder_cue_decode": visible_encoder_decode,
                "visible_encoder_cue_distance": visible_encoder_distance,
                "visible_encoder_snr": visible_encoder_snr,

                "latent_effective_rank": (
                    latent_health["effective_rank"]),

                "latent_std_mean": (
                    latent_health["std_mean"]),

                "latent_std_min": (
                    latent_health["std_min"]),

                "latent_std_max": (
                    latent_health["std_max"]),

                "latent_saturation": (
                    latent_health["saturation"]),

                "sigreg_loss": sums["sigreg_loss"] / decisions,

                "latent_correlation_rms": (
                    latent_health["correlation_rms"]
                ),

                "latent_correlation_max": (
                    latent_health["correlation_max"]
                ),

                "latent_top_eigen_fraction": (
                    latent_health["top_eigen_fraction"]
                ),

                "strategy_encoder_trace": (
                    sums["strategy_encoder_trace"]
                    / decisions
                ),

                "strategy_encoder_memory": (
                    sums["strategy_encoder_memory"]
                    / decisions
                ),

                "strategy_encoder_score": (
                    sums["strategy_encoder_score"]
                    / decisions
                ),

                "strategy_encoder_gradient": (
                    sums["strategy_encoder_gradient"]
                    / decisions
                ),

                "strategy_encoder_step": (
                    sums["strategy_encoder_step"]
                    / decisions
                ),

                "strategy_sigreg_loss": (
                    sums["strategy_sigreg_loss"] / decisions
                ),

                "strategy_sigreg_gradient": (
                    sums["strategy_sigreg_gradient"] / decisions
                ),

                "shuffle_tv": sums["shuffle_tv"]/decisions,
                "zero_tv": sums["zero_tv"]/decisions,
                "strategy_stability": (
                    sums["strategy_stability"]/decisions),
                "strategy_gate": sums["strategy_gate"]/decisions,
                "actor_step": sums["actor_step"]/decisions,
                "encoder_step": sums["encoder_step"]/decisions,
                "critic_encoder_step": (
                    sums["critic_encoder_step"]/decisions),
                "strategy_step": sums["strategy_step"]/decisions,

            }
            progress["curriculum_blend_fraction"]=env.curriculum_blend_fraction
            progress["current_length_distribution"]=env.current_length_distribution
            progress.update(env.summarize_hallways(window_hallways,
                cfg.curriculum_min_episodes_per_length,env.curriculum_length_distributions))
            if progress_callback is not None:
                progress_callback(progress)
            print(
                f"condition={condition:21s} seed={seed} transitions={completed} "
                f"episodes={episodes} successes={successes} "
                f"curriculum={env.curriculum_stage+1}/{len(env.start_rows)} "
                f"length_distribution={env.current_length_distribution} blend={env.curriculum_blend_fraction:.2f} "
                f"stage_limit={env.episode_limits[env.curriculum_stage]} "
                f"cue_success=[{curriculum_rates[0]:.2f},"
                f"{curriculum_rates[1]:.2f}] "
                f"cue_assignments=[{cue_assignments[0]},"
                f"{cue_assignments[1]}] "
                f"cue_p_left={window_cue_probability_left:.3f} "
                f"window_rate={window_successes/max(window_episodes,1):.3f} "
                f"wrong_rate={window_wrong/max(window_episodes,1):.3f} "
                f"timeout_rate={window_timeouts/max(window_episodes,1):.3f} "
                f"reward={sums['reward']/decisions:+.4f} "
                f"explore={sums['exploration_rate']/decisions:.3f} "
                +(
                    "success_ema=["
                    f"{exploration_scheduler.success_ema[0]:.3f},"
                    f"{exploration_scheduler.success_ema[1]:.3f}] "
                    if exploration_scheduler is not None else ""
                )+
                (
                    f"timer_weight={sums['timer_influence']/decisions:.3f} "
                    f"timer_stalled={timer_arbitration.stalled} "
                    f"timer_latched={timer_arbitration.rescue_latched} "
                    f"timer_state={getattr(timer_arbitration, 'state', 'legacy')} "
                    f"timer_progress={timer_arbitration.last_improvement:+.3f} "
                    if timer_arbitration is not None else ""
                )+
                f"entropy={sums['entropy']/decisions:.3f} "
                f"timer=[mean:{sums['timer_duration']/decisions:.2f},"
                f"entropy:{sums['timer_entropy']/decisions:.3f},"
                f"counts:{timer_counts.tolist()}] "
                f"td={sums['td_abs']/decisions:.3f} "
                f"desire={sums['desirability']/decisions:+.3f} "
                f"outcome_std={sums['outcome_std']/decisions:.3f} "
                f"outcome_nll={sums['outcome_nll']/decisions:+.4f} "
                f"latent_q_mae={sums['representation_critic_mae']/decisions:.3f} "
                f"terminal_mae={sums['terminal_outcome_abs_error']/max(sums['terminal_outcome_count'],1):.3f} "
                f"terminal_desire=[win:{sums['success_desirability']/max(sums['success_count'],1):+.3f},"
                f"wrong:{sums['wrong_desirability']/max(sums['wrong_count'],1):+.3f},"
                f"timeout:{sums['timeout_desirability']/max(sums['timeout_count'],1):+.3f}] "
                f"pred_mse={sums['prediction_mse']/decisions:.4f} "
                f"joy_mse={sums['joy_prediction_error']/max(sums['joy_event_count'],1):.4f} "
                f"joy_shift={sums['reward_latent_shift']/decisions:.4f} "

                f"predictor_eprop=[elig:"
                f"{sums['predictor_eligibility']/decisions:.2f},"
                f"grad:"
                f"{sums['predictor_eprop_gradient']/decisions:.3f}] "

                f"predictor_encoder_eprop=[elig:"
                f"{sums['predictor_encoder_eligibility']/decisions:.2f},"
                f"grad:"
                f"{sums['predictor_encoder_eprop_gradient']/decisions:.3f}] "

                f"sigreg={sums['sigreg_loss']/decisions:.4f} "

                f"latent_health=["
                f"rank:{latent_health['effective_rank']:.2f}/"
                f"{cfg.latent_dim},"
                f"std:{latent_health['std_mean']:.3f}/"
                f"{latent_health['std_min']:.3f}/"
                f"{latent_health['std_max']:.3f},"
                f"sat:{latent_health['saturation']:.3f},"
                f"corr_rms:{latent_health['correlation_rms']:.3f},"
                f"corr_max:{latent_health['correlation_max']:.3f},"
                f"top1:{latent_health['top_eigen_fraction']:.3f}] "


                f"strategy_encoder_eprop=["
                f"elig:{sums['strategy_encoder_trace']/decisions:.2f},"
                f"memory:{sums['strategy_encoder_memory']/decisions:.2f},"
                f"score:{sums['strategy_encoder_score']/decisions:.2f},"
                f"grad:{sums['strategy_encoder_gradient']/decisions:.3f},"
                f"step:{sums['strategy_encoder_step']/decisions:.3e}] "



                f"steps=[actor:{sums['actor_step']/decisions:.5f},"
                f"encoder:{sums['encoder_step']/decisions:.5f},"
                f"strategy_encoder:"
                f"{sums['strategy_encoder_step']/decisions:.3e},"
                f"critic_encoder:"
                f"{sums['critic_encoder_step']/decisions:.5f},"
                f"strategy:{sums['strategy_step']/decisions:.5f}]"


                f"encoder_cue={sums['encoder_cue_accuracy']/decisions:.3f} "
                f"visible_cue={sums['visible_cue_correct']/max(sums['visible_cue_count'],1):.3f} "
                f"cue_decode=[encoder:{encoder_decode:.3f},"
                f"visible_encoder_decode={visible_encoder_decode:.3f}",
                f"visible_encoder_strength=[",
                f"distance:{visible_encoder_distance:.3e},",
                f"snr:{visible_encoder_snr:.3e}]",
                f"latent_health=["
                f"rank:{latent_health['effective_rank']:.2f}/"
                f"{cfg.latent_dim},"
                f"std:{latent_health['std_mean']:.3f}/"
                f"{latent_health['std_min']:.3f}/"
                f"{latent_health['std_max']:.3f},"
                f"sat:{latent_health['saturation']:.3f}] "
                f"predictor:{predictor_decode:.3f},"
                f"strategy:{strategy_decode:.3f}] "
                f"strategy_tv=[shuffle:{sums['shuffle_tv']/decisions:.4f},"
                f"zero:{sums['zero_tv']/decisions:.4f}] "
                f"stability={sums['strategy_stability']/decisions:.3f} "
                f"gate={sums['strategy_gate']/decisions:.3f} "
                f"gate_sat={sums['strategy_gate_saturation']/decisions:.3f} "
                f"eligibility=[rec:{sums['strategy_recurrent_e']/decisions:.2f},"
                f"memory:{sums['strategy_memory_e']/decisions:.2f},"
                f"score:{sums['strategy_score_e']/decisions:.2f}] "
                f"steps=[actor:{sums['actor_step']/decisions:.5f},"
                f"encoder:{sums['encoder_step']/decisions:.5f},"
                f"critic_encoder:{sums['critic_encoder_step']/decisions:.5f},"
                f"strategy:{sums['strategy_step']/decisions:.5f}]")
            window_steps = window_episodes = window_successes = window_wrong = 0
            window_hallways.clear()
            window_timeouts = 0
            timer_counts.fill(0)
            reported_cue_assignments = env.cue_assignment_counts.copy()
            reported_cue_probability_sum = env.cue_assignment_probability_sum
            sums.clear()
            strategy_values.clear()
            encoder_values.clear()
            predictor_values.clear()
            strategy_labels.clear()
            visible_encoder_values.clear()
            visible_encoder_labels.clear()
            latent_distribution_values.clear()

            for values in strategy_delay_values.values():
                values.clear()

            for labels in strategy_delay_labels.values():
                labels.clear()

    if timer_arbitration is not None:
        system.timer_influence = timer_arbitration.gate
    result = {"condition": condition, "seed": seed,
              "episodes": episodes, "successes": successes,
              "rate": successes/max(episodes, 1),
              "wrong": wrong_total/max(episodes, 1),
              "timeout": timeout_total/max(episodes, 1),
              "final_exploration_rate": (
                  exploration_scheduler.rate
                  if exploration_scheduler is not None
                  else cfg.exploration_rate),
              "final_success_ema": (
                  exploration_scheduler.success_ema.tolist()
                  if exploration_scheduler is not None else None),
              "final_timer_influence": system.timer_influence,
              "timer_arbitration_stalled": (
                  timer_arbitration.stalled
                  if timer_arbitration is not None else None),
              "timer_rescue_latched": (
                  timer_arbitration.rescue_latched
                  if timer_arbitration is not None else None),
              "timer_controller_state": (
                  getattr(timer_arbitration, "state", None)
                  if timer_arbitration is not None else None),
              "timer_controller_timeout_ema": (
                  getattr(timer_arbitration, "timeout_ema", None)
                  if timer_arbitration is not None else None),
              "timer_arbitration_improvement": (
                  timer_arbitration.last_improvement
                  if timer_arbitration is not None else None),
              "system": system}
    if diagnostics is not None:
        diagnostics.save(cfg.critic_diagnostics_dir)
    if capture is not None:
        capture.flush()
    return result


@torch.no_grad()
def evaluate(system: System, cfg: Config, seed: int, intervention: str):
    # Curriculum is a training aid only.  Every intervention is evaluated
    # from the final, longest-biased hallway distribution.
    env = BatchedTMaze(cfg, seed+80_000, curriculum=False)
    system.strategizer.core.initial(cfg.worlds, system.device)
    system.predictor.core.initial(cfg.worlds, system.device)
    system.actor.core.initial(cfg.worlds, system.device)
    system.feedback.zero_()
    system.strategy_memory.zero_(); system.desirability_memory.zero_()
    system.outcome_logvar_memory.zero_()
    timed_predictions = (
        TimedPredictionBuffer(cfg, system.device)
        if cfg.use_strategic_prediction_timer else None
    )
    episodes = successes = wrong = 0

    while episodes < cfg.evaluation_episodes:
        observation = torch.tensor(env.observation(), device=system.device)
        latent, _, _ = system.encoder(observation)
        strategy = system.strategizer(
            latent.detach(), system.feedback.detach(), deterministic=False,
            timer_influence=system.timer_credit_influence,
            previous_strategy=(system.strategy_memory
                if cfg.learned_strategy_memory else None))
        if system.condition == "actor_only":
            actor_strategy = torch.zeros_like(strategy["strategy"])
            actor_desirability = torch.zeros_like(strategy["desirability"])
            actor_outcome_logvar = torch.zeros_like(
                strategy["outcome_logvar"])
        elif system.condition == "separated":
            if cfg.learned_strategy_memory:
                actor_strategy = strategy["strategy"]
                actor_desirability = strategy["desirability"]
                actor_outcome_logvar = strategy["outcome_logvar"]
            else:
                keep = cfg.strategy_retention
                actor_strategy = (keep*system.strategy_memory
                                  +(1-keep)*strategy["strategy"])
                actor_desirability = (keep*system.desirability_memory
                                      +(1-keep)*strategy["desirability"])
                actor_outcome_logvar = (
                    keep*system.outcome_logvar_memory
                    +(1-keep)*strategy["outcome_logvar"])
            system.strategy_memory = actor_strategy
            system.desirability_memory = actor_desirability
            system.outcome_logvar_memory = actor_outcome_logvar
        else:
            actor_strategy = strategy["strategy"]
            actor_desirability = strategy["desirability"]
            actor_outcome_logvar = strategy["outcome_logvar"]
        if intervention == "shuffle":
            permutation = torch.roll(torch.arange(cfg.worlds,
                                                   device=system.device), 1)
            actor_strategy = actor_strategy[permutation]
            actor_desirability = actor_desirability[permutation]
            actor_outcome_logvar = actor_outcome_logvar[permutation]
        elif intervention == "zero":
            actor_strategy = torch.zeros_like(actor_strategy)
            actor_desirability = torch.zeros_like(actor_desirability)
            actor_outcome_logvar = torch.zeros_like(actor_outcome_logvar)
        # Evaluate the learned stochastic policy without the forced training
        # exploration mixture.  This distinguishes learned competence from
        # accidental epsilon-driven terminal visits.
        actor = system.actor(
            latent, actor_strategy, actor_desirability,
            actor_outcome_logvar, deterministic=False, exploration=0.0)
        next_np, reward_np, done_np, success_np, wrong_np, _, _ = env.step(
            actor["action"].cpu().numpy())
        next_observation = torch.tensor(next_np, device=system.device)
        next_latent, _, _ = system.encoder(next_observation)
        done = torch.tensor(
            done_np, device=system.device, dtype=torch.bool)
        reward = torch.tensor(reward_np, device=system.device)

        if timed_predictions is not None:
            prediction_observation = torch.tensor(
                env.transition_observation, device=system.device)
            target_encoder = (
                system.target_encoder
                if system.target_encoder is not None else system.encoder
            )
            target_latent = target_encoder.encode(
                prediction_observation, reward)
            predictor_feedback = timed_predictor_update(
                system,
                timed_predictions,
                latent,
                actor_strategy,
                actor_desirability,
                actor["action"],
                strategy["timer_index"],
                strategy["timer_duration"],
                target_latent,
                reward,
                done,
                train=False,
            )[0]
            predictor_feedback = system.timer_influence*predictor_feedback
        else:
            predicted_next_latent = system.predictor(
                latent,
                actor_strategy,
                actor_desirability,
                actor["action"],
            )
            predicted_change = predicted_next_latent-latent
            prediction_error = next_latent-predicted_next_latent
            predictor_feedback = torch.cat(
                (predicted_change, prediction_error), dim=-1)
        if intervention == "predictor_shuffle":
            permutation = torch.roll(torch.arange(
                cfg.worlds, device=system.device), 1)
            predictor_feedback = predictor_feedback[permutation]
        elif intervention == "predictor_zero":
            predictor_feedback = torch.zeros_like(predictor_feedback)
        system.feedback = predictor_feedback


        system.reset(done)
        episodes += int(done_np.sum()); successes += int(success_np.sum())
        wrong += int(wrong_np.sum())
    return successes/max(episodes, 1), wrong/max(episodes, 1)


def save(path: Path, cfg: Config, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 7,
               "architecture": ("learned_gated_strategy_memory_eprop"
                    if cfg.learned_strategy_memory
                    else "fixed_leaky_strategy_memory_eprop"),
               "encoder_learning": cfg.encoder_learning_mode,
               "config": asdict(cfg), "conditions": {}}
    for result in results:
        system = result["system"]
        state = {
            name: getattr(system, name).state_dict() for name in
            ("encoder", "strategizer", "actor", "predictor")}
        state["timer_influence"] = system.timer_influence
        if system.target_encoder is not None:
            state["target_encoder"] = system.target_encoder.state_dict()
        if system.representation_critic is not None:
            state["representation_critic"] = (
                system.representation_critic.state_dict())
            state["target_representation_critic"] = (
                system.target_representation_critic.state_dict())
        payload["conditions"][
            f"{result['condition']}_seed{result['seed']}"] = state
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(payload, temporary); temporary.replace(path)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    mo.md(r"""
    # Stateful strategizer · delayed-cue T-maze

    Configure and compare the separated architecture and its controls.
    Settings are committed only when **Train and evaluate** is clicked, so
    changing a widget never starts an expensive run.
    """)
    return (mo,)


@app.cell
def _():
    # Keep the learning code importable while exposing one explicit dependency
    # to marimo's reactive graph.
    terminal_api = {
        "Config": Config,
        "conditions": CONDITIONS,
        "choose_device": choose_device,
        "environment": BatchedTMaze,
        "run_condition": run_condition,
        "evaluate": evaluate,
        "save": save,
    }
    return (terminal_api,)


@app.cell(hide_code=True)
def _(mo, terminal_api):
    defaults = terminal_api["Config"]()

    def number(value, label, *, step=None, start=None, stop=None):
        return mo.ui.number(
            value=value, label=label, step=step, start=start, stop=stop,
            full_width=True,
        )

    run_controls = mo.ui.dictionary({
        "seed": number(defaults.seed, "Random seed", step=1, start=0),
        "device": mo.ui.dropdown(
            ["auto", "cpu", "mps", "cuda"], value=defaults.device,
            label="Device", full_width=True),
        "worlds": number(
            defaults.worlds, "Parallel worlds", step=1, start=1),
        "transitions": number(
            defaults.transitions, "Training transitions", step=1, start=1),
        "report_every": number(
            defaults.report_every, "Log every N transitions",
            step=1, start=1),
        "seeds": number(3, "Independent seeds", step=1, start=1),
        "conditions": mo.ui.multiselect(
            options=list(terminal_api["conditions"]), value=["separated"],
            label="Architectures", full_width=True),
        "evaluation_episodes": number(
            defaults.evaluation_episodes, "Evaluation episodes",
            step=1, start=1),
        "checkpoint": mo.ui.text(
            value="", label="Checkpoint filename (blank selects a default)",
            full_width=True),
        "save_checkpoint": mo.ui.switch(
            value=True, label="Save checkpoint"),
    }, label="Run")

    curriculum_controls = mo.ui.dictionary({
        "curriculum_success_threshold": number(
            defaults.curriculum_success_threshold,
            "Advancement success threshold", step=0.01,
            start=0.0, stop=1.0),
        "curriculum_min_episodes_per_cue": number(
            defaults.curriculum_min_episodes_per_cue,
            "Minimum episodes per cue", step=1, start=1),
        "curriculum_history_per_cue": number(
            defaults.curriculum_history_per_cue,
            "History length per cue", step=1, start=1),
    }, label="Curriculum")

    architecture_controls = mo.ui.dictionary({
        "learned_strategy_memory": mo.ui.switch(
            value=defaults.learned_strategy_memory,
            label="Learned gated strategy memory"),
        "latent_dim": number(
            defaults.latent_dim, "Latent dimension", step=1, start=1),
        "strategy_dim": number(
            defaults.strategy_dim, "Strategy dimension", step=1, start=1),
        "hidden_dim": number(
            defaults.hidden_dim, "Hidden dimension", step=1, start=1),
        "conditioning_dim": number(
            defaults.conditioning_dim, "Conditioning dimension",
            step=1, start=1),
        "snn_ticks": number(
            defaults.snn_ticks, "SNN ticks per decision", step=1, start=1),
        "strategy_retention": number(
            defaults.strategy_retention, "Fixed-memory retention",
            step=0.01, start=0.0, stop=1.0),
    }, label="Architecture")

    learning_controls = mo.ui.dictionary({
        "gamma": number(
            defaults.gamma, "Discount factor",
            step=0.001, start=0.0, stop=1.0),
        "exploration_rate": number(
            defaults.exploration_rate, "Training exploration",
            step=0.01, start=0.0, stop=1.0),
        "encoder_lr": number(
            defaults.encoder_lr, "Encoder learning rate",
            step=1e-5, start=0.0),
        "predictor_lr": number(
            defaults.predictor_lr, "Predictor learning rate",
            step=1e-5, start=0.0),
        "actor_eprop_lr": number(
            defaults.actor_eprop_lr, "Actor e-prop learning rate",
            step=1e-5, start=0.0),
        "strategy_eprop_lr": number(
            defaults.strategy_eprop_lr,
            "Strategizer e-prop learning rate", step=1e-5, start=0.0),
        "critic_lr": number(
            defaults.critic_lr, "Outcome-head learning rate",
            step=1e-5, start=0.0),
        "actor_trace_decay": number(
            defaults.actor_trace_decay, "Actor trace decay",
            step=0.01, start=0.0, stop=1.0),
        "strategy_trace_decay": number(
            defaults.strategy_trace_decay, "Strategy trace decay",
            step=0.01, start=0.0, stop=1.0),
    }, label="Learning")

    experiment_form = mo.ui.dictionary({
        "run": run_controls,
        "curriculum": curriculum_controls,
        "architecture": architecture_controls,
        "learning": learning_controls,
    }, label="Experiment configuration").form(
        label="Delayed-cue T-maze experiment",
        submit_button_label="Train and evaluate",
        submit_button_tooltip="Apply these values and start a fresh run",
        bordered=True,
    )
    mo.vstack([
        mo.md("## Experiment controls"),
        mo.md(
            "A full default run is compute-intensive. For a quick smoke test, "
            "reduce transitions, evaluation episodes, and independent seeds."),
        experiment_form,
    ])
    return (experiment_form,)


@app.cell
def _(experiment_form, mo, terminal_api):
    mo.stop(
        experiment_form.value is None,
        mo.md("Submit the form above when you are ready to run."),
    )

    submitted = experiment_form.value
    run_values = dict(submitted["run"])
    cfg_values = {
        **dict(submitted["curriculum"]),
        **dict(submitted["architecture"]),
        **dict(submitted["learning"]),
    }
    for field in (
        "seed", "worlds", "transitions", "report_every",
        "evaluation_episodes",
    ):
        run_values[field] = int(run_values[field])
    for field in (
        "latent_dim", "strategy_dim", "hidden_dim", "conditioning_dim",
        "snn_ticks", "curriculum_min_episodes_per_cue",
        "curriculum_history_per_cue",
    ):
        cfg_values[field] = int(cfg_values[field])

    conditions = list(run_values.pop("conditions"))
    seed_count = int(run_values.pop("seeds"))
    save_checkpoint = bool(run_values.pop("save_checkpoint"))
    requested_checkpoint = run_values.pop("checkpoint").strip()
    cfg_values.update(run_values)

    mo.stop(
        not conditions,
        mo.md("Select at least one architecture before starting."),
    )
    mo.stop(
        cfg_values["curriculum_history_per_cue"]
        < cfg_values["curriculum_min_episodes_per_cue"],
        mo.md(
            "Curriculum history must be at least as large as the minimum "
            "episodes per cue."),
    )

    if requested_checkpoint:
        cfg_values["checkpoint"] = requested_checkpoint
    elif cfg_values["learned_strategy_memory"]:
        cfg_values["checkpoint"] = (
            "online_delayed_cue_strategy_tmaze_gated_memory_v8.pt")
    else:
        cfg_values["checkpoint"] = (
            "online_delayed_cue_strategy_tmaze_recurrent_eprop_v7.pt")

    cfg = terminal_api["Config"](**cfg_values)
    device = terminal_api["choose_device"](cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    preview_env = terminal_api["environment"](
        cfg, cfg.seed, curriculum=True)

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        print(f"device={device} task=DELAYED_CUE_TMAZE "
              f"reward=SPARSE_SIGNED_TERMINAL(+1/-1/timeout"
              f"{cfg.timeout_penalty:+.1f}) worlds={cfg.worlds} "
              f"episode_limit={cfg.episode_limit}")
        print(f"length_curriculum={preview_env.curriculum_length_distributions} "
              f"stage_limits={preview_env.episode_limits} "
              f"advance=min_cue_success>="
              f"{cfg.curriculum_success_threshold:.2f} "
              f"after>={cfg.curriculum_min_episodes_per_cue}"
              f"_episodes_per_cue "
              f"evaluation_lengths={preview_env.curriculum_length_distributions[-1]}")
        print(f"encoder=STATELESS_SNN predictor=STATEFUL_SNN "
              f"strategizer=STATEFUL_EXCEPT_CONTROL "
              f"actor=STATELESS_EXCEPT_CONTROL ticks={cfg.snn_ticks} "
              f"temporal_bptt=False")
        print("strategy_memory="+(
            "LEARNED_ELEMENTWISE_GATE+DIRECT_ACTOR_INPUT"
            if cfg.learned_strategy_memory else "FIXED_LEAKY_GATE"))

        results = []
        for seed in range(seed_count):
            for condition in conditions:
                result = terminal_api["run_condition"](
                    cfg, condition, seed, device)
                system = result["system"]
                real = terminal_api["evaluate"](
                    system, cfg, seed, "real")
                shuffled = terminal_api["evaluate"](
                    system, cfg, seed, "shuffle")
                zero = terminal_api["evaluate"](
                    system, cfg, seed, "zero")
                result.update(
                    eval_real=real[0], eval_shuffle=shuffled[0],
                    eval_zero=zero[0], eval_wrong=real[1])
                print(
                    f"evaluation condition={condition} seed={seed} "
                    f"train={result['successes']}/{result['episodes']} "
                    f"real={real[0]:.3f} shuffled={shuffled[0]:.3f} "
                    f"zero={zero[0]:.3f} wrong={real[1]:.3f}")
                results.append(result)

        aggregate_rows = []
        print("aggregate")
        for condition in conditions:
            rows = [
                row for row in results
                if row["condition"] == condition
            ]
            aggregate = {
                "condition": condition,
                "train_rate": float(np.mean(
                    [row["rate"] for row in rows])),
                "eval_real": float(np.mean(
                    [row["eval_real"] for row in rows])),
                "eval_shuffle": float(np.mean(
                    [row["eval_shuffle"] for row in rows])),
                "eval_zero": float(np.mean(
                    [row["eval_zero"] for row in rows])),
            }
            aggregate_rows.append(aggregate)
            print(
                f"  {condition:21s} "
                f"train_rate={aggregate['train_rate']:.3f} "
                f"eval_real={aggregate['eval_real']:.3f} "
                f"eval_shuffle={aggregate['eval_shuffle']:.3f} "
                f"eval_zero={aggregate['eval_zero']:.3f}")

        checkpoint_path = Path(__file__).resolve().parent / cfg.checkpoint
        if save_checkpoint:
            terminal_api["save"](checkpoint_path, cfg, results)
            print(f"saved_checkpoint={checkpoint_path}")
        else:
            checkpoint_path = None
            print("checkpoint_not_saved")

    result_rows = [{
        "condition": row["condition"],
        "seed": row["seed"],
        "episodes": row["episodes"],
        "train rate": round(row["rate"], 4),
        "wrong rate": round(row["wrong"], 4),
        "timeout rate": round(row["timeout"], 4),
        "eval real": round(row["eval_real"], 4),
        "eval shuffled": round(row["eval_shuffle"], 4),
        "eval zero": round(row["eval_zero"], 4),
    } for row in results]
    aggregate_display = [{
        "condition": row["condition"],
        "train rate": round(row["train_rate"], 4),
        "eval real": round(row["eval_real"], 4),
        "eval shuffled": round(row["eval_shuffle"], 4),
        "eval zero": round(row["eval_zero"], 4),
        "shuffle drop": round(
            row["eval_real"] - row["eval_shuffle"], 4),
        "zero drop": round(
            row["eval_real"] - row["eval_zero"], 4),
    } for row in aggregate_rows]
    run_log = log_buffer.getvalue()
    return aggregate_display, checkpoint_path, result_rows, run_log


@app.cell(hide_code=True)
def _(
    aggregate_display,
    checkpoint_path,
    mo,
    result_rows,
    run_log,
    ):
    checkpoint_message = (
        f"Checkpoint saved to `{checkpoint_path}`"
        if checkpoint_path is not None
        else "Checkpoint saving was disabled."
    )
    mo.vstack([
        mo.md("## Results"),
        mo.md(checkpoint_message),
        mo.md("### Aggregate comparison"),
        mo.ui.table(aggregate_display),
        mo.md(
            "Shuffle and zero drops measure how much evaluation performance "
            "depends on the learned strategy signal."),
        mo.md("### Per-seed results"),
        mo.ui.table(result_rows),
        mo.md("### Training log"),
        mo.md(f"```text\n{run_log}\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
