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
* Stateful spiking predictor: one-step Gaussian prediction of a stop-gradient
  EMA target-encoder latent.  Reward is revealed only to the target, so its
  recurrent state must retain earlier evidence to predict the resulting
  affective modulation.  Online LIF eligibility carries prediction gradients
  across otherwise detached environment decisions.
* Stateful spiking strategizer: predictor-feedback conditioned, deterministic
  strategy readout plus independent context and augmented outcome critics.
  The causal strategy-only critic conditions the actor; the full strategy/
  actor-feature critic supplies the TD baseline.
  Bellec-style recurrent LIF eligibility and exact leaky-strategy-memory
  Jacobians carry causal derivatives online; sparse TD error modulates their
  reward trace using Adam.  Outcome statistics use distributional TD.
* Optional strategy-conditioned representation critic: distributional TD on
  ``Q(online_latent, detached_strategy)`` with EMA targets and a scaled direct
  gradient into the online encoder.
* Stateless spiking actor: current latent input plus strategy, desirability,
  and uncertainty conditioning, followed by one immediate action.  Its
  normalized recurrent feature is supplied, stop-gradient, to the full
  outcome critic so value estimation can assess how it interpreted the
  strategy.  Its reward-modulated score eligibility can also update the
  encoder directly.  Neural state resets each environment decision, while
  eligibility does not.

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

import marimo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__generated_with = "0.23.15"
app = marimo.App(width="full")


ACTIONS = ("left", "right", "forward")
ACTION_DIM = len(ACTIONS)
CONDITIONS = ("separated", "stateless_strategizer", "actor_only")


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
    encoder_persistent: bool = True
    encoder_membrane_decay: float = 0.97
    train_encoder_core_eprop: bool = True
    encoder_core_trace_decay: float = 0.995
    encoder_core_eprop_lr: float = 3e-6
    predictor_membrane_decay: float = 0.97
    strategy_membrane_decay: float = 0.98
    surrogate_scale: float = 0.30
    gamma: float = 0.99
    actor_trace_decay: float = 0.85
    strategy_trace_decay: float = 0.99
    encoder_trace_decay: float = 0.99
    predictor_trace_decay: float = 0.995
    strategy_retention: float = 0.95
    learned_strategy_memory: bool = True
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

    #Vision Configuration
    visual_observations: bool = True
    image_size: int = 40
    image_channels: int = 3
    visual_feature_dim: int = 64
    train_visual_projection: bool = True
    visual_projection_eprop_lr: float = 1e-6


    use_reward_adaln: bool = False
    reward_adaln_strength: float = 0.25
    use_actor_encoder_eprop: bool = False
    use_representation_critic: bool = False
    representation_critic_lr: float = 3e-4
    representation_critic_target_tau: float = 0.005
    critic_encoder_weight: float = 0.0
    predictor_lr: float = 3e-4
    use_predictor_eprop: bool = True
    use_predictor_encoder_eprop: bool = False

    # Predictor gradients never backpropagate directly into the encoder.
    # Instead, prediction loss supplies a strategy-space learning signal that
    # is contracted with the strategizer and encoder temporal Jacobians.
    predictor_strategy_weight: float = 1e-2
    predictor_mediated_encoder_weight: float = 1e-3
    align_predictor_with_task_gradient: bool = True

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
    curriculum_episode_limits: Tuple[int, ...] = (16, 24, 36, 48)
    encoder_learning_mode: str = "reward_eprop"
    cue_aux_weight: float = 2.0
    exploration_rate: float = 0.10
    evaluation_episodes: int = 192
    checkpoint: str = "online_delayed_cue_visual_tmaze.pt"


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
        requested_rows = (1, 3, 5, full_start)
        requested_limits = cfg.curriculum_episode_limits
        if len(requested_rows) != len(requested_limits):
            raise ValueError("curriculum rows and episode limits must align")
        # Keep this robust to smaller custom mazes by dropping out-of-range
        # stages and merging any duplicate final row into the harder limit.
        curriculum_pairs = []
        for row, limit in zip(requested_rows, requested_limits):
            if row > full_start:
                continue
            if curriculum_pairs and curriculum_pairs[-1][0] == row:
                curriculum_pairs[-1] = (row, limit)
            else:
                curriculum_pairs.append((row, limit))
        self.start_rows = tuple(row for row, _ in curriculum_pairs)
        self.episode_limits = tuple(limit for _, limit in curriculum_pairs)
        self.curriculum_enabled = curriculum
        self.curriculum_stage = 0 if curriculum else len(self.start_rows)-1
        self.curriculum_history = (
            deque(maxlen=cfg.curriculum_history_per_cue),
            deque(maxlen=cfg.curriculum_history_per_cue))
        b = cfg.worlds
        self.x = np.zeros(b, np.int64)
        self.y = np.zeros(b, np.int64)
        self.direction = np.zeros(b, np.int64)
        self.cue = np.zeros(b, np.int64)
        self.age = np.zeros(b, np.int64)
        self.previous_action = np.full(b, 2, np.int64)
        self.episode = np.zeros(b, np.int64)
        self.reset(np.ones(b, bool))

    def reset(self, mask: np.ndarray) -> None:
        count = int(mask.sum())
        if not count:
            return
        self.x[mask] = self.center
        self.y[mask] = self.start_rows[self.curriculum_stage]
        self.direction[mask] = 0
        self.cue[mask] = self.rng.integers(0, 2, count)
        self.age[mask] = 0
        self.previous_action[mask] = 2
        self.episode[mask] += 1

    def _wall(self, world: int, relative_turn: int) -> float:
        direction = (int(self.direction[world])+relative_turn) % 4
        dx, dy = self.DELTAS[direction]
        target = (int(self.x[world]+dx), int(self.y[world]+dy))
        return float(target not in self.valid)

    def _vector_observation(self) -> np.ndarray:
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

    def _visual_observation(self) -> np.ndarray:
        size = self.cfg.image_size
        channels = self.cfg.image_channels
        batch = self.cfg.worlds

        if channels != 3:
            raise ValueError(
                "The initial visual renderer expects three channels"
            )

        images = np.zeros(
            (batch, channels, size, size),
            dtype=np.float32,
        )

        # Reserve a narrow strip at the top for the transient cue
        # and previous-action indicators.
        panel_height = 5
        available_height = size - panel_height - 2

        cell_size = max(
            1,
            min(
                (size - 2) // self.cfg.maze_width,
                available_height // self.cfg.maze_height,
            ),
        )

        map_width = cell_size * self.cfg.maze_width
        map_height = cell_size * self.cfg.maze_height

        offset_x = (size - map_width) // 2
        offset_y = (
            panel_height
            + (available_height - map_height) // 2
        )

        # Draw all traversable maze cells.
        for x, y in self.valid:
            x0 = offset_x + x * cell_size
            y0 = offset_y + y * cell_size

            images[
                :,
                :,
                y0:y0 + cell_size,
                x0:x0 + cell_size,
            ] = 0.20

        # Both goals look identical. The cue determines which is correct.
        for goal_x, goal_y in (
            self.left_goal,
            self.right_goal,
        ):
            x0 = offset_x + goal_x * cell_size
            y0 = offset_y + goal_y * cell_size

            images[
                :,
                :,
                y0:y0 + cell_size,
                x0:x0 + cell_size,
            ] = 0.45

        action_panel_start = (
            size - (ACTION_DIM * 3 + 1)
        )

        for world in range(batch):
            # Show the cue only during the original cue interval.
            if self.age[world] < self.cfg.cue_steps:
                cue_channel = int(self.cue[world])

                images[
                    world,
                    cue_channel,
                    1:4,
                    1:4,
                ] = 1.0

            # Render the previous action as one of three generic lights.
            previous_action = int(
                self.previous_action[world]
            )

            action_x = (
                action_panel_start
                + 3 * previous_action
            )

            images[
                world,
                :,
                1:4,
                action_x:action_x + 2,
            ] = 1.0

            # Agent position.
            agent_x = (
                offset_x
                + int(self.x[world]) * cell_size
                + cell_size // 2
            )

            agent_y = (
                offset_y
                + int(self.y[world]) * cell_size
                + cell_size // 2
            )

            images[
                world,
                :,
                agent_y,
                agent_x,
            ] = 1.0

            # A blue heading pixel shows orientation.
            direction = int(self.direction[world])
            dx, dy = self.DELTAS[direction]

            heading_x = int(
                np.clip(agent_x + dx, 0, size - 1)
            )

            heading_y = int(
                np.clip(agent_y + dy, 0, size - 1)
            )

            images[
                world,
                2,
                heading_y,
                heading_x,
            ] = 1.0

        return images

    def observation(self) -> np.ndarray:
        if self.cfg.visual_observations:
            return self._visual_observation()

        return self._vector_observation()

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
        timeout = self.age >= self.episode_limits[self.curriculum_stage]
        pure_timeout = timeout & ~successes & ~wrong
        rewards[pure_timeout] = self.cfg.timeout_penalty
        done = successes | wrong | timeout
        # Preserve the actual post-action observation for world-model
        # learning.  The ordinary return value remains the automatically reset
        # observation consumed by the policy on the following decision.
        self.transition_observation = self.observation()
        cue = self.cue.copy()
        terminal_age = self.age.copy()
        if self.curriculum_enabled:
            for world in np.flatnonzero(done):
                self.curriculum_history[int(cue[world])].append(
                    float(successes[world]))
            enough_evidence = all(
                len(history) >= self.cfg.curriculum_min_episodes_per_cue
                for history in self.curriculum_history)
            mastered = enough_evidence and all(
                float(np.mean(history)) >=
                self.cfg.curriculum_success_threshold
                for history in self.curriculum_history)
            if mastered and self.curriculum_stage < len(self.start_rows)-1:
                self.curriculum_stage += 1
                for history in self.curriculum_history:
                    history.clear()
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

    def forward(
            self,
            value: torch.Tensor,
            update_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
        if (not self.persistent or self.mem is None
                or self.mem.shape[0] != value.shape[0]):
            previous_mem = torch.zeros(
                value.shape[0], self.hidden_dim,
                device=value.device, dtype=value.dtype)
            previous_spk = torch.zeros_like(previous_mem)
        else:
            previous_mem = self.mem.detach()
            previous_spk = self.spk.detach()
        mem, spk = previous_mem, previous_spk
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
            committed_mem, committed_spk = mem.detach(), spk.detach()
            if update_mask is not None:
                mask = update_mask.to(
                    device=value.device, dtype=torch.bool)[:, None]
                committed_mem = torch.where(
                    mask, committed_mem, previous_mem)
                committed_spk = torch.where(
                    mask, committed_spk, previous_spk)
            self.mem, self.spk = committed_mem, committed_spk
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


class VisualStem(nn.Module):
    """Convert an RGB observation into a small fixed-size feature vector."""

    def __init__(
            self,
            image_channels: int,
            output_dim: int,
        ):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv2d(
                image_channels,
                16,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.GroupNorm(4, 16),
            nn.GELU(),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.projection = nn.Linear(
            32 * 4 * 4,
            output_dim,
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        features = self.convolution(observation)
        features = features.flatten(start_dim=1)
        features = self.projection(features)
        features = self.output_norm(features)

        return torch.tanh(features)


class StatelessEncoder(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__(); self.cfg = cfg

        if cfg.visual_observations:
            self.visual_stem = VisualStem(
                image_channels=cfg.image_channels,
                output_dim=cfg.visual_feature_dim,
            )

            # Keep low-level visual filters fixed initially.
            for parameter in self.visual_stem.parameters():
                parameter.requires_grad_(False)

            # Allow the random convolutional features to be remixed into a
            # task-useful visual basis.
            if cfg.train_visual_projection:
                for parameter in self.visual_stem.projection.parameters():
                    parameter.requires_grad_(True)

                for parameter in self.visual_stem.output_norm.parameters():
                    parameter.requires_grad_(True)

            core_input_dim = cfg.visual_feature_dim
        
        else:
            self.visual_stem = None
            core_input_dim = cfg.observation_dim

        self.core = RecurrentSNN(
            core_input_dim,
            cfg.hidden_dim,
            cfg,
            persistent=cfg.encoder_persistent,
            decay=cfg.encoder_membrane_decay,
            record_eligibility=cfg.train_encoder_core_eprop,
        )
        

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


    def prepare_input(
            self,
            observation: torch.Tensor,
        ) -> torch.Tensor:
        if self.visual_stem is None:
            if observation.ndim != 2:
                raise RuntimeError(
                    "Vector encoder expected [batch, features], "
                    f"but received {tuple(observation.shape)}"
                )

            return observation

        if observation.ndim != 4:
            raise RuntimeError(
                "Visual encoder expected [batch, channels, height, width], "
                f"but received {tuple(observation.shape)}"
            )

        return self.visual_stem(observation)


    def encode_with_pre_tanh(
            self,
            observation: torch.Tensor,
            reward: torch.Tensor | None = None,
            update_mask: torch.Tensor | None = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        encoder_input = self.prepare_input(observation)

        hidden = self.norm(
            self.core(encoder_input, update_mask=update_mask),
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
            reward: torch.Tensor | None = None,
            update_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:

        latent, _=self.encode_with_pre_tanh(
            observation,
            reward,
            update_mask=update_mask,
        )

        return latent

    

    def forward(self, observation: torch.Tensor,
                reward: torch.Tensor | None = None,
                update_mask: torch.Tensor | None = None):
        latent = self.encode(
            observation, reward, update_mask=update_mask)
        return latent, self.decoder(latent), self.cue_head(latent)

    def snapshot(self):
        return self.core.snapshot()

    def restore(self, state) -> None:
        self.core.restore(state)

    def reset(self, mask: torch.Tensor | None = None) -> None:
        self.core.reset(mask)

    def preview_encode(
            self,
            observation: torch.Tensor,
            reward: torch.Tensor | None = None,
        ) -> torch.Tensor:
        """Encode without consuming persistent state or eligibility records."""
        state = self.core.snapshot()
        previous_output = self.core.last_output
        previous_records = self.core.last_eligibility_records
        try:
            return self.encode(observation, reward)
        finally:
            self.core.restore(state)
            self.core.last_output = previous_output
            self.core.last_eligibility_records = previous_records


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
        self.gate_head = (nn.Linear(2*cfg.hidden_dim, cfg.strategy_dim)
                          if cfg.learned_strategy_memory else None)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias,
                              math.log(0.05/0.95))
        # The actor-facing context critic is intentionally independent of the
        # full actor-feature critic.  Training both views through one linear
        # head would make the actor block a residual that is implicitly driven
        # toward zero whenever the strategy-only view fits the same target.
        context_input_dim = 2*cfg.hidden_dim+cfg.strategy_dim
        outcome_input_dim = context_input_dim+2*cfg.hidden_dim
        self.context_head = nn.Linear(context_input_dim, 2)

        # The legacy architecture created exactly the context-width head here.
        # Construct the extra augmented head without advancing the global RNG,
        # preserving actor and predictor initialisation for controlled same-seed
        # comparisons.  Both heads are explicitly zero-initialised below.
        downstream_rng_state = torch.random.get_rng_state()
        self.outcome_head = nn.Linear(outcome_input_dim, 2)
        torch.random.set_rng_state(downstream_rng_state)
        # With sparse reward an arbitrary initial critic would manufacture TD
        # credit before any outcome had occurred.  A zero critic makes early
        # no-reward transitions genuinely censored until terminal evidence.
        for head in (self.context_head, self.outcome_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, latent: torch.Tensor, feedback: torch.Tensor,
                deterministic: bool = False,
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
        return {
            "feature": feature,
            "proposal_pre_tanh": proposal_pre_tanh,
            "proposal": proposal,
            "gate": gate,
            "strategy": strategy,
            "previous_strategy": previous_strategy,
        }

    def evaluate_outcome(
            self,
            feature: torch.Tensor,
            strategy: torch.Tensor,
            actor_feature: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a strategy together with the actor's interpretation.

        The outcome loss trains only this calibration head.  Task gradients
        reach the actor, strategy core, and encoder through their TD-modulated
        eligibility paths rather than direct critic backpropagation.
        """
        outcome = self.outcome_head(torch.cat(
            (feature.detach(), strategy.detach(), actor_feature.detach()), -1))
        desirability = outcome[:, 0]
        raw_logvar = outcome[:, 1]
        # Bounded forward value with identity backward gradient.  A hard clamp
        # permanently trapped the uncertainty head at std=exp(1) once its raw
        # log-variance crossed the upper limit.
        outcome_logvar = raw_logvar+(
            raw_logvar.clamp(-5.0, 2.0)-raw_logvar).detach()
        return desirability, outcome_logvar

    def evaluate_context(
            self,
            feature: torch.Tensor,
            strategy: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Causal strategy-only critic used to condition the current actor."""
        outcome = self.context_head(torch.cat(
            (feature.detach(), strategy.detach()), -1))
        desirability = outcome[:, 0]
        raw_logvar = outcome[:, 1]
        outcome_logvar = raw_logvar+(
            raw_logvar.clamp(-5.0, 2.0)-raw_logvar).detach()
        return desirability, outcome_logvar

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
        # These are the causal outputs of the independent strategy-only critic.
        # The augmented value is evaluated only after this actor feature exists.
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
        return {"feature": feature, "logits": logits, "action": action,
                "logp": distribution.log_prob(action),
                "entropy": distribution.entropy()}

    def snapshot(self):
        return self.core.snapshot()

    def restore(self, state) -> None:
        self.core.restore(state)

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

        self.core = RecurrentSNN(
            cfg.latent_dim+cfg.conditioning_dim+ACTION_DIM,
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
        feature = self.norm(self.core(torch.cat(
            (latent, conditioning, action_code), -1)))

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

    def per_world_trace_norm(self) -> torch.Tensor:
        """Current score-eligibility magnitude for each environment world."""
        trace_square = torch.zeros(
            self.traces[0].shape[0],
            device=self.traces[0].device,
            dtype=self.traces[0].dtype,
        )
        with torch.no_grad():
            for trace in self.traces:
                trace_square.add_(
                    trace.reshape(trace.shape[0], -1).square().sum(dim=1))
        return trace_square.sqrt()

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



def contract_strategy_memory_signal(
        memory_jacobians: Sequence[torch.Tensor],
        memory_signal: torch.Tensor,
        weight: float,
    ) -> List[torch.Tensor]:
    """Map d(loss)/d(strategy-memory) through stored temporal Jacobians.

    ``memory_signal`` is already the gradient of a scalar, batch-reduced loss,
    so contributions are summed over worlds rather than averaged a second
    time. The returned tensors are ordinary loss gradients to be minimized.
    """
    if memory_signal.ndim != 2:
        raise RuntimeError(
            "Predictor strategy signal must have shape [worlds, strategy_dim]")

    gradients = []
    with torch.no_grad():
        for memory in memory_jacobians:
            if memory.shape[:2] != memory_signal.shape:
                raise RuntimeError(
                    "Strategy-memory Jacobian and predictor signal mismatch: "
                    f"{tuple(memory.shape[:2])} versus "
                    f"{tuple(memory_signal.shape)}")
            view = memory_signal.shape + (1,)*(memory.ndim-2)
            gradient = (
                memory*memory_signal.view(view)
            ).sum(dim=(0, 1))*weight
            gradients.append(gradient)
    return gradients


def align_predictive_descent_with_task(
        task_directions: Sequence[torch.Tensor],
        predictive_loss_gradients: Sequence[torch.Tensor] | None,
        enabled: bool,
    ) -> Tuple[List[torch.Tensor], float, float, float]:
    """Return prediction-loss descent directions that do not oppose TD.

    Task tensors are ascent directions. Predictor tensors are gradients of a
    loss, so their desired parameter direction is their negative. When the
    global dot product is negative, project the predictive descent direction
    onto the hyperplane orthogonal to the current task direction.
    """
    if predictive_loss_gradients is None:
        zeros = [torch.zeros_like(value) for value in task_directions]
        return zeros, 0.0, 0.0, 0.0

    if len(predictive_loss_gradients) != len(task_directions):
        raise RuntimeError(
            "Predictor-mediated and task-gradient parameter counts differ: "
            f"{len(predictive_loss_gradients)} versus {len(task_directions)}")

    predictive_directions = []
    task_square = torch.zeros(
        (), device=task_directions[0].device,
        dtype=task_directions[0].dtype)
    predictive_square = torch.zeros_like(task_square)
    dot = torch.zeros_like(task_square)

    for task, loss_gradient in zip(
            task_directions, predictive_loss_gradients):
        predictive = -loss_gradient
        predictive_directions.append(predictive)
        task_square += task.square().sum()
        predictive_square += predictive.square().sum()
        dot += (task*predictive).sum()

    denominator = (
        task_square.sqrt()*predictive_square.sqrt()
    ).clamp_min(1e-12)
    cosine = dot/denominator

    if enabled and bool((dot < 0).item()) and bool((task_square > 0).item()):
        coefficient = dot/task_square.clamp_min(1e-12)
        predictive_directions = [
            predictive-coefficient*task
            for predictive, task in zip(
                predictive_directions, task_directions)
        ]

    aligned_square = torch.stack([
        value.square().sum() for value in predictive_directions
    ]).sum()

    return (
        predictive_directions,
        float(predictive_square.sqrt().detach()),
        float(aligned_square.sqrt().detach()),
        float(cosine.detach()),
    )


class RecurrentStrategyEprop:
    """Bellec-style LIF eligibility plus exact leaky-memory eligibility.

    The recurrent SNN carries neuron-local membrane eligibility across every
    SNN tick and environment decision.  The resulting derivative of each
    strategy write is then carried through the external strategy memory:

        dm_t/dtheta = keep*dm_{t-1}/dtheta + write*ds_t/dtheta.

    The actor's current d log pi / dm contracts with this Jacobian before the
    ordinary reward eligibility trace is advanced. Prediction loss can also
    contract with the same Jacobian, providing an explicitly mediated
    predictor -> strategy-memory -> strategizer learning path. No temporal
    autograd graph or BPTT is retained.
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
        self.last_predictive_gradient_norm = 0.0
        self.last_predictive_aligned_norm = 0.0
        self.last_predictive_task_cosine = 0.0

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
                   learned_gate: bool) -> float:
        if learned_gate:
            combined = torch.cat((proposal, gate), -1)
            combined_jacobians = self._current_output_jacobians(combined)
            k = proposal.shape[-1]
            proposal_jacobians = [value[:, :k]
                                  for value in combined_jacobians]
            gate_jacobians = [value[:, k:]
                              for value in combined_jacobians]
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
            proposal_jacobians = self._current_output_jacobians(proposal)
            gate_jacobians = []
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
                reward_trace.mul_(self.decay).add_(score)
                memory_square += memory.square().sum()
                score_square += score.square().sum()
        self.last_memory_norm = float(memory_square.sqrt())
        self.last_score_norm = float(score_square.sqrt())
        return float(torch.stack([trace.square().sum()
                                  for trace in self.reward_traces]).sum().sqrt())

    def predictive_gradients(
            self,
            memory_signal: torch.Tensor,
            weight: float,
        ) -> List[torch.Tensor]:
        return contract_strategy_memory_signal(
            self.memory_jacobians,
            memory_signal,
            weight,
        )

    def apply(
            self,
            td_error: torch.Tensor,
            minimizing_gradients: Sequence[torch.Tensor | None] | None = None,
            predictive_gradients: Sequence[torch.Tensor] | None = None,
            align_predictive: bool = True,
        ) -> Tuple[float, float]:

        self.optimizer.zero_grad(set_to_none=True)

        task_direction_square = torch.zeros(
            (),
            device=td_error.device,
        )

        task_directions = []
        for trace in self.reward_traces:
            view = (
                td_error.shape[0],
            ) + (1,) * (trace.ndim - 1)
            task_direction = (
                trace * td_error.view(view)
            ).mean(dim=0)
            task_directions.append(task_direction)
            task_direction_square += task_direction.square().sum()

        (
            predictive_directions,
            self.last_predictive_gradient_norm,
            self.last_predictive_aligned_norm,
            self.last_predictive_task_cosine,
        ) = align_predictive_descent_with_task(
            task_directions,
            predictive_gradients,
            align_predictive,
        )

        for index, (parameter, task_direction, predictive_direction) in (
                enumerate(zip(
                    self.parameters,
                    task_directions,
                    predictive_directions))):
            combined_direction = task_direction+predictive_direction

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
            visual_learning_rate: float,
            core_learning_rate: float,
            core_trace_decay: float,
            gradient_clip: float,
        ) -> None:
        self.encoder = encoder
        self.decay = decay
        self.core_trace_decay = core_trace_decay
        self.gradient_clip = gradient_clip

        # Existing encoder parameters receiving delayed strategy credit.
        self.latent_parameters = [
            encoder.latent_head.weight,
            encoder.latent_head.bias,
        ]

        # Optional visual parameters receiving the same causal credit signal.
        self.visual_parameters = []

        if encoder.visual_stem is not None:
            self.visual_parameters.extend(
                parameter
                for parameter in encoder.visual_stem.projection.parameters()
                if parameter.requires_grad
            )

            self.visual_parameters.extend(
                parameter
                for parameter in encoder.visual_stem.output_norm.parameters()
                if parameter.requires_grad
            )

        # Recurrent encoder parameters require an explicit temporal
        # eligibility path because persistent membrane state is detached
        # between environment decisions.
        self.core_parameters = []
        if encoder.cfg.train_encoder_core_eprop:
            self.core_parameters = [
                encoder.core.input.weight,
                encoder.core.input.bias,
                encoder.core.recurrent.weight,
                encoder.core.bias,
            ]

        self.parameters = (
            self.latent_parameters
            + self.visual_parameters
            + self.core_parameters
        )

        self.visual_parameter_ids = {
            id(parameter) for parameter in self.visual_parameters
        }
        self.core_parameter_ids = {
            id(parameter) for parameter in self.core_parameters
        }

        core = encoder.core
        self.core_parameter_kind = {
            id(core.input.weight): "input_weight",
            id(core.input.bias): "input_bias",
            id(core.recurrent.weight): "recurrent_weight",
            id(core.bias): "bias",
        }
        self.non_core_indices = [
            index for index, parameter in enumerate(self.parameters)
            if id(parameter) not in self.core_parameter_kind
        ]

        device = encoder.latent_head.weight.device

        hidden = core.hidden_dim
        input_dim = core.input.in_features
        self.epsilon_in = torch.zeros(
            worlds, hidden, input_dim, device=device)
        self.epsilon_rec = torch.zeros(
            worlds, hidden, hidden, device=device)
        self.epsilon_bias = torch.zeros(
            worlds, hidden, device=device)

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

        optimizer_groups = [
            {
                "params": self.latent_parameters,
                "lr": learning_rate,
            }
        ]

        if self.visual_parameters:
            optimizer_groups.append(
                {
                    "params": self.visual_parameters,
                    "lr": visual_learning_rate,
                }
            )

        if self.core_parameters:
            optimizer_groups.append(
                {
                    "params": self.core_parameters,
                    "lr": core_learning_rate,
                }
            )

        self.optimizer = torch.optim.Adam(
            optimizer_groups,
            maximize=True,
        )

        self.last_memory_norm = 0.0
        self.last_score_norm = 0.0
        self.last_eligibility_norm = 0.0
        self.last_gradient_norm = 0.0
        self.last_core_eligibility_norm = 0.0
        self.last_visual_step_norm = 0.0
        self.last_core_step_norm = 0.0
        self.last_predictive_gradient_norm = 0.0
        self.last_predictive_aligned_norm = 0.0
        self.last_predictive_task_cosine = 0.0

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

        jacobians = [
            torch.zeros(
                (batch, output_dim) + tuple(parameter.shape),
                device=output.device,
                dtype=output.dtype,
            )
            for parameter in self.parameters
        ]

        # Feed-forward encoder parameters have exact local Jacobians.
        if self.non_core_indices:
            basis = torch.eye(
                batch,
                device=output.device,
                dtype=output.dtype,
            )
            non_core_parameters = [
                self.parameters[index] for index in self.non_core_indices
            ]
            for coordinate in range(output_dim):
                gradients = torch.autograd.grad(
                    output[:, coordinate],
                    non_core_parameters,
                    grad_outputs=basis,
                    is_grads_batched=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                for index, gradient in zip(
                        self.non_core_indices, gradients):
                    if gradient is not None:
                        jacobians[index][:, coordinate].copy_(
                            gradient.detach())

        # Persistent recurrent state is detached between decisions. Carry
        # neuron-local encoder eligibility explicitly across those decisions.
        if self.core_parameters:
            core = self.encoder.core
            if core.last_output is None:
                raise RuntimeError("encoder core has no eligibility output")

            hidden = core.hidden_dim
            feature_gradients = []
            for coordinate in range(output_dim):
                feature_gradients.append(torch.autograd.grad(
                    output[:, coordinate].sum(),
                    core.last_output,
                    retain_graph=True,
                )[0].detach())
            feature_gradient = torch.stack(feature_gradients, dim=1)

            jacobian_in = torch.zeros(
                batch, output_dim, hidden, self.epsilon_in.shape[-1],
                device=output.device, dtype=output.dtype)
            jacobian_rec = torch.zeros(
                batch, output_dim, hidden, hidden,
                device=output.device, dtype=output.dtype)
            jacobian_bias = torch.zeros(
                batch, output_dim, hidden,
                device=output.device, dtype=output.dtype)
            recurrent_mask = 1-torch.eye(
                hidden, device=output.device, dtype=output.dtype)
            ticks = max(len(core.last_eligibility_records), 1)

            with torch.no_grad():
                for tick, (value, presynaptic, pseudo_derivative) in enumerate(
                        core.last_eligibility_records):
                    carry = core.decay*(
                        self.core_trace_decay if tick == 0 else 1.0)
                    self.epsilon_in.mul_(carry).add_(value[:, None, :])
                    self.epsilon_rec.mul_(carry).add_(
                        presynaptic[:, None, :]*recurrent_mask[None])
                    self.epsilon_bias.mul_(carry).add_(1)
                    coefficient = (
                        feature_gradient[:, :, :hidden]
                        +feature_gradient[:, :, hidden:]
                         *pseudo_derivative[:, None, :])/ticks
                    jacobian_in.add_(
                        coefficient[:, :, :, None]
                        *self.epsilon_in[:, None, :, :])
                    jacobian_rec.add_(
                        coefficient[:, :, :, None]
                        *self.epsilon_rec[:, None, :, :])
                    jacobian_bias.add_(
                        coefficient*self.epsilon_bias[:, None, :])

                core_square = (
                    self.epsilon_in.square().sum()
                    +self.epsilon_rec.square().sum()
                    +self.epsilon_bias.square().sum())
                self.last_core_eligibility_norm = float(core_square.sqrt())

            for index, parameter in enumerate(self.parameters):
                kind = self.core_parameter_kind.get(id(parameter))
                if kind == "input_weight":
                    jacobians[index].copy_(jacobian_in)
                elif kind in ("input_bias", "bias"):
                    jacobians[index].copy_(jacobian_bias)
                elif kind == "recurrent_weight":
                    jacobians[index].copy_(jacobian_rec)

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

    def predictive_gradients(
            self,
            memory_signal: torch.Tensor,
            weight: float,
        ) -> List[torch.Tensor]:
        return contract_strategy_memory_signal(
            self.memory_jacobians,
            memory_signal,
            weight,
        )

    def apply(
            self,
            td_error: torch.Tensor,
            predictive_gradients: Sequence[torch.Tensor] | None = None,
            align_predictive: bool = True,
        ) -> tuple[float, float]:

        self.optimizer.zero_grad(set_to_none=True)

        gradient_square = torch.zeros(
            (),
            device=td_error.device,
        )

        task_directions = []
        for trace in self.reward_traces:
            view = (
                td_error.shape[0],
            ) + (1,) * (trace.ndim - 1)

            gradient = (
                trace * td_error.view(view)
            ).mean(dim=0)
            task_directions.append(gradient)
            gradient_square += gradient.square().sum()

        (
            predictive_directions,
            self.last_predictive_gradient_norm,
            self.last_predictive_aligned_norm,
            self.last_predictive_task_cosine,
        ) = align_predictive_descent_with_task(
            task_directions,
            predictive_gradients,
            align_predictive,
        )

        combined_square = torch.zeros_like(gradient_square)
        for parameter, task_direction, predictive_direction in zip(
                self.parameters,
                task_directions,
                predictive_directions):
            parameter.grad = task_direction+predictive_direction
            combined_square += parameter.grad.square().sum()

        raw_gradient_norm = gradient_square.sqrt()
        raw_combined_norm = combined_square.sqrt()

        if self.gradient_clip > 0:
            scale = torch.clamp(
                self.gradient_clip
                / raw_combined_norm.clamp_min(1e-12),
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

        visual_step_square = torch.zeros(
            (),
            device=td_error.device,
        )
        core_step_square = torch.zeros(
            (),
            device=td_error.device,
        )

        for parameter, old in zip(self.parameters, before):
            if id(parameter) in self.visual_parameter_ids:
                visual_step_square += (
                    parameter - old
                ).square().sum()
            if id(parameter) in self.core_parameter_ids:
                core_step_square += (
                    parameter - old
                ).square().sum()

        self.last_visual_step_norm = float(
            visual_step_square.sqrt().detach()
        )
        self.last_core_step_norm = float(
            core_step_square.sqrt().detach()
        )


        step_square = torch.stack([
            (parameter - old).square().sum()
            for parameter, old in zip(self.parameters, before)
        ]).sum()

        return (
            self.last_gradient_norm,
            float(step_square.sqrt()),
        )

    def reset(self, mask: torch.Tensor) -> None:
        if not bool(mask.any()):
            return

        with torch.no_grad():
            self.epsilon_in[mask] = 0
            self.epsilon_rec[mask] = 0
            self.epsilon_bias[mask] = 0

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
    def __init__(self, cfg: Config, condition: str,
                 device: torch.device, seed: int) -> None:
        if cfg.encoder_learning_mode not in (
                "cue_auxiliary", "reward_eprop", "hybrid"):
            raise ValueError(
                "encoder_learning_mode must be cue_auxiliary, "
                "reward_eprop, or hybrid")

        if (
            cfg.visual_observations
            and cfg.encoder_learning_mode != "reward_eprop"
        ):
            raise ValueError(
                "Visual observations currently require "
                "encoder_learning_mode='reward_eprop'."
            )
           
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
        if cfg.predictor_strategy_weight < 0.0:
            raise ValueError(
                "predictor_strategy_weight must be non-negative")
        if cfg.predictor_mediated_encoder_weight < 0.0:
            raise ValueError(
                "predictor_mediated_encoder_weight must be non-negative")
        if (cfg.predictor_mediated_encoder_weight > 0.0
                and not cfg.use_strategy_encoder_eprop):
            raise ValueError(
                "predictor_mediated_encoder_weight requires "
                "use_strategy_encoder_eprop=True")
        if (cfg.predictor_mediated_encoder_weight > 0.0
                and not cfg.detach_predictor_from_encoder):
            raise ValueError(
                "Mediated predictor credit requires "
                "detach_predictor_from_encoder=True; direct predictor-to-"
                "encoder gradients must remain disabled.")
        if (cfg.encoder_persistent
                and cfg.encoder_learning_mode != "reward_eprop"):
            raise ValueError(
                "Persistent encoder currently requires "
                "encoder_learning_mode='reward_eprop' so each observation "
                "is encoded exactly once.")
        if cfg.encoder_persistent and cfg.use_actor_encoder_eprop:
            raise ValueError(
                "Persistent encoder core learning is owned by "
                "StrategyEncoderEprop; disable use_actor_encoder_eprop.")
        if cfg.encoder_persistent and cfg.use_predictor_encoder_eprop:
            raise ValueError(
                "Persistent encoder currently requires "
                "use_predictor_encoder_eprop=False to avoid two temporal "
                "eligibility systems owning the same encoder parameters.")
        if cfg.encoder_persistent and cfg.use_representation_critic:
            raise ValueError(
                "Persistent encoder currently requires "
                "use_representation_critic=False because that auxiliary "
                "path re-encodes observations.")
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
                visual_learning_rate=cfg.visual_projection_eprop_lr,
                core_learning_rate=cfg.encoder_core_eprop_lr,
                core_trace_decay=cfg.encoder_core_trace_decay,
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
        strategy_encoder_parameter_ids = (
            {
                id(parameter)
                for parameter in self.strategy_encoder_eprop.parameters
            }
            if self.strategy_encoder_eprop is not None
            else set()
        )
        auxiliary_encoder_parameters = [
            parameter
            for parameter in self.encoder.parameters()
            if id(parameter) not in strategy_encoder_parameter_ids
        ]
        self.encoder_optimizer = torch.optim.Adam(
            auxiliary_encoder_parameters, lr=cfg.encoder_lr)
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), lr=cfg.predictor_lr)
        self.critic_optimizer = torch.optim.Adam(
            module_parameters((self.strategizer.context_head,
                               self.strategizer.outcome_head)),
            lr=cfg.critic_lr)
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

    def reset(
            self,
            mask: torch.Tensor,
            reset_encoder: bool = True,
        ) -> None:
        if reset_encoder:
            self.encoder.reset(mask)
            if self.target_encoder is not None:
                self.target_encoder.reset(mask)
        self.strategizer.reset(mask); self.predictor.reset(mask)
        self.actor.reset(mask)
        self.actor_eprop.reset(mask); self.strategy_eprop.reset(mask)

        if self.predictor_eprop is not None:
            self.predictor_eprop.reset(mask)
        
        if self.predictor_encoder_eprop is not None:
            self.predictor_encoder_eprop.reset(mask)

        if self.encoder_eprop is not None:
            self.encoder_eprop.reset(mask)

        if self.strategy_encoder_eprop is not None:
            self.strategy_encoder_eprop.reset(mask)

        self.feedback[mask] = 0
        self.strategy_memory[mask] = 0
        self.desirability_memory[mask] = 0
        self.outcome_logvar_memory[mask] = 0

    def strategy_and_action(self, latent: torch.Tensor,
                            deterministic: bool = False):
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
            previous_strategy=(self.strategy_memory.detach().requires_grad_(True)
                               if self.cfg.learned_strategy_memory else None))
        context_desirability, context_outcome_logvar = (
            self.strategizer.evaluate_context(
                strategy["feature"], strategy["strategy"]))
        strategy["context_desirability"] = context_desirability
        strategy["context_outcome_logvar"] = context_outcome_logvar

        if self.condition == "actor_only":
            actor_strategy = torch.zeros_like(strategy["strategy"])
            actor_desirability = torch.zeros_like(context_desirability)
            actor_outcome_logvar = torch.zeros_like(context_outcome_logvar)
        elif self.condition == "separated":
            if self.cfg.learned_strategy_memory:
                actor_strategy = strategy["strategy"]
                actor_desirability = context_desirability
                actor_outcome_logvar = context_outcome_logvar
            else:
                keep = self.cfg.strategy_retention
                actor_strategy = (keep*self.strategy_memory.detach()
                                  +(1-keep)*strategy["strategy"])
                actor_desirability = (
                    keep*self.desirability_memory.detach()
                    +(1-keep)*context_desirability)
                actor_outcome_logvar = (
                    keep*self.outcome_logvar_memory.detach()
                    +(1-keep)*context_outcome_logvar)
            self.strategy_memory = actor_strategy.detach()
            self.desirability_memory = actor_desirability.detach()
            self.outcome_logvar_memory = actor_outcome_logvar.detach()
        else:
            actor_strategy = strategy["strategy"]
            actor_desirability = context_desirability
            actor_outcome_logvar = context_outcome_logvar
        actor = self.actor(
            actor_latent, actor_strategy, actor_desirability,
            actor_outcome_logvar, deterministic,
            exploration=0.0 if deterministic else self.cfg.exploration_rate)
        desirability, outcome_logvar = self.strategizer.evaluate_outcome(
            strategy["feature"], strategy["strategy"], actor["feature"])
        strategy["desirability"] = desirability
        strategy["outcome_logvar"] = outcome_logvar
        return (strategy, actor, actor_strategy, actor_desirability,
                actor_outcome_logvar)


def latent_reconstruction_update(
        system: System, observation: torch.Tensor,
        reward_context: torch.Tensor | None = None,
        policy_graph: bool = False,
        cue_target: torch.Tensor | None = None,
        advance_state: bool = True):
    

    if cue_target is None:
        if observation.ndim == 2:
            cue_target = observation[:, 9:12].argmax(dim=-1)
        else:
            raise RuntimeError(
                "Pixel observations require an explicit cue_target "
                "for cue diagnostics."
            )


    keep_encoder_graph = policy_graph and (
        system.use_jepa
        or system.encoder_eprop is not None
        or system.predictor_encoder_eprop is not None
    )


    if system.use_jepa:
        # JEPA mode never reconstructs observations and never consumes cue
        # labels as a learning target.  Its encoder update happens in
        # predictor_update against the EMA target encoder.
        if advance_state:
            latent, sigreg_state = system.encoder.encode_with_pre_tanh(
                observation, reward_context)
        else:
            state = system.encoder.snapshot()
            previous_output = system.encoder.core.last_output
            previous_records = system.encoder.core.last_eligibility_records
            try:
                latent, sigreg_state = system.encoder.encode_with_pre_tanh(
                    observation, reward_context)
            finally:
                system.encoder.restore(state)
                system.encoder.core.last_output = previous_output
                system.encoder.core.last_eligibility_records = previous_records
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

    # Prediction sees a detached strategy leaf. Its gradient therefore cannot
    # backpropagate directly into either strategizer or encoder; it is routed
    # explicitly through their stored strategy-memory Jacobians below.
    use_strategy_mediation = (
        system.cfg.predictor_strategy_weight > 0.0
        or system.cfg.predictor_mediated_encoder_weight > 0.0
    )
    predictor_strategy = strategy.detach()
    if use_strategy_mediation:
        predictor_strategy = predictor_strategy.requires_grad_(True)

    #Predictor Call
    predicted_next_latent = system.predictor(
        predictor_latent,
        predictor_strategy,
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

    if use_strategy_mediation:
        prediction_strategy_signal = torch.autograd.grad(
            prediction_loss,
            predictor_strategy,
            retain_graph=True,
        )[0].detach()
    else:
        prediction_strategy_signal = torch.zeros_like(strategy)

    prediction_strategy_signal_norm = float(
        prediction_strategy_signal.square().sum().sqrt())

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
        float(sigreg_loss.detach()),
        prediction_strategy_signal,
        prediction_strategy_signal_norm,
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

def compatible_action_statistics(
        probabilities: torch.Tensor,
        actions: torch.Tensor,
        current_direction: torch.Tensor,
        desired_direction: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Probability and realization of an action that turns toward a heading.

    When facing away from the desired heading, either turn is an equally short
    first correction. Actions are left=0, right=1, and forward=2.
    """
    turn = torch.remainder(desired_direction-current_direction, 4)
    compatible_probability = torch.where(
        turn == 0,
        probabilities[:, 2],
        torch.where(
            turn == 1,
            probabilities[:, 1],
            torch.where(
                turn == 3,
                probabilities[:, 0],
                probabilities[:, 0]+probabilities[:, 1],
            ),
        ),
    )
    compatible_action = (
        ((turn == 0) & (actions == 2))
        | ((turn == 1) & (actions == 1))
        | ((turn == 3) & (actions == 0))
        | ((turn == 2) & ((actions == 0) | (actions == 1)))
    )
    return compatible_probability, compatible_action.float()


def opposite_cue_matched_partners(
        mask: torch.Tensor,
        cue: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
    """Match each selected world to the closest-age opposite-cue world."""
    partners = torch.full_like(cue, -1)
    selected = torch.nonzero(mask, as_tuple=False).flatten()
    for index in selected.tolist():
        candidates = selected[cue[selected] != cue[index]]
        if candidates.numel():
            closest = torch.argmin((age[candidates]-age[index]).abs())
            partners[index] = candidates[closest]
    return partners

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
    system = System(cfg, condition, device, seed)
    print(
        "predictor_encoder_eprop_enabled=",
        system.predictor_encoder_eprop is not None
    )
    observation = torch.tensor(env.observation(), device=device)
    completed = episodes = successes = wrong_total = timeout_total = 0
    window_steps = window_episodes = window_successes = window_wrong = 0
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

    encoder_delay_values = {
        "cue": [],
        "delay_1_4": [],
        "delay_5_8": [],
        "delay_9_plus": [],
    }

    cue_delay_labels = {
        "cue": [],
        "delay_1_4": [],
        "delay_5_8": [],
        "delay_9_plus": [],
    }
    cue_timecourse_history: List[Dict[str, float | int | str]] = []
    policy_diagnostic_history: List[Dict[str, float | int]] = []
    credit_diagnostic_history: List[Dict[str, float | int | str]] = []

    visible_encoder_values: List[torch.Tensor] = []
    visible_encoder_labels: List[torch.Tensor] = []
    latent_distribution_values: List[torch.Tensor] = []
    previous_strategy = torch.zeros(
        cfg.worlds, cfg.strategy_dim, device=device)
    reward_context = torch.zeros(cfg.worlds, device=device)

    # The online encoder advances at the top of each decision. The target
    # encoder is advanced here once for the initial observation, then once per
    # true transition below.
    system.encoder.reset()
    if system.target_encoder is not None:
        system.target_encoder.reset()
        with torch.no_grad():
            system.target_encoder.encode(observation, reward_context)

    def add(name: str, value: float) -> None:
        sums[name] = sums.get(name, 0.0)+value

    def add_masked(
            name: str,
            values: torch.Tensor,
            mask: torch.Tensor,
        ) -> None:
        detached = values.detach()
        selected = mask.bool()
        add(f"{name}_sum", float(detached[selected].sum()))
        add(f"{name}_count", float(selected.sum()))

    while completed < cfg.transitions:
        decision_age_np = env.age.copy()
        decision_cue_np = env.cue.copy()

        observation_cue_target = torch.tensor(
            np.where(
                decision_age_np < cfg.cue_steps,
                decision_cue_np,
                2,
            ),
            device=device,
            dtype=torch.long,
        )
        (latent, reconstruction, cue_accuracy, cue_visible_correct, cue_visible_count, sigreg_state) = latent_reconstruction_update(system, observation, reward_context, policy_graph=True, cue_target = observation_cue_target)
        
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
        (strategy, actor, actor_strategy, actor_desirability,
         actor_outcome_logvar) = system.strategy_and_action(latent)

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

        decision_x = torch.tensor(
            env.x.copy(), device=device, dtype=torch.long)
        decision_y = torch.tensor(
            env.y.copy(), device=device, dtype=torch.long)
        decision_direction = torch.tensor(
            env.direction.copy(), device=device, dtype=torch.long)

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

                encoder_delay_values[delay_name].append(
                    latent.detach()[
                        delay_mask
                    ].cpu()
                )

                cue_delay_labels[delay_name].append(
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
                learned_gate=cfg.learned_strategy_memory)

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
            raw_prob = actor["logits"].softmax(-1)
            real_prob = (
                (1-cfg.exploration_rate)*raw_prob
                +cfg.exploration_rate/ACTION_DIM
            )

            stem_mask = (
                (decision_x == env.center)
                & (decision_y > 1)
            )
            junction_mask = (
                (decision_x == env.center)
                & (decision_y == 1)
            )
            canonical_junction_mask = (
                junction_mask & (decision_direction == 0)
            )

            north = torch.zeros_like(decision_direction)
            desired_goal_direction = torch.where(
                decision_cue == 0,
                torch.full_like(decision_direction, 3),
                torch.ones_like(decision_direction),
            )
            stem_navigation_probability, stem_navigation_selected = (
                compatible_action_statistics(
                    real_prob,
                    actor["action"],
                    decision_direction,
                    north,
                )
            )
            junction_goal_probability, junction_goal_selected = (
                compatible_action_statistics(
                    real_prob,
                    actor["action"],
                    decision_direction,
                    desired_goal_direction,
                )
            )

            correct_turn_probability = real_prob.gather(
                1, decision_cue[:, None]).squeeze(1)
            correct_turn_selected = (
                actor["action"] == decision_cue).float()
            correct_turn_greedy = (
                raw_prob.argmax(dim=-1) == decision_cue).float()
            junction_blocked_probability = real_prob[:, 2]
            junction_blocked_selected = (
                actor["action"] == 2).float()

            forward_blocked = torch.tensor(
                [env._wall(world, 0) for world in range(cfg.worlds)],
                device=device,
                dtype=torch.bool,
            )
            blocked_forward_probability = real_prob[:, 2]*forward_blocked
            blocked_forward_selected = (
                (actor["action"] == 2) & forward_blocked).float()

            # Actor is stateless in the separated conditions, so these probes
            # cannot disturb a hidden action state.
            if condition != "actor_only":
                shuffled = system.actor(
                    latent, actor_strategy[permutation],
                    actor_desirability[permutation],
                    actor_outcome_logvar[permutation],
                    deterministic=True)[
                        "logits"].softmax(-1)
                zeroed = system.actor(
                    latent, torch.zeros_like(actor_strategy),
                    torch.zeros_like(actor_desirability),
                    torch.zeros_like(actor_outcome_logvar),
                    deterministic=True)[
                        "logits"].softmax(-1)
                shuffled = (
                    (1-cfg.exploration_rate)*shuffled
                    +cfg.exploration_rate/ACTION_DIM
                )
                zeroed = (
                    (1-cfg.exploration_rate)*zeroed
                    +cfg.exploration_rate/ACTION_DIM
                )
                shuffle_tv = float(0.5*(real_prob-shuffled).abs().sum(-1).mean())
                zero_tv = float(0.5*(real_prob-zeroed).abs().sum(-1).mean())

                partners = opposite_cue_matched_partners(
                    canonical_junction_mask,
                    decision_cue,
                    decision_age,
                )
                paired_junction_mask = partners >= 0
                safe_partners = partners.clamp_min(0)
                cue_flipped = system.actor(
                    latent,
                    actor_strategy[safe_partners],
                    actor_desirability,
                    actor_outcome_logvar,
                    deterministic=True,
                )["logits"].softmax(-1)
                cue_flipped = (
                    (1-cfg.exploration_rate)*cue_flipped
                    +cfg.exploration_rate/ACTION_DIM
                )
                cue_flip_tv = 0.5*(
                    real_prob-cue_flipped).abs().sum(dim=-1)
                flipped_correct_probability = cue_flipped.gather(
                    1, decision_cue[:, None]).squeeze(1)
                cue_flip_correct_drop = (
                    correct_turn_probability-flipped_correct_probability)
            else:
                shuffle_tv = zero_tv = 0.0
                paired_junction_mask = torch.zeros_like(junction_mask)
                cue_flip_tv = torch.zeros_like(correct_turn_probability)
                cue_flip_correct_drop = torch.zeros_like(
                    correct_turn_probability)



            next_np, reward_np, done_np, success_np, wrong_np, cue_np, age_np = (
                env.step(
                    actor["action"].detach().cpu().numpy()
                )
            )

            next_observation = torch.tensor(
                next_np,
                device=device,
            )

            # This target describes next_observation. env.step() has already reset
            # completed worlds, so use the environment's current age and cue.
            next_observation_cue_target = torch.tensor(
                np.where(
                    env.age < cfg.cue_steps,
                    env.cue,
                    2,
                ),
                device=device,
                dtype=torch.long,
            )

            prediction_observation = torch.tensor(
                env.transition_observation,
                device=device,
            )

            done = torch.tensor(
                done_np,
                device=device,
                dtype=torch.bool,
            )

            reward = torch.tensor(
                reward_np,
                device=device,
            )

            success_mask = torch.tensor(
                success_np, device=device, dtype=torch.bool)
            wrong_mask = torch.tensor(
                wrong_np, device=device, dtype=torch.bool)
            timeout_mask = done & ~success_mask & ~wrong_mask
            nonterminal_mask = ~done



        # Terminal outcomes are the only non-zero rewards in this task, so
        # terminal transitions must train the predictor.  Its target is the
        # true post-action observation captured before the environment reset.
        valid_prediction = torch.ones_like(done)
        next_reward_context = torch.where(
            done, torch.zeros_like(reward), reward)


        if system.use_jepa:
            (
                _,
                next_reconstruction,
                next_cue_accuracy,
                next_cue_visible_correct,
                next_cue_visible_count,
                _,
            ) = latent_reconstruction_update(
                system,
                next_observation,
                next_reward_context,
                cue_target=next_observation_cue_target,
                advance_state=False,
            )


            with torch.no_grad():
                target_state_before = system.target_encoder.snapshot()
                target_next_latent = system.target_encoder.encode(
                    prediction_observation, reward)
                target_state_after = system.target_encoder.snapshot()
                target_output_after = system.target_encoder.core.last_output
                target_records_after = (
                    system.target_encoder.core.last_eligibility_records)
                system.target_encoder.restore(target_state_before)
                neutral_target = system.target_encoder.encode(
                    prediction_observation, torch.zeros_like(reward))
                system.target_encoder.restore(target_state_after)
                system.target_encoder.core.last_output = target_output_after
                system.target_encoder.core.last_eligibility_records = (
                    target_records_after)
                reward_latent_shift = (
                    target_next_latent-neutral_target).square().mean(-1)
                rewarded = reward != 0
                reward_latent_shift = float(
                    reward_latent_shift[rewarded].mean()
                    if bool(rewarded.any()) else 0.0)

            (new_feedback, predictor_loss, prediction_mse,
            joy_prediction_error, joy_event_count,
            predictor_eligibility, predictor_eprop_gradient,
            predictor_encoder_eligibility,
            predictor_encoder_eprop_gradient,
            sigreg_loss,
            prediction_strategy_signal,
            prediction_strategy_signal_norm) = predictor_update(
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
                next_latent = system.encoder.preview_encode(
                    next_observation, next_reward_context)
        else:
            (
                next_latent,
                next_reconstruction,
                next_cue_accuracy,
                next_cue_visible_correct,
                next_cue_visible_count,
                _,
            ) = latent_reconstruction_update(
                system,
                next_observation,
                next_reward_context,
                cue_target=next_observation_cue_target,
            )


            
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
                prediction_strategy_signal,
                prediction_strategy_signal_norm,
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

        if condition != "actor_only":
            predictive_strategy_gradients = (
                system.strategy_eprop.predictive_gradients(
                    prediction_strategy_signal,
                    cfg.predictor_strategy_weight,
                )
                if cfg.predictor_strategy_weight > 0.0
                else None
            )
            predictive_encoder_gradients = (
                system.strategy_encoder_eprop.predictive_gradients(
                    prediction_strategy_signal,
                    cfg.predictor_mediated_encoder_weight,
                )
                if (
                    system.strategy_encoder_eprop is not None
                    and cfg.predictor_mediated_encoder_weight > 0.0
                )
                else None
            )
        else:
            predictive_strategy_gradients = None
            predictive_encoder_gradients = None

        # Provisional next value advances both recurrent policy stages, then
        # immediately rolls them back: no state or graph is consumed twice.
        strategy_snapshot = system.strategizer.snapshot()
        actor_snapshot = system.actor.snapshot()
        strategy_last_output = system.strategizer.core.last_output
        strategy_last_records = (
            system.strategizer.core.last_eligibility_records)
        actor_last_output = system.actor.core.last_output
        actor_last_records = system.actor.core.last_eligibility_records
        system.feedback = new_feedback.detach()
        with torch.no_grad():
            next_strategy = system.strategizer(
                next_latent, system.feedback, deterministic=True,
                previous_strategy=(system.strategy_memory.detach()
                    if cfg.learned_strategy_memory else None))
            next_context_desirability, next_context_outcome_logvar = (
                system.strategizer.evaluate_context(
                    next_strategy["feature"], next_strategy["strategy"]))
            if condition == "actor_only":
                next_actor_strategy = torch.zeros_like(
                    next_strategy["strategy"])
                next_actor_desirability = torch.zeros_like(
                    next_context_desirability)
                next_actor_outcome_logvar = torch.zeros_like(
                    next_context_outcome_logvar)
            elif condition == "separated" and not cfg.learned_strategy_memory:
                keep = cfg.strategy_retention
                next_actor_strategy = (
                    keep*system.strategy_memory.detach()
                    +(1-keep)*next_strategy["strategy"])
                next_actor_desirability = (
                    keep*system.desirability_memory.detach()
                    +(1-keep)*next_context_desirability)
                next_actor_outcome_logvar = (
                    keep*system.outcome_logvar_memory.detach()
                    +(1-keep)*next_context_outcome_logvar)
            else:
                next_actor_strategy = next_strategy["strategy"]
                next_actor_desirability = next_context_desirability
                next_actor_outcome_logvar = next_context_outcome_logvar
            next_actor = system.actor(
                next_latent, next_actor_strategy,
                next_actor_desirability, next_actor_outcome_logvar,
                deterministic=True)
            next_value, next_outcome_logvar = (
                system.strategizer.evaluate_outcome(
                    next_strategy["feature"],
                    next_strategy["strategy"],
                    next_actor["feature"],
                ))
        system.strategizer.restore(strategy_snapshot)
        system.actor.restore(actor_snapshot)
        system.strategizer.core.last_output = strategy_last_output
        system.strategizer.core.last_eligibility_records = (
            strategy_last_records)
        system.actor.core.last_output = actor_last_output
        system.actor.core.last_eligibility_records = actor_last_records

        (representation_critic_loss,
         representation_critic_mae,
         critic_encoder_step) = representation_critic_update(
             system, observation, actor_strategy, reward, done,
             next_observation, next_actor_strategy)
        td = (reward+cfg.gamma*(~done).float()*next_value
              -strategy["desirability"].detach()).clamp(
                  -cfg.td_clip, cfg.td_clip)

        actor_trace_per_world = (
            system.actor_eprop.per_world_trace_norm().detach())

        actor_direction, actor_step = system.actor_eprop.apply(td)
        if system.encoder_eprop is not None:
            encoder_direction, encoder_step = system.encoder_eprop.apply(td)
        else:
            encoder_direction = encoder_step = 0.0

        if system.strategy_encoder_eprop is not None:
            (
                strategy_encoder_direction,
                strategy_encoder_step,
            ) = system.strategy_encoder_eprop.apply(
                td,
                predictive_gradients=predictive_encoder_gradients,
                align_predictive=cfg.align_predictor_with_task_gradient,
            )
        else:
            strategy_encoder_direction = 0.0
            strategy_encoder_step = 0.0

        system.update_target_encoder()
        system.update_target_representation_critic()
        if condition != "actor_only":
            strategy_direction, strategy_step = system.strategy_eprop.apply(
                td,
                minimizing_gradients=strategy_sigreg_gradients,
                predictive_gradients=predictive_strategy_gradients,
                align_predictive=cfg.align_predictor_with_task_gradient,
            )
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
        context_target = (
            reward+cfg.gamma*(~done).float()
            *next_context_desirability.detach())
        context_target_variance = torch.where(
            done,
            torch.full_like(context_target, cfg.terminal_outcome_variance),
            cfg.gamma**2*next_context_outcome_logvar.detach().exp())
        outcome_residual = target-strategy["desirability"]
        augmented_critic_loss = 0.5*(
            (outcome_residual.square()+target_variance)*torch.exp(
                -strategy["outcome_logvar"])
            +strategy["outcome_logvar"]).mean()
        context_residual = (
            context_target-strategy["context_desirability"])
        context_critic_loss = 0.5*(
            (context_residual.square()+context_target_variance)*torch.exp(
                -strategy["context_outcome_logvar"])
            +strategy["context_outcome_logvar"]).mean()
        # Each independent critic owns its Bellman recursion.  Averaging keeps
        # the combined optimizer step near the original critic-loss scale.
        critic_loss = 0.5*(augmented_critic_loss+context_critic_loss)
        system.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward(); system.critic_optimizer.step()

        system.feedback = new_feedback.detach()
        # Completed online worlds are now ready for the reset observation that
        # will be encoded exactly once at the top of the next decision.
        system.reset(done)
        if system.target_encoder is not None and bool(done.any()):
            with torch.no_grad():
                system.target_encoder.encode(
                    next_observation,
                    next_reward_context,
                    update_mask=done,
                )
        observation = next_observation
        reward_context = next_reward_context
        batch = cfg.worlds
        completed += batch; window_steps += batch
        timed_out = int((done_np & ~success_np & ~wrong_np).sum())
        finished, won, lost = (int(done_np.sum()), int(success_np.sum()),
                               int(wrong_np.sum()))
        episodes += finished; successes += won; wrong_total += lost
        timeout_total += timed_out
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
        for name, value in (
            ("reward", float(reward.mean())),
            ("entropy", float(actor["entropy"].detach().mean())),
            ("desirability", float(
                strategy["desirability"].detach().mean())),
            ("outcome_std", float(
                (0.5*strategy["outcome_logvar"].detach()).exp().mean())),
            ("context_desirability", float(
                strategy["context_desirability"].detach().mean())),
            ("context_outcome_std", float(
                (0.5*strategy[
                    "context_outcome_logvar"].detach()).exp().mean())),
            ("outcome_residual", float(
                outcome_residual.detach().abs().mean())),
            ("context_outcome_residual", float(
                context_residual.detach().abs().mean())),
            ("outcome_nll", float(critic_loss.detach())),
            ("augmented_outcome_nll", float(
                augmented_critic_loss.detach())),
            ("context_outcome_nll", float(context_critic_loss.detach())),
            ("critic_actor_value_delta", float((
                strategy["desirability"].detach()
                -strategy["context_desirability"].detach()).abs().mean())),
            ("critic_actor_logvar_delta", float((
                strategy["outcome_logvar"].detach()
                -strategy["context_outcome_logvar"].detach()).abs().mean())),
            ("critic_actor_weight_norm", float(
                system.strategizer.outcome_head.weight.detach()[:,
                    2*cfg.hidden_dim+cfg.strategy_dim:].norm())),
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
            ("prediction_strategy_signal", prediction_strategy_signal_norm),
            ("predictor_strategy_gradient",
             system.strategy_eprop.last_predictive_gradient_norm),
            ("predictor_strategy_aligned",
             system.strategy_eprop.last_predictive_aligned_norm),
            ("predictor_strategy_task_cosine",
             system.strategy_eprop.last_predictive_task_cosine),

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
            ("encoder_core_eligibility",
             system.strategy_encoder_eprop.last_core_eligibility_norm
             if system.strategy_encoder_eprop is not None else 0.0),
            ("encoder_core_step",
             system.strategy_encoder_eprop.last_core_step_norm
             if system.strategy_encoder_eprop is not None else 0.0),
            ("visual_projection_step",
             system.strategy_encoder_eprop.last_visual_step_norm
             if system.strategy_encoder_eprop is not None else 0.0),
            ("predictor_mediated_encoder_gradient",
             system.strategy_encoder_eprop.last_predictive_gradient_norm
             if system.strategy_encoder_eprop is not None else 0.0),
            ("predictor_mediated_encoder_aligned",
             system.strategy_encoder_eprop.last_predictive_aligned_norm
             if system.strategy_encoder_eprop is not None else 0.0),
            ("predictor_encoder_task_cosine",
             system.strategy_encoder_eprop.last_predictive_task_cosine
             if system.strategy_encoder_eprop is not None else 0.0),

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

        # Navigation versus cue-use diagnostics. These are observational only
        # and never participate in the training graph.
        add_masked(
            "stem_navigation_probability",
            stem_navigation_probability,
            stem_mask,
        )
        add_masked(
            "stem_navigation_selected",
            stem_navigation_selected,
            stem_mask,
        )
        add_masked(
            "junction_goal_probability",
            junction_goal_probability,
            junction_mask,
        )
        add_masked(
            "junction_goal_selected",
            junction_goal_selected,
            junction_mask,
        )
        add_masked(
            "junction_correct_turn_probability",
            correct_turn_probability,
            canonical_junction_mask,
        )
        add_masked(
            "junction_correct_turn_selected",
            correct_turn_selected,
            canonical_junction_mask,
        )
        add_masked(
            "junction_correct_turn_greedy",
            correct_turn_greedy,
            canonical_junction_mask,
        )
        add_masked(
            "junction_blocked_probability",
            junction_blocked_probability,
            canonical_junction_mask,
        )
        add_masked(
            "junction_blocked_selected",
            junction_blocked_selected,
            canonical_junction_mask,
        )
        add_masked(
            "blocked_forward_probability",
            blocked_forward_probability,
            forward_blocked,
        )
        add_masked(
            "blocked_forward_selected",
            blocked_forward_selected,
            forward_blocked,
        )
        add_masked(
            "junction_cue_flip_tv",
            cue_flip_tv,
            paired_junction_mask,
        )
        add_masked(
            "junction_cue_flip_correct_drop",
            cue_flip_correct_drop,
            paired_junction_mask,
        )

        outcome_std_per_world = (
            0.5*strategy["outcome_logvar"].detach()).exp()
        outcome_masks = {
            "nonterminal": nonterminal_mask,
            "success": success_mask,
            "wrong": wrong_mask,
            "timeout": timeout_mask,
        }
        for outcome_name, outcome_mask in outcome_masks.items():
            add_masked(f"credit_{outcome_name}_td", td, outcome_mask)
            add_masked(
                f"credit_{outcome_name}_td_abs", td.abs(), outcome_mask)
            add_masked(
                f"credit_{outcome_name}_actor_trace",
                actor_trace_per_world,
                outcome_mask,
            )
            add_masked(
                f"credit_{outcome_name}_value",
                strategy["desirability"],
                outcome_mask,
            )
            add_masked(
                f"credit_{outcome_name}_outcome_std",
                outcome_std_per_world,
                outcome_mask,
            )

        for phase_name, phase_mask in delay_masks.items():
            add_masked(f"phase_{phase_name}_td_abs", td.abs(), phase_mask)
            add_masked(
                f"phase_{phase_name}_actor_trace",
                actor_trace_per_world,
                phase_mask,
            )
            add_masked(
                f"phase_{phase_name}_value",
                strategy["desirability"],
                phase_mask,
            )
            add_masked(
                f"phase_{phase_name}_outcome_std",
                outcome_std_per_world,
                phase_mask,
            )

        if window_steps >= cfg.report_every:
            decisions = window_steps/cfg.worlds

            def window_mean(name: str) -> float:
                return (
                    sums.get(f"{name}_sum", 0.0)
                    /max(sums.get(f"{name}_count", 0.0), 1.0)
                )

            strategy_decode = centroid_accuracy(
                strategy_values, strategy_labels)

            strategy_delay_decode = {
                delay_name: centroid_accuracy(
                    strategy_delay_values[delay_name],
                    cue_delay_labels[delay_name],
                )
                for delay_name in strategy_delay_values
            }

            strategy_delay_strength = {
                delay_name: cue_strength(
                    strategy_delay_values[delay_name],
                    cue_delay_labels[delay_name],
                )
                for delay_name in strategy_delay_values
            }

            encoder_delay_decode = {
                delay_name: centroid_accuracy(
                    encoder_delay_values[delay_name],
                    cue_delay_labels[delay_name],
                )
                for delay_name in encoder_delay_values
            }

            encoder_delay_strength = {
                delay_name: cue_strength(
                    encoder_delay_values[delay_name],
                    cue_delay_labels[delay_name],
                )
                for delay_name in encoder_delay_values
            }

            cue_timecourse = []
            for delay_name in strategy_delay_values:
                row = {
                    "transitions": completed,
                    "episode_phase": delay_name,
                    "samples": sum(
                        labels.shape[0]
                        for labels in cue_delay_labels[delay_name]),
                    "encoder_decode": encoder_delay_decode[delay_name],
                    "encoder_distance": encoder_delay_strength[delay_name][0],
                    "encoder_snr": encoder_delay_strength[delay_name][1],
                    "strategy_decode": strategy_delay_decode[delay_name],
                    "strategy_distance": strategy_delay_strength[delay_name][0],
                    "strategy_snr": strategy_delay_strength[delay_name][1],
                }
                cue_timecourse.append(row)
                cue_timecourse_history.append(dict(row))

            policy_diagnostics = {
                "transitions": completed,
                "stem_samples": int(sums.get(
                    "stem_navigation_probability_count", 0.0)),
                "stem_navigation_probability": window_mean(
                    "stem_navigation_probability"),
                "stem_navigation_selected": window_mean(
                    "stem_navigation_selected"),
                "junction_samples": int(sums.get(
                    "junction_goal_probability_count", 0.0)),
                "junction_goal_probability": window_mean(
                    "junction_goal_probability"),
                "junction_goal_selected": window_mean(
                    "junction_goal_selected"),
                "canonical_junction_samples": int(sums.get(
                    "junction_correct_turn_probability_count", 0.0)),
                "junction_correct_turn_probability": window_mean(
                    "junction_correct_turn_probability"),
                "junction_correct_turn_selected": window_mean(
                    "junction_correct_turn_selected"),
                "junction_correct_turn_greedy": window_mean(
                    "junction_correct_turn_greedy"),
                "junction_blocked_probability": window_mean(
                    "junction_blocked_probability"),
                "junction_blocked_selected": window_mean(
                    "junction_blocked_selected"),
                "blocked_forward_probability": window_mean(
                    "blocked_forward_probability"),
                "blocked_forward_selected": window_mean(
                    "blocked_forward_selected"),
                "cue_flip_pairs": int(sums.get(
                    "junction_cue_flip_tv_count", 0.0)),
                "cue_flip_tv": window_mean("junction_cue_flip_tv"),
                "cue_flip_correct_drop": window_mean(
                    "junction_cue_flip_correct_drop"),
            }
            policy_diagnostic_history.append(dict(policy_diagnostics))

            outcome_credit = {}
            for outcome_name in (
                    "nonterminal", "success", "wrong", "timeout"):
                credit = {
                    "samples": int(sums.get(
                        f"credit_{outcome_name}_td_count", 0.0)),
                    "td": window_mean(f"credit_{outcome_name}_td"),
                    "td_abs": window_mean(
                        f"credit_{outcome_name}_td_abs"),
                    "actor_trace": window_mean(
                        f"credit_{outcome_name}_actor_trace"),
                    "value": window_mean(
                        f"credit_{outcome_name}_value"),
                    "outcome_std": window_mean(
                        f"credit_{outcome_name}_outcome_std"),
                }
                outcome_credit[outcome_name] = credit
                credit_diagnostic_history.append({
                    "transitions": completed,
                    "axis": "outcome",
                    "group": outcome_name,
                    **credit,
                })

            phase_credit = {}
            for phase_name in delay_masks:
                credit = {
                    "samples": int(sums.get(
                        f"phase_{phase_name}_td_abs_count", 0.0)),
                    "td_abs": window_mean(f"phase_{phase_name}_td_abs"),
                    "actor_trace": window_mean(
                        f"phase_{phase_name}_actor_trace"),
                    "value": window_mean(f"phase_{phase_name}_value"),
                    "outcome_std": window_mean(
                        f"phase_{phase_name}_outcome_std"),
                }
                phase_credit[phase_name] = credit
                credit_diagnostic_history.append({
                    "transitions": completed,
                    "axis": "episode_phase",
                    "group": phase_name,
                    "td": 0.0,
                    **credit,
                })

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

            for representation_name, delay_decode, delay_strength in (
                    ("encoder", encoder_delay_decode, encoder_delay_strength),
                    ("strategy", strategy_delay_decode,
                     strategy_delay_strength)):
                print(
                    f"{representation_name}_cue_timecourse",
                    "decode=["
                    f"cue:{delay_decode['cue']:.3f},"
                    f"d1_4:{delay_decode['delay_1_4']:.3f},"
                    f"d5_8:{delay_decode['delay_5_8']:.3f},"
                    f"d9+:{delay_decode['delay_9_plus']:.3f}]",
                    "snr=["
                    f"cue:{delay_strength['cue'][1]:.3f},"
                    f"d1_4:{delay_strength['delay_1_4'][1]:.3f},"
                    f"d5_8:{delay_strength['delay_5_8'][1]:.3f},"
                    f"d9+:{delay_strength['delay_9_plus'][1]:.3f}]",
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
            progress = {
                "condition": condition,
                "seed": seed,
                "transitions": completed,
                "fraction": min(completed/max(cfg.transitions, 1), 1.0),
                "episodes": episodes,
                "successes": successes,
                "curriculum_stage": env.curriculum_stage+1,
                "curriculum_stages": len(env.start_rows),
                "window_success": (
                    window_successes/max(window_episodes, 1)),
                "window_wrong": window_wrong/max(window_episodes, 1),
                "window_timeout": window_timeouts/max(window_episodes, 1),
                "reward": sums["reward"]/decisions,
                "entropy": sums["entropy"]/decisions,
                "td_abs": sums["td_abs"]/decisions,
                "desirability": sums["desirability"]/decisions,
                "outcome_std": sums["outcome_std"]/decisions,
                "context_desirability": (
                    sums["context_desirability"]/decisions),
                "context_outcome_std": (
                    sums["context_outcome_std"]/decisions),
                "critic_actor_value_delta": (
                    sums["critic_actor_value_delta"]/decisions),
                "critic_actor_logvar_delta": (
                    sums["critic_actor_logvar_delta"]/decisions),
                "critic_actor_weight_norm": (
                    sums["critic_actor_weight_norm"]/decisions),
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

                "prediction_strategy_signal": (
                    sums["prediction_strategy_signal"] / decisions),
                "predictor_strategy_gradient": (
                    sums["predictor_strategy_gradient"] / decisions),
                "predictor_strategy_aligned": (
                    sums["predictor_strategy_aligned"] / decisions),
                "predictor_strategy_task_cosine": (
                    sums["predictor_strategy_task_cosine"] / decisions),
                "predictor_mediated_encoder_gradient": (
                    sums["predictor_mediated_encoder_gradient"] / decisions),
                "predictor_mediated_encoder_aligned": (
                    sums["predictor_mediated_encoder_aligned"] / decisions),
                "predictor_encoder_task_cosine": (
                    sums["predictor_encoder_task_cosine"] / decisions),

                "reward_latent_shift": (
                    sums["reward_latent_shift"]/decisions),
                "cue_decode": strategy_decode,
                "encoder_cue_decode": encoder_decode,
                "predictor_cue_decode": predictor_decode,
                "visible_encoder_cue_decode": visible_encoder_decode,
                "visible_encoder_cue_distance": visible_encoder_distance,
                "visible_encoder_snr": visible_encoder_snr,
                "cue_timecourse": cue_timecourse,
                "policy_diagnostics": policy_diagnostics,
                "outcome_credit_diagnostics": outcome_credit,
                "phase_credit_diagnostics": phase_credit,

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
            print(
                "policy_diagnostics",
                "stem=["
                f"n:{policy_diagnostics['stem_samples']},"
                f"p_nav:{policy_diagnostics['stem_navigation_probability']:.3f},"
                f"chosen:{policy_diagnostics['stem_navigation_selected']:.3f}]",
                "junction=["
                f"n:{policy_diagnostics['junction_samples']},"
                f"p_goal:{policy_diagnostics['junction_goal_probability']:.3f},"
                f"chosen:{policy_diagnostics['junction_goal_selected']:.3f}]",
                "canonical_turn=["
                f"n:{policy_diagnostics['canonical_junction_samples']},"
                f"p_correct:{policy_diagnostics['junction_correct_turn_probability']:.3f},"
                f"chosen:{policy_diagnostics['junction_correct_turn_selected']:.3f},"
                f"greedy:{policy_diagnostics['junction_correct_turn_greedy']:.3f},"
                f"p_blocked:{policy_diagnostics['junction_blocked_probability']:.3f},"
                f"blocked:{policy_diagnostics['junction_blocked_selected']:.3f}]",
                "cue_flip=["
                f"pairs:{policy_diagnostics['cue_flip_pairs']},"
                f"tv:{policy_diagnostics['cue_flip_tv']:.3f},"
                f"correct_drop:{policy_diagnostics['cue_flip_correct_drop']:+.3f}]",
                "all_blocked_forward=["
                f"p:{policy_diagnostics['blocked_forward_probability']:.3f},"
                f"chosen:{policy_diagnostics['blocked_forward_selected']:.3f}]",
            )
            print(
                "credit_by_outcome",
                "td_abs=["
                +",".join(
                    f"{name}:{outcome_credit[name]['td_abs']:.3f}"
                    for name in outcome_credit)
                +"]",
                "actor_trace=["
                +",".join(
                    f"{name}:{outcome_credit[name]['actor_trace']:.2f}"
                    for name in outcome_credit)
                +"]",
                "value=["
                +",".join(
                    f"{name}:{outcome_credit[name]['value']:+.3f}"
                    for name in outcome_credit)
                +"]",
                "std=["
                +",".join(
                    f"{name}:{outcome_credit[name]['outcome_std']:.3f}"
                    for name in outcome_credit)
                +"]",
            )
            print(
                "credit_by_phase",
                "td_abs=["
                +",".join(
                    f"{name}:{phase_credit[name]['td_abs']:.3f}"
                    for name in phase_credit)
                +"]",
                "actor_trace=["
                +",".join(
                    f"{name}:{phase_credit[name]['actor_trace']:.2f}"
                    for name in phase_credit)
                +"]",
                "value=["
                +",".join(
                    f"{name}:{phase_credit[name]['value']:+.3f}"
                    for name in phase_credit)
                +"]",
                "std=["
                +",".join(
                    f"{name}:{phase_credit[name]['outcome_std']:.3f}"
                    for name in phase_credit)
                +"]",
            )
            if progress_callback is not None:
                progress_callback(progress)
            print(
                f"condition={condition:21s} seed={seed} transitions={completed} "
                f"episodes={episodes} successes={successes} "
                f"curriculum={env.curriculum_stage+1}/{len(env.start_rows)} "
                f"start_y={env.start_rows[env.curriculum_stage]} "
                f"stage_limit={env.episode_limits[env.curriculum_stage]} "
                f"cue_success=[{curriculum_rates[0]:.2f},"
                f"{curriculum_rates[1]:.2f}] "
                f"window_rate={window_successes/max(window_episodes,1):.3f} "
                f"wrong_rate={window_wrong/max(window_episodes,1):.3f} "
                f"timeout_rate={window_timeouts/max(window_episodes,1):.3f} "
                f"reward={sums['reward']/decisions:+.4f} "
                f"entropy={sums['entropy']/decisions:.3f} "
                f"td={sums['td_abs']/decisions:.3f} "
                f"desire={sums['desirability']/decisions:+.3f} "
                f"outcome_std={sums['outcome_std']/decisions:.3f} "
                f"outcome_nll={sums['outcome_nll']/decisions:+.4f} "
                f"critic_views=[context:"
                f"{sums['context_desirability']/decisions:+.3f},"
                f"context_std:{sums['context_outcome_std']/decisions:.3f},"
                f"value_delta:"
                f"{sums['critic_actor_value_delta']/decisions:.3f},"
                f"logvar_delta:"
                f"{sums['critic_actor_logvar_delta']/decisions:.3f},"
                f"context_mae:"
                f"{sums['context_outcome_residual']/decisions:.3f},"
                f"full_mae:{sums['outcome_residual']/decisions:.3f},"
                f"context_nll:"
                f"{sums['context_outcome_nll']/decisions:+.4f},"
                f"full_nll:"
                f"{sums['augmented_outcome_nll']/decisions:+.4f},"
                f"actor_w:"
                f"{sums['critic_actor_weight_norm']/decisions:.3f}] "
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

                f"predictor_mediation=[signal:"
                f"{sums['prediction_strategy_signal']/decisions:.3e},"
                f"strategy_grad:"
                f"{sums['predictor_strategy_gradient']/decisions:.3e},"
                f"strategy_aligned:"
                f"{sums['predictor_strategy_aligned']/decisions:.3e},"
                f"strategy_cos:"
                f"{sums['predictor_strategy_task_cosine']/decisions:+.3f},"
                f"encoder_grad:"
                f"{sums['predictor_mediated_encoder_gradient']/decisions:.3e},"
                f"encoder_aligned:"
                f"{sums['predictor_mediated_encoder_aligned']/decisions:.3e},"
                f"encoder_cos:"
                f"{sums['predictor_encoder_task_cosine']/decisions:+.3f}] "

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
                f"step:{sums['strategy_encoder_step']/decisions:.3e},"
                f"core_elig:"
                f"{sums['encoder_core_eligibility']/decisions:.2f},"
                f"core_step:{sums['encoder_core_step']/decisions:.3e},"
                f"visual_step:"
                f"{sums['visual_projection_step']/decisions:.3e}] "



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
            window_timeouts = 0
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

            for values in encoder_delay_values.values():
                values.clear()

            for labels in cue_delay_labels.values():
                labels.clear()

    result = {"condition": condition, "seed": seed,
              "episodes": episodes, "successes": successes,
              "rate": successes/max(episodes, 1),
              "wrong": wrong_total/max(episodes, 1),
              "timeout": timeout_total/max(episodes, 1),
              "cue_timecourse": cue_timecourse_history,
              "policy_diagnostics": policy_diagnostic_history,
              "credit_diagnostics": credit_diagnostic_history,
              "system": system}
    return result


@torch.no_grad()
def evaluate(system: System, cfg: Config, seed: int, intervention: str):
    # Curriculum is a training aid only.  Every intervention is evaluated
    # from the original, maximally delayed start state.
    env = BatchedTMaze(cfg, seed+80_000, curriculum=False)
    system.strategizer.core.initial(cfg.worlds, system.device)
    system.predictor.core.initial(cfg.worlds, system.device)
    system.actor.core.initial(cfg.worlds, system.device)
    system.encoder.reset()
    system.feedback.zero_()
    system.strategy_memory.zero_(); system.desirability_memory.zero_()
    system.outcome_logvar_memory.zero_()
    episodes = successes = wrong = 0
    diagnostic_sums: Dict[str, float] = {}

    def add_evaluation_masked(
            name: str,
            values: torch.Tensor,
            mask: torch.Tensor,
        ) -> None:
        diagnostic_sums[f"{name}_sum"] = (
            diagnostic_sums.get(f"{name}_sum", 0.0)
            +float(values[mask].sum()))
        diagnostic_sums[f"{name}_count"] = (
            diagnostic_sums.get(f"{name}_count", 0.0)
            +float(mask.sum()))

    observation = torch.tensor(env.observation(), device=system.device)
    latent, _, _ = system.encoder(observation)

    while episodes < cfg.evaluation_episodes:
        strategy = system.strategizer(
            latent.detach(), system.feedback.detach(), deterministic=False,
            previous_strategy=(system.strategy_memory
                if cfg.learned_strategy_memory else None))
        context_desirability, context_outcome_logvar = (
            system.strategizer.evaluate_context(
                strategy["feature"], strategy["strategy"]))
        strategy["context_desirability"] = context_desirability
        strategy["context_outcome_logvar"] = context_outcome_logvar
        if system.condition == "actor_only":
            actor_strategy = torch.zeros_like(strategy["strategy"])
            actor_desirability = torch.zeros_like(context_desirability)
            actor_outcome_logvar = torch.zeros_like(context_outcome_logvar)
        elif system.condition == "separated":
            if cfg.learned_strategy_memory:
                actor_strategy = strategy["strategy"]
                actor_desirability = context_desirability
                actor_outcome_logvar = context_outcome_logvar
            else:
                keep = cfg.strategy_retention
                actor_strategy = (keep*system.strategy_memory
                                  +(1-keep)*strategy["strategy"])
                actor_desirability = (
                    keep*system.desirability_memory
                    +(1-keep)*context_desirability)
                actor_outcome_logvar = (
                    keep*system.outcome_logvar_memory
                    +(1-keep)*context_outcome_logvar)
            system.strategy_memory = actor_strategy
            system.desirability_memory = actor_desirability
            system.outcome_logvar_memory = actor_outcome_logvar
        else:
            actor_strategy = strategy["strategy"]
            actor_desirability = context_desirability
            actor_outcome_logvar = context_outcome_logvar
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
        desirability, outcome_logvar = system.strategizer.evaluate_outcome(
            strategy["feature"], strategy["strategy"], actor["feature"])
        strategy["desirability"] = desirability
        strategy["outcome_logvar"] = outcome_logvar

        evaluation_x = torch.tensor(
            env.x.copy(), device=system.device, dtype=torch.long)
        evaluation_y = torch.tensor(
            env.y.copy(), device=system.device, dtype=torch.long)
        evaluation_direction = torch.tensor(
            env.direction.copy(), device=system.device, dtype=torch.long)
        evaluation_cue = torch.tensor(
            env.cue.copy(), device=system.device, dtype=torch.long)
        evaluation_prob = actor["logits"].softmax(dim=-1)
        stem_mask = (
            (evaluation_x == env.center) & (evaluation_y > 1))
        junction_mask = (
            (evaluation_x == env.center) & (evaluation_y == 1))
        canonical_junction_mask = (
            junction_mask & (evaluation_direction == 0))
        north = torch.zeros_like(evaluation_direction)
        desired_goal_direction = torch.where(
            evaluation_cue == 0,
            torch.full_like(evaluation_direction, 3),
            torch.ones_like(evaluation_direction),
        )
        stem_probability, stem_selected = compatible_action_statistics(
            evaluation_prob,
            actor["action"],
            evaluation_direction,
            north,
        )
        junction_probability, junction_selected = (
            compatible_action_statistics(
                evaluation_prob,
                actor["action"],
                evaluation_direction,
                desired_goal_direction,
            )
        )
        correct_turn_probability = evaluation_prob.gather(
            1, evaluation_cue[:, None]).squeeze(1)
        correct_turn_selected = (
            actor["action"] == evaluation_cue).float()
        correct_turn_greedy = (
            evaluation_prob.argmax(dim=-1) == evaluation_cue).float()
        blocked_probability = evaluation_prob[:, 2]
        blocked_selected = (actor["action"] == 2).float()

        for name, values, mask in (
                ("stem_navigation_probability", stem_probability, stem_mask),
                ("stem_navigation_selected", stem_selected, stem_mask),
                ("junction_goal_probability", junction_probability,
                 junction_mask),
                ("junction_goal_selected", junction_selected, junction_mask),
                ("junction_correct_turn_probability",
                 correct_turn_probability, canonical_junction_mask),
                ("junction_correct_turn_selected",
                 correct_turn_selected, canonical_junction_mask),
                ("junction_correct_turn_greedy",
                 correct_turn_greedy, canonical_junction_mask),
                ("junction_blocked_probability",
                 blocked_probability, canonical_junction_mask),
                ("junction_blocked_selected",
                 blocked_selected, canonical_junction_mask)):
            add_evaluation_masked(name, values, mask)

        next_np, _, done_np, success_np, wrong_np, _, _ = env.step(
            actor["action"].cpu().numpy())
        next_observation = torch.tensor(next_np, device=system.device)
        done = torch.tensor(
            done_np, device=system.device, dtype=torch.bool)
        system.encoder.reset(done)
        next_latent, _, _ = system.encoder(next_observation)


        predicted_next_latent = system.predictor(
            latent,
            actor_strategy,
            actor_desirability,
            actor["action"],
        )

        predicted_change = (
            predicted_next_latent
            - latent
        )

        prediction_error = (
            next_latent
            - predicted_next_latent
        )

        system.feedback = torch.cat(
            (
                predicted_change,
                prediction_error,
            ),
            dim=-1,
        )


        system.reset(done, reset_encoder=False)
        observation = next_observation
        latent = next_latent
        episodes += int(done_np.sum()); successes += int(success_np.sum())
        wrong += int(wrong_np.sum())

    def evaluation_mean(name: str) -> float:
        return (
            diagnostic_sums.get(f"{name}_sum", 0.0)
            /max(diagnostic_sums.get(f"{name}_count", 0.0), 1.0)
        )

    diagnostics = {
        "stem_samples": int(diagnostic_sums.get(
            "stem_navigation_probability_count", 0.0)),
        "stem_navigation_probability": evaluation_mean(
            "stem_navigation_probability"),
        "stem_navigation_selected": evaluation_mean(
            "stem_navigation_selected"),
        "junction_samples": int(diagnostic_sums.get(
            "junction_goal_probability_count", 0.0)),
        "junction_goal_probability": evaluation_mean(
            "junction_goal_probability"),
        "junction_goal_selected": evaluation_mean(
            "junction_goal_selected"),
        "canonical_junction_samples": int(diagnostic_sums.get(
            "junction_correct_turn_probability_count", 0.0)),
        "junction_correct_turn_probability": evaluation_mean(
            "junction_correct_turn_probability"),
        "junction_correct_turn_selected": evaluation_mean(
            "junction_correct_turn_selected"),
        "junction_correct_turn_greedy": evaluation_mean(
            "junction_correct_turn_greedy"),
        "junction_blocked_probability": evaluation_mean(
            "junction_blocked_probability"),
        "junction_blocked_selected": evaluation_mean(
            "junction_blocked_selected"),
    }
    return (
        successes/max(episodes, 1),
        wrong/max(episodes, 1),
        diagnostics,
    )


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
        "predictor_strategy_weight": number(
            defaults.predictor_strategy_weight,
            "Predictor-mediated strategizer weight",
            step=1e-3, start=0.0),
        "predictor_mediated_encoder_weight": number(
            defaults.predictor_mediated_encoder_weight,
            "Predictor-mediated encoder weight",
            step=1e-4, start=0.0),
        "align_predictor_with_task_gradient": mo.ui.switch(
            value=defaults.align_predictor_with_task_gradient,
            label="Project conflicting predictor gradients"),
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
        print(f"start_curriculum={preview_env.start_rows} "
              f"stage_limits={preview_env.episode_limits} "
              f"advance=min_cue_success>="
              f"{cfg.curriculum_success_threshold:.2f} "
              f"after>={cfg.curriculum_min_episodes_per_cue}"
              f"_episodes_per_cue "
              f"evaluation_start_y={preview_env.start_rows[-1]}")
        print(f"encoder={'STATEFUL_SNN' if cfg.encoder_persistent else 'STATELESS_SNN'} "
              f"predictor=STATEFUL_SNN "
              f"strategizer=STATEFUL_EXCEPT_CONTROL "
              f"actor=STATELESS_EXCEPT_CONTROL ticks={cfg.snn_ticks} "
              f"temporal_bptt=False")
        print(
            "predictor_mediation="
            f"strategy:{cfg.predictor_strategy_weight:g},"
            f"encoder:{cfg.predictor_mediated_encoder_weight:g},"
            f"conflict_projection:{cfg.align_predictor_with_task_gradient} "
            f"direct_predictor_encoder:{not cfg.detach_predictor_from_encoder}")
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
                    eval_zero=zero[0], eval_wrong=real[1],
                    eval_diagnostics=real[2],
                    eval_shuffle_diagnostics=shuffled[2],
                    eval_zero_diagnostics=zero[2])
                print(
                    f"evaluation condition={condition} seed={seed} "
                    f"train={result['successes']}/{result['episodes']} "
                    f"real={real[0]:.3f} shuffled={shuffled[0]:.3f} "
                    f"zero={zero[0]:.3f} wrong={real[1]:.3f}")
                print(
                    "evaluation_policy_diagnostics",
                    f"condition={condition}",
                    f"seed={seed}",
                    "stem=["
                    f"n:{real[2]['stem_samples']},"
                    f"p_nav:{real[2]['stem_navigation_probability']:.3f},"
                    f"chosen:{real[2]['stem_navigation_selected']:.3f}]",
                    "junction=["
                    f"n:{real[2]['junction_samples']},"
                    f"p_goal:{real[2]['junction_goal_probability']:.3f},"
                    f"chosen:{real[2]['junction_goal_selected']:.3f}]",
                    "canonical_turn=["
                    f"n:{real[2]['canonical_junction_samples']},"
                    f"p_correct:{real[2]['junction_correct_turn_probability']:.3f},"
                    f"chosen:{real[2]['junction_correct_turn_selected']:.3f},"
                    f"greedy:{real[2]['junction_correct_turn_greedy']:.3f},"
                    f"p_blocked:{real[2]['junction_blocked_probability']:.3f}]",
                )
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
        "eval stem navigation": round(
            row["eval_diagnostics"]["stem_navigation_probability"], 4),
        "eval junction goal": round(
            row["eval_diagnostics"]["junction_goal_probability"], 4),
        "eval correct turn": round(
            row["eval_diagnostics"][
                "junction_correct_turn_probability"], 4),
        "eval blocked at junction": round(
            row["eval_diagnostics"]["junction_blocked_probability"], 4),
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
    cue_timecourse_rows = [{
        "condition": result["condition"],
        "seed": result["seed"],
        "transitions": row["transitions"],
        "episode phase": row["episode_phase"],
        "samples": row["samples"],
        "encoder decode": round(row["encoder_decode"], 4),
        "encoder SNR": round(row["encoder_snr"], 4),
        "strategy decode": round(row["strategy_decode"], 4),
        "strategy SNR": round(row["strategy_snr"], 4),
    } for result in results for row in result["cue_timecourse"]]
    policy_diagnostic_rows = [{
        "condition": result["condition"],
        "seed": result["seed"],
        **{
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in row.items()
        },
    } for result in results for row in result["policy_diagnostics"]]
    credit_diagnostic_rows = [{
        "condition": result["condition"],
        "seed": result["seed"],
        **{
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in row.items()
        },
    } for result in results for row in result["credit_diagnostics"]]
    run_log = log_buffer.getvalue()
    return (
        aggregate_display,
        checkpoint_path,
        credit_diagnostic_rows,
        cue_timecourse_rows,
        policy_diagnostic_rows,
        result_rows,
        run_log,
    )


@app.cell(hide_code=True)
def _(
    aggregate_display,
    checkpoint_path,
    credit_diagnostic_rows,
    cue_timecourse_rows,
    mo,
    policy_diagnostic_rows,
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
        mo.md("### Cue information over episode and training time"),
        mo.md(
            "Each reporting window measures cue decoding and normalized "
            "left/right separation in both representations. Filter by seed "
            "and condition, then compare episode phases across transitions."),
        mo.ui.table(cue_timecourse_rows),
        mo.md("### Navigation and cue-conditioned action diagnostics"),
        mo.md(
            "Stem navigation isolates progress toward the junction. Junction "
            "metrics isolate correct branch selection; cue-flip metrics swap "
            "only the strategy between matched opposite-cue worlds."),
        mo.ui.table(policy_diagnostic_rows),
        mo.md("### Credit assignment by outcome and episode phase"),
        mo.md(
            "These rows separate TD magnitude, actor eligibility, critic "
            "value, and uncertainty by terminal outcome and temporal phase."),
        mo.ui.table(credit_diagnostic_rows),
        mo.md("### Training log"),
        mo.md(f"```text\n{run_log}\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
