import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import copy
    from collections import deque
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from typing import List, NamedTuple, Sequence

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return (
        F, List, NamedTuple, Path, Sequence, asdict, copy, dataclass, deque,
        mo, nn, np, plt, torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configuration
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Config:
        seed: int = 11
        device: str = "cpu"
        checkpoint_path: str = "Ghost/ghost_v3_checkpoint.pt"
        worlds: int = 24
        transitions: int = 163_840
        observation_dim: int = 3
        latent_dim: int = 16
        ghost_dim: int = 8
        hidden_dim: int = 40
        conditioning_dim: int = 24
        snn_ticks: int = 5
        membrane_decay: float = 0.90
        trajectory_predictor_membrane_decay: float = 0.97
        trajectory_critic_membrane_decay: float = 0.97
        surrogate_scale: float = 0.30

        gamma: float = 0.99
        encoder_lr: float = 3e-4
        trajectory_predictor_lr: float = 3e-4
        trajectory_critic_lr: float = 3e-4
        ghost_lr: float = 1e-4
        encoder_gradient_clip: float = 1.0
        trajectory_predictor_gradient_clip: float = 1.0
        trajectory_critic_gradient_clip: float = 1.0
        ghost_score_gradient_clip: float = 1.0
        ghost_step_max_norm: float = 0.02
        trajectory_critic_target_tau: float = 0.005
        encoder_target_tau: float = 0.005
        trajectory_critic_encoder_weight: float = 0.20
        external_intent_weight: float = 1.0
        trajectory_feasibility_weight: float = 0.25
        tracking_action_weight: float = 0.25
        external_horizon_weight: float = 0.25
        feasibility_horizon_weight: float = 0.15
        ghost_score_warmup_transitions: int = 8_192
        horizon_entropy_weight: float = 0.002
        trajectory_feasibility_gradient_clip: float = 1.0
        tracking_gradient_clip: float = 1.0
        tracking_action_cost: float = 0.002
        tracking_baseline_decay: float = 0.95
        external_horizon_baseline_decay: float = 0.95
        feasibility_horizon_baseline_decay: float = 0.95
        minimum_waypoint_displacement: float = 0.01

        initial_exploration_std: float = 0.25
        minimum_exploration_std: float = 0.08
        exploration_decay_transitions: int = 120_000

        uncertainty_penalty: float = 0.15
        trajectory_replacement_margin: float = 0.02
        trajectory_switch_penalty: float = 0.01
        trajectory_max_horizon: int = 16
        spline_control_points: int = 6
        spline_degree: int = 3
        spline_max_control_step: float = 0.15
        minimum_spline_arc_length: float = 0.02
        spline_curvature_regularization: float = 0.0

        encoder_latent_decay: float = 0.80
        encoder_membrane_readout: float = 0.05
        predictor_logvar_min: float = -5.0
        predictor_logvar_max: float = 2.0
        horizon_uniform_eta: float = 0.10
        actual_horizon_min_probability: float = 0.02

        step_scale: float = 0.05
        target_radius: float = 0.02
        progress_weight: float = 2.0
        action_cost: float = 0.002
        overshoot_penalty: float = 0.05
        curriculum_stage: int = 0
        curriculum_threshold: float = 0.65
        curriculum_window_episodes: int = 128
        curriculum_distance_ranges: tuple = (
            (0.03, 0.10), (0.08, 0.35), (0.15, 1.00), (0.20, 2.00))
        curriculum_hold_steps: tuple = (1, 2, 2, 3)
        curriculum_episode_limits: tuple = (8, 24, 40, 64)

        plot_every: int = 1_024
        plot_window: int = 128

    RELATIVE_HORIZONS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
    CHECKPOINT_FORMAT_VERSION = 7
    return CHECKPOINT_FORMAT_VERSION, Config, RELATIVE_HORIZONS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Shared neural utilities
    """)
    return


@app.cell
def _(Config, F, Sequence, nn, torch):
    class SurrogateSpike(torch.autograd.Function):
        @staticmethod
        def forward(ctx, voltage: torch.Tensor, scale: float):
            ctx.save_for_backward(voltage)
            ctx.scale = scale
            return (voltage >= 0).to(voltage.dtype)

        @staticmethod
        def backward(ctx, gradient: torch.Tensor):
            (voltage,) = ctx.saved_tensors
            return (
                gradient * ctx.scale
                * torch.clamp(1 - voltage.abs(), min=0),
                None,
            )

    class RecurrentSNN(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            cfg: Config,
            persistent: bool,
            decay: float | None = None,
            record_eligibility: bool = False,
        ) -> None:
            super().__init__()
            self.cfg = cfg
            self.hidden_dim = hidden_dim
            self.persistent = persistent
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
                self.mem.zero_()
                self.spk.zero_()
            elif bool(mask.any()):
                self.mem[mask] = 0
                self.spk[mask] = 0

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            if (
                not self.persistent
                or self.mem is None
                or self.mem.shape[0] != value.shape[0]
            ):
                mem = torch.zeros(
                    value.shape[0],
                    self.hidden_dim,
                    device=value.device,
                    dtype=value.dtype,
                )
                spk = torch.zeros_like(mem)
            else:
                mem, spk = self.mem.detach(), self.spk.detach()

            features = []
            self.last_eligibility_records = []
            recurrent_mask = 1 - torch.eye(
                self.hidden_dim, device=value.device, dtype=value.dtype)
            for _ in range(self.cfg.snn_ticks):
                presynaptic_spikes = spk
                current = (
                    self.input(value)
                    + F.linear(
                        spk, self.recurrent.weight * recurrent_mask)
                    + self.bias
                )
                mem = self.decay * mem + current - spk
                spk = SurrogateSpike.apply(
                    mem - 1.0, self.cfg.surrogate_scale)
                if self.record_eligibility:
                    pseudo_derivative = (
                        self.cfg.surrogate_scale
                        * torch.clamp(
                            1 - (mem.detach() - 1.0).abs(), min=0)
                    )
                    self.last_eligibility_records.append((
                        value.detach(),
                        presynaptic_spikes.detach(),
                        pseudo_derivative,
                    ))
                features.append(torch.cat((mem, spk), -1))

            if self.persistent:
                self.mem, self.spk = mem.detach(), spk.detach()
            self.last_output = torch.stack(features).mean(0)
            return self.last_output

    def mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
            nn.Tanh(),
        )

    def zeros_like_parameters(parameters: Sequence[nn.Parameter]):
        return [torch.zeros_like(parameter) for parameter in parameters]

    def gradient_list_norm(gradients) -> torch.Tensor:
        present = [g for g in gradients if g is not None]
        if not present:
            return torch.tensor(0.0)
        return torch.stack([g.square().sum() for g in present]).sum().sqrt()

    def clip_gradient_list(gradients, max_norm: float):
        raw = gradient_list_norm(gradients)
        if float(raw) == 0.0:
            return gradients
        scale = min(1.0, float(max_norm) / max(float(raw), 1e-12))
        return [None if value is None else value * scale for value in gradients]

    def add_gradient_lists(*weighted_lists):
        if not weighted_lists:
            return []
        length = len(weighted_lists[0][1])
        result = []
        for index in range(length):
            total = None
            for weight, gradients in weighted_lists:
                value = gradients[index]
                if value is None:
                    continue
                contribution = value * weight
                total = contribution if total is None else total + contribution
            result.append(total)
        return result

    def parameter_gradients(loss, parameters, retain_graph=True):
        parameters = list(parameters)
        if loss is None:
            return [None for _ in parameters]
        return list(torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        ))

    def optimizer_step_from_gradients(
        optimizer, parameters, gradients, max_norm: float
    ) -> float:
        parameters = list(parameters)
        gradients = clip_gradient_list(gradients, max_norm)
        optimizer.zero_grad(set_to_none=True)
        before = [parameter.detach().clone() for parameter in parameters]
        for parameter, gradient in zip(parameters, gradients):
            if gradient is not None:
                parameter.grad = gradient.detach().clone()
        optimizer.step()
        if not before:
            return 0.0
        return float(torch.stack([
            (parameter.detach() - old).square().sum()
            for parameter, old in zip(parameters, before)
        ]).sum().sqrt())

    @torch.no_grad()
    def polyak_update(target, online, tau):
        for target_parameter, parameter in zip(
            target.parameters(), online.parameters()
        ):
            target_parameter.mul_(1 - tau).add_(parameter, alpha=tau)

    def slice_state(state, index: int):
        if state is None:
            return None
        return tuple(
            value[index:index + 1].detach().clone() for value in state)

    return (
        RecurrentSNN,
        SurrogateSpike,
        add_gradient_lists,
        clip_gradient_list,
        gradient_list_norm,
        mlp,
        optimizer_step_from_gradients,
        parameter_gradients,
        polyak_update,
        slice_state,
        zeros_like_parameters,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Shared latent encoder
    """)
    return


@app.cell
def _(Config, RecurrentSNN, SurrogateSpike, nn, torch):
    class Encoder(nn.Module):
        """Stateless spiking encoder shared by both predictive/control levels."""

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg
            self.core = RecurrentSNN(
                cfg.observation_dim,
                cfg.hidden_dim,
                cfg,
                persistent=False,
            )
            self.norm = nn.LayerNorm(2 * cfg.hidden_dim)
            self.latent_current = nn.Linear(
                2 * cfg.hidden_dim, cfg.latent_dim, bias=False)
            nn.init.normal_(
                self.latent_current.weight,
                std=0.12 / max((2 * cfg.hidden_dim) ** 0.5, 1.0),
            )

        def forward(self, observation: torch.Tensor):
            feature = self.norm(self.core(observation))
            raw = self.latent_current(feature)
            current = 2.0 * raw + 0.34
            membrane = torch.zeros_like(current)
            spikes = torch.zeros_like(current)
            spike_sum = torch.zeros_like(current)
            for _ in range(self.cfg.snn_ticks):
                membrane = (
                    self.cfg.encoder_latent_decay * membrane
                    + current - spikes
                )
                membrane = 12.0 * torch.tanh(membrane / 12.0)
                spikes = SurrogateSpike.apply(
                    membrane - 1.0, self.cfg.surrogate_scale)
                spike_sum += spikes
            return (
                spike_sum / self.cfg.snn_ticks
                + self.cfg.encoder_membrane_readout
                * torch.tanh(membrane)
            )

    return (Encoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Persistent recurrent Ghost
    """)
    return


@app.cell
def _(
    Config,
    List,
    RELATIVE_HORIZONS,
    RecurrentSNN,
    Sequence,
    add_gradient_lists,
    clip_gradient_list,
    gradient_list_norm,
    nn,
    torch,
):
    class Ghost(nn.Module):
        """Persistent intent, horizon, and continuous-action policy."""

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg
            # Latent, leader delta, score, uncertainty, time, valid, tracking.
            input_dim = 2 * cfg.latent_dim + 5
            self.core = RecurrentSNN(
                input_dim,
                cfg.hidden_dim,
                cfg,
                persistent=True,
                record_eligibility=True,
            )
            self.norm = nn.LayerNorm(2 * cfg.hidden_dim)
            self.intent_head = nn.Linear(
                2 * cfg.hidden_dim, cfg.ghost_dim)
            self.horizon_head = nn.Linear(
                2 * cfg.hidden_dim, len(RELATIVE_HORIZONS))
            self.action_head = nn.Sequential(
                nn.Linear(
                    2 * cfg.hidden_dim + cfg.latent_dim + 2,
                    cfg.hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, 1),
            )
            nn.init.normal_(self.intent_head.weight, std=0.02)
            nn.init.zeros_(self.intent_head.bias)
            nn.init.zeros_(self.horizon_head.bias)
            nn.init.normal_(self.action_head[-1].weight, std=0.005)
            nn.init.zeros_(self.action_head[-1].bias)

        def update_context(
            self,
            latent,
            leader_next_delta,
            leader_score,
            leader_uncertainty,
            normalized_remaining_horizon,
            leader_valid,
            previous_tracking_reward,
        ):
            value = torch.cat((
                latent,
                leader_next_delta,
                leader_score[:, None],
                leader_uncertainty[:, None],
                normalized_remaining_horizon[:, None],
                leader_valid.float()[:, None],
                previous_tracking_reward[:, None],
            ), -1)
            return self.norm(self.core(value))

        def plan_from_feature(self, feature):
            intent = torch.tanh(self.intent_head(feature))
            horizon_logits = self.horizon_head(feature)
            return intent, horizon_logits

        def action_from_feature(
            self,
            feature,
            selected_next_waypoint_delta,
            selected_remaining_horizon,
            selected_plan_valid,
        ):
            return self.action_head(torch.cat((
                feature,
                selected_next_waypoint_delta,
                selected_remaining_horizon[:, None],
                selected_plan_valid.float()[:, None],
            ), -1)).squeeze(-1)

        def reset(self, mask):
            self.core.reset(mask)

    class RecurrentGhostEprop:
        """Bellec-style local output Jacobians for the persistent LIF core."""

        def __init__(
            self,
            ghost: Ghost,
            parameters: Sequence[nn.Parameter],
            worlds: int,
        ) -> None:
            self.ghost = ghost
            self.parameters = list(parameters)
            cfg = ghost.cfg
            b, h = worlds, cfg.hidden_dim
            d = ghost.core.input.in_features
            device = next(ghost.parameters()).device
            self.epsilon_in = torch.zeros(b, h, d, device=device)
            self.epsilon_rec = torch.zeros(b, h, h, device=device)
            self.epsilon_bias = torch.zeros(b, h, device=device)
            core = ghost.core
            self.core_parameter_kind = {
                id(core.input.weight): "input_weight",
                id(core.input.bias): "input_bias",
                id(core.recurrent.weight): "recurrent_weight",
                id(core.bias): "bias",
            }
            self.other_indices = [
                index for index, parameter in enumerate(self.parameters)
                if id(parameter) not in self.core_parameter_kind
            ]

        def current_output_jacobians(
            self, output: torch.Tensor
        ) -> List[torch.Tensor]:
            cfg, core = self.ghost.cfg, self.ghost.core
            b, output_dim, h = output.shape[0], output.shape[1], cfg.hidden_dim
            current = [
                torch.zeros(
                    (b, output_dim) + tuple(parameter.shape),
                    device=output.device,
                )
                for parameter in self.parameters
            ]
            if core.last_output is None:
                raise RuntimeError("ghost core has no eligibility output")

            feature_gradient = torch.stack([
                torch.autograd.grad(
                    output[:, coordinate].sum(),
                    core.last_output,
                    retain_graph=True,
                )[0].detach()
                for coordinate in range(output_dim)
            ], 1)

            jacobian_in = torch.zeros(
                b, output_dim, h, self.epsilon_in.shape[-1],
                device=output.device,
            )
            jacobian_rec = torch.zeros(
                b, output_dim, h, h, device=output.device)
            jacobian_bias = torch.zeros(
                b, output_dim, h, device=output.device)
            recurrent_mask = 1 - torch.eye(h, device=output.device)
            ticks = max(len(core.last_eligibility_records), 1)
            with torch.no_grad():
                for value, presynaptic, pseudo_derivative in (
                    core.last_eligibility_records
                ):
                    self.epsilon_in.mul_(core.decay).add_(
                        value[:, None, :])
                    self.epsilon_rec.mul_(core.decay).add_(
                        presynaptic[:, None, :]
                        * recurrent_mask[None]
                    )
                    self.epsilon_bias.mul_(core.decay).add_(1)
                    coefficient = (
                        feature_gradient[:, :, :h]
                        + feature_gradient[:, :, h:]
                        * pseudo_derivative[:, None, :]
                    ) / ticks
                    jacobian_in.add_(
                        coefficient[:, :, :, None]
                        * self.epsilon_in[:, None, :, :]
                    )
                    jacobian_rec.add_(
                        coefficient[:, :, :, None]
                        * self.epsilon_rec[:, None, :, :]
                    )
                    jacobian_bias.add_(
                        coefficient * self.epsilon_bias[:, None, :]
                    )

            if self.other_indices:
                basis = torch.eye(
                    b, device=output.device, dtype=output.dtype)
                other_parameters = [
                    self.parameters[index] for index in self.other_indices]
                for coordinate in range(output_dim):
                    gradients = torch.autograd.grad(
                        output[:, coordinate],
                        other_parameters,
                        grad_outputs=basis,
                        is_grads_batched=True,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for index, gradient in zip(
                        self.other_indices, gradients
                    ):
                        if gradient is not None:
                            current[index][:, coordinate].copy_(
                                gradient.detach())

            for index, parameter in enumerate(self.parameters):
                kind = self.core_parameter_kind.get(id(parameter))
                if kind == "input_weight":
                    current[index].copy_(jacobian_in)
                elif kind in ("input_bias", "bias"):
                    current[index].copy_(jacobian_bias)
                elif kind == "recurrent_weight":
                    current[index].copy_(jacobian_rec)
            return current

        def reset(self, mask: torch.Tensor) -> None:
            if not bool(mask.any()):
                return
            with torch.no_grad():
                self.epsilon_in[mask] = 0
                self.epsilon_rec[mask] = 0
                self.epsilon_bias[mask] = 0

    class GhostLearner:
        """One optimizer for all independently routed maximizing directions."""

        def __init__(self, ghost: Ghost, cfg: Config):
            self.ghost = ghost
            self.cfg = cfg
            self.parameters = list(ghost.parameters())
            self.eprop = RecurrentGhostEprop(
                ghost, self.parameters, cfg.worlds)
            self.optimizer = torch.optim.Adam(
                self.parameters, lr=cfg.ghost_lr, maximize=True)

        def capture_directions(
            self,
            intent,
            raw_action_mean,
            outer_score,
            action_log_probability,
        ):
            joined = torch.cat(
                (intent, raw_action_mean[:, None]), -1)
            jacobians = self.eprop.current_output_jacobians(joined)
            intent_signal = torch.autograd.grad(
                outer_score.sum(), intent, retain_graph=True)[0].detach()
            action_signal = torch.autograd.grad(
                action_log_probability.sum(),
                raw_action_mean,
                retain_graph=True,
            )[0].detach()

            external_intent_directions = []
            intent_eligibilities = []
            action_eligibilities = []
            intent_dim = intent.shape[-1]
            for jacobian in jacobians:
                intent_view = (
                    intent_signal.shape
                    + (1,) * (jacobian.ndim - 2)
                )
                external_intent_directions.append(
                    (
                        jacobian[:, :intent_dim]
                        * intent_signal.view(intent_view)
                    ).sum(1).mean(0)
                )
                action_jacobian = jacobian[:, intent_dim]
                action_view = (
                    action_signal.shape
                    + (1,) * (action_jacobian.ndim - 1)
                )
                action_eligibilities.append(
                    action_jacobian
                    * action_signal.view(action_view)
                )
                intent_eligibilities.append(
                    jacobian[:, :intent_dim].detach().clone())
            return (
                external_intent_directions,
                intent_eligibilities,
                action_eligibilities,
            )

        @staticmethod
        def _normalize(directions, clip):
            clipped = clip_gradient_list(directions, clip)
            norm = gradient_list_norm(clipped)
            if float(norm) <= 0:
                return clipped
            return [
                None if value is None else value / norm
                for value in clipped
            ]

        def apply(
            self,
            external_intent_directions,
            feasibility_intent_directions,
            tracking_action_directions,
            external_horizon_directions,
            feasibility_horizon_directions,
            horizon_entropy_directions,
        ):
            combined = add_gradient_lists(
                (
                    self.cfg.external_intent_weight,
                    self._normalize(
                        external_intent_directions,
                        self.cfg.ghost_score_gradient_clip,
                    ),
                ),
                (
                    self.cfg.trajectory_feasibility_weight,
                    self._normalize(
                        feasibility_intent_directions,
                        self.cfg.trajectory_feasibility_gradient_clip,
                    ),
                ),
                (
                    self.cfg.tracking_action_weight,
                    self._normalize(
                        tracking_action_directions,
                        self.cfg.tracking_gradient_clip,
                    ),
                ),
                (
                    self.cfg.external_horizon_weight,
                    self._normalize(
                        external_horizon_directions,
                        self.cfg.ghost_score_gradient_clip,
                    ),
                ),
                (
                    self.cfg.feasibility_horizon_weight,
                    self._normalize(
                        feasibility_horizon_directions,
                        self.cfg.trajectory_feasibility_gradient_clip,
                    ),
                ),
                (
                    self.cfg.horizon_entropy_weight,
                    self._normalize(
                        horizon_entropy_directions,
                        self.cfg.ghost_score_gradient_clip,
                    ),
                ),
            )
            combined = clip_gradient_list(
                combined, self.cfg.ghost_score_gradient_clip)
            self.optimizer.zero_grad(set_to_none=True)
            before = [
                parameter.detach().clone()
                for parameter in self.parameters
            ]
            for parameter, direction in zip(
                self.parameters, combined
            ):
                if direction is not None:
                    parameter.grad = direction.detach().clone()
            self.optimizer.step()

            step = torch.stack([
                (parameter.detach() - old).square().sum()
                for parameter, old in zip(self.parameters, before)
            ]).sum().sqrt()
            if (
                self.cfg.ghost_step_max_norm > 0
                and float(step) > self.cfg.ghost_step_max_norm
            ):
                scale = self.cfg.ghost_step_max_norm / max(
                    float(step), 1e-12)
                with torch.no_grad():
                    for parameter, old in zip(
                        self.parameters, before
                    ):
                        parameter.copy_(
                            old + scale * (parameter - old))
                step = torch.tensor(
                    self.cfg.ghost_step_max_norm,
                    device=step.device,
                )
            return float(step)

        def reset(self, mask):
            self.eprop.reset(mask)

    def per_world_parameter_gradients(score, parameters):
        basis = torch.eye(
            len(score), device=score.device, dtype=score.dtype)
        return [
            (
                torch.zeros(
                    (len(score),) + tuple(parameter.shape),
                    device=parameter.device,
                )
                if gradient is None else gradient.detach()
            )
            for parameter, gradient in zip(
                parameters,
                torch.autograd.grad(
                    score,
                    list(parameters),
                    grad_outputs=basis,
                    is_grads_batched=True,
                    retain_graph=True,
                    allow_unused=True,
                ),
            )
        ]

    return (
        Ghost,
        GhostLearner,
        per_world_parameter_gradients,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Intent-conditioned trajectory predictor and rolling records
    """)
    return


@app.cell
def _(Config, NamedTuple, RecurrentSNN, dataclass, mlp, nn, torch):
    def bspline_basis(
        phases,
        number_of_control_points,
        degree,
        device=None,
        dtype=None,
    ):
        """Clamped-uniform B-spline basis evaluated at phases in [0, 1]."""
        if degree < 1:
            raise ValueError("spline degree must be at least one")
        if number_of_control_points < degree + 1:
            raise ValueError(
                "spline control points must be at least degree + 1")
        phases = torch.as_tensor(
            phases,
            device=device,
            dtype=torch.float32 if dtype is None else dtype,
        ).clamp(0, 1)
        interior_count = number_of_control_points - degree - 1
        interior = (
            torch.arange(
                1,
                interior_count + 1,
                device=phases.device,
                dtype=phases.dtype,
            ) / (interior_count + 1)
            if interior_count > 0
            else phases.new_empty(0)
        )
        knots = torch.cat((
            phases.new_zeros(degree + 1),
            interior,
            phases.new_ones(degree + 1),
        ))
        flattened = phases.reshape(-1)
        interval_count = len(knots) - 1
        basis = torch.stack([
            (
                (flattened >= knots[index])
                & (flattened < knots[index + 1])
            ).to(phases.dtype)
            for index in range(interval_count)
        ], -1)
        for order in range(1, degree + 1):
            next_basis = []
            for index in range(interval_count - order):
                left_width = knots[index + order] - knots[index]
                right_width = (
                    knots[index + order + 1] - knots[index + 1])
                left = (
                    (flattened - knots[index])
                    / left_width.clamp_min(torch.finfo(phases.dtype).eps)
                    * basis[:, index]
                    if float(left_width) > 0
                    else torch.zeros_like(flattened)
                )
                right = (
                    (knots[index + order + 1] - flattened)
                    / right_width.clamp_min(torch.finfo(phases.dtype).eps)
                    * basis[:, index + 1]
                    if float(right_width) > 0
                    else torch.zeros_like(flattened)
                )
                next_basis.append(left + right)
            basis = torch.stack(next_basis, -1)
        endpoint = flattened == 1
        if bool(endpoint.any()):
            basis = basis.clone()
            basis[endpoint] = 0
            basis[endpoint, -1] = 1
        return basis.reshape(
            phases.shape + (number_of_control_points,))

    def build_spline_tables(cfg, device, dtype=torch.float32):
        """Precompute padded bases, masks, and absolute phases by horizon."""
        k_max = cfg.trajectory_max_horizon
        controls = cfg.spline_control_points
        basis_table = torch.zeros(
            k_max + 1, k_max, controls,
            device=device, dtype=dtype)
        mask_table = torch.zeros(
            k_max + 1, k_max,
            device=device, dtype=torch.bool)
        phase_table = torch.zeros(
            k_max + 1, k_max,
            device=device, dtype=dtype)
        for horizon in range(1, k_max + 1):
            phases = (
                torch.arange(
                    1, horizon + 1,
                    device=device, dtype=dtype)
                / horizon
            )
            basis_table[horizon, :horizon] = bspline_basis(
                phases,
                controls,
                cfg.spline_degree,
                device,
                dtype,
            )
            mask_table[horizon, :horizon] = True
            phase_table[horizon, :horizon] = phases
        return basis_table, mask_table, phase_table

    def _remaining_spline_basis(
        horizons,
        cursors,
        basis_table,
        mask_table,
        phase_table,
    ):
        batch = horizons.shape[0]
        k_max = basis_table.shape[1]
        controls = basis_table.shape[2]
        basis = basis_table.new_zeros(batch, k_max, controls)
        mask = torch.zeros(
            batch, k_max, dtype=torch.bool, device=horizons.device)
        phases = phase_table.new_zeros(batch, k_max)
        if cursors is None:
            return (
                basis_table[horizons],
                mask_table[horizons],
                phase_table[horizons],
            )
        for world in range(batch):
            horizon = int(horizons[world])
            cursor = int(cursors[world])
            remaining = max(horizon - cursor, 0)
            if remaining:
                basis[world, :remaining] = basis_table[
                    horizon, cursor:horizon]
                mask[world, :remaining] = True
                phases[world, :remaining] = phase_table[
                    horizon, cursor:horizon]
        return basis, mask, phases

    def render_spline_positions(
        control_points,
        horizons,
        cursors=None,
        basis_table=None,
        mask_table=None,
        phase_table=None,
    ):
        if basis_table is None:
            raise ValueError("a precomputed spline basis table is required")
        basis, _mask, _phases = _remaining_spline_basis(
            horizons,
            cursors,
            basis_table,
            mask_table,
            phase_table,
        )
        return torch.einsum("bkc,bcl->bkl", basis, control_points)

    def render_spline_logvars(
        control_logvars,
        horizons,
        cursors=None,
        basis_table=None,
        mask_table=None,
        phase_table=None,
        logvar_min=None,
        logvar_max=None,
    ):
        if basis_table is None:
            raise ValueError("a precomputed spline basis table is required")
        basis, _mask, _phases = _remaining_spline_basis(
            horizons,
            cursors,
            basis_table,
            mask_table,
            phase_table,
        )
        variance = torch.einsum(
            "bkc,bcl->bkl",
            basis.square(),
            control_logvars.exp(),
        )
        rendered = variance.clamp_min(1e-8).log()
        if logvar_min is not None or logvar_max is not None:
            rendered = rendered.clamp(
                min=logvar_min, max=logvar_max)
        return rendered

    def render_remaining_spline(
        control_points,
        control_logvars,
        horizons,
        cursors,
        basis_table,
        mask_table,
        phase_table,
        logvar_min=None,
        logvar_max=None,
    ):
        basis, mask, phases = _remaining_spline_basis(
            horizons,
            cursors,
            basis_table,
            mask_table,
            phase_table,
        )
        positions = torch.einsum(
            "bkc,bcl->bkl", basis, control_points)
        variance = torch.einsum(
            "bkc,bcl->bkl",
            basis.square(),
            control_logvars.exp(),
        )
        logvars = variance.clamp_min(1e-8).log()
        if logvar_min is not None or logvar_max is not None:
            logvars = logvars.clamp(
                min=logvar_min, max=logvar_max)
        return positions, logvars, mask, phases

    class TrajectoryPredictor(nn.Module):
        """Persistent intent-conditioned latent B-spline forecast.

        The source is an online-encoder latent; factual waypoint targets use
        target-encoder latents. Live critic deltas therefore re-anchor the
        absolute forecast at the detached target-current latent.
        """

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg
            self.intent_encoder = mlp(
                cfg.ghost_dim, cfg.hidden_dim, cfg.conditioning_dim)
            self.core = RecurrentSNN(
                cfg.latent_dim + cfg.conditioning_dim + 2,
                cfg.hidden_dim,
                cfg,
                persistent=True,
                decay=cfg.trajectory_predictor_membrane_decay,
            )
            self.norm = nn.LayerNorm(2 * cfg.hidden_dim)
            output_dim = (
                (cfg.spline_control_points - 1) * cfg.latent_dim)
            self.spline_increment_head = nn.Linear(
                2 * cfg.hidden_dim, output_dim)
            self.spline_control_logvar_head = nn.Linear(
                2 * cfg.hidden_dim, output_dim)

        def forward(
            self, latent, intent, relative_horizon, budget_fraction
        ):
            condition = self.intent_encoder(intent)
            feature = self.norm(self.core(torch.cat((
                latent,
                condition,
                relative_horizon[:, None],
                budget_fraction[:, None],
            ), -1)))
            batch = latent.shape[0]
            increments = (
                self.cfg.spline_max_control_step
                * torch.tanh(
                    self.spline_increment_head(feature)
                ).reshape(
                    batch,
                    self.cfg.spline_control_points - 1,
                    self.cfg.latent_dim,
                )
            )
            controls_after_anchor = (
                latent[:, None, :] + increments.cumsum(dim=1))
            control_points = torch.cat((
                latent[:, None, :],
                controls_after_anchor,
            ), 1)
            learned_logvars = (
                self.spline_control_logvar_head(feature).reshape(
                    batch,
                    self.cfg.spline_control_points - 1,
                    self.cfg.latent_dim,
                ).clamp(
                    self.cfg.predictor_logvar_min,
                    self.cfg.predictor_logvar_max,
                )
            )
            anchor_logvar = torch.full(
                (batch, 1, self.cfg.latent_dim),
                self.cfg.predictor_logvar_min,
                device=latent.device,
                dtype=latent.dtype,
            )
            control_logvars = torch.cat((
                anchor_logvar,
                learned_logvars,
            ), 1)
            return control_points, control_logvars

        def reset(self, mask):
            self.core.reset(mask)

    @dataclass
    class LeaderTrajectory:
        episode: int
        plan_version: int
        source_observation: torch.Tensor
        source_target_latent: torch.Tensor
        intent: torch.Tensor
        control_points: torch.Tensor
        control_logvars: torch.Tensor
        selected_horizon: int
        cursor: int
        relative_horizon: torch.Tensor
        budget_fraction: torch.Tensor
        predictor_state: object
        intent_eligibility: object
        control_intent_jacobian: torch.Tensor
        horizon_gradients: object
        original_adjusted_score: float
        valid: bool

    class ReadyWaypointPrediction(NamedTuple):
        source_observation: torch.Tensor
        intent: torch.Tensor
        relative_horizon: torch.Tensor
        budget_fraction: torch.Tensor
        predictor_state: object
        selected_horizon: int
        executed_cursor: int
        selected_phase: torch.Tensor
        target: torch.Tensor

    class PendingTrajectoryTD(NamedTuple):
        episode: int
        source_observation: torch.Tensor
        source_target_latent: torch.Tensor
        waypoints: torch.Tensor
        mask: torch.Tensor
        spline_phases: torch.Tensor
        critic_state: object
        source_q: torch.Tensor
        reward: torch.Tensor
        done: bool

    class ReadyTrajectoryTD(NamedTuple):
        record: object
        target: torch.Tensor

    return (
        LeaderTrajectory,
        PendingTrajectoryTD,
        ReadyTrajectoryTD,
        ReadyWaypointPrediction,
        TrajectoryPredictor,
        bspline_basis,
        build_spline_tables,
        render_remaining_spline,
        render_spline_logvars,
        render_spline_positions,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Recurrent external trajectory critic
    """)
    return


@app.cell
def _(Config, RecurrentSNN, nn, torch):
    class TrajectoryCritic(nn.Module):
        """Persistent external value of a masked remaining trajectory."""

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.cfg = cfg
            self.core = RecurrentSNN(
                2 * cfg.latent_dim + 2,
                cfg.hidden_dim,
                cfg,
                persistent=True,
                decay=cfg.trajectory_critic_membrane_decay,
            )
            self.norm = nn.LayerNorm(2 * cfg.hidden_dim)
            self.score_head = nn.Linear(2 * cfg.hidden_dim, 1)
            nn.init.normal_(self.score_head.weight, std=1e-3)
            nn.init.zeros_(self.score_head.bias)

        def snapshot_runtime(self):
            return {
                "state": self.core.snapshot(),
                "last_output": (
                    None if self.core.last_output is None
                    else self.core.last_output.detach().clone()
                ),
            }

        def restore_runtime(self, runtime):
            self.core.restore(runtime["state"])
            self.core.last_output = (
                None
                if runtime["last_output"] is None
                else runtime["last_output"].clone()
            )
            self.core.last_eligibility_records = []

        @staticmethod
        def blend_runtime(leader, contender, choose_contender):
            def blend(left, right):
                if left is None:
                    return None if right is None else right.clone()
                if right is None:
                    return left.clone()
                view = (
                    choose_contender.shape
                    + (1,) * (left.ndim - 1)
                )
                return torch.where(
                    choose_contender.view(view), right, left)

            leader_state, contender_state = (
                leader["state"], contender["state"])
            state = None
            if leader_state is not None or contender_state is not None:
                state = tuple(blend(left, right) for left, right in zip(
                    leader_state, contender_state))
            return {
                "state": state,
                "last_output": blend(
                    leader["last_output"], contender["last_output"]),
            }

        def evaluate_trajectory(
            self, latent, deltas, mask, spline_phases
        ):
            batch, horizon, _ = deltas.shape
            last_feature = torch.zeros(
                batch,
                2 * self.cfg.hidden_dim,
                device=latent.device,
                dtype=latent.dtype,
            )
            for position in range(horizon):
                active = mask[:, position].bool()
                pre_state = self.core.snapshot()
                if pre_state is None:
                    pre_state = (
                        torch.zeros(
                            batch,
                            self.cfg.hidden_dim,
                            device=latent.device,
                            dtype=latent.dtype,
                        ),
                        torch.zeros(
                            batch,
                            self.cfg.hidden_dim,
                            device=latent.device,
                            dtype=latent.dtype,
                        ),
                    )
                remaining = (
                    mask[:, position:].sum(-1).float()
                    / max(self.cfg.trajectory_max_horizon, 1)
                )
                raw_feature = self.core(torch.cat((
                    latent,
                    deltas[:, position],
                    remaining[:, None],
                    spline_phases[:, position:position + 1],
                ), -1))
                view = active[:, None]
                self.core.mem = torch.where(
                    view, self.core.mem, pre_state[0]).detach()
                self.core.spk = torch.where(
                    view, self.core.spk, pre_state[1]).detach()
                last_feature = torch.where(
                    view, raw_feature, last_feature)
            self.core.last_output = last_feature
            return self.score_head(
                self.norm(last_feature)).squeeze(-1)

        def reset(self, mask=None):
            self.core.reset(mask)

    return (TrajectoryCritic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Environment
    """)
    return


@app.cell
def _(Config, dataclass, np):
    class PointControl:
        def __init__(self, cfg: Config, seed: int) -> None:
            self.cfg = cfg
            self.rng = np.random.default_rng(seed)
            b = cfg.worlds
            self.x = np.zeros(b, np.float32)
            self.goal = np.zeros(b, np.float32)
            self.age = np.zeros(b, np.int64)
            self.hold = np.zeros(b, np.int64)
            self.episode = np.zeros(b, np.int64)
            self.curriculum_stage = int(cfg.curriculum_stage)
            self.curriculum_stage_count = len(
                cfg.curriculum_distance_ranges)
            self.reset(np.ones(b, bool))

        @property
        def distance_range(self):
            return self.cfg.curriculum_distance_ranges[
                self.curriculum_stage]

        @property
        def hold_steps(self):
            return self.cfg.curriculum_hold_steps[
                self.curriculum_stage]

        @property
        def current_episode_limit(self):
            return self.cfg.curriculum_episode_limits[
                self.curriculum_stage]

        def set_curriculum_stage(self, stage: int) -> None:
            self.curriculum_stage = int(np.clip(
                stage, 0, self.curriculum_stage_count - 1))
            self.cfg.curriculum_stage = self.curriculum_stage
            self.reset(np.ones(self.cfg.worlds, bool))

        def reset(self, mask: np.ndarray) -> None:
            for index in np.flatnonzero(mask):
                minimum, maximum = self.distance_range
                goal = self.rng.uniform(-1, 1)
                for _ in range(10_000):
                    x = self.rng.uniform(-1, 1)
                    distance = abs(x - goal)
                    if minimum <= distance <= maximum:
                        break
                else:
                    raise RuntimeError(
                        "could not sample curriculum start state")
                self.goal[index] = goal
                self.x[index] = x
                self.age[index] = 0
                self.hold[index] = 0
                self.episode[index] += 1

        def observation(self):
            return np.stack((
                self.x, self.goal, self.goal - self.x
            ), -1).astype(np.float32)

    @dataclass
    class Transition:
        observation: np.ndarray
        target_observation: np.ndarray
        dense_reward: np.ndarray
        done: np.ndarray
        success: np.ndarray
        episode: np.ndarray

    class TaskRewardPointControl(PointControl):
        def step(self, action: np.ndarray):
            action = np.asarray(action, np.float32).clip(-1, 1)
            episode = self.episode.copy()
            previous = self.goal - self.x
            self.x = np.clip(
                self.x + self.cfg.step_scale * action,
                -1.25,
                1.25,
            )
            error = self.goal - self.x
            progress = np.abs(previous) - np.abs(error)
            overshot = (
                (previous * error < 0)
                & (np.abs(error) >= self.cfg.target_radius)
            )
            inside = np.abs(error) < self.cfg.target_radius
            self.hold = np.where(inside, self.hold + 1, 0)
            self.age += 1
            reached = self.hold >= self.hold_steps
            timeout = (
                self.age >= self.current_episode_limit
            ) & ~reached
            shaping = (
                self.cfg.progress_weight * progress
                - self.cfg.action_cost * np.square(action)
                - self.cfg.overshoot_penalty
                * overshot.astype(np.float32)
            )
            done = reached | timeout
            success = reached.copy()
            terminal = np.zeros(self.cfg.worlds, np.float32)
            terminal[success] = 1.0
            terminal[timeout] = -1.0
            reward = shaping.astype(np.float32) + terminal
            target = self.observation().copy()
            self.reset(done)
            return Transition(
                self.observation(),
                target,
                reward,
                done,
                success,
                episode,
            )

    return (TaskRewardPointControl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training controls
    """)
    return


@app.cell
def _(Config, RELATIVE_HORIZONS, mo):
    defaults = Config()

    def number(value, label, *, step=None, start=None, stop=None):
        return mo.ui.number(
            value=value,
            label=label,
            step=step,
            start=start,
            stop=stop,
            full_width=True,
        )

    controls = {
        "seed": number(defaults.seed, "Run · Seed", step=1, start=0),
        "device": mo.ui.dropdown(
            ["cpu", "mps", "cuda"],
            value=defaults.device,
            label="Run · Device",
            full_width=True,
        ),
        "checkpoint_path": mo.ui.text(
            value=defaults.checkpoint_path,
            label="Run · Checkpoint path",
            full_width=True,
        ),
        "worlds": number(
            defaults.worlds, "Run · Parallel worlds", step=1, start=1),
        "transitions": number(
            defaults.transitions, "Run · Transitions", step=1, start=1),
        "latent_dim": number(
            defaults.latent_dim, "Network · Latent dimension",
            step=1, start=1),
        "ghost_dim": number(
            defaults.ghost_dim, "Network · Intent dimension",
            step=1, start=1),
        "hidden_dim": number(
            defaults.hidden_dim, "Network · Hidden dimension",
            step=1, start=1),
        "conditioning_dim": number(
            defaults.conditioning_dim, "Network · Conditioning dimension",
            step=1, start=1),
        "snn_ticks": number(
            defaults.snn_ticks, "Network · SNN ticks", step=1, start=1),
        "membrane_decay": number(
            defaults.membrane_decay, "Network · Membrane decay",
            step=0.01, start=0.0, stop=1.0),
        "trajectory_predictor_membrane_decay": number(
            defaults.trajectory_predictor_membrane_decay,
            "Network · Trajectory predictor decay",
            step=0.01, start=0.0, stop=1.0),
        "trajectory_critic_membrane_decay": number(
            defaults.trajectory_critic_membrane_decay,
            "Network · Trajectory critic decay",
            step=0.01, start=0.0, stop=1.0),
        "surrogate_scale": number(
            defaults.surrogate_scale, "Network · Surrogate scale",
            step=0.01, start=0.0),
        "encoder_latent_decay": number(
            defaults.encoder_latent_decay,
            "Encoder · Latent decay",
            step=0.01, start=0.0, stop=1.0),
        "encoder_membrane_readout": number(
            defaults.encoder_membrane_readout,
            "Encoder · Membrane readout",
            step=0.01, start=0.0),
        "encoder_lr": number(
            defaults.encoder_lr, "Learning · Encoder rate",
            step=1e-5, start=0.0),
        "trajectory_predictor_lr": number(
            defaults.trajectory_predictor_lr,
            "Learning · Trajectory predictor rate", step=1e-5, start=0.0),
        "trajectory_critic_lr": number(
            defaults.trajectory_critic_lr,
            "Learning · Trajectory critic rate", step=1e-5, start=0.0),
        "ghost_lr": number(
            defaults.ghost_lr, "Learning · Ghost rate",
            step=1e-5, start=0.0),
        "trajectory_critic_gradient_clip": number(
            defaults.trajectory_critic_gradient_clip,
            "Learning · Trajectory critic clip", step=0.1, start=0.0),
        "trajectory_predictor_gradient_clip": number(
            defaults.trajectory_predictor_gradient_clip,
            "Learning · Trajectory predictor clip",
            step=0.1, start=0.0),
        "trajectory_critic_target_tau": number(
            defaults.trajectory_critic_target_tau,
            "Learning · Trajectory target rate", step=0.001, start=0.0),
        "encoder_target_tau": number(
            defaults.encoder_target_tau,
            "Learning · Encoder target rate", step=0.001, start=0.0),
        "trajectory_critic_encoder_weight": number(
            defaults.trajectory_critic_encoder_weight,
            "Learning · Trajectory critic→encoder", step=0.05, start=0.0),
        "external_intent_weight": number(
            defaults.external_intent_weight,
            "Ghost · External intent weight", step=0.05, start=0.0),
        "trajectory_feasibility_weight": number(
            defaults.trajectory_feasibility_weight,
            "Ghost · Trajectory feasibility weight", step=0.05, start=0.0),
        "tracking_action_weight": number(
            defaults.tracking_action_weight,
            "Ghost · Tracking action weight", step=0.05, start=0.0),
        "external_horizon_weight": number(
            defaults.external_horizon_weight,
            "Ghost · External horizon weight", step=0.05, start=0.0),
        "feasibility_horizon_weight": number(
            defaults.feasibility_horizon_weight,
            "Ghost · Feasibility horizon weight", step=0.05, start=0.0),
        "trajectory_feasibility_gradient_clip": number(
            defaults.trajectory_feasibility_gradient_clip,
            "Ghost · Feasibility gradient clip", step=0.1, start=0.0),
        "tracking_gradient_clip": number(
            defaults.tracking_gradient_clip,
            "Ghost · Tracking gradient clip", step=0.1, start=0.0),
        "ghost_score_warmup_transitions": number(
            defaults.ghost_score_warmup_transitions,
            "Learning · Ghost score warmup", step=1, start=0),
        "gamma": number(
            defaults.gamma, "Learning · Outer discount",
            step=0.001, start=0.0, stop=1.0),
        "initial_exploration_std": number(
            defaults.initial_exploration_std,
            "Exploration · Initial std", step=0.01, start=0.0),
        "minimum_exploration_std": number(
            defaults.minimum_exploration_std,
            "Exploration · Minimum std", step=0.01, start=0.0),
        "exploration_decay_transitions": number(
            defaults.exploration_decay_transitions,
            "Exploration · Decay transitions", step=1, start=1),
        "uncertainty_penalty": number(
            defaults.uncertainty_penalty,
            "Trajectory · Uncertainty penalty", step=0.01, start=0.0),
        "trajectory_replacement_margin": number(
            defaults.trajectory_replacement_margin,
            "Trajectory · Replacement margin", step=0.01, start=0.0),
        "trajectory_switch_penalty": number(
            defaults.trajectory_switch_penalty,
            "Trajectory · Switch penalty", step=0.01, start=0.0),
        "trajectory_max_horizon": number(
            defaults.trajectory_max_horizon,
            "Trajectory · Maximum horizon", step=1, start=1),
        "spline_control_points": number(
            defaults.spline_control_points,
            "Spline · Control points", step=1, start=2),
        "spline_degree": number(
            defaults.spline_degree,
            "Spline · Degree", step=1, start=1),
        "spline_max_control_step": number(
            defaults.spline_max_control_step,
            "Spline · Maximum control step", step=0.01, start=0.001),
        "minimum_spline_arc_length": number(
            defaults.minimum_spline_arc_length,
            "Spline · Minimum arc length", step=0.005, start=0.0),
        "spline_curvature_regularization": number(
            defaults.spline_curvature_regularization,
            "Spline · Curvature regularization",
            step=0.001, start=0.0),
        "tracking_action_cost": number(
            defaults.tracking_action_cost,
            "Tracking · Action cost", step=0.001, start=0.0),
        "tracking_baseline_decay": number(
            defaults.tracking_baseline_decay,
            "Tracking · Baseline decay", step=0.01,
            start=0.0, stop=1.0),
        "external_horizon_baseline_decay": number(
            defaults.external_horizon_baseline_decay,
            "Horizon · External baseline decay", step=0.01,
            start=0.0, stop=1.0),
        "feasibility_horizon_baseline_decay": number(
            defaults.feasibility_horizon_baseline_decay,
            "Horizon · Feasibility baseline decay", step=0.01,
            start=0.0, stop=1.0),
        "minimum_waypoint_displacement": number(
            defaults.minimum_waypoint_displacement,
            "Trajectory · Minimum waypoint displacement",
            step=0.005, start=0.0),
        "horizon_uniform_eta": number(
            defaults.horizon_uniform_eta,
            "Horizon · Uniform mixing", step=0.01,
            start=0.0, stop=1.0),
        "actual_horizon_min_probability": number(
            defaults.actual_horizon_min_probability,
            "Horizon · Minimum probability", step=0.01,
            start=0.0, stop=1.0),
        "horizon_entropy_weight": number(
            defaults.horizon_entropy_weight,
            "Horizon · Entropy weight", step=0.001, start=0.0),
        "predictor_logvar_min": number(
            defaults.predictor_logvar_min,
            "Predictor · Minimum log variance", step=0.1),
        "predictor_logvar_max": number(
            defaults.predictor_logvar_max,
            "Predictor · Maximum log variance", step=0.1),
        "step_scale": number(
            defaults.step_scale, "Environment · Step scale",
            step=0.01, start=0.001),
        "target_radius": number(
            defaults.target_radius, "Environment · Target radius",
            step=0.005, start=0.001),
        "progress_weight": number(
            defaults.progress_weight, "Environment · Progress weight",
            step=0.1, start=0.0),
        "action_cost": number(
            defaults.action_cost, "Environment · External action cost",
            step=0.001, start=0.0),
        "overshoot_penalty": number(
            defaults.overshoot_penalty,
            "Environment · Overshoot penalty", step=0.01, start=0.0),
        "curriculum_stage": number(
            defaults.curriculum_stage, "Curriculum · Initial stage",
            step=1, start=0,
            stop=len(defaults.curriculum_distance_ranges) - 1),
        "curriculum_threshold": number(
            defaults.curriculum_threshold,
            "Curriculum · Advancement threshold",
            step=0.01, start=0.0, stop=1.0),
        "curriculum_window_episodes": number(
            defaults.curriculum_window_episodes,
            "Curriculum · Episode window", step=1, start=1),
        "plot_every": number(
            defaults.plot_every, "Dashboard · Refresh transitions",
            step=1, start=1),
        "plot_window": number(
            defaults.plot_window, "Dashboard · Rolling window",
            step=1, start=1),
    }

    training_form = mo.ui.dictionary(controls).form(
        label="Rolling leader–contender trajectory configuration",
        submit_button_label="Start training",
        submit_button_tooltip="Apply values and start a new run",
        bordered=True,
    )
    mo.vstack([
        mo.md(
            "## Training controls\n"
            "Values are applied only when **Start training** is clicked."
        ),
        training_form,
    ])
    return (training_form,)


@app.cell
def _(
    CHECKPOINT_FORMAT_VERSION,
    Config,
    Encoder,
    F,
    Ghost,
    GhostLearner,
    LeaderTrajectory,
    Path,
    PendingTrajectoryTD,
    RELATIVE_HORIZONS,
    ReadyTrajectoryTD,
    ReadyWaypointPrediction,
    TaskRewardPointControl,
    TrajectoryCritic,
    TrajectoryPredictor,
    add_gradient_lists,
    asdict,
    bspline_basis,
    build_spline_tables,
    clip_gradient_list,
    copy,
    deque,
    gradient_list_norm,
    mo,
    np,
    optimizer_step_from_gradients,
    parameter_gradients,
    per_world_parameter_gradients,
    plt,
    polyak_update,
    render_remaining_spline,
    render_spline_logvars,
    render_spline_positions,
    slice_state,
    torch,
    training_form,
    zeros_like_parameters,
):
    mo.stop(
        training_form.value is None,
        mo.md("Configure the run above, then click **Start training**."),
    )
    submitted = dict(training_form.value)
    integer_fields = (
        "seed", "worlds", "transitions", "latent_dim", "ghost_dim",
        "hidden_dim", "conditioning_dim", "snn_ticks",
        "ghost_score_warmup_transitions",
        "exploration_decay_transitions", "trajectory_max_horizon",
        "spline_control_points", "spline_degree",
        "curriculum_stage", "curriculum_window_episodes",
        "plot_every", "plot_window",
    )
    for field in integer_fields:
        submitted[field] = int(submitted[field])
    cfg = Config(**submitted)
    if cfg.minimum_exploration_std > cfg.initial_exploration_std:
        raise ValueError(
            "minimum exploration std cannot exceed initial std")
    if cfg.spline_degree < 1:
        raise ValueError("spline_degree must be at least one")
    if cfg.spline_control_points < cfg.spline_degree + 1:
        raise ValueError(
            "spline_control_points must be at least spline_degree + 1")
    if cfg.spline_max_control_step <= 0:
        raise ValueError("spline_max_control_step must be positive")
    if cfg.trajectory_max_horizon < 1:
        raise ValueError("trajectory_max_horizon must be at least one")
    existing_checkpoint = Path(cfg.checkpoint_path)
    if existing_checkpoint.exists():
        try:
            checkpoint_metadata = torch.load(
                existing_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            existing_format = checkpoint_metadata.get(
                "format_version")
        except Exception as error:
            existing_format = None
            print(
                "checkpoint metadata could not be inspected; "
                f"starting clean ({error})"
            )
        if (
            existing_format is not None
            and existing_format != CHECKPOINT_FORMAT_VERSION
        ):
            print(
                f"checkpoint format {existing_format} is incompatible "
                f"with B-spline format {CHECKPOINT_FORMAT_VERSION}; "
                "starting with clean spline predictor heads"
            )

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    env = TaskRewardPointControl(cfg, cfg.seed)

    encoder = Encoder(cfg).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    ghost = Ghost(cfg).to(device)
    trajectory_predictor = TrajectoryPredictor(cfg).to(device)
    trajectory_critic = TrajectoryCritic(cfg).to(device)
    target_trajectory_critic = copy.deepcopy(
        trajectory_critic).to(device)
    for model in (target_encoder, target_trajectory_critic):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=cfg.encoder_lr)
    trajectory_predictor_optimizer = torch.optim.Adam(
        trajectory_predictor.parameters(),
        lr=cfg.trajectory_predictor_lr,
    )
    trajectory_critic_optimizer = torch.optim.Adam(
        trajectory_critic.parameters(),
        lr=cfg.trajectory_critic_lr,
    )
    ghost_learner = GhostLearner(ghost, cfg)

    encoder_parameters = list(encoder.parameters())
    predictor_parameters = list(trajectory_predictor.parameters())
    critic_parameters = list(trajectory_critic.parameters())
    ghost_parameters = list(ghost.parameters())

    b = cfg.worlds
    k_max = cfg.trajectory_max_horizon
    (
        spline_basis_table,
        spline_mask_table,
        spline_phase_table,
    ) = build_spline_tables(cfg, device)
    leaders = [None for _ in range(b)]
    pending_td = [None for _ in range(b)]
    previous_tracking_reward = torch.zeros(b, device=device)
    plan_versions = torch.zeros(b, dtype=torch.long, device=device)
    exhausted_since_last_step = torch.zeros(
        b, dtype=torch.bool, device=device)
    tracking_baseline = torch.zeros(k_max + 1, device=device)
    external_horizon_baseline = torch.zeros(
        k_max + 1, device=device)
    feasibility_horizon_baseline = torch.zeros(
        k_max + 1, k_max, device=device)

    metric_names = (
        "reward", "success", "trajectory_predictor_loss",
        "trajectory_critic_loss", "absolute_td_error",
        "leader_q", "contender_q", "adjusted_leader",
        "adjusted_contender", "replacement_rate",
        "leader_remaining_horizon", "selected_horizon",
        "leader_cursor_fraction", "waypoint_distance_before",
        "waypoint_distance_after", "tracking_reward",
        "tracking_advantage", "waypoint_prediction_error",
        "trajectory_feasibility_direction",
        "action_tracking_direction", "external_intent_direction",
        "external_horizon_direction",
        "feasibility_horizon_direction", "leader_uncertainty",
        "contender_uncertainty", "switching_cost",
        "action_entropy", "action_saturation", "absolute_action",
        "ghost_step", "encoder_step", "predictor_step",
        "critic_step", "critic_membrane_norm",
        "predictor_membrane_norm", "forced_replacement_rate",
        "rejected_contender_rate", "degenerate_waypoint_rate",
        "spline_arc_length", "spline_endpoint_displacement",
        "mean_control_increment", "max_control_increment",
        "spline_curvature", "degenerate_contender_rate",
        "degenerate_leader_rate", "current_leader_phase",
        "selected_phase_displacement",
        "control_intent_jacobian_norm",
        "waypoint_intent_jacobian_norm", "spline_basis_error",
        "spline_anchor_error",
    )
    metric_windows = {
        name: deque(maxlen=cfg.plot_window) for name in metric_names
    }
    plot_history = {
        name: [] for name in ("transition", "curriculum_stage")
        + metric_names
    }
    trajectory_step_history = []
    recent_episode_success = deque(
        maxlen=cfg.curriculum_window_episodes)
    observation = env.observation()
    transitions = 0
    next_plot_transition = max(cfg.plot_every, 1)

    def rolling_metric(name):
        values = metric_windows[name]
        return float(np.mean(values)) if values else float("nan")

    def rms(value):
        return value.square().mean(dim=-1).add(1e-8).sqrt()

    def trajectory_deltas(waypoints, mask, target_current):
        deltas = torch.zeros_like(waypoints)
        deltas[:, 0] = waypoints[:, 0] - target_current
        if k_max > 1:
            deltas[:, 1:] = waypoints[:, 1:] - waypoints[:, :-1]
        return deltas * mask[:, :, None]

    def trajectory_uncertainty(logvars, mask):
        valid = mask[:, :, None].expand_as(logvars)
        denominator = valid.sum((1, 2)).clamp_min(1)
        variance = (
            torch.where(valid, logvars.exp(), torch.zeros_like(logvars))
            .sum((1, 2)) / denominator
        )
        return variance.clamp_min(1e-12).sqrt()

    def slice_runtime(runtime, world):
        state = runtime["state"]
        return {
            "state": slice_state(state, world),
            "last_output": (
                None if runtime["last_output"] is None
                else runtime["last_output"][
                    world:world + 1].detach().clone()
            ),
        }

    def snapshot_predictor_runtime():
        return (
            trajectory_predictor.core.snapshot(),
            None if trajectory_predictor.core.last_output is None
            else trajectory_predictor.core.last_output.detach().clone(),
            list(trajectory_predictor.core.last_eligibility_records),
        )

    def restore_predictor_runtime(runtime):
        state, last_output, eligibility = runtime
        trajectory_predictor.core.restore(state)
        trajectory_predictor.core.last_output = (
            None if last_output is None else last_output.clone())
        trajectory_predictor.core.last_eligibility_records = eligibility

    def evaluate_branches(
        critic,
        latent_value,
        leader_deltas,
        leader_mask,
        leader_phases,
        contender_deltas,
        contender_mask,
        contender_phases,
        choose_contender=None,
    ):
        base = critic.snapshot_runtime()
        critic.restore_runtime(base)
        leader_score = critic.evaluate_trajectory(
            latent_value, leader_deltas, leader_mask, leader_phases)
        leader_post = critic.snapshot_runtime()
        critic.restore_runtime(base)
        contender_score = critic.evaluate_trajectory(
            latent_value,
            contender_deltas,
            contender_mask,
            contender_phases,
        )
        contender_post = critic.snapshot_runtime()
        if choose_contender is None:
            critic.restore_runtime(base)
        else:
            critic.restore_runtime(TrajectoryCritic.blend_runtime(
                leader_post, contender_post, choose_contender))
        return (
            leader_score, contender_score, base,
            leader_post, contender_post,
        )

    def weighted_world_direction(eligibilities, signal):
        result = []
        for eligibility in eligibilities:
            view = signal.shape + (1,) * (eligibility.ndim - 1)
            result.append(
                (eligibility * signal.view(view)).mean(0))
        return result

    def add_in_place(destination, source, scale=1.0):
        for index, value in enumerate(source):
            if value is not None:
                destination[index].add_(value, alpha=scale)

    def render_training_dashboard():
        mo.output.clear()
        fig, axes = plt.subplots(13, 4, figsize=(19, 41))
        fig.patch.set_facecolor("black")
        labels = [
            ("reward", "External reward"),
            ("success", "Episode success"),
            ("trajectory_predictor_loss", "Waypoint predictor NLL"),
            ("trajectory_critic_loss", "Trajectory critic TD loss"),
            ("absolute_td_error", "Mean absolute TD error"),
            ("leader_q", "Leader Q"),
            ("contender_q", "Contender Q"),
            ("replacement_rate", "Contender replacement rate"),
            ("leader_remaining_horizon", "Leader remaining horizon"),
            ("selected_horizon", "Selected horizon"),
            ("leader_cursor_fraction", "Leader cursor fraction"),
            ("tracking_reward", "Immediate tracking reward"),
            ("waypoint_distance_before", "Waypoint distance before"),
            ("waypoint_distance_after", "Waypoint distance after"),
            ("tracking_advantage", "Tracking advantage"),
            ("waypoint_prediction_error", "Waypoint prediction error"),
            ("leader_uncertainty", "Leader uncertainty"),
            ("contender_uncertainty", "Contender uncertainty"),
            ("switching_cost", "Switching cost"),
            ("rejected_contender_rate", "Rejected contender rate"),
            ("external_intent_direction", "External intent direction"),
            ("trajectory_feasibility_direction", "Feasibility direction"),
            ("action_tracking_direction", "Action tracking direction"),
            ("external_horizon_direction", "External horizon direction"),
            ("feasibility_horizon_direction", "Feasibility horizon direction"),
            ("action_entropy", "Action-policy entropy"),
            ("action_saturation", "Action saturation"),
            ("absolute_action", "Mean absolute action"),
            ("encoder_step", "Encoder update norm"),
            ("ghost_step", "Ghost update norm"),
            ("predictor_step", "Predictor update norm"),
            ("critic_step", "Critic update norm"),
            ("critic_membrane_norm", "Critic membrane norm"),
            ("predictor_membrane_norm", "Predictor membrane norm"),
            (
                "forced_replacement_rate",
                "Replacement forced by exhaustion",
            ),
            ("spline_arc_length", "Spline arc length"),
            (
                "spline_endpoint_displacement",
                "Spline endpoint displacement",
            ),
            ("mean_control_increment", "Mean control increment"),
            ("max_control_increment", "Maximum control increment"),
            ("spline_curvature", "Spline control curvature"),
            (
                "degenerate_contender_rate",
                "Spline-degenerate contender rate",
            ),
            (
                "degenerate_leader_rate",
                "Spline-degenerate leader rate",
            ),
            ("current_leader_phase", "Current leader phase"),
            (
                "selected_phase_displacement",
                "Selected next-phase displacement",
            ),
            (
                "control_intent_jacobian_norm",
                "Control→intent Jacobian norm",
            ),
            (
                "waypoint_intent_jacobian_norm",
                "Waypoint→intent Jacobian norm",
            ),
            ("spline_basis_error", "Basis partition error"),
            ("spline_anchor_error", "Spline start-anchor error"),
        ]
        x = plot_history["transition"]
        for axis, (name, title) in zip(axes.flat, labels):
            axis.set_facecolor("black")
            axis.plot(x, plot_history[name])
            axis.set_title(title)
            axis.grid(alpha=0.25)
        step_axis = axes.flat[len(labels)]
        step_axis.set_facecolor("black")
        if trajectory_step_history:
            recent_steps = np.stack(
                trajectory_step_history[-cfg.plot_window:])
            step_axis.plot(
                np.arange(1, k_max + 1),
                recent_steps.mean(0),
            )
        step_axis.set_title("Mean rendered spline-step norm")
        step_axis.set_xlabel("Normalized sampled phase index")
        step_axis.grid(alpha=0.25)
        for axis in axes.flat[len(labels) + 1:]:
            axis.set_axis_off()
        fig.suptitle(
            f"Rolling trajectory training — {transitions:,} / "
            f"{cfg.transitions:,} transitions",
            fontsize=16,
        )
        fig.tight_layout()
        mo.output.replace(fig)

    while transitions < cfg.transitions:
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=device)
        latent = encoder(observation_tensor)
        with torch.no_grad():
            target_current = target_encoder(observation_tensor)

        leader_valid = torch.zeros(b, dtype=torch.bool, device=device)
        leader_next_delta = torch.zeros(
            b, cfg.latent_dim, device=device)
        leader_feedback_score = torch.zeros(b, device=device)
        leader_feedback_uncertainty = torch.zeros(b, device=device)
        leader_normalized_remaining = torch.zeros(b, device=device)
        leader_controls = torch.zeros(
            b, cfg.spline_control_points, cfg.latent_dim, device=device)
        leader_control_logvars = torch.full_like(
            leader_controls, cfg.predictor_logvar_min)
        leader_selected_horizon = torch.ones(
            b, dtype=torch.long, device=device)
        leader_cursors = torch.ones(
            b, dtype=torch.long, device=device)
        for world, leader in enumerate(leaders):
            if (
                leader is None
                or not leader.valid
                or leader.episode != int(env.episode[world])
                or leader.cursor >= leader.selected_horizon
            ):
                continue
            count = leader.selected_horizon - leader.cursor
            leader_valid[world] = True
            leader_selected_horizon[world] = leader.selected_horizon
            leader_cursors[world] = leader.cursor
            leader_controls[world] = leader.control_points
            leader_control_logvars[world] = leader.control_logvars
            leader_feedback_score[world] = (
                leader.original_adjusted_score)
            leader_normalized_remaining[world] = (
                count / max(k_max, 1))
        (
            leader_waypoints,
            leader_logvars,
            leader_mask,
            leader_phases,
        ) = render_remaining_spline(
            leader_controls,
            leader_control_logvars,
            leader_selected_horizon,
            leader_cursors,
            spline_basis_table,
            spline_mask_table,
            spline_phase_table,
            cfg.predictor_logvar_min,
            cfg.predictor_logvar_max,
        )
        leader_next_delta = torch.where(
            leader_valid[:, None],
            leader_waypoints[:, 0] - target_current,
            torch.zeros_like(leader_next_delta),
        )
        leader_feedback_uncertainty = trajectory_uncertainty(
            leader_logvars, leader_mask)
        leader_feedback_uncertainty = torch.where(
            leader_valid,
            leader_feedback_uncertainty,
            torch.zeros_like(leader_feedback_uncertainty),
        )

        feature = ghost.update_context(
            latent.detach(),
            leader_next_delta.detach(),
            leader_feedback_score.detach(),
            leader_feedback_uncertainty.detach(),
            leader_normalized_remaining.detach(),
            leader_valid,
            previous_tracking_reward.detach(),
        )
        intent, horizon_logits = ghost.plan_from_feature(feature)

        remaining_budget = torch.as_tensor(
            np.maximum(
                env.current_episode_limit - env.age, 1),
            dtype=torch.float32,
            device=device,
        )
        relative_options = torch.as_tensor(
            RELATIVE_HORIZONS,
            dtype=torch.float32,
            device=device,
        )
        option_horizons = (
            remaining_budget[:, None] * relative_options[None]
        ).round().long().clamp(1, k_max)
        option_probabilities = torch.softmax(horizon_logits, -1)
        option_probabilities = (
            (1 - cfg.horizon_uniform_eta) * option_probabilities
            + cfg.horizon_uniform_eta / len(RELATIVE_HORIZONS)
        )
        actual_probabilities = torch.zeros(
            b, k_max + 1, device=device)
        actual_probabilities.scatter_add_(
            1, option_horizons, option_probabilities)
        floor = cfg.actual_horizon_min_probability
        valid_horizons = torch.arange(
            k_max + 1, device=device)[None] >= 1
        actual_probabilities = (
            actual_probabilities
            + floor * valid_horizons.float()
        )
        actual_probabilities[:, 0] = 0
        actual_probabilities = (
            actual_probabilities
            / actual_probabilities.sum(-1, keepdim=True)
        )
        actual_horizon = torch.multinomial(
            actual_probabilities, 1).squeeze(-1)
        selected_probability = actual_probabilities.gather(
            1, actual_horizon[:, None]).squeeze(1).clamp_min(1e-8)
        horizon_log_probability = selected_probability.log()
        horizon_entropy = -(
            actual_probabilities
            * actual_probabilities.clamp_min(1e-8).log()
        ).sum(-1)
        horizon_gradients = per_world_parameter_gradients(
            horizon_log_probability, ghost_parameters)
        entropy_gradients = per_world_parameter_gradients(
            horizon_entropy, ghost_parameters)
        relative_horizon = (
            actual_horizon.float() / remaining_budget
        ).clamp(0, 1)
        budget_fraction = (
            remaining_budget / max(env.current_episode_limit, 1)
        ).clamp(0, 1)

        predictor_pre = trajectory_predictor.core.snapshot()
        (
            contender_control_points,
            contender_control_logvars,
        ) = trajectory_predictor(
            latent.detach(),
            intent,
            relative_horizon,
            budget_fraction,
        )
        contender_cursors = torch.zeros(
            b, dtype=torch.long, device=device)
        (
            contender_waypoints,
            contender_logvars,
            contender_mask,
            contender_phases,
        ) = render_remaining_spline(
            contender_control_points,
            contender_control_logvars,
            actual_horizon,
            contender_cursors,
            spline_basis_table,
            spline_mask_table,
            spline_phase_table,
            cfg.predictor_logvar_min,
            cfg.predictor_logvar_max,
        )
        contender_uncertainty = trajectory_uncertainty(
            contender_logvars, contender_mask)
        leader_deltas = trajectory_deltas(
            leader_waypoints, leader_mask, target_current.detach())
        contender_deltas = trajectory_deltas(
            contender_waypoints,
            contender_mask,
            target_current.detach(),
        )

        (
            leader_q,
            contender_q,
            critic_base,
            leader_post,
            contender_post,
        ) = evaluate_branches(
            trajectory_critic,
            latent.detach(),
            leader_deltas.detach(),
            leader_mask,
            leader_phases,
            contender_deltas,
            contender_mask,
            contender_phases,
        )
        leader_adjusted = (
            leader_q
            - cfg.uncertainty_penalty
            * leader_feedback_uncertainty.detach()
        )
        switching_cost = torch.where(
            leader_valid,
            rms(contender_waypoints[:, 0].detach()
                - leader_waypoints[:, 0]),
            torch.zeros(b, device=device),
        )
        contender_adjusted = (
            contender_q
            - cfg.uncertainty_penalty * contender_uncertainty
            - cfg.trajectory_switch_penalty * switching_cost
        )
        forced_replacement = ~leader_valid
        forced_by_exhaustion = (
            forced_replacement & exhausted_since_last_step)
        choose_contender = (
            forced_replacement
            | (
                contender_adjusted.detach()
                > (
                    leader_adjusted.detach()
                    + cfg.trajectory_replacement_margin
                )
            )
        )
        exhausted_since_last_step.zero_()
        trajectory_critic.restore_runtime(
            TrajectoryCritic.blend_runtime(
                leader_post, contender_post, choose_contender))

        with torch.no_grad():
            (
                target_leader_q,
                target_contender_q,
                _target_base,
                target_leader_post,
                target_contender_post,
            ) = evaluate_branches(
                target_trajectory_critic,
                target_current,
                leader_deltas,
                leader_mask,
                leader_phases,
                contender_deltas.detach(),
                contender_mask,
                contender_phases,
            )
            target_trajectory_critic.restore_runtime(
                TrajectoryCritic.blend_runtime(
                    target_leader_post,
                    target_contender_post,
                    choose_contender,
                )
            )
            target_selected_q = torch.where(
                choose_contender,
                target_contender_q,
                target_leader_q,
            )

        ready_td = []
        for world, record in enumerate(pending_td):
            if record is None:
                continue
            if record.episode == int(env.episode[world]):
                ready_td.append(ReadyTrajectoryTD(
                    record,
                    (
                        record.reward
                        + cfg.gamma
                        * target_selected_q[
                            world:world + 1].detach()
                    ),
                ))
            pending_td[world] = None

        planning_objective = (
            contender_q
            - cfg.uncertainty_penalty * contender_uncertainty
        )
        flat_controls = contender_control_points.reshape(
            b, cfg.spline_control_points * cfg.latent_dim)
        control_intent_jacobian = torch.stack([
            torch.autograd.grad(
                flat_controls[:, coordinate].sum(),
                intent,
                retain_graph=True,
            )[0].detach()
            for coordinate in range(
                cfg.spline_control_points * cfg.latent_dim)
        ], 1).reshape(
            b,
            cfg.spline_control_points,
            cfg.latent_dim,
            cfg.ghost_dim,
        )

        # Install accepted contenders before action readout. Rejected
        # contenders are intentionally not stored or supervised.
        selected_plan_versions = torch.zeros(
            b, dtype=torch.long, device=device)
        for world in range(b):
            if not bool(choose_contender[world]):
                continue
            plan_versions[world] += 1
            intent_eligibility_placeholder = None
            leaders[world] = LeaderTrajectory(
                episode=int(env.episode[world]),
                plan_version=int(plan_versions[world]),
                source_observation=observation_tensor[
                    world:world + 1].detach().clone(),
                source_target_latent=target_current[
                    world:world + 1].detach().clone(),
                intent=intent[world:world + 1].detach().clone(),
                control_points=contender_control_points[
                    world].detach().clone(),
                control_logvars=contender_control_logvars[
                    world].detach().clone(),
                selected_horizon=int(actual_horizon[world]),
                cursor=0,
                relative_horizon=relative_horizon[
                    world:world + 1].detach().clone(),
                budget_fraction=budget_fraction[
                    world:world + 1].detach().clone(),
                predictor_state=slice_state(predictor_pre, world),
                intent_eligibility=intent_eligibility_placeholder,
                control_intent_jacobian=(
                    control_intent_jacobian[world].clone()),
                horizon_gradients=[
                    value[world].detach().clone()
                    for value in horizon_gradients
                ],
                original_adjusted_score=float(
                    contender_adjusted[world].detach()),
                valid=True,
            )
            if (
                leaders[world].control_points.requires_grad
                or leaders[world].control_logvars.requires_grad
                or leaders[world].control_intent_jacobian.requires_grad
            ):
                raise RuntimeError(
                    "leader spline state retained a live autograd graph")
            if not bool(torch.allclose(
                leaders[world].control_intent_jacobian[0],
                torch.zeros_like(
                    leaders[world].control_intent_jacobian[0]),
                atol=1e-7,
                rtol=0,
            )):
                raise RuntimeError(
                    "anchored spline control acquired an intent Jacobian")

        selected_remaining = torch.zeros(b, device=device)
        selected_valid = torch.zeros(
            b, dtype=torch.bool, device=device)
        selected_cursor = torch.zeros(
            b, dtype=torch.long, device=device)
        selected_horizon = torch.ones(
            b, dtype=torch.long, device=device)
        selected_q = torch.where(
            choose_contender, contender_q, leader_q)
        selected_controls = torch.zeros_like(leader_controls)
        selected_control_logvars = torch.full_like(
            leader_control_logvars, cfg.predictor_logvar_min)
        for world, leader in enumerate(leaders):
            if leader is None or not leader.valid:
                continue
            cursor = leader.cursor
            remaining = leader.selected_horizon - cursor
            selected_valid[world] = True
            selected_plan_versions[world] = leader.plan_version
            selected_cursor[world] = cursor
            selected_horizon[world] = leader.selected_horizon
            selected_remaining[world] = remaining / max(k_max, 1)
            selected_controls[world] = leader.control_points
            selected_control_logvars[world] = leader.control_logvars
        (
            selected_waypoint_buffers,
            selected_logvar_buffers,
            selected_masks,
            selected_phases,
        ) = render_remaining_spline(
            selected_controls,
            selected_control_logvars,
            selected_horizon,
            selected_cursor,
            spline_basis_table,
            spline_mask_table,
            spline_phase_table,
            cfg.predictor_logvar_min,
            cfg.predictor_logvar_max,
        )
        selected_waypoint = selected_waypoint_buffers[:, 0].detach()
        selected_phase = selected_phases[:, 0].detach()

        raw_action_mean = ghost.action_from_feature(
            feature,
            (selected_waypoint - target_current).detach(),
            selected_remaining.detach(),
            selected_valid,
        )
        exploration_fraction = min(
            transitions / max(cfg.exploration_decay_transitions, 1),
            1.0,
        )
        exploration_std = (
            cfg.initial_exploration_std
            + exploration_fraction
            * (
                cfg.minimum_exploration_std
                - cfg.initial_exploration_std
            )
        )
        normal = torch.distributions.Normal(
            raw_action_mean,
            torch.full_like(raw_action_mean, exploration_std),
        )
        # The sampled action is a fixed stochastic outcome for REINFORCE.
        # Detaching it here preserves the Normal score-function derivative
        # with respect to the mean; rsample() would cancel that derivative.
        raw_action = normal.sample()
        executed_action = torch.tanh(raw_action)
        action_log_probability = (
            normal.log_prob(raw_action)
            - torch.log(
                1 - executed_action.square() + 1e-6)
        )
        (
            external_intent_directions,
            intent_eligibilities,
            action_eligibilities,
        ) = ghost_learner.capture_directions(
            intent,
            raw_action_mean,
            planning_objective,
            action_log_probability,
        )
        if transitions < cfg.ghost_score_warmup_transitions:
            external_intent_directions = zeros_like_parameters(
                ghost_parameters)

        # Accepted leaders keep detached e-prop eligibilities from the
        # proposal that created them. No autograd graph crosses a transition.
        for world in range(b):
            if bool(choose_contender[world]):
                leaders[world].intent_eligibility = [
                    value[world].detach().clone()
                    for value in intent_eligibilities
                ]

        transition = env.step(
            executed_action.detach().cpu().numpy())
        endpoint_observation = torch.as_tensor(
            transition.target_observation,
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            endpoint_target = target_encoder(endpoint_observation)

        distance_before = rms(
            target_current.detach() - selected_waypoint.detach())
        distance_after = rms(
            endpoint_target.detach() - selected_waypoint.detach())
        tracking_reward = (
            distance_before
            - distance_after
            - cfg.tracking_action_cost
            * executed_action.detach().square()
        )
        remaining_index = (
            selected_horizon - selected_cursor
        ).clamp(1, k_max)
        tracking_advantage = (
            tracking_reward
            - tracking_baseline[remaining_index]
        )
        tracking_action_directions = weighted_world_direction(
            action_eligibilities,
            tracking_advantage.detach(),
        )
        with torch.no_grad():
            decay = cfg.tracking_baseline_decay
            for world in range(b):
                index = int(remaining_index[world])
                tracking_baseline[index].mul_(decay).add_(
                    tracking_reward[world], alpha=1 - decay)

        feasibility_directions = zeros_like_parameters(
            ghost_parameters)
        feasibility_horizon_directions = zeros_like_parameters(
            ghost_parameters)
        ready_waypoints = []
        degenerate = torch.zeros(
            b, dtype=torch.bool, device=device)
        dense_phases = torch.linspace(
            0, 1, max(4 * k_max + 1, 17),
            device=device,
        )
        dense_basis = bspline_basis(
            dense_phases,
            cfg.spline_control_points,
            cfg.spline_degree,
            device,
            contender_control_points.dtype,
        )
        contender_dense_positions = torch.einsum(
            "sc,bcl->bsl",
            dense_basis,
            contender_control_points,
        )
        contender_arc_length = (
            (
                contender_dense_positions[:, 1:]
                - contender_dense_positions[:, :-1]
            ).square().mean(-1).sqrt().sum(-1)
        )
        contender_spline_degenerate = (
            (contender_arc_length < cfg.minimum_spline_arc_length)
            | (
                rms(
                    contender_waypoints[:, 0].detach()
                    - target_current.detach()
                )
                < cfg.minimum_waypoint_displacement
            )
        )
        selected_dense_positions = torch.einsum(
            "sc,bcl->bsl",
            dense_basis,
            selected_controls,
        )
        selected_arc_length = (
            (
                selected_dense_positions[:, 1:]
                - selected_dense_positions[:, :-1]
            ).square().mean(-1).sqrt().sum(-1)
        )
        leader_dense_positions = torch.einsum(
            "sc,bcl->bsl",
            dense_basis,
            leader_controls,
        )
        leader_arc_length = (
            (
                leader_dense_positions[:, 1:]
                - leader_dense_positions[:, :-1]
            ).square().mean(-1).sqrt().sum(-1)
        )
        leader_spline_degenerate = (
            (leader_arc_length < cfg.minimum_spline_arc_length)
            | (
                rms(
                    leader_waypoints[:, 0]
                    - target_current.detach()
                )
                < cfg.minimum_waypoint_displacement
            )
        ) & leader_valid
        previous_selected_phase = (
            selected_cursor.float()
            / selected_horizon.float().clamp_min(1)
        )
        previous_selected_basis = bspline_basis(
            previous_selected_phase,
            cfg.spline_control_points,
            cfg.spline_degree,
            device,
            selected_controls.dtype,
        )
        previous_selected_position = torch.einsum(
            "bc,bcl->bl",
            previous_selected_basis,
            selected_controls,
        )
        selected_phase_displacement = rms(
            selected_waypoint - previous_selected_position)
        waypoint_prediction_errors = []
        waypoint_intent_jacobian_norms = []
        for world, leader in enumerate(leaders):
            if (
                leader is None
                or not selected_valid[world]
                or leader.plan_version
                != int(selected_plan_versions[world])
            ):
                continue
            cursor = int(selected_cursor[world])
            waypoint = selected_waypoint[world].detach()
            displacement = rms(
                waypoint - target_current[world].detach())
            degenerate[world] = (
                displacement
                < cfg.minimum_waypoint_displacement
                or selected_arc_length[world]
                < cfg.minimum_spline_arc_length
            )
            ready_waypoints.append(ReadyWaypointPrediction(
                leader.source_observation,
                leader.intent,
                leader.relative_horizon,
                leader.budget_fraction,
                leader.predictor_state,
                leader.selected_horizon,
                cursor,
                selected_phase[
                    world:world + 1].detach().clone(),
                endpoint_target[
                    world:world + 1].detach().clone(),
            ))
            waypoint_prediction_errors.append(
                float(distance_after[world]))
            if not bool(degenerate[world]):
                waypoint_variable = (
                    waypoint.clone().requires_grad_(True))
                feasibility_objective = -rms(
                    endpoint_target[world].detach()
                    - waypoint_variable
                )
                waypoint_signal = torch.autograd.grad(
                    feasibility_objective,
                    waypoint_variable,
                )[0].detach()
                phase_basis = spline_basis_table[
                    leader.selected_horizon, cursor]
                waypoint_intent_jacobian = torch.einsum(
                    "c,clg->lg",
                    phase_basis,
                    leader.control_intent_jacobian,
                )
                waypoint_intent_jacobian_norms.append(
                    float(waypoint_intent_jacobian.norm()))
                intent_signal = torch.einsum(
                    "l,lg->g",
                    waypoint_signal,
                    waypoint_intent_jacobian,
                )
                for index, eligibility in enumerate(
                    leader.intent_eligibility
                ):
                    view = intent_signal.shape + (
                        1,) * (eligibility.ndim - 1)
                    feasibility_directions[index].add_(
                        (
                            eligibility
                            * intent_signal.view(view)
                        ).sum(0),
                        alpha=1 / b,
                    )

            horizon_index = leader.selected_horizon
            horizon_advantage = (
                tracking_reward[world]
                - feasibility_horizon_baseline[
                    horizon_index, cursor]
            )
            for index, eligibility in enumerate(
                leader.horizon_gradients
            ):
                feasibility_horizon_directions[index].add_(
                    eligibility * horizon_advantage.detach(),
                    alpha=1 / b,
                )
            with torch.no_grad():
                decay = cfg.feasibility_horizon_baseline_decay
                baseline = feasibility_horizon_baseline[
                    horizon_index, cursor]
                baseline.mul_(decay).add_(
                    tracking_reward[world], alpha=1 - decay)

        external_horizon_advantage = (
            contender_adjusted.detach()
            - external_horizon_baseline[actual_horizon]
        )
        external_horizon_directions = weighted_world_direction(
            horizon_gradients,
            external_horizon_advantage,
        )
        entropy_directions = [
            value.mean(0) for value in entropy_gradients
        ]
        with torch.no_grad():
            decay = cfg.external_horizon_baseline_decay
            for world in range(b):
                index = int(actual_horizon[world])
                external_horizon_baseline[index].mul_(decay).add_(
                    contender_adjusted[world].detach(),
                    alpha=1 - decay,
                )

        # Advance exactly the plan version and cursor used by this action.
        for world, leader in enumerate(leaders):
            if (
                leader is None
                or leader.plan_version
                != int(selected_plan_versions[world])
            ):
                continue
            leader.cursor += 1
            if leader.cursor >= leader.selected_horizon:
                leader.valid = False
                leaders[world] = None
                exhausted_since_last_step[world] = True
        previous_tracking_reward.copy_(tracking_reward.detach())

        reward_tensor = torch.as_tensor(
            transition.dense_reward,
            dtype=torch.float32,
            device=device,
        )
        done_tensor = torch.as_tensor(
            transition.done, dtype=torch.bool, device=device)
        for world in range(b):
            record = PendingTrajectoryTD(
                episode=int(transition.episode[world]),
                source_observation=observation_tensor[
                    world:world + 1].detach().clone(),
                source_target_latent=target_current[
                    world:world + 1].detach().clone(),
                waypoints=selected_waypoint_buffers[
                    world:world + 1].detach().clone(),
                mask=selected_masks[
                    world:world + 1].detach().clone(),
                spline_phases=selected_phases[
                    world:world + 1].detach().clone(),
                critic_state=slice_runtime(critic_base, world),
                source_q=selected_q[
                    world:world + 1].detach().clone(),
                reward=reward_tensor[
                    world:world + 1].detach().clone(),
                done=bool(done_tensor[world]),
            )
            if bool(done_tensor[world]):
                ready_td.append(ReadyTrajectoryTD(
                    record, record.reward))
            else:
                pending_td[world] = record

        predictor_live = snapshot_predictor_runtime()
        predictor_losses = []
        for record in ready_waypoints:
            trajectory_predictor.core.restore(record.predictor_state)
            replay_latent = encoder(record.source_observation)
            replay_controls, replay_control_logvars = trajectory_predictor(
                replay_latent,
                record.intent,
                record.relative_horizon,
                record.budget_fraction,
            )
            replay_basis = bspline_basis(
                record.selected_phase,
                cfg.spline_control_points,
                cfg.spline_degree,
                device,
                replay_controls.dtype,
            )
            predicted = torch.einsum(
                "bc,bcl->bl",
                replay_basis,
                replay_controls,
            )
            rendered_variance = torch.einsum(
                "bc,bcl->bl",
                replay_basis.square(),
                replay_control_logvars.exp(),
            )
            logvar = rendered_variance.clamp_min(1e-8).log().clamp(
                cfg.predictor_logvar_min,
                cfg.predictor_logvar_max,
            )
            error = record.target - predicted
            waypoint_nll = (
                0.5
                * (
                    error.square() * (-logvar).exp()
                    + logvar
                ).mean()
            )
            if (
                cfg.spline_curvature_regularization > 0
                and cfg.spline_control_points >= 3
            ):
                second_difference = (
                    replay_controls[:, 2:]
                    - 2 * replay_controls[:, 1:-1]
                    + replay_controls[:, :-2]
                )
                waypoint_nll = (
                    waypoint_nll
                    + cfg.spline_curvature_regularization
                    * second_difference.square().mean()
                )
            predictor_losses.append(waypoint_nll)
        restore_predictor_runtime(predictor_live)
        predictor_loss = (
            torch.stack(predictor_losses).mean()
            if predictor_losses else None
        )

        critic_live = trajectory_critic.snapshot_runtime()
        critic_losses = []
        td_errors = []
        for ready in ready_td:
            record = ready.record
            trajectory_critic.restore_runtime(record.critic_state)
            source_latent = encoder(record.source_observation)
            stored_deltas = trajectory_deltas(
                record.waypoints,
                record.mask,
                record.source_target_latent,
            )
            prediction = trajectory_critic.evaluate_trajectory(
                source_latent,
                stored_deltas,
                record.mask,
                record.spline_phases,
            )
            target = ready.target.detach()
            critic_losses.append(F.smooth_l1_loss(
                prediction, target))
            td_errors.append(
                float((prediction.detach() - target).abs().mean()))
        trajectory_critic.restore_runtime(critic_live)
        critic_loss = (
            torch.stack(critic_losses).mean()
            if critic_losses else None
        )

        predictor_model_gradients = parameter_gradients(
            predictor_loss, predictor_parameters)
        critic_model_gradients = parameter_gradients(
            critic_loss, critic_parameters)
        predictor_encoder_gradients = clip_gradient_list(
            parameter_gradients(predictor_loss, encoder_parameters),
            cfg.encoder_gradient_clip,
        )
        critic_encoder_gradients = clip_gradient_list(
            parameter_gradients(critic_loss, encoder_parameters),
            cfg.encoder_gradient_clip,
        )
        encoder_gradients = add_gradient_lists(
            (1.0, predictor_encoder_gradients),
            (
                cfg.trajectory_critic_encoder_weight,
                critic_encoder_gradients,
            ),
        )
        encoder_step = optimizer_step_from_gradients(
            encoder_optimizer,
            encoder_parameters,
            encoder_gradients,
            cfg.encoder_gradient_clip,
        )
        predictor_step = optimizer_step_from_gradients(
            trajectory_predictor_optimizer,
            predictor_parameters,
            predictor_model_gradients,
            cfg.trajectory_predictor_gradient_clip,
        )
        critic_step = optimizer_step_from_gradients(
            trajectory_critic_optimizer,
            critic_parameters,
            critic_model_gradients,
            cfg.trajectory_critic_gradient_clip,
        )

        feasibility_directions = clip_gradient_list(
            feasibility_directions,
            cfg.trajectory_feasibility_gradient_clip,
        )
        tracking_action_directions = clip_gradient_list(
            tracking_action_directions,
            cfg.tracking_gradient_clip,
        )
        external_intent_direction_norm = float(
            gradient_list_norm(external_intent_directions))
        feasibility_direction_norm = float(
            gradient_list_norm(feasibility_directions))
        tracking_direction_norm = float(
            gradient_list_norm(tracking_action_directions))
        external_horizon_direction_norm = float(
            gradient_list_norm(external_horizon_directions))
        feasibility_horizon_direction_norm = float(
            gradient_list_norm(feasibility_horizon_directions))
        ghost_step = ghost_learner.apply(
            external_intent_directions,
            feasibility_directions,
            tracking_action_directions,
            external_horizon_directions,
            feasibility_horizon_directions,
            entropy_directions,
        )
        polyak_update(
            target_encoder, encoder, cfg.encoder_target_tau)
        if critic_loss is not None:
            polyak_update(
                target_trajectory_critic,
                trajectory_critic,
                cfg.trajectory_critic_target_tau,
            )

        finite_tensors = {
            "spline controls": contender_control_points,
            "spline control log variance": contender_control_logvars,
            "rendered contender trajectory": contender_waypoints,
            "rendered contender log variance": contender_logvars,
            "leader score": leader_q,
            "contender score": contender_q,
            "tracking reward": tracking_reward,
            "action": executed_action,
        }
        if predictor_loss is not None:
            finite_tensors["predictor loss"] = predictor_loss
        if critic_loss is not None:
            finite_tensors["critic loss"] = critic_loss
        for name, value in finite_tensors.items():
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(
                    f"non-finite {name} at transition {transitions}")
        for name, model in (
            ("encoder", encoder),
            ("ghost", ghost),
            ("trajectory predictor", trajectory_predictor),
            ("trajectory critic", trajectory_critic),
            ("target encoder", target_encoder),
            ("target trajectory critic", target_trajectory_critic),
        ):
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"non-finite {name} parameter at "
                    f"transition {transitions}"
                )
        finite_scalars = {
            "encoder update norm": encoder_step,
            "predictor update norm": predictor_step,
            "critic update norm": critic_step,
            "Ghost update norm": ghost_step,
            "external intent direction norm":
                external_intent_direction_norm,
            "feasibility direction norm": feasibility_direction_norm,
            "tracking direction norm": tracking_direction_norm,
            "external horizon direction norm":
                external_horizon_direction_norm,
            "feasibility horizon direction norm":
                feasibility_horizon_direction_norm,
        }
        for name, value in finite_scalars.items():
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"non-finite {name} at transition {transitions}")

        done_indices = np.flatnonzero(transition.done)
        for world in done_indices:
            recent_episode_success.append(
                float(transition.success[world]))
            leaders[world] = None
            pending_td[world] = None
            previous_tracking_reward[world] = 0
            exhausted_since_last_step[world] = False
            plan_versions[world] += 1
        if bool(done_tensor.any()):
            ghost.reset(done_tensor)
            ghost_learner.reset(done_tensor)
            trajectory_predictor.reset(done_tensor)
            trajectory_critic.reset(done_tensor)
            target_trajectory_critic.reset(done_tensor)

        advanced_curriculum = False
        if (
            len(recent_episode_success)
            == recent_episode_success.maxlen
            and np.mean(recent_episode_success)
            >= cfg.curriculum_threshold
            and env.curriculum_stage
            < env.curriculum_stage_count - 1
        ):
            env.set_curriculum_stage(env.curriculum_stage + 1)
            recent_episode_success.clear()
            reset_all = torch.ones(
                b, dtype=torch.bool, device=device)
            ghost.reset(reset_all)
            ghost_learner.reset(reset_all)
            trajectory_predictor.reset(reset_all)
            trajectory_critic.reset(reset_all)
            target_trajectory_critic.reset(reset_all)
            leaders = [None for _ in range(b)]
            pending_td = [None for _ in range(b)]
            previous_tracking_reward.zero_()
            exhausted_since_last_step.zero_()
            plan_versions.add_(1)
            advanced_curriculum = True

        with torch.no_grad():
            critic_membrane_norm = (
                0.0 if trajectory_critic.core.mem is None
                else float(
                    trajectory_critic.core.mem.norm(
                        dim=-1).mean())
            )
            predictor_membrane_norm = (
                0.0 if trajectory_predictor.core.mem is None
                else float(
                    trajectory_predictor.core.mem.norm(
                        dim=-1).mean())
            )
            rendered_steps = torch.zeros_like(
                contender_waypoints.detach())
            rendered_steps[:, 0] = (
                contender_waypoints[:, 0].detach()
                - contender_control_points[:, 0].detach()
            )
            if k_max > 1:
                rendered_steps[:, 1:] = (
                    contender_waypoints[:, 1:].detach()
                    - contender_waypoints[:, :-1].detach()
                )
            rendered_steps.mul_(contender_mask[:, :, None])
            step_norms = rms(rendered_steps)
            trajectory_step_history.append(
                step_norms.mean(0).cpu().numpy())
            control_increments = (
                contender_control_points[:, 1:]
                - contender_control_points[:, :-1]
            )
            control_increment_norms = rms(control_increments)
            endpoint_displacement = rms(
                contender_control_points[:, -1]
                - contender_control_points[:, 0]
            )
            if cfg.spline_control_points >= 3:
                control_curvature = rms(
                    contender_control_points[:, 2:]
                    - 2 * contender_control_points[:, 1:-1]
                    + contender_control_points[:, :-2]
                )
            else:
                control_curvature = torch.zeros(
                    b, 1, device=device)
            valid_basis = spline_basis_table[spline_mask_table]
            spline_basis_error = (
                valid_basis.sum(-1) - 1).abs().max()
            start_basis = bspline_basis(
                torch.zeros(b, device=device),
                cfg.spline_control_points,
                cfg.spline_degree,
                device,
                contender_control_points.dtype,
            )
            rendered_start = torch.einsum(
                "bc,bcl->bl",
                start_basis,
                contender_control_points,
            )
            spline_anchor_error = rms(
                rendered_start
                - contender_control_points[:, 0]
            )
        values = {
            "reward": float(reward_tensor.mean()),
            "success": (
                float(np.mean(transition.success[transition.done]))
                if bool(transition.done.any()) else float("nan")
            ),
            "trajectory_predictor_loss": (
                float(predictor_loss.detach())
                if predictor_loss is not None else float("nan")
            ),
            "trajectory_critic_loss": (
                float(critic_loss.detach())
                if critic_loss is not None else float("nan")
            ),
            "absolute_td_error": (
                float(np.mean(td_errors))
                if td_errors else float("nan")
            ),
            "leader_q": float(
                leader_q.detach()[leader_valid].mean())
            if bool(leader_valid.any()) else float("nan"),
            "contender_q": float(contender_q.detach().mean()),
            "adjusted_leader": float(
                leader_adjusted.detach()[leader_valid].mean())
            if bool(leader_valid.any()) else float("nan"),
            "adjusted_contender": float(
                contender_adjusted.detach().mean()),
            "replacement_rate": float(
                choose_contender.float().mean()),
            "leader_remaining_horizon": float(
                (
                    leader_selected_horizon[leader_valid]
                    - leader_cursors[leader_valid]
                ).float().mean())
            if bool(leader_valid.any()) else 0.0,
            "selected_horizon": float(
                selected_horizon.float().mean()),
            "leader_cursor_fraction": float(
                (
                    selected_cursor.float()
                    / selected_horizon.float().clamp_min(1)
                ).mean()
            ),
            "waypoint_distance_before": float(distance_before.mean()),
            "waypoint_distance_after": float(distance_after.mean()),
            "tracking_reward": float(tracking_reward.mean()),
            "tracking_advantage": float(tracking_advantage.mean()),
            "waypoint_prediction_error": (
                float(np.mean(waypoint_prediction_errors))
                if waypoint_prediction_errors else float("nan")
            ),
            "trajectory_feasibility_direction": (
                feasibility_direction_norm),
            "action_tracking_direction": tracking_direction_norm,
            "external_intent_direction": (
                external_intent_direction_norm),
            "external_horizon_direction": (
                external_horizon_direction_norm),
            "feasibility_horizon_direction": (
                feasibility_horizon_direction_norm),
            "leader_uncertainty": float(
                leader_feedback_uncertainty[
                    leader_valid].mean())
            if bool(leader_valid.any()) else 0.0,
            "contender_uncertainty": float(
                contender_uncertainty.detach().mean()),
            "switching_cost": float(switching_cost.mean()),
            "action_entropy": float(
                normal.entropy().detach().mean()),
            "action_saturation": float(
                (executed_action.detach().abs() > 0.98).float().mean()),
            "absolute_action": float(
                executed_action.detach().abs().mean()),
            "ghost_step": ghost_step,
            "encoder_step": encoder_step,
            "predictor_step": predictor_step,
            "critic_step": critic_step,
            "critic_membrane_norm": critic_membrane_norm,
            "predictor_membrane_norm": predictor_membrane_norm,
            "forced_replacement_rate": float(
                forced_by_exhaustion.float().mean()),
            "rejected_contender_rate": float(
                (~choose_contender).float().mean()),
            "degenerate_waypoint_rate": float(
                degenerate.float().mean()),
            "spline_arc_length": float(
                contender_arc_length.detach().mean()),
            "spline_endpoint_displacement": float(
                endpoint_displacement.detach().mean()),
            "mean_control_increment": float(
                control_increment_norms.detach().mean()),
            "max_control_increment": float(
                control_increment_norms.detach().max()),
            "spline_curvature": float(
                control_curvature.detach().mean()),
            "degenerate_contender_rate": float(
                contender_spline_degenerate.float().mean()),
            "degenerate_leader_rate": (
                float(
                    leader_spline_degenerate[
                        leader_valid].float().mean())
                if bool(leader_valid.any()) else 0.0
            ),
            "current_leader_phase": (
                float(
                    (
                        leader_cursors[leader_valid].float()
                        / leader_selected_horizon[
                            leader_valid].float().clamp_min(1)
                    ).mean()
                )
                if bool(leader_valid.any()) else 0.0
            ),
            "selected_phase_displacement": float(
                selected_phase_displacement.detach().mean()),
            "control_intent_jacobian_norm": float(
                control_intent_jacobian.detach().square().sum(
                    dim=(1, 2, 3)).sqrt().mean()),
            "waypoint_intent_jacobian_norm": (
                float(np.mean(waypoint_intent_jacobian_norms))
                if waypoint_intent_jacobian_norms else 0.0
            ),
            "spline_basis_error": float(spline_basis_error),
            "spline_anchor_error": float(
                spline_anchor_error.mean()),
        }
        for name, value in values.items():
            if np.isfinite(value):
                metric_windows[name].append(value)

        transition_after_step = transitions + b
        if transition_after_step >= next_plot_transition:
            while next_plot_transition <= transition_after_step:
                next_plot_transition += max(cfg.plot_every, 1)
            plot_history["transition"].append(transition_after_step)
            plot_history["curriculum_stage"].append(
                env.curriculum_stage)
            for name in metric_names:
                plot_history[name].append(rolling_metric(name))
            render_training_dashboard()

        observation = (
            env.observation()
            if advanced_curriculum else transition.observation
        )
        transitions += b

    final_path = Path(cfg.checkpoint_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": "rolling_leader_contender_bspline",
        "recurrent_runtime_state_saved": False,
        "live_leaders_saved": False,
        "config": asdict(cfg),
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "ghost": ghost.state_dict(),
        "trajectory_predictor": trajectory_predictor.state_dict(),
        "trajectory_critic": trajectory_critic.state_dict(),
        "target_trajectory_critic": (
            target_trajectory_critic.state_dict()),
        "optimizers": {
            "encoder": encoder_optimizer.state_dict(),
            "trajectory_predictor": (
                trajectory_predictor_optimizer.state_dict()),
            "trajectory_critic": (
                trajectory_critic_optimizer.state_dict()),
            "ghost": ghost_learner.optimizer.state_dict(),
        },
        "tracking_baseline": tracking_baseline.detach().cpu(),
        "external_horizon_baseline": (
            external_horizon_baseline.detach().cpu()),
        "feasibility_horizon_baseline": (
            feasibility_horizon_baseline.detach().cpu()),
        "previous_tracking_reward": (
            previous_tracking_reward.detach().cpu()),
        "curriculum_stage": env.curriculum_stage,
        "mean_trajectory_step_norm": (
            np.mean(trajectory_step_history, axis=0)
            if trajectory_step_history else np.zeros(k_max)
        ),
    }, final_path)
    print(f"saved={final_path}")
    return

if __name__ == "__main__":
    app.run()
