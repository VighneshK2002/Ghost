import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Imports
    """)
    return


@app.cell
def _():
    import marimo as mo
    from dataclasses import asdict, dataclass
    from pathlib import Path
    from typing import List, NamedTuple, Sequence
    from collections import deque
    import copy
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return (
        F,
        List,
        NamedTuple,
        Path,
        Sequence,
        asdict,
        copy,
        dataclass,
        deque,
        math,
        mo,
        nn,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Config
    """)
    return


@app.cell
def _(dataclass):
    @dataclass
    class Config:
        seed: int = 11
        device: str = "cpu"
        checkpoint_path: str = "Ghost/ghost_checkpoint.pt"
        worlds: int = 24
        transitions: int = 163_840
        observation_dim: int = 3
        latent_dim: int = 16
        ghost_dim: int = 8
        hidden_dim: int = 40
        conditioning_dim: int = 24
        snn_ticks: int = 5
        membrane_decay: float = 0.90
        predictor_membrane_decay: float = 0.97
        surrogate_scale: float = 0.30
        gamma: float = 0.99
        actor_trace_decay: float = 0.95
        ghost_trace_decay: float = 0.99
        encoder_lr: float = 3e-4
        predictor_lr: float = 3e-4
        actor_eprop_lr: float = 1e-4
        ghost_eprop_lr: float = 1e-4
        state_critic_lr: float = 3e-4
        critic_updates_per_step: int = 1
        critic_gradient_clip: float = 1.0
        freeze_critic_below_td: bool = False
        critic_freeze_td_threshold: float = 0.05
        freeze_all_above_success: bool = False
        freeze_all_success_threshold: float = 0.90

        initial_action_std: float = 0.05
        actor_min_std: float = 0.03
        actor_max_std: float = 0.25
        actor_fixed_std: float = 0.05
        actor_std_warmup_transitions: int = 0

        # Deprecated compatibility fields; value-adaptive exploration is no
        # longer active.
        use_value_adaptive_exploration: bool = False
        adaptive_exploration_min_std: float = 0.03
        adaptive_exploration_max_std: float = 0.25
        adaptive_exploration_warmup_transitions: int = 20_000
        adaptive_exploration_stage_warmup_transitions: int = 2_048
        exploration_value_ema_decay: float = 0.99
        exploration_value_temperature: float = 1.0
        exploration_td_ema_decay: float = 0.99
        exploration_reliable_td_threshold: float = 0.10
        exploration_td_temperature: float = 0.025

        encoder_latent_decay: float = 0.80
        encoder_membrane_readout: float = 0.05

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

        target_tau: float = 0.005
        encoder_target_tau: float = 0.005
        encoder_actor_credit_weight: float = 0.0
        encoder_critic_credit_weight: float = 0.05
        encoder_critic_max_predictor_ratio: float = 0.25
        encoder_critic_warmup_transitions: int = 20_000
        actor_to_ghost_credit_weight: float = 1.0
        ghost_gradient_clip: float = 1.0
        ghost_predictive_credit_weight: float = 0.10
        ghost_predictive_max_policy_ratio: float = 0.25
        ghost_predictive_warmup_transitions: int = 20_000

        use_world_model_curiosity: bool = True
        curiosity_warmup_transitions: int = 0
        curiosity_error_scale: float = 0.01
        curiosity_progress_clip: float = 1.0
        curiosity_baseline_decay: float = 0.99
        actor_curiosity_credit_weight: float = 0.10
        ghost_curiosity_credit_weight: float = 0.50
        actor_curiosity_max_external_ratio: float = 0.25
        ghost_curiosity_max_external_ratio: float = 1.00

        horizon_uniform_eta: float = 0.10
        actual_horizon_min_probability: float = 0.02
        predictor_logvar_min: float = -5.0
        predictor_logvar_max: float = 2.0
        return_loss_weight: float = 1.0
        plot_every: int = 1_024
        plot_window: int = 128

        # Weak online diagonal-Fisher consolidation for actor parameters only.
        use_actor_fisher: bool = True
        actor_fisher_lambda: float = 0.15
        actor_fisher_decay: float = 0.9995
        actor_fisher_warmup_transitions: int = 20_000
        actor_fisher_activation_success: float = 0.55
        actor_fisher_anchor_lr: float = 1e-4
        actor_fisher_importance_scale: float = 10.0
        actor_fisher_normalized_max: float = 20.0
        actor_fisher_stage_initial_multiplier: float = 0.25
        actor_fisher_stage_recovery_transitions: int = 10_000

        # Dormant compatibility fields. Encoder Fisher is disabled because
        # actor-TD gradients no longer update or define importance for encoder.
        use_encoder_fisher: bool = False
        encoder_fisher_lambda: float = 0.0
        encoder_fisher_decay: float = 0.9995
        encoder_fisher_warmup_transitions: int = 20_000
        encoder_fisher_activation_success: float = 0.55
        encoder_fisher_anchor_lr: float = 1e-4
        encoder_fisher_importance_scale: float = 10.0
        encoder_fisher_normalized_max: float = 20.0
        encoder_fisher_stage_initial_multiplier: float = 0.25
        encoder_fisher_stage_recovery_transitions: int = 10_000
    RELATIVE_HORIZONS = (0.05, 0.10, 0.20, 0.40, 0.70, 1.00)
    return Config, RELATIVE_HORIZONS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # MLP Encoder
    """)
    return


@app.cell
def _(nn):
    def mlp(input_dim: int, hidden: int, output_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden),
                             nn.GELU(), nn.Linear(hidden, output_dim), nn.Tanh())


    return (mlp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Recurrent Spiking Neural Network
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
            return (gradient*ctx.scale
                    *torch.clamp(1-voltage.abs(), min=0)), None


    class RecurrentSNN(nn.Module):

        def __init__(self, input_dim: int, hidden_dim: int, cfg: Config, persistent: bool,
                     decay: float | None = None,  record_eligibility: bool = False) -> None:
            super().__init__()
            self.cfg, self.hidden_dim, self.persistent = cfg, hidden_dim,persistent
            self.decay = cfg.membrane_decay if decay is None else decay
            self.record_eligibility = record_eligibility
            self.input = nn.Linear(input_dim, hidden_dim)
            self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.orthogonal_(self.recurrent.weight, gain = 0.35)
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


    class ScoreEligibility:
        def __init__(self, parameters: Sequence[nn.Parameter], cfg: Config):
            self.parameters = list(parameters); self.decay = cfg.actor_trace_decay
            self.traces = [torch.zeros(
                (cfg.worlds,)+tuple(parameter.shape), device=parameter.device)
                for parameter in self.parameters]
            self.optimizer = torch.optim.Adam(
                self.parameters, lr=cfg.actor_eprop_lr, maximize=True)

        def accumulate(self, logp: torch.Tensor):
            basis = torch.eye(logp.shape[0], device=logp.device, dtype=logp.dtype)
            gradients = torch.autograd.grad(
                logp, self.parameters, grad_outputs=basis,
                is_grads_batched=True, retain_graph=True, allow_unused=True)
            with torch.no_grad():
                for trace, gradient in zip(self.traces, gradients):
                    trace.mul_(self.decay)
                    if gradient is not None:
                        trace.add_(gradient.detach())

        def reset(self, mask: torch.Tensor) -> None:
            if bool(mask.any()):
                with torch.no_grad():
                    for trace in self.traces:
                        trace[mask] = 0


    def clip_gradient_list(gradients, max_norm: float):
        """Clip an explicit gradient list without attaching an autograd graph."""
        present = [value for value in gradients if value is not None]
        if not present:
            return gradients
        raw = torch.stack([value.square().sum() for value in present]).sum().sqrt()
        scale = min(1.0, float(max_norm)/max(float(raw), 1e-12))
        return [None if value is None else value*scale for value in gradients]

    @torch.no_grad()
    def polyak_update(target, online, tau):
        for target_parameter, parameter in zip(
                target.parameters(), online.parameters()):
            target_parameter.mul_(1-tau).add_(parameter, alpha=tau)


    return (
        RecurrentSNN,
        ScoreEligibility,
        SurrogateSpike,
        clip_gradient_list,
        polyak_update,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ghost
    """)
    return


@app.cell
def _(Config, List, RELATIVE_HORIZONS, RecurrentSNN, Sequence, nn, torch):
    #Ghost
    class Ghost(nn.Module):
        """Persistent recurrent SNN whose current readout is the live ghost."""

        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg
            input_dim = 2*cfg.latent_dim+2
            self.core = RecurrentSNN(
                input_dim, cfg.hidden_dim, cfg, persistent=True,
                record_eligibility=True)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.ghost_head = nn.Linear(2*cfg.hidden_dim, cfg.ghost_dim)
            self.horizon_head = nn.Linear(
                2*cfg.hidden_dim, len(RELATIVE_HORIZONS))
            nn.init.normal_(self.ghost_head.weight, std=0.02)
            nn.init.zeros_(self.ghost_head.bias)
            nn.init.zeros_(self.horizon_head.bias)

        def forward(self, latent, previous_endpoint,
                    previous_advantage, previous_uncertainty):
            value = torch.cat((
                latent, previous_endpoint, previous_advantage[:, None],
                previous_uncertainty[:, None]), -1)
            feature = self.norm(self.core(value))
            ghost = torch.tanh(self.ghost_head(feature))
            horizon_logits = self.horizon_head(feature)
            return ghost, horizon_logits

        def reset(self, mask):
            self.core.reset(mask)


    class RecurrentGhostEprop:
        """Bellec-style LIF output Jacobians for the recurrent ghost."""

        def __init__(self, ghost: Ghost,
                     parameters: Sequence[nn.Parameter], worlds: int,
                     learning_rate: float) -> None:
            self.ghost = ghost
            self.parameters = list(parameters)
            cfg = ghost.cfg
            b, h = worlds, cfg.hidden_dim
            d = ghost.core.input.in_features
            device = next(ghost.parameters()).device
            self.epsilon_in = torch.zeros(b, h, d, device=device)
            self.epsilon_rec = torch.zeros(b, h, h, device=device)
            self.epsilon_bias = torch.zeros(b, h, device=device)
            self.optimizer = torch.optim.Adam(
                self.parameters, lr=learning_rate, maximize=True)
            core = ghost.core
            self.core_parameter_kind = {
                id(core.input.weight): "input_weight",
                id(core.input.bias): "input_bias",
                id(core.recurrent.weight): "recurrent_weight",
                id(core.bias): "bias",
            }
            self.other_indices = [index for index, parameter in
                                  enumerate(self.parameters)
                                  if id(parameter) not in self.core_parameter_kind]

        def _current_output_jacobians(
                self, output: torch.Tensor) -> List[torch.Tensor]:
            cfg, core = self.ghost.cfg, self.ghost.core
            b, output_dim, h = output.shape[0], output.shape[1], cfg.hidden_dim
            current = [torch.zeros(
                (b, output_dim)+tuple(parameter.shape), device=output.device)
                for parameter in self.parameters]
            if core.last_output is None:
                raise RuntimeError("ghost core has no eligibility output")

            # Spatial learning signal from each ghost coordinate to the
            # ghost's mean membrane/spike feature.  Batch elements are
            # independent, so differentiating the coordinate sum gives one local
            # derivative per world without mixing samples.
            feature_gradients = []
            for coordinate in range(output_dim):
                feature_gradients.append(torch.autograd.grad(
                    output[:, coordinate].sum(), core.last_output,
                    retain_graph=True)[0].detach())
            feature_gradient = torch.stack(feature_gradients, 1)

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
            # output Jacobian.
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
            return current

        def reset(self, mask: torch.Tensor) -> None:
            if not bool(mask.any()):
                return
            with torch.no_grad():
                self.epsilon_in[mask] = 0
                self.epsilon_rec[mask] = 0
                self.epsilon_bias[mask] = 0


    class GhostEligibility:
        """Actor-score eligibility through the live recurrent ghost output."""

        def __init__(self, ghost: Ghost, cfg: Config):
            self.parameters = list(ghost.parameters())
            self.decay = cfg.ghost_trace_decay
            self.weight = cfg.actor_to_ghost_credit_weight
            self.max_norm = cfg.ghost_gradient_clip
            self.predictive_weight = cfg.ghost_predictive_credit_weight
            self.predictive_max_policy_ratio = (
                cfg.ghost_predictive_max_policy_ratio)
            self.curiosity_weight = cfg.ghost_curiosity_credit_weight
            self.curiosity_max_external_ratio = (
                cfg.ghost_curiosity_max_external_ratio)
            self.recurrent_eprop = RecurrentGhostEprop(
                ghost, self.parameters, cfg.worlds,
                cfg.ghost_eprop_lr)
            self.optimizer = self.recurrent_eprop.optimizer
            self.reward_traces = [
                torch.zeros((cfg.worlds,)+tuple(parameter.shape),
                            device=parameter.device)
                for parameter in self.parameters]

        def accumulate(self, ghost, actor_logp):
            if ghost.grad_fn is None:
                raise RuntimeError(
                    "live ghost was detached before actor-mediated credit")
            jacobians = self.recurrent_eprop._current_output_jacobians(ghost)
            ghost_signal = torch.autograd.grad(
                actor_logp.sum(), ghost, retain_graph=True)[0].detach()
            with torch.no_grad():
                for jacobian, trace in zip(jacobians, self.reward_traces):
                    view = ghost_signal.shape+(1,)*(jacobian.ndim-2)
                    score = (jacobian*ghost_signal.view(view)).sum(1)
                    trace.mul_(self.decay).add_(score)

        @staticmethod
        def _global_norm(values):
            return torch.stack([
                value.square().sum() for value in values]).sum().sqrt()

        def apply(
                self, actor_td, curiosity_advantage=None,
                predictive_gradients=None,
                predictive_record_count=0, predictive_loss=float("nan")):
            external_policy_directions = []
            for trace in self.reward_traces:
                view = (len(actor_td),)+(1,)*(trace.ndim-1)
                external_policy_directions.append(
                    self.weight*(trace*actor_td.view(view)).mean(0))
            external_policy_norm = self._global_norm(
                external_policy_directions)
            if curiosity_advantage is None:
                curiosity_advantage = torch.zeros_like(actor_td)
            curiosity_advantage = curiosity_advantage.detach()
            raw_curiosity_directions = []
            for trace in self.reward_traces:
                view = (
                    (len(curiosity_advantage),)
                    +(1,)*(trace.ndim-1))
                raw_curiosity_directions.append(
                    (trace*curiosity_advantage.view(view)).mean(0))
            raw_curiosity_norm = self._global_norm(
                raw_curiosity_directions)
            weighted_curiosity_directions = [
                self.curiosity_weight*direction
                for direction in raw_curiosity_directions]
            weighted_curiosity_norm = self._global_norm(
                weighted_curiosity_directions)
            if float(external_policy_norm) == 0.0:
                curiosity_scale = 0.0
            else:
                curiosity_scale = min(
                    1.0,
                    (self.curiosity_max_external_ratio
                     *float(external_policy_norm))
                    /(float(weighted_curiosity_norm)+1e-12))
            behavioral_directions = [
                external_direction
                +curiosity_scale*curiosity_direction
                for external_direction, curiosity_direction in zip(
                    external_policy_directions,
                    weighted_curiosity_directions)]
            behavioral_norm = self._global_norm(
                behavioral_directions)
            if predictive_gradients is None:
                predictive_gradients = [
                    torch.zeros_like(parameter)
                    for parameter in self.parameters]
            weighted_predictive_gradients = [
                self.predictive_weight*gradient.detach()
                for gradient in predictive_gradients]
            weighted_predictive_norm = self._global_norm(
                weighted_predictive_gradients)
            if float(behavioral_norm) == 0.0:
                predictive_scale = 0.0
            else:
                predictive_scale = min(
                    1.0,
                    (self.predictive_max_policy_ratio
                     *float(behavioral_norm))
                    /(float(weighted_predictive_norm)+1e-12))
            joint_directions = [
                behavioral_direction
                -predictive_scale*weighted_predictive_gradient
                for behavioral_direction,
                weighted_predictive_gradient in zip(
                    behavioral_directions,
                    weighted_predictive_gradients)]
            joint_norm = self._global_norm(joint_directions)
            joint_scale = min(
                1.0, self.max_norm/max(float(joint_norm), 1e-12))
            self.optimizer.zero_grad(set_to_none=True)
            before = [parameter.detach().clone()
                      for parameter in self.parameters]
            for parameter, direction in zip(
                    self.parameters, joint_directions):
                parameter.grad = direction*joint_scale
            self.optimizer.step()
            step = torch.stack([
                (parameter-old).square().sum()
                for parameter, old in zip(self.parameters, before)
            ]).sum().sqrt()
            applied_predictive_norm = (
                predictive_scale*float(weighted_predictive_norm))
            predictive_ratio = (
                applied_predictive_norm/float(behavioral_norm)
                if float(behavioral_norm) > 0.0 else 0.0)
            applied_curiosity_norm = (
                curiosity_scale*float(weighted_curiosity_norm))
            curiosity_ratio = (
                applied_curiosity_norm/float(external_policy_norm)
                if float(external_policy_norm) > 0.0 else 0.0)
            return {
                "step_norm": float(step),
                "policy_gradient_norm": float(
                    external_policy_norm),
                "external_gradient_norm": float(
                    external_policy_norm),
                "curiosity_raw_gradient_norm": float(
                    raw_curiosity_norm),
                "curiosity_applied_gradient_norm": (
                    applied_curiosity_norm),
                "curiosity_external_ratio": curiosity_ratio,
                "weighted_predictive_gradient_norm": float(
                    weighted_predictive_norm),
                "applied_predictive_gradient_norm": (
                    applied_predictive_norm),
                "predictive_to_policy_ratio": predictive_ratio,
                "predictive_record_count": int(
                    predictive_record_count),
                "predictive_loss": float(predictive_loss),
            }

        def reset(self, mask):
            if not bool(mask.any()):
                return
            with torch.no_grad():
                for trace in self.reward_traces:
                    trace[mask] = 0
            self.recurrent_eprop.reset(mask)





    return Ghost, GhostEligibility


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Actor
    """)
    return


@app.cell
def _(Config, RecurrentSNN, math, mlp, nn, torch):
    class Actor(nn.Module):
        """Stateless Gaussian policy conditioned on the current ghost."""

        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg
            active_std = (cfg.actor_fixed_std if cfg.actor_fixed_std > 0
                          else cfg.initial_action_std)
            if not 0 < cfg.actor_min_std <= active_std <= cfg.actor_max_std:
                raise ValueError(
                    "actor standard deviations must satisfy 0 < min <= active <= max")
            self.ghost_encoder = mlp(
                cfg.ghost_dim, cfg.hidden_dim, cfg.conditioning_dim)
            self.core = RecurrentSNN(
                cfg.latent_dim+cfg.conditioning_dim+cfg.ghost_dim,
                cfg.hidden_dim, cfg, persistent=False)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.mean_head = nn.Linear(2*cfg.hidden_dim, 1)
            self.log_std_head = nn.Linear(2*cfg.hidden_dim, 1)
            nn.init.normal_(self.mean_head.weight, std=0.01)
            nn.init.zeros_(self.mean_head.bias)
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.zeros_(self.log_std_head.bias)
            self.log_std = nn.Parameter(torch.tensor(
                math.log(cfg.initial_action_std), dtype=torch.float32))

        def forward(
                self, latent: torch.Tensor, ghost: torch.Tensor,
                deterministic: bool = False):
            conditioning = self.ghost_encoder(ghost)
            feature = self.norm(self.core(torch.cat(
                (latent, conditioning, ghost), -1)))
            raw_mean = self.mean_head(feature).squeeze(-1)
            mean = torch.tanh(raw_mean)
            learned_log_std = (
                self.log_std
                +self.log_std_head(feature).squeeze(-1)).clamp(
                    math.log(self.cfg.actor_min_std),
                    math.log(self.cfg.actor_max_std))
            if self.cfg.actor_fixed_std > 0:
                std = torch.full_like(
                    mean, self.cfg.actor_fixed_std)
                # Fixed exploration is injected directly in bounded action space.
                distribution = torch.distributions.Normal(mean, std)
                if deterministic:
                    pre_action = mean
                    action = mean
                else:
                    pre_action = distribution.sample()
                    action = pre_action.clamp(-1, 1)
                logp = distribution.log_prob(pre_action)
                behavior_log_std = std.log()
            else:
                std = learned_log_std.exp()
                distribution = torch.distributions.Normal(raw_mean, std)
                pre_action = (
                    raw_mean if deterministic else distribution.sample())
                action = torch.tanh(pre_action)
                logp = distribution.log_prob(pre_action)-torch.log(
                    1-action.square()+1e-6)
                behavior_log_std = learned_log_std
            return {
                "action": action,
                "logp": logp,
                "mean": mean,
                "raw_mean": raw_mean,
                "std": std,
                "log_std": learned_log_std,
                "behavior_log_std": behavior_log_std,
            }

    return (Actor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Latent Encoder
    """)
    return


@app.cell
def _(Config, RecurrentSNN, SurrogateSpike, nn, torch):
    class Encoder(nn.Module):
        """Stateless spiking latent encoder."""

        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg
            self.core = RecurrentSNN(
                cfg.observation_dim, cfg.hidden_dim, cfg, persistent=False)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.latent_current = nn.Linear(
                2*cfg.hidden_dim, cfg.latent_dim, bias=False)
            nn.init.normal_(self.latent_current.weight,
                            std=0.12/max((2*cfg.hidden_dim)**0.5, 1.0))

        def forward(self, observation: torch.Tensor):
            feature = self.norm(self.core(observation))
            raw = self.latent_current(feature)
            # Preserve the fixed operating point previously used while the
            # adaptive controller was disabled: 0.20/sqrt(0.01) == 2.0.
            current = 2.0*raw+0.34
            membrane = torch.zeros_like(current); spikes = torch.zeros_like(current)
            spike_sum = torch.zeros_like(current)
            for _ in range(self.cfg.snn_ticks):
                membrane = (self.cfg.encoder_latent_decay*membrane+current
                            -spikes)
                membrane = 12.0*torch.tanh(membrane/12.0)
                spikes = SurrogateSpike.apply(
                    membrane-1.0, self.cfg.surrogate_scale)
                spike_sum += spikes
            spike_rate = spike_sum/self.cfg.snn_ticks
            latent = spike_rate+self.cfg.encoder_membrane_readout*torch.tanh(
                membrane)
            return latent


    class EncoderScoreTrace:
        """Online actor-score eligibility for all current encoder policy paths."""

        def __init__(self, parameters, worlds, decay):
            self.parameters = list(parameters); self.decay = decay
            self.traces = [
                torch.zeros((worlds,)+tuple(parameter.shape),
                            device=parameter.device)
                for parameter in self.parameters]

        def accumulate(self, score):
            basis = torch.eye(
                len(score), device=score.device, dtype=score.dtype)
            gradients = torch.autograd.grad(
                score, self.parameters, grad_outputs=basis,
                is_grads_batched=True, retain_graph=True, allow_unused=True)
            with torch.no_grad():
                for trace, gradient in zip(self.traces, gradients):
                    trace.mul_(self.decay)
                    if gradient is not None:
                        trace.add_(gradient.detach())

        def direction(self, credit):
            result = []
            for trace in self.traces:
                view = (len(credit),)+(1,)*(trace.ndim-1)
                result.append((trace*credit.view(view)).mean(0))
            return result

        def reset(self, mask):
            if bool(mask.any()):
                with torch.no_grad():
                    for trace in self.traces:
                        trace[mask] = 0


    def slice_state(state, index: int):
        if state is None:
            return None
        return tuple(value[index:index+1].detach().clone() for value in state)


    return Encoder, slice_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Latent Predictor
    """)
    return


@app.cell
def _(Config, NamedTuple, RecurrentSNN, mlp, nn, torch):
    class Predictor(nn.Module):
        """Forecast under the evolving closed-loop policy from ghost state now."""

        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg
            self.ghost_encoder = mlp(
                cfg.ghost_dim, cfg.hidden_dim, cfg.conditioning_dim)
            self.core = RecurrentSNN(
                cfg.latent_dim+cfg.conditioning_dim+3, cfg.hidden_dim, cfg,
                persistent=True, decay=cfg.predictor_membrane_decay)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.mean_head = nn.Linear(2*cfg.hidden_dim, cfg.latent_dim)
            self.logvar_head = nn.Linear(2*cfg.hidden_dim, cfg.latent_dim)
            self.return_head = nn.Linear(2*cfg.hidden_dim, 1)

        def forward(
                self, latent, ghost, relative_horizon, budget_fraction,
                source_action_std):
            condition = self.ghost_encoder(ghost)
            source_action_std = source_action_std.detach().reshape(-1, 1)
            if source_action_std.shape != (latent.shape[0], 1):
                raise ValueError(
                    "source_action_std must contain one value per batch item")
            if (not bool(torch.isfinite(source_action_std).all())
                    or not bool((source_action_std > 0).all())):
                raise ValueError(
                    "source_action_std values must be finite and positive")
            feature = self.norm(self.core(torch.cat((
                latent, condition, relative_horizon[:, None],
                budget_fraction[:, None], source_action_std), -1)))
            mean = self.mean_head(feature)
            logvar = self.logvar_head(feature).clamp(
                self.cfg.predictor_logvar_min, self.cfg.predictor_logvar_max)
            predicted_return = self.return_head(feature).squeeze(-1)
            return mean, logvar, predicted_return

        def reset(self, mask):
            self.core.reset(mask)


    class PendingPrediction(NamedTuple):
        episode: int
        remaining: int
        source_observation: torch.Tensor
        source_ghost_intent: torch.Tensor
        relative_horizon: torch.Tensor
        budget_fraction: torch.Tensor
        predictor_state: object
        discounted_return: float
        discount: float
        source_latent: torch.Tensor
        ghost_pre_state: object
        source_previous_endpoint: torch.Tensor
        source_previous_advantage: torch.Tensor
        source_previous_uncertainty: torch.Tensor
        source_action_std: torch.Tensor



    return PendingPrediction, Predictor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Critic
    """)
    return


@app.cell
def _(Config, nn, torch):
    class StateValueCritic(nn.Module):
        """Plan-conditioned V(z, predicted endpoint delta, normalized remaining horizon)."""

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2*cfg.latent_dim+1, cfg.hidden_dim),
                nn.SiLU(), nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1))

        def forward(
                self, latent, predicted_delta,
                remaining_horizon_fraction):
            if remaining_horizon_fraction.ndim == 1:
                remaining_horizon_fraction = (
                    remaining_horizon_fraction[:, None])
            critic_input = torch.cat(
                (latent, predicted_delta, remaining_horizon_fraction),
                dim=-1)
            return self.net(critic_input).squeeze(-1)


    return (StateValueCritic,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ENV
    **PointControl**
    """)
    return


@app.cell
def _(Config, dataclass, np):
    class PointControl:
        def __init__(self, cfg: Config, seed: int) -> None:
            self.cfg = cfg; self.rng = np.random.default_rng(seed)
            b = cfg.worlds
            self.x = np.zeros(b, np.float32); self.goal = np.zeros(b, np.float32)
            self.age = np.zeros(b, np.int64); self.hold = np.zeros(b, np.int64)
            self.episode = np.zeros(b, np.int64)
            self.curriculum_stage = int(cfg.curriculum_stage)
            self.curriculum_stage_count = len(cfg.curriculum_distance_ranges)
            self.reset(np.ones(b, bool))

        @property
        def distance_range(self):
            return self.cfg.curriculum_distance_ranges[self.curriculum_stage]

        @property
        def hold_steps(self):
            return self.cfg.curriculum_hold_steps[self.curriculum_stage]

        @property
        def current_episode_limit(self):
            return self.cfg.curriculum_episode_limits[self.curriculum_stage]

        def set_curriculum_stage(self, stage: int) -> None:
            self.curriculum_stage = int(np.clip(
                stage, 0, self.curriculum_stage_count-1))
            self.cfg.curriculum_stage = self.curriculum_stage
            self.reset(np.ones(self.cfg.worlds, bool))

        def reset(self, mask: np.ndarray) -> None:
            for index in np.flatnonzero(mask):
                minimum, maximum = self.distance_range
                goal = self.rng.uniform(-1, 1)
                for _ in range(10_000):
                    x = self.rng.uniform(-1, 1)
                    distance = abs(x-goal)
                    if minimum <= distance <= maximum: break
                else:
                    raise RuntimeError("could not sample curriculum start state")
                self.goal[index] = goal; self.x[index] = x
                self.age[index] = 0; self.hold[index] = 0
                self.episode[index] += 1

        def observation(self):
            return np.stack((self.x, self.goal, self.goal-self.x), -1).astype(np.float32)

    @dataclass
    class Transition:
        observation: np.ndarray
        target_observation: np.ndarray
        dense_reward: np.ndarray
        done: np.ndarray
        success: np.ndarray
        episode: np.ndarray

    class TaskRewardPointControl(PointControl):
        """PointControl with one explicit shaped task reward."""

        def step(self, action: np.ndarray):
            action = np.asarray(action, np.float32).clip(-1, 1)
            episode = self.episode.copy()
            previous = self.goal-self.x
            self.x = np.clip(self.x+self.cfg.step_scale*action, -1.25, 1.25)
            error = self.goal-self.x
            progress = np.abs(previous)-np.abs(error)
            overshot = ((previous*error < 0)
                        &(np.abs(error) >= self.cfg.target_radius))
            inside = np.abs(error) < self.cfg.target_radius
            self.hold = np.where(inside, self.hold+1, 0); self.age += 1
            reached = self.hold >= self.hold_steps
            timeout = (self.age >= self.current_episode_limit) & ~reached
            shaping = (self.cfg.progress_weight*progress
                       -self.cfg.action_cost*np.square(action)
                       -self.cfg.overshoot_penalty*overshot.astype(np.float32))
            done = reached | timeout; success = reached.copy()
            terminal = np.zeros(self.cfg.worlds, np.float32)
            terminal[success] = 1.0; terminal[timeout] = -1.0
            task_reward = shaping.astype(np.float32)+terminal
            target = self.observation().copy()
            self.reset(done)
            return Transition(
                self.observation(), target, task_reward, done, success, episode)



    return (TaskRewardPointControl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #Train
    """)
    return


@app.cell
def _(Config, mo):
    defaults = Config()

    def number(value, label, *, step=None, start=None, stop=None):
        return mo.ui.number(
            value=value, label=label, step=step, start=start, stop=stop,
            full_width=True)

    training_form = mo.ui.dictionary({
        "seed": number(defaults.seed, "Run · Seed", step=1, start=0),
        "device": mo.ui.dropdown(
            ["cpu", "mps", "cuda"], value=defaults.device,
            label="Run · Device", full_width=True),
        "checkpoint_path": mo.ui.text(
            value=defaults.checkpoint_path,
            label="Run · Checkpoint path", full_width=True),
        "worlds": number(
            defaults.worlds, "Run · Parallel worlds", step=1, start=1),
        "transitions": number(
            defaults.transitions, "Run · Transitions", step=1, start=1),

        "latent_dim": number(
            defaults.latent_dim, "Network · Latent dimension",
            step=1, start=1),
        "ghost_dim": number(
            defaults.ghost_dim, "Network · Ghost dimension",
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
        "predictor_membrane_decay": number(
            defaults.predictor_membrane_decay,
            "Network · Predictor membrane decay",
            step=0.01, start=0.0, stop=1.0),
        "surrogate_scale": number(
            defaults.surrogate_scale, "Network · Surrogate scale",
            step=0.01, start=0.0),
        "encoder_latent_decay": number(
            defaults.encoder_latent_decay,
            "Network · Encoder latent decay",
            step=0.01, start=0.0, stop=1.0),
        "encoder_membrane_readout": number(
            defaults.encoder_membrane_readout,
            "Network · Encoder membrane readout",
            step=0.01, start=0.0),

        "encoder_lr": number(
            defaults.encoder_lr, "Learning · Encoder rate",
            step=1e-5, start=0.0),
        "state_critic_lr": number(
            defaults.state_critic_lr, "Learning · Critic rate",
            step=1e-5, start=0.0),
        "predictor_lr": number(
            defaults.predictor_lr, "Learning · Predictor rate",
            step=1e-5, start=0.0),
        "actor_eprop_lr": number(
            defaults.actor_eprop_lr, "Learning · Actor rate",
            step=1e-5, start=0.0),
        "ghost_eprop_lr": number(
            defaults.ghost_eprop_lr, "Learning · Ghost rate",
            step=1e-5, start=0.0),
        "critic_updates_per_step": number(
            defaults.critic_updates_per_step,
            "Learning · Critic updates per step",
            step=1, start=1),
        "critic_gradient_clip": number(
            defaults.critic_gradient_clip,
            "Learning · Critic gradient clip",
            step=0.1, start=0.0),
        "freeze_critic_below_td": mo.ui.switch(
            value=defaults.freeze_critic_below_td,
            label="Learning · Freeze critic below TD threshold"),
        "critic_freeze_td_threshold": number(
            defaults.critic_freeze_td_threshold,
            "Learning · Critic freeze TD threshold",
            step=0.01, start=0.0),
        "freeze_all_above_success": mo.ui.switch(
            value=defaults.freeze_all_above_success,
            label=(
                "Learning · Freeze all models above success threshold")),
        "freeze_all_success_threshold": number(
            defaults.freeze_all_success_threshold,
            "Learning · All-model freeze success threshold",
            step=0.01, start=0.0, stop=1.0),
        "actor_trace_decay": number(
            defaults.actor_trace_decay, "Learning · Actor trace decay",
            step=0.01, start=0.0, stop=1.0),
        "ghost_trace_decay": number(
            defaults.ghost_trace_decay, "Learning · Ghost trace decay",
            step=0.01, start=0.0, stop=1.0),
        "gamma": number(
            defaults.gamma, "Learning · Discount factor",
            step=0.001, start=0.0, stop=1.0),
        "target_tau": number(
            defaults.target_tau, "Learning · Critic target rate",
            step=0.001, start=0.0, stop=1.0),
        "encoder_target_tau": number(
            defaults.encoder_target_tau, "Learning · Encoder target rate",
            step=0.001, start=0.0, stop=1.0),
        "encoder_critic_credit_weight": number(
            defaults.encoder_critic_credit_weight,
            "Learning · Critic-to-encoder credit", step=0.01, start=0.0),
        "encoder_critic_max_predictor_ratio": number(
            defaults.encoder_critic_max_predictor_ratio,
            "Learning · Critic-to-encoder max predictor ratio",
            step=0.01, start=0.0),
        "encoder_critic_warmup_transitions": number(
            defaults.encoder_critic_warmup_transitions,
            "Learning · Critic-to-encoder warmup",
            step=1, start=0),
        "actor_to_ghost_credit_weight": number(
            defaults.actor_to_ghost_credit_weight,
            "Learning · Actor-to-ghost credit", step=0.05, start=0.0),
        "ghost_gradient_clip": number(
            defaults.ghost_gradient_clip,
            "Learning · Ghost gradient clip", step=0.1, start=0.0),
        "ghost_predictive_credit_weight": number(
            defaults.ghost_predictive_credit_weight,
            "Learning · Predictor-to-ghost credit",
            step=0.01, start=0.0),
        "ghost_predictive_max_policy_ratio": number(
            defaults.ghost_predictive_max_policy_ratio,
            "Learning · Predictor-to-ghost max policy ratio",
            step=0.01, start=0.0),
        "ghost_predictive_warmup_transitions": number(
            defaults.ghost_predictive_warmup_transitions,
            "Learning · Predictor-to-ghost warmup",
            step=1, start=0),
        "use_world_model_curiosity": mo.ui.switch(
            value=defaults.use_world_model_curiosity,
            label="Curiosity · Enable world-model progress"),
        "curiosity_warmup_transitions": number(
            defaults.curiosity_warmup_transitions,
            "Curiosity · Warmup transitions",
            step=1, start=0),
        "curiosity_error_scale": number(
            defaults.curiosity_error_scale,
            "Curiosity · Error gate scale",
            step=0.001, start=0.001),
        "curiosity_progress_clip": number(
            defaults.curiosity_progress_clip,
            "Curiosity · Progress clip",
            step=0.05, start=0.001),
        "curiosity_baseline_decay": number(
            defaults.curiosity_baseline_decay,
            "Curiosity · Baseline decay",
            step=0.01, start=0.0, stop=0.999),
        "actor_curiosity_credit_weight": number(
            defaults.actor_curiosity_credit_weight,
            "Curiosity · Actor credit",
            step=0.01, start=0.0),
        "ghost_curiosity_credit_weight": number(
            defaults.ghost_curiosity_credit_weight,
            "Curiosity · Ghost credit",
            step=0.05, start=0.0),
        "actor_curiosity_max_external_ratio": number(
            defaults.actor_curiosity_max_external_ratio,
            "Curiosity · Actor max external ratio",
            step=0.05, start=0.0),
        "ghost_curiosity_max_external_ratio": number(
            defaults.ghost_curiosity_max_external_ratio,
            "Curiosity · Ghost max external ratio",
            step=0.05, start=0.0),

        "initial_action_std": number(
            defaults.initial_action_std, "Actor · Initial std",
            step=0.01, start=0.001),
        "actor_min_std": number(
            defaults.actor_min_std, "Actor · Minimum std",
            step=0.01, start=0.001),
        "actor_max_std": number(
            defaults.actor_max_std, "Actor · Maximum std",
            step=0.01, start=0.001),

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
            defaults.action_cost, "Environment · Action cost",
            step=0.001, start=0.0),
        "overshoot_penalty": number(
            defaults.overshoot_penalty,
            "Environment · Overshoot penalty",
            step=0.01, start=0.0),

        "curriculum_stage": number(
            defaults.curriculum_stage, "Curriculum · Initial stage",
            step=1, start=0,
            stop=len(defaults.curriculum_distance_ranges)-1),
        "curriculum_threshold": number(
            defaults.curriculum_threshold,
            "Curriculum · Advancement threshold",
            step=0.01, start=0.0, stop=1.0),
        "curriculum_window_episodes": number(
            defaults.curriculum_window_episodes,
            "Curriculum · Episode window", step=1, start=1),

        "horizon_uniform_eta": number(
            defaults.horizon_uniform_eta,
            "Predictor · Uniform horizon mixing",
            step=0.01, start=0.0, stop=1.0),
        "actual_horizon_min_probability": number(
            defaults.actual_horizon_min_probability,
            "Predictor · Minimum horizon probability",
            step=0.01, start=0.0, stop=1.0),
        "predictor_logvar_min": number(
            defaults.predictor_logvar_min,
            "Predictor · Minimum log variance", step=0.1),
        "predictor_logvar_max": number(
            defaults.predictor_logvar_max,
            "Predictor · Maximum log variance", step=0.1),
        "return_loss_weight": number(
            defaults.return_loss_weight,
            "Predictor · Return loss weight", step=0.1, start=0.0),

        "use_actor_fisher": mo.ui.switch(
            value=defaults.use_actor_fisher,
            label="Fisher · Enable actor consolidation"),
        "actor_fisher_lambda": number(
            defaults.actor_fisher_lambda,
            "Fisher · Actor strength", step=0.01, start=0.0),
        "actor_fisher_warmup_transitions": number(
            defaults.actor_fisher_warmup_transitions,
            "Fisher · Actor warmup", step=1, start=0),
        "use_encoder_fisher": mo.ui.switch(
            value=defaults.use_encoder_fisher,
            label="Fisher · Encoder consolidation (inactive)"),
        "encoder_fisher_lambda": number(
            defaults.encoder_fisher_lambda,
            "Fisher · Encoder strength (inactive)",
            step=0.01, start=0.0),
        "encoder_fisher_warmup_transitions": number(
            defaults.encoder_fisher_warmup_transitions,
            "Fisher · Encoder warmup (inactive)", step=1, start=0),

        "plot_every": number(
            defaults.plot_every, "Dashboard · Refresh transitions",
            step=1, start=1),
        "plot_window": number(
            defaults.plot_window, "Dashboard · Rolling window",
            step=1, start=1),
    }).form(
        label="Training configuration",
        submit_button_label="Start training",
        submit_button_tooltip=(
            "Apply these values and start a new training run"),
        bordered=True,
    )

    mo.vstack([
        mo.md(
            "## Training controls\n"
            "Values are applied only when **Start training** is clicked."),
        training_form,
    ])
    return (training_form,)


@app.cell
def _(
    Actor,
    Config,
    Encoder,
    F,
    Ghost,
    GhostEligibility,
    Path,
    PendingPrediction,
    Predictor,
    RELATIVE_HORIZONS,
    ScoreEligibility,
    StateValueCritic,
    TaskRewardPointControl,
    asdict,
    clip_gradient_list,
    copy,
    deque,
    mo,
    nn,
    np,
    plt,
    polyak_update,
    slice_state,
    torch,
    training_form,
):
    mo.stop(
        training_form.value is None,
        mo.md("Configure the run above, then click **Start training**."))
    submitted_config = dict(training_form.value)
    for integer_field in (
            "seed", "worlds", "transitions", "latent_dim", "ghost_dim",
            "hidden_dim", "conditioning_dim", "snn_ticks",
            "critic_updates_per_step", "curriculum_stage",
            "curriculum_window_episodes", "actor_fisher_warmup_transitions",
            "encoder_fisher_warmup_transitions",
            "encoder_critic_warmup_transitions",
            "ghost_predictive_warmup_transitions",
            "curiosity_warmup_transitions",
            "plot_every", "plot_window"):
        submitted_config[integer_field] = int(
            submitted_config[integer_field])
    cfg = Config(**submitted_config)
    if cfg.ghost_predictive_credit_weight < 0:
        raise ValueError(
            "ghost_predictive_credit_weight must be nonnegative")
    if cfg.ghost_predictive_max_policy_ratio < 0:
        raise ValueError(
            "ghost_predictive_max_policy_ratio must be nonnegative")
    if cfg.ghost_predictive_warmup_transitions < 0:
        raise ValueError(
            "ghost_predictive_warmup_transitions must be nonnegative")
    if cfg.encoder_critic_credit_weight < 0:
        raise ValueError(
            "encoder_critic_credit_weight must be nonnegative")
    if cfg.encoder_critic_max_predictor_ratio < 0:
        raise ValueError(
            "encoder_critic_max_predictor_ratio must be nonnegative")
    if cfg.encoder_critic_warmup_transitions < 0:
        raise ValueError(
            "encoder_critic_warmup_transitions must be nonnegative")
    if cfg.actor_fixed_std != 0.05:
        raise ValueError(
            "fixed-exploration ablation requires actor_fixed_std == 0.05")
    if cfg.curiosity_warmup_transitions < 0:
        raise ValueError(
            "curiosity_warmup_transitions must be nonnegative")
    if cfg.curiosity_error_scale <= 0:
        raise ValueError(
            "curiosity_error_scale must be positive")
    if cfg.curiosity_progress_clip <= 0:
        raise ValueError(
            "curiosity_progress_clip must be positive")
    if not 0 <= cfg.curiosity_baseline_decay < 1:
        raise ValueError(
            "curiosity_baseline_decay must be in [0, 1)")
    for curiosity_name, curiosity_value in (
            ("actor_curiosity_credit_weight",
             cfg.actor_curiosity_credit_weight),
            ("ghost_curiosity_credit_weight",
             cfg.ghost_curiosity_credit_weight),
            ("actor_curiosity_max_external_ratio",
             cfg.actor_curiosity_max_external_ratio),
            ("ghost_curiosity_max_external_ratio",
             cfg.ghost_curiosity_max_external_ratio)):
        if curiosity_value < 0:
            raise ValueError(
                f"{curiosity_name} must be nonnegative")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    env = TaskRewardPointControl(cfg, cfg.seed)
    #Encoder
    encoder = Encoder(cfg).to(device)
    encoder_parameters = list(encoder.parameters())
    target_encoder = copy.deepcopy(encoder).to(device)
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)


    #Ghost    
    ghost = Ghost(cfg).to(device)


    #Actor
    actor = Actor(cfg).to(device)
    actor_named_parameters = dict(actor.named_parameters())
    actor_fisher = {
        name: torch.zeros_like(
            parameter, memory_format=torch.preserve_format)
        for name, parameter in actor_named_parameters.items()}
    actor_fisher_anchor = {
        name: parameter.detach().clone()
        for name, parameter in actor_named_parameters.items()}
    actor_fisher_has_activated = False
    actor_fisher_stage_start_transition = 0
    actor_fisher_has_stage_transitioned = False
    critic_frozen = False
    all_models_frozen = False


    #Predictor
    predictor = Predictor(cfg).to(device)

    #Critic
    critic = StateValueCritic(cfg).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    for parameter in target_critic.parameters():
        parameter.requires_grad_(False)



    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=cfg.encoder_lr)
    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(), lr=cfg.predictor_lr)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=cfg.state_critic_lr)
    actor_trace = ScoreEligibility(actor.parameters(), cfg)
    ghost_intent_trace = GhostEligibility(ghost, cfg)

    b, d = cfg.worlds, cfg.latent_dim
    previous_endpoint = torch.zeros(b, d, device=device)
    previous_advantage = torch.zeros(b, device=device)
    previous_uncertainty = torch.ones(b, device=device)
    previous_remaining_horizon_fraction = torch.zeros(
        b, device=device)
    previous_prediction_valid = torch.zeros(
        b, dtype=torch.bool, device=device)

    maximum_horizon = max(cfg.curriculum_episode_limits)
    pending = [[] for _ in range(b)]
    stage_history = deque(maxlen=cfg.curriculum_window_episodes)
    rolling_success = deque(maxlen=128)
    metric_windows = {
        name: deque(maxlen=cfg.plot_window)
        for name in (
            "reward", "td_error", "critic_value", "predictor_loss",
            "critic_loss", "actor_step", "ghost_step",
            "ghost_policy_gradient_norm",
            "ghost_predictive_gradient_norm",
            "ghost_predictive_applied_ratio",
            "ghost_predictive_loss",
            "encoder_predictor_gradient_norm",
            "encoder_critic_raw_gradient_norm",
            "encoder_critic_applied_gradient_norm",
            "encoder_critic_applied_ratio",
            "encoder_predictor_critic_cosine",
            "actor_std_mean", "actor_std_min", "actor_std_max",
            "sampled_action_deviation",
            "action_saturation_fraction",
            "curiosity_error_before", "curiosity_error_after",
            "curiosity_raw_improvement",
            "curiosity_learning_progress",
            "curiosity_progress_baseline",
            "curiosity_advantage_mean",
            "curiosity_valid_world_fraction",
            "curiosity_terminal_matured_fraction",
            "actor_external_gradient_norm",
            "actor_curiosity_raw_gradient_norm",
            "actor_curiosity_applied_gradient_norm",
            "actor_curiosity_external_ratio",
            "ghost_external_gradient_norm",
            "ghost_curiosity_raw_gradient_norm",
            "ghost_curiosity_applied_gradient_norm",
            "ghost_curiosity_external_ratio")}
    plot_history = {
        name: [] for name in (
            "transition", "reward", "success", "td_error",
            "critic_value", "predictor_loss", "critic_loss",
            "actor_step", "ghost_step", "curriculum_stage",
            "critic_frozen", "all_models_frozen",
            "ghost_policy_gradient_norm",
            "ghost_predictive_gradient_norm",
            "ghost_predictive_applied_ratio",
            "ghost_predictive_loss",
            "encoder_predictor_gradient_norm",
            "encoder_critic_raw_gradient_norm",
            "encoder_critic_applied_gradient_norm",
            "encoder_critic_applied_ratio",
            "encoder_predictor_critic_cosine",
            "actor_std_mean", "actor_std_min", "actor_std_max",
            "sampled_action_deviation",
            "action_saturation_fraction",
            "curiosity_error_before", "curiosity_error_after",
            "curiosity_raw_improvement",
            "curiosity_learning_progress",
            "curiosity_progress_baseline",
            "curiosity_advantage_mean",
            "curiosity_valid_world_fraction",
            "curiosity_terminal_matured_fraction",
            "actor_external_gradient_norm",
            "actor_curiosity_raw_gradient_norm",
            "actor_curiosity_applied_gradient_norm",
            "actor_curiosity_external_ratio",
            "ghost_external_gradient_norm",
            "ghost_curiosity_raw_gradient_norm",
            "ghost_curiosity_applied_gradient_norm",
            "ghost_curiosity_external_ratio")}
    next_plot_transition = cfg.plot_every

    observation = env.observation()
    transitions = 0
    curiosity_progress_baseline = torch.zeros((), device=device)


    def rolling_metric(name):
        values = metric_windows[name]
        return float(np.mean(values)) if values else float("nan")


    def render_training_dashboard():
        x = plot_history["transition"]
        figure, axes = plt.subplots(6, 2, figsize=(13, 19))
        panels = axes.ravel()

        panels[0].plot(x, plot_history["reward"], color="tab:blue")
        panels[0].set_title("Rolling task reward")

        panels[1].plot(x, plot_history["success"], color="tab:green")
        panels[1].set_ylim(-0.02, 1.02)
        panels[1].set_title("Episode success rate")

        panels[2].plot(x, plot_history["td_error"], color="tab:orange")
        panels[2].set_title("Mean absolute TD error")

        panels[3].plot(
            x, plot_history["predictor_loss"],
            label="Predictor", color="tab:purple")
        panels[3].plot(
            x, plot_history["critic_loss"],
            label="Critic", color="tab:red")
        critic_status = (
            "FROZEN" if plot_history["critic_frozen"][-1] else "training")
        panels[3].set_title(f"Rolling losses — critic {critic_status}")
        panels[3].legend()

        actor_steps = np.maximum(
            np.asarray(plot_history["actor_step"]), 1e-12)
        ghost_steps = np.maximum(
            np.asarray(plot_history["ghost_step"]), 1e-12)
        panels[4].semilogy(
            x, actor_steps, label="Actor", color="tab:brown")
        panels[4].semilogy(
            x, ghost_steps, label="Ghost", color="tab:pink")
        panels[4].set_title("Parameter update norms")
        panels[4].legend()

        panels[5].step(
            x, plot_history["curriculum_stage"],
            where="post", color="tab:cyan")
        panels[5].set_yticks(range(env.curriculum_stage_count))
        panels[5].set_title("Curriculum stage")

        panels[6].plot(
            x, plot_history["critic_value"], color="tab:olive")
        panels[6].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        panels[6].set_title(
            "Mean V(state, predicted delta, horizon)")

        panels[7].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history["ghost_policy_gradient_norm"]), 1e-12),
            label="Policy direction", color="tab:blue")
        panels[7].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history["ghost_predictive_gradient_norm"]), 1e-12),
            label="Applied predictive", color="tab:orange")
        ratio_axis = panels[7].twinx()
        ratio_axis.plot(
            x, plot_history["ghost_predictive_applied_ratio"],
            label="Predictive / policy", color="tab:green", alpha=0.75)
        ratio_axis.set_ylabel("Applied ratio")
        panel_lines, panel_labels = panels[7].get_legend_handles_labels()
        ratio_lines, ratio_labels = ratio_axis.get_legend_handles_labels()
        panels[7].legend(
            panel_lines+ratio_lines, panel_labels+ratio_labels,
            loc="best")
        panels[7].set_title("Ghost learning directions")
        panels[7].set_xlabel("Transitions")
        panels[7].grid(alpha=0.25)

        panels[8].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history["encoder_predictor_gradient_norm"]),
                1e-12),
            label="Predictor", color="tab:purple")
        panels[8].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history[
                        "encoder_critic_applied_gradient_norm"]),
                1e-12),
            label="Applied critic", color="tab:red")
        encoder_ratio_axis = panels[8].twinx()
        encoder_ratio_axis.plot(
            x, plot_history["encoder_critic_applied_ratio"],
            label="Critic / predictor", color="tab:orange")
        encoder_ratio_axis.plot(
            x, plot_history["encoder_predictor_critic_cosine"],
            label="Predictor/critic cosine",
            color="tab:green", alpha=0.75)
        encoder_ratio_axis.axhline(
            0.0, color="black", linewidth=0.8, alpha=0.35)
        encoder_ratio_axis.set_ylabel("Ratio / cosine")
        encoder_lines, encoder_labels = (
            panels[8].get_legend_handles_labels())
        encoder_ratio_lines, encoder_ratio_labels = (
            encoder_ratio_axis.get_legend_handles_labels())
        panels[8].legend(
            encoder_lines+encoder_ratio_lines,
            encoder_labels+encoder_ratio_labels,
            loc="best")
        panels[8].set_title("Encoder learning directions")
        panels[8].set_xlabel("Transitions")
        panels[8].grid(alpha=0.25)

        panels[9].plot(
            x, plot_history["actor_std_mean"],
            label="Mean pre-squash std", color="tab:blue")
        panels[9].fill_between(
            x, plot_history["actor_std_min"],
            plot_history["actor_std_max"],
            label="Min–max std", color="tab:blue", alpha=0.18)
        panels[9].axhline(
            cfg.actor_fixed_std,
            color="tab:red", linestyle="--", linewidth=0.9,
            label="Configured fixed std")
        exploration_stability_axis = panels[9].twinx()
        exploration_stability_axis.plot(
            x, plot_history["sampled_action_deviation"],
            label="Mean |action − mean|", color="tab:green")
        exploration_stability_axis.plot(
            x, plot_history["action_saturation_fraction"],
            label="Action saturation", color="tab:orange")
        exploration_stability_axis.set_ylim(bottom=-0.02)
        exploration_stability_axis.set_ylabel(
            "Deviation / saturation fraction")
        exploration_lines, exploration_labels = (
            panels[9].get_legend_handles_labels())
        stability_lines, stability_labels = (
            exploration_stability_axis.get_legend_handles_labels())
        panels[9].legend(
            exploration_lines+stability_lines,
            exploration_labels+stability_labels,
            loc="best")
        panels[9].set_title("Fixed exploration ablation")
        panels[9].set_xlabel("Transitions")
        panels[9].grid(alpha=0.25)
        panels[10].plot(
            x, plot_history["curiosity_error_before"],
            label="Error before", color="tab:red")
        panels[10].plot(
            x, plot_history["curiosity_error_after"],
            label="Error after", color="tab:orange")
        progress_axis = panels[10].twinx()
        progress_axis.plot(
            x, plot_history["curiosity_learning_progress"],
            label="Learning progress", color="tab:green")
        progress_axis.plot(
            x, plot_history["curiosity_progress_baseline"],
            label="Progress baseline", color="tab:blue",
            linestyle="--")
        progress_axis.plot(
            x, plot_history["curiosity_valid_world_fraction"],
            label="Valid-world fraction", color="tab:purple",
            alpha=0.65)
        progress_axis.set_ylabel("Progress / fraction")
        curiosity_error_lines, curiosity_error_labels = (
            panels[10].get_legend_handles_labels())
        progress_lines, progress_labels = (
            progress_axis.get_legend_handles_labels())
        panels[10].legend(
            curiosity_error_lines+progress_lines,
            curiosity_error_labels+progress_labels,
            loc="best")
        panels[10].set_title("World-model learning progress")
        panels[10].set_xlabel("Transitions")
        panels[10].grid(alpha=0.25)

        panels[11].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history["actor_external_gradient_norm"]),
                1e-12),
            label="Actor external", color="tab:blue")
        panels[11].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history[
                        "actor_curiosity_applied_gradient_norm"]),
                1e-12),
            label="Actor applied curiosity", color="tab:cyan")
        panels[11].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history["ghost_external_gradient_norm"]),
                1e-12),
            label="Ghost external", color="tab:orange")
        panels[11].semilogy(
            x, np.maximum(
                np.asarray(
                    plot_history[
                        "ghost_curiosity_applied_gradient_norm"]),
                1e-12),
            label="Ghost applied curiosity", color="tab:red")
        curiosity_ratio_axis = panels[11].twinx()
        curiosity_ratio_axis.plot(
            x, plot_history["actor_curiosity_external_ratio"],
            label="Actor curiosity/external",
            color="tab:green", alpha=0.75)
        curiosity_ratio_axis.plot(
            x, plot_history["ghost_curiosity_external_ratio"],
            label="Ghost curiosity/external",
            color="tab:purple", alpha=0.75)
        curiosity_ratio_axis.set_ylabel("Applied ratio")
        direction_lines, direction_labels = (
            panels[11].get_legend_handles_labels())
        curiosity_ratio_lines, curiosity_ratio_labels = (
            curiosity_ratio_axis.get_legend_handles_labels())
        panels[11].legend(
            direction_lines+curiosity_ratio_lines,
            direction_labels+curiosity_ratio_labels,
            loc="best")
        panels[11].set_title("Curiosity eligibility directions")
        panels[11].set_xlabel("Transitions")
        panels[11].grid(alpha=0.25)

        for axis in panels[:7]:
            axis.set_xlabel("Transitions")
            axis.grid(alpha=0.25)
        freeze_status = (
            " — ALL MODELS FROZEN"
            if plot_history["all_models_frozen"][-1] else "")
        figure.suptitle(
            f"Live training — {x[-1]:,} / {cfg.transitions:,} transitions"
            f"{freeze_status}")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        mo.output.replace(figure)
        plt.close(figure)


    while transitions < cfg.transitions:
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=device)
        latent = encoder(obs)
        current_predicted_delta = torch.where(
            previous_prediction_valid[:, None],
            previous_endpoint-latent,
            torch.zeros_like(latent))
        current_remaining_horizon_fraction = torch.where(
            previous_prediction_valid,
            previous_remaining_horizon_fraction,
            torch.zeros_like(previous_remaining_horizon_fraction))
        assert current_predicted_delta.shape == latent.shape
        assert (
            current_remaining_horizon_fraction.shape == latent.shape[:1])
        assert torch.isfinite(current_predicted_delta).all()
        assert torch.isfinite(current_remaining_horizon_fraction).all()
        ghost_pre = ghost.core.snapshot()
        ghost_intent, horizon_logits = ghost(
            latent, previous_endpoint,
            previous_advantage, previous_uncertainty)

        current_value = critic(
            latent,
            current_predicted_delta.detach(),
            current_remaining_horizon_fraction.detach())
        category_policy = horizon_logits.softmax(-1)
        relative_table = torch.tensor(RELATIVE_HORIZONS, device=device)
        remaining_budget = torch.as_tensor(
            np.maximum(1, env.current_episode_limit-env.age),
            dtype=torch.long, device=device)
        category_horizons = torch.maximum(
            torch.ones((b, len(RELATIVE_HORIZONS)),
                       dtype=torch.long, device=device),
            torch.round(
                remaining_budget[:, None]*relative_table[None]).long())
        actual_policy = torch.zeros(
            b, maximum_horizon+1, device=device)
        actual_policy.scatter_add_(
            1, category_horizons, category_policy)
        support = torch.zeros_like(actual_policy, dtype=torch.bool)
        support.scatter_(1, category_horizons, True)
        uniform_actual = (
            support.float()
            /support.sum(-1, keepdim=True).clamp_min(1))
        mixed_policy = (
            (1-cfg.horizon_uniform_eta)*actual_policy
            +cfg.horizon_uniform_eta*uniform_actual)
        option_count = support.sum(-1, keepdim=True).float()
        residual = (
            1-cfg.actual_horizon_min_probability*option_count).clamp_min(0)
        sampling_policy = (
            cfg.actual_horizon_min_probability*support.float()
            +residual*mixed_policy)
        with torch.no_grad():
            actual_horizon = torch.multinomial(
                sampling_policy.detach(), 1).squeeze(-1)
        relative_horizon = (
            actual_horizon.float()
            /remaining_budget.float().clamp_min(1))
        next_remaining_horizon = torch.clamp(
            actual_horizon-1, min=0)
        next_remaining_budget = torch.clamp(
            remaining_budget-1, min=1)
        next_remaining_horizon_fraction = (
            next_remaining_horizon.float()
            /next_remaining_budget.float())
        budget_fraction = (
            remaining_budget.float()/float(env.current_episode_limit))

        actor_output = actor(latent, ghost_intent)
        assert torch.allclose(
            actor_output["std"],
            torch.full_like(actor_output["std"], 0.05))
        actor_trace.accumulate(actor_output["logp"])
        # The same actor score carries TD credit for Ghost influence over
        # both action mean and learned exploration standard deviation.
        ghost_intent_trace.accumulate(
            ghost_intent, actor_output["logp"])

        predictor_pre = predictor.core.snapshot()
        predicted_mean, predicted_logvar, predicted_return = predictor(
            latent.detach(), ghost_intent.detach(),
            relative_horizon, budget_fraction,
            actor_output["std"].detach())
        uncertainty = predicted_logvar.exp().mean(-1).sqrt()
        with torch.no_grad():
            endpoint_value = target_critic(
                predicted_mean.detach(),
                torch.zeros_like(predicted_mean),
                torch.zeros_like(relative_horizon))
            forecast_value = (
                predicted_return.detach()
                +torch.pow(
                    torch.full_like(predicted_return, cfg.gamma),
                    actual_horizon.float())*endpoint_value)
            forecast_advantage = forecast_value-current_value.detach()

        # Ordinary predictions are created every step.  Their source ghost_intent
        # is context at prediction time; later ghost_intent evolution is expected
        # closed-loop behavior and never interrupts or cancels the record.
        for world in range(b):
            horizon = int(actual_horizon[world])
            pending[world].append(PendingPrediction(
                episode=int(env.episode[world]),
                remaining=horizon,
                source_observation=(
                    obs[world:world+1].detach().clone()),
                source_ghost_intent=(
                    ghost_intent[world:world+1].detach().clone()),
                relative_horizon=(
                    relative_horizon[world:world+1].detach().clone()),
                budget_fraction=(
                    budget_fraction[world:world+1].detach().clone()),
                predictor_state=slice_state(predictor_pre, world),
                discounted_return=0.0,
                discount=1.0,
                source_latent=(
                    latent[world:world+1].detach().clone()),
                ghost_pre_state=slice_state(ghost_pre, world),
                source_previous_endpoint=(
                    previous_endpoint[
                        world:world+1].detach().clone()),
                source_previous_advantage=(
                    previous_advantage[
                        world:world+1].detach().clone()),
                source_previous_uncertainty=(
                    previous_uncertainty[
                        world:world+1].detach().clone()),
                source_action_std=(
                    actor_output["std"][
                        world:world+1].detach().clone())))

        action_detached = actor_output["action"].detach()
        transition = env.step(action_detached.cpu().numpy())
        reward = torch.as_tensor(
            transition.dense_reward, dtype=torch.float32, device=device)
        done = torch.as_tensor(
            transition.done, dtype=torch.bool, device=device)
        success = torch.as_tensor(
            transition.success, dtype=torch.bool, device=device)
        for is_done, is_success in zip(
                done.detach().cpu().tolist(),
                success.detach().cpu().tolist()):
            if is_done:
                rolling_success.append(float(is_success))
        rolling_success_128 = (
            sum(rolling_success)/max(len(rolling_success), 1))
        if (cfg.freeze_all_above_success
                and len(rolling_success) == rolling_success.maxlen
                and rolling_success_128
                >= cfg.freeze_all_success_threshold):
            all_models_frozen = True
            critic_frozen = True
        endpoint_obs = torch.as_tensor(
            transition.target_observation,
            dtype=torch.float32, device=device)
        with torch.no_grad():
            endpoint_target_latent = target_encoder(endpoint_obs)
            next_predicted_delta = (
                predicted_mean.detach()-endpoint_target_latent)
            assert next_predicted_delta.shape == endpoint_target_latent.shape
            assert (
                next_remaining_horizon_fraction.shape
                == endpoint_target_latent.shape[:1])
            assert torch.isfinite(next_predicted_delta).all()
            assert torch.isfinite(
                next_remaining_horizon_fraction).all()
            next_target = target_critic(
                endpoint_target_latent,
                next_predicted_delta.detach(),
                next_remaining_horizon_fraction.detach())
            td_target = reward+cfg.gamma*(~done).float()*next_target
            actor_td = td_target-current_value.detach()

        mean_abs_td_error = float(actor_td.detach().abs().mean())
        if (cfg.freeze_critic_below_td
                and mean_abs_td_error
                <= cfg.critic_freeze_td_threshold):
            critic_frozen = True

        transition_after_step = transitions+b
        if (cfg.use_actor_fisher
                and not actor_fisher_has_activated
                and transition_after_step
                >= cfg.actor_fisher_warmup_transitions
                and rolling_success_128
                >= cfg.actor_fisher_activation_success):
            actor_fisher_has_activated = True
        if actor_fisher_has_stage_transitioned:
            actor_fisher_stage_age = max(
                0, transition_after_step
                -actor_fisher_stage_start_transition)
            actor_fisher_stage_multiplier = min(
                1.0,
                cfg.actor_fisher_stage_initial_multiplier
                +(1.0-cfg.actor_fisher_stage_initial_multiplier)
                *actor_fisher_stage_age
                /max(1, cfg.actor_fisher_stage_recovery_transitions))
        else:
            actor_fisher_stage_multiplier = 1.0
        effective_fisher_lambda = (
            cfg.actor_fisher_lambda*actor_fisher_stage_multiplier)

        critic_per_world_loss = (current_value-td_target).square()
        critic_encoder_jacobians = torch.autograd.grad(
            critic_per_world_loss, encoder_parameters,
            grad_outputs=torch.eye(b, device=device),
            is_grads_batched=True, retain_graph=True, allow_unused=True)
        critic_encoder_jacobians = [
            (torch.zeros(
                (b,)+tuple(parameter.shape),
                device=parameter.device, dtype=parameter.dtype)
             if jacobian is None else jacobian.detach())
            for parameter, jacobian in zip(
                encoder_parameters, critic_encoder_jacobians)]
        raw_critic_encoder_gradients = [
            jacobian.mean(0).detach()
            for jacobian in critic_encoder_jacobians]
        critic_update_losses = []
        critic_latent = latent.detach()
        critic_delta = current_predicted_delta.detach()
        critic_horizon = current_remaining_horizon_fraction.detach()
        fixed_td_target = td_target.detach()
        critic_update_count = (
            0
            if critic_frozen or all_models_frozen
            else cfg.critic_updates_per_step)
        for _ in range(critic_update_count):
            critic_prediction = critic(
                critic_latent, critic_delta, critic_horizon)
            critic_loss = F.mse_loss(
                critic_prediction, fixed_td_target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                critic.parameters(), cfg.critic_gradient_clip)
            critic_optimizer.step()
            polyak_update(target_critic, critic, cfg.target_tau)
            critic_update_losses.append(critic_loss.detach())
        if critic_update_losses:
            critic_loss = torch.stack(critic_update_losses).mean()
        else:
            with torch.no_grad():
                critic_loss = F.mse_loss(
                    critic(critic_latent, critic_delta, critic_horizon),
                    fixed_td_target)

        matured = []
        for world in range(b):
            updated = []
            for record in pending[world]:
                if record.episode != int(transition.episode[world]):
                    continue
                new_record = record._replace(
                    remaining=record.remaining-1,
                    discounted_return=(
                        record.discounted_return
                        +record.discount*float(reward[world])),
                    discount=record.discount*cfg.gamma)
                if new_record.remaining <= 0 or bool(done[world]):
                    terminated_early = (
                        bool(done[world])
                        and new_record.remaining > 0)
                    matured.append(
                        (world, new_record, terminated_early))
                else:
                    updated.append(new_record)
            pending[world] = updated

        predictor_live = predictor.core.snapshot()
        predictor_live_last_output = predictor.core.last_output
        predictor_live_eligibility_records = list(
            predictor.core.last_eligibility_records)
        ghost_live_state = ghost.core.snapshot()
        ghost_live_last_output = ghost.core.last_output
        ghost_live_eligibility_records = list(
            ghost.core.last_eligibility_records)
        predictor_losses = []
        predictive_gradient_sums = [
            torch.zeros_like(parameter)
            for parameter in ghost_intent_trace.parameters]
        ghost_predictive_losses = []
        ghost_predictive_record_count = 0
        curiosity_measurements = []
        for world, record, terminated_early in matured:
            predictor.core.restore(record.predictor_state)
            source_latent = encoder(record.source_observation)
            replay_mean, replay_logvar, replay_return = predictor(
                source_latent, record.source_ghost_intent,
                record.relative_horizon, record.budget_fraction,
                record.source_action_std.detach())
            target = endpoint_target_latent[world:world+1]
            assert not target.requires_grad
            squared = (target-replay_mean).square()
            with torch.no_grad():
                fixed_precision = torch.exp(
                    -replay_logvar.detach())
                curiosity_error_before = (
                    (target.detach()-replay_mean.detach()).square()
                    *fixed_precision).mean()
                curiosity_measurements.append((
                    world, record,
                    target.detach().clone(),
                    fixed_precision.detach().clone(),
                    curiosity_error_before.detach().clone(),
                    terminated_early))
            nll = 0.5*(
                squared*torch.exp(-replay_logvar)+replay_logvar).mean()
            return_target = torch.tensor(
                [record.discounted_return], device=device)
            return_loss = F.mse_loss(replay_return, return_target)
            prediction_loss = nll+cfg.return_loss_weight*return_loss
            predictor_losses.append(prediction_loss)
            if (not all_models_frozen
                    and transitions
                    >= cfg.ghost_predictive_warmup_transitions):
                ghost.core.restore(record.ghost_pre_state)
                replay_ghost_intent, _ = ghost(
                    record.source_latent.detach(),
                    record.source_previous_endpoint.detach(),
                    record.source_previous_advantage.detach(),
                    record.source_previous_uncertainty.detach())
                predictor.core.restore(record.predictor_state)
                auxiliary_mean, auxiliary_logvar, _ = predictor(
                    record.source_latent.detach(),
                    replay_ghost_intent,
                    record.relative_horizon.detach(),
                    record.budget_fraction.detach(),
                    record.source_action_std.detach())
                squared_error = (
                    target.detach()-auxiliary_mean).square()
                detached_precision = torch.exp(
                    -auxiliary_logvar.detach())
                ghost_predictive_loss = (
                    0.5*squared_error*detached_precision).mean()
                auxiliary_gradients = torch.autograd.grad(
                    ghost_predictive_loss,
                    ghost_intent_trace.parameters,
                    allow_unused=True)
                with torch.no_grad():
                    for total, parameter, gradient in zip(
                            predictive_gradient_sums,
                            ghost_intent_trace.parameters,
                            auxiliary_gradients):
                        total.add_(
                            torch.zeros_like(parameter)
                            if gradient is None else gradient.detach())
                ghost_predictive_losses.append(
                    ghost_predictive_loss.detach())
                ghost_predictive_record_count += 1
        predictor.core.restore(predictor_live)
        predictor.core.last_output = predictor_live_last_output
        predictor.core.last_eligibility_records = (
            predictor_live_eligibility_records)
        ghost.core.restore(ghost_live_state)
        ghost.core.last_output = ghost_live_last_output
        ghost.core.last_eligibility_records = (
            ghost_live_eligibility_records)
        if ghost_predictive_record_count:
            predictive_gradients = [
                gradient_sum/ghost_predictive_record_count
                for gradient_sum in predictive_gradient_sums]
            mean_ghost_predictive_loss = float(torch.stack(
                ghost_predictive_losses).mean())
        else:
            predictive_gradients = None
            mean_ghost_predictive_loss = float("nan")

        encoder_optimizer.zero_grad(set_to_none=True)
        predictor_optimizer.zero_grad(set_to_none=True)
        prediction_loss_value = float("nan")
        if predictor_losses:
            prediction_loss = torch.stack(predictor_losses).mean()
            prediction_loss_value = float(prediction_loss.detach())
            prediction_loss.backward()
        predictor_encoder_gradients = [
            (torch.zeros_like(parameter)
             if parameter.grad is None
             else parameter.grad.detach().clone())
            for parameter in encoder_parameters]
        predictor_gradient_norm = torch.stack([
            gradient.square().sum()
            for gradient in predictor_encoder_gradients]).sum().sqrt()
        raw_critic_gradient_norm = torch.stack([
            gradient.square().sum()
            for gradient in raw_critic_encoder_gradients]).sum().sqrt()
        weighted_critic_gradients = [
            cfg.encoder_critic_credit_weight*gradient
            for gradient in raw_critic_encoder_gradients]
        weighted_critic_gradient_norm = torch.stack([
            gradient.square().sum()
            for gradient in weighted_critic_gradients]).sum().sqrt()
        critic_credit_enabled = (
            transitions >= cfg.encoder_critic_warmup_transitions
            and not critic_frozen
            and not all_models_frozen
            and float(predictor_gradient_norm) > 0.0
            and cfg.encoder_critic_credit_weight > 0.0
            and cfg.encoder_critic_max_predictor_ratio > 0.0)
        if critic_credit_enabled:
            critic_cap = (
                cfg.encoder_critic_max_predictor_ratio
                *float(predictor_gradient_norm))
            critic_scale = min(
                1.0,
                critic_cap/(float(weighted_critic_gradient_norm)+1e-12))
        else:
            critic_scale = 0.0
        applied_critic_gradients = [
            critic_scale*gradient
            for gradient in weighted_critic_gradients]
        applied_critic_gradient_norm = torch.stack([
            gradient.square().sum()
            for gradient in applied_critic_gradients]).sum().sqrt()
        encoder_critic_applied_ratio = (
            float(applied_critic_gradient_norm)
            /(float(predictor_gradient_norm)+1e-12))
        assert (
            encoder_critic_applied_ratio
            <= cfg.encoder_critic_max_predictor_ratio+1e-6)
        if float(predictor_gradient_norm) == 0.0:
            assert float(applied_critic_gradient_norm) == 0.0
        predictor_critic_dot = torch.stack([
            (predictor_gradient*critic_gradient).sum()
            for predictor_gradient, critic_gradient in zip(
                predictor_encoder_gradients,
                applied_critic_gradients)]).sum()
        encoder_predictor_critic_cosine = (
            float(predictor_critic_dot)
            /(float(predictor_gradient_norm)
              *float(applied_critic_gradient_norm)+1e-12))
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if not all_models_frozen:
            predictor_optimizer.step()
        for parameter, predictor_gradient, critic_gradient in zip(
                encoder_parameters, predictor_encoder_gradients,
                applied_critic_gradients):
            # Encoder Adam minimizes both factual prediction error and the
            # bounded critic TD regularizer, so the two gradients are added.
            parameter.grad = (
                predictor_gradient+critic_gradient).detach().clone()
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        if not all_models_frozen and predictor_losses:
            encoder_optimizer.step()

        curiosity_error_before_value = float("nan")
        curiosity_error_after_value = float("nan")
        curiosity_raw_improvement_value = float("nan")
        curiosity_learning_progress_value = float("nan")
        curiosity_advantage_mean = 0.0
        progress_sum_per_world = torch.zeros(b, device=device)
        progress_count_per_world = torch.zeros(b, device=device)
        curiosity_valid_mask = torch.zeros(
            b, dtype=torch.bool, device=device)
        curiosity_advantage = torch.zeros(b, device=device)
        curiosity_terminal_matured_fraction = (
            sum(float(measurement[-1])
                for measurement in curiosity_measurements)
            /max(len(curiosity_measurements), 1))
        curiosity_active = (
            cfg.use_world_model_curiosity
            and transitions >= cfg.curiosity_warmup_transitions
            and not all_models_frozen
            and bool(curiosity_measurements))
        if curiosity_active:
            after_live_state = predictor.core.snapshot()
            after_live_last_output = predictor.core.last_output
            after_live_eligibility_records = list(
                predictor.core.last_eligibility_records)
            error_before_values = []
            error_after_values = []
            raw_improvement_values = []
            learning_progress_values = []
            with torch.no_grad():
                for (
                        world, record, fixed_target,
                        fixed_precision, error_before,
                        _terminated_early) in curiosity_measurements:
                    predictor.core.restore(record.predictor_state)
                    updated_source_latent = encoder(
                        record.source_observation)
                    replay_mean_after, _, _ = predictor(
                        updated_source_latent,
                        record.source_ghost_intent,
                        record.relative_horizon,
                        record.budget_fraction,
                        record.source_action_std.detach())
                    error_after = (
                        (fixed_target-replay_mean_after).square()
                        *fixed_precision).mean()
                    raw_improvement = torch.clamp(
                        error_before-error_after, min=0)
                    relative_improvement = (
                        raw_improvement/(error_before+1e-8))
                    novelty_gate = (
                        error_before
                        /(error_before+cfg.curiosity_error_scale))
                    learning_progress = torch.clamp(
                        relative_improvement*novelty_gate,
                        min=0, max=cfg.curiosity_progress_clip
                    ).detach()
                    assert torch.isfinite(error_before)
                    assert torch.isfinite(error_after)
                    assert torch.isfinite(learning_progress)
                    error_before_values.append(error_before)
                    error_after_values.append(error_after)
                    raw_improvement_values.append(raw_improvement)
                    learning_progress_values.append(
                        learning_progress)
                    progress_sum_per_world[world].add_(
                        learning_progress)
                    progress_count_per_world[world].add_(1)
                predictor.core.restore(after_live_state)
                predictor.core.last_output = after_live_last_output
                predictor.core.last_eligibility_records = (
                    after_live_eligibility_records)
                curiosity_valid_mask = (
                    progress_count_per_world > 0)
                mean_progress_per_world = (
                    progress_sum_per_world
                    /progress_count_per_world.clamp_min(1))
                baseline_before = (
                    curiosity_progress_baseline.detach().clone())
                curiosity_advantage = torch.where(
                    curiosity_valid_mask,
                    mean_progress_per_world-baseline_before,
                    torch.zeros_like(mean_progress_per_world)
                ).detach()
                assert torch.isfinite(curiosity_advantage).all()
                if bool(curiosity_valid_mask.any()):
                    batch_progress_mean = mean_progress_per_world[
                        curiosity_valid_mask].mean()
                    curiosity_progress_baseline.mul_(
                        cfg.curiosity_baseline_decay).add_(
                            batch_progress_mean,
                            alpha=1-cfg.curiosity_baseline_decay)
                curiosity_error_before_value = float(
                    torch.stack(error_before_values).mean())
                curiosity_error_after_value = float(
                    torch.stack(error_after_values).mean())
                curiosity_raw_improvement_value = float(
                    torch.stack(raw_improvement_values).mean())
                curiosity_learning_progress_value = float(
                    torch.stack(learning_progress_values).mean())
                curiosity_advantage_mean = float(
                    curiosity_advantage[
                        curiosity_valid_mask].mean())

        # External actor TD remains the sole Fisher source.
        external_actor_directions = []
        for trace in actor_trace.traces:
            view = (len(actor_td),)+(1,)*(trace.ndim-1)
            external_actor_directions.append(
                (trace*actor_td.view(view)).mean(0))
        actor_external_gradient_norm = torch.stack([
            direction.square().sum()
            for direction in external_actor_directions]).sum().sqrt()
        raw_actor_curiosity_directions = []
        for trace in actor_trace.traces:
            view = (
                (len(curiosity_advantage),)
                +(1,)*(trace.ndim-1))
            raw_actor_curiosity_directions.append(
                (trace*curiosity_advantage.view(view)).mean(0))
        actor_curiosity_raw_gradient_norm = torch.stack([
            direction.square().sum()
            for direction in raw_actor_curiosity_directions
        ]).sum().sqrt()
        weighted_actor_curiosity_directions = [
            cfg.actor_curiosity_credit_weight*direction
            for direction in raw_actor_curiosity_directions]
        weighted_actor_curiosity_norm = torch.stack([
            direction.square().sum()
            for direction in weighted_actor_curiosity_directions
        ]).sum().sqrt()
        actor_curiosity_enabled = (
            curiosity_active
            and bool(curiosity_valid_mask.any())
            and float(actor_external_gradient_norm) > 0.0
            and cfg.actor_curiosity_credit_weight > 0.0
            and cfg.actor_curiosity_max_external_ratio > 0.0)
        if actor_curiosity_enabled:
            actor_curiosity_scale = min(
                1.0,
                (cfg.actor_curiosity_max_external_ratio
                 *float(actor_external_gradient_norm))
                /(float(weighted_actor_curiosity_norm)+1e-12))
        else:
            actor_curiosity_scale = 0.0
        applied_actor_curiosity_directions = [
            actor_curiosity_scale*direction
            for direction in weighted_actor_curiosity_directions]
        actor_curiosity_applied_gradient_norm = (
            actor_curiosity_scale
            *float(weighted_actor_curiosity_norm))
        actor_curiosity_external_ratio = (
            actor_curiosity_applied_gradient_norm
            /float(actor_external_gradient_norm)
            if float(actor_external_gradient_norm) > 0.0 else 0.0)
        actor_behavioral_directions = [
            external_direction+curiosity_direction
            for external_direction, curiosity_direction in zip(
                external_actor_directions,
                applied_actor_curiosity_directions)]
        actor_control_gradients = clip_gradient_list(
            actor_behavioral_directions, 1.0)
        fisher_external_directions = clip_gradient_list(
            external_actor_directions, 1.0)

        with torch.no_grad():
            if cfg.use_actor_fisher:
                for ((name, _), external_direction) in zip(
                        actor_named_parameters.items(),
                        fisher_external_directions):
                    if not bool(torch.count_nonzero(
                            external_direction.detach())):
                        continue
                    fisher = actor_fisher[name]
                    fisher.mul_(cfg.actor_fisher_decay)
                    fisher.addcmul_(
                        external_direction.detach(),
                        external_direction.detach(),
                        value=1.0-cfg.actor_fisher_decay)

        actor_fisher_gradients = []
        with torch.no_grad():
            for name, parameter in actor_named_parameters.items():
                fisher = actor_fisher[name]
                normalized_fisher = (
                    fisher/(fisher.mean()+1e-8)).clamp(
                        min=0.0,
                        max=cfg.actor_fisher_normalized_max)
                if (cfg.use_actor_fisher
                        and actor_fisher_has_activated):
                    fisher_gradient = (
                        effective_fisher_lambda
                        *normalized_fisher
                        *(parameter.detach()
                          -actor_fisher_anchor[name]))
                else:
                    fisher_gradient = torch.zeros_like(parameter)
                actor_fisher_gradients.append(fisher_gradient)

        actor_trace.optimizer.zero_grad(set_to_none=True)
        for parameter, control_gradient, fisher_gradient in zip(
                actor_trace.parameters, actor_control_gradients,
                actor_fisher_gradients):
            parameter.grad = (
                control_gradient-fisher_gradient).detach().clone()
        if all_models_frozen:
            actor_step_norm = 0.0
        else:
            actor_before = [
                parameter.detach().clone()
                for parameter in actor_trace.parameters]
            actor_trace.optimizer.step()
            actor_step_norm = float(torch.stack([
                (parameter-old).square().sum()
                for parameter, old in zip(
                    actor_trace.parameters, actor_before)
            ]).sum().sqrt())
        with torch.no_grad():
            if cfg.use_actor_fisher and not all_models_frozen:
                for name, parameter in actor_named_parameters.items():
                    fisher = actor_fisher[name]
                    normalized_fisher = (
                        fisher/(fisher.mean()+1e-8)).clamp(
                            min=0.0,
                            max=cfg.actor_fisher_normalized_max)
                    local_anchor_rate = (
                        cfg.actor_fisher_anchor_lr
                        /(1.0+cfg.actor_fisher_importance_scale
                          *normalized_fisher))
                    anchor = actor_fisher_anchor[name]
                    anchor.add_(
                        local_anchor_rate
                        *(parameter.detach()-anchor))

        if not all_models_frozen:
            polyak_update(
                target_encoder, encoder,
                cfg.encoder_target_tau)

        if all_models_frozen:
            ghost_diagnostics = {
                "step_norm": 0.0,
                "policy_gradient_norm": 0.0,
                "external_gradient_norm": 0.0,
                "curiosity_raw_gradient_norm": 0.0,
                "curiosity_applied_gradient_norm": 0.0,
                "curiosity_external_ratio": 0.0,
                "weighted_predictive_gradient_norm": 0.0,
                "applied_predictive_gradient_norm": 0.0,
                "predictive_to_policy_ratio": 0.0,
                "predictive_record_count": 0,
                "predictive_loss": float("nan"),
            }
        else:
            ghost_diagnostics = ghost_intent_trace.apply(
                actor_td,
                curiosity_advantage=curiosity_advantage,
                predictive_gradients=predictive_gradients,
                predictive_record_count=ghost_predictive_record_count,
                predictive_loss=mean_ghost_predictive_loss)
        ghost_step_norm = ghost_diagnostics["step_norm"]


        previous_endpoint = predicted_mean.detach()
        previous_remaining_horizon_fraction = (
            next_remaining_horizon_fraction.detach())
        previous_prediction_valid.fill_(True)
        previous_advantage = forecast_advantage.detach()
        previous_uncertainty = uncertainty.detach()

        done_values = done.detach().cpu().tolist()
        success_values = success.detach().cpu().tolist()
        for is_done, is_success in zip(done_values, success_values):
            if is_done:
                stage_history.append(float(is_success))

        if bool(done.any()):
            # This is the only runtime reset of the ghost neuronal state.
            ghost.reset(done)
            predictor.reset(done)
            actor_trace.reset(done)
            ghost_intent_trace.reset(done)
            previous_endpoint[done] = 0
            previous_remaining_horizon_fraction[done] = 0
            previous_prediction_valid[done] = False
            previous_advantage[done] = 0
            previous_uncertainty[done] = 1
            for world in torch.nonzero(
                    done, as_tuple=False).flatten().tolist():
                pending[world].clear()

        advanced_curriculum = False
        if (len(stage_history) == stage_history.maxlen
                and float(np.mean(stage_history))
                >= cfg.curriculum_threshold
                and env.curriculum_stage
                < env.curriculum_stage_count-1):
            env.set_curriculum_stage(env.curriculum_stage+1)
            actor_fisher_stage_start_transition = transitions+b
            actor_fisher_has_stage_transitioned = True
            reset_all = torch.ones(
                b, dtype=torch.bool, device=device)
            # Deliberately do not reset ghost.core here.  Curriculum
            # changes are not ghost_intent-reset events.
            predictor.reset(reset_all)
            actor_trace.reset(reset_all)
            ghost_intent_trace.reset(reset_all)
            previous_endpoint.zero_()
            previous_remaining_horizon_fraction.zero_()
            previous_prediction_valid.zero_()
            previous_advantage.zero_()
            previous_uncertainty.fill_(1)
            for queue in pending:
                queue.clear()
            stage_history.clear()
            curiosity_progress_baseline.zero_()
            advanced_curriculum = True

        with torch.no_grad():
            behavior_std = actor_output["std"].detach()
            sampled_action_deviation = (
                actor_output["action"].detach()
                -actor_output["mean"].detach()).abs().mean()
            action_saturation_fraction = (
                actor_output["action"].detach().abs()
                >= 1-1e-6).float().mean()

        metric_windows["reward"].append(float(reward.detach().mean()))
        metric_windows["td_error"].append(mean_abs_td_error)
        metric_windows["critic_value"].append(
            float(current_value.detach().mean()))
        metric_windows["critic_loss"].append(float(critic_loss.detach()))
        metric_windows["actor_step"].append(actor_step_norm)
        metric_windows["ghost_step"].append(ghost_step_norm)
        metric_windows["ghost_policy_gradient_norm"].append(
            ghost_diagnostics["policy_gradient_norm"])
        metric_windows["ghost_predictive_gradient_norm"].append(
            ghost_diagnostics["applied_predictive_gradient_norm"])
        metric_windows["ghost_predictive_applied_ratio"].append(
            ghost_diagnostics["predictive_to_policy_ratio"])
        metric_windows["encoder_predictor_gradient_norm"].append(
            float(predictor_gradient_norm))
        metric_windows["encoder_critic_raw_gradient_norm"].append(
            float(raw_critic_gradient_norm))
        metric_windows["encoder_critic_applied_gradient_norm"].append(
            float(applied_critic_gradient_norm))
        metric_windows["encoder_critic_applied_ratio"].append(
            encoder_critic_applied_ratio)
        metric_windows["encoder_predictor_critic_cosine"].append(
            encoder_predictor_critic_cosine)
        metric_windows["actor_std_mean"].append(
            float(behavior_std.mean()))
        metric_windows["actor_std_min"].append(
            float(behavior_std.min()))
        metric_windows["actor_std_max"].append(
            float(behavior_std.max()))
        metric_windows["sampled_action_deviation"].append(
            float(sampled_action_deviation))
        metric_windows["action_saturation_fraction"].append(
            float(action_saturation_fraction))
        if np.isfinite(curiosity_error_before_value):
            metric_windows["curiosity_error_before"].append(
                curiosity_error_before_value)
        if np.isfinite(curiosity_error_after_value):
            metric_windows["curiosity_error_after"].append(
                curiosity_error_after_value)
        if np.isfinite(curiosity_raw_improvement_value):
            metric_windows["curiosity_raw_improvement"].append(
                curiosity_raw_improvement_value)
        if np.isfinite(curiosity_learning_progress_value):
            metric_windows["curiosity_learning_progress"].append(
                curiosity_learning_progress_value)
        metric_windows["curiosity_progress_baseline"].append(
            float(curiosity_progress_baseline))
        metric_windows["curiosity_advantage_mean"].append(
            curiosity_advantage_mean)
        metric_windows["curiosity_valid_world_fraction"].append(
            float(curiosity_valid_mask.float().mean()))
        metric_windows["curiosity_terminal_matured_fraction"].append(
            curiosity_terminal_matured_fraction)
        metric_windows["actor_external_gradient_norm"].append(
            float(actor_external_gradient_norm))
        metric_windows["actor_curiosity_raw_gradient_norm"].append(
            float(actor_curiosity_raw_gradient_norm))
        metric_windows["actor_curiosity_applied_gradient_norm"].append(
            actor_curiosity_applied_gradient_norm)
        metric_windows["actor_curiosity_external_ratio"].append(
            actor_curiosity_external_ratio)
        metric_windows["ghost_external_gradient_norm"].append(
            ghost_diagnostics["external_gradient_norm"])
        metric_windows["ghost_curiosity_raw_gradient_norm"].append(
            ghost_diagnostics["curiosity_raw_gradient_norm"])
        metric_windows["ghost_curiosity_applied_gradient_norm"].append(
            ghost_diagnostics["curiosity_applied_gradient_norm"])
        metric_windows["ghost_curiosity_external_ratio"].append(
            ghost_diagnostics["curiosity_external_ratio"])
        if np.isfinite(ghost_diagnostics["predictive_loss"]):
            metric_windows["ghost_predictive_loss"].append(
                ghost_diagnostics["predictive_loss"])
        if np.isfinite(prediction_loss_value):
            metric_windows["predictor_loss"].append(prediction_loss_value)

        if transition_after_step >= next_plot_transition:
            while next_plot_transition <= transition_after_step:
                next_plot_transition += max(cfg.plot_every, 1)
            plot_history["transition"].append(transition_after_step)
            plot_history["reward"].append(rolling_metric("reward"))
            plot_history["success"].append(rolling_success_128)
            plot_history["td_error"].append(rolling_metric("td_error"))
            plot_history["critic_value"].append(
                rolling_metric("critic_value"))
            plot_history["predictor_loss"].append(
                rolling_metric("predictor_loss"))
            plot_history["critic_loss"].append(
                rolling_metric("critic_loss"))
            plot_history["actor_step"].append(
                rolling_metric("actor_step"))
            plot_history["ghost_step"].append(
                rolling_metric("ghost_step"))
            plot_history["ghost_policy_gradient_norm"].append(
                rolling_metric("ghost_policy_gradient_norm"))
            plot_history["ghost_predictive_gradient_norm"].append(
                rolling_metric("ghost_predictive_gradient_norm"))
            plot_history["ghost_predictive_applied_ratio"].append(
                rolling_metric("ghost_predictive_applied_ratio"))
            plot_history["ghost_predictive_loss"].append(
                rolling_metric("ghost_predictive_loss"))
            plot_history["encoder_predictor_gradient_norm"].append(
                rolling_metric("encoder_predictor_gradient_norm"))
            plot_history["encoder_critic_raw_gradient_norm"].append(
                rolling_metric("encoder_critic_raw_gradient_norm"))
            plot_history[
                "encoder_critic_applied_gradient_norm"].append(
                    rolling_metric(
                        "encoder_critic_applied_gradient_norm"))
            plot_history["encoder_critic_applied_ratio"].append(
                rolling_metric("encoder_critic_applied_ratio"))
            plot_history["encoder_predictor_critic_cosine"].append(
                rolling_metric("encoder_predictor_critic_cosine"))
            plot_history["actor_std_mean"].append(
                rolling_metric("actor_std_mean"))
            plot_history["actor_std_min"].append(
                rolling_metric("actor_std_min"))
            plot_history["actor_std_max"].append(
                rolling_metric("actor_std_max"))
            plot_history["sampled_action_deviation"].append(
                rolling_metric("sampled_action_deviation"))
            plot_history["action_saturation_fraction"].append(
                rolling_metric("action_saturation_fraction"))
            for curiosity_metric in (
                    "curiosity_error_before",
                    "curiosity_error_after",
                    "curiosity_raw_improvement",
                    "curiosity_learning_progress",
                    "curiosity_progress_baseline",
                    "curiosity_advantage_mean",
                    "curiosity_valid_world_fraction",
                    "curiosity_terminal_matured_fraction",
                    "actor_external_gradient_norm",
                    "actor_curiosity_raw_gradient_norm",
                    "actor_curiosity_applied_gradient_norm",
                    "actor_curiosity_external_ratio",
                    "ghost_external_gradient_norm",
                    "ghost_curiosity_raw_gradient_norm",
                    "ghost_curiosity_applied_gradient_norm",
                    "ghost_curiosity_external_ratio"):
                plot_history[curiosity_metric].append(
                    rolling_metric(curiosity_metric))
            plot_history["curriculum_stage"].append(
                env.curriculum_stage)
            plot_history["critic_frozen"].append(critic_frozen)
            plot_history["all_models_frozen"].append(
                all_models_frozen)
            render_training_dashboard()

        observation = (
            env.observation() if advanced_curriculum
            else transition.observation)
        transitions += b

    final_path = Path(cfg.checkpoint_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": asdict(cfg),
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "ghost": ghost.state_dict(),
        "actor": actor.state_dict(),
        "predictor": predictor.state_dict(),
        "critic": critic.state_dict(),
        "target_critic": target_critic.state_dict(),
        "critic_frozen": critic_frozen,
        "all_models_frozen": all_models_frozen,
    }, final_path)
    print(f"saved={final_path}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
