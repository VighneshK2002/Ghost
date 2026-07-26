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

        initial_action_std: float = 0.25
        actor_min_std: float = 0.10
        actor_max_std: float = 0.60
        actor_fixed_std: float = 0.25

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
        encoder_actor_credit_weight: float = 1.0
        encoder_critic_credit_weight: float = 0.0
        actor_to_ghost_credit_weight: float = 1.0
        ghost_gradient_clip: float = 1.0

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

        # Online diagonal Fisher consolidation for encoder parameters.  Its
        # importance estimate is sourced only from actor-TD control credit.
        use_encoder_fisher: bool = True
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

        def apply(self, actor_td):
            directions = []
            for trace in self.reward_traces:
                view = (len(actor_td),)+(1,)*(trace.ndim-1)
                directions.append(
                    self.weight*(trace*actor_td.view(view)).mean(0))
            raw = torch.stack([
                value.square().sum() for value in directions]).sum().sqrt()
            scale = min(1.0, self.max_norm/max(float(raw), 1e-12))
            self.optimizer.zero_grad(set_to_none=True)
            before = [parameter.detach().clone()
                      for parameter in self.parameters]
            for parameter, direction in zip(self.parameters, directions):
                parameter.grad = direction*scale
            self.optimizer.step()
            step = torch.stack([
                (parameter-old).square().sum()
                for parameter, old in zip(self.parameters, before)
            ]).sum().sqrt()
            return float(step)

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

        def forward(self, latent: torch.Tensor, ghost: torch.Tensor):
            conditioning = self.ghost_encoder(ghost)
            feature = self.norm(self.core(torch.cat(
                (latent, conditioning, ghost), -1)))
            raw_mean = self.mean_head(feature).squeeze(-1)
            if self.cfg.actor_fixed_std > 0:
                log_std = torch.full_like(raw_mean, math.log(
                    self.cfg.actor_fixed_std))
            else:
                log_std = (self.log_std
                    +self.log_std_head(feature).squeeze(-1)).clamp(
                        math.log(self.cfg.actor_min_std),
                        math.log(self.cfg.actor_max_std))
            std = log_std.exp()
            mean = torch.tanh(raw_mean)
            if self.cfg.actor_fixed_std > 0:
                # Fixed exploration is injected directly in bounded action space.
                distribution = torch.distributions.Normal(mean, std)
                pre_action = distribution.sample()
                action = pre_action.clamp(-1, 1)
                logp = distribution.log_prob(pre_action)
            else:
                distribution = torch.distributions.Normal(raw_mean, std)
                pre_action = distribution.sample()
                action = torch.tanh(pre_action)
                logp = distribution.log_prob(pre_action)-torch.log(
                    1-action.square()+1e-6)
            return {"action": action, "logp": logp}

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


    return Encoder, EncoderScoreTrace, slice_state


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
                cfg.latent_dim+cfg.conditioning_dim+2, cfg.hidden_dim, cfg,
                persistent=True, decay=cfg.predictor_membrane_decay)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.mean_head = nn.Linear(2*cfg.hidden_dim, cfg.latent_dim)
            self.logvar_head = nn.Linear(2*cfg.hidden_dim, cfg.latent_dim)
            self.return_head = nn.Linear(2*cfg.hidden_dim, 1)

        def forward(self, latent, ghost, relative_horizon, budget_fraction):
            condition = self.ghost_encoder(ghost)
            feature = self.norm(self.core(torch.cat((
                latent, condition, relative_horizon[:, None],
                budget_fraction[:, None]), -1)))
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
        """Stationary V(z, current recurrent ghost output)."""

        def __init__(self, cfg: Config) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.latent_dim+cfg.ghost_dim, cfg.hidden_dim),
                nn.SiLU(), nn.LayerNorm(cfg.hidden_dim),
                nn.Linear(cfg.hidden_dim, 1))

        def forward(self, latent, ghost):
            return self.net(torch.cat((latent, ghost), -1)).squeeze(-1)


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
        "encoder_actor_credit_weight": number(
            defaults.encoder_actor_credit_weight,
            "Learning · Encoder actor credit", step=0.05, start=0.0),
        "encoder_critic_credit_weight": number(
            defaults.encoder_critic_credit_weight,
            "Learning · Encoder critic credit", step=0.05, start=0.0),
        "actor_to_ghost_credit_weight": number(
            defaults.actor_to_ghost_credit_weight,
            "Learning · Actor-to-ghost credit", step=0.05, start=0.0),
        "ghost_gradient_clip": number(
            defaults.ghost_gradient_clip,
            "Learning · Ghost gradient clip", step=0.1, start=0.0),

        "initial_action_std": number(
            defaults.initial_action_std, "Actor · Initial std",
            step=0.01, start=0.001),
        "actor_min_std": number(
            defaults.actor_min_std, "Actor · Minimum std",
            step=0.01, start=0.001),
        "actor_max_std": number(
            defaults.actor_max_std, "Actor · Maximum std",
            step=0.01, start=0.001),
        "actor_fixed_std": number(
            defaults.actor_fixed_std, "Actor · Fixed std (0 = learned)",
            step=0.01, start=0.0),

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
            label="Fisher · Enable encoder consolidation"),
        "encoder_fisher_lambda": number(
            defaults.encoder_fisher_lambda,
            "Fisher · Encoder strength", step=0.01, start=0.0),
        "encoder_fisher_warmup_transitions": number(
            defaults.encoder_fisher_warmup_transitions,
            "Fisher · Encoder warmup", step=1, start=0),

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
    EncoderScoreTrace,
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
            "encoder_fisher_warmup_transitions", "plot_every",
            "plot_window"):
        submitted_config[integer_field] = int(
            submitted_config[integer_field])
    cfg = Config(**submitted_config)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    env = TaskRewardPointControl(cfg, cfg.seed)
    #Encoder
    encoder = Encoder(cfg).to(device)
    encoder_named_parameters = dict(encoder.named_parameters())
    encoder_fisher = {
        name: torch.zeros_like(
            parameter, memory_format=torch.preserve_format)
        for name, parameter in encoder_named_parameters.items()}
    encoder_fisher_anchor = {
        name: parameter.detach().clone()
        for name, parameter in encoder_named_parameters.items()}
    encoder_fisher_has_activated = False
    encoder_fisher_stage_start_transition = 0
    encoder_fisher_has_stage_transitioned = False
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
    encoder_policy_named_parameters = list(encoder.named_parameters())
    encoder_policy_parameters = [
        parameter for _, parameter in encoder_policy_named_parameters]
    encoder_actor_trace = EncoderScoreTrace(
        encoder_policy_parameters, cfg.worlds, cfg.actor_trace_decay)

    b, d = cfg.worlds, cfg.latent_dim
    previous_endpoint = torch.zeros(b, d, device=device)
    previous_advantage = torch.zeros(b, device=device)
    previous_uncertainty = torch.ones(b, device=device)

    maximum_horizon = max(cfg.curriculum_episode_limits)
    pending = [[] for _ in range(b)]
    stage_history = deque(maxlen=cfg.curriculum_window_episodes)
    rolling_success = deque(maxlen=128)
    metric_windows = {
        name: deque(maxlen=cfg.plot_window)
        for name in (
            "reward", "td_error", "critic_value", "predictor_loss",
            "critic_loss", "actor_step", "ghost_step")}
    plot_history = {
        name: [] for name in (
            "transition", "reward", "success", "td_error",
            "critic_value", "predictor_loss", "critic_loss",
            "actor_step", "ghost_step", "curriculum_stage",
            "critic_frozen", "all_models_frozen")}
    next_plot_transition = cfg.plot_every

    observation = env.observation()
    transitions = 0


    def rolling_metric(name):
        values = metric_windows[name]
        return float(np.mean(values)) if values else float("nan")


    def render_training_dashboard():
        x = plot_history["transition"]
        figure, axes = plt.subplots(4, 2, figsize=(13, 13))
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
            "Mean current critic value Q(state, strategy)")

        panels[7].axis("off")

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


    def recurrent_state_snapshot():
        state = ghost.core.snapshot()
        if state is None:
            return None
        return tuple(value.detach().clone() for value in state)

    def restore_auxiliary_core_fields(last_output, records):
        ghost.core.last_output = last_output
        ghost.core.last_eligibility_records = records

    @torch.no_grad()
    def query_next_ghost_intent(
            next_latent, endpoint, advantage, uncertainty):
        """Query s(t+1) without changing live state or e-prop records."""
        live_state = recurrent_state_snapshot()
        live_output = ghost.core.last_output
        live_records = list(ghost.core.last_eligibility_records)
        result, _ = ghost(
            next_latent, endpoint, advantage, uncertainty)
        ghost.core.restore(live_state)
        restore_auxiliary_core_fields(live_output, live_records)
        return result.detach()


    while transitions < cfg.transitions:
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=device)
        latent = encoder(obs)
        ghost_intent, horizon_logits = ghost(
            latent, previous_endpoint,
            previous_advantage, previous_uncertainty)

        current_value = critic(latent, ghost_intent.detach())
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
        budget_fraction = (
            remaining_budget.float()/float(env.current_episode_limit))

        predictor_pre = predictor.core.snapshot()
        predicted_mean, predicted_logvar, predicted_return = predictor(
            latent.detach(), ghost_intent.detach(),
            relative_horizon, budget_fraction)
        uncertainty = predicted_logvar.exp().mean(-1).sqrt()
        with torch.no_grad():
            endpoint_value = target_critic(
                predicted_mean.detach(), ghost_intent.detach())
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
                int(env.episode[world]), horizon,
                obs[world:world+1].detach().clone(),
                ghost_intent[world:world+1].detach().clone(),
                relative_horizon[world:world+1].detach().clone(),
                budget_fraction[world:world+1].detach().clone(),
                slice_state(predictor_pre, world),
                0.0, 1.0))

        actor_output = actor(latent, ghost_intent)
        actor_trace.accumulate(actor_output["logp"])
        # Preserve the full live graph here. EncoderScoreTrace therefore
        # contains both encoder->actor and encoder->ghost->actor score
        # credit. Detaching ghost_intent would silently remove the latter.
        encoder_actor_trace.accumulate(actor_output["logp"])
        ghost_intent_trace.accumulate(ghost_intent, actor_output["logp"])

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
            next_ghost_intent = query_next_ghost_intent(
                endpoint_target_latent, predicted_mean.detach(),
                forecast_advantage.detach(), uncertainty.detach())
            next_target = target_critic(
                endpoint_target_latent, next_ghost_intent)
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
        if (cfg.use_encoder_fisher
                and not encoder_fisher_has_activated
                and transition_after_step
                >= cfg.encoder_fisher_warmup_transitions
                and rolling_success_128
                >= cfg.encoder_fisher_activation_success):
            encoder_fisher_has_activated = True
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
        if encoder_fisher_has_stage_transitioned:
            encoder_fisher_stage_age = max(
                0, transition_after_step
                -encoder_fisher_stage_start_transition)
            encoder_fisher_stage_multiplier = min(
                1.0,
                cfg.encoder_fisher_stage_initial_multiplier
                +(1.0-cfg.encoder_fisher_stage_initial_multiplier)
                *encoder_fisher_stage_age
                /max(1, cfg.encoder_fisher_stage_recovery_transitions))
        else:
            encoder_fisher_stage_multiplier = 1.0
        effective_encoder_fisher_lambda = (
            cfg.encoder_fisher_lambda
            *encoder_fisher_stage_multiplier)

        # Reconstruct the actor's ordinary eligibility-trace control direction.
        # This is the sole Fisher source.
        actor_control_gradients = []
        for trace in actor_trace.traces:
            view = (len(actor_td),)+(1,)*(trace.ndim-1)
            actor_control_gradients.append(
                (trace*actor_td.view(view)).mean(0))
        actor_control_gradients = clip_gradient_list(
            actor_control_gradients, 1.0)

        with torch.no_grad():
            if cfg.use_actor_fisher:
                for ((name, _), control_grad) in zip(
                        actor_named_parameters.items(),
                        actor_control_gradients):
                    if (control_grad is None
                            or not bool(torch.count_nonzero(
                                control_grad.detach()))):
                        continue
                    fisher = actor_fisher[name]
                    fisher.mul_(cfg.actor_fisher_decay)
                    fisher.addcmul_(
                        control_grad.detach(), control_grad.detach(),
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
        for parameter, control_grad, fisher_gradient in zip(
                actor_trace.parameters, actor_control_gradients,
                actor_fisher_gradients):
            # Actor Adam uses maximize=True.  The positive derivative of the
            # EWC penalty is therefore subtracted from the maximizing control
            # direction so it remains a restoring force.
            parameter.grad = (
                control_grad-fisher_gradient).detach().clone()
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

        critic_per_world_loss = (current_value-td_target).square()
        critic_encoder_jacobians = torch.autograd.grad(
            critic_per_world_loss, encoder_policy_parameters,
            grad_outputs=torch.eye(b, device=device),
            is_grads_batched=True, retain_graph=True, allow_unused=True)
        critic_encoder_jacobians = [
            (torch.zeros(
                (b,)+tuple(parameter.shape),
                device=parameter.device, dtype=parameter.dtype)
             if jacobian is None else jacobian.detach())
            for parameter, jacobian in zip(
                encoder_policy_parameters, critic_encoder_jacobians)]
        critic_update_losses = []
        critic_latent = latent.detach()
        critic_ghost = ghost_intent.detach()
        fixed_td_target = td_target.detach()
        critic_update_count = (
            0
            if critic_frozen or all_models_frozen
            else cfg.critic_updates_per_step)
        for _ in range(critic_update_count):
            critic_prediction = critic(critic_latent, critic_ghost)
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
                    critic(critic_latent, critic_ghost),
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
                    matured.append((world, new_record))
                else:
                    updated.append(new_record)
            pending[world] = updated

        predictor_live = predictor.core.snapshot()
        predictor_losses = []
        for world, record in matured:
            predictor.core.restore(record.predictor_state)
            source_latent = encoder(record.source_observation)
            replay_mean, replay_logvar, replay_return = predictor(
                source_latent, record.source_ghost_intent,
                record.relative_horizon, record.budget_fraction)
            target = endpoint_target_latent[world:world+1]
            assert not target.requires_grad
            squared = (target-replay_mean).square()
            nll = 0.5*(
                squared*torch.exp(-replay_logvar)+replay_logvar).mean()
            return_target = torch.tensor(
                [record.discounted_return], device=device)
            return_loss = F.mse_loss(replay_return, return_target)
            prediction_loss = nll+cfg.return_loss_weight*return_loss
            predictor_losses.append(prediction_loss)
        predictor.core.restore(predictor_live)

        actor_encoder_direction = encoder_actor_trace.direction(actor_td)
        # This is the exact minimizing-gradient equivalent of the encoder's
        # complete actor-TD control update. Because the score trace was taken
        # through live ``ghost_intent``, this includes both the direct actor path
        # and the actor-mediated ghost path exactly once. Predictor and
        # critic directions remain excluded from the Fisher estimate.
        encoder_control_gradients = {
            name: (
                -cfg.encoder_actor_credit_weight*direction.detach())
            for (name, _), direction in zip(
                encoder_policy_named_parameters,
                actor_encoder_direction)}
        with torch.no_grad():
            if cfg.use_encoder_fisher:
                for name, control_gradient in (
                        encoder_control_gradients.items()):
                    if not bool(torch.count_nonzero(control_gradient)):
                        continue
                    fisher = encoder_fisher[name]
                    fisher.mul_(cfg.encoder_fisher_decay)
                    fisher.addcmul_(
                        control_gradient, control_gradient,
                        value=1.0-cfg.encoder_fisher_decay)

        encoder_fisher_gradients = {}
        with torch.no_grad():
            for name, parameter in encoder_named_parameters.items():
                fisher = encoder_fisher[name]
                normalized_fisher = (
                    fisher/(fisher.mean()+1e-8)).clamp(
                        min=0.0,
                        max=cfg.encoder_fisher_normalized_max)
                if (cfg.use_encoder_fisher
                        and encoder_fisher_has_activated):
                    encoder_fisher_gradients[name] = (
                        effective_encoder_fisher_lambda
                        *normalized_fisher
                        *(parameter.detach()
                          -encoder_fisher_anchor[name]))
                else:
                    encoder_fisher_gradients[name] = torch.zeros_like(
                        parameter)

        critic_encoder_direction = [
            jacobian.mean(0) for jacobian in critic_encoder_jacobians]
        encoder_optimizer.zero_grad(set_to_none=True)
        predictor_optimizer.zero_grad(set_to_none=True)
        prediction_loss_value = float("nan")
        if predictor_losses:
            prediction_loss = torch.stack(predictor_losses).mean()
            prediction_loss_value = float(prediction_loss.detach())
            prediction_loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        if not all_models_frozen:
            predictor_optimizer.step()
        for parameter, actor_direction, critic_direction in zip(
                encoder_policy_parameters, actor_encoder_direction,
                critic_encoder_direction):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(
                actor_direction, alpha=-cfg.encoder_actor_credit_weight)
            parameter.grad.add_(
                critic_direction, alpha=cfg.encoder_critic_credit_weight)
        # Encoder Adam minimizes.  The positive Fisher gradient is therefore
        # added to the ordinary prediction/control gradient as a restoring
        # force before the existing global clip and optimizer step.
        for name, parameter in encoder_named_parameters.items():
            fisher_gradient = encoder_fisher_gradients[name]
            if not bool(torch.count_nonzero(fisher_gradient)):
                continue
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(fisher_gradient)
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        if not all_models_frozen:
            encoder_optimizer.step()
        with torch.no_grad():
            if cfg.use_encoder_fisher and not all_models_frozen:
                for name, parameter in encoder_named_parameters.items():
                    fisher = encoder_fisher[name]
                    normalized_fisher = (
                        fisher/(fisher.mean()+1e-8)).clamp(
                            min=0.0,
                            max=cfg.encoder_fisher_normalized_max)
                    local_anchor_rate = (
                        cfg.encoder_fisher_anchor_lr
                        /(1.0+cfg.encoder_fisher_importance_scale
                          *normalized_fisher))
                    anchor = encoder_fisher_anchor[name]
                    anchor.add_(
                        local_anchor_rate
                        *(parameter.detach()-anchor))
        if not all_models_frozen:
            polyak_update(target_encoder, encoder, cfg.encoder_target_tau)

        ghost_step_norm = (
            0.0
            if all_models_frozen
            else ghost_intent_trace.apply(actor_td))


        previous_endpoint = predicted_mean.detach()
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
            encoder_actor_trace.reset(done)
            previous_endpoint[done] = 0
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
            encoder_fisher_stage_start_transition = transitions+b
            encoder_fisher_has_stage_transitioned = True
            reset_all = torch.ones(
                b, dtype=torch.bool, device=device)
            # Deliberately do not reset ghost.core here.  Curriculum
            # changes are not ghost_intent-reset events.
            predictor.reset(reset_all)
            actor_trace.reset(reset_all)
            ghost_intent_trace.reset(reset_all)
            encoder_actor_trace.reset(reset_all)
            previous_endpoint.zero_()
            previous_advantage.zero_()
            previous_uncertainty.fill_(1)
            for queue in pending:
                queue.clear()
            stage_history.clear()
            advanced_curriculum = True

        metric_windows["reward"].append(float(reward.detach().mean()))
        metric_windows["td_error"].append(mean_abs_td_error)
        metric_windows["critic_value"].append(
            float(current_value.detach().mean()))
        metric_windows["critic_loss"].append(float(critic_loss.detach()))
        metric_windows["actor_step"].append(actor_step_norm)
        metric_windows["ghost_step"].append(ghost_step_norm)
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
