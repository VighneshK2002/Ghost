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
        q_trace_decay: float = 0.95
        ghost_trace_decay: float = 0.99
        encoder_lr: float = 3e-4
        predictor_lr: float = 3e-4
        naf_lr: float = 3e-4
        ghost_eprop_lr: float = 1e-4
        naf_gradient_clip: float = 1.0
        freeze_all_above_success: bool = False
        freeze_all_success_threshold: float = 0.90

        initial_exploration_std: float = 0.25
        minimum_exploration_std: float = 0.05
        exploration_decay_transitions: int = 100_000
        naf_precision_epsilon: float = 1e-4

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

        naf_target_tau: float = 0.005
        encoder_target_tau: float = 0.005
        encoder_q_credit_weight: float = 1.0
        q_to_ghost_credit_weight: float = 1.0
        ghost_gradient_clip: float = 1.0

        horizon_uniform_eta: float = 0.10
        actual_horizon_min_probability: float = 0.02
        predictor_logvar_min: float = -5.0
        predictor_logvar_max: float = 2.0
        return_loss_weight: float = 1.0
        plot_every: int = 1_024
        plot_window: int = 128

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
def _(Config, F, nn, torch):
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


    @torch.no_grad()
    def polyak_update(target, online, tau):
        for target_parameter, parameter in zip(
                target.parameters(), online.parameters()):
            target_parameter.mul_(1-tau).add_(parameter, alpha=tau)


    return RecurrentSNN, SurrogateSpike, polyak_update


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


    class GhostQEligibility:
        """Q semi-gradient eligibility through the recurrent Ghost output."""

        def __init__(self, ghost: Ghost, cfg: Config):
            self.parameters = list(ghost.parameters())
            self.decay = cfg.ghost_trace_decay
            self.weight = cfg.q_to_ghost_credit_weight
            self.max_norm = cfg.ghost_gradient_clip
            self.recurrent_eprop = RecurrentGhostEprop(
                ghost, self.parameters, cfg.worlds,
                cfg.ghost_eprop_lr)
            self.optimizer = self.recurrent_eprop.optimizer
            self.q_traces = [
                torch.zeros((cfg.worlds,)+tuple(parameter.shape),
                            device=parameter.device)
                for parameter in self.parameters]

        def accumulate(self, ghost, current_q):
            if ghost.grad_fn is None:
                raise RuntimeError(
                    "live ghost was detached before Q-mediated credit")
            jacobians = self.recurrent_eprop._current_output_jacobians(ghost)
            ghost_signal = torch.autograd.grad(
                current_q.sum(), ghost, retain_graph=True)[0].detach()
            with torch.no_grad():
                for jacobian, trace in zip(jacobians, self.q_traces):
                    view = ghost_signal.shape+(1,)*(jacobian.ndim-2)
                    score = (jacobian*ghost_signal.view(view)).sum(1)
                    trace.mul_(self.decay).add_(score)

        def apply(self, td_error):
            directions = []
            for trace in self.q_traces:
                view = (len(td_error),)+(1,)*(trace.ndim-1)
                directions.append(
                    self.weight*(trace*td_error.view(view)).mean(0))
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
                for trace in self.q_traces:
                    trace[mask] = 0
            self.recurrent_eprop.reset(mask)





    return Ghost, GhostQEligibility


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Normalized Advantage Q-function
    """)
    return


@app.cell
def _(Config, F, RecurrentSNN, mlp, nn, torch):
    class NormalizedAdvantageQ(nn.Module):
        """Scalar-action NAF conditioned on latent state and recurrent Ghost.

        NAF is deliberately unimodal and quadratic in action.  A vector-action
        extension would replace scalar precision with a lower-triangular L and
        use P = L @ L.T, but PointControl currently has one bounded action.
        """

        def __init__(self, cfg: Config) -> None:
            super().__init__(); self.cfg = cfg
            self.ghost_encoder = mlp(
                cfg.ghost_dim, cfg.hidden_dim, cfg.conditioning_dim)
            self.core = RecurrentSNN(
                cfg.latent_dim+cfg.conditioning_dim+cfg.ghost_dim,
                cfg.hidden_dim, cfg, persistent=False)
            self.norm = nn.LayerNorm(2*cfg.hidden_dim)
            self.value_head = nn.Linear(2*cfg.hidden_dim, 1)
            self.mean_head = nn.Linear(2*cfg.hidden_dim, 1)
            self.precision_head = nn.Linear(2*cfg.hidden_dim, 1)
            nn.init.normal_(self.mean_head.weight, std=0.01)
            nn.init.zeros_(self.mean_head.bias)
            nn.init.zeros_(self.precision_head.weight)
            nn.init.zeros_(self.precision_head.bias)

        def components(self, latent, ghost):
            conditioning = self.ghost_encoder(ghost)
            feature = self.norm(self.core(torch.cat(
                (latent, conditioning, ghost), -1)))
            value = self.value_head(feature).squeeze(-1)
            mean = torch.tanh(
                self.mean_head(feature).squeeze(-1))
            precision = (
                F.softplus(
                    self.precision_head(feature).squeeze(-1))
                +self.cfg.naf_precision_epsilon)
            return value, mean, precision

        def value(self, latent, ghost):
            value, _, _ = self.components(latent, ghost)
            return value

        def q(self, latent, ghost, action):
            value, mean, precision = self.components(latent, ghost)
            advantage = -0.5*precision*(action-mean).square()
            return value+advantage

        def forward(self, latent, ghost, action=None):
            value, mean, precision = self.components(latent, ghost)
            result = {
                "value": value,
                "mean": mean,
                "precision": precision,
            }
            if action is not None:
                result["q"] = (
                    value
                    -0.5*precision*(action-mean).square())
            return result


    def _check_normalized_advantage_q():
        cfg = Config(
            worlds=3, hidden_dim=6, latent_dim=4,
            ghost_dim=3, conditioning_dim=5, snn_ticks=2)
        model = NormalizedAdvantageQ(cfg)
        latent = torch.randn(cfg.worlds, cfg.latent_dim)
        ghost = torch.randn(cfg.worlds, cfg.ghost_dim)
        output = model(latent, ghost)
        assert bool(torch.all(output["precision"] > 0))
        assert bool(torch.all(output["mean"].abs() <= 1))
        q_at_mean = model.q(
            latent, ghost, output["mean"].detach())
        assert torch.allclose(
            q_at_mean, output["value"], atol=1e-6, rtol=1e-5)
        displaced_action = torch.where(
            output["mean"].detach() >= 0,
            output["mean"].detach()-0.25,
            output["mean"].detach()+0.25)
        displaced_q = model.q(latent, ghost, displaced_action)
        assert bool(torch.all(
            displaced_q < model.value(latent, ghost)))

        model.zero_grad(set_to_none=True)
        fixed_action = torch.full(
            (cfg.worlds,), 0.75)
        model.q(latent, ghost, fixed_action).sum().backward()
        assert model.mean_head.bias.grad is not None
        assert bool(torch.count_nonzero(model.mean_head.bias.grad))

        target = NormalizedAdvantageQ(cfg)
        for parameter in target.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            target.value(latent, ghost)
        assert all(parameter.grad is None
                   for parameter in target.parameters())


    _check_normalized_advantage_q()
    return (NormalizedAdvantageQ,)


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


    class EncoderQTrace:
        """Per-world Q eligibility for direct and Ghost-mediated encoder paths."""

        def __init__(self, parameters, worlds, decay):
            self.parameters = list(parameters); self.decay = decay
            self.traces = [
                torch.zeros((worlds,)+tuple(parameter.shape),
                            device=parameter.device)
                for parameter in self.parameters]

        def accumulate(self, current_q):
            basis = torch.eye(
                len(current_q), device=current_q.device,
                dtype=current_q.dtype)
            gradients = torch.autograd.grad(
                current_q, self.parameters, grad_outputs=basis,
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


    return Encoder, EncoderQTrace, slice_state


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

    training_controls = {
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
        "naf_lr": number(
            defaults.naf_lr, "Learning · NAF rate",
            step=1e-5, start=0.0),
        "predictor_lr": number(
            defaults.predictor_lr, "Learning · Predictor rate",
            step=1e-5, start=0.0),
        "ghost_eprop_lr": number(
            defaults.ghost_eprop_lr, "Learning · Ghost rate",
            step=1e-5, start=0.0),
        "naf_gradient_clip": number(
            defaults.naf_gradient_clip,
            "Learning · NAF gradient clip",
            step=0.1, start=0.0),
        "freeze_all_above_success": mo.ui.switch(
            value=defaults.freeze_all_above_success,
            label=(
                "Learning · Freeze all models above success threshold")),
        "freeze_all_success_threshold": number(
            defaults.freeze_all_success_threshold,
            "Learning · All-model freeze success threshold",
            step=0.01, start=0.0, stop=1.0),
        "q_trace_decay": number(
            defaults.q_trace_decay, "Learning · Q trace decay",
            step=0.01, start=0.0, stop=1.0),
        "ghost_trace_decay": number(
            defaults.ghost_trace_decay, "Learning · Ghost trace decay",
            step=0.01, start=0.0, stop=1.0),
        "gamma": number(
            defaults.gamma, "Learning · Discount factor",
            step=0.001, start=0.0, stop=1.0),
        "naf_target_tau": number(
            defaults.naf_target_tau, "Learning · NAF target rate",
            step=0.001, start=0.0, stop=1.0),
        "encoder_target_tau": number(
            defaults.encoder_target_tau, "Learning · Encoder target rate",
            step=0.001, start=0.0, stop=1.0),
        "encoder_q_credit_weight": number(
            defaults.encoder_q_credit_weight,
            "Learning · Encoder Q credit", step=0.05, start=0.0),
        "q_to_ghost_credit_weight": number(
            defaults.q_to_ghost_credit_weight,
            "Learning · Q-to-Ghost credit", step=0.05, start=0.0),
        "ghost_gradient_clip": number(
            defaults.ghost_gradient_clip,
            "Learning · Ghost gradient clip", step=0.1, start=0.0),

        "initial_exploration_std": number(
            defaults.initial_exploration_std,
            "Exploration · Initial std", step=0.01, start=0.0),
        "minimum_exploration_std": number(
            defaults.minimum_exploration_std,
            "Exploration · Minimum std", step=0.01, start=0.0),
        "exploration_decay_transitions": number(
            defaults.exploration_decay_transitions,
            "Exploration · Decay transitions", step=1, start=1),
        "naf_precision_epsilon": number(
            defaults.naf_precision_epsilon,
            "NAF · Precision epsilon", step=1e-5, start=1e-8),

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

        "plot_every": number(
            defaults.plot_every, "Dashboard · Refresh transitions",
            step=1, start=1),
        "plot_window": number(
            defaults.plot_window, "Dashboard · Rolling window",
            step=1, start=1),
    }
    section_keys = {
        "run": (
            "seed", "device", "checkpoint_path", "worlds", "transitions"),
        "network": (
            "latent_dim", "ghost_dim", "hidden_dim", "conditioning_dim",
            "snn_ticks", "membrane_decay", "predictor_membrane_decay",
            "surrogate_scale", "encoder_latent_decay",
            "encoder_membrane_readout"),
        "learning": (
            "encoder_lr", "naf_lr", "predictor_lr", "ghost_eprop_lr",
            "naf_gradient_clip", "freeze_all_above_success",
            "freeze_all_success_threshold", "q_trace_decay",
            "ghost_trace_decay", "gamma", "naf_target_tau",
            "encoder_target_tau", "encoder_q_credit_weight",
            "q_to_ghost_credit_weight", "ghost_gradient_clip"),
        "exploration": (
            "initial_exploration_std", "minimum_exploration_std",
            "exploration_decay_transitions", "naf_precision_epsilon"),
        "environment": (
            "step_scale", "target_radius", "progress_weight",
            "action_cost", "overshoot_penalty"),
        "curriculum": (
            "curriculum_stage", "curriculum_threshold",
            "curriculum_window_episodes"),
        "predictor": (
            "horizon_uniform_eta", "actual_horizon_min_probability",
            "predictor_logvar_min", "predictor_logvar_max",
            "return_loss_weight"),
        "dashboard": ("plot_every", "plot_window"),
    }
    assigned_control_keys = [
        key for keys in section_keys.values() for key in keys]
    assert len(assigned_control_keys) == len(set(assigned_control_keys))
    assert set(assigned_control_keys) == set(training_controls)
    training_sections = {
        name: mo.ui.dictionary(
            {key: training_controls[key] for key in keys},
            label=name.title())
        for name, keys in section_keys.items()}
    training_form = mo.ui.dictionary(
        training_sections, label="Training configuration").form(
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
    Config,
    Encoder,
    EncoderQTrace,
    F,
    Ghost,
    GhostQEligibility,
    NormalizedAdvantageQ,
    Path,
    PendingPrediction,
    Predictor,
    RELATIVE_HORIZONS,
    TaskRewardPointControl,
    asdict,
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
    submitted_config = {
        key: value
        for section_values in training_form.value.values()
        for key, value in section_values.items()
    }
    for integer_field in (
            "seed", "worlds", "transitions", "latent_dim", "ghost_dim",
            "hidden_dim", "conditioning_dim", "snn_ticks",
            "exploration_decay_transitions", "curriculum_stage",
            "curriculum_window_episodes", "plot_every", "plot_window"):
        submitted_config[integer_field] = int(
            submitted_config[integer_field])
    cfg = Config(**submitted_config)
    if not (
            0 <= cfg.minimum_exploration_std
            <= cfg.initial_exploration_std):
        raise ValueError(
            "exploration std must satisfy 0 <= minimum <= initial")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    env = TaskRewardPointControl(cfg, cfg.seed)
    encoder = Encoder(cfg).to(device)
    target_encoder = copy.deepcopy(encoder).to(device)
    for parameter in target_encoder.parameters():
        parameter.requires_grad_(False)

    ghost = Ghost(cfg).to(device)
    predictor = Predictor(cfg).to(device)
    naf = NormalizedAdvantageQ(cfg).to(device)
    target_naf = copy.deepcopy(naf).to(device)
    for parameter in target_naf.parameters():
        parameter.requires_grad_(False)
    all_models_frozen = False

    encoder_optimizer = torch.optim.Adam(
        encoder.parameters(), lr=cfg.encoder_lr)
    predictor_optimizer = torch.optim.Adam(
        predictor.parameters(), lr=cfg.predictor_lr)
    naf_optimizer = torch.optim.Adam(
        naf.parameters(), lr=cfg.naf_lr)
    ghost_q_trace = GhostQEligibility(ghost, cfg)
    encoder_parameters = list(encoder.parameters())
    encoder_q_trace = EncoderQTrace(
        encoder_parameters, cfg.worlds, cfg.q_trace_decay)

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
            "reward", "td_error", "predictor_loss", "naf_loss",
            "q_value", "max_value", "precision", "exploration_std",
            "encoder_step", "ghost_step", "naf_step")}
    plot_history = {
        name: [] for name in (
            "transition", "reward", "success", "td_error",
            "predictor_loss", "naf_loss", "q_value", "max_value",
            "precision", "exploration_std", "encoder_step",
            "ghost_step", "naf_step", "curriculum_stage",
            "all_models_frozen")}
    next_plot_transition = cfg.plot_every

    observation = env.observation()
    transitions = 0


    def rolling_metric(name):
        values = metric_windows[name]
        return float(np.mean(values)) if values else float("nan")


    def render_training_dashboard():
        x = plot_history["transition"]
        figure, axes = plt.subplots(5, 2, figsize=(14, 16))
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
            x, plot_history["naf_loss"],
            label="NAF TD", color="tab:red")
        panels[3].set_title("Rolling losses")
        panels[3].legend()

        for name, label, color in (
                ("encoder_step", "Encoder", "tab:blue"),
                ("ghost_step", "Ghost", "tab:pink"),
                ("naf_step", "NAF", "tab:brown")):
            panels[4].semilogy(
                x, np.maximum(
                    np.asarray(plot_history[name]), 1e-12),
                label=label, color=color)
        panels[4].set_title("Parameter update norms")
        panels[4].legend()

        panels[5].step(
            x, plot_history["curriculum_stage"],
            where="post", color="tab:cyan")
        panels[5].set_yticks(range(env.curriculum_stage_count))
        panels[5].set_title("Curriculum stage")

        panels[6].plot(
            x, plot_history["q_value"],
            label="Executed-action Q", color="tab:olive")
        panels[6].plot(
            x, plot_history["max_value"],
            label="Maximum V", color="tab:green")
        panels[6].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
        panels[6].set_title("NAF value diagnostics")
        panels[6].legend()

        panels[7].plot(
            x, plot_history["precision"], color="tab:orange")
        panels[7].set_title("Mean NAF precision P")

        panels[8].plot(
            x, plot_history["exploration_std"], color="tab:gray")
        panels[8].set_title("Exploration standard deviation")

        panels[9].axis("off")

        for axis in panels[:9]:
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

        naf_output = naf(latent, ghost_intent)
        current_max_value = naf_output["value"]
        greedy_action = naf_output["mean"]
        precision = naf_output["precision"]
        exploration_fraction = min(
            1.0,
            transitions/max(cfg.exploration_decay_transitions, 1))
        exploration_std = (
            cfg.initial_exploration_std
            +exploration_fraction
            *(cfg.minimum_exploration_std
              -cfg.initial_exploration_std))
        exploration_noise = (
            torch.randn_like(greedy_action)*exploration_std)
        action_detached = (
            greedy_action.detach()+exploration_noise).clamp(-1, 1)
        # The executed action is fixed in Q evaluation.  Its generation graph
        # must not cancel the intended gradient into the NAF mean head.
        current_q = naf.q(
            latent, ghost_intent, action_detached.detach())
        encoder_q_trace.accumulate(current_q)
        ghost_q_trace.accumulate(ghost_intent, current_q)
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
            endpoint_value = target_naf.value(
                predicted_mean.detach(), ghost_intent.detach())
            forecast_value = (
                predicted_return.detach()
                +torch.pow(
                    torch.full_like(predicted_return, cfg.gamma),
                    actual_horizon.float())*endpoint_value)
            forecast_advantage = (
                forecast_value-current_max_value.detach())

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
        endpoint_obs = torch.as_tensor(
            transition.target_observation,
            dtype=torch.float32, device=device)
        with torch.no_grad():
            endpoint_target_latent = target_encoder(endpoint_obs)
            next_ghost_intent = query_next_ghost_intent(
                endpoint_target_latent, predicted_mean.detach(),
                forecast_advantage.detach(), uncertainty.detach())
            next_value = target_naf.value(
                endpoint_target_latent, next_ghost_intent)
            td_target = (
                reward+cfg.gamma*(~done).float()*next_value)
        td_error = td_target-current_q.detach()
        mean_abs_td_error = float(td_error.abs().mean())
        transition_after_step = transitions+b

        training_q = naf.q(
            latent.detach(), ghost_intent.detach(),
            action_detached.detach())
        naf_loss = F.smooth_l1_loss(
            training_q, td_target.detach())
        naf_step_norm = 0.0
        if not all_models_frozen:
            naf_optimizer.zero_grad(set_to_none=True)
            naf_loss.backward()
            nn.utils.clip_grad_norm_(
                naf.parameters(), cfg.naf_gradient_clip)
            naf_before = [
                parameter.detach().clone()
                for parameter in naf.parameters()]
            naf_optimizer.step()
            naf_step_norm = float(torch.stack([
                (parameter-old).square().sum()
                for parameter, old in zip(
                    naf.parameters(), naf_before)
            ]).sum().sqrt())
            polyak_update(
                target_naf, naf, cfg.naf_target_tau)

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

        # EncoderQTrace was accumulated through the live current_q graph, so
        # this direction includes encoder->NAF and encoder->Ghost->NAF exactly
        # once.  The NAF Huber loss uses detached inputs and cannot duplicate it.
        encoder_q_direction = encoder_q_trace.direction(td_error)
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
        for parameter, q_direction in zip(
                encoder_parameters, encoder_q_direction):
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            parameter.grad.add_(
                q_direction, alpha=-cfg.encoder_q_credit_weight)
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        encoder_step_norm = 0.0
        if not all_models_frozen:
            encoder_before = [
                parameter.detach().clone()
                for parameter in encoder.parameters()]
            encoder_optimizer.step()
            encoder_step_norm = float(torch.stack([
                (parameter-old).square().sum()
                for parameter, old in zip(
                    encoder.parameters(), encoder_before)
            ]).sum().sqrt())
            polyak_update(target_encoder, encoder, cfg.encoder_target_tau)

        ghost_step_norm = (
            0.0
            if all_models_frozen
            else ghost_q_trace.apply(td_error))


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
            ghost_q_trace.reset(done)
            encoder_q_trace.reset(done)
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
            reset_all = torch.ones(
                b, dtype=torch.bool, device=device)
            # Deliberately do not reset ghost.core here.  Curriculum
            # changes are not ghost_intent-reset events.
            predictor.reset(reset_all)
            ghost_q_trace.reset(reset_all)
            encoder_q_trace.reset(reset_all)
            previous_endpoint.zero_()
            previous_advantage.zero_()
            previous_uncertainty.fill_(1)
            for queue in pending:
                queue.clear()
            stage_history.clear()
            advanced_curriculum = True

        metric_windows["reward"].append(float(reward.detach().mean()))
        metric_windows["td_error"].append(mean_abs_td_error)
        metric_windows["naf_loss"].append(float(naf_loss.detach()))
        metric_windows["q_value"].append(
            float(current_q.detach().mean()))
        metric_windows["max_value"].append(
            float(current_max_value.detach().mean()))
        metric_windows["precision"].append(
            float(precision.detach().mean()))
        metric_windows["exploration_std"].append(exploration_std)
        metric_windows["encoder_step"].append(encoder_step_norm)
        metric_windows["ghost_step"].append(ghost_step_norm)
        metric_windows["naf_step"].append(naf_step_norm)
        if np.isfinite(prediction_loss_value):
            metric_windows["predictor_loss"].append(prediction_loss_value)

        if transition_after_step >= next_plot_transition:
            while next_plot_transition <= transition_after_step:
                next_plot_transition += max(cfg.plot_every, 1)
            plot_history["transition"].append(transition_after_step)
            plot_history["reward"].append(rolling_metric("reward"))
            plot_history["success"].append(rolling_success_128)
            plot_history["td_error"].append(rolling_metric("td_error"))
            plot_history["predictor_loss"].append(
                rolling_metric("predictor_loss"))
            plot_history["naf_loss"].append(
                rolling_metric("naf_loss"))
            plot_history["q_value"].append(
                rolling_metric("q_value"))
            plot_history["max_value"].append(
                rolling_metric("max_value"))
            plot_history["precision"].append(
                rolling_metric("precision"))
            plot_history["exploration_std"].append(
                rolling_metric("exploration_std"))
            plot_history["encoder_step"].append(
                rolling_metric("encoder_step"))
            plot_history["ghost_step"].append(
                rolling_metric("ghost_step"))
            plot_history["naf_step"].append(
                rolling_metric("naf_step"))
            plot_history["curriculum_stage"].append(
                env.curriculum_stage)
            plot_history["all_models_frozen"].append(
                all_models_frozen)
            render_training_dashboard()

        observation = (
            env.observation() if advanced_curriculum
            else transition.observation)
        transitions += b

    final_path = Path(cfg.checkpoint_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    # Architecture-breaking NAF format; pre-NAF checkpoints are incompatible.
    torch.save({
        "checkpoint_format": "naf_v1",
        "config": asdict(cfg),
        "transitions": transitions,
        "encoder": encoder.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "ghost": ghost.state_dict(),
        "predictor": predictor.state_dict(),
        "naf": naf.state_dict(),
        "target_naf": target_naf.state_dict(),
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
