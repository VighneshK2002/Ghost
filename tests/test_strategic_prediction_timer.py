import torch

from ghost import Config, System
from ghost.ghost_terminal_core import TimedPredictionBuffer


def timer_system(*, worlds: int = 2, durations=(2,)) -> tuple[System, Config]:
    config = Config(
        worlds=worlds,
        use_strategic_prediction_timer=True,
        prediction_timer_durations=durations,
        use_strategy_encoder_eprop=False,
    )
    return System(config, "separated", torch.device("cpu"), seed=11), config


def test_timer_is_opt_in_and_preserves_legacy_topology() -> None:
    config = Config(worlds=2)
    system = System(config, "separated", torch.device("cpu"), seed=11)

    assert system.strategizer.timer_head is None
    assert system.predictor.core.input.in_features == (
        config.latent_dim+config.conditioning_dim+3
    )


def test_timed_prediction_resolves_only_at_selected_horizon() -> None:
    system, config = timer_system(worlds=1, durations=(2,))
    buffer = TimedPredictionBuffer(config, torch.device("cpu"))
    latent = torch.zeros(1, config.latent_dim)
    strategy = torch.zeros(1, config.strategy_dim)
    desirability = torch.zeros(1)
    action = torch.zeros(1, dtype=torch.long)
    timer_index = torch.zeros(1, dtype=torch.long)
    duration = torch.full((1,), 2, dtype=torch.long)
    target = torch.ones_like(latent)
    reward = torch.zeros(1)
    done = torch.zeros(1, dtype=torch.bool)

    buffer.forecast(
        system, latent, strategy, desirability, action,
        timer_index, duration,
    )
    _, _, first_mse, _, _, _ = buffer.resolve(
        system, target, reward, done, train=False)
    assert first_mse == 0.0

    buffer.forecast(
        system, latent, strategy, desirability, action,
        timer_index, duration,
    )
    _, _, second_mse, _, _, _ = buffer.resolve(
        system, target, reward, done, train=False)
    assert second_mse > 0.0


def test_episode_end_cancels_unexpired_predictions() -> None:
    system, config = timer_system(worlds=1, durations=(4,))
    buffer = TimedPredictionBuffer(config, torch.device("cpu"))
    latent = torch.zeros(1, config.latent_dim)
    strategy = torch.zeros(1, config.strategy_dim)
    scalar = torch.zeros(1)
    index = torch.zeros(1, dtype=torch.long)

    buffer.forecast(
        system, latent, strategy, scalar, index,
        index, torch.full((1,), 4, dtype=torch.long),
    )
    buffer.resolve(
        system, latent, scalar, torch.ones(1, dtype=torch.bool), train=False)

    assert buffer.pending == []


def test_td_credit_updates_timer_head() -> None:
    system, config = timer_system(worlds=4, durations=(1, 2, 4))
    latent = torch.randn(config.worlds, config.latent_dim)
    strategy, actor, actor_strategy, _, _ = system.strategy_and_action(latent)
    before = [parameter.detach().clone()
              for parameter in system.strategizer.timer_head.parameters()]

    system.strategy_eprop.accumulate(
        strategy["proposal"],
        strategy["gate"],
        strategy["previous_strategy"],
        actor_strategy,
        actor["logp"],
        config.strategy_retention,
        learned_gate=config.learned_strategy_memory,
        timer_logp=strategy["timer_logp"],
    )
    system.strategy_eprop.apply(torch.ones(config.worlds))

    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, system.strategizer.timer_head.parameters())
    )
