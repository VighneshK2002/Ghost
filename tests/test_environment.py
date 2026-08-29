import numpy as np

from environments.delayed_cue_tmaze import BatchedTMaze
from ghost import Config


def test_observation_shape_and_cue_visibility() -> None:
    config = Config(worlds=2)
    environment = BatchedTMaze(config, seed=7, curriculum=False)
    environment.cue[:] = (0, 1)
    environment.age[:] = 0
    visible = environment.observation()

    assert visible.shape == (2, config.observation_dim)
    np.testing.assert_array_equal(visible[:, 9:12], ((1, 0, 0), (0, 1, 0)))

    environment.age[:] = config.cue_steps
    hidden = environment.observation()
    np.testing.assert_array_equal(hidden[:, 9:12], ((0, 0, 1), (0, 0, 1)))


def test_delayed_junction_is_ambiguous_without_memory() -> None:
    config = Config(worlds=2)
    environment = BatchedTMaze(config, seed=9, curriculum=False)
    environment.x[:] = environment.center
    environment.y[:] = 1
    environment.direction[:] = 0
    environment.previous_action[:] = 2
    environment.cue[:] = (0, 1)
    environment.age[:] = config.cue_steps + 5

    observation = environment.observation()
    np.testing.assert_array_equal(observation[0], observation[1])


def test_terminal_step_preserves_pre_reset_transition_observation() -> None:
    config = Config(worlds=1)
    environment = BatchedTMaze(config, seed=3, curriculum=False)
    environment.x[0], environment.y[0] = environment.center, 1
    environment.direction[0] = 3  # west
    environment.cue[0] = 0

    _, reward, done, success, wrong, _, _ = environment.step(np.array([2]))

    assert not bool(done[0])
    assert reward[0] == 0.0
    assert not bool(success[0])
    assert not bool(wrong[0])
    assert environment.transition_observation.shape == (1, config.observation_dim)
