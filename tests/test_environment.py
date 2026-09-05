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


def test_training_cue_schedule_enforces_each_segment_quota() -> None:
    config = Config(
        worlds=10,
        cue_probability_schedule=((0, 0.8), (20, 0.2)),
    )
    environment = BatchedTMaze(config, seed=3, curriculum=True)

    np.testing.assert_array_equal(environment.cue_assignment_counts, (8, 2))
    assert environment.current_cue_probability_left == 0.8
    assert environment.cue_assignment_probability_sum == 8.0

    before = environment.cue_assignment_counts.copy()
    environment.completed_transitions = 20
    environment.reset(np.ones(config.worlds, dtype=bool))

    np.testing.assert_array_equal(
        environment.cue_assignment_counts - before,
        (2, 8),
    )
    assert environment.current_cue_probability_left == 0.2
    assert environment.cue_assignment_probability_sum == 10.0


def test_training_without_schedule_preserves_legacy_random_sampler() -> None:
    config = Config(worlds=8)
    expected = np.random.default_rng(5).integers(0, 2, config.worlds)

    environment = BatchedTMaze(config, seed=5, curriculum=True)

    np.testing.assert_array_equal(environment.cue, expected)


def test_evaluation_ignores_training_cue_schedule() -> None:
    config = Config(
        worlds=8,
        cue_probability_schedule=((0, 1.0),),
    )
    expected = np.random.default_rng(5).integers(0, 2, config.worlds)

    environment = BatchedTMaze(config, seed=5, curriculum=False)

    np.testing.assert_array_equal(environment.cue, expected)
    assert environment.current_cue_probability_left == 0.5
