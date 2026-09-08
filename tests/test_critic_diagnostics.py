import numpy as np
from ghost.ghost_terminal_core import Config, BatchedTMaze
from experiments.delayed_cue_tmaze.critic_diagnostics import CriticDiagnostics, oracle_distances, successor


def test_oracle_accounts_for_orientation_and_wrong_terminal():
    env = BatchedTMaze(Config(worlds=1), 12)
    distances = oracle_distances(env, 0)
    assert (*env.right_goal, 0) not in distances
    for state, distance in distances.items():
        if distance:
            assert min(distances.get(successor(env, state, a), float("inf")) for a in range(3)) == distance-1


def test_return_backfill_and_censoring():
    env = BatchedTMaze(Config(worlds=2), 12)
    logger = CriticDiagnostics(env, 12, "separated")
    for rewards, done in [(np.array([0., 0.]), np.array([False, False])),
                          (np.array([1., 0.]), np.array([True, False]))]:
        logger.before(env, np.array([0, 0]), np.array([.2, .2]), np.zeros(2))
        logger.after(rewards, done, .9)
    assert logger.rows[0]["discounted_return"] == .9
    assert logger.rows[0]["critic_error"] == .2-.9
    assert logger.rows[2]["discounted_return"] == 1.
    assert logger.rows[1]["censored"]
