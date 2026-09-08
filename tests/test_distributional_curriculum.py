"""Distributional starts must not leak cue, stage, or mutable time limits."""
import numpy as np
import pytest
from ghost import Config
from environments.delayed_cue_tmaze import BatchedTMaze


def test_each_stage_varies_overlaps_and_shifts_longer():
    env = BatchedTMaze(Config(worlds=10000), 12)
    previous = None
    means = []
    for stage, distribution in enumerate(env.curriculum_length_distributions):
        env.curriculum_stage = stage
        env.reset(np.ones(env.cfg.worlds, bool))
        assert len(set(env.hallway_length)) > 1
        assert np.array_equal(env.y, env.hallway_length)
        assert sum(distribution.values()) == pytest.approx(1.)
        means.append(sum(h*p for h,p in distribution.items()))
        for length,p in distribution.items():
            assert np.mean(env.hallway_length == length) == pytest.approx(p,abs=.02)
            rates = [np.mean(env.hallway_length[env.cue==cue] == length) for cue in (0,1)]
            assert abs(rates[0]-rates[1]) < .035
        if previous is not None: assert set(previous)&set(distribution)
        previous=distribution
    assert all(a<b for a,b in zip(means,means[1:]))
    assert all(1 in d for d in env.curriculum_length_distributions)


def test_reset_mask_seed_and_observation_invariance():
    cfg=Config(worlds=24)
    a,b=BatchedTMaze(cfg,5),BatchedTMaze(cfg,5)
    np.testing.assert_array_equal(a.hallway_length,b.hallway_length)
    mask=np.arange(24)%2==0
    before=a.hallway_length.copy(); limits=a.episode_time_limit.copy()
    a.curriculum_stage=3; a.reset(mask)
    np.testing.assert_array_equal(a.hallway_length[~mask],before[~mask])
    np.testing.assert_array_equal(a.episode_time_limit[~mask],limits[~mask])
    observation=a.observation()
    a.hallway_length[:]=99; a.episode_stage[:]=99
    np.testing.assert_array_equal(a.observation(),observation)
    assert observation.shape==(24,15)


@pytest.mark.parametrize('height',[4,5,7,9,12])
def test_geometry_and_optimal_route_timeout(height):
    cfg=Config(worlds=2,maze_height=height)
    env=BatchedTMaze(cfg,7,curriculum=False)
    assert all(len(d)>=2 for d in env.curriculum_length_distributions)
    assert max(env.curriculum_length_distributions[-1])==height-2
    for length in range(1,height-1):
        env.reset(np.ones(2,bool))
        env.y[:]=length; env.hallway_length[:]=length
        env.episode_time_limit[:]=env._limit_for_length(length)
        env.cue[:]=[0,1]
        for _ in range(length-1):
            _,_,done,*_=env.step(np.array([2,2])); assert not done.any()
        _,_,done,*_=env.step(np.array([0,1])); assert not done.any()
        for _ in range(env.center-1):
            _,reward,done,success,wrong,_,_=env.step(np.array([2,2]))
        assert done.all() and success.all() and not wrong.any()
        np.testing.assert_array_equal(reward,[1.,1.])


def test_promotion_requires_both_cues_and_preserves_inflight_limits():
    cfg=Config(worlds=2,curriculum_min_episodes_per_cue=1)
    env=BatchedTMaze(cfg,3)
    env.curriculum_history[0].append(1.)
    env.x[0]=2;env.y[0]=1;env.direction[0]=3;env.cue[0]=0
    env.step(np.array([2,0]))
    assert env.curriculum_stage==0  # Missing right-cue evidence.
    env.curriculum_history[1].append(1.)
    old_limit=env.episode_time_limit[1]
    env.x[0]=2;env.y[0]=1;env.direction[0]=3;env.cue[0]=0
    env.step(np.array([2,0]))
    assert env.curriculum_stage==1
    assert env.episode_stage.tolist()==[1,0]
    assert env.episode_time_limit[1]==old_limit
    env.x[1]=2;env.y[1]=1;env.direction[1]=3;env.cue[1]=1
    env.step(np.array([0,2]))
    assert all(not h for h in env.curriculum_history)  # Old-stage failure excluded.
    assert env.transition_episode_stage[1]==0 and env.episode_stage[1]==1


def test_timeout_rewards_and_terminal_length_metadata():
    env=BatchedTMaze(Config(worlds=24),8)
    lengths=env.hallway_length.copy()
    env.age[:]=env.episode_time_limit-1
    _,rewards,done,success,wrong,_,ages=env.step(np.zeros(24,dtype=int))
    assert done.all() and not success.any() and not wrong.any()
    np.testing.assert_allclose(rewards,env.cfg.timeout_penalty)
    np.testing.assert_array_equal(env.transition_hallway_length,lengths)
    np.testing.assert_array_equal(ages,env.transition_episode_time_limit)


def test_length_diagnostics_do_not_hide_rare_or_failed_lengths():
    records=[dict(hallway_length=1,cue=i%2,success=1.,curriculum_stage=1) for i in range(8)]
    records += [dict(hallway_length=3,cue=i%2,success=0.,curriculum_stage=1) for i in range(8)]
    stats=BatchedTMaze.summarize_hallways(records,8,({1:.5,2:.1,3:.4},))
    assert stats['worst_length_success']==0.
    assert stats['worst_cue_success']==.5
    assert stats['mean_hallway_length']==2.
    assert stats['success_by_hallway_length'][1]['success'] is None
    assert stats['success_by_hallway_length'][2]['success_left']==0.
    assert stats['hallway_distribution_by_stage'][1]['samples']==0


@pytest.mark.parametrize('distributions',[
    (((1,1.),),),
    (((1,.5),(2,.5)),((3,.5),(4,.5))),
    (((2,.5),(3,.5)),((1,.5),(2,.5))),
    (((1,.5),(99,.5)),),
])
def test_invalid_distributions_rejected(distributions):
    with pytest.raises(ValueError):
        BatchedTMaze(Config(curriculum_length_distributions=distributions),1)


def test_ten_stages_interpolate_anchors_without_changing_endpoints():
    env=BatchedTMaze(Config(worlds=1),12)
    stages=env.curriculum_length_distributions
    assert len(stages)==10
    assert stages[0]==pytest.approx({1:.8,2:.15,3:.05})
    assert stages[-1]==pytest.approx({1:.02,2:.03,3:.05,4:.10,5:.20,6:.25,7:.35})
    for start in (0,3,6):
        left,right=stages[start],stages[start+3]
        support=set(left)|set(right)
        distance=sum(abs(left.get(h,0)-right.get(h,0)) for h in support)
        for offset in (1,2,3):
            alpha=offset/3
            expected={h:(1-alpha)*left.get(h,0)+alpha*right.get(h,0) for h in support}
            assert stages[start+offset]==pytest.approx(expected)
            prior,current=stages[start+offset-1],stages[start+offset]
            assert sum(abs(prior.get(h,0)-current.get(h,0)) for h in support)==pytest.approx(distance/3)
    custom=(((1,.8),(2,.2)),((1,.2),(2,.8)))
    assert len(BatchedTMaze(Config(curriculum_length_distributions=custom),12).curriculum_length_distributions)==2


def test_promotion_blends_by_completed_episodes_and_defers_mastery():
    cfg=Config(worlds=2,curriculum_min_episodes_per_cue=1,curriculum_blend_episodes=4)
    env=BatchedTMaze(cfg,12)
    for h in env.curriculum_history:h.extend([1.]*8)
    env.age[:]=env.episode_time_limit-1
    env.step(np.zeros(2,dtype=int))
    assert env.curriculum_stage==1
    assert env.curriculum_blend_fraction==0.
    assert env.current_length_distribution==env.curriculum_length_distributions[0]
    # Reset both at the beginning of the blend, then complete those episodes.
    env.reset(np.ones(2,bool))
    for expected in (.5,1.):
        env.age[:]=env.episode_time_limit-1
        env.step(np.zeros(2,dtype=int))
        assert env.curriculum_blend_fraction==expected
        assert all(not h for h in env.curriculum_history)
        assert env.episode_blend.tolist()==[expected,expected]
    assert env.current_length_distribution==env.curriculum_length_distributions[1]
    env.age[:]=env.episode_time_limit-1
    env.cue[:]=[0,1]
    env.step(np.zeros(2,dtype=int))
    assert [len(h) for h in env.curriculum_history]==[1,1]


def test_blend_interpolation_endpoints_and_evaluation():
    env=BatchedTMaze(Config(worlds=24,curriculum_blend_episodes=10),7)
    env.curriculum_stage=1
    left,right=env.curriculum_length_distributions[:2]
    for complete in (0,1,5,10):
        env.curriculum_blend_completed=complete
        actual=env.current_length_distribution
        for h in set(left)|set(right):
            assert actual.get(h,0.)==pytest.approx((1-complete/10)*left.get(h,0.)+complete/10*right.get(h,0.))
    evaluation=BatchedTMaze(env.cfg,7,curriculum=False)
    assert evaluation.curriculum_blend_fraction==1.
    assert evaluation.current_length_distribution==evaluation.curriculum_length_distributions[-1]
    immediate=BatchedTMaze(Config(curriculum_blend_episodes=0),7)
    immediate.curriculum_stage=1
    immediate.curriculum_blend_completed=0
    assert immediate.current_length_distribution==immediate.curriculum_length_distributions[1]
