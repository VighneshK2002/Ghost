"""Run notebook model cells headlessly; protect canonical Ghost mechanics."""
import ast
import itertools
from pathlib import Path

import pytest
import torch

from ghost import ghost_terminal_core as core


NOTEBOOK = Path(__file__).resolve().parents[1] / "ghost_notebook.py"


@pytest.fixture(scope="module")
def notebook():
    namespace = {}
    cells = [node for node in ast.parse(NOTEBOOK.read_text()).body
             if isinstance(node, ast.FunctionDef)]
    for cell in cells[:5]:
        body = [node for node in cell.body if not isinstance(node, ast.Return)]
        exec(compile(ast.Module(body=body, type_ignores=[]), str(NOTEBOOK), "exec"),
             namespace)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield namespace
    torch.set_num_threads(previous_threads)


def test_canonical_definitions_are_unchanged():
    source = ast.parse(Path(core.__file__).read_text())
    original = {n.name: n for n in source.body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    cells = [n for n in ast.parse(NOTEBOOK.read_text()).body
             if isinstance(n, ast.FunctionDef)]
    copied = [n for n in cells[1].body
              if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert len(copied) == 16
    class NormalizeDocstrings(ast.NodeTransformer):
        def visit_Constant(self, node):
            # Nesting verbatim definitions in a marimo cell adds indentation
            # inside multiline docstrings, but changes no executable code.
            if isinstance(node.value, str) and "\n" in node.value:
                import inspect
                node.value = inspect.cleandoc(node.value)
            return node

    for definition in copied:
        assert ast.dump(NormalizeDocstrings().visit(definition)) == ast.dump(
            NormalizeDocstrings().visit(original[definition.name])), definition.name


def test_canonical_forward_parity(notebook):
    cfg = core.Config(worlds=1, use_reward_adaln=False)
    for name, kwargs in [("StatelessEncoder", {}), ("Strategizer", {"persistent": True}),
                         ("Actor", {"persistent": False}), ("Predictor", {})]:
        torch.manual_seed(19)
        reference = getattr(core, name)(cfg, **kwargs)
        torch.manual_seed(19)
        copied = notebook[name](cfg, **kwargs)
        for _ in range(3):
            z = torch.randn(1, cfg.latent_dim)
            strategy = torch.randn(1, cfg.strategy_dim)
            if name == "StatelessEncoder":
                args = (torch.randn(1, cfg.observation_dim),)
                a, b = reference.encode(*args), copied.encode(*args)
            elif name == "Strategizer":
                args = (z, torch.randn(1, 2*cfg.latent_dim))
                a, b = reference(*args)["strategy"], copied(*args)["strategy"]
            elif name == "Actor":
                args = (z, strategy, torch.zeros(1), torch.zeros(1))
                a, b = reference(*args, deterministic=True)["logits"], copied(*args, deterministic=True)["logits"]
            else:
                args = (z, strategy, torch.zeros(1), torch.tensor([2]))
                a, b = reference(*args), copied(*args)
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def make_agent(notebook, **kwargs):
    torch.manual_seed(12)
    return notebook["PredictiveAgent"](notebook["Settings"](**kwargs))


def test_balanced_cues_and_curriculum(notebook):
    cfg=notebook["Config"](worlds=1)
    env=notebook["BalancedMaze"](cfg,12)
    cues=[]
    for _ in range(20):
        cues.append(int(env.cue[0]))
        env.reset(notebook["np"].ones(1,bool))
    assert all(sorted(cues[i:i+2])==[0,1] for i in range(0,20,2))
    cfg=notebook["Config"](worlds=24)
    env=notebook["BalancedMaze"](cfg,12)
    assert (env.cue==0).sum()==12 and (env.cue==1).sum()==12
    # Subclass changes reset assignment only; canonical step retains both-cue mastery.
    assert notebook["BalancedMaze"].step is notebook["BatchedTMaze"].step


def test_prediction_survives_discard_and_full_bank(notebook):
    agent=make_agent(notebook,prediction_horizons=(2,),max_prediction_age=1)
    z=torch.zeros(1,agent.cfg.latent_dim)
    s=torch.zeros(1,agent.cfg.strategy_dim)
    agent.active[:]=True
    assert not agent.create_prediction(z,s,torch.tensor([2]),True)
    assert len(agent.queue)==1  # supervision still issued when capacity is full
    record=agent.queue[0]
    assert not record["prediction"].requires_grad
    assert all(j.grad_fn is None for j in record["jacobians"])
    agent.ages[:]=1
    agent.step_index=1
    agent.manage(z,False)  # hard expiry frees behavioral capacity
    assert not agent.active.any()
    observation=torch.zeros(1,agent.cfg.observation_dim)
    assert agent.supervise(observation,True)==[]
    assert agent.queue[0] is record
    before={n:p.detach().clone() for n,p in agent.named_parameters()}
    agent.step_index=2
    due=agent.supervise(observation,True)
    assert len(due)==1 and due[0]["target_step"]==2
    assert not agent.queue
    changed=[n for n,p in agent.named_parameters() if not torch.equal(before[n],p)]
    assert changed and all(n.startswith("predictor.") for n in changed)


def test_creation_jacobian_matches_predictor_readout(notebook):
    agent=make_agent(notebook,prediction_horizons=(2,))
    z=torch.randn(1,agent.cfg.latent_dim)
    agent.create_prediction(z,torch.randn(1,agent.cfg.strategy_dim),torch.tensor([2]),True)
    params=dict(agent.predictor.named_parameters())
    jac=dict(zip(params,agent.queue[0]["jacobians"]))["head.weight"]
    with torch.no_grad():
        feature=agent.predictor.norm(agent.predictor.core.last_output)
        delta=agent.predictor.head(feature)
        expected=torch.zeros_like(jac)
        for coordinate in range(agent.cfg.latent_dim):
            expected[coordinate,coordinate]=.5*(1-torch.tanh(delta[0,coordinate])**2)*feature[0]
    torch.testing.assert_close(jac,expected)
    # Capturing sixteen output derivatives must advance eligibility once.
    expected_epsilon=torch.zeros_like(agent.predictor_credit.epsilon_in)
    for tick,(value,_,_) in enumerate(agent.predictor.core.last_eligibility_records):
        carry=agent.predictor.core.decay*(agent.cfg.predictor_trace_decay if tick==0 else 1.)
        expected_epsilon=carry*expected_epsilon+value[:,None,:]
    torch.testing.assert_close(agent.predictor_credit.epsilon_in,expected_epsilon)


def test_discard_does_not_cancel_scheduled_prediction(notebook):
    agent=make_agent(notebook,prediction_horizons=(2,),management=True)
    z=torch.zeros(1,agent.cfg.latent_dim)
    agent.create_prediction(z,torch.zeros(1,agent.cfg.strategy_dim),torch.tensor([2]),True)
    with torch.no_grad():
        agent.manager.head.weight.zero_()
        agent.manager.head.bias.copy_(torch.tensor([-100.,-100.,100.]))
    agent.step_index=1;agent.ages[agent.active]+=1
    decision=agent.manage(z,False)
    assert decision['discard']==1 and not agent.active.any()
    assert len(agent.queue)==1
    agent.step_index=2
    assert len(agent.supervise(torch.zeros(1,agent.cfg.observation_dim),True))==1


def test_target_encoder_ema_and_frozen_targets(notebook):
    agent=make_agent(notebook,encoder_target_tau=.25)
    assert all(not p.requires_grad for p in agent.target_encoder.parameters())
    old=agent.target_encoder.latent_head.weight.detach().clone()
    with torch.no_grad():
        agent.encoder.latent_head.weight.add_(1.)
    agent.update_target_encoder()
    torch.testing.assert_close(agent.target_encoder.latent_head.weight,old+.25)


def test_strategy_score_has_correct_direction_and_noise_floor(notebook):
    agent=make_agent(notebook)
    clean=torch.zeros(1,agent.cfg.strategy_dim,requires_grad=True)
    sample,score,std,_=agent.sample_strategy(clean,True)
    signal=torch.autograd.grad(score.sum(),clean)[0]
    assert (signal*sample).sum()>0  # positive return reinforces the sampled direction
    with torch.no_grad():agent.strategy_logstd.fill_(-100.)
    _,_,std,_=agent.sample_strategy(clean,True)
    assert std.min()>=agent.settings.strategy_min_std
    first=agent.sample_strategy(clean,False)[0]
    second=agent.sample_strategy(clean,False)[0]
    torch.testing.assert_close(first,second,rtol=0,atol=0)


def test_shaping_telescopes_across_target_switches(notebook):
    agent=make_agent(notebook)
    phi=torch.zeros(())
    rewards=[];shaping=[]
    for t in range(8):
        z=torch.randn(1,agent.cfg.latent_dim)
        nxt=torch.randn_like(z)
        target=torch.randn_like(z)*10
        signal,phi,bonus,correction=agent.actor_reward(
            1. if t==7 else 0.,agent.cfg.gamma,1.,z,nxt,target,t%3!=0,t==7,phi)
        rewards.append(float(signal));shaping.append(float(bonus))
    assert sum(agent.cfg.gamma**t*b for t,b in enumerate(shaping))==pytest.approx(0.,abs=1e-4)
    assert sum(agent.cfg.gamma**t*r for t,r in enumerate(rewards))==pytest.approx(agent.cfg.gamma**7,abs=1e-4)


@pytest.mark.parametrize("management",[False,True])
@pytest.mark.parametrize("horizons",[(1,),(1,2,4,8,16)])
def test_episode_supervision_and_firewall(notebook,management,horizons):
    agent=make_agent(notebook,management=management,prediction_horizons=horizons)
    env=notebook["BalancedMaze"](agent.cfg,12)
    seen={"actor":[],"strategy":[],"manager":[]}
    original_actor=agent.actor_credit.apply
    original_strategy=agent.strategy_credit.apply
    def actor_apply(signal):
        seen["actor"].append(signal.clone())
        return original_actor(signal)
    def strategy_apply(signal,*args,**kwargs):
        seen["strategy"].append(signal.clone())
        return original_strategy(signal,*args,**kwargs)
    original_reward=agent.reward_step
    def reward_step(credit,advantage,*args,**kwargs):
        if credit is agent.management_credit:seen["manager"].append(advantage.clone())
        return original_reward(credit,advantage,*args,**kwargs)
    agent.actor_credit.apply=actor_apply
    agent.strategy_credit.apply=strategy_apply
    agent.reward_step=reward_step
    report,frames=notebook["episode"](agent,env,True)
    assert report["prediction_creation_count"]==report["steps"]
    assert report["prediction_validation_count"]+report["prediction_censored_count"]==report["steps"]
    if horizons==(1,):
        assert report["prediction_validation_count"]==report["steps"]
        assert report["prediction_censored_count"]==0
    for i,frame in enumerate(frames):
        assert float(seen["actor"][i])==pytest.approx(frame["actor_signal"])
        assert float(seen["strategy"][i])==pytest.approx(frame["external_td"])
        if management:
            torch.testing.assert_close(seen["manager"][i],torch.full((agent.settings.slots,),frame["external_td"]))
    assert all(torch.isfinite(p).all() for p in agent.parameters())
    assert not agent.queue and not agent.active.any()
    assert all(t.count_nonzero()==0 for t in agent.management_credit.traces)
    assert report["maximum_prediction_age"]<=agent.settings.max_prediction_age
    before={n:p.detach().clone() for n,p in agent.named_parameters()}
    with torch.no_grad():
        evaluation,_=notebook["episode"](agent,notebook["BalancedMaze"](agent.cfg,10012,False))
    assert all(torch.equal(before[n],p) for n,p in agent.named_parameters())


@pytest.mark.parametrize("kwargs",[
    {"stochastic_strategy":False},
    {"strategy_conditioning":False},
    {"use_target_encoder":False},
    {"encoder_eprop":False},
    {"prediction_shaping_beta":0.},
])
def test_required_ablations(notebook,kwargs):
    agent=make_agent(notebook,prediction_horizons=(1,),**kwargs)
    report,frames=notebook["episode"](agent,notebook["BalancedMaze"](agent.cfg,12),True)
    assert frames and report["prediction_validation_count"]==report["steps"]
    if kwargs.get("prediction_shaping_beta")==0:
        assert all(f["actor_signal"]==pytest.approx(f["reward"]) for f in frames)


def test_predecision_baseline_and_summary(notebook):
    agent=make_agent(notebook,prediction_horizons=(1,))
    z=torch.zeros(1,agent.cfg.latent_dim)
    before=agent.meta_state(z).clone()
    agent.sample_strategy(agent.memory,True)
    torch.testing.assert_close(before,agent.meta_state(z),rtol=0,atol=0)
    env=notebook["BalancedMaze"](agent.cfg,12)
    history=[]
    with torch.no_grad():
        for _ in range(2):
            r,_=notebook["episode"](agent,env,False);history.append(r)
    summary=notebook["summarize"](history)
    assert summary["episodes_left"]==summary["episodes_right"]==1
    assert summary["worst_cue_success"]==min(summary["success_left"],summary["success_right"])
    assert summary["accuracy_by_horizon"][0]["validations"]==sum(r["steps"] for r in history)
    assert summary["all_z_separation"] is not None


def test_training_plot_and_episode_stage_at_promotion(notebook, tmp_path):
    cell = next(cell for cell in ast.parse(NOTEBOOK.read_text()).body
                if isinstance(cell, ast.FunctionDef) and any(
                    isinstance(node, ast.FunctionDef) and node.name == "training_plot"
                    for node in cell.body))
    exec(compile(ast.Module(body=cell.body[:-1], type_ignores=[]), str(NOTEBOOK), "exec"), notebook)
    agent = make_agent(notebook)
    agent.cfg.curriculum_min_episodes_per_cue = 1
    agent.cfg.curriculum_success_threshold = 0.0
    env = notebook["BalancedMaze"](agent.cfg, 12)
    for group in env.curriculum_history:
        group.append(1.0)
    with torch.no_grad():
        first, _ = notebook["episode"](agent, env)
        second, _ = notebook["episode"](agent, env)
    assert first["curriculum_stage"] == 1
    assert first["next_curriculum_stage"] == 2
    assert second["curriculum_stage"] == 2
    first["prediction_validation_count"] = 0
    first["prediction_validation_loss"] = None
    for entry in first["horizon_accuracy"].values():
        entry.update(count=0, mse_sum=0)
    fig = notebook["training_plot"]([first, second], window=100)
    assert len(fig.axes) == 12
    assert notebook["np"].isnan(fig.axes[1].lines[0].get_ydata()[0])
    assert fig.axes[0].lines[0].get_ydata()[1] == second["success"]
    fig.savefig(tmp_path / "training.png")



def loss_entries(agent, value, count=1):
    return {h: dict(count=count, mse_sum=value*count) for h in agent.settings.prediction_horizons}


def detector_agent(notebook, **kwargs):
    return make_agent(notebook, adaptive_strategy_lr=True, prediction_change_window=8,
                      prediction_change_warmup=5, strategy_lr_hold_episodes=2,
                      strategy_lr_recovery_episodes=3, prediction_stable_episodes=2, **kwargs)


def test_change_detector_noise_trend_jump_and_rearm(notebook):
    agent = detector_agent(notebook)
    for i in range(15):
        row = agent.adapt_strategy_lr(loss_entries(agent, 1.-.01*i+(.003 if i%2 else -.003)))
        assert not row['prediction_change_triggered']
    row = agent.adapt_strategy_lr(loss_entries(agent, 1.5))
    assert row['prediction_change_triggered']
    assert agent.strategy_lr_scale == .1
    assert row['prediction_change_state'] == 'hold'
    for _ in range(30):
        row = agent.adapt_strategy_lr(loss_entries(agent, 1.5))
        assert not row['prediction_change_triggered']
    assert row['prediction_change_state'] == 'armed'
    assert agent.strategy_lr_scale == 1.
    assert agent.adapt_strategy_lr(loss_entries(agent, 2.))['prediction_change_triggered']


def test_change_detector_missing_evidence_and_episode_reset(notebook):
    agent = detector_agent(notebook)
    for _ in range(5): agent.adapt_strategy_lr(loss_entries(agent, 1.))
    agent.adapt_strategy_lr(loss_entries(agent, 2.))
    state = (agent.prediction_hold_remaining, agent.prediction_stable_count, agent.strategy_lr_scale)
    histories = {h: list(v) for h,v in agent.prediction_loss_history.items()}
    agent.reset_episode()
    row = agent.adapt_strategy_lr(loss_entries(agent, 0., count=0))
    assert row['prediction_change_score'] is None
    assert state == (agent.prediction_hold_remaining, agent.prediction_stable_count, agent.strategy_lr_scale)
    assert histories == agent.prediction_loss_history


def test_change_detector_horizon_mix_does_not_trigger(notebook):
    agent = detector_agent(notebook)
    for i in range(12):
        entries = {h:dict(count=(100 if (h+i)%2 else 1),
                    mse_sum=h*(100 if (h+i)%2 else 1)) for h in agent.settings.prediction_horizons}
        row = agent.adapt_strategy_lr(entries)
        assert not row['prediction_change_triggered']
    assert abs(row['prediction_change_score']) < 1e-8


def test_change_detector_scales_actual_adam_steps_and_scope(notebook):
    baseline = make_agent(notebook)
    slowed = detector_agent(notebook)
    untouched = [slowed.actor_credit.optimizer, slowed.management_credit.optimizer,
                 slowed.predictor_optimizer, slowed.critic_optimizer]
    rates = [[g['lr'] for g in opt.param_groups] for opt in untouched]
    for _ in range(5): slowed.adapt_strategy_lr(loss_entries(slowed, 1.))
    slowed.adapt_strategy_lr(loss_entries(slowed, 2.))
    for name in ('strategy_credit', 'encoder_credit'):
        normal_credit, slow_credit = getattr(baseline, name), getattr(slowed, name)
        for credit in (normal_credit, slow_credit):
            for trace in credit.reward_traces: trace.fill_(1.)
        _, normal_norm = normal_credit.apply(torch.ones(1))
        _, slow_norm = slow_credit.apply(torch.ones(1))
        assert slow_norm == pytest.approx(.1*normal_norm, rel=.01)
        assert all(torch.all(trace == 1.) for trace in slow_credit.reward_traces)
    assert slowed.std_credit.optimizer.param_groups[0]['lr'] == pytest.approx(slowed.cfg.strategy_eprop_lr*.1)
    assert rates == [[g['lr'] for g in opt.param_groups] for opt in untouched]


def test_change_detector_episode_integration_and_evaluation_readonly(notebook):
    agent = detector_agent(notebook)
    original = agent.adapt_strategy_lr
    inputs = []
    def record(entries):
        inputs.append(entries)
        return original(entries)
    agent.adapt_strategy_lr = record
    env = notebook['BalancedMaze'](agent.cfg, 12)
    report, frames = notebook['episode'](agent, env, train=True)
    assert inputs == [report['horizon_accuracy']]
    assert agent.prediction_episode_index == 1
    assert report['strategy_update_norm'] > 0
    assert all(f['strategy_lr_scale'] == 1. for f in frames)
    assert notebook['summarize']([report])['prediction_change_events'] == 0
    with torch.no_grad(): notebook['episode'](agent, env)
    assert agent.prediction_episode_index == 1


def test_change_detector_disabled_keeps_learning_rates(notebook):
    agent = make_agent(notebook)
    for _ in range(10): agent.adapt_strategy_lr(loss_entries(agent, 1.))
    row = agent.adapt_strategy_lr(loss_entries(agent, 2.))
    assert row['prediction_change_score'] > agent.settings.prediction_change_threshold
    assert not row['prediction_change_triggered']
    assert agent.strategy_lr_scale == 1.


def test_prediction_plot_loss_window_continues_across_stages(notebook):
    import copy
    agent = make_agent(notebook)
    with torch.no_grad():
        report, _ = notebook['episode'](agent, notebook['BalancedMaze'](agent.cfg, 12))
    a, b = copy.deepcopy(report), copy.deepcopy(report)
    a.update(curriculum_stage=1,prediction_validation_loss=1.,prediction_validation_count=1,critic_loss=1.)
    b.update(curriculum_stage=2,prediction_validation_loss=3.,prediction_validation_count=1,critic_loss=9.)
    fig = notebook['training_plot']([a,b],window=100)
    assert list(fig.axes[1].lines[0].get_ydata()) == [1.,2.]
    assert list(fig.axes[4].lines[0].get_ydata()) == [1.,5.]
    assert list(fig.axes[4].lines[1].get_ydata()) == [1.,9.]


def test_raw_prediction_loss_preserves_spikes_gaps_and_counts(notebook, tmp_path):
    history = [dict(curriculum_stage=stage,horizon_accuracy={
        1:dict(count=count,mse_sum=total),
        4:dict(count=2,mse_sum=2.)})
        for stage,count,total in [(1,2,2.),(1,0,0.),(2,1,9.),(2,3,3.)]]
    fig = notebook['prediction_loss_plot'](history,radius=1)
    assert len(fig.axes) == 4  # Overview and one transition, each with counts.
    actual = fig.axes[0].lines[0].get_ydata()
    notebook['np'].testing.assert_allclose(actual,[1.,float('nan'),9.,1.],equal_nan=True)
    assert list(fig.axes[1].lines[0].get_ydata()) == [2,0,1,3]
    assert list(fig.axes[2].lines[0].get_xdata()) == [2,3,4]
    assert fig.axes[2].lines[0].get_ydata()[1] == 9.
    fig.savefig(tmp_path / 'raw_prediction_loss.png')


def test_critic_trigger_ignores_prediction_spikes_and_detects_critic_shift(notebook):
    agent = detector_agent(notebook, slowdown_trigger_source='critic')
    for _ in range(5):
        assert not agent.adapt_strategy_lr(loss_entries(agent,1.),critic_loss=.1)['prediction_change_triggered']
    assert not agent.adapt_strategy_lr(loss_entries(agent,100.),critic_loss=.1)['prediction_change_triggered']
    row = agent.adapt_strategy_lr(loss_entries(agent,1.),critic_loss=.3)
    assert row['prediction_change_triggered']
    assert row['slowdown_trigger_source'] == 'critic'
    assert agent.strategy_lr_scale == .1
    assert row['prediction_change_details'][0]['loss'] == .3


def test_critic_trigger_episode_input_and_prediction_mode_isolation(notebook):
    agent = detector_agent(notebook, slowdown_trigger_source='critic')
    original = agent.adapt_strategy_lr
    inputs = []
    def record(entries, critic_loss=None):
        inputs.append(critic_loss)
        return original(entries,critic_loss)
    agent.adapt_strategy_lr = record
    report,frames = notebook['episode'](agent,notebook['BalancedMaze'](agent.cfg,12),train=True)
    assert inputs == [report['critic_loss']]
    assert inputs[0] == pytest.approx(sum(f['critic_loss'] for f in frames)/len(frames))
    assert notebook['summarize']([report])['slowdown_trigger_source'] == 'critic'
    prediction_agent = detector_agent(notebook)
    for _ in range(5): prediction_agent.adapt_strategy_lr(loss_entries(prediction_agent,1.),critic_loss=.1)
    assert not prediction_agent.adapt_strategy_lr(loss_entries(prediction_agent,1.),critic_loss=100.)['prediction_change_triggered']


def test_critic_spike_density_continuous_and_counts_episodes(notebook):
    history = [dict(critic_loss=loss,curriculum_stage=1 if i<2 else 2)
               for i,loss in enumerate([.1,.3,.5,.2,.1])]
    result = notebook['critic_spike_density'](history,threshold=.2,window=3)
    assert list(result['spikes']) == [False,True,True,False,False]
    assert result['counts'] == [0,1,2,2,1]
    assert result['fractions'] == pytest.approx([0.,.5,2/3,2/3,1/3])
    assert result['mean_excess'] == pytest.approx([0.,.05,.4/3,.4/3,.1])


def test_spike_density_times_magnitude(notebook):
    history = [dict(critic_loss=value) for value in [.1,.5,.1,.5]]
    result = notebook['critic_spike_density'](history,threshold=.1,window=4)
    assert result['spike_magnitude'] == pytest.approx([0.,.4,.4,.4])
    assert result['density_times_magnitude'] == pytest.approx([0.,.2,.4/3,.2])
    assert result['density_times_magnitude'] == pytest.approx(result['mean_excess'])


def test_balanced_cues_independent_of_sampled_hallway_lengths(notebook):
    cfg=notebook['Config'](worlds=8000)
    env=notebook['BalancedMaze'](cfg,12)
    assert (env.cue==0).sum()==4000
    for h in env.curriculum_length_distributions[0]:
        left=(env.hallway_length[env.cue==0]==h).mean()
        right=(env.hallway_length[env.cue==1]==h).mean()
        assert abs(left-right)<.04
    # Cue balancing must not consume or perturb the independent length stream.
    plain=notebook['BatchedTMaze'](cfg,12)
    notebook['np'].testing.assert_array_equal(env.hallway_length,plain.hallway_length)


def test_notebook_records_completed_length_and_limits(notebook):
    agent=make_agent(notebook)
    env=notebook['BalancedMaze'](agent.cfg,12)
    length=int(env.hallway_length[0]);limit=int(env.episode_time_limit[0])
    with torch.no_grad(): report,_=notebook['episode'](agent,env)
    assert report['hallway_length']==length
    assert report['episode_time_limit']==limit
    assert report['steps']<=limit
    summary=notebook['summarize']([report])
    assert summary['mean_hallway_length']==length
    assert sum(r['samples'] for r in summary['success_by_hallway_length'])==1
    assert summary['worst_length_success'] is None  # Insufficient sample count.
