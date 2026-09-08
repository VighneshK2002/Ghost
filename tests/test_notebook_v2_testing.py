"""Regression checks for the v2 live evaluator, without running training UI cells."""
import ast, copy, io
from pathlib import Path
import numpy as np
import torch


def test_live_checkpoint_and_cue_replay():
    previous_threads = torch.get_num_threads()
    try:
        path=Path(__file__).resolve().parents[1] / 'ghost_notebook_v2.py'
        cells=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)]
        ns={}
        def run(cell):
            exec(compile(ast.Module(body=[n for n in cell.body if not isinstance(n,ast.Return)],type_ignores=[]),str(path),'exec'),ns)
        for cell in cells[:4]: run(cell)
        run(next(c for c in cells if any(isinstance(n,ast.FunctionDef) and n.name=='load_testing_model' for n in c.body)))
        torch.set_num_threads(1)
        a=ns['PredictiveAgent'](ns['Settings']())
        # Adaptive controls use completed episodes, respect bounds, and bypass eval.
        adaptive=ns['PredictiveAgent'](ns['Settings'](
            adaptive_strategy_exploration=True,adaptive_actor_exploration=True,
            exploration_window=4,exploration_warmup=2))
        adaptive.prepare_exploration(True)
        assert adaptive.actor_temperature==adaptive.strategy_exploration_scale==1.
        adaptive.record_exploration_success(False)
        adaptive.prepare_exploration(True)
        assert adaptive.actor_temperature==1.
        adaptive.record_exploration_success(False)
        adaptive.prepare_exploration(True)
        assert adaptive.actor_temperature==adaptive.strategy_exploration_scale==2.
        clean=torch.zeros(1,adaptive.cfg.strategy_dim)
        _,_,base_std,_=adaptive.sample_strategy(clean,False)
        _,_,scaled_std,_=adaptive.sample_strategy(clean,True)
        torch.testing.assert_close(scaled_std,2*base_std)
        z=torch.zeros(1,adaptive.cfg.latent_dim)
        context=torch.zeros(1,2*adaptive.cfg.latent_dim+3)
        cold=copy.deepcopy(adaptive.actor)(z,clean,context,temperature=1.)
        hot=copy.deepcopy(adaptive.actor)(z,clean,context,temperature=2.)
        torch.testing.assert_close(hot['logits'],cold['logits']/2)
        expected=torch.distributions.Categorical(logits=hot['logits'])
        torch.testing.assert_close(hot['logp'],expected.log_prob(hot['action']))
        assert hot['entropy'].item() >= cold['entropy'].item()-1e-6
        adaptive.prepare_exploration(False)
        assert adaptive.actor_temperature==adaptive.strategy_exploration_scale==1.
        for _ in range(4): adaptive.record_exploration_success(True)
        adaptive.prepare_exploration(True)
        assert len(adaptive.exploration_success_history)==4
        assert adaptive.actor_temperature==adaptive.strategy_exploration_scale==1.
        adaptive.settings.adaptive_actor_exploration=False
        adaptive.exploration_success_history=[0.,0.]
        adaptive.prepare_exploration(True)
        assert adaptive.actor_temperature==1. and adaptive.strategy_exploration_scale==2.
        # Bonus uses prior per-cue rates, preserves raw outcomes, and is training-only.
        a.settings.cue_success_bonus=True
        a.settings.cue_bonus_scale=.2
        a.settings.cue_bonus_window=4
        a.settings.cue_bonus_min_samples=2
        for cue,success in [(0,False),(1,True),(0,False)]:
            a.record_cue_success(cue,success)
        assert a.success_bonus(0,True,True)==0.  # Insufficient right-cue evidence.
        a.record_cue_success(1,True)
        assert a.success_bonus(0,True,True)==.2
        assert a.success_bonus(1,True,True)==0.
        assert a.success_bonus(0,False,True)==0.
        assert a.success_bonus(0,True,False)==0.
        a.reset_episode()
        assert a.success_bonus(0,True,True)==.2
        a.record_cue_success(0,True)
        assert len(a.cue_success_history)==4
        assert a.success_bonus(0,True,True)==.1
        a.record_cue_success(1,False)
        assert a.success_bonus(0,True,True)==0.  # Equal rates.
        a.settings.cue_success_bonus=False
        assert a.success_bonus(0,True,True)==0.
        a=ns['PredictiveAgent'](ns['Settings']())
        payload=dict(architecture='ghost_external_predictive_control_v6',config=ns['asdict'](a.settings),ghost_config=ns['asdict'](a.cfg),weights=a.state_dict(),reliability={h:.25 for h in a.settings.prediction_horizons})
        buf=io.BytesIO();torch.save(payload,buf);buf.seek(0)
        a=ns['load_testing_model'](torch.load(buf,weights_only=True))
        weights={k:v.clone() for k,v in a.state_dict().items()}
        def replay(cue):
            agent=copy.deepcopy(a)
            env=ns['BalancedMaze'](agent.cfg,10012,curriculum=False,balanced=False);env.cue[0]=cue
            with torch.no_grad(): return ns['episode'](agent,env,False)
        l,lf=replay(0);r,rf=replay(1);_,again=replay(0)
        assert not np.array_equal(lf[0]['observation'],rf[0]['observation'])
        assert not np.array_equal(lf[0]['z'],rf[0]['z'])
        assert all(np.array_equal(x['actor'],y['actor']) for x,y in zip(lf,again))
        assert all(torch.equal(v,a.state_dict()[k]) for k,v in weights.items())
        assert all(np.isclose(f['actor'].sum(),1) for f in lf+rf)
        assert not lf[0]['managed_active'].any()
        assert lf[0]['active'].any()
        assert np.array_equal(lf[1]['bank_before'],lf[0]['predictions'])
        assert a.reliability[1]==.25
        buf=io.BytesIO(); torch.save(dict(payload, history=[l], evaluation=r, summary=ns['summarize']([l,r])),buf); buf.seek(0)
        ns['load_testing_model'](torch.load(buf, weights_only=True))
        # Reproduce the historical module construction order independently, so
        # matching toggle paths cannot hide a changed baseline RNG sequence.
        torch.manual_seed(12)
        reference={}
        cfg=a.cfg
        reference['encoder']=ns['StatelessEncoder'](cfg)
        reference['target_encoder']=copy.deepcopy(reference['encoder'])
        reference['strategizer']=ns['Strategizer'](cfg,persistent=True)
        reference['actor']=ns['PredictiveActor'](cfg)
        predictor_cfg=copy.deepcopy(cfg)
        predictor_cfg.use_strategic_prediction_timer=True
        predictor_cfg.prediction_timer_durations=a.settings.prediction_horizons
        reference['predictor']=ns['Predictor'](predictor_cfg)
        reference['manager']=ns['PredictionManager'](cfg)
        reference['critic']=ns['ExternalValueCritic'](cfg)
        historical_rng=torch.get_rng_state().clone()
        # Permanent removal retains historical weights and RNG, and repeatable training.
        torch.manual_seed(12)
        present=ns['PredictiveAgent'](ns['Settings']())
        assert torch.equal(torch.get_rng_state(),historical_rng)
        for name,module in reference.items():
            if name in ('manager', 'target_encoder'):
                continue
            for key,value in module.state_dict().items():
                assert torch.equal(value,present.state_dict()[name+'.'+key]), (name,key)
        torch.manual_seed(12)
        absent=ns['PredictiveAgent'](ns['Settings']())
        assert torch.equal(torch.get_rng_state(),historical_rng)
        assert not hasattr(absent, 'manager') and not hasattr(absent, 'management_credit')
        assert not hasattr(absent, 'target_encoder')
        assert not any(k.startswith('manager.') for k in absent.state_dict())
        for k,v in absent.state_dict().items():
            assert torch.equal(v,present.state_dict()[k]), k
        environments=[ns['BalancedMaze'](agent.cfg,12) for agent in (present,absent)]
        for step in range(2):
            torch.manual_seed(100+step)
            _,pf=ns['episode'](present,environments[0],train=True)
            torch.manual_seed(100+step)
            _,af=ns['episode'](absent,environments[1],train=True)
            assert len(pf)==len(af)
            for left,right in zip(pf,af):
                for key in ('actor','strategy','predictions','context','used'):
                    np.testing.assert_array_equal(left[key],right[key])
                assert 'management' not in right['spike_rates']
            for k,v in absent.state_dict().items():
                assert torch.equal(v,present.state_dict()[k]), k
        removed_payload=dict(payload,config=ns['asdict'](absent.settings),weights=absent.state_dict())
        restored=ns['load_testing_model'](removed_payload)
        assert not hasattr(restored, 'manager')
        legacy=dict(payload,config=dict(payload['config']))
        legacy['config'].update(management=False, remove_manager=False, manager_entropy_weight=.001)
        legacy['weights']=dict(payload['weights'])
        legacy['weights'].update({'manager.'+k:v for k,v in reference['manager'].state_dict().items()})
        assert not hasattr(ns['load_testing_model'](legacy), 'manager')
        legacy['config']['management']=True
        import pytest
        with pytest.raises(ValueError, match='learned management'):
            ns['load_testing_model'](legacy)
    finally:
        torch.set_num_threads(previous_threads)
