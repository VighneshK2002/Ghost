"""Numerical baseline credit audit, not a full BPTT equivalence claim."""
import argparse
from dataclasses import asdict, replace
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import json
from pathlib import Path

import torch

from experiments.delayed_cue_tmaze.train import parser as training_parser, config_from_args
from ghost.ghost_terminal_core import System, BatchedTMaze, RewardEprop, RecurrentStrategyEprop, run_condition


def extended_checks(cfg, seed, check):
    # Isolate the production memory recursion from approximate SNN Jacobians.
    system = System(cfg, "separated", torch.device("cpu"), seed)
    p = torch.nn.Parameter(torch.tensor(.3))
    learner = RecurrentStrategyEprop(system.strategizer, [p], 2, .9, 1e-4, True)
    def jacobian(output):
        return [torch.stack([torch.stack([
            torch.autograd.grad(output[w, k], p, retain_graph=True)[0]
            for k in range(output.shape[1])]) for w in range(2)])]
    learner._current_output_jacobians = jacobian
    reference = torch.zeros(2, cfg.strategy_dim)
    score_reference = torch.zeros(2)
    for step in range(4):
        previous = reference.detach().requires_grad_(True)
        proposal = torch.tanh(p+.2*previous+step*.1)
        gate = torch.sigmoid(p-.1*previous)
        readout = (1-gate)*previous+gate*proposal
        logp = -torch.nn.functional.softplus(-readout.sum(-1))
        learner.accumulate(proposal, gate, previous, readout, logp, .95, True)
        ref_proposal = torch.tanh(p+.2*reference+step*.1)
        ref_gate = torch.sigmoid(p-.1*reference)
        reference = (1-ref_gate)*reference+ref_gate*ref_proposal
        expected = torch.stack([torch.stack([
            torch.autograd.grad(reference[w, k], p, retain_graph=True)[0]
            for k in range(cfg.strategy_dim)]) for w in range(2)])
        check(f"unrolled memory Jacobian step {step}", (expected-learner.memory_jacobians[0]).abs().max().detach())
        ref_logp = -torch.nn.functional.softplus(-reference.sum(-1))
        score_reference = .9*score_reference+torch.stack([
            torch.autograd.grad(ref_logp[w], p, retain_graph=True)[0].detach() for w in range(2)])
        check(f"delayed strategy score step {step}", (score_reference-learner.reward_traces[0]).abs().max())

    # Observe actual production TD calls: one-step timeouts force automatic
    # environment resets and must still produce reward-only terminal targets.
    def integration(mode, terminal="timeout"):
        records, forwards, feedbacks = [], [], []
        original_forward = type(system.strategizer).forward
        original_apply = RewardEprop.apply
        original_step = BatchedTMaze.step
        transition = {}
        def forward(instance, latent, feedback, *args, **kwargs):
            result = original_forward(instance, latent, feedback, *args, **kwargs)
            forwards.append(result["desirability"].detach().clone())
            feedbacks.append(feedback.detach().clone())
            return result
        def step(instance, actions):
            if terminal == "timeout":
                # Force the final step explicitly: production timeouts now
                # enforce route feasibility for each sampled hallway length.
                instance.age[:] = instance.episode_time_limit-1
                actions = actions*0
            else:
                instance.x[:] = 2
                instance.y[:] = 1
                instance.direction[:] = 3
                instance.cue[:] = 0 if terminal == "success" else 1
                actions = actions*0+2
            result = original_step(instance, actions)
            if terminal != "timeout":
                check(f"{terminal} terminal exercised", int(not bool(result[3 if terminal == "success" else 4].all())))
            transition["reward"] = torch.tensor(result[1])
            transition["done"] = torch.tensor(result[2])
            return result
        def apply(instance, td):
            check(f"terminal mask exercised {mode}", int(not bool(transition["done"].all())))
            expected = (transition["reward"]+cfg.gamma*(~transition["done"]).float()*forwards[-1]-forwards[-2]).clamp(-cfg.td_clip, cfg.td_clip)
            records.append((td.detach().clone(), expected))
            return original_apply(instance, td)
        config = replace(cfg, transitions=8, report_every=1000,
                         training_predictor_feedback=mode)
        with patch.object(type(system.strategizer), "forward", forward), patch.object(RewardEprop, "apply", apply), patch.object(BatchedTMaze, "step", step), redirect_stdout(io.StringIO()):
            result = run_condition(config, "separated", seed, torch.device("cpu"))
        for i, (actual, expected) in enumerate(records):
            check(f"production terminal TD {mode}/{i}", (actual-expected).abs().max())
        return result["system"], feedbacks
    normal, normal_feedback = integration("normal")
    blocked, blocked_feedback = integration("zero")
    integration("normal", "success")
    integration("normal", "wrong")
    check("blocked decision and bootstrap feedback", max(float(x.abs().max()) for x in blocked_feedback))
    check("normal feedback path exercised", int(not any(bool(x.abs().max()>0) for x in normal_feedback)))
    # First predictor update is compared in a separate one-decision run: later
    # trajectories legitimately diverge when feedback changes value estimates.
    states = []
    for mode in ("normal", "zero"):
        with redirect_stdout(io.StringIO()):
            result = run_condition(replace(cfg, transitions=2, report_every=1000,
                training_predictor_feedback=mode), "separated", seed, torch.device("cpu"))
        states.append(result["system"].predictor.state_dict())
    check("blocking preserves first predictor update", max(float((states[0][k]-states[1][k]).abs().max()) for k in states[0]))


def audit(seed=12):
    cfg = config_from_args(training_parser().parse_args([
        "--preset", "historical-reward-eprop", "--seed", str(seed), "--worlds", "2",
    ]))
    checks = []
    def check(name, error, tolerance=1e-5):
        checks.append(dict(name=name, max_error=float(error), tolerance=tolerance,
                           passed=bool(error <= tolerance)))
    check("baseline flags", int(cfg.use_strategic_prediction_timer or cfg.adaptive_exploration))
    # Independent two-step score reference, signed TD and selective reset.
    p = torch.nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float64))
    learner = RewardEprop([p], 2, .85, 1e-4)
    reference = torch.zeros(2, 2, dtype=torch.float64)
    for actions in ([0, 1], [1, 1]):
        probabilities = p.softmax(0)
        scores = p.log_softmax(0)[actions]
        reference = .85*reference + torch.eye(2, dtype=p.dtype)[actions]-probabilities.detach()
        learner.accumulate(scores)
        check("two-step analytic categorical eligibility", (learner.traces[0]-reference).abs().max())
    td = torch.tensor([1., -1.], dtype=p.dtype)
    expected = (reference*td[:, None]).mean(0)
    learner.apply(td)
    check("signed TD direction", (p.grad-expected).abs().max())
    learner.reset(torch.tensor([True, False]))
    check("selective trace reset", learner.traces[0][0].abs().max())
    check("other world trace preserved", (learner.traces[0][1]-reference[1]).abs().max())

    for sign in (1., -1.):
        system = System(cfg, "separated", torch.device("cpu"), seed)
        env = BatchedTMaze(cfg, seed*1000+17)
        latent = system.encoder(torch.tensor(env.observation()))[0].detach()
        strategy, actor, readout, desire, variance = system.strategy_and_action(latent)
        params = system.actor_eprop.parameters
        direct = torch.autograd.grad(actor["logp"].sum(), params, retain_graph=True, allow_unused=True)
        system.actor_eprop.accumulate(actor["logp"])
        for index, (trace, grad) in enumerate(zip(system.actor_eprop.traces, direct)):
            if grad is not None:
                check(f"actor autograd reference {sign}/{index}", (trace.sum(0)-grad).abs().max())
        actions = actor["action"].detach()
        before = actor["logp"].detach().mean()
        system.actor_eprop.apply(torch.full((2,), sign))
        after = system.actor(latent, readout.detach(), desire.detach(), variance.detach(),
                             deterministic=True, exploration=cfg.exploration_rate)
        # Actor logits are policy logits; reconstruct the fixed exploration mix.
        probs = after["logits"].softmax(-1)
        probs = (1-cfg.exploration_rate)*probs+cfg.exploration_rate/probs.shape[-1]
        delta = probs.gather(1, actions[:, None]).log().mean()-before
        check(f"actor sampled-action probability TD sign {sign}", max(0., -sign*float(delta.detach())), 1e-7)

    system = System(cfg, "separated", torch.device("cpu"), seed)
    env = BatchedTMaze(cfg, seed*1000+17)
    latent = system.encoder(torch.tensor(env.observation()))[0].detach()
    strategy, actor, readout, _, _ = system.strategy_and_action(latent)
    system.strategy_eprop.accumulate(strategy["proposal"], strategy["gate"],
        strategy["previous_strategy"], readout, actor["logp"], cfg.strategy_retention,
        learned_gate=cfg.learned_strategy_memory)
    # Output-head derivatives are exact on a fresh state; recurrent e-prop is approximate.
    head_ids = {id(p) for p in system.strategizer.strategy_head.parameters()}
    for index, (p, trace) in enumerate(zip(system.strategy_eprop.parameters, system.strategy_eprop.reward_traces)):
        if id(p) in head_ids:
            direct = torch.autograd.grad(actor["logp"].sum(), p, retain_graph=True)[0]
            check(f"strategy head first-step chain rule {index}", (trace.sum(0)-direct).abs().max())
    system.reset(torch.ones(2, dtype=torch.bool))
    for index, trace in enumerate(system.strategy_eprop.reward_traces):
        check(f"strategy episode trace reset {index}", trace.abs().max())
    check("feedback reset", system.feedback.abs().max())
    check("strategy memory reset", system.strategy_memory.abs().max())
    extended_checks(cfg, seed, check)
    return dict(config=asdict(cfg), checks=checks, passed=all(c["passed"] for c in checks),
                limitations=["No timer or adaptive/noisy strategy in this baseline.",
                    "Recurrent e-prop is not validated against multi-step BPTT here.",
                    "Memory recursion checked with exact synthetic local Jacobians; full SNN temporal derivatives remain approximate and unaudited.",
                    "Terminal integration forces timeout/success/wrong-goal transitions; it is not a natural-policy performance test.",
                    "Passing these checks does not establish learning robustness."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for check in report["checks"]:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['max_error']:.3g}")
    print("Limitations:", *report["limitations"], sep="\n  ")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
