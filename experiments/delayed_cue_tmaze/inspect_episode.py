"""Record module outputs from an isolated, inference-only evaluation."""
import copy
from unittest.mock import patch
import numpy as np
import torch
from ghost.checkpoint import load_system
from ghost.ghost_terminal_core import BatchedTMaze, evaluate


def record_episode(path, seed=12, rollout_seed=123):
    with torch.random.fork_rng(devices=[]):
        system, cfg, _ = load_system(path, "separated", seed, torch.device("cpu"), evaluation_episodes=1)
        torch.manual_seed(rollout_seed)
        latest, frames, handles = {}, [], []
        def array(t):
            return t[0].detach().cpu().numpy().copy()
        def encoder(module, args, out):
            latest["encoder"] = array(out[0])
        def strategy(module, args, out):
            latest["feedback"] = array(args[1])
            latest["strategy"] = array(out["strategy"])
            latest["gate"] = array(out["gate"])
            latest["value"] = float(out["desirability"][0])
            latest["std"] = float(out["outcome_logvar"][0].exp().sqrt())
        def actor(module, args, out):
            latest["actor"] = array(out["logits"].softmax(-1))
        def predictor(module, args, out):
            if frames and (not finished[0] or "predictor" not in frames[-1]):
                frames[-1]["predictor"] = array(out)
        for module, hook in ((system.encoder,encoder),(system.strategizer,strategy),
                             (system.actor,actor),(system.predictor,predictor)):
            handles.append(module.register_forward_hook(hook))
        original = BatchedTMaze.step
        finished = [False]
        class EpisodeComplete(Exception):
            pass
        def step(env, actions):
            if finished[0]:
                raise EpisodeComplete
            recording = not finished[0]
            if recording:
                frame = copy.deepcopy(latest)
                frame.update(x=int(env.x[0]), y=int(env.y[0]), direction=int(env.direction[0]),
                    cue=int(env.cue[0]), age=int(env.age[0]), action=int(actions[0]),
                    observation=env.observation()[0].copy(), valid=sorted(env.valid),
                    width=cfg.maze_width, height=cfg.maze_height)
                frames.append(frame)
            out = original(env, actions)
            if recording:
                frames[-1].update(reward=float(out[1][0]), done=bool(out[2][0]), success=bool(out[3][0]))
                # Finish after predictor hook has processed this transition.
                if out[2][0]:
                    finished[0] = True
            return out
        # Ensure world zero completes even if another world terminates sooner.
        from dataclasses import replace
        try:
            with patch.object(BatchedTMaze, "step", step):
                try:
                    evaluate(system, replace(cfg, evaluation_episodes=cfg.worlds*cfg.episode_limit+1), rollout_seed, "normal")
                except EpisodeComplete:
                    pass
        finally:
            for handle in handles:
                handle.remove()
    return frames


def draw_frame(frames, index):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    frame = frames[index]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    axes = axes.flatten()
    valid = np.zeros((frame["height"], frame["width"]))
    for x,y in frame["valid"]:
        valid[y,x] = 1
    axes[0].imshow(valid, cmap="Greys_r", vmin=0, vmax=1)
    axes[0].plot(frame["x"],frame["y"],"ro")
    axes[0].set_title(f"Observation / maze | cue {'L' if frame['cue']==0 else 'R'} (oracle label)")
    axes[0].set_xlabel(f"orientation={frame['direction']} | cue age={frame['age']}")
    for ax,key,title in zip(axes[1:4],("encoder","strategy","actor"),
            ("Encoder latent z(t)","Strategizer readout s(t)","Actor probabilities")):
        values=frame[key]
        ax.bar(range(len(values)),values)
        ax.set_title(title)
        ax.set_ylim((0,1) if key=="actor" else (-1.1,1.1))
    axes[3].set_xticks([0,1,2],["turn left","turn right","forward"])
    axes[4].bar(range(len(frame["feedback"])),frame["feedback"])
    axes[4].set_title("Incoming predictor feedback f(t−1)")
    axes[5].bar(range(len(frame["gate"])),frame["gate"])
    axes[5].set_ylim(0,1)
    axes[5].set_title("Strategy memory-update gate")
    if "predictor" in frame:
        axes[6].bar(range(len(frame["predictor"])),frame["predictor"])
    axes[6].set_title("Predictor forecast → feedback at next decision")
    axes[6].set_ylim(-1.6,1.6)
    axes[7].plot([f["value"] for f in frames],label="expected return")
    axes[7].plot([f["std"] for f in frames],label="predicted std")
    axes[7].axvline(index,color="red")
    axes[7].legend()
    axes[7].set_title("Outcome head across episode")
    for a,b in ((0,1),(1,2),(2,3),(2,6),(4,2),(2,7)):
        fig.add_artist(ConnectionPatch((1,.5),(0,.5),"axes fraction","axes fraction",
            axesA=axes[a],axesB=axes[b],arrowstyle="->",alpha=.3))
    fig.suptitle(f"Step {index} | chosen action {frame['action']} | reward {frame['reward']:+.1f} | "
                 f"episode success={frames[-1]['success']} — arrows show selected dependencies, not gradients")
    return fig
