import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import contextlib
    import io
    from pathlib import Path

    import anywidget
    import marimo as mo
    import numpy as np
    import torch
    import traitlets

    from ghost_vision_core import (
        BatchedTMaze,
        CONDITIONS,
        Config,
        choose_device,
        evaluate,
        run_condition,
        save,
    )

    return (
        BatchedTMaze,
        CONDITIONS,
        Config,
        Path,
        anywidget,
        choose_device,
        contextlib,
        evaluate,
        io,
        mo,
        np,
        run_condition,
        save,
        torch,
        traitlets,
    )


@app.cell
def _(BatchedTMaze, Config, np):
    import matplotlib.pyplot as plt

    preview_cfg = Config(
        worlds=4,
        visual_observations=True,
        image_size=40,
        encoder_learning_mode="reward_eprop",
        use_reward_adaln=False,
    )

    preview_env = BatchedTMaze(
        preview_cfg,
        seed=11,
        curriculum=True,
    )

    frames = []
    ages = []

    # Observation at age 0: cue should be visible.
    frames.append(preview_env.observation()[0].copy())
    ages.append(int(preview_env.age[0]))

    # Turn left twice. This advances time without moving the agent.
    for _ in range(preview_cfg.cue_steps):
        preview_env.step(
            np.zeros(preview_cfg.worlds, dtype=np.int64)
        )
        frames.append(preview_env.observation()[0].copy())
        ages.append(int(preview_env.age[0]))

    fig, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 4))

    for axis, frame, age in zip(axes, frames, ages):
        image = np.transpose(frame, (1, 2, 0))
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(
            f"age={age}, cue={int(preview_env.cue[0])}"
        )
        axis.axis("off")

    fig.suptitle(
        f"Rendered observations: shape={frames[0].shape}, "
        f"range=[{frames[0].min():.2f}, {frames[0].max():.2f}]"
    )
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stateful strategizer · delayed-cue T-maze

    Configure, train, and compare the separated strategizer/actor architecture
    with its causal controls. Values are committed only when **Train and
    evaluate** is clicked, so changing a widget never starts an expensive run.
    """)
    return


@app.cell(hide_code=True)
def _(anywidget, mo, traitlets):
    class TrainingDashboard(anywidget.AnyWidget):
        _esm = r"""
        const COLORS = [
          "#7c3aed", "#06b6d4", "#f97316", "#22c55e",
          "#ec4899", "#eab308", "#3b82f6", "#ef4444"
        ];

        function render({ model, el }) {
          el.classList.add("ghost-dashboard");
          el.innerHTML = `
            <div class="dash-head">
              <div>
                <div class="eyebrow">LIVE TRAINING TELEMETRY</div>
                <div class="dash-title">Causal strategy monitor</div>
              </div>
              <div class="run-state">Waiting for a run</div>
            </div>
            <div class="progress-shell"><div class="progress-fill"></div></div>
            <div class="stat-grid"></div>
            <div class="legend"></div>
            <div class="chart-grid">
              <section><h3>Terminal outcomes</h3><canvas data-chart="outcomes"></canvas></section>
              <section><h3>Reward signal</h3><canvas data-chart="reward"></canvas></section>
              <section><h3>Learning diagnostics</h3><canvas data-chart="learning"></canvas></section>
              <section><h3>Strategy intervention sensitivity</h3><canvas data-chart="strategy"></canvas></section>
            </div>`;

          const stateNode = el.querySelector(".run-state");
          const progressNode = el.querySelector(".progress-fill");
          const statsNode = el.querySelector(".stat-grid");
          const legendNode = el.querySelector(".legend");
          const canvases = Object.fromEntries(
            [...el.querySelectorAll("canvas")].map(
              canvas => [canvas.dataset.chart, canvas])
          );

          const seriesKey = row => `${row.condition} · seed ${row.seed}`;
          const grouped = rows => {
            const groups = new Map();
            rows.forEach(row => {
              const key = seriesKey(row);
              if (!groups.has(key)) groups.set(key, []);
              groups.get(key).push(row);
            });
            return groups;
          };
          const fmt = (value, digits=3) =>
            Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : "—";

          function drawChart(canvas, rows, lines, yDomain=null) {
            const rect = canvas.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const width = Math.max(rect.width, 320);
            const height = 210;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            const ctx = canvas.getContext("2d");
            ctx.scale(dpr, dpr);
            ctx.clearRect(0, 0, width, height);

            const pad = { left: 48, right: 14, top: 16, bottom: 30 };
            const plotW = width - pad.left - pad.right;
            const plotH = height - pad.top - pad.bottom;
            const groups = grouped(rows);
            const values = rows.flatMap(row =>
              lines.map(line => Number(row[line.key])).filter(Number.isFinite));
            let minY = yDomain ? yDomain[0] : Math.min(...values, 0);
            let maxY = yDomain ? yDomain[1] : Math.max(...values, 1e-6);
            if (minY === maxY) { minY -= 0.5; maxY += 0.5; }
            const maxX = Math.max(...rows.map(row => row.transitions), 1);
            const x = value => pad.left + (value / maxX) * plotW;
            const y = value => pad.top + (1 - (value-minY)/(maxY-minY)) * plotH;

            ctx.strokeStyle = "rgba(148,163,184,.22)";
            ctx.fillStyle = "#94a3b8";
            ctx.font = "11px ui-monospace, SFMono-Regular, monospace";
            ctx.lineWidth = 1;
            for (let tick = 0; tick <= 4; tick++) {
              const yy = pad.top + tick * plotH / 4;
              const value = maxY - tick * (maxY-minY) / 4;
              ctx.beginPath(); ctx.moveTo(pad.left, yy);
              ctx.lineTo(width-pad.right, yy); ctx.stroke();
              ctx.fillText(fmt(value, 2), 4, yy + 4);
            }
            ctx.fillText("0", pad.left-3, height-8);
            ctx.fillText(`${Math.round(maxX).toLocaleString()} transitions`,
              Math.max(pad.left, width-145), height-8);

            [...groups.entries()].forEach(([key, points], groupIndex) => {
              lines.forEach((line, lineIndex) => {
                const color = COLORS[groupIndex % COLORS.length];
                ctx.strokeStyle = color;
                ctx.globalAlpha = line.alpha ?? (lineIndex ? 0.48 : 1);
                ctx.setLineDash(line.dash || []);
                ctx.lineWidth = line.width || 2;
                ctx.beginPath();
                points.forEach((point, index) => {
                  const xx = x(point.transitions);
                  const yy = y(Number(point[line.key]));
                  index ? ctx.lineTo(xx, yy) : ctx.moveTo(xx, yy);
                });
                ctx.stroke();
              });
            });
            ctx.globalAlpha = 1;
            ctx.setLineDash([]);
          }

          function update() {
            const rows = model.get("records") || [];
            const status = model.get("status") || {};
            const latest = rows.at(-1);
            const fraction = Number(status.fraction || 0);
            progressNode.style.width = `${Math.max(0, Math.min(1, fraction))*100}%`;
            stateNode.textContent = status.running
              ? `${status.condition} · seed ${status.seed} · ${Math.round(fraction*100)}%`
              : (rows.length ? "Run complete" : "Waiting for a run");
            stateNode.classList.toggle("active", Boolean(status.running));

            const cards = [
              ["Transitions", latest?.transitions?.toLocaleString() ?? "—"],
              ["Episodes", latest?.episodes?.toLocaleString() ?? "—"],
              ["Window success", latest ? `${fmt(100*latest.window_success, 1)}%` : "—"],
              ["TD magnitude", latest ? fmt(latest.td_abs) : "—"],
              ["Actor→encoder step", latest ? fmt(latest.encoder_step, 5) : "—"],
              ["Reward latent shift", latest ? fmt(latest.reward_latent_shift, 5) : "—"],
              ["Joy prediction MSE", latest ? fmt(latest.joy_prediction_mse, 4) : "—"],
              ["Goal events/window", latest ? Math.round(latest.joy_event_count).toLocaleString() : "—"],
              ["Predictor eligibility", latest ? fmt(latest.predictor_eligibility, 2) : "—"],
              ["Predictor temporal grad", latest ? fmt(latest.predictor_eprop_gradient, 3) : "—"],
              ["Critic→encoder step", latest ? fmt(latest.critic_encoder_step, 5) : "—"],
              ["Latent critic MAE", latest ? fmt(latest.representation_critic_mae) : "—"],
              ["Encoder cue memory", latest ? `${fmt(100*latest.encoder_cue_decode, 1)}%` : "—"],
              ["Predictor cue memory", latest ? `${fmt(100*latest.predictor_cue_decode, 1)}%` : "—"],
              ["Strategy cue memory", latest ? `${fmt(100*latest.cue_decode, 1)}%` : "—"],
              ["Curriculum", latest
                ? `${latest.curriculum_stage}/${latest.curriculum_stages}` : "—"],
            ];
            statsNode.innerHTML = cards.map(([label, value]) =>
              `<div class="stat"><span>${label}</span><strong>${value}</strong></div>`
            ).join("");

            const keys = [...grouped(rows).keys()];
            legendNode.innerHTML = keys.map((key, index) =>
              `<span><i style="background:${COLORS[index%COLORS.length]}"></i>${key}</span>`
            ).join("");

            if (!rows.length) return;
            drawChart(canvases.outcomes, rows, [
              {key:"window_success"}, {key:"window_wrong", dash:[6,4]},
              {key:"window_timeout", dash:[2,4]}
            ], [0, 1]);
            drawChart(canvases.reward, rows, [{key:"reward"}]);
            drawChart(canvases.learning, rows, [
              {key:"td_abs"}, {key:"prediction_mse", dash:[6,4]},
              {key:"joy_prediction_mse", dash:[8,3], width:2.5},
              {key:"reward_latent_shift", dash:[2,4]},
              {key:"representation_critic_mae", dash:[1,5], alpha:.35}
            ]);
            drawChart(canvases.strategy, rows, [
              {key:"shuffle_tv"}, {key:"zero_tv", dash:[6,4]}
            ], [0, Math.max(...rows.flatMap(
              row => [row.shuffle_tv, row.zero_tv]), .001)]);
          }

          model.on("change:records", update);
          model.on("change:status", update);
          const observer = new ResizeObserver(update);
          observer.observe(el);
          update();
          return () => observer.disconnect();
        }
        export default { render };
        """

        _css = r"""
        .ghost-dashboard {
          display: block; padding: 20px; border-radius: 18px;
          color: #e2e8f0;
          background:
            radial-gradient(circle at 15% 0%, rgba(124,58,237,.24), transparent 32%),
            linear-gradient(145deg, #111827, #07101e 70%);
          border: 1px solid rgba(148,163,184,.18);
          box-shadow: 0 20px 45px rgba(2,6,23,.28);
          font-family: Inter, ui-sans-serif, system-ui, sans-serif;
        }
        .dash-head { display:flex; align-items:center; justify-content:space-between; gap:16px; }
        .eyebrow { color:#a78bfa; font-size:10px; font-weight:800; letter-spacing:.18em; }
        .dash-title { font-size:22px; font-weight:750; margin-top:3px; }
        .run-state {
          padding:7px 11px; border-radius:999px; color:#94a3b8;
          background:rgba(148,163,184,.1); font-size:12px; font-weight:700;
        }
        .run-state.active { color:#67e8f9; background:rgba(6,182,212,.13); }
        .progress-shell {
          height:6px; overflow:hidden; margin:16px 0 18px; border-radius:999px;
          background:rgba(148,163,184,.12);
        }
        .progress-fill {
          width:0; height:100%; border-radius:inherit; transition:width .22s ease;
          background:linear-gradient(90deg,#7c3aed,#06b6d4);
        }
        .stat-grid {
          display:grid;
          grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
          gap:9px;
        }
        .stat {
          padding:10px 11px; border-radius:11px;
          background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.11);
        }
        .stat span { display:block; color:#94a3b8; font-size:10px; margin-bottom:4px; }
        .stat strong { font-size:15px; font-variant-numeric:tabular-nums; }
        .legend { display:flex; flex-wrap:wrap; gap:12px; min-height:18px; margin:15px 0 6px; }
        .legend span { color:#94a3b8; font-size:11px; }
        .legend i { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:5px; }
        .chart-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:11px; }
        .chart-grid section {
          min-width:0; padding:12px; border-radius:13px;
          background:rgba(15,23,42,.63); border:1px solid rgba(148,163,184,.1);
        }
        .chart-grid h3 { color:#cbd5e1; font-size:12px; margin:0 0 4px; font-weight:650; }
        .chart-grid canvas { display:block; width:100%; height:210px; }
        @media (max-width: 900px) {
          .stat-grid { grid-template-columns:repeat(3,1fr); }
          .chart-grid { grid-template-columns:1fr; }
        }
        """

        records = traitlets.List(default_value=[]).tag(sync=True)
        status = traitlets.Dict(default_value={}).tag(sync=True)

    dashboard_model = TrainingDashboard()
    training_dashboard = mo.ui.anywidget(dashboard_model)
    mo.vstack([
        mo.md("## Live dashboard"),
        training_dashboard,
        mo.md(
            "Solid lines show the primary metric; dashed lines show the "
            "comparison metric. Series colors identify architecture and seed."
        ),
    ])
    return (dashboard_model,)


@app.cell(hide_code=True)
def _(CONDITIONS, Config, mo):
    defaults = Config()

    def number(value, label, *, step=None, start=None, stop=None):
        return mo.ui.number(
            value=value,
            label=label,
            step=step,
            start=start,
            stop=stop,
            full_width=True,
        )

    run_controls = mo.ui.dictionary({
        "seed": number(defaults.seed, "Random seed", step=1, start=0),
        "device": mo.ui.dropdown(
            ["auto", "cpu", "mps", "cuda"],
            value=defaults.device,
            label="Device",
            full_width=True,
        ),
        "worlds": number(
            defaults.worlds, "Parallel worlds", step=1, start=1),
        "transitions": number(
            defaults.transitions, "Training transitions", step=1, start=1),
        "report_every": number(
            defaults.report_every,
            "Log every N transitions",
            step=1,
            start=1,
        ),
        "seeds": number(3, "Independent seeds", step=1, start=1),
        "conditions": mo.ui.multiselect(
            options=list(CONDITIONS),
            value=["separated"],
            label="Architectures",
            full_width=True,
        ),
        "evaluation_episodes": number(
            defaults.evaluation_episodes,
            "Evaluation episodes",
            step=1,
            start=1,
        ),
        "checkpoint": mo.ui.text(
            value="",
            label="Checkpoint filename (blank selects a default)",
            full_width=True,
        ),
        "save_checkpoint": mo.ui.switch(
            value=True, label="Save checkpoint"),
    }, label="Run")

    curriculum_controls = mo.ui.dictionary({
        "curriculum_success_threshold": number(
            defaults.curriculum_success_threshold,
            "Advancement success threshold",
            step=0.01,
            start=0.0,
            stop=1.0,
        ),
        "curriculum_min_episodes_per_cue": number(
            defaults.curriculum_min_episodes_per_cue,
            "Minimum episodes per cue",
            step=1,
            start=1,
        ),
        "curriculum_history_per_cue": number(
            defaults.curriculum_history_per_cue,
            "History length per cue",
            step=1,
            start=1,
        ),
    }, label="Curriculum")

    encoder_mode_options = {
        "Cue auxiliary (current baseline)": "cue_auxiliary",
        "Reward-AdaLN JEPA + actor e-prop": "reward_eprop",
        "Legacy reconstruction + cue + reward e-prop": "hybrid",
    }
    encoder_mode_default = next(
        label for label, mode in encoder_mode_options.items()
        if mode == defaults.encoder_learning_mode
    )

    architecture_controls = mo.ui.dictionary({
        "encoder_learning_mode": mo.ui.dropdown(
            options=encoder_mode_options,
            value=encoder_mode_default,
            label="Encoder task signal",
            full_width=True,
        ),
        "learned_strategy_memory": mo.ui.switch(
            value=defaults.learned_strategy_memory,
            label="Learned gated strategy memory",
        ),
        "use_reward_adaln": mo.ui.switch(
            value=defaults.use_reward_adaln,
            label="Reward-conditioned encoder AdaLN",
        ),
        "use_actor_encoder_eprop": mo.ui.switch(
            value=defaults.use_actor_encoder_eprop,
            label="Allow actor gradients into encoder",
        ),
        "use_predictor_eprop": mo.ui.switch(
            value=defaults.use_predictor_eprop,
            label="Temporal e-prop for recurrent predictor",
        ),
        "use_representation_critic": mo.ui.switch(
            value=defaults.use_representation_critic,
            label="Value-aware latent critic Q(z, strategy)",
        ),
        "latent_dim": number(
            defaults.latent_dim, "Latent dimension", step=1, start=1),
        "strategy_dim": number(
            defaults.strategy_dim, "Strategy dimension", step=1, start=1),
        "hidden_dim": number(
            defaults.hidden_dim, "Hidden dimension", step=1, start=1),
        "conditioning_dim": number(
            defaults.conditioning_dim,
            "Conditioning dimension",
            step=1,
            start=1,
        ),
        "snn_ticks": number(
            defaults.snn_ticks,
            "SNN ticks per decision",
            step=1,
            start=1,
        ),
        "strategy_retention": number(
            defaults.strategy_retention,
            "Fixed-memory retention",
            step=0.01,
            start=0.0,
            stop=1.0,
        ),
    }, label="Architecture")

    learning_controls = mo.ui.dictionary({
        "gamma": number(
            defaults.gamma,
            "Discount factor",
            step=0.001,
            start=0.0,
            stop=1.0,
        ),
        "exploration_rate": number(
            defaults.exploration_rate,
            "Training exploration",
            step=0.01,
            start=0.0,
            stop=1.0,
        ),
        "encoder_lr": number(
            defaults.encoder_lr,
            "Encoder auxiliary / JEPA learning rate",
            step=1e-5,
            start=0.0,
        ),
        "encoder_eprop_lr": number(
            defaults.encoder_eprop_lr,
            "Encoder reward e-prop learning rate",
            step=1e-5,
            start=0.0,
        ),
        "encoder_target_tau": number(
            defaults.encoder_target_tau,
            "JEPA target EMA update rate",
            step=0.001,
            start=0.001,
            stop=1.0,
        ),
        "jepa_variance_weight": number(
            defaults.jepa_variance_weight,
            "JEPA anti-collapse variance weight",
            step=0.01,
            start=0.0,
        ),
        "reward_adaln_strength": number(
            defaults.reward_adaln_strength,
            "Fixed reward-AdaLN strength",
            step=0.01,
            start=0.0,
        ),
        "cue_aux_weight": number(
            defaults.cue_aux_weight,
            "Cue auxiliary weight",
            step=0.1,
            start=0.0,
        ),
        "predictor_lr": number(
            defaults.predictor_lr,
            "Predictor learning rate",
            step=1e-5,
            start=0.0,
        ),
        "predictor_trace_decay": number(
            defaults.predictor_trace_decay,
            "Predictor eligibility decay",
            step=0.001,
            start=0.0,
            stop=1.0,
        ),
        "predictor_reward_event_weight": number(
            defaults.predictor_reward_event_weight,
            "Reward-event JEPA weight",
            step=0.5,
            start=1.0,
        ),
        "predictor_eprop_clip": number(
            defaults.predictor_eprop_clip,
            "Predictor e-prop gradient clip",
            step=0.1,
            start=0.0,
        ),
        "actor_eprop_lr": number(
            defaults.actor_eprop_lr,
            "Actor e-prop learning rate",
            step=1e-5,
            start=0.0,
        ),
        "strategy_eprop_lr": number(
            defaults.strategy_eprop_lr,
            "Strategizer e-prop learning rate",
            step=1e-5,
            start=0.0,
        ),
        "critic_lr": number(
            defaults.critic_lr,
            "Outcome-head learning rate",
            step=1e-5,
            start=0.0,
        ),
        "representation_critic_lr": number(
            defaults.representation_critic_lr,
            "Latent critic learning rate",
            step=1e-5,
            start=0.0,
        ),
        "representation_critic_target_tau": number(
            defaults.representation_critic_target_tau,
            "Latent critic EMA update rate",
            step=0.001,
            start=0.001,
            stop=1.0,
        ),
        "critic_encoder_weight": number(
            defaults.critic_encoder_weight,
            "Critic → encoder gradient weight",
            step=0.01,
            start=0.0,
        ),
        "actor_trace_decay": number(
            defaults.actor_trace_decay,
            "Actor trace decay",
            step=0.01,
            start=0.0,
            stop=1.0,
        ),
        "encoder_trace_decay": number(
            defaults.encoder_trace_decay,
            "Encoder reward-trace decay",
            step=0.01,
            start=0.0,
            stop=1.0,
        ),
        "strategy_trace_decay": number(
            defaults.strategy_trace_decay,
            "Strategy trace decay",
            step=0.01,
            start=0.0,
            stop=1.0,
        ),
    }, label="Learning")

    experiment_form = mo.ui.dictionary({
        "run": run_controls,
        "curriculum": curriculum_controls,
        "architecture": architecture_controls,
        "learning": learning_controls,
    }, label="Experiment configuration").form(
        label="Delayed-cue T-maze experiment",
        submit_button_label="Train and evaluate",
        submit_button_tooltip="Apply these values and start a fresh run",
        bordered=True,
    )

    mo.vstack([
        mo.md("## Experiment controls"),
        mo.md(
            "A full default run is compute-intensive. For a quick smoke test, "
            "reduce transitions, evaluation episodes, and independent seeds."
        ),
        mo.md(
            "**Reward-AdaLN JEPA** removes raw-observation reconstruction and "
            "cue classification. After an action, the signed reward modulates "
            "the stop-gradient target encoder through Adaptive LayerNorm; the "
            "recurrent predictor must predict that affected latent before the "
            "reward is revealed. Persistent predictor e-prop traces carry the "
            "terminal JEPA error back to recurrent activity from earlier "
            "decisions, and signed goal outcomes receive extra loss weight. "
            "Actor score gradients can independently flow into the encoder "
            "through a lower-rate reward e-prop path. "
            "The latent critic remains available as an ablation and is off by "
            "default."
        ),
        experiment_form,
    ])
    return (experiment_form,)


@app.cell
def _(
    BatchedTMaze,
    Config,
    Path,
    choose_device,
    contextlib,
    dashboard_model,
    evaluate,
    experiment_form,
    io,
    mo,
    np,
    run_condition,
    save,
    torch,
):
    mo.stop(
        experiment_form.value is None,
        mo.md("Submit the form above when you are ready to run."),
    )

    submitted = experiment_form.value
    run_values = dict(submitted["run"])
    config_values = {
        **dict(submitted["curriculum"]),
        **dict(submitted["architecture"]),
        **dict(submitted["learning"]),
    }
    for field in (
        "seed",
        "worlds",
        "transitions",
        "report_every",
        "evaluation_episodes",
    ):
        run_values[field] = int(run_values[field])
    for field in (
        "latent_dim",
        "strategy_dim",
        "hidden_dim",
        "conditioning_dim",
        "snn_ticks",
        "curriculum_min_episodes_per_cue",
        "curriculum_history_per_cue",
    ):
        config_values[field] = int(config_values[field])

    selected_conditions = list(run_values.pop("conditions"))
    seed_count = int(run_values.pop("seeds"))
    should_save = bool(run_values.pop("save_checkpoint"))
    requested_checkpoint = run_values.pop("checkpoint").strip()
    config_values.update(run_values)

    mo.stop(
        not selected_conditions,
        mo.md("Select at least one architecture before starting."),
    )
    mo.stop(
        config_values["curriculum_history_per_cue"]
        < config_values["curriculum_min_episodes_per_cue"],
        mo.md(
            "Curriculum history must be at least as large as the minimum "
            "episodes per cue."
        ),
    )

    if requested_checkpoint:
        config_values["checkpoint"] = requested_checkpoint
    else:
        memory_name = (
            "gated_memory" if config_values["learned_strategy_memory"]
            else "fixed_memory")
        encoder_name = config_values["encoder_learning_mode"]
        config_values["checkpoint"] = (
            f"online_delayed_cue_tmaze_{memory_name}_"
            f"{encoder_name}.pt")

    config = Config(**config_values)
    device = choose_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    preview = BatchedTMaze(config, config.seed, curriculum=True)

    live_records = []
    dashboard_model.records = []
    dashboard_model.status = {
        "running": True,
        "fraction": 0.0,
        "condition": selected_conditions[0],
        "seed": config.seed,
    }

    def update_dashboard(progress):
        live_records.append(dict(progress))
        dashboard_model.records = list(live_records)
        dashboard_model.status = {
            "running": True,
            "fraction": progress["fraction"],
            "condition": progress["condition"],
            "seed": progress["seed"],
        }

    log_buffer = io.StringIO()
    with contextlib.redirect_stdout(log_buffer):
        print(
            f"device={device} task=DELAYED_CUE_TMAZE "
            f"reward=SPARSE_SIGNED_TERMINAL(+1/-1/timeout"
            f"{config.timeout_penalty:+.1f}) worlds={config.worlds} "
            f"episode_limit={config.episode_limit}"
        )
        print(
            f"start_curriculum={preview.start_rows} "
            f"stage_limits={preview.episode_limits} "
            f"advance=min_cue_success>="
            f"{config.curriculum_success_threshold:.2f} "
            f"after>={config.curriculum_min_episodes_per_cue}"
            f"_episodes_per_cue "
            f"evaluation_start_y={preview.start_rows[-1]}"
        )
        print(
            "strategy_memory="
            + (
                "LEARNED_ELEMENTWISE_GATE+DIRECT_ACTOR_INPUT"
                if config.learned_strategy_memory
                else "FIXED_LEAKY_GATE"
            )
        )
        print(
            f"encoder_learning={config.encoder_learning_mode} "
            f"auxiliary_or_jepa_lr={config.encoder_lr} "
            f"reward_eprop_lr={config.encoder_eprop_lr} "
            f"reward_trace={config.encoder_trace_decay} "
            f"target_tau={config.encoder_target_tau} "
            f"jepa_variance={config.jepa_variance_weight} "
            f"reward_adaln={config.use_reward_adaln} "
            f"reward_adaln_strength={config.reward_adaln_strength} "
            f"actor_to_encoder={config.use_actor_encoder_eprop} "
            f"predictor_eprop={config.use_predictor_eprop} "
            f"predictor_trace={config.predictor_trace_decay} "
            f"reward_event_weight={config.predictor_reward_event_weight} "
            f"predictor_eprop_clip={config.predictor_eprop_clip} "
            f"latent_critic={config.use_representation_critic} "
            f"critic_to_encoder={config.critic_encoder_weight}"
        )

        results = []
        for seed_index in range(seed_count):
            seed_value = config.seed + seed_index
            for condition in selected_conditions:
                result = run_condition(
                    config,
                    condition,
                    seed_value,
                    device,
                    progress_callback=update_dashboard,
                )
                system = result["system"]
                real = evaluate(
                    system, config, seed_value, "real")
                shuffled = evaluate(
                    system, config, seed_value, "shuffle")
                zero = evaluate(
                    system, config, seed_value, "zero")
                result.update(
                    eval_real=real[0],
                    eval_shuffle=shuffled[0],
                    eval_zero=zero[0],
                    eval_wrong=real[1],
                )
                print(
                    f"evaluation condition={condition} seed={seed_value} "
                    f"train={result['successes']}/{result['episodes']} "
                    f"real={real[0]:.3f} shuffled={shuffled[0]:.3f} "
                    f"zero={zero[0]:.3f} wrong={real[1]:.3f}"
                )
                results.append(result)

        aggregates = []
        print("aggregate")
        for condition in selected_conditions:
            matching = [
                row for row in results
                if row["condition"] == condition
            ]
            aggregate = {
                "condition": condition,
                "train_rate": float(np.mean(
                    [row["rate"] for row in matching])),
                "eval_real": float(np.mean(
                    [row["eval_real"] for row in matching])),
                "eval_shuffle": float(np.mean(
                    [row["eval_shuffle"] for row in matching])),
                "eval_zero": float(np.mean(
                    [row["eval_zero"] for row in matching])),
            }
            aggregates.append(aggregate)
            print(
                f"  {condition:21s} "
                f"train_rate={aggregate['train_rate']:.3f} "
                f"eval_real={aggregate['eval_real']:.3f} "
                f"eval_shuffle={aggregate['eval_shuffle']:.3f} "
                f"eval_zero={aggregate['eval_zero']:.3f}"
            )

        dashboard_model.status = {
            "running": False,
            "fraction": 1.0,
        }
        checkpoint = Path(__file__).resolve().parent / config.checkpoint
        if should_save:
            save(checkpoint, config, results)
            print(f"saved_checkpoint={checkpoint}")
        else:
            checkpoint = None
            print("checkpoint_not_saved")

    seed_rows = [{
        "condition": row["condition"],
        "encoder mode": config.encoder_learning_mode,
        "seed": row["seed"],
        "episodes": row["episodes"],
        "train rate": round(row["rate"], 4),
        "wrong rate": round(row["wrong"], 4),
        "timeout rate": round(row["timeout"], 4),
        "eval real": round(row["eval_real"], 4),
        "eval shuffled": round(row["eval_shuffle"], 4),
        "eval zero": round(row["eval_zero"], 4),
    } for row in results]
    summary_rows = [{
        "condition": row["condition"],
        "encoder mode": config.encoder_learning_mode,
        "train rate": round(row["train_rate"], 4),
        "eval real": round(row["eval_real"], 4),
        "eval shuffled": round(row["eval_shuffle"], 4),
        "eval zero": round(row["eval_zero"], 4),
        "shuffle drop": round(
            row["eval_real"] - row["eval_shuffle"], 4),
        "zero drop": round(
            row["eval_real"] - row["eval_zero"], 4),
    } for row in aggregates]
    training_log = log_buffer.getvalue()
    return checkpoint, seed_rows, summary_rows, training_log


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Diagnostics

    **Cue Strength**

    $$
    \text{cue strength} = \frac{\mu_{left} - \mu_{right}}{\sqrt{1/2 (\sigma_{left}^2 + \sigma_{right}^2)}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(checkpoint, mo, seed_rows, summary_rows, training_log):
    checkpoint_message = (
        f"Checkpoint saved to `{checkpoint}`"
        if checkpoint is not None
        else "Checkpoint saving was disabled."
    )
    mo.vstack([
        mo.md("## Results"),
        mo.md(checkpoint_message),
        mo.md("### Aggregate comparison"),
        mo.ui.table(summary_rows),
        mo.md(
            "Shuffle and zero drops measure how much evaluation performance "
            "depends on the learned strategy signal."
        ),
        mo.md("### Per-seed results"),
        mo.ui.table(seed_rows),
        mo.md("### Training log"),
        mo.md(f"```text\n{training_log}\n```"),
    ])
    return


if __name__ == "__main__":
    app.run()
