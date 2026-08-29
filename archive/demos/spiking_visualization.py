import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Toy spiking neural network

    This notebook simulates a small feed-forward network of
    leaky-integrate-and-fire (LIF) neurons:

    **2 Poisson inputs → 8 hidden neurons → 2 output neurons**

    Halfway through each trial, the active input switches. The synaptic
    weights are structured so output neuron 0 prefers the first stimulus
    and output neuron 1 prefers the second.
    """)
    return


@app.cell
def _(mo):
    duration = mo.ui.slider(
        start=200,
        stop=800,
        step=50,
        value=400,
        label="Duration (ms)",
    )
    input_rate = mo.ui.slider(
        start=20,
        stop=120,
        step=5,
        value=75,
        label="Active input rate (Hz)",
    )
    threshold = mo.ui.slider(
        start=0.7,
        stop=1.5,
        step=0.05,
        value=1.0,
        label="Spike threshold",
    )
    seed = mo.ui.number(start=0, stop=10_000, value=7, label="Random seed")

    controls = mo.hstack(
        [duration, input_rate, threshold, seed],
        justify="space-around",
        gap=2,
    )
    controls
    return duration, input_rate, seed, threshold


@app.cell
def _(np):
    def simulate_snn(
        duration_ms: int,
        active_rate_hz: float,
        spike_threshold: float,
        random_seed: int,
    ):
        """Simulate a two-layer current-based LIF spiking network."""
        dt_ms = 1.0
        steps = int(duration_ms / dt_ms)
        time_ms = np.arange(steps) * dt_ms
        switch_step = steps // 2
        rng = np.random.default_rng(random_seed)

        # The first input is active before the switch; the second is active after.
        low_rate_hz = 5.0
        rates_hz = np.full((steps, 2), low_rate_hz)
        rates_hz[:switch_step, 0] = active_rate_hz
        rates_hz[switch_step:, 1] = active_rate_hz
        input_spikes = rng.random((steps, 2)) < rates_hz * dt_ms / 1000.0

        n_hidden = 8
        n_output = 2

        # Hidden neurons 0–3 prefer input 0; neurons 4–7 prefer input 1.
        weights_in = np.full((2, n_hidden), 0.08)
        weights_in[0, :4] = rng.uniform(0.42, 0.58, size=4)
        weights_in[1, 4:] = rng.uniform(0.42, 0.58, size=4)

        # Each output pools spikes from one preferred hidden population.
        weights_out = np.full((n_hidden, n_output), 0.025)
        weights_out[:4, 0] = rng.uniform(0.24, 0.34, size=4)
        weights_out[4:, 1] = rng.uniform(0.24, 0.34, size=4)

        tau_hidden_ms = 18.0
        tau_output_ms = 25.0
        hidden_voltage = np.zeros((steps, n_hidden))
        output_voltage = np.zeros((steps, n_output))
        hidden_spikes = np.zeros((steps, n_hidden), dtype=bool)
        output_spikes = np.zeros((steps, n_output), dtype=bool)

        hidden_state = np.zeros(n_hidden)
        output_state = np.zeros(n_output)

        for step in range(1, steps):
            hidden_current = input_spikes[step].astype(float) @ weights_in
            hidden_state += (
                -hidden_state / tau_hidden_ms + hidden_current
            ) * dt_ms
            fired_hidden = hidden_state >= spike_threshold
            hidden_spikes[step] = fired_hidden
            hidden_voltage[step] = hidden_state
            hidden_state[fired_hidden] = 0.0

            output_current = fired_hidden.astype(float) @ weights_out
            output_state += (
                -output_state / tau_output_ms + output_current
            ) * dt_ms
            fired_output = output_state >= spike_threshold
            output_spikes[step] = fired_output
            output_voltage[step] = output_state
            output_state[fired_output] = 0.0

        half_duration_s = (duration_ms / 2.0) / 1000.0
        output_rates_hz = np.vstack(
            [
                output_spikes[:switch_step].sum(axis=0) / half_duration_s,
                output_spikes[switch_step:].sum(axis=0) / half_duration_s,
            ]
        )

        return {
            "time_ms": time_ms,
            "switch_ms": switch_step * dt_ms,
            "rates_hz": rates_hz,
            "input_spikes": input_spikes,
            "hidden_spikes": hidden_spikes,
            "output_spikes": output_spikes,
            "hidden_voltage": hidden_voltage,
            "output_voltage": output_voltage,
            "output_rates_hz": output_rates_hz,
        }

    return (simulate_snn,)


@app.cell
def _(duration, input_rate, seed, simulate_snn, threshold):
    result = simulate_snn(
        duration_ms=duration.value,
        active_rate_hz=input_rate.value,
        spike_threshold=threshold.value,
        random_seed=int(seed.value),
    )
    return (result,)


@app.cell
def _(np, plt, result, threshold):
    _time = result["time_ms"]
    _switch = result["switch_ms"]
    _fig, _axes = plt.subplots(
        5,
        1,
        figsize=(11, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.5, 1.3, 1.0]},
        constrained_layout=True,
    )

    # Input rates
    _axes[0].plot(_time, result["rates_hz"][:, 0], label="Input 0", lw=2)
    _axes[0].plot(_time, result["rates_hz"][:, 1], label="Input 1", lw=2)
    _axes[0].set_ylabel("Rate (Hz)")
    _axes[0].set_title("Stimulus schedule")
    _axes[0].legend(loc="upper center", ncol=2)

    # Input spike raster
    for _neuron in range(2):
        _spike_times = _time[result["input_spikes"][:, _neuron]]
        _axes[1].scatter(
            _spike_times,
            np.full_like(_spike_times, _neuron),
            marker="|",
            s=90,
        )
    _axes[1].set_yticks([0, 1], ["Input 0", "Input 1"])
    _axes[1].set_ylabel("Input")
    _axes[1].set_title("Poisson input spikes")

    # Hidden-layer spike raster
    for _neuron in range(result["hidden_spikes"].shape[1]):
        _spike_times = _time[result["hidden_spikes"][:, _neuron]]
        _axes[2].scatter(
            _spike_times,
            np.full_like(_spike_times, _neuron),
            marker="|",
            s=65,
            color="C0" if _neuron < 4 else "C1",
        )
    _axes[2].set_yticks(range(8))
    _axes[2].set_ylabel("Neuron")
    _axes[2].set_title("Hidden-layer spike raster")

    # Output membrane potentials (recorded before reset)
    for _neuron in range(2):
        _axes[3].plot(
            _time,
            result["output_voltage"][:, _neuron],
            label=f"Output {_neuron}",
            lw=1.5,
        )
    _axes[3].axhline(
        threshold.value,
        color="black",
        ls="--",
        lw=1,
        label="Threshold",
    )
    _axes[3].set_ylabel("Voltage")
    _axes[3].set_title("Output membrane potentials")
    _axes[3].legend(loc="upper center", ncol=3)

    # Output spike raster
    for _neuron in range(2):
        _spike_times = _time[result["output_spikes"][:, _neuron]]
        _axes[4].scatter(
            _spike_times,
            np.full_like(_spike_times, _neuron),
            marker="|",
            s=110,
        )
    _axes[4].set_yticks([0, 1], ["Output 0", "Output 1"])
    _axes[4].set_ylabel("Readout")
    _axes[4].set_title("Output spikes")
    _axes[4].set_xlabel("Time (ms)")

    for _axis in _axes:
        _axis.axvline(_switch, color="0.35", ls=":", lw=1.25)
        _axis.grid(alpha=0.2)

    _fig.suptitle("Toy LIF spiking neural network", fontsize=15)
    _fig
    return


@app.cell
def _(mo, result):
    _rates = result["output_rates_hz"]
    mo.md(f"""
    ### Output firing-rate summary

    | Trial segment | Output 0 | Output 1 |
    |:--|--:|--:|
    | First stimulus | {_rates[0, 0]:.1f} Hz | {_rates[0, 1]:.1f} Hz |
    | Second stimulus | {_rates[1, 0]:.1f} Hz | {_rates[1, 1]:.1f} Hz |

    The dotted vertical line in every plot marks the stimulus switch.
    Change any control above and Marimo will rerun the simulation and plots.
    """)
    return


if __name__ == "__main__":
    app.run()
