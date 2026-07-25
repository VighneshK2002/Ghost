import marimo

__generated_with = "0.23.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return mo, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Config Parameters
    """)
    return


@app.cell
def _(dataclass, shared):
    @dataclass
    class Config(shared.Config):
        latent_dim = 0.0

    return (Config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Recurrent Spiking Neural Network
    """)
    return


@app.cell
def _(Config, gain, nn, torch):
    class RecurrentSNN(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, cfg: Config, persistent: bool,
                     decay: float | None = None,  record_eligibility: bool = False) -> None:
            self.cfg, self.hidden_dim, self.persistent = cfg, hidden_dim,persistent
            self.decay = cfg.membrane_decay if decay is None else decay
            self.record_eligibility = record_eligibility
            self.input = nn.Linear(input_dim, hidden_dim)
            self.recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
            nn.init.orthagonal_(self.recurrent.weight, gain)
            self.bias = nn.Parameter(torch.full)
    

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Ghost
    """)
    return


app._unparsable_cell(
    r"""
    #Ghost
    class Ghost(nn.Module)
        def __init__(self) --> None:
            input_dim = 2*latent_dim + 2
            self.core = RecurrentSNN(
            
            )
        

    """,
    name="_"
)


@app.cell
def _():
    #Actor
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    """)
    return


if __name__ == "__main__":
    app.run()
