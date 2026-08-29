"""Strict checkpoint loading for the canonical experiment."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import torch

from .config import Config
from .system import System

MODEL_NAMES = ("encoder", "strategizer", "actor", "predictor")


class CheckpointCompatibilityError(RuntimeError):
    """Raised when an artifact does not match the available implementation."""


def read_checkpoint(path: str | Path) -> dict:
    """Read a Ghost checkpoint without enabling arbitrary pickle globals."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "config" not in payload:
        raise CheckpointCompatibilityError("not a recognized Ghost checkpoint")
    if not isinstance(payload.get("conditions"), dict):
        raise CheckpointCompatibilityError("checkpoint has no condition states")
    return payload


def config_from_checkpoint(payload: dict, **overrides) -> Config:
    """Rebuild a config, filling fields absent from older payloads with defaults."""
    known = {field.name for field in fields(Config)}
    values = {key: value for key, value in payload["config"].items() if key in known}
    config = Config(**values)
    return replace(config, **overrides) if overrides else config


def available_runs(payload: dict) -> list[tuple[str, int]]:
    runs = []
    for key in payload["conditions"]:
        try:
            condition, seed_text = key.rsplit("_seed", 1)
            runs.append((condition, int(seed_text)))
        except ValueError:
            continue
    return sorted(runs)


def load_system(
    path: str | Path,
    condition: str,
    seed: int,
    device: torch.device,
    *,
    evaluation_episodes: int | None = None,
) -> tuple[System, Config, dict]:
    """Strictly load one condition/seed and return the system, config, and payload."""
    payload = read_checkpoint(path)
    overrides = {"device": str(device)}
    if evaluation_episodes is not None:
        overrides["evaluation_episodes"] = evaluation_episodes
    config = config_from_checkpoint(payload, **overrides)
    key = f"{condition}_seed{seed}"
    if key not in payload["conditions"]:
        choices = ", ".join(
            f"{name}/seed-{run_seed}" for name, run_seed in available_runs(payload)
        )
        raise KeyError(f"{key!r} is absent; available runs: {choices or 'none'}")

    system = System(config, condition, device, seed)
    state = payload["conditions"][key]
    try:
        for name in MODEL_NAMES:
            if name not in state:
                raise CheckpointCompatibilityError(f"checkpoint is missing {name!r}")
            getattr(system, name).load_state_dict(state[name], strict=True)
        for name in (
            "target_encoder",
            "representation_critic",
            "target_representation_critic",
        ):
            module = getattr(system, name, None)
            if name in state and module is not None:
                module.load_state_dict(state[name], strict=True)
    except (RuntimeError, CheckpointCompatibilityError) as error:
        raise CheckpointCompatibilityError(
            "checkpoint tensors are incompatible with the canonical engine; "
            "use the source revision that created the artifact or retrain with "
            "the documented command"
        ) from error
    return system, config, payload
