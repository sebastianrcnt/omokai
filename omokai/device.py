from __future__ import annotations

from contextlib import nullcontext

import torch


def is_mps_available() -> bool:
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        return torch.device(requested)
    if requested == "mps":
        return torch.device("mps")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


def amp_context(device: torch.device, enabled: bool) -> object:
    if not enabled or device.type in {"cpu", "mps"}:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)
