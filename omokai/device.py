from __future__ import annotations

from contextlib import nullcontext

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    if requested.startswith("cuda:"):
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


def amp_context(device: torch.device, enabled: bool) -> object:
    if not enabled or device.type == "cpu":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16)

