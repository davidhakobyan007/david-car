"""Device selection + GPU-saturation knobs."""
from __future__ import annotations

import torch


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("cuda", "gpu") or (prefer == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def tune_for_throughput(device: torch.device) -> None:
    """Enable the settings that let a GPU run flat-out."""
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        i = torch.cuda.current_device()
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        return f"CUDA GPU: {name} ({total:.1f} GB)"
    return "CPU (no CUDA GPU detected -- training will be slow)"
