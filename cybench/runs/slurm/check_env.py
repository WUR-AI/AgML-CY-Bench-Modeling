#!/usr/bin/env python3
"""Pre-flight checks for SLURM benchmark jobs (torch stack, CUDA, HF encoders)."""

from __future__ import annotations

import argparse
import importlib
import re
import site
import sys
from pathlib import Path
from typing import Sequence

# Models whose Hydra config uses hf_temporal_encoder.*
HF_TEMPORAL_MODELS = frozenset(
    {"patchtst_lf", "autoformer_lf", "informer_lf", "tst_lf"}
)

# (torch major.minor, torchvision major.minor) pairs from PyTorch release notes.
_TORCH_TV_PAIRS: Sequence[tuple[tuple[int, int], tuple[int, int]]] = (
    ((2, 6), (0, 21)),
    ((2, 7), (0, 22)),
    ((2, 8), (0, 23)),
    ((2, 9), (0, 24)),
    ((2, 10), (0, 25)),
    ((2, 11), (0, 26)),
    ((2, 12), (0, 27)),
)


def _parse_version(prefix: str, version: str) -> tuple[int, int]:
    match = re.match(rf"^{prefix}\.?(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Could not parse version from {version!r}")
    return int(match.group(1)), int(match.group(2))


def nvidia_wheel_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    for root in site.getsitepackages():
        base = Path(root) / "nvidia"
        if not base.is_dir():
            continue
        for lib_dir in sorted(base.glob("*/lib")):
            if lib_dir.is_dir():
                dirs.append(lib_dir)
    return dirs


def check_nccl_wheel() -> None:
    """PyTorch CUDA wheels need libnccl.so.2 from the nvidia-nccl-cu12 pip package."""
    for lib_dir in nvidia_wheel_lib_dirs():
        if lib_dir.parent.name != "nccl":
            continue
        libnccl = lib_dir / "libnccl.so.2"
        if libnccl.is_file():
            return
        raise RuntimeError(
            f"Incomplete nvidia-nccl-cu12 install: missing {libnccl}. "
            "Reinstall with: poetry run pip install --force-reinstall --no-cache-dir "
            "nvidia-nccl-cu12==2.21.5"
        )


def check_torch_torchvision_compat() -> None:
    check_nccl_wheel()
    import torch

    torch_mm = _parse_version("", torch.__version__.split("+", 1)[0])
    try:
        import torchvision
    except ImportError:
        # transformers may pull torchvision lazily; HF temporal models need it.
        raise RuntimeError(
            "torchvision is not installed. Add it via poetry (torchvision ^0.21 for torch 2.6) "
            "and run `poetry sync` on the cluster venv."
        ) from None

    tv_mm = _parse_version("", torchvision.__version__.split("+", 1)[0])
    if any(torch_mm == t and tv_mm == v for t, v in _TORCH_TV_PAIRS):
        return

    expected = ", ".join(f"torch {t[0]}.{t[1]} + torchvision {v[0]}.{v[1]}" for t, v in _TORCH_TV_PAIRS)
    raise RuntimeError(
        f"Incompatible torch/torchvision: torch {torch.__version__}, "
        f"torchvision {torchvision.__version__}. "
        f"Expected one of: {expected}. "
        "Run `poetry sync` or install a matching torchvision wheel."
    )


def check_hf_temporal_encoders() -> None:
    importlib.import_module(
        "cybench.models.torch.model_components.hf_temporal_encoder"
    )


def probe_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    x = torch.randn(4, 4, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    float(y.sum().item())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Hydra model slug for this array task (enables HF encoder import check)",
    )
    parser.add_argument(
        "--probe-cuda",
        action="store_true",
        help="Run a small CUDA matmul; exit 1 on arch/driver errors",
    )
    parser.add_argument(
        "--check-torch-stack",
        action="store_true",
        help="Verify torch and torchvision versions are compatible",
    )
    parser.add_argument(
        "--check-hf-encoders",
        action="store_true",
        help="Import hf_temporal_encoder (or auto when --model is an HF temporal slug)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_torch = args.check_torch_stack or not any(
        (args.probe_cuda, args.check_hf_encoders, args.model)
    )
    run_hf = args.check_hf_encoders or (
        args.model is not None and args.model in HF_TEMPORAL_MODELS
    )

    try:
        if run_torch:
            check_torch_torchvision_compat()
        if run_hf:
            check_hf_temporal_encoders()
        if args.probe_cuda:
            probe_cuda()
    except Exception as exc:
        print(f"[check_env] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
