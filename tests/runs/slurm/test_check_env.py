"""Tests for SLURM pre-flight environment checks."""

from __future__ import annotations

import pytest

from cybench.models.tabular_foundation_model import _is_cuda_recoverable_error
from cybench.runs.slurm import check_env


def test_is_cuda_recoverable_error_matches_arch_mismatch():
    assert _is_cuda_recoverable_error(
        RuntimeError("CUDA error: no kernel image is available for execution on the device")
    )
    assert _is_cuda_recoverable_error(RuntimeError("CUBLAS_STATUS_ARCH_MISMATCH"))
    assert not _is_cuda_recoverable_error(ValueError("shape mismatch"))


def test_torch_torchvision_compat_accepts_paired_versions(monkeypatch, tmp_path):
    nccl_lib = tmp_path / "nvidia" / "nccl" / "lib"
    nccl_lib.mkdir(parents=True)
    (nccl_lib / "libnccl.so.2").write_bytes(b"x")
    monkeypatch.setattr(check_env.site, "getsitepackages", lambda: [str(tmp_path)])

    class FakeTorch:
        __version__ = "2.6.0+cu124"

    class FakeTorchvision:
        __version__ = "0.21.0"

    monkeypatch.setitem(check_env.sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(check_env.sys.modules, "torchvision", FakeTorchvision)
    check_env.check_torch_torchvision_compat()


def test_check_nccl_wheel_detects_missing_lib(tmp_path, monkeypatch):
    nccl_lib = tmp_path / "nvidia" / "nccl" / "lib"
    nccl_lib.mkdir(parents=True)
    monkeypatch.setattr(check_env.site, "getsitepackages", lambda: [str(tmp_path)])
    with pytest.raises(RuntimeError, match="Incomplete nvidia-nccl-cu12"):
        check_env.check_nccl_wheel()


def test_check_nccl_wheel_passes_when_present(tmp_path, monkeypatch):
    nccl_lib = tmp_path / "nvidia" / "nccl" / "lib"
    nccl_lib.mkdir(parents=True)
    (nccl_lib / "libnccl.so.2").write_bytes(b"x")
    monkeypatch.setattr(check_env.site, "getsitepackages", lambda: [str(tmp_path)])
    check_env.check_nccl_wheel()


def test_torch_torchvision_compat_rejects_mismatch(monkeypatch, tmp_path):
    nccl_lib = tmp_path / "nvidia" / "nccl" / "lib"
    nccl_lib.mkdir(parents=True)
    (nccl_lib / "libnccl.so.2").write_bytes(b"x")
    monkeypatch.setattr(check_env.site, "getsitepackages", lambda: [str(tmp_path)])

    class FakeTorch:
        __version__ = "2.12.1+cu130"

    class FakeTorchvision:
        __version__ = "0.20.1"

    monkeypatch.setitem(check_env.sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(check_env.sys.modules, "torchvision", FakeTorchvision)
    with pytest.raises(RuntimeError, match="Incompatible torch/torchvision"):
        check_env.check_torch_torchvision_compat()
