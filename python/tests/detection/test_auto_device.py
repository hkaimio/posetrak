# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for _auto_device()'s CPU/GPU fallback in the rtmlib detection backends.

Found via the installer prototype's Windows Sandbox testing, 2026-08-23: the
previous torch-absent fallback trusted onnxruntime.get_available_providers(),
which only reflects what onnxruntime-gpu was *compiled* with, not whether
CUDA is actually installed and loadable -- a CPU-only machine without torch
(onnxruntime-gpu is a core dependency; torch is only in the optional
segmentation extras group) reported "cuda" as available regardless.
"""
from __future__ import annotations

import sys

import pytest

import posetrak.detection.backends_rtmdet as rtmdet
import posetrak.detection.backends_rtmpose as rtmpose


@pytest.mark.parametrize("module", [rtmdet, rtmpose])
def test_auto_device_without_torch_defaults_to_cpu(
    module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without torch there's no reliable signal CUDA is actually usable --
    must not trust onnxruntime's compiled-in provider list."""
    monkeypatch.setitem(sys.modules, "torch", None)  # forces ImportError on `import torch`
    assert module._auto_device() == "cpu"


@pytest.mark.parametrize("module", [rtmdet, rtmpose])
def test_auto_device_follows_torch_cuda_available(
    module, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert module._auto_device() == "cuda"


@pytest.mark.parametrize("module", [rtmdet, rtmpose])
def test_auto_device_follows_torch_cuda_unavailable(
    module, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert module._auto_device() == "cpu"
