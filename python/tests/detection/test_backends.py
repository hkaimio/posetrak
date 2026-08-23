# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for backends.py's corrupt-checkpoint self-heal retry.

Found via the installer prototype's Windows Sandbox testing, 2026-08-23:
rtmlib's download_checkpoint() treats "a file already exists at the cache
path" as "already downloaded", and its download never verifies the byte
count against Content-Length before atomically renaming into place -- a
connection dropped mid-transfer leaves a permanently-corrupt cached file
that fails onnxruntime on every subsequent attempt with no way to recover.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from posetrak.detection.backends import construct_with_corrupt_checkpoint_retry


def test_returns_factory_result_on_success() -> None:
    assert construct_with_corrupt_checkpoint_retry(lambda: 42, "https://example/x.onnx") == 42


def test_retries_once_after_clearing_stale_cache_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "model.onnx"
    stale.write_bytes(b"corrupt")
    monkeypatch.setattr(
        "posetrak.detection.backends._rtmlib_checkpoint_cache_paths",
        lambda url: [stale],
    )

    calls = {"n": 0}

    def factory() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated onnxruntime InvalidProtobuf")
        return "ok"

    result = construct_with_corrupt_checkpoint_retry(factory, "https://example/model.onnx")

    assert result == "ok"
    assert calls["n"] == 2
    assert not stale.exists()


def test_reraises_original_error_when_nothing_to_clean_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "posetrak.detection.backends._rtmlib_checkpoint_cache_paths",
        lambda url: [],
    )
    calls = {"n": 0}

    def factory() -> None:
        calls["n"] += 1
        raise RuntimeError("real failure, not a corrupt-cache one")

    with pytest.raises(RuntimeError, match="real failure"):
        construct_with_corrupt_checkpoint_retry(factory, "https://example/model.onnx")
    assert calls["n"] == 1  # not retried -- there was nothing to clean up


def test_reraises_if_retry_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = tmp_path / "model.onnx"
    stale.write_bytes(b"corrupt")
    monkeypatch.setattr(
        "posetrak.detection.backends._rtmlib_checkpoint_cache_paths",
        lambda url: [stale],
    )

    def factory() -> None:
        raise RuntimeError("still broken after retry")

    with pytest.raises(RuntimeError, match="still broken after retry"):
        construct_with_corrupt_checkpoint_retry(factory, "https://example/model.onnx")
    assert not stale.exists()  # cleaned up even though the retry itself failed too
