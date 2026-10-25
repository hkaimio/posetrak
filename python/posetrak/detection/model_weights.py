# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""model_weights.py — the GEM-X 2D pose weights: licences, download on first use, location.

The weights of the SOMA-77 model are not part of the repository or the installer; they are governed by the
NVIDIA Open Model License and the DINOv3 License (design:
docs/roadmap/features/detection-pipeline/detection-pipeline-design.md, section 4). They are downloaded from the
``nvidia/GEM-X`` Hugging Face repository into the model directory after the user has been shown the text of both
licences and has accepted them. The texts are bundled with posetrak (``posetrak/data/licences``), so showing and
accepting them needs no network access; only the download does. The acceptance is stored beside the files with
the SHA-256 of each bundled text; :func:`accepted_licences` counts a record whose hash differs from the text now
bundled as not accepted, so a release that ships a changed text asks again.

``POSETRAK_MODEL_DIR`` moves the model directory (default ``~/.posetrak/models``). ``POSETRAK_SOMA77_MODEL``
names an ``onnx/vitpose.onnx`` the user obtained and licensed on their own; it bypasses the download and the
acceptance record.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

_log = logging.getLogger(__name__)

GEMX_REPO = "nvidia/GEM-X"
GEMX_FILES = ("onnx/vitpose.onnx", "onnx/vitpose.onnx.data")
MODEL_PATH_ENV = "POSETRAK_SOMA77_MODEL"
MODEL_DIR_ENV = "POSETRAK_MODEL_DIR"

#: Shown in the about box and the user documentation, and kept with the weights.
NVIDIA_NOTICE = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
DINOV3_NOTICE = "Built with DINOv3"


@dataclass(frozen=True)
class Licence:
    id: str
    name: str
    url: str  # where the bundled text was copied from
    file: str  # file name in posetrak/data/licences


LICENCES = (
    Licence(
        "nvidia-open-model-license", "NVIDIA Open Model License Agreement",
        "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
        "nvidia-open-model-license.txt",
    ),
    Licence(
        "dinov3-license", "DINOv3 License Agreement",
        "https://ai.meta.com/resources/models-and-libraries/dinov3-license/",
        "dinov3-license.txt",
    ),
)


class LicenceDeclined(Exception):
    """The user did not accept the licences."""


def model_dir() -> Path:
    return Path(os.environ.get(MODEL_DIR_ENV) or Path.home() / ".posetrak" / "models")


def _gemx_dir() -> Path:
    return model_dir() / "gem-x"


def _acceptance_file() -> Path:
    return _gemx_dir() / "licence-acceptance.json"


def soma77_model_path() -> Path | None:
    """The ONNX file to run, or None when the weights are not available (not downloaded, or not accepted)."""
    override = os.environ.get(MODEL_PATH_ENV)
    if override:
        path = Path(override)
        return path if path.is_file() else None
    path = _gemx_dir() / GEMX_FILES[0]
    complete = all((_gemx_dir() / f).is_file() for f in GEMX_FILES)
    return path if complete and accepted_licences() else None


def licence_text(licence: Licence) -> str:
    """The bundled text of *licence*, with line ends normalised to LF so that its hash is the same everywhere."""
    raw = resources.files("posetrak.data.licences").joinpath(licence.file).read_bytes().decode("utf-8")
    return raw.replace("\r\n", "\n")


def licence_hash(licence: Licence) -> str:
    """SHA-256 (hex) of the UTF-8 bytes of the bundled text of *licence*; the version recorded when it is accepted."""
    return hashlib.sha256(licence_text(licence).encode("utf-8")).hexdigest()


def bundled_licences() -> list[tuple[Licence, str]]:
    """Both licences with their bundled texts, in the order they are shown."""
    return [(licence, licence_text(licence)) for licence in LICENCES]


def accepted_licences() -> list[dict]:
    """The stored acceptance records (id, text_sha256, ...), or [] unless every licence in :data:`LICENCES` has a
    record whose ``text_sha256`` equals the hash of the text bundled now. A missing or unreadable record, a missing
    licence, or a record of an older text all count as not accepted."""
    try:
        record = json.loads(_acceptance_file().read_text(encoding="utf-8"))
        by_id = {entry["id"]: entry["text_sha256"] for entry in record["licences"]}
        if all(by_id.get(lic.id) == licence_hash(lic) for lic in LICENCES):
            return record["licences"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return []


AcceptCallback = Callable[[list[tuple[Licence, str]]], bool]


def install_gemx(progress: Callable[[str], None] | None = None) -> Path:
    """Download the weights for a user who has accepted the bundled licences, and record the acceptance."""
    from huggingface_hub import hf_hub_download

    target = _gemx_dir()
    target.mkdir(parents=True, exist_ok=True)
    for filename in GEMX_FILES:
        if progress:
            progress(f"Downloading {GEMX_REPO}/{filename}")
        hf_hub_download(GEMX_REPO, filename, local_dir=str(target))
    record = {
        "repo": GEMX_REPO,
        "notices": [NVIDIA_NOTICE, DINOV3_NOTICE],
        "accepted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "licences": [
            {"id": lic.id, "name": lic.name, "url": lic.url,
             "text_sha256": licence_hash(lic)}
            for lic in LICENCES
        ],
    }
    _acceptance_file().write_text(json.dumps(record, indent=1), encoding="utf-8")
    path = soma77_model_path()
    assert path is not None
    return path


def download_gemx(
    accept: AcceptCallback,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Show the bundled licence texts to *accept*, and on acceptance download the weights; returns the ONNX path.

    *accept* receives ``[(licence, text), ...]`` and returns whether the user accepted all of them. Raises
    :class:`LicenceDeclined` when it returns False, and the usual network errors from the download.
    """
    if not accept(bundled_licences()):
        raise LicenceDeclined("The licences of the SOMA-77 pose model were not accepted.")
    return install_gemx(progress)
