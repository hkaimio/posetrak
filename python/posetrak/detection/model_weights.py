# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""model_weights.py — the GEM-X 2D pose weights: licences, download on first use, location.

The weights of the SOMA-77 model are not part of the repository or the installer; they are governed by the
NVIDIA Open Model License and, assumed until NVIDIA answers, the DINOv3 License (design:
docs/roadmap/features/detection-pipeline/detection-pipeline-design.md, section 4). They are downloaded from the
``nvidia/GEM-X`` Hugging Face repository into the model directory after the user has been shown the text of both
licences and has accepted them. The acceptance is stored beside the files, with the hash of each text shown, so a
changed text asks again.

``POSETRAK_MODEL_DIR`` moves the model directory (default ``~/.posetrak/models``). ``POSETRAK_SOMA77_MODEL``
names an ``onnx/vitpose.onnx`` the user obtained and licensed on their own; it bypasses the download and the
acceptance record.
"""
from __future__ import annotations

import datetime
import hashlib
import html.parser
import json
import logging
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
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
    url: str


LICENCES = (
    Licence(
        "nvidia-open-model-license", "NVIDIA Open Model License Agreement",
        "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/",
    ),
    Licence(
        "dinov3-license", "DINOv3 License Agreement",
        "https://ai.meta.com/resources/models-and-libraries/dinov3-license/",
    ),
)


class LicenceDeclined(Exception):
    """The user did not accept the licences, or they could not be shown."""


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


def accepted_licences() -> list[dict]:
    """The stored acceptance records (id, text_sha256, accepted_at); [] when none or unreadable."""
    try:
        record = json.loads(_acceptance_file().read_text(encoding="utf-8"))
        have = {entry["id"] for entry in record["licences"]}
        return record["licences"] if {lic.id for lic in LICENCES} <= have else []
    except (OSError, ValueError, KeyError, TypeError):
        return []


class _Text(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        self._skip += tag in ("script", "style")

    def handle_endtag(self, tag):
        self._skip -= tag in ("script", "style")
        if tag in ("p", "li", "h1", "h2", "h3", "h4", "div", "br"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def fetch_licence_text(licence: Licence, timeout: float = 30.0) -> str:
    """The text of *licence* from its published page; raises OSError when it cannot be read as text."""
    request = urllib.request.Request(licence.url, headers={"User-Agent": "posetrak"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8", errors="replace")
    parser = _Text()
    parser.feed(body)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    text = "\n".join(line for line in lines if line)
    if len(text) < 2000:
        raise OSError(f"{licence.url} did not return a readable licence text")
    return text


AcceptCallback = Callable[[list[tuple[Licence, str]]], bool]


def download_gemx(
    accept: AcceptCallback,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Show both licence texts to *accept*, and on acceptance download the weights; returns the ONNX path.

    *accept* receives ``[(licence, text), ...]`` and returns whether the user accepted all of them. Raises
    :class:`LicenceDeclined` when it returns False or a text cannot be fetched (a licence that cannot be shown
    cannot be accepted), and the usual network errors from the download.
    """
    texts = []
    for licence in LICENCES:
        try:
            texts.append((licence, fetch_licence_text(licence)))
        except OSError as exc:
            raise LicenceDeclined(
                f"The text of the {licence.name} could not be fetched ({exc}); read it at {licence.url}"
            ) from exc
    if not accept(texts):
        raise LicenceDeclined("The licences of the SOMA-77 pose model were not accepted.")

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
             "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            for lic, text in texts
        ],
    }
    _acceptance_file().write_text(json.dumps(record, indent=1), encoding="utf-8")
    path = soma77_model_path()
    assert path is not None
    return path
