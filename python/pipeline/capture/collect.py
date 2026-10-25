# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Copying recordings off the cameras.

Each copied file is verified against the size the camera announced and gets a
JSON file next to it with the camera it came from, its path on the camera, its
SHA-256, and whatever metadata the camera keeps about the recording. A file that
was copied before is not copied again, and a copy that was interrupted resumes.

Layout under the destination: ``<camera label>/<camera folder>/<file name>``,
with ``<file name>.json`` beside it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pipeline.capture.devices import CameraDriver, MediaFile, OffloadError
from posetrak.db.capture_control import utc_now_iso

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = (".mp4", ".mov")


def is_video(file: MediaFile) -> bool:
    return file.name.lower().endswith(VIDEO_SUFFIXES)


@dataclass(frozen=True)
class OffloadedFile:
    """What happened to one recording."""

    file: MediaFile
    path: Path
    size: int
    sha256: str | None
    #: True when the file had been copied before and was left alone
    skipped: bool = False


def target_path(dest_root: Path, label: str, file: MediaFile) -> Path:
    return dest_root.joinpath(label, *file.path.split("/"))


def sidecar_path(target: Path) -> Path:
    return target.with_name(target.name + ".json")


def already_copied(target: Path, file: MediaFile) -> bool:
    """Whether *target* is a complete copy that was recorded in its sidecar."""
    if not target.exists() or not sidecar_path(target).exists():
        return False
    size = target.stat().st_size
    return size == file.size if file.size is not None else size > 0


async def offload_camera(
    driver: CameraDriver,
    dest_root: Path,
    *,
    files: list[MediaFile] | None = None,
    include_all: bool = False,
    last: int | None = None,
    progress: Callable[[MediaFile, int, int | None], None] | None = None,
) -> list[OffloadedFile]:
    """Copy a camera's recordings to *dest_root*.

    Parameters
    ----------
    files:
        The recordings to copy; the whole media list when None.
    include_all:
        Copy photos and other files too, not only videos.
    last:
        Copy only this many of the most recent recordings.
    progress:
        Called with ``(file, bytes done, bytes total)`` while a file is copied.

    Raises
    ------
    OffloadError
        If the media cannot be listed or a file cannot be copied.
    """
    chosen = files if files is not None else await driver.list_media()
    if not include_all:
        chosen = [f for f in chosen if is_video(f)]
    chosen = sorted(chosen, key=lambda f: (f.created or 0, f.path))
    if last is not None:
        chosen = chosen[-last:] if last > 0 else []
    results = []
    for file in chosen:
        target = target_path(dest_root, driver.label, file)
        if already_copied(target, file):
            results.append(OffloadedFile(file, target, target.stat().st_size, None, skipped=True))
            continue
        info = await driver.media_info(file)

        def report(done: int, total: int | None, file: MediaFile = file) -> None:
            if progress is not None:
                progress(file, done, total)

        copied = await driver.download(file, target, report)
        sidecar_path(target).write_text(
            json.dumps(
                {
                    "camera": driver.label,
                    "camera_path": file.path,
                    "size": copied.size,
                    "sha256": copied.sha256,
                    "created": file.created,
                    "copied_at": utc_now_iso(),
                    "camera_info": info,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("%s: copied %s (%d bytes)", driver.label, file.path, copied.size)
        results.append(OffloadedFile(file, copied.path, copied.size, copied.sha256))
    return results


__all__ = ["OffloadError", "OffloadedFile", "already_copied", "is_video", "offload_camera", "sidecar_path", "target_path"]
