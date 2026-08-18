# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""video_probe.py — Extract metadata from a video file.

Two-level probe:
1. **cv2.VideoCapture** — always available; provides width, height, container
   fps, and frame count.
2. **exiftool** (optional, via pyexiftool) — provides camera make/model,
   serial number, lens info, actual capture fps (e.g. Android slow-mo), and
   firmware version.  Gracefully disabled when exiftool is not installed.

Insta360 cameras embed their model/serial in a non-standard MP4 atom that
only exiftool's ``-ExtractEmbedded`` pass finds.  Large files (>4 GB atoms)
require ``-api LargeFileSupport=1``.  Both flags are always applied.

Usage::

    result = probe_video(Path("/path/to/clip.mp4"))
    print(result.make, result.model, result.serial_number)
    print(result.capture_fps, result.mode_hint)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VideoProbeResult:
    """Metadata extracted from a video file.

    Fields that require exiftool are ``None`` when exiftool is unavailable
    or when the camera does not embed the corresponding tag.
    """

    # --- always present (cv2) ---
    width: int
    height: int
    container_fps: float     # fps reported by the container header
    frame_count: int

    # --- exiftool-enriched (None when unavailable) ---
    capture_fps: float | None = None
    """Actual capture fps.  Differs from ``container_fps`` for slow-motion
    files (e.g. Google Pixel shoots at 120 fps but stores 30 fps in the
    container for slow-mo playback)."""

    make: str | None = None
    model: str | None = None
    serial_number: str | None = None
    lens_model: str | None = None
    lens_serial: str | None = None
    focal_length_mm: float | None = None
    field_of_view: str | None = None    # GoPro: 'Linear', 'Wide', etc.
    codec: str | None = None            # 'hvc1', 'avc1', etc.
    firmware: str | None = None
    created_at: str | None = None       # ISO-ish datetime string from video

    mode_hint: str | None = None
    """Best-effort human-readable mode description, e.g.
    ``"Linear 4K 120fps"`` or ``"EF17-40mm 17mm 4K 60fps"``."""

    exiftool_available: bool = False
    raw_exiftool: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> VideoProbeResult:
    """Return a ``VideoProbeResult`` for *path*.

    Always performs the cv2 probe.  Attempts an exiftool probe if exiftool
    is available (both the ``pyexiftool`` package and the system binary).
    """
    result = _probe_cv2(path)
    _try_enrich_with_exiftool(path, result)
    return result


def exiftool_available() -> bool:
    """Return ``True`` if both ``pyexiftool`` and the system ``exiftool``
    binary are available."""
    try:
        import exiftool  # noqa: F401
        import shutil
        return shutil.which("exiftool") is not None
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# cv2 probe
# ---------------------------------------------------------------------------


def _probe_cv2(path: Path) -> VideoProbeResult:
    cap = cv2.VideoCapture(str(path))
    try:
        width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps        = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    return VideoProbeResult(
        width=width,
        height=height,
        container_fps=fps,
        frame_count=frame_count,
    )


# ---------------------------------------------------------------------------
# exiftool probe
# ---------------------------------------------------------------------------

# Tags that appear with a group prefix in pyexiftool output (e.g.
# "QuickTime:VideoFrameRate").  We build a stripped lookup on the fly.

def _strip_groups(raw: dict) -> dict:
    """Return a copy of *raw* with group prefixes removed from keys.

    When two tags share the same unqualified name, the first one wins
    (preserves priority from exiftool's output order).
    """
    out: dict = {}
    for k, v in raw.items():
        simple = k.split(":")[-1]
        out.setdefault(simple, v)
    return out


def _get(tags: dict, *keys: str) -> str | None:
    """Return the first non-empty value for any of *keys*, or ``None``."""
    for k in keys:
        v = tags.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def _parse_focal_length(raw: str | None) -> float | None:
    if raw is None:
        return None
    m = re.search(r"[\d.]+", raw)
    return float(m.group()) if m else None


def _infer_make(tags: dict, model: str | None) -> str | None:
    """Infer manufacturer when no ``Make`` tag is present."""
    # Canon / standard EXIF cameras → already in Make
    # GoPro embeds maker name in CompressorName or Model prefix
    comp = tags.get("CompressorName", "") or ""
    if "gopro" in comp.lower():
        return "GoPro"
    if model:
        ml = model.lower()
        if ml.startswith("hero") or "gopro" in ml:
            return "GoPro"
        if "insta360" in ml:
            return "Insta360"
        if "pixel" in ml:
            return "Google"
    return None


def _build_mode_hint(tags: dict, result: VideoProbeResult) -> str | None:
    """Construct a human-readable mode string from available tags."""
    fps = result.capture_fps or result.container_fps
    dims = f"{result.width}×{result.height}"
    fps_str = f"{fps:.0f}fps" if fps == int(fps) else f"{fps:.2f}fps"

    make_l = (result.make or "").lower()

    if make_l == "gopro":
        fov = result.field_of_view or ""
        parts = [p for p in [fov, dims, fps_str] if p]
        return " ".join(parts) or None

    if make_l == "canon":
        lens = result.lens_model or ""
        fl = f"{result.focal_length_mm:.0f}mm" if result.focal_length_mm else ""
        parts = [p for p in [lens, fl, dims, fps_str] if p]
        return " ".join(parts) or None

    if make_l == "google":
        # Annotate slow-mo if capture fps ≠ container fps
        if (result.capture_fps and result.container_fps
                and abs(result.capture_fps - result.container_fps) > 1):
            return f"{dims} {fps_str} capture (slow-mo {result.container_fps:.0f}fps)"

    # Generic fallback
    return f"{dims} {fps_str}"


def _try_enrich_with_exiftool(path: Path, result: VideoProbeResult) -> None:
    """Attempt to enrich *result* in-place using exiftool; silently skip on
    any failure (missing binary, import error, parse error)."""
    try:
        import exiftool
    except ImportError:
        return

    try:
        with exiftool.ExifToolHelper() as et:
            meta_list = et.get_metadata(
                [str(path)],
                params=["-api", "LargeFileSupport=1", "-ee"],
            )
    except Exception:  # noqa: BLE001
        return

    if not meta_list:
        return

    tags = _strip_groups(meta_list[0])
    result.raw_exiftool = tags
    result.exiftool_available = True

    # Make / model
    model   = _get(tags, "Model", "AndroidModel")
    make    = _get(tags, "Make", "AndroidManufacturer") or _infer_make(tags, model)
    result.make  = make
    result.model = model

    # Serial number — prefer camera body serial over lens
    result.serial_number = _get(tags,
        "CameraSerialNumber",  # GoPro
        "SerialNumber",        # Canon, Insta360, others
    )

    # Lens
    result.lens_model       = _get(tags, "LensModel", "Lens")
    result.lens_serial      = _get(tags, "LensSerialNumber")
    result.focal_length_mm  = _parse_focal_length(_get(tags, "FocalLength"))
    result.field_of_view    = _get(tags, "FieldOfView")

    # Codec / technical
    result.codec      = _get(tags, "CompressorID")
    result.created_at = _get(tags, "SubSecDateTimeOriginal", "DateTimeOriginal",
                              "CreateDate")

    # Firmware
    raw_fw = _get(tags, "FirmwareVersion", "CanonFirmwareVersion", "Firmware")
    if raw_fw:
        # Strip Canon's "Firmware Version " prefix
        result.firmware = re.sub(r"^(?:firmware\s+version\s+)", "", raw_fw,
                                 flags=re.IGNORECASE).strip()

    # Capture fps — prefer Android metadata (real capture rate for slow-mo)
    android_fps = tags.get("AndroidCaptureFPS")
    et_fps = tags.get("VideoFrameRate")
    if android_fps is not None:
        try:
            result.capture_fps = float(android_fps)
        except (TypeError, ValueError):
            pass
    elif et_fps is not None:
        try:
            result.capture_fps = float(et_fps)
        except (TypeError, ValueError):
            pass

    result.mode_hint = _build_mode_hint(tags, result)
