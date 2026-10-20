# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""frame_source.py — Sequential BGR frame decoding with rotation correction.

Extracted from ``pipeline.py`` so a later pass over the same run (e.g. the
hand-detection refinement stage in ``hand_refinement.py``) can re-decode the
exact frames a detection run's keypoints were built from, without
duplicating the pyav/cv2 rotation-handling logic.
"""
from __future__ import annotations

import logging
import math
import struct

import cv2
import numpy as np

_log = logging.getLogger(__name__)

try:
    import av as _av  # noqa: F401
    _AV_AVAILABLE = True
except ImportError:
    _AV_AVAILABLE = False


def _stream_rotation(stream) -> int:
    """Return clockwise rotation degrees from stream metadata (older cameras)."""
    rotate_str = (stream.metadata or {}).get("rotate", "0") or "0"
    try:
        return int(rotate_str) % 360
    except (ValueError, TypeError):
        return 0


def _parse_displaymatrix(data: bytes) -> int:
    """Parse clockwise rotation from a DISPLAYMATRIX side-data blob.

    Modern Android phones (Pixel 7+, OnePlus 10 Pro confirmed 2026-08-15)
    store rotation as a 3×3 fixed-point matrix in frame side data rather
    than a plain 'rotate' metadata tag. The blob is 36 bytes: nine 32-bit
    little-endian integers (16.16 fixed).

    No negation here, unlike FFmpeg's own ``av_display_rotation_get()``
    (which returns degrees to rotate *counter*-clockwise) -- this function
    returns clockwise degrees directly, matching ``_apply_rotation``'s own
    convention. A stray negation here (matching FFmpeg's CCW convention
    literally, then treating the result as clockwise degrees) previously
    produced exactly 180° of extra rotation -- confirmed against a real
    OnePlus 10 Pro portrait capture whose matrix is
    ``[0, 65536, 0, -65536, 0, 0, 0, 0, 1<<30]``: the negated formula
    computed 270° clockwise, which is empirically upside-down (verified by
    comparing against the same frame read via a backend that auto-rotates
    correctly); the un-negated formula below computes the correct 90°.
    """
    if len(data) < 36:
        return 0
    m = struct.unpack("<9i", data[:36])
    scale_x = math.hypot(m[0], m[3])
    scale_y = math.hypot(m[1], m[4])
    if scale_x == 0 or scale_y == 0:
        return 0
    return round(math.atan2(m[1] / scale_y, m[0] / scale_x) * 180 / math.pi) % 360


def _apply_rotation(img: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate *img* clockwise by *degrees* (must be 0/90/180/270)."""
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# A target frame this close ahead of the decoder is reached by decoding on rather than seeking:
# a seek decodes forward from the preceding keyframe, on average half a GOP (60 frames for the
# 120-frame GOPs of the GoPro cameras).
_MAX_DECODE_AHEAD = 60


def iter_frames(path: str, first_frame: int, last_frame: int, step: int = 1):
    """Yield (video_frame_index, bgr_array) for frames in [first_frame, last_frame).

    With *step* > 1 only every step-th frame, counted from *first_frame*, is yielded; the
    others are decoded but not converted to BGR.
    """
    _log.debug("iter_frames: opening %s (av=%s)", path, _AV_AVAILABLE)
    if _AV_AVAILABLE:
        yield from _iter_frames_av(path, first_frame, last_frame, step)
    else:
        yield from _iter_frames_cv2(path, first_frame, last_frame, step)
    _log.debug("iter_frames: closed %s", path)


def _open_av_stream(container):
    """Configure the first video stream; return (stream, time_base, container_fps, rotation)."""
    stream = container.streams.video[0]
    # NOT "AUTO": callers decode only part of the file and then close the container, i.e.
    # always close *before* the decoder reaches the file's real EOF.
    # With threaded decoding enabled, that close (avcodec_free_context())
    # can hang indefinitely waiting on an FFmpeg-internal decode worker
    # thread that's still mid-frame and never gets flushed -- confirmed
    # live via py-spy: the main decode thread parked in
    # avcodec_free_context()/avpriv_split_xiph_headers on a condition
    # variable, an FFmpeg-spawned worker thread (visible as its own OS
    # thread, created via beginthreadex) stuck inside av_parser_iterate.
    # Single-threaded decode has no such worker thread to leak, so
    # closing early is always safe -- the seek in _seek_av() already avoids
    # most of the cost multi-threaded decode would have saved anyway.
    stream.thread_type = "NONE"
    time_base = float(stream.time_base)
    # Use the container's own fps for seek/pts arithmetic — never
    # actual_fps from the DB, which may reflect real-world capture rate
    # (e.g. 120 for slow-motion) rather than the container's pts cadence.
    container_fps = float(stream.average_rate)

    # Rotation correction: PyAV does not apply container rotation
    # automatically in to_ndarray().  Try stream metadata first (older
    # cameras); fall back to DISPLAYMATRIX in frame side data (Pixel 7+).
    rotation = _stream_rotation(stream)
    rotation_source = "metadata"
    if rotation == 0:
        for probe_frame in container.decode(stream):
            for sd in (probe_frame.side_data or []):
                if "DISPLAYMATRIX" in str(sd.type).upper():
                    rotation = _parse_displaymatrix(bytes(sd))
                    rotation_source = "DISPLAYMATRIX"
            break
        container.seek(0, stream=stream, backward=True, any_frame=False)
    if rotation:
        _log.info("_open_av_stream: rotate=%d° from %s", rotation, rotation_source)
    return stream, time_base, container_fps, rotation


def _seek_av(container, stream, time_base: float, container_fps: float, first_frame: int) -> None:
    """Position *container* at the keyframe at or before *first_frame*."""
    if first_frame > 0:
        seek_s = max(0.0, (first_frame - 1) / container_fps)
        container.seek(int(seek_s / time_base), stream=stream, backward=True, any_frame=False)


def _iter_frames_av(path: str, first_frame: int, last_frame: int, step: int):
    import av
    _log.debug("_iter_frames_av: av.open(%s)", path)
    with av.open(path) as container:
        _log.debug("_iter_frames_av: opened ok")
        stream, time_base, container_fps, rotation = _open_av_stream(container)
        _seek_av(container, stream, time_base, container_fps, first_frame)

        frame_idx: int | None = None
        for av_frame in container.decode(stream):
            if av_frame.pts is None:
                continue
            if frame_idx is None:
                frame_s = float(av_frame.pts) * time_base
                pts_idx = round(frame_s * container_fps)
                if pts_idx < first_frame:
                    continue
                frame_idx = first_frame
            if frame_idx >= last_frame:
                break
            # The colour conversion costs about as much as the decode at 4K, so frames the
            # caller does not want are decoded (they are references of later ones) but not
            # converted.
            if (frame_idx - first_frame) % step == 0:
                img = av_frame.to_ndarray(format="bgr24")
                if rotation:
                    img = _apply_rotation(img, rotation)
                yield frame_idx, img
            frame_idx += 1


def iter_frames_at(path: str, frame_indices):
    """Yield (frame_index, bgr_array) at *frame_indices* of one video, through one open container.

    Indices are visited in ascending order. A target shortly after the previous one is reached
    by decoding on; a farther one by a seek, which decodes forward from the preceding keyframe.
    Frames are labelled as ``iter_frames`` labels them: the first frame at or after the
    target's presentation index is the target. An index past the end of the video is skipped.
    """
    wanted = sorted(set(frame_indices))
    if not _AV_AVAILABLE:
        for idx in wanted:
            yield from _iter_frames_cv2(path, idx, idx + 1)
        return
    import av
    with av.open(path) as container:
        stream, time_base, container_fps, rotation = _open_av_stream(container)
        decoder = None
        last_pts_idx: int | None = None  # presentation index of the last frame decoded
        for target in wanted:
            if decoder is None or last_pts_idx is None or not 0 < target - last_pts_idx <= _MAX_DECODE_AHEAD:
                _seek_av(container, stream, time_base, container_fps, target)
                decoder = container.decode(stream)
                last_pts_idx = None
            for av_frame in decoder:
                if av_frame.pts is None:
                    continue
                last_pts_idx = round(float(av_frame.pts) * time_base * container_fps)
                if last_pts_idx >= target:
                    img = av_frame.to_ndarray(format="bgr24")
                    yield target, _apply_rotation(img, rotation) if rotation else img
                    break
            else:
                decoder = None  # end of video


def _iter_frames_cv2(path: str, first_frame: int, last_frame: int, step: int = 1):
    cap = cv2.VideoCapture(path)
    # cv2 does not auto-rotate on most backends; read the orientation flag
    # and apply it manually so the fallback path behaves the same as av.
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    if rotation:
        _log.info("_iter_frames_cv2: %s rotate=%d° from metadata", path, rotation)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
    frame_idx = first_frame
    while frame_idx < last_frame:
        ok, img = cap.read()
        if not ok:
            break
        if (frame_idx - first_frame) % step == 0:
            yield frame_idx, _apply_rotation(img, rotation) if rotation else img
        frame_idx += 1
    cap.release()
