"""timeline_status.py — per-(keypoint, frame) status for the keypoint-editing timeline.

Status has two independent axes (see
docs/roadmap/features/keypoint-editing/keypoint-editing-design.md,
"Improvements" section, *Timeline view*):

Axis 1 (implemented here) — edit state: stable, independent of any tracking
run.  One of the STATUS_* codes below, in ascending display precedence.

Axis 2 — last tracking-run outlier verdict, rendered as a separate overlay
once a tracking run is selected.  Not implemented yet (deferred, no phase
number assigned in the design doc).
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict

import numpy as np

from posetrak.db.observation_merge import (
    infer_body_width,
    merge_observation_sources,
    refined_indices,
)

# Ascending precedence: when aggregating several keypoints/cameras into one
# cell, the maximum code wins (GREY > BLUE > ORANGE > YELLOW > GREEN).
STATUS_GREEN = 0   # original detection, inside person segmentation (or no segmentation data)
STATUS_YELLOW = 1  # original detection, outside person segmentation
STATUS_ORANGE = 2  # from a '<base>.refined' source (Idea 3: automated post-edit
                   # redetection) -- not yet human-verified
STATUS_BLUE = 3    # edited/moved, or explicitly kept as a keyframe (not disabled)
STATUS_GREY = 4    # disabled (user-forced outlier), or no usable detection at all

# quality_blob values: 1.0=inside, 0.5=boundary, 0.0=outside, -1.0=unavailable.
# Boundary counts as "inside enough" (not flagged yellow); unavailable (-1)
# falls through to the GREEN default, same as no segmentation run at all.
_SEG_INSIDE_THRESHOLD = 0.5


def read_timeline_status(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    shot_video_id: str | None = None,
    seg_run_id: str | None = None,
    track_id_by_frame: dict[int, int] | None = None,
) -> dict[int, np.ndarray]:
    """Return {video_frame: int8[N]} axis-1 status codes for one camera.

    `shot_video_id` / `seg_run_id` / `track_id_by_frame` are optional: when
    any is missing, segmentation status is unavailable and original
    detections default to STATUS_GREEN (segmentation is a refinement signal,
    not a requirement — see *Status signal* in the design doc).
    """
    obs_rows = session.execute(
        "SELECT video_frame, source, kp_blob FROM pose_observations"
        " WHERE sequence_id = ? AND camera_instance_id = ?"
        " ORDER BY video_frame",
        (sequence_id, camera_instance_id),
    ).fetchall()

    edit_rows = session.execute(
        "SELECT video_frame, kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id = ? AND camera_instance_id = ?"
        " ORDER BY video_frame",
        (sequence_id, camera_instance_id),
    ).fetchall()
    edits: dict[int, tuple[np.ndarray, bytes]] = {
        r["video_frame"]: (
            np.frombuffer(bytes(r["kp_blob"]), dtype=np.float32).reshape(-1, 3),
            bytes(r["kp_mask"]),
        )
        for r in edit_rows
    }

    by_frame_rows: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for r in obs_rows:
        kp = np.frombuffer(bytes(r["kp_blob"]), dtype=np.float32).reshape(-1, 3)
        by_frame_rows[r["video_frame"]].append((r["source"], kp))

    default_width = infer_body_width(by_frame_rows.values())
    if default_width is None and edits:
        default_width = next(iter(edits.values()))[0].shape[0]

    obs_by_frame: dict[int, np.ndarray] = {}
    refined_by_frame: dict[int, frozenset[int]] = {}
    for frame, rows in by_frame_rows.items():
        merged = merge_observation_sources(rows, default_width=default_width)
        obs_by_frame[frame] = merged if merged is not None else rows[0][1]
        refined_by_frame[frame] = refined_indices(rows)

    frames = sorted(set(obs_by_frame) | set(edits))
    if not frames:
        return {}

    if obs_by_frame:
        n_kp = next(iter(obs_by_frame.values())).shape[0]
    else:
        n_kp = next(iter(edits.values()))[0].shape[0]

    seg_available = (
        shot_video_id is not None and seg_run_id is not None and track_id_by_frame is not None
    )

    result: dict[int, np.ndarray] = {}
    for frame in frames:
        status = np.full(n_kp, STATUS_GREY, dtype=np.int8)

        obs = obs_by_frame.get(frame)
        if obs is not None:
            status[obs[:, 2] > 0.0] = STATUS_GREEN

        if seg_available:
            track_id = track_id_by_frame.get(frame)
            if track_id is not None:
                quality = _read_quality(session, seg_run_id, shot_video_id, frame, track_id, n_kp)
                if quality is not None:
                    outside = (quality >= 0.0) & (quality < _SEG_INSIDE_THRESHOLD)
                    status[(status == STATUS_GREEN) & outside] = STATUS_YELLOW

        # Idea 3: a slot backed by a '<base>.refined' source is flagged
        # ORANGE regardless of its GREEN/YELLOW sub-state above -- "this came
        # from automated redetection, review it" is a more specific signal
        # than the segmentation-quality flag, but still ranks below an
        # actual human edit (checked next).
        for i in refined_by_frame.get(frame, ()):
            if i < n_kp and status[i] in (STATUS_GREEN, STATUS_YELLOW):
                status[i] = STATUS_ORANGE

        edit = edits.get(frame)
        if edit is not None:
            edit_kp, mask = edit
            if edit_kp.shape[0] == n_kp:
                for i in range(n_kp):
                    byte_idx, bit_idx = divmod(i, 8)
                    if byte_idx < len(mask) and (mask[byte_idx] >> bit_idx) & 1:
                        status[i] = STATUS_GREY if edit_kp[i, 2] != 0.0 else STATUS_BLUE

        result[frame] = status
    return result


def _read_quality(
    session: sqlite3.Connection,
    seg_run_id: str,
    shot_video_id: str,
    video_frame: int,
    track_id: int,
    n_kp: int,
) -> np.ndarray | None:
    row = session.execute(
        "SELECT quality_blob FROM keypoint_obs_quality"
        " WHERE seg_run_id = ? AND shot_video_id = ? AND video_frame = ? AND track_id = ?",
        (seg_run_id, shot_video_id, video_frame, track_id),
    ).fetchone()
    if row is None:
        return None
    quality = np.frombuffer(bytes(row["quality_blob"]), dtype=np.float32)
    if quality.shape[0] != n_kp:
        return None
    return quality


def compute_inlier_camera_counts(
    obs_kp_by_camera: dict[str, dict[int, np.ndarray]],
) -> dict[int, np.ndarray]:
    """Return {video_frame: int16[N]} count of cameras with confidence > 0, per keypoint.

    Pure aggregation over already-merged per-camera observation dicts (as
    returned by `read_observations_with_edits`); no DB access — the caller
    already has this data loaded for display.
    """
    counts: dict[int, np.ndarray] = {}
    for frame_kp in obs_kp_by_camera.values():
        for frame, kp in frame_kp.items():
            acc = counts.setdefault(frame, np.zeros(kp.shape[0], dtype=np.int16))
            acc += (kp[:, 2] > 0.0).astype(np.int16)
    return counts
