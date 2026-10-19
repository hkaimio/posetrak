# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Safe camera relabelling for a capture.

A wrong camera label is fixed by changing which camera each video belongs to,
never by swapping file paths between ``capture_videos`` rows: sync points hang
off the video row (``sync_points.shot_video_id``), so moving a path leaves the
synchronisation with the wrong file.

``reassign_cameras`` changes ``capture_videos.camera_instance_id`` (and, when
the new camera is of another model, the camera mode and the intrinsics
calibration that follows it) and, in the video's sync points, the
``camera_instance_id`` column where it holds the video's camera. (Sync points
written by the synchronisation dialog hold the video id there instead; those
stay as they are.) File paths, frame ranges and sync points otherwise stay with
their videos.

Detection and tracking results carry camera ids of their own and are not
rewritten; the operation therefore refuses when any exist for the capture,
unless the caller confirms that they will be recomputed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class CameraReassignError(ValueError):
    """The reassignment is invalid or refused; nothing was changed."""


@dataclass
class Assignment:
    """Move *video_id* to *camera_instance_id*.

    *camera_mode_id* is only needed when the new camera is of another model and the model
    has more than one camera mode.
    """
    video_id: str
    camera_instance_id: str
    camera_mode_id: str | None = None


def runs_referencing_capture(conn: sqlite3.Connection, capture_id: str) -> dict[str, int]:
    """Counts of the detection runs, pose observation sequences and tracking runs of a capture."""
    q = lambda sql: conn.execute(sql, (capture_id,)).fetchone()[0]  # noqa: E731
    return {
        "detection runs": q("SELECT COUNT(*) FROM detection_runs WHERE shot_id = ?"),
        "pose observation sequences": q(
            "SELECT COUNT(*) FROM pose_observation_sequences WHERE shot_id = ?"),
        "tracking runs": q(
            "SELECT COUNT(*) FROM tracking_runs tr "
            "JOIN pose_observation_sequences s ON s.id = tr.observation_sequence_id "
            "WHERE s.shot_id = ?"),
    }


def _mode_and_intrinsics(
    conn: sqlite3.Connection, camera_id: str, mode_id: str | None,
) -> tuple[str, str]:
    """The camera mode of the camera's model and the intrinsics calibration that follows it."""
    modes = conn.execute(
        "SELECT cm.id, cm.default_intrinsics_calibration_id FROM camera_instances ci "
        "JOIN camera_modes cm ON cm.camera_model_id = ci.camera_model_id WHERE ci.id = ?",
        (camera_id,),
    ).fetchall()
    if mode_id is not None:
        modes = [m for m in modes if m[0] == mode_id]
    if len(modes) != 1:
        raise CameraReassignError(
            f"Camera {camera_id} has {len(modes)} matching camera modes; choose one explicitly"
        )
    mode, intrinsics = modes[0]
    if intrinsics is None:
        row = conn.execute(
            "SELECT id FROM intrinsics_calibrations WHERE camera_mode_id = ? "
            "ORDER BY calibrated_at DESC LIMIT 1", (mode,),
        ).fetchone()
        intrinsics = row[0] if row else None
    if intrinsics is None:
        raise CameraReassignError(f"Camera mode {mode} has no intrinsics calibration")
    return mode, intrinsics


def reassign_cameras(
    conn: sqlite3.Connection,
    capture_id: str,
    assignments: list[Assignment],
    *,
    confirm_recompute: bool = False,
) -> list[str]:
    """Apply *assignments* to a capture's videos in one transaction.

    Several videos may exchange cameras in one call (a swap). Returns warnings for the
    caller to show; raises ``CameraReassignError`` without changing anything when the
    result would give two videos of the capture the same camera, an id is unknown, or
    results exist and *confirm_recompute* is false.
    """
    videos = {
        r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT id, camera_instance_id, camera_mode_id FROM capture_videos WHERE shot_id = ?",
            (capture_id,))
    }
    final = {vid: cam for vid, (cam, _) in videos.items()}
    for a in assignments:
        if a.video_id not in videos:
            raise CameraReassignError(f"Video {a.video_id} is not in capture {capture_id}")
        if conn.execute("SELECT 1 FROM camera_instances WHERE id = ?", (a.camera_instance_id,)).fetchone() is None:
            raise CameraReassignError(f"Unknown camera {a.camera_instance_id}")
        final[a.video_id] = a.camera_instance_id
    if len(set(final.values())) != len(final):
        raise CameraReassignError("Two videos of the capture would have the same camera")

    changes = [a for a in assignments if a.camera_instance_id != videos[a.video_id][0]]
    if not changes:
        return []
    existing = {k: n for k, n in runs_referencing_capture(conn, capture_id).items() if n}
    if existing and not confirm_recompute:
        what = ", ".join(f"{n} {k}" for k, n in existing.items())
        raise CameraReassignError(
            f"The capture already has {what}, which keep the old camera ids. "
            f"Confirm that they will be recomputed to reassign cameras."
        )

    updates = []
    for a in changes:
        old_model = conn.execute(
            "SELECT camera_model_id FROM camera_instances WHERE id = ?", (videos[a.video_id][0],)).fetchone()
        new_model = conn.execute(
            "SELECT camera_model_id FROM camera_instances WHERE id = ?", (a.camera_instance_id,)).fetchone()
        if a.camera_mode_id is None and old_model == new_model:
            updates.append((a, None, None))  # same model: mode and intrinsics stay
        else:
            updates.append((a, *_mode_and_intrinsics(conn, a.camera_instance_id, a.camera_mode_id)))

    with conn:
        # sync_points is keyed by (config, camera, frame): park the moved rows on unique
        # placeholders first so that exchanging two cameras cannot collide.
        for a, _, _ in updates:
            conn.execute(
                "UPDATE sync_points SET camera_instance_id = ? "
                "WHERE shot_video_id = ? AND camera_instance_id = ?",
                (f"reassigning:{a.video_id}", a.video_id, videos[a.video_id][0]))
        for a, mode, intrinsics in updates:
            conn.execute("UPDATE sync_points SET camera_instance_id = ? WHERE camera_instance_id = ?",
                         (a.camera_instance_id, f"reassigning:{a.video_id}"))
            if mode is None:
                conn.execute("UPDATE capture_videos SET camera_instance_id = ? WHERE id = ?",
                             (a.camera_instance_id, a.video_id))
            else:
                conn.execute(
                    "UPDATE capture_videos SET camera_instance_id = ?, camera_mode_id = ?, "
                    "intrinsics_calibration_id = ? WHERE id = ?",
                    (a.camera_instance_id, mode, intrinsics, a.video_id))
            conn.execute(
                "INSERT OR IGNORE INTO session_cameras (session_id, camera_instance_id) "
                "SELECT session_id, ? FROM captures WHERE id = ?", (a.camera_instance_id, capture_id))

    warnings = []
    cal = conn.execute("SELECT extrinsic_calibration_id FROM captures WHERE id = ?", (capture_id,)).fetchone()
    if cal and cal[0]:
        warnings.append(
            f"The capture is linked to extrinsic calibration {cal[0]}, whose entries belong to the "
            f"cameras by id: check that they still match the videos, or calibrate again."
        )
    if existing:
        warnings.append("Recompute the detection and tracking results of this capture.")
    return warnings
