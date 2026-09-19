# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""object_marker_run.py — give one tracked object its own marker run out of a shared one.

Decoding the video is the expensive part of coded-marker detection, so several
props are best detected together: one run over the ids of all of them, bound to
no object. A tracked object needs a run of its own, though, because
``detection_runs.capture_object_id`` is what links its observation sequence to
the object, and its corner slots follow the object's own marker body.

:func:`derive_object_marker_run` makes that run without touching the shared one
(detection runs are append-only): a new object-bound run whose corner rows are the
shared run's, re-slotted to the object's marker ids. The run records where it came
from in ``config_json["derived_from_detection_run_id"]``. Finalise it with
``finalise_object_to_db`` like any other object-bound run.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np


def derive_object_marker_run(
    session: sqlite3.Connection,
    detection_run_id: str,
    capture_object_id: str,
    *,
    with_dots: bool = False,
) -> str:
    """Create an object-bound marker run from a shared marker run.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    detection_run_id:
        A marker run (``detector_type='aruco'``) of the object's capture.
    capture_object_id:
        The object the new run is bound to. Its marker body gives the marker ids
        and their slot order.
    with_dots:
        Also copy the shared run's reflective-dot candidates, for an object whose
        body carries dots. They are scene-wide, so every object that asks for
        them gets its own copy.

    Returns
    -------
    str
        The new run's ID.

    Raises
    ------
    ValueError
        If the run or object does not exist, they belong to different captures,
        the object's body has no coded markers, the run has none of them, or the
        object already has a run derived from this one.
    """
    from app.pose.db_cache import (
        DOT_REGION_TYPE,
        DOT_TRACK_ID,
        MARKER_REGION_TYPE,
        MARKER_TRACK_ID,
        create_marker_detection_run,
        mark_run_complete,
    )
    from app.setup.fiducial_markers import load_marker_body_yaml

    run = session.execute(
        "SELECT shot_id, sync_config_id, trial_id, time_start_s, time_end_s, detector_type, config_json "
        "FROM detection_runs WHERE id = ?",
        (detection_run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"detection_runs row not found: {detection_run_id!r}")
    if run["detector_type"] != "aruco":
        raise ValueError(f"detection run {detection_run_id!r} is a {run['detector_type']!r} run, not a marker run")

    obj = session.execute(
        "SELECT capture_id, name, marker_body_definition_id FROM capture_objects WHERE id = ?",
        (capture_object_id,),
    ).fetchone()
    if obj is None:
        raise ValueError(f"capture_objects row not found: {capture_object_id!r}")
    if obj["capture_id"] != run["shot_id"]:
        raise ValueError(f"object '{obj['name']}' and detection run {detection_run_id!r} belong to different captures")

    already = session.execute(
        "SELECT id FROM detection_runs WHERE capture_object_id = ? "
        "AND json_extract(config_json, '$.derived_from_detection_run_id') = ?",
        (capture_object_id, detection_run_id),
    ).fetchone()
    if already is not None:
        raise ValueError(
            f"object '{obj['name']}' already has run {already['id']} derived from this one; "
            "finalise that run instead"
        )

    body_id = obj["marker_body_definition_id"]
    body = session.execute(
        "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (body_id,)
    ).fetchone()
    if body is None or body["yaml_content"] is None:
        raise ValueError(f"marker_body_definitions row not found or empty: {body_id!r}")
    object_ids = list(load_marker_body_yaml(body["yaml_content"], rig_id=body_id).marker_corners)
    if not object_ids:
        raise ValueError(f"the marker body of object '{obj['name']}' has no coded markers")

    config = json.loads(run["config_json"] or "{}")
    run_ids: list[str] = config.get("marker_ids") or []
    if not set(object_ids) & set(run_ids):
        raise ValueError(
            f"detection run {detection_run_id!r} has none of the marker ids of object '{obj['name']}' "
            f"(object: {object_ids}, run: {run_ids})"
        )

    rows = _reslotted_marker_rows(session, detection_run_id, run_ids, object_ids, MARKER_TRACK_ID, MARKER_REGION_TYPE)
    if not rows:
        raise ValueError(f"object '{obj['name']}' is not seen in detection run {detection_run_id!r}")

    new_run_id = create_marker_detection_run(
        session,
        shot_id=run["shot_id"],
        sync_config_id=run["sync_config_id"],
        time_start_s=run["time_start_s"],
        time_end_s=run["time_end_s"],
        dictionary=config.get("dictionary", "DICT_4X4_50"),
        marker_ids=object_ids,
        min_marker_perimeter_rate=config.get("min_marker_perimeter_rate"),
        frame_step=config.get("frame_step", 1),
        trial_id=run["trial_id"],
        capture_object_id=capture_object_id,
        marker_body_definition_id=body_id,
        dot_detection_config=config.get("dot_detection") if with_dots else None,
    )
    new_config = json.loads(session.execute(
        "SELECT config_json FROM detection_runs WHERE id = ?", (new_run_id,)
    ).fetchone()["config_json"])
    new_config["derived_from_detection_run_id"] = detection_run_id
    session.execute(
        "UPDATE detection_runs SET config_json = ? WHERE id = ?", (json.dumps(new_config), new_run_id)
    )
    session.executemany(
        "INSERT INTO detection_keypoints "
        "(detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale) "
        "VALUES (?,?,?,?,?,?,?)",
        [(new_run_id, *row) for row in rows],
    )
    if with_dots:
        session.execute(
            "INSERT INTO detection_keypoints "
            "(detection_run_id, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale) "
            "SELECT ?, shot_video_id, video_frame, track_id, region_type, keypoints, noise_scale "
            "FROM detection_keypoints WHERE detection_run_id = ? AND track_id = ? AND region_type = ?",
            (new_run_id, detection_run_id, DOT_TRACK_ID, DOT_REGION_TYPE),
        )
    mark_run_complete(session, new_run_id)
    return new_run_id


def _reslotted_marker_rows(
    session: sqlite3.Connection,
    detection_run_id: str,
    run_ids: list[str],
    object_ids: list[str],
    track_id: int,
    region_type: str,
) -> list[tuple]:
    """The run's corner rows re-slotted to *object_ids*, without the frames that show none of them.

    A corner blob is float32[4 * len(ids), 3] (x, y, confidence), list-position-major by
    marker id with corners 0-3 within each marker. A marker of the object that the run
    did not look for keeps the occluded-slot convention: NaN position, confidence 0.
    """
    slot_in_run = {marker_id: i for i, marker_id in enumerate(run_ids)}
    out: list[tuple] = []
    for r in session.execute(
        "SELECT shot_video_id, video_frame, keypoints, noise_scale FROM detection_keypoints "
        "WHERE detection_run_id = ? AND track_id = ? AND region_type = ? "
        "ORDER BY shot_video_id, video_frame",
        (detection_run_id, track_id, region_type),
    ):
        blob = np.frombuffer(r["keypoints"], dtype=np.float32).reshape(-1, 3)
        new = np.full((4 * len(object_ids), 3), np.nan, dtype=np.float32)
        new[:, 2] = 0.0
        for new_slot, marker_id in enumerate(object_ids):
            old_slot = slot_in_run.get(marker_id)
            if old_slot is not None:
                new[new_slot * 4:new_slot * 4 + 4] = blob[old_slot * 4:old_slot * 4 + 4]
        if np.any(new[:, 2] > 0.5):
            out.append((r["shot_video_id"], r["video_frame"], track_id, region_type, new.tobytes(), r["noise_scale"]))
    return out
