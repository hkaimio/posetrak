# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""finalise.py — Convert detection keypoints into pose_observation_sequences."""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import json

import numpy as np

from posetrak.db.db import generate_id
from posetrak.db.manage_person import find_or_create_person
from app.setup.db_context import SyncPoint, SyncTable


@dataclass
class TrackAssignment:
    shot_video_id: str
    track_id: int
    person_name: str
    first_frame: int   # inclusive
    last_frame: int    # inclusive


def _scale_kp_confidence(kp_blob: bytes, scale: float) -> bytes:
    """Multiply the confidence channel of a float32[N,3] keypoint blob by *scale*."""
    arr = np.frombuffer(kp_blob, dtype="<f4").copy().reshape(-1, 3)
    arr[:, 2] *= scale
    return arr.tobytes()


def conf_scale_for_model(pose_model: str) -> float:
    """Return the confidence normalisation factor for *pose_model*.

    RTMPose outputs SimCC logit scores (3–8 for well-detected joints).
    VITpose outputs heatmap peak values (0–1).  The C++ UKF uses
    ``measurement_noise_std = base_noise / confidence``, so a scale factor
    is needed to bring ViTPose values into the same effective noise range.
    """
    try:
        from app.pose.backends_rtmpose import _KNOWN_MODELS
        entry = _KNOWN_MODELS.get(pose_model)
        if entry is not None:
            return float(entry[3])
    except Exception:
        pass
    return 1.0


def finalise_to_db(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_id: str,
    sync_config_id: str,
    assignments: list[TrackAssignment],
    pose_model: str,
    notes: str = "",
    confidence_scale: float = 1.0,
) -> list[str]:
    """Write pose_observation_sequences + pose_observations from detection_keypoints.

    Creates one sequence per person.  Replaces any existing sequences for this
    detection run (including their tracking runs and results).

    Returns the list of new sequence IDs (one per person).
    """
    # Build SyncTable from DB
    rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp "
        "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()

    points = [
        SyncPoint(
            camera_instance_id="",
            shot_video_id=r["shot_video_id"],
            video_frame=r["video_frame"],
            timestamp_s=r["timestamp_s"],
        )
        for r in rows
    ]
    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in rows}
    sync_table = SyncTable(points, fps_by_video)

    # Get camera_instance_id per shot_video_id
    sv_rows = session.execute(
        "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?",
        (shot_id,),
    ).fetchall()
    camera_by_svid = {r["id"]: r["camera_instance_id"] for r in sv_rows}

    # Get time_start/end_s from detection_runs row
    run_row = session.execute(
        "SELECT time_start_s, time_end_s FROM detection_runs WHERE id = ?",
        (detection_run_id,),
    ).fetchone()
    time_start_s = float(run_row["time_start_s"])
    time_end_s = float(run_row["time_end_s"])

    # Delete existing sequences for this detection run (cascade manually).
    # This supports re-finalising while track assignments are still being
    # tuned (stitch -> finalise -> notice a mistake -> restitch -> finalise
    # again) -- but only up to the point where real work has been built on
    # top of a sequence. Once a sequence has tracking results and/or manual
    # keypoint edits, detection runs are meant to be immutable: refuse
    # instead of silently destroying that work (and instead of the FK
    # violation on pose_observation_edits.sequence_id this cascade used to
    # hit uncontrolled, since edits were never deleted here).
    existing_ids = [
        r[0] for r in session.execute(
            "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?",
            (detection_run_id,),
        )
    ]
    for sid in existing_ids:
        has_tracking = session.execute(
            "SELECT 1 FROM tracking_runs WHERE observation_sequence_id = ? LIMIT 1", (sid,)
        ).fetchone()
        has_edits = session.execute(
            "SELECT 1 FROM pose_observation_edits WHERE sequence_id = ? LIMIT 1", (sid,)
        ).fetchone()
        if has_tracking or has_edits:
            raise RuntimeError(
                f"Cannot re-finalise detection run {detection_run_id}: sequence {sid} "
                "already has tracking results and/or manual keypoint edits. Detection "
                "runs are immutable once tracked or edited -- create a new detection "
                "run instead of re-finalising this one."
            )
    for sid in existing_ids:
        session.execute(
            "DELETE FROM tracking_results WHERE run_id IN "
            "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,)
        )
        session.execute(
            "DELETE FROM tracking_obs_results WHERE run_id IN "
            "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,)
        )
        session.execute(
            "DELETE FROM tracking_runs WHERE observation_sequence_id = ?", (sid,)
        )
        session.execute("DELETE FROM sequence_persons WHERE sequence_id = ?", (sid,))
        session.execute("DELETE FROM pose_observations WHERE sequence_id = ?", (sid,))
    session.execute(
        "DELETE FROM pose_observation_sequences WHERE detection_run_id = ?",
        (detection_run_id,),
    )

    # Group assignments by person name
    by_person: dict[str, list[TrackAssignment]] = defaultdict(list)
    for asgn in assignments:
        by_person[asgn.person_name].append(asgn)

    seq_ids: list[str] = []
    # Resolves each assigned name to this capture's capture_persons row
    # (config-improvements design doc, "Person model"), creating one if this
    # is the first time that name's been used in this capture -- reused
    # below for both sequence_persons and detection_track_assignments so a
    # person's identity carries across detection runs within the capture.
    capture_person_ids: dict[str, str] = {}

    for person_name, person_assignments in by_person.items():
        seq_id = generate_id()
        seq_ids.append(seq_id)
        capture_person_id = find_or_create_person(session, shot_id, person_name)
        capture_person_ids[person_name] = capture_person_id

        session.execute(
            "INSERT INTO pose_observation_sequences "
            "(id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model, notes, "
            " pixels_are_undistorted, detection_run_id) "
            "VALUES (?,?,?,?,?,?,?,0,?)",
            (seq_id, shot_id, sync_config_id, time_start_s, time_end_s,
             pose_model, notes, detection_run_id),
        )

        # One person per sequence — person_id is always 0 within the sequence
        session.execute(
            "INSERT OR IGNORE INTO sequence_persons "
            "(sequence_id, person_id, person_name, capture_person_id) "
            "VALUES (?, 0, ?, ?)",
            (seq_id, person_name, capture_person_id),
        )

        # Keyed by (camera_instance_id, video_frame, source) so that adjacent
        # segments sharing a boundary frame don't produce duplicates (later
        # segment wins), while 'body' and 'hand_l'/'hand_r' rows for the same
        # frame coexist as separate pose_observations rows.
        obs_by_key: dict[tuple, tuple] = {}
        for asgn in person_assignments:
            camera_instance_id = camera_by_svid.get(asgn.shot_video_id)
            if camera_instance_id is None:
                continue

            kp_rows = session.execute(
                "SELECT video_frame, region_type, keypoints, noise_scale "
                "FROM detection_keypoints "
                "WHERE detection_run_id=? AND shot_video_id=? AND track_id=? "
                "AND video_frame BETWEEN ? AND ? "
                "AND region_type IN ('full_body','hand_l','hand_r') "
                "ORDER BY video_frame",
                (detection_run_id, asgn.shot_video_id, asgn.track_id,
                 asgn.first_frame, asgn.last_frame),
            ).fetchall()

            for kp_row in kp_rows:
                frame_idx = int(kp_row["video_frame"])
                timestamp_s = sync_table.frame_to_global_time(frame_idx, asgn.shot_video_id)
                if timestamp_s is None:
                    continue
                # detection_keypoints.region_type='full_body' becomes
                # pose_observations.source='body'; hand_l/hand_r pass through
                # unchanged (same spelling, both enums already agree).
                source = "body" if kp_row["region_type"] == "full_body" else kp_row["region_type"]
                kp_blob = bytes(kp_row["keypoints"])
                # confidence_scale corrects the whole-body pose model's raw
                # score range (see conf_scale_for_model) — hand rows already
                # carry a hand-model-appropriate scale from hand_refinement.py.
                if source == "body" and confidence_scale != 1.0:
                    kp_blob = _scale_kp_confidence(kp_blob, confidence_scale)
                obs_by_key[(camera_instance_id, frame_idx, source)] = (
                    seq_id, camera_instance_id, frame_idx,
                    timestamp_s, 0,  # person_id=0, single person per sequence
                    source, detection_run_id,
                    kp_blob, kp_row["noise_scale"],
                )

        if obs_by_key:
            session.executemany(
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
                " person_id, source, detection_run_id, kp_blob, noise_scale) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                obs_by_key.values(),
            )

    # Persist the track→person mapping for future restore
    session.execute(
        "DELETE FROM detection_track_assignments WHERE detection_run_id = ?",
        (detection_run_id,),
    )
    session.executemany(
        "INSERT INTO detection_track_assignments "
        "(detection_run_id, shot_video_id, track_id, person_name, capture_person_id, "
        " first_frame, last_frame) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (detection_run_id, a.shot_video_id, a.track_id,
             a.person_name, capture_person_ids[a.person_name], a.first_frame, a.last_frame)
            for a in assignments
        ],
    )

    session.commit()
    return seq_ids


def auto_assign_and_finalise(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_id: str,
    sync_config_id: str,
    persons_ordered: list[str],
    pose_model: str,
    notes: str = "",
    confidence_scale: float = 1.0,
) -> list[str]:
    """Finalise a segmentation-sourced detection run with no manual
    stitcher pass, deriving track->person assignments directly from
    person_tracks instead of building them interactively.

    Works because a segmentation mask's per-person label is already a
    stable identity -- mask label i+1 maps to persons_ordered[i], the same
    convention app.pose.pose_worker._bboxes_from_mask uses when writing
    detections, so track_id (== mask label) -> person name is a pure
    lookup, not something that needs stitching across ambiguous YOLO
    tracks. See docs/roadmap/features/segmentation-reuse/
    segmentation-reuse-design.md, "Auto-assignment" (gap 3).

    A track_id outside persons_ordered's range (stray/corrupt data) is
    skipped rather than raising -- finalise_to_db still runs for whatever
    valid assignments remain.
    """
    track_rows = session.execute(
        "SELECT shot_video_id, track_id, first_frame, last_frame "
        "FROM person_tracks WHERE detection_run_id = ?",
        (detection_run_id,),
    ).fetchall()
    assignments = [
        TrackAssignment(
            shot_video_id=row["shot_video_id"],
            track_id=row["track_id"],
            person_name=persons_ordered[row["track_id"] - 1],
            first_frame=row["first_frame"],
            last_frame=row["last_frame"],
        )
        for row in track_rows
        if 1 <= row["track_id"] <= len(persons_ordered)
    ]
    return finalise_to_db(
        session=session,
        detection_run_id=detection_run_id,
        shot_id=shot_id,
        sync_config_id=sync_config_id,
        assignments=assignments,
        pose_model=pose_model,
        notes=notes,
        confidence_scale=confidence_scale,
    )


def finalise_object_to_db(
    session: sqlite3.Connection,
    detection_run_id: str,
    notes: str = "",
) -> str:
    """Finalise a marker detection run into one `pose_observation_sequence`
    for its tracked object (marker-based-mocap design doc §4.3, §7.1
    sub-phase 1d).

    Unlike `finalise_to_db`, there is no stitching/track-assignment step:
    a marker detection run already has exactly one implicit subject
    (`track_id=0` throughout, "one prop = one track" — design §4.1), so
    this reads every camera's `detection_keypoints` directly and writes
    one sequence, automatically — the finalisation call itself *is* the
    only decision (design §7.1's 1d/1e ordering note: an object has no
    pre-finalisation review step to make first, unlike a person's
    track-to-person stitching).

    Also writes the `pose_sequence_keypoints` manifest (design §4.3): each
    landmark's name and source, resolved from the run's own `config_json`
    (`marker_ids`, and the marker body definition's own marker names when
    known) — so the sequence is self-describing without needing to
    re-resolve the marker body definition later.

    Raises
    ------
    ValueError
        If *detection_run_id* is not a marker (`detector_type='aruco'`)
        run, or has no `capture_object_id` (only an object-bound run can
        be finalised this way — a standalone/scripted phase-1a run with
        no known object has nothing to attach the resulting sequence's
        identity to).
    RuntimeError
        If a sequence for this run already has tracking results and/or
        manual keypoint edits (same immutability guard as `finalise_to_db`).
    """
    from app.pose.db_cache import MARKER_REGION_TYPE, MARKER_TRACK_ID
    from app.setup.fiducial_markers import load_marker_body_yaml

    run = session.execute(
        "SELECT shot_id, sync_config_id, time_start_s, time_end_s, detector_type, "
        "       detector_model, config_json, capture_object_id "
        "FROM detection_runs WHERE id = ?",
        (detection_run_id,),
    ).fetchone()
    if run is None:
        raise ValueError(f"detection_runs row not found: {detection_run_id!r}")
    if run["detector_type"] != "aruco":
        raise ValueError(
            f"detection run {detection_run_id!r} is a {run['detector_type']!r} run, "
            "not a marker run -- use finalise_to_db for pose detection runs"
        )
    if run["capture_object_id"] is None:
        raise ValueError(
            f"detection run {detection_run_id!r} has no capture_object_id -- only "
            "an object-bound run (design phase 1c) can be finalised into an object "
            "sequence; a standalone/scripted run has no object to attach one to"
        )

    shot_id = run["shot_id"]
    sync_config_id = run["sync_config_id"]
    time_start_s = float(run["time_start_s"])
    time_end_s = float(run["time_end_s"])
    config = json.loads(run["config_json"] or "{}")
    marker_ids: list[str] = config.get("marker_ids") or []
    marker_body_definition_id = config.get("marker_body_definition_id")

    # Landmark names: prefer the marker body definition's own names
    # (survives even if a future run re-decodes the same physical markers
    # under different dictionary ids); fall back to the bare marker id
    # for a standalone run with no known body.
    marker_names: dict[str, str] = {}
    if marker_body_definition_id is not None:
        body_row = session.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?",
            (marker_body_definition_id,),
        ).fetchone()
        if body_row is not None and body_row["yaml_content"] is not None:
            marker_names = load_marker_body_yaml(body_row["yaml_content"]).marker_names

    landmark_names: list[str] = []
    for marker_id in marker_ids:
        name = marker_names.get(marker_id, marker_id)
        landmark_names.extend(f"{name}:c{i}" for i in range(4))

    # Same immutability guard as finalise_to_db -- refuse to blow away
    # real work rather than the FK violation an uncontrolled cascade used
    # to hit (see that function's own comment).
    existing_ids = [
        r[0] for r in session.execute(
            "SELECT id FROM pose_observation_sequences WHERE detection_run_id = ?",
            (detection_run_id,),
        )
    ]
    for sid in existing_ids:
        has_tracking = session.execute(
            "SELECT 1 FROM tracking_runs WHERE observation_sequence_id = ? LIMIT 1", (sid,)
        ).fetchone()
        has_edits = session.execute(
            "SELECT 1 FROM pose_observation_edits WHERE sequence_id = ? LIMIT 1", (sid,)
        ).fetchone()
        if has_tracking or has_edits:
            raise RuntimeError(
                f"Cannot re-finalise detection run {detection_run_id}: sequence {sid} "
                "already has tracking results and/or manual keypoint edits. Detection "
                "runs are immutable once tracked or edited -- create a new detection "
                "run instead of re-finalising this one."
            )
    for sid in existing_ids:
        session.execute(
            "DELETE FROM tracking_results WHERE run_id IN "
            "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,)
        )
        session.execute(
            "DELETE FROM tracking_obs_results WHERE run_id IN "
            "(SELECT id FROM tracking_runs WHERE observation_sequence_id = ?)", (sid,)
        )
        session.execute(
            "DELETE FROM tracking_runs WHERE observation_sequence_id = ?", (sid,)
        )
        session.execute("DELETE FROM pose_sequence_keypoints WHERE sequence_id = ?", (sid,))
        session.execute("DELETE FROM pose_observations WHERE sequence_id = ?", (sid,))
    session.execute(
        "DELETE FROM pose_observation_sequences WHERE detection_run_id = ?",
        (detection_run_id,),
    )

    seq_id = generate_id()
    session.execute(
        "INSERT INTO pose_observation_sequences "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model, notes, "
        " pixels_are_undistorted, detection_run_id) "
        "VALUES (?,?,?,?,?,?,?,0,?)",
        (seq_id, shot_id, sync_config_id, time_start_s, time_end_s,
         run["detector_model"], notes, detection_run_id),
    )
    if landmark_names:
        session.executemany(
            "INSERT INTO pose_sequence_keypoints (sequence_id, keypoint_idx, name, source) "
            "VALUES (?, ?, ?, ?)",
            [(seq_id, i, name, "aruco") for i, name in enumerate(landmark_names)],
        )

    # Build SyncTable + camera_instance_id lookup, same as finalise_to_db.
    rows = session.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, sv.actual_fps "
        "FROM sync_points sp "
        "JOIN capture_videos sv ON sv.id = sp.shot_video_id "
        "WHERE sp.sync_config_id = ?",
        (sync_config_id,),
    ).fetchall()
    points = [
        SyncPoint(
            camera_instance_id="", shot_video_id=r["shot_video_id"],
            video_frame=r["video_frame"], timestamp_s=r["timestamp_s"],
        )
        for r in rows
    ]
    fps_by_video = {r["shot_video_id"]: float(r["actual_fps"]) for r in rows}
    sync_table = SyncTable(points, fps_by_video)

    sv_rows = session.execute(
        "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?", (shot_id,)
    ).fetchall()
    camera_by_svid = {r["id"]: r["camera_instance_id"] for r in sv_rows}

    kp_rows = session.execute(
        "SELECT shot_video_id, video_frame, keypoints, noise_scale FROM detection_keypoints "
        "WHERE detection_run_id = ? AND track_id = ? AND region_type = ? "
        "ORDER BY shot_video_id, video_frame",
        (detection_run_id, MARKER_TRACK_ID, MARKER_REGION_TYPE),
    ).fetchall()

    obs_rows = []
    for kp_row in kp_rows:
        camera_instance_id = camera_by_svid.get(kp_row["shot_video_id"])
        if camera_instance_id is None:
            continue
        frame_idx = int(kp_row["video_frame"])
        timestamp_s = sync_table.frame_to_global_time(frame_idx, kp_row["shot_video_id"])
        if timestamp_s is None:
            continue
        obs_rows.append((
            seq_id, camera_instance_id, frame_idx, timestamp_s,
            0,  # person_id column, repurposed as "the object" -- always 0, one subject per sequence
            "markers", detection_run_id, bytes(kp_row["keypoints"]), kp_row["noise_scale"],
        ))
    if obs_rows:
        session.executemany(
            "INSERT INTO pose_observations "
            "(sequence_id, camera_instance_id, video_frame, timestamp_s, "
            " person_id, source, detection_run_id, kp_blob, noise_scale) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            obs_rows,
        )

    session.commit()
    return seq_id
