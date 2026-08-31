# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""db_cache.py — Read/write helpers for detection pipeline DB tables."""
from __future__ import annotations

import datetime
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from posetrak.db.db import generate_id
from posetrak.db.observation_merge import BODY_SOURCE, infer_body_width, merge_observation_sources

_CROP_JPEG_QUALITY = 75
_CROP_TARGET_HEIGHT = 240
_CROP_MARGIN = 0.20  # fraction of bbox dimension added on each side


# ---------------------------------------------------------------------------
# Detection run
# ---------------------------------------------------------------------------

@dataclass
class DetectionRunInfo:
    id: str
    shot_id: str
    sync_config_id: str
    time_start_s: float
    time_end_s: float
    detector_model: str
    pose_model: str
    detector_version: str
    pose_version: str
    detector_conf: float
    pose_conf_threshold: float
    pose_input_width: int
    pose_input_height: int
    status: str
    created_at: str
    completed_at: str | None


def create_detection_run(
    session: sqlite3.Connection,
    shot_id: str,
    sync_config_id: str,
    time_start_s: float,
    time_end_s: float,
    detector_model: str,
    pose_model: str,
    trial_id: str | None = None,
    detector_version: str = "",
    pose_version: str = "",
    detector_conf: float = 0.3,
    pose_conf_threshold: float = 0.3,
    pose_input_width: int = 0,
    pose_input_height: int = 0,
    detector_type: str = "pose",
    config_json: str | None = None,
) -> str:
    run_id = generate_id()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    session.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s, "
        " detector_model, pose_model, detector_version, pose_version, "
        " detector_conf, pose_conf_threshold, "
        " pose_input_width, pose_input_height, status, created_at, "
        " detector_type, config_json) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'running',?,?,?)",
        (run_id, shot_id, sync_config_id, trial_id, time_start_s, time_end_s,
         detector_model, pose_model, detector_version, pose_version,
         detector_conf, pose_conf_threshold,
         pose_input_width, pose_input_height, now,
         detector_type, config_json),
    )
    session.commit()
    return run_id


def create_marker_detection_run(
    session: sqlite3.Connection,
    shot_id: str,
    sync_config_id: str,
    time_start_s: float,
    time_end_s: float,
    dictionary: str,
    marker_ids: list[str],
    min_marker_perimeter_rate: float | None = None,
    frame_step: int = 1,
    trial_id: str | None = None,
    capture_object_id: str | None = None,
    marker_body_definition_id: str | None = None,
) -> str:
    """Create a detection_runs row for an ArUco marker detection pass.

    Design phase 1a (marker-mocap-design.md §7.1) covers the standalone
    case: *dictionary*/*marker_ids* given directly, no
    `marker_body_definitions`/`capture_objects` row involved.
    Phase 1c's `MarkerDetectionPipeline`/`load_pipeline_for_capture_object`
    additionally passes *capture_object_id* and *marker_body_definition_id*
    once a real object/registered body drives the run. `config_json`'s
    `marker_ids` is the corner-blob decode key for `detection_keypoints`
    (§4.1) in both cases: a run's coded-marker corner slots are ordered
    list-position-major by this list, so re-deriving the blob layout later
    only ever needs this one field, not a second lookup -- even for a
    marker-body-driven run, where *dictionary* itself is only a best-effort
    single value (a body spanning more than one ArUco dictionary has no one
    right answer for it; the per-marker dictionaries live in the marker
    body definition itself, reachable via `marker_body_definition_id`).

    `capture_object_id` also becomes `detection_runs.capture_object_id`
    (not just a `config_json` field) so a run's object is directly
    queryable -- `config_json.marker_ids` alone can't disambiguate two
    `capture_objects` rows that reference the *same* marker body
    definition (e.g. two physically-identical props in one capture),
    mirroring why `tracking_run_persons.capture_object_id` (design §4.2)
    is an explicit column rather than left as convention.
    """
    config = {
        "dictionary": dictionary,
        "marker_ids": list(marker_ids),
        "min_marker_perimeter_rate": min_marker_perimeter_rate,
        "frame_step": frame_step,
    }
    if marker_body_definition_id is not None:
        config["marker_body_definition_id"] = marker_body_definition_id
    if capture_object_id is not None:
        config["capture_object_id"] = capture_object_id
    run_id = create_detection_run(
        session,
        shot_id=shot_id,
        sync_config_id=sync_config_id,
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        detector_model=f"aruco:{dictionary}",
        pose_model="",  # no pose model for a marker run; NOT NULL, so "" not NULL
        trial_id=trial_id,
        detector_type="aruco",
        config_json=json.dumps(config),
    )
    if capture_object_id is not None:
        session.execute(
            "UPDATE detection_runs SET capture_object_id = ? WHERE id = ?",
            (capture_object_id, run_id),
        )
        session.commit()
    return run_id


def mark_run_complete(session: sqlite3.Connection, run_id: str, status: str = "complete") -> None:
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    session.execute(
        "UPDATE detection_runs SET status=?, completed_at=? WHERE id=?",
        (status, now, run_id),
    )
    session.commit()


def list_detection_runs(
    session: sqlite3.Connection, shot_id: str
) -> list[dict]:
    rows = session.execute(
        "SELECT id, detector_model, pose_model, time_start_s, time_end_s, "
        "       status, created_at, completed_at "
        "FROM detection_runs WHERE shot_id=? ORDER BY created_at DESC",
        (shot_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Per-frame detection batch writer
# ---------------------------------------------------------------------------

_BATCH_SIZE = 200


def _encode_crop(img: "np.ndarray", bbox: tuple) -> "tuple[bytes, int, int, int, int, int, int] | None":
    """Crop *img* by *bbox* (cx, cy, w, h) with margin and return
    (jpeg, jpeg_w, jpeg_h, src_x, src_y, src_w, src_h).

    src_* are the crop coordinates in the original full-resolution frame
    before any JPEG downscale.
    """
    cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    mx, my = w * _CROP_MARGIN, h * _CROP_MARGIN
    x1 = max(0, int(cx - w / 2 - mx))
    y1 = max(0, int(cy - h / 2 - my))
    x2 = min(img.shape[1], int(cx + w / 2 + mx))
    y2 = min(img.shape[0], int(cy + h / 2 + my))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    src_w, src_h = x2 - x1, y2 - y1
    if crop.shape[0] > _CROP_TARGET_HEIGHT:
        scale = _CROP_TARGET_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), _CROP_TARGET_HEIGHT))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, _CROP_JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes(), crop.shape[1], crop.shape[0], x1, y1, src_w, src_h


class DetectionBatchWriter:
    """Accumulates detection rows and flushes to DB in batches."""

    def __init__(
        self,
        session: sqlite3.Connection,
        detection_run_id: str,
        shot_video_id: str,
        pose_input_width: int,
    ) -> None:
        self._session = session
        self._run_id = detection_run_id
        self._svid = shot_video_id
        self._pose_input_width = pose_input_width
        self._det_rows: list[tuple] = []
        self._kp_rows: list[tuple] = []
        self._crop_rows: list[tuple] = []
        # track_id -> (first_frame, last_frame)
        self._track_spans: dict[int, tuple[int, int]] = {}

    def add_frame(
        self,
        video_frame: int,
        detections,           # list[PersonDetection]
        pose_results,         # list[PoseResult]
        model_name: str,
        img: np.ndarray | None = None,  # BGR frame for crop storage
    ) -> None:
        kp_by_track = {pr.track_id: pr.keypoints for pr in pose_results}

        for det in detections:
            self._det_rows.append((
                self._run_id, self._svid, video_frame,
                det.track_id, "full_body", model_name,
                float(det.bbox[0]), float(det.bbox[1]),
                float(det.bbox[2]), float(det.bbox[3]),
                float(det.confidence),
            ))

            if det.track_id in kp_by_track:
                kp = kp_by_track[det.track_id]
                noise_scale = (
                    float(det.bbox[2]) / self._pose_input_width
                    if self._pose_input_width > 0 else None
                )
                self._kp_rows.append((
                    self._run_id, self._svid, video_frame,
                    det.track_id, "full_body",
                    kp.astype(np.float32).tobytes(),
                    noise_scale,
                ))

            if img is not None:
                result = _encode_crop(img, det.bbox)
                if result is not None:
                    jpeg, wpx, hpx, src_x, src_y, src_w, src_h = result
                    self._crop_rows.append((
                        self._svid, video_frame, "person_crop",
                        det.track_id, "full_body", wpx, hpx, jpeg, self._run_id,
                        src_x, src_y, src_w, src_h,
                    ))

            first, last = self._track_spans.get(det.track_id, (video_frame, video_frame))
            self._track_spans[det.track_id] = (min(first, video_frame), max(last, video_frame))

        if len(self._det_rows) >= _BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if self._det_rows:
            self._session.executemany(
                "INSERT OR REPLACE INTO person_detections "
                "(detection_run_id, shot_video_id, video_frame, track_id, region_type, "
                " model_name, bbox_x, bbox_y, bbox_w, bbox_h, confidence) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                self._det_rows,
            )
            self._det_rows.clear()
        if self._kp_rows:
            self._session.executemany(
                "INSERT OR REPLACE INTO detection_keypoints "
                "(detection_run_id, shot_video_id, video_frame, track_id, region_type, "
                " keypoints, noise_scale) "
                "VALUES (?,?,?,?,?,?,?)",
                self._kp_rows,
            )
            self._kp_rows.clear()
        if self._crop_rows:
            self._session.executemany(
                "INSERT OR REPLACE INTO frame_cache_entries "
                "(shot_video_id, frame_idx, cache_type, track_id, region_type, "
                " width_px, height_px, image_data, detection_run_id, "
                " src_x, src_y, src_w, src_h) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self._crop_rows,
            )
            self._crop_rows.clear()
        self._session.commit()

    def finalise(self) -> None:
        """Flush remaining rows and write track spans."""
        self._flush()
        rows = [
            (generate_id(), self._run_id, self._svid, tid, first, last)
            for tid, (first, last) in self._track_spans.items()
        ]
        if rows:
            self._session.executemany(
                "INSERT OR REPLACE INTO person_tracks "
                "(id, detection_run_id, shot_video_id, track_id, first_frame, last_frame) "
                "VALUES (?,?,?,?,?,?)",
                rows,
            )
            self._session.commit()


# ---------------------------------------------------------------------------
# Coded-marker (ArUco) keypoint writer -- design phase 1a
# ---------------------------------------------------------------------------

MARKER_TRACK_ID = 0       # one prop = one track (aruco-prop-tracking-design.md)
MARKER_REGION_TYPE = "markers"

# The tracker's measurement noise splits into two independent pieces
# (Observation::measurement_noise_std(ep, ec) = ep*crop_scale + ec):
# ep ("pose_noise_std") is the detection algorithm's own localization error,
# scaled by crop_scale -- how much the algorithm's fixed input resolution
# was stretched to cover the real-world crop, so a farther/bigger bbox
# means more real pixels of error per network-input pixel. ec
# ("calib_noise_std") is camera-specific error (extrinsics inaccuracy,
# autofocus drift affecting intrinsics, ...) that applies regardless of
# detection method. A coded ArUco corner is found by direct sub-pixel
# corner refinement on the full-resolution frame -- there is no
# fixed-input-resolution network stage for crop_scale to describe, and the
# corner-finding error itself is negligible next to ec -- so crop_scale
# should be ~0 for markers (letting ec alone dominate), not the person
# pipeline's 1.0 default. See status.md's 2026-08-31 entry (code review
# finding #4): before this, marker observations silently reused the full
# ep contribution meant for a markerless pose network, under-trusting
# sub-pixel-precise corners relative to a ~5-25px interpolated keypoint.
_MARKER_CROP_SCALE = 0.0


class MarkerKeypointWriter:
    """Accumulates coded-marker corner rows and flushes to DB in batches.

    Fixed-slot layout per marker-mocap-design.md §4.1: one row per (frame,
    camera), track_id=0, region_type='markers'. The blob is
    float32[4 * len(marker_ids), 3] (x, y, confidence), ordered
    list-position-major by *marker_ids* with corners 0-3 within each
    marker (real ``cv2.aruco`` corner order -- see
    ``fiducial_markers.ArucoDetector``'s own docstring). A marker not seen
    in a given frame keeps NaN x/y and confidence 0 at its slot -- exactly
    an occluded keypoint, so the same NaN-handling code paths used for pose
    keypoints apply unchanged.
    """

    def __init__(
        self,
        session: sqlite3.Connection,
        detection_run_id: str,
        shot_video_id: str,
        marker_ids: list[str],
    ) -> None:
        self._session = session
        self._run_id = detection_run_id
        self._svid = shot_video_id
        self._marker_ids = list(marker_ids)
        self._slot_of = {mid: i for i, mid in enumerate(self._marker_ids)}
        self._rows: list[tuple] = []

    def add_frame(self, video_frame: int, detections: list) -> None:
        """*detections* is the ``ArucoDetector.detect()`` result for this frame."""
        n = len(self._marker_ids)
        kp = np.full((4 * n, 3), np.nan, dtype=np.float32)
        kp[:, 2] = 0.0  # confidence -- overwritten to 1.0 per corner actually seen
        for det in detections:
            slot = self._slot_of.get(det.marker_id)
            if slot is None:
                continue  # a marker outside this prop's configured id list -- ignore
            for corner in det.corners:
                idx = slot * 4 + corner.corner_index
                kp[idx, 0] = corner.px
                kp[idx, 1] = corner.py
                kp[idx, 2] = 1.0
        self._rows.append((
            self._run_id, self._svid, video_frame, MARKER_TRACK_ID, MARKER_REGION_TYPE,
            kp.tobytes(), _MARKER_CROP_SCALE,
        ))
        if len(self._rows) >= _BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if self._rows:
            self._session.executemany(
                "INSERT OR REPLACE INTO detection_keypoints "
                "(detection_run_id, shot_video_id, video_frame, track_id, region_type, "
                " keypoints, noise_scale) "
                "VALUES (?,?,?,?,?,?,?)",
                self._rows,
            )
            self._rows.clear()
            self._session.commit()

    def finalise(self) -> None:
        """Flush remaining rows."""
        self._flush()


def read_marker_keypoints_for_run(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
) -> dict[int, np.ndarray]:
    """Return {video_frame: float32[4*n_markers, 3]} for a coded-marker run."""
    rows = session.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints "
        "WHERE detection_run_id=? AND shot_video_id=? AND track_id=? AND region_type=? "
        "ORDER BY video_frame",
        (detection_run_id, shot_video_id, MARKER_TRACK_ID, MARKER_REGION_TYPE),
    ).fetchall()
    result = {}
    for row in rows:
        kp_bytes = bytes(row["keypoints"])
        n = len(kp_bytes) // (3 * 4)
        result[row["video_frame"]] = np.frombuffer(kp_bytes, dtype=np.float32).reshape(n, 3)
    return result


def read_detections_for_run(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
) -> list[dict]:
    """Return all detection rows for one camera in a run, ordered by frame."""
    rows = session.execute(
        "SELECT video_frame, track_id, bbox_x, bbox_y, bbox_w, bbox_h, confidence "
        "FROM person_detections "
        "WHERE detection_run_id=? AND shot_video_id=? "
        "ORDER BY video_frame, track_id",
        (detection_run_id, shot_video_id),
    ).fetchall()
    return [dict(r) for r in rows]


def read_keypoints_for_run(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
    track_id: int,
) -> dict[int, np.ndarray]:
    """Return {video_frame: keypoints float32[N,3]} for one track."""
    rows = session.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints "
        "WHERE detection_run_id=? AND shot_video_id=? AND track_id=? "
        "AND region_type='full_body' ORDER BY video_frame",
        (detection_run_id, shot_video_id, track_id),
    ).fetchall()
    result = {}
    for row in rows:
        kp_bytes = bytes(row["keypoints"])
        n = len(kp_bytes) // (3 * 4)  # float32, 3 values per kp
        result[row["video_frame"]] = np.frombuffer(kp_bytes, dtype=np.float32).reshape(n, 3)
    return result


def read_observations_with_edits(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    primary_source: str = BODY_SOURCE,
) -> dict[int, np.ndarray]:
    """Return {video_frame: float32[N,3]} for one camera, with pose_observation_edits applied.

    Each frame's keypoint array matches the float32[N,3] format used in
    pose_observations.kp_blob.  If a frame has rows from multiple detection
    sources (e.g. 'body' plus refined 'hand_l'/'hand_r' passes), they are
    merged first via observation_merge.merge_observation_sources.  Edited
    slots (bit set in kp_mask) have their x/y replaced by the edit values;
    if the edit marks a keypoint as outlier (is_outlier != 0) its confidence
    is zeroed.

    *primary_source* (default `BODY_SOURCE`) names the source this
    sequence's single "real" row uses as its base layer -- pass a
    sequence's own source (e.g. 'markers' for a marker-based-mocap object
    sequence, design doc §7.1 sub-phase 1e) when it isn't 'body', so that
    sequence's row is correctly treated as the base layer instead of a
    same-width zero body silently overwriting it once any edit exists (see
    `merge_observation_sources`'s own docstring and status.md's 2026-08-30
    note for why this matters).
    """
    obs_rows = session.execute(
        "SELECT video_frame, source, kp_blob FROM pose_observations"
        " WHERE sequence_id = ? AND camera_instance_id = ?"
        " ORDER BY video_frame",
        (sequence_id, camera_instance_id),
    ).fetchall()
    if not obs_rows:
        return {}

    by_frame: dict[int, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for row in obs_rows:
        kp = np.frombuffer(bytes(row["kp_blob"]), dtype=np.float32).reshape(-1, 3).copy()
        by_frame[row["video_frame"]].append((row["source"], kp))

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

    default_width = infer_body_width(by_frame.values(), primary_source=primary_source)
    if default_width is None and edits:
        default_width = next(iter(edits.values()))[0].shape[0]

    def _apply_edit(kp: np.ndarray, edit_kp: np.ndarray, mask: bytes) -> None:
        if edit_kp.shape[0] != kp.shape[0]:
            return
        for i in range(kp.shape[0]):
            byte_idx, bit_idx = divmod(i, 8)
            if byte_idx < len(mask) and (mask[byte_idx] >> bit_idx) & 1:
                kp[i, 0] = edit_kp[i, 0]
                kp[i, 1] = edit_kp[i, 1]
                if edit_kp[i, 2] != 0.0:  # is_outlier → zero confidence
                    kp[i, 2] = 0.0
                else:
                    kp[i, 2] = 1.0  # manually placed → full confidence

    result: dict[int, np.ndarray] = {}
    for frame, rows in by_frame.items():
        kp = merge_observation_sources(rows, default_width=default_width, primary_source=primary_source)
        if kp is None:
            # No primary-source row for this frame and no other frame in
            # this camera had one either (default_width also came up
            # empty) -- nothing establishes the true width, so fall back
            # to whichever row is
            # present rather than dropping the frame.
            kp = rows[0][1]
        if frame in edits:
            _apply_edit(kp, *edits[frame])
        result[frame] = kp

    # Include ghost frames: edit rows with no backing pose_observations row.
    for frame, (edit_kp, mask) in edits.items():
        if frame in result:
            continue
        kp = np.zeros_like(edit_kp)  # all-zero confidence base
        _apply_edit(kp, edit_kp, mask)
        result[frame] = kp

    return result


def write_observation_edit(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    video_frame: int,
    kp: np.ndarray,
    kp_mask: bytes,
) -> None:
    """Upsert one pose_observation_edits row.

    Parameters
    ----------
    kp:
        float32[N,3] array (x, y, is_outlier) for all N keypoint slots.
        Only slots with the corresponding bit set in kp_mask are applied at
        read time; the others are stored but ignored.
    kp_mask:
        uint8 bytes, ceil(N/8) length.  Bit i set → slot i is overridden.
    """
    kp_blob = kp.astype(np.float32).tobytes()
    edit_id = generate_id()
    session.execute(
        "INSERT INTO pose_observation_edits"
        " (id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(sequence_id, camera_instance_id, video_frame)"
        " DO UPDATE SET kp_blob=excluded.kp_blob, kp_mask=excluded.kp_mask",
        (edit_id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask),
    )
    session.commit()


def update_single_keypoint_edit(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    video_frame: int,
    kp_idx: int,
    new_x: float,
    new_y: float,
    is_outlier: bool = False,
    source: str = BODY_SOURCE,
) -> None:
    """Update one keypoint slot in pose_observation_edits, preserving other slots.

    Reads the existing edit row (if any), merges the new slot into it, and
    writes back with an upsert.  is_outlier=True marks the slot as rejected
    (confidence → 0); is_outlier=False sets the new x/y position (confidence → 1).

    Works on ghost frames (no pose_observations row) by inferring the keypoint
    count from any other observation in the same camera.

    The keypoint count is inferred from a *source*-tagged row (default
    `BODY_SOURCE`, i.e. 'body' — every existing person-panel call site):
    a frame may also have 'hand_l'/'hand_r' rows (narrower, 21-point
    arrays) that must not be mistaken for the frame's full keypoint width.
    A sequence with no 'body' source at all (marker-based-mocap object
    sequences, source='markers' — design doc §7.1 sub-phase 1e) passes
    its own *source* here instead, since 'body' will never exist for it.
    """
    obs_row = session.execute(
        "SELECT kp_blob FROM pose_observations"
        " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?"
        " AND source = ?",
        (sequence_id, camera_instance_id, video_frame, source),
    ).fetchone()

    if obs_row is not None:
        n_kp = len(bytes(obs_row["kp_blob"])) // (3 * 4)
    else:
        # Ghost frame: infer n_kp from any other same-source observation in this camera.
        any_obs = session.execute(
            "SELECT kp_blob FROM pose_observations"
            " WHERE sequence_id = ? AND camera_instance_id = ? AND source = ? LIMIT 1",
            (sequence_id, camera_instance_id, source),
        ).fetchone()
        if any_obs is None:
            return  # no observations at all — cannot determine keypoint count
        n_kp = len(bytes(any_obs["kp_blob"])) // (3 * 4)

    n_mask_bytes = (n_kp + 7) // 8

    edit_row = session.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
        (sequence_id, camera_instance_id, video_frame),
    ).fetchone()

    if edit_row is not None:
        edit_kp = np.frombuffer(bytes(edit_row["kp_blob"]), dtype=np.float32).reshape(-1, 3).copy()
        mask = bytearray(bytes(edit_row["kp_mask"]))
    else:
        edit_kp = np.zeros((n_kp, 3), dtype=np.float32)
        mask = bytearray(n_mask_bytes)

    edit_kp[kp_idx, 0] = new_x
    edit_kp[kp_idx, 1] = new_y
    edit_kp[kp_idx, 2] = 1.0 if is_outlier else 0.0
    mask[kp_idx // 8] |= 1 << (kp_idx % 8)

    write_observation_edit(session, sequence_id, camera_instance_id, video_frame, edit_kp, bytes(mask))


def clear_single_keypoint_edit(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    video_frame: int,
    kp_idx: int,
) -> None:
    """Clear one keypoint slot's override, reverting it to the original detection.

    Clears the kp_mask bit for kp_idx; if no slots remain overridden, deletes
    the row entirely.  No-op if there is no edit row for this frame.
    """
    edit_row = session.execute(
        "SELECT kp_mask FROM pose_observation_edits"
        " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
        (sequence_id, camera_instance_id, video_frame),
    ).fetchone()
    if edit_row is None:
        return

    mask = bytearray(bytes(edit_row["kp_mask"]))
    byte_idx, bit_idx = divmod(kp_idx, 8)
    if byte_idx >= len(mask):
        return
    mask[byte_idx] &= ~(1 << bit_idx)

    if any(mask):
        session.execute(
            "UPDATE pose_observation_edits SET kp_mask = ?"
            " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
            (bytes(mask), sequence_id, camera_instance_id, video_frame),
        )
    else:
        session.execute(
            "DELETE FROM pose_observation_edits"
            " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
            (sequence_id, camera_instance_id, video_frame),
        )
    session.commit()


# Idea 3 (automated post-edit hand redetection): interactively-redetected
# hands are their own pose_observations row, source='hand_l.refined'/
# 'hand_r.refined' -- see posetrak.db.observation_merge's generic
# <base>.refined precedence convention. 'side' matches the vocabulary
# posetrak.detection.hand_refinement already uses ("left"/"right").
_HAND_REFINED_SOURCE = {"left": "hand_l.refined", "right": "hand_r.refined"}


def write_hand_refinement(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    video_frame: int,
    person_id: int,
    timestamp_s: float,
    side: str,
    kp: np.ndarray,
    noise_scale: float,
) -> None:
    """Upsert one interactively-redetected hand as its own pose_observations row.

    Never overwrites the original batch 'hand_l'/'hand_r' row for the same
    frame -- a different `source` value is a different primary-key row
    (`sequence_id, camera_instance_id, video_frame, person_id, source`) --
    and is itself always overridden by a human edit on the same slot at
    load time (pose_observation_edits applies last, unchanged). Callers in
    "auto-detect" mode should immediately follow a successful write with
    `clear_disabled_hand_edits` for the same (frame, side) -- see that
    function's docstring for why.

    `detection_run_id` is left NULL: there is no dedicated "interactive
    redetection" detection run, since minting one per edit wouldn't scale
    (a real editing session produces thousands of these writes). Re-writing
    for the same (camera, frame, side) is a plain overwrite, not
    accumulation -- consistent with how edits and the batch hand pass
    already behave.

    Parameters
    ----------
    kp:
        float32[21,3] hand21-order keypoints (x, y, confidence), full-frame
        pixel coordinates -- same shape/format as a batch hand_l/hand_r row.
    """
    source = _HAND_REFINED_SOURCE[side]
    kp_blob = kp.astype(np.float32).tobytes()
    session.execute(
        "INSERT OR REPLACE INTO pose_observations"
        " (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id,"
        "  source, detection_run_id, kp_blob, noise_scale)"
        " VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)",
        (sequence_id, camera_instance_id, video_frame, timestamp_s, person_id,
         source, kp_blob, noise_scale),
    )
    session.commit()


def revert_hand_refinement(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    video_frame: int,
    person_id: int,
    side: str,
) -> None:
    """Delete an interactively-redetected hand row ("reject", per the design doc).

    Reverting falls back to whatever the batch 'hand_l'/'hand_r' row (or
    nothing) provides for that slot, the same graceful-degradation
    tolerance as any other sparse observation. No-op if there is no such
    row.
    """
    source = _HAND_REFINED_SOURCE[side]
    session.execute(
        "DELETE FROM pose_observations"
        " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?"
        " AND person_id = ? AND source = ?",
        (sequence_id, camera_instance_id, video_frame, person_id, source),
    )
    session.commit()


def clear_disabled_hand_edits(
    session: sqlite3.Connection,
    sequence_id: str,
    camera_instance_id: str,
    video_frame: int,
    side: str,
) -> None:
    """Clear disable-edits (not repositioning edits) within *side*'s
    21-keypoint range for this frame, as the "auto-detect" half of Idea 3's
    two-mode design (see the design doc's "Idea 3" section, "auto-detect vs
    keep existing state"): once the wrist/elbow has moved enough to trigger
    a fresh redetection, whatever was true about this hand's fingers before
    -- the original detection, or a prior "disable this" edit -- is
    presumed stale, and the fresh redetection should not be silently
    shadowed by it.

    A prior *repositioning* edit (is_outlier=False, an actual hand-placed
    x/y) is deliberately left untouched -- that is a much stronger claim
    ("here is the correct value") that a redetection should never discard;
    this is what makes "user decides to edit the auto-refined kps" still
    work after this function runs. Call this once, right after a
    successful `write_hand_refinement`, only when "auto-detect" mode is
    active -- in "keep existing state" mode, redetection (and this
    function) never runs at all, so prior edits are never touched by
    anything automatic.
    """
    from posetrak.detection.hand_refinement import _HAND_BASE_IDX, _HAND_N_KP

    edit_row = session.execute(
        "SELECT kp_blob, kp_mask FROM pose_observation_edits"
        " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
        (sequence_id, camera_instance_id, video_frame),
    ).fetchone()
    if edit_row is None:
        return

    edit_kp = np.frombuffer(bytes(edit_row["kp_blob"]), dtype=np.float32).reshape(-1, 3)
    mask = bytearray(bytes(edit_row["kp_mask"]))
    base = _HAND_BASE_IDX[side]
    changed = False
    for i in range(base, min(base + _HAND_N_KP, edit_kp.shape[0])):
        byte_idx, bit_idx = divmod(i, 8)
        if byte_idx >= len(mask) or not (mask[byte_idx] >> bit_idx) & 1:
            continue  # not edited at all
        if edit_kp[i, 2] == 0.0:
            continue  # a repositioning edit, not a disable -- leave it alone
        mask[byte_idx] &= ~(1 << bit_idx)
        changed = True

    if not changed:
        return
    if any(mask):
        session.execute(
            "UPDATE pose_observation_edits SET kp_mask = ?"
            " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
            (bytes(mask), sequence_id, camera_instance_id, video_frame),
        )
    else:
        session.execute(
            "DELETE FROM pose_observation_edits"
            " WHERE sequence_id = ? AND camera_instance_id = ? AND video_frame = ?",
            (sequence_id, camera_instance_id, video_frame),
        )
    session.commit()


def read_track_spans(
    session: sqlite3.Connection,
    detection_run_id: str,
    shot_video_id: str,
) -> list[dict]:
    rows = session.execute(
        "SELECT track_id, first_frame, last_frame FROM person_tracks "
        "WHERE detection_run_id=? AND shot_video_id=? ORDER BY track_id",
        (detection_run_id, shot_video_id),
    ).fetchall()
    return [dict(r) for r in rows]
