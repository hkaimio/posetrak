"""db_cache.py — Read/write helpers for detection pipeline DB tables."""
from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass

import cv2
import numpy as np

from posetrak.db.db import generate_id

_CROP_JPEG_QUALITY = 75
_CROP_TARGET_HEIGHT = 240


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
    detector_version: str = "",
    pose_version: str = "",
    detector_conf: float = 0.3,
    pose_conf_threshold: float = 0.3,
    pose_input_width: int = 0,
    pose_input_height: int = 0,
) -> str:
    run_id = generate_id()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    session.execute(
        "INSERT INTO detection_runs "
        "(id, shot_id, sync_config_id, time_start_s, time_end_s, "
        " detector_model, pose_model, detector_version, pose_version, "
        " detector_conf, pose_conf_threshold, "
        " pose_input_width, pose_input_height, status, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'running',?)",
        (run_id, shot_id, sync_config_id, time_start_s, time_end_s,
         detector_model, pose_model, detector_version, pose_version,
         detector_conf, pose_conf_threshold,
         pose_input_width, pose_input_height, now),
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


def _encode_crop(img: "np.ndarray", bbox: tuple) -> "tuple[bytes, int, int] | None":
    """Crop *img* by *bbox* (cx, cy, w, h) and return (jpeg, width_px, height_px)."""
    cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x2, y2 = min(img.shape[1], int(cx + w / 2)), min(img.shape[0], int(cy + h / 2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] > _CROP_TARGET_HEIGHT:
        scale = _CROP_TARGET_HEIGHT / crop.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), _CROP_TARGET_HEIGHT))
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, _CROP_JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes(), crop.shape[1], crop.shape[0]


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
                    jpeg, wpx, hpx = result
                    self._crop_rows.append((
                        self._svid, video_frame, "person_crop",
                        det.track_id, "full_body", wpx, hpx, jpeg, self._run_id,
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
                " width_px, height_px, image_data, detection_run_id) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
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
