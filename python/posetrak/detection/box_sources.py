# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""box_sources.py — where the person boxes of a detection run come from.

A box source gives the loop the person boxes of every frame of every camera:

* :class:`YoloxBoxSource` — a YOLOX detector with an IoU tracker per camera (track ids are fragments of a
  person's path, assigned to persons later);
* :class:`MaskBoxSource` — the tight box of each label of a segmentation run's masks (the label is the track id and
  names the subject), with the grown mask attached so that an estimator can gate its heatmaps;
* :class:`RunBoxSource` — the boxes and track ids of an existing detection run, so a new pose model can be run on
  them without detecting again.

The loop calls a source in two places so that each runs where it is cheap: :meth:`BoxSource.prepare` on the
decode thread of the camera (CPU work only: mask decoding, image resizing for the detector; it must be thread-safe),
and :meth:`BoxSource.detect` on the inference thread (the detector's model call and its tracker), frames of one
camera in order.

Design: docs/roadmap/features/detection-pipeline/detection-pipeline-design.md.
"""
from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from posetrak.db.seg_revisions import mask_at, newest_closed_revision
from posetrak.detection import stage_timing as st
from posetrak.detection.backends import PersonDetection

#: Boxes smaller than this (pixels, either side) are not persons.
MIN_BOX_PX = 4


@dataclass
class FrameWork:
    """One decoded frame on its way through the loop."""

    cam: Any                         # pipeline.CameraInfo
    video_frame: int
    image: np.ndarray | None         # None: the source wants no detection on this frame
    prepared: Any = None             # what BoxSource.prepare returned


class BoxSource(Protocol):
    name: str                        # recorded as detection_runs.detector_model
    version: str
    conf: float                      # recorded as detection_runs.detector_conf

    def start_camera(self, cam: Any, first_frame: int, last_frame: int) -> None:
        """Called once per camera before its frames are decoded (resets a tracker, loads boxes)."""

    def wants(self, cam: Any, video_frame: int) -> bool:
        """False for a frame that cannot have boxes; the loop then does not hand its image over."""

    def prepare(self, cam: Any, video_frame: int, image: np.ndarray) -> Any: ...

    def detect(self, works: Sequence[FrameWork]) -> list[list[PersonDetection]]: ...

    def run_inputs(self) -> dict[str, Any]:
        """``detection_runs`` columns that record what the boxes came from."""

    def options(self) -> dict:
        """Options that change the boxes, for ``detection_runs.config_json``."""


# ---------------------------------------------------------------------------
# YOLOX
# ---------------------------------------------------------------------------


class YoloxBoxSource:
    """A :class:`posetrak.detection.backends_rtmdet.YOLOXDetector` as a box source, one tracker per camera.

    The model is exported with a fixed batch size of one, so the frames are not batched into one call; the image
    is resized to the model input on the decode thread, and the model calls of all cameras are made by the single
    inference thread.
    """

    def __init__(self, detector) -> None:
        self._det = detector
        self.name, self.version, self.conf = detector.name, detector.version, detector._conf
        self._trackers: dict[str, Any] = {}

    def start_camera(self, cam, first_frame, last_frame) -> None:
        self._trackers[cam.shot_video_id] = self._det.new_tracker()

    def wants(self, cam, video_frame) -> bool:
        return True

    def prepare(self, cam, video_frame, image):
        with st.stage("box_source"):
            return self._det.model.preprocess(image)

    def detect(self, works):
        out = []
        for work in works:
            with st.stage("box_source"):
                padded, ratio = work.prepared
                outputs = self._det.model.inference(padded)[0]
                boxes = self._det.model.postprocess(outputs, ratio)
                out.append(self._trackers[work.cam.shot_video_id].update(boxes))
        return out

    def run_inputs(self) -> dict:
        return {}

    def options(self) -> dict:
        return {"detector": self.name, "conf": self.conf, "iou_thr": self._det._iou_thr}


# ---------------------------------------------------------------------------
# Segmentation masks
# ---------------------------------------------------------------------------


def load_mask(
    conn: sqlite3.Connection, seg_quality_run_id: str, shot_video_id: str, frame_idx: int, revision: int | None = None
) -> np.ndarray | None:
    """(H, W) uint8 label mask of a frame, at the resolution it is stored in, or None if not stored.

    With *revision* the mask the frame had at that revision (what a recorded result saw); without it the
    current mask.
    """
    if revision is None:
        row = conn.execute(
            "SELECT mask_blob FROM seg_masks_latest WHERE seg_quality_run_id=? AND shot_video_id=? AND frame_idx=?",
            (seg_quality_run_id, shot_video_id, frame_idx),
        ).fetchone()
        blob = None if row is None else bytes(row[0])
    else:
        blob = mask_at(conn, seg_quality_run_id, shot_video_id, frame_idx, revision)
    if blob is None:
        return None
    mask = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    return mask[:, :, 0] if mask.ndim == 3 else mask


def boxes_from_labels(mask: np.ndarray, labels: Sequence[int]) -> dict[int, tuple[int, int, int, int]]:
    """Inclusive pixel box (x1, y1, x2, y2) of each of *labels* present in *mask*, in the mask's own pixels."""
    out = {}
    for label in labels:
        on = mask == label
        cols, rows = on.any(0), on.any(1)
        if not cols.any():
            continue
        out[label] = (int(cols.argmax()), int(rows.argmax()),
                      len(cols) - 1 - int(cols[::-1].argmax()), len(rows) - 1 - int(rows[::-1].argmax()))
    return out


def frame_boxes(
    mask: np.ndarray, labels: Sequence[int], frame_width: int, frame_height: int
) -> dict[int, tuple[float, float, float, float]]:
    """Box (x1, y1, x2, y2) of each of *labels* in the frame's pixels, from a mask stored at a lower resolution.

    A label whose box is smaller than ``MIN_BOX_PX`` in either direction is left out.
    """
    mask_h, mask_w = mask.shape
    sx, sy = frame_width / mask_w, frame_height / mask_h
    out = {}
    for label, (x1, y1, x2, y2) in boxes_from_labels(mask, labels).items():
        box = (x1 * sx, y1 * sy, (x2 + 1) * sx - 1, (y2 + 1) * sy - 1)
        if box[2] - box[0] >= MIN_BOX_PX and box[3] - box[1] >= MIN_BOX_PX:
            out[label] = box
    return out


class _ReadOnlyDb:
    """A read-only connection per thread to the database file of *conn* (decode threads read masks in parallel)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._path = next((r[2] for r in conn.execute("PRAGMA database_list") if r[1] == "main"), "")
        if not self._path:
            raise ValueError("masks can only be read from a session database that is a file")
        self._local = threading.local()

    def get(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(Path(self._path).as_uri() + "?mode=ro", uri=True, timeout=60.0)
        return self._local.conn


class MaskBoxSource:
    """Person boxes from the labels of a segmentation run: label ``i + 1`` is ``persons_ordered[i]``.

    Boxes are found on the mask at its stored resolution and scaled to the frame, instead of resizing the mask
    to the frame and scanning it per label. With *gate_margin_px* the person's mask, grown by that margin, is
    attached to the detection (at *work_width* pixels across) for estimators that gate heatmaps with it.
    """

    name = "segmentation"
    version = ""
    conf = 0.3

    def __init__(self, conn: sqlite3.Connection, seg_quality_run_id: str, persons_ordered: Sequence[str], *,
                 gate_margin_px: float = 40.0, work_width: int = 480, revision: int | None = None) -> None:
        self.seg_quality_run_id = seg_quality_run_id
        # Pinned for the whole run: a save made while the run goes on does not change what it reads.
        self.revision = newest_closed_revision(conn, seg_quality_run_id) if revision is None else revision
        self._labels = list(range(1, len(persons_ordered) + 1))
        self._margin, self._work_width = gate_margin_px, work_width
        self._db = _ReadOnlyDb(conn)

    def start_camera(self, cam, first_frame, last_frame) -> None:
        pass

    def wants(self, cam, video_frame) -> bool:
        return True

    def prepare(self, cam, video_frame, image):
        with st.stage("box_source"):
            mask = load_mask(self._db.get(), self.seg_quality_run_id, cam.shot_video_id, video_frame, self.revision)
            if mask is None:
                return []
            height, width = image.shape[:2]
            work = (self._work_width, max(1, round(self._work_width * height / width)))
            radius = max(1, round(self._margin * work[0] / width))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
            detections = []
            for label, (bx1, by1, bx2, by2) in frame_boxes(mask, self._labels, width, height).items():
                w, h = bx2 - bx1, by2 - by1
                small = cv2.resize((mask == label).astype(np.uint8), work, interpolation=cv2.INTER_NEAREST)
                detections.append(PersonDetection(
                    track_id=label,
                    bbox=np.array([(bx1 + bx2) / 2.0, (by1 + by2) / 2.0, w, h], dtype=np.float32),
                    confidence=1.0,
                    mask=cv2.dilate(small, kernel) if self._margin > 0 else None,
                ))
            return detections

    def detect(self, works):
        return [work.prepared for work in works]

    def run_inputs(self) -> dict:
        return {"seg_quality_run_id": self.seg_quality_run_id, "seg_revision": self.revision}

    def options(self) -> dict:
        return {"mask_gate_margin_px": self._margin, "mask_work_width": self._work_width}


# ---------------------------------------------------------------------------
# An existing detection run
# ---------------------------------------------------------------------------


class RunBoxSource:
    """The boxes, confidences and track ids of the full-body detections of an existing detection run."""

    name = "detection-run"
    version = ""
    conf = 0.3

    def __init__(self, conn: sqlite3.Connection, source_run_id: str) -> None:
        self.source_run_id = source_run_id
        self._db = _ReadOnlyDb(conn)
        self._boxes: dict[tuple[str, int], list[PersonDetection]] = {}

    def start_camera(self, cam, first_frame, last_frame) -> None:
        for r in self._db.get().execute(
            "SELECT video_frame, track_id, bbox_x, bbox_y, bbox_w, bbox_h, confidence FROM person_detections"
            " WHERE detection_run_id=? AND shot_video_id=? AND region_type='full_body'"
            " AND video_frame >= ? AND video_frame < ? ORDER BY video_frame, track_id",
            (self.source_run_id, cam.shot_video_id, first_frame, last_frame),
        ):
            self._boxes.setdefault((cam.shot_video_id, r[0]), []).append(PersonDetection(
                track_id=r[1], bbox=np.array(r[2:6], dtype=np.float32), confidence=float(r[6])))

    def wants(self, cam, video_frame) -> bool:
        return (cam.shot_video_id, video_frame) in self._boxes

    def prepare(self, cam, video_frame, image):
        return self._boxes.get((cam.shot_video_id, video_frame), [])

    def detect(self, works):
        return [work.prepared for work in works]

    def run_inputs(self) -> dict:
        return {"source_detection_run_id": self.source_run_id}

    def options(self) -> dict:
        return {}
