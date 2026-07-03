"""Background wide-crop, person-cluster frame cache.

Implements "Background wide-crop frame cache" from
docs/roadmap/features/keypoint-editing/keypoint-editing-design.md:

- Per camera, walk the video sequentially in fixed-length "epochs".
- Per epoch, compute each tracked person's generous padded crop window
  (union of real detections widened past the epoch boundary, falling back
  to the nearest real detections before/after the epoch for gaps longer
  than the epoch -- see `_TrackWindow.raw_rect`).
- Cluster overlapping windows (with a merge-area guard) so nearby people
  share one cached crop instead of one each.
- Cache is scoped to the detection run, not a single person/panel, and is
  reference-counted across open `PersonPanel`s via `FrameCropCacheManager`
  so multiple people edited in the same trial share one cache.

Result tuples are `(jpeg_bytes, width_px, height_px, src_x, src_y, src_w,
src_h)` -- the same shape `_encode_crop` (`db_cache.py`) and the Phase 6
in-memory backfill results use, so callers can render them with the same
code path regardless of which layer produced them.
"""

from __future__ import annotations

import bisect
import collections
import logging
import sqlite3
import threading

from PySide6.QtCore import QThread, Signal

_log = logging.getLogger(__name__)

Rect = tuple[float, float, float, float]  # x0, y0, x1, y1

EPOCH_SECONDS = 0.4
PAD_FRAC = 0.35
MERGE_GUARD = 1.3
GAP_MARGIN_FRAMES = 10       # Phase 6's existing +/-10 frame margin
GAP_SEARCH_SECONDS = 3.0     # bounded outward search for gaps longer than one epoch
MAX_LONG_EDGE = 1200
JPEG_QUALITY = 90


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def _pad_rect(x0: float, y0: float, x1: float, y1: float, pad_frac: float) -> Rect:
    w, h = x1 - x0, y1 - y0
    px, py = w * pad_frac, h * pad_frac
    return (x0 - px, y0 - py, x1 + px, y1 + py)


def _rects_overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _union_rect(a: Rect, b: Rect) -> Rect:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _rect_area(r: Rect) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def cluster_rects(
    rects: dict[int, Rect], guard: float = MERGE_GUARD
) -> list[tuple[list[int], Rect]]:
    """Union-find over overlapping padded rects.

    Merges two clusters only if the resulting union rect's area doesn't
    exceed `guard`x the sum of the merged rects' individual areas -- two
    rects that only graze at a corner produce a union much larger than
    either alone, and are better left as separate crops.
    """
    clusters = [([tid], r, _rect_area(r)) for tid, r in rects.items()]
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i] is None or clusters[j] is None:
                    continue
                ci, cj = clusters[i], clusters[j]
                if _rects_overlap(ci[1], cj[1]):
                    merged_rect = _union_rect(ci[1], cj[1])
                    merged_area = _rect_area(merged_rect)
                    if merged_area <= guard * (ci[2] + cj[2]):
                        clusters[i] = (ci[0] + cj[0], merged_rect, merged_area)
                        clusters[j] = None
                        changed = True
        clusters = [c for c in clusters if c is not None]
    return [(ids, rect) for ids, rect, _area in clusters]


class _TrackWindow:
    """Sorted per-track bbox lookup: windowed union + nearest-anchor gap search."""

    def __init__(self, det: dict[int, tuple[float, float, float, float]]) -> None:
        self.det = det  # frame -> (cx, cy, w, h)
        self.frames = sorted(det)

    def raw_rect(
        self, epoch_start: int, epoch_end: int, margin: int, gap_radius: int
    ) -> Rect | None:
        """Raw (unpadded) crop rect for this track over one epoch.

        First tries the union of real detections within the epoch widened by
        `margin` frames on each side (matching Phase 6's existing behaviour).
        If that window is empty -- a gap longer than the epoch itself, which
        can happen during a fast, high-motion sequence -- searches outward for
        the nearest real detection before/after the epoch, up to `gap_radius`
        frames, and unions whichever anchor(s) are found. Returns None only if
        the track has no detections within range on either side.
        """
        lo = bisect.bisect_left(self.frames, epoch_start - margin)
        hi = bisect.bisect_right(self.frames, epoch_end - 1 + margin)
        window = self.frames[lo:hi]
        if window:
            return self._union(window)

        pos = bisect.bisect_left(self.frames, epoch_start)
        before = self.frames[pos - 1] if pos > 0 else None
        after = self.frames[pos] if pos < len(self.frames) else None
        if before is not None and epoch_start - before > gap_radius:
            before = None
        if after is not None and after - (epoch_end - 1) > gap_radius:
            after = None
        anchors = [f for f in (before, after) if f is not None]
        if not anchors:
            return None
        return self._union(anchors)

    def _union(self, frames: list[int]) -> Rect:
        x0 = min(self.det[f][0] - self.det[f][2] / 2 for f in frames)
        y0 = min(self.det[f][1] - self.det[f][3] / 2 for f in frames)
        x1 = max(self.det[f][0] + self.det[f][2] / 2 for f in frames)
        y1 = max(self.det[f][1] + self.det[f][3] / 2 for f in frames)
        return (x0, y0, x1, y1)


def _encode_rect(img, rect: Rect):
    """Crop *img* to *rect* (already padded, full-frame pixel space) and JPEG
    encode, capping the long edge at MAX_LONG_EDGE. Returns
    (jpeg, wpx, hpx, src_x, src_y, src_w, src_h), matching `_encode_crop`'s
    shape, or None if the rect doesn't intersect the frame.
    """
    import cv2

    ih, iw = img.shape[:2]
    x0, y0, x1, y1 = rect
    xi0, yi0 = max(0, int(x0)), max(0, int(y0))
    xi1, yi1 = min(iw, int(x1)), min(ih, int(y1))
    if xi1 <= xi0 or yi1 <= yi0:
        return None
    crop = img[yi0:yi1, xi0:xi1]
    src_w, src_h = xi1 - xi0, yi1 - yi0
    long_edge = max(crop.shape[0], crop.shape[1])
    if long_edge > MAX_LONG_EDGE:
        scale = MAX_LONG_EDGE / long_edge
        crop = cv2.resize(
            crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale)))
        )
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return None
    return buf.tobytes(), crop.shape[1], crop.shape[0], xi0, yi0, src_w, src_h


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class WideCropExtractWorker(QThread):
    """Sequentially decode each camera's video and cache one JPEG per
    (camera, frame, person-cluster), keyed by an in-memory index.

    Uses its own SQLite connection so it never contends with the main
    thread's reads. `cameras` is a list of
    {shot_video_id, file_path, fps} covering every camera in the shot (not
    just the ones a given person happens to be assigned to), since
    clustering needs to see every tracked person's bboxes.
    """

    frame_ready = Signal(str, int)

    def __init__(
        self,
        db_path: str,
        det_run_id: str,
        cameras: list[dict],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._det_run_id = det_run_id
        self._cameras = cameras
        self._stop_event = threading.Event()
        self._priority_lock = threading.Lock()
        self._priority_request: tuple[str, int] | None = None
        self._index_lock = threading.Lock()
        # (svid, frame_idx) -> [(frozenset(track_ids), result_tuple), ...]
        self._index: dict[tuple[str, int], list[tuple[frozenset, tuple]]] = {}

    def stop(self) -> None:
        self._stop_event.set()
        self.wait(3000)

    def prioritise(self, svid: str, frame_idx: int) -> None:
        """Ensure the epoch containing (svid, frame_idx) is processed next."""
        with self._priority_lock:
            self._priority_request = (svid, frame_idx)

    def get_cluster_result(self, svid: str, frame_idx: int, track_id: int):
        with self._index_lock:
            entries = self._index.get((svid, frame_idx))
            if not entries:
                return None
            for track_ids, result in entries:
                if track_id in track_ids:
                    return result
        return None

    def run(self) -> None:  # noqa: C901
        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
        except Exception:
            _log.exception("wide-crop worker: failed to open DB")
            return
        try:
            for cam in self._cameras:
                if self._stop_event.is_set():
                    return
                try:
                    self._process_camera(conn, cam)
                except Exception:
                    _log.exception("wide-crop worker: error processing svid=%s", cam.get("shot_video_id"))
        finally:
            conn.close()

    def _process_camera(self, conn: sqlite3.Connection, cam: dict) -> None:
        import cv2

        svid = cam["shot_video_id"]
        fps = float(cam.get("fps") or 30.0)
        epoch_frames = max(1, round(EPOCH_SECONDS * fps))
        gap_radius = max(epoch_frames, round(GAP_SEARCH_SECONDS * fps))

        rows = conn.execute(
            "SELECT video_frame, track_id, bbox_x, bbox_y, bbox_w, bbox_h "
            "FROM person_detections WHERE detection_run_id=? AND shot_video_id=? "
            "AND region_type='full_body'",
            (self._det_run_id, svid),
        ).fetchall()
        if not rows:
            return

        by_track: dict[int, dict[int, tuple]] = collections.defaultdict(dict)
        for r in rows:
            by_track[r["track_id"]][r["video_frame"]] = (
                r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"],
            )

        cap = cv2.VideoCapture(cam.get("file_path", ""))
        if not cap.isOpened():
            _log.warning("wide-crop worker: cannot open video svid=%s", svid)
            return

        # Detection bboxes are stored in the resolution the detection pipeline
        # decoded frames at (see PoseWorker._run_pose). If this worker's own
        # cv2.VideoCapture ever decodes the same file at a *different*
        # resolution -- a stale/mismatched calibration record, a different
        # OpenCV/ffmpeg build, a re-encoded file -- bboxes would silently land
        # on the wrong region, and the error is far more visible on a 4K
        # camera than a 1080p one for the same relative mismatch. Rescale
        # defensively whenever the expected resolution (camera_modes /
        # intrinsics_calibrations, same source `_video_dims` in
        # content_panels.py uses) disagrees with what's actually decoded.
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        expected_w, expected_h = cam.get("expected_w"), cam.get("expected_h")
        if expected_w and expected_h and (actual_w != expected_w or actual_h != expected_h):
            sx, sy = actual_w / expected_w, actual_h / expected_h
            _log.warning(
                "wide-crop worker: resolution mismatch svid=%s expected=%dx%d "
                "decoded=%dx%d -- rescaling bboxes by (%.4f, %.4f)",
                svid, expected_w, expected_h, actual_w, actual_h, sx, sy,
            )
            for det in by_track.values():
                for f, (cx, cy, w, h) in det.items():
                    det[f] = (cx * sx, cy * sy, w * sx, h * sy)

        windows = {tid: _TrackWindow(det) for tid, det in by_track.items()}
        frame_min = min(min(det) for det in by_track.values())
        frame_max = max(max(det) for det in by_track.values())
        _log.info(
            "wide-crop worker: svid=%s file=%s decoded=%dx%d fps=%.2f "
            "tracks=%s frames=[%d,%d] epoch_frames=%d",
            svid, cam.get("file_path"), actual_w, actual_h, fps,
            sorted(by_track.keys()), frame_min, frame_max, epoch_frames,
        )

        starts = list(range((frame_min // epoch_frames) * epoch_frames, frame_max + 1, epoch_frames))
        queue: collections.deque = collections.deque(starts)
        queued_set = set(starts)
        cur_pos: int | None = None

        try:
            while queue:
                if self._stop_event.is_set():
                    return

                epoch_start = self._next_epoch(queue, queued_set, svid, epoch_frames)
                epoch_end = min(epoch_start + epoch_frames, frame_max + 1)

                # A rect can exist for this epoch even if zero real detections
                # fall strictly inside it -- that's exactly the long-gap case
                # `raw_rect`'s gap search handles, and precisely the frames
                # most in need of a cached crop for editing.
                rects: dict[int, Rect] = {}
                for tid, win in windows.items():
                    raw = win.raw_rect(epoch_start, epoch_end, GAP_MARGIN_FRAMES, gap_radius)
                    if raw is not None:
                        rects[tid] = _pad_rect(*raw, PAD_FRAC)

                if not rects:
                    continue

                clusters = cluster_rects(rects, MERGE_GUARD)

                if cur_pos != epoch_start:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, epoch_start)
                    cur_pos = epoch_start

                for f in range(epoch_start, epoch_end):
                    if self._stop_event.is_set():
                        return
                    ok, bgr = cap.read()
                    cur_pos += 1
                    if not ok:
                        continue
                    entries = []
                    for track_ids, rect in clusters:
                        result = _encode_rect(bgr, rect)
                        if result is not None:
                            entries.append((frozenset(track_ids), result))
                    if entries:
                        with self._index_lock:
                            self._index[(svid, f)] = entries
                        self.frame_ready.emit(svid, f)
        finally:
            cap.release()

    def _next_epoch(self, queue, queued_set: set, svid: str, epoch_frames: int) -> int:
        """Pop the next epoch to process, preferring a pending priority request."""
        with self._priority_lock:
            req = self._priority_request
            if req is not None and req[0] == svid:
                self._priority_request = None
            else:
                req = None
        if req is not None:
            req_epoch = (req[1] // epoch_frames) * epoch_frames
            if req_epoch in queued_set:
                queue.remove(req_epoch)
                queued_set.discard(req_epoch)
                return req_epoch
        epoch_start = queue.popleft()
        queued_set.discard(epoch_start)
        return epoch_start


# ---------------------------------------------------------------------------
# Reference-counted, detection-run-scoped cache manager
# ---------------------------------------------------------------------------

class FrameCropCacheManager:
    """Owns one `WideCropExtractWorker` per detection run, shared across every
    `PersonPanel` open on that run so multiple people edited in the same
    trial reuse one cache instead of each rebuilding their own.
    """

    _instances: dict[str, "FrameCropCacheManager"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, det_run_id: str) -> None:
        self.det_run_id = det_run_id
        self._refcount = 0
        self._worker: WideCropExtractWorker | None = None

    @classmethod
    def acquire(cls, db_path: str, det_run_id: str) -> "FrameCropCacheManager":
        with cls._registry_lock:
            mgr = cls._instances.get(det_run_id)
            if mgr is None:
                mgr = cls(det_run_id)
                cls._instances[det_run_id] = mgr
            mgr._refcount += 1
            if mgr._worker is None:
                mgr._start(db_path)
            return mgr

    def release(self) -> None:
        with FrameCropCacheManager._registry_lock:
            self._refcount -= 1
            if self._refcount <= 0:
                if self._worker is not None:
                    self._worker.stop()
                    self._worker = None
                FrameCropCacheManager._instances.pop(self.det_run_id, None)

    def _start(self, db_path: str) -> None:
        cameras: list[dict] = []
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT shot_id FROM detection_runs WHERE id=?", (self.det_run_id,)
            ).fetchone()
            if row is not None:
                cam_rows = conn.execute(
                    "SELECT cv.id AS shot_video_id, cv.file_path, cv.actual_fps, "
                    "       COALESCE(cm.width_px, ic.image_width) AS expected_w, "
                    "       COALESCE(cm.height_px, ic.image_height) AS expected_h "
                    "FROM capture_videos cv "
                    "LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id "
                    "LEFT JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id "
                    "WHERE cv.shot_id=?",
                    (row["shot_id"],),
                ).fetchall()
                cameras = [
                    {
                        "shot_video_id": r["shot_video_id"],
                        "file_path": r["file_path"] or "",
                        "fps": float(r["actual_fps"] or 30.0),
                        "expected_w": int(r["expected_w"]) if r["expected_w"] else None,
                        "expected_h": int(r["expected_h"]) if r["expected_h"] else None,
                    }
                    for r in cam_rows
                ]
            conn.close()
        except Exception:
            _log.exception("wide-crop cache: failed to load cameras for det_run=%s", self.det_run_id)

        worker = WideCropExtractWorker(db_path, self.det_run_id, cameras)
        worker.start()
        self._worker = worker

    @property
    def frame_ready(self) -> Signal:
        return self._worker.frame_ready

    def prioritise(self, svid: str, frame_idx: int) -> None:
        if self._worker is not None:
            self._worker.prioritise(svid, frame_idx)

    def get_cluster_result(self, svid: str, frame_idx: int, track_id: int):
        if self._worker is None:
            return None
        return self._worker.get_cluster_result(svid, frame_idx, track_id)
