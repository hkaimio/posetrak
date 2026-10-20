# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Staleness of mask-sourced detection runs, and their refresh into a new run.

A detection run made from segmentation masks used the person boxes of one mask revision
(``detection_runs.seg_revision``). After the masks were edited, a frame is stale when a person's box
now differs from the one the run used by more than a tolerance measured in pixels of the pose model's
input: a box edge that moved by ``d`` frame pixels moves the crop the model sees by
``d * input_width / box_width`` input pixels. Refreshing creates a new run (detection runs are
append-only): frames that are not stale are copied with ``INSERT ... SELECT``, stale frames are run
again, and the new run records the revision it read. See
docs/roadmap/features/segmentation-versioning/segmentation-versioning-design.md §2.5.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from posetrak.db.seg_revisions import newest_closed_revision
from posetrak.detection.box_sources import frame_boxes, load_mask

#: Pose model input used when the run did not record one (width, height).
_DEFAULT_INPUT = (192, 256)


@dataclass(frozen=True)
class StaleFrame:
    """A frame whose boxes changed beyond the tolerance."""

    shot_video_id: str
    video_frame: int
    labels: tuple[int, ...]  # mask labels whose box moved, appeared or disappeared


def _run_row(conn: sqlite3.Connection, run_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM detection_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"No detection run {run_id}.")
    if row["seg_quality_run_id"] is None:
        raise ValueError(f"Detection run {run_id} was not made from segmentation masks.")
    return row


def _frame_size(conn: sqlite3.Connection, shot_video_id: str) -> tuple[int, int] | None:
    row = conn.execute(
        "SELECT COALESCE(ic.image_width, cm.width_px) AS w, COALESCE(ic.image_height, cm.height_px) AS h"
        " FROM capture_videos cv"
        " LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id"
        " LEFT JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id"
        " WHERE cv.id = ?",
        (shot_video_id,),
    ).fetchone()
    return (int(row["w"]), int(row["h"])) if row is not None and row["w"] and row["h"] else None


def _label_count(conn: sqlite3.Connection, run_id: str, seg_id: str) -> int:
    """Number of mask labels the run's boxes came from: the segmentation's subjects or the highest track id it holds."""
    from posetrak.db.manage_person import persons_ordered_for_seg_run

    highest = conn.execute(
        "SELECT MAX(track_id) FROM person_detections WHERE detection_run_id = ? AND region_type = 'full_body'",
        (run_id,)).fetchone()[0]
    return max(len(persons_ordered_for_seg_run(conn, seg_id)), highest or 0)


def _moved(old: Sequence[float], new: tuple[float, float, float, float], input_wh: tuple[int, int], tol: float) -> bool:
    """Whether a box (centre x, centre y, w, h) moved by more than *tol* input pixels against (x1, y1, x2, y2)."""
    ox, oy, ow, oh = old
    dx = max(abs((ox - ow / 2) - new[0]), abs((ox + ow / 2) - new[2]))
    dy = max(abs((oy - oh / 2) - new[1]), abs((oy + oh / 2) - new[3]))
    return max(dx * input_wh[0] / max(ow, 1.0), dy * input_wh[1] / max(oh, 1.0)) > tol


def stale_frames(
    conn: sqlite3.Connection, run_id: str, *, tolerance: float = 1.0, revision: int | None = None
) -> list[StaleFrame]:
    """Frames of a mask-sourced detection run whose boxes differ from the ones *revision* gives.

    Parameters
    ----------
    tolerance:
        Largest box-edge movement, in pixels of the pose model's input, that does not make a frame stale.
    revision:
        Revision to compare against; the newest saved one by default.
    """
    run = _run_row(conn, run_id)
    seg_id, base = run["seg_quality_run_id"], run["seg_revision"]
    current = newest_closed_revision(conn, seg_id) if revision is None else revision
    if current is None or base == current:
        return []
    input_wh = (run["pose_input_width"] or _DEFAULT_INPUT[0], run["pose_input_height"] or _DEFAULT_INPUT[1])

    labels = list(range(1, _label_count(conn, run_id, seg_id) + 1))

    old: dict[tuple[str, int], dict[int, tuple]] = defaultdict(dict)
    for r in conn.execute(
        "SELECT shot_video_id, video_frame, track_id, bbox_x, bbox_y, bbox_w, bbox_h FROM person_detections"
        " WHERE detection_run_id = ? AND region_type = 'full_body'", (run_id,)
    ):
        old[(r[0], r[1])][r[2]] = tuple(r[3:7])
    spans: dict[str, tuple[int, int]] = {}
    for (svid, frame) in old:
        lo, hi = spans.get(svid, (frame, frame))
        spans[svid] = (min(lo, frame), max(hi, frame))

    # Frames to compare: with a known base revision only those the revisions after it touched.
    if base is not None:
        touched = {
            (r[0], r[1]) for r in conn.execute(
                "SELECT DISTINCT shot_video_id, frame_idx FROM seg_masks"
                " WHERE seg_quality_run_id = ? AND revision > ? AND revision <= ?", (seg_id, base, current))
        }
    else:
        from posetrak.db.seg_revisions import masked_frames

        touched = set(old)
        for svid in spans:
            touched |= {(svid, f) for f in masked_frames(conn, seg_id, svid, current)}

    sizes: dict[str, tuple[int, int] | None] = {}
    out: list[StaleFrame] = []
    for svid, frame in sorted(touched):
        if svid not in spans or not spans[svid][0] <= frame <= spans[svid][1]:
            continue
        if svid not in sizes:
            sizes[svid] = _frame_size(conn, svid)
        size = sizes[svid]
        if size is None:
            raise ValueError(f"No frame size known for video {svid}; the camera has no mode or intrinsics.")
        mask = load_mask(conn, seg_id, svid, frame, current)
        new = {} if mask is None else frame_boxes(mask, labels, *size)
        before = old.get((svid, frame), {})
        changed = tuple(sorted(
            label for label in before.keys() | new.keys()
            if (label in before) != (label in new) or _moved(before[label], new[label], input_wh, tolerance)
        ))
        if changed:
            out.append(StaleFrame(svid, frame, changed))
    return out


_COPIED_TABLES = ("person_detections", "detection_keypoints")


def refresh_run(
    conn: sqlite3.Connection,
    run_id: str,
    stale: Sequence[StaleFrame],
    *,
    revision: int | None = None,
    estimator=None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> str:
    """Create a new detection run from *run_id*, with the *stale* frames run again on *revision*.

    Frames that are not stale keep the detections and keypoints of *run_id*; the new run records
    the revision (the newest saved one by default) and names *run_id* as its source run. Returns
    the new run's id; its status is ``complete`` or ``failed``.

    *estimator* defaults to the pose model the old run used.
    """
    from app.pose.db_cache import create_detection_run, mark_run_complete
    from posetrak.detection.box_sources import MaskBoxSource
    from posetrak.detection.frame_readers import get_reader
    from posetrak.detection.pipeline import (
        DEFAULT_DECODE_MODE, CameraInfo, CameraJob, DetectionLoop, recompute_track_spans, run_config,
    )
    from posetrak.detection.pose_model_choice import create_estimator
    run = _run_row(conn, run_id)
    seg_id = run["seg_quality_run_id"]
    revision = newest_closed_revision(conn, seg_id) if revision is None else revision
    estimator = estimator or create_estimator(run["pose_model"], device=None)
    box_source = MaskBoxSource(conn, seg_id, [""] * _label_count(conn, run_id, seg_id), revision=revision)
    wanted: dict[str, set[int]] = defaultdict(set)
    for sf in stale:
        wanted[sf.shot_video_id].add(sf.video_frame)

    class _StaleOnly:
        """The mask box source, asked only for the stale frames."""

        def __getattr__(self, name):
            return getattr(box_source, name)

        def wants(self, cam, video_frame) -> bool:
            return video_frame in wanted[cam.shot_video_id]

    hand_pass = None
    if json.loads(run["config_json"] or "{}").get("hand_pass"):
        from posetrak.detection.hand_refinement import HandPass

        if HandPass.supports(run["pose_model"]):
            hand_pass = HandPass(pose_model=run["pose_model"])
    config = run_config(box_source, estimator, hand_pass, DEFAULT_DECODE_MODE, 6)
    config["refreshed_from"] = run_id
    new_id = create_detection_run(
        conn, run["shot_id"], run["sync_config_id"], run["time_start_s"], run["time_end_s"], run["detector_model"],
        run["pose_model"], trial_id=run["trial_id"], detector_version=run["detector_version"] or "",
        pose_version=run["pose_version"] or "", detector_conf=run["detector_conf"],
        pose_conf_threshold=run["pose_conf_threshold"], pose_input_width=run["pose_input_width"] or 0,
        pose_input_height=run["pose_input_height"] or 0, detector_type=run["detector_type"],
        config_json=json.dumps(config), seg_quality_run_id=seg_id, seg_revision=revision,
        source_detection_run_id=run_id,
    )
    try:
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS stale_frames (shot_video_id TEXT, video_frame INTEGER)")
        conn.execute("DELETE FROM stale_frames")
        conn.executemany("INSERT INTO stale_frames VALUES (?, ?)", [(s.shot_video_id, s.video_frame) for s in stale])
        for table in _COPIED_TABLES:
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if r[1] != "detection_run_id"]
            names = ", ".join(cols)
            conn.execute(
                f"INSERT INTO {table} (detection_run_id, {names}) SELECT ?, {names} FROM {table} t"
                " WHERE t.detection_run_id = ? AND NOT EXISTS (SELECT 1 FROM stale_frames s"
                "   WHERE s.shot_video_id = t.shot_video_id AND s.video_frame = t.video_frame)",
                (new_id, run_id),
            )
        # The preview crops of the unchanged frames (keyed by frame_idx), so the new run shows them too.
        crop_cols = [r[1] for r in conn.execute("PRAGMA table_info(frame_cache_entries)") if r[1] != "detection_run_id"]
        crop_names = ", ".join(crop_cols)
        conn.execute(
            f"INSERT INTO frame_cache_entries (detection_run_id, {crop_names}) SELECT ?, {crop_names}"
            " FROM frame_cache_entries t WHERE t.detection_run_id = ? AND NOT EXISTS (SELECT 1 FROM stale_frames s"
            "   WHERE s.shot_video_id = t.shot_video_id AND s.video_frame = t.frame_idx)",
            (new_id, run_id),
        )
        conn.execute(
            "INSERT INTO person_tracks (id, detection_run_id, shot_video_id, track_id, first_frame, last_frame)"
            " SELECT lower(hex(randomblob(16))), ?, shot_video_id, track_id, first_frame, last_frame"
            " FROM person_tracks WHERE detection_run_id = ?", (new_id, run_id))
        # A mask run's track ids are the mask labels, so the old run's track-to-person assignments
        # hold for the new one and it can be finalised the same way.
        conn.execute(
            "INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name,"
            " capture_person_id, subject_id, first_frame, last_frame) SELECT ?, shot_video_id, track_id, person_name,"
            " capture_person_id, subject_id, first_frame, last_frame FROM detection_track_assignments"
            " WHERE detection_run_id = ?", (new_id, run_id))
        conn.commit()

        jobs = []
        for svid, frames in wanted.items():
            cam = conn.execute(
                "SELECT cv.camera_instance_id, cv.file_path, ci.label FROM capture_videos cv"
                " LEFT JOIN camera_instances ci ON ci.id = cv.camera_instance_id WHERE cv.id = ?", (svid,)
            ).fetchone()
            info = CameraInfo(svid, cam["camera_instance_id"], cam["file_path"], 0.0, 0, 0.0, label=cam["label"] or "")
            jobs.append(CameraJob(info, min(frames), max(frames) + 1))
        if jobs:
            DetectionLoop(
                conn, new_id, _StaleOnly(), estimator, hand_pass=hand_pass, reader=get_reader(DEFAULT_DECODE_MODE)
            ).run(jobs, on_progress)
        recompute_track_spans(conn, new_id, sorted(wanted) or [])
        mark_run_complete(conn, new_id)
    except Exception:
        mark_run_complete(conn, new_id, "failed")
        raise
    return new_id
