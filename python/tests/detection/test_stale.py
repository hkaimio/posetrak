# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Staleness of a mask-sourced detection run after the masks were edited, and its refresh."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from posetrak.db import seg_revisions as sr
from posetrak.detection.box_sources import MaskBoxSource
from posetrak.detection.pipeline import CameraJob, DetectionLoop
from posetrak.detection.stale import refresh_run, stale_frames

from .test_pipeline_loop import H, SHOT, SYNC, W, Estimator, cam_jobs, fake_reader, session  # noqa: F401

SEG, VID = "seg", "v-cam-a"
MASK_H, MASK_W = 45, 80          # half the frame's size: a mask pixel is 2x2 frame pixels


def png(x1: int, y1: int, x2: int, y2: int) -> bytes:
    mask = np.zeros((MASK_H, MASK_W), np.uint8)
    mask[y1:y2 + 1, x1:x2 + 1] = 1
    return cv2.imencode(".png", mask)[1].tobytes()


@pytest.fixture
def seg_run(session):  # noqa: F811
    session.execute("PRAGMA foreign_keys = OFF")
    session.execute("INSERT INTO camera_modes (id, camera_model_id, width_px, height_px) VALUES ('mode', 'm', ?, ?)", (W, H))
    session.execute("UPDATE capture_videos SET camera_mode_id = 'mode'")
    session.execute(
        "INSERT INTO seg_quality_runs (id, shot_id, time_start_s, time_end_s, created_at) VALUES (?, ?, 0, 1, '2026-01-01')",
        (SEG, SHOT))
    # Frames 0-4: the person occupies mask pixels 10..49 x 10..29, i.e. 40 x 20 mask pixels.
    session.executemany("INSERT INTO seg_masks (seg_quality_run_id, shot_video_id, frame_idx, mask_blob) VALUES (?, ?, ?, ?)",
                        [(SEG, VID, f, png(10, 10, 49, 29)) for f in range(5)])
    session.commit()
    return SEG


def detect(session, estimator=None):
    """A mask-sourced run over frames 0-4 of camera A, as ``detect run --segmentation`` makes it."""
    from app.pose.db_cache import create_detection_run, mark_run_complete

    source = MaskBoxSource(session, SEG, ["ann"], gate_margin_px=0)
    estimator = estimator or Estimator()
    run_id = create_detection_run(
        session, SHOT, SYNC, 0.0, 1.0, detector_model=source.name, pose_model=estimator.name,
        pose_input_width=estimator.input_size[1], pose_input_height=estimator.input_size[0], **source.run_inputs())
    DetectionLoop(session, run_id, source, estimator, reader=fake_reader).run(cam_jobs(0, 5)[:1])
    mark_run_complete(session, run_id)
    return run_id


def edit(session, changes: dict[int, bytes | None]):
    rev = sr.open_revision(session, SEG)
    for frame, blob in changes.items():
        sr.write_mask(session, SEG, VID, frame, rev, blob)
    sr.close_revision(session, SEG, rev)
    return rev


def test_a_run_is_not_stale_until_the_masks_change(session, seg_run):
    run_id = detect(session)
    assert session.execute("SELECT seg_revision FROM detection_runs WHERE id = ?", (run_id,)).fetchone()[0] == 0
    assert stale_frames(session, run_id) == []


def test_only_frames_whose_box_moved_beyond_the_tolerance_are_listed(session, seg_run):
    run_id = detect(session)
    # Model input is 24 px wide for a 80 px wide box (40 mask pixels): one input pixel is 80/24 = 3.33 frame pixels,
    # i.e. 1.67 mask pixels. Frame 1: right edge 1 mask pixel (2 frame px, 0.6 input px) -> within tolerance.
    # Frame 2: right edge 3 mask pixels (6 frame px, 1.8 input px) -> stale. Frame 3: person removed. Frame 4 unchanged.
    edit(session, {1: png(10, 10, 50, 29), 2: png(10, 10, 52, 29), 3: None})
    got = stale_frames(session, run_id)
    assert [(s.shot_video_id, s.video_frame, s.labels) for s in got] == [(VID, 2, (1,)), (VID, 3, (1,))]
    # A looser tolerance drops the small move; a tighter one adds frame 1.
    assert [s.video_frame for s in stale_frames(session, run_id, tolerance=2.0)] == [3]
    assert [s.video_frame for s in stale_frames(session, run_id, tolerance=0.1)] == [1, 2, 3]


def test_a_person_appearing_makes_the_frame_stale(session, seg_run):
    run_id = detect(session)
    session.execute("DELETE FROM person_detections WHERE video_frame = 2")
    session.commit()
    edit(session, {2: png(10, 10, 49, 29)})
    assert [s.video_frame for s in stale_frames(session, run_id)] == [2]


def test_a_run_not_made_from_masks_is_refused(session, seg_run):
    from app.pose.db_cache import create_detection_run

    run_id = create_detection_run(session, SHOT, SYNC, 0.0, 1.0, detector_model="yolox-x", pose_model="fake-pose")
    with pytest.raises(ValueError, match="not made from segmentation"):
        stale_frames(session, run_id)


def test_refresh_copies_unchanged_frames_and_reruns_the_stale_ones(session, seg_run, monkeypatch):
    monkeypatch.setattr("posetrak.detection.frame_readers.get_reader", lambda mode: fake_reader)
    run_id = detect(session)
    edit(session, {2: png(10, 10, 52, 29)})
    stale = stale_frames(session, run_id)
    assert [s.video_frame for s in stale] == [2]

    new_id = refresh_run(session, run_id, stale, estimator=Estimator())
    assert new_id != run_id
    new = session.execute("SELECT * FROM detection_runs WHERE id = ?", (new_id,)).fetchone()
    assert (new["status"], new["seg_quality_run_id"], new["seg_revision"], new["source_detection_run_id"]) == (
        "complete", SEG, 1, run_id)
    rows = session.execute(
        "SELECT video_frame, bbox_w FROM person_detections WHERE detection_run_id = ? ORDER BY video_frame", (new_id,)
    ).fetchall()
    assert [r["video_frame"] for r in rows] == [0, 1, 2, 3, 4]
    assert [r["bbox_w"] for r in rows] == [79.0, 79.0, 85.0, 79.0, 79.0]
    assert session.execute("SELECT COUNT(*) FROM detection_keypoints WHERE detection_run_id = ?", (new_id,)).fetchone()[0] == 5
    spans = session.execute("SELECT first_frame, last_frame FROM person_tracks WHERE detection_run_id = ?", (new_id,)).fetchall()
    assert [tuple(s) for s in spans] == [(0, 4)]
    session.execute(
        "INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name,"
        " first_frame, last_frame) SELECT ?, shot_video_id, track_id, 'A', first_frame, last_frame"
        " FROM person_tracks WHERE detection_run_id = ?", (run_id, run_id))
    again = refresh_run(session, run_id, stale, estimator=Estimator())
    assert [tuple(r) for r in session.execute(
        "SELECT track_id, person_name FROM detection_track_assignments WHERE detection_run_id = ?", (again,))] == [
        (r["track_id"], "A") for r in session.execute(
            "SELECT track_id FROM person_tracks WHERE detection_run_id = ?", (run_id,))]
    # The old run is untouched and is now stale; the new one is current.
    assert session.execute("SELECT COUNT(*) FROM person_detections WHERE detection_run_id = ?", (run_id,)).fetchone()[0] == 5
    assert stale_frames(session, new_id) == []


def test_refresh_copies_candidate_rows_and_links_the_rerun_frames_after_them(session, seg_run, monkeypatch):
    from .test_kp_candidates import CandidateEstimator, candidate_rows

    monkeypatch.setattr("posetrak.detection.frame_readers.get_reader", lambda mode: fake_reader)
    run_id = detect(session, CandidateEstimator())
    old = candidate_rows(session, run_id, VID)
    assert sorted(old) == [0, 1, 2, 3, 4]
    edit(session, {2: png(10, 10, 52, 29)})

    new_id = refresh_run(session, run_id, stale_frames(session, run_id), estimator=CandidateEstimator())
    new = candidate_rows(session, new_id, VID)
    assert sorted(new) == [0, 1, 2, 3, 4]
    for frame in (0, 1, 3, 4):                                    # unchanged frames keep their rows and ids
        np.testing.assert_array_equal(new[frame], old[frame])
    used = {int(t) for f in (0, 1, 3, 4) for t in new[f][:, 5]}
    assert used.isdisjoint({int(t) for t in new[2][:, 5]})        # the re-run frame starts tracklets of its own
    import json
    assert "candidates" in json.loads(session.execute(
        "SELECT config_json FROM detection_runs WHERE id = ?", (new_id,)).fetchone()[0])
