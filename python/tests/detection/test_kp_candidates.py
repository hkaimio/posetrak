# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Keypoint candidates: blob format, heatmap maxima, online linking, and the rows the detection loop writes."""
from __future__ import annotations

import numpy as np
import pytest

from app.pose.db_cache import KP_CANDIDATES_REGION
from posetrak.detection.backends import PoseResult
from posetrak.detection.kp_candidates import (
    CandidateOptions, PersonLinker, TrackletIds, decode_kp_candidates_blob, encode_kp_candidates_blob,
    extract_candidates, heatmap_maxima,
)
from tests.detection.test_pipeline_loop import Boxes, cam_jobs, make_loop, new_run, session  # noqa: F401

# Three candidates (keypoint_idx, x, y, peak, rank, tracklet_id): (0, 1, 2, 0.5, 0, 0), (0, 3, 4, 0.25, 1, -1),
# (76, 640.25, 360.5, 0.125, 0, 12). The C++ test (cpp/tests/test_kp_candidates_blob.cpp) decodes the same bytes.
FABRICATED_BLOB_HEX = (
    "0300000006000000"
    "000000000000803f000000400000003f0000000000000000"
    "0000000000004040000080400000803e0000803f000080bf"
    "0000984200102044" "0040b4430000003e0000000000004041"
)
FABRICATED = dict(keypoint_idx=[0, 0, 76], x=[1.0, 3.0, 640.25], y=[2.0, 4.0, 360.5], peak=[0.5, 0.25, 0.125],
                  rank=[0, 1, 0], tracklet_id=[0, -1, 12])


def test_blob_bytes_are_the_documented_layout():
    blob = encode_kp_candidates_blob(**FABRICATED)
    assert blob[:8] == np.array([3, 6], "<i4").tobytes()
    assert len(blob) == 8 + 3 * 6 * 4
    assert blob.hex() == FABRICATED_BLOB_HEX


def test_blob_round_trip():
    table = decode_kp_candidates_blob(bytes.fromhex(FABRICATED_BLOB_HEX))
    assert table.shape == (3, 6)
    np.testing.assert_array_equal(table[:, 0], [0, 0, 76])
    np.testing.assert_array_equal(table[:, 4], [0, 1, 0])
    np.testing.assert_array_equal(table[:, 5], [0, -1, 12])
    assert table[2, 1] == pytest.approx(640.25)


def test_decoder_accepts_extra_fields_and_refuses_fewer():
    wide = np.array([1, 7], "<i4").tobytes() + np.arange(7, dtype="<f4").tobytes()
    assert decode_kp_candidates_blob(wide).shape == (1, 7)
    with pytest.raises(ValueError):
        decode_kp_candidates_blob(np.array([1, 5], "<i4").tobytes() + np.zeros(5, "<f4").tobytes())


def gaussian(cx, cy, height, sigma=1.5):
    ys, xs = np.mgrid[:64, :48]
    return height * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


def test_maxima_are_ranked_thresholded_and_refined():
    heat = np.zeros((2, 64, 48), np.float32)
    heat[0] = gaussian(10.3, 20.0, 0.9) + gaussian(30, 40, 0.4) + gaussian(5, 5, 0.03)
    found = heatmap_maxima(heat, CandidateOptions())
    first = [m for m in found if m[0] == 0]
    assert [m[1] for m in first] == [0, 1]                    # the 0.03 bump is below min_peak
    assert first[0][4] > first[1][4]
    assert first[0][2] == pytest.approx(10.3, abs=0.1) and first[0][3] == pytest.approx(20.0, abs=0.1)
    assert [m for m in found if m[0] == 1] == []             # an empty heatmap has no maxima
    assert len(heatmap_maxima(heat, CandidateOptions(max_per_keypoint=1))) == 1


def test_extract_maps_cells_to_frame():
    heat = np.zeros((1, 64, 48), np.float32)
    heat[0] = gaussian(24, 32, 0.8)
    table = extract_candidates(heat, CandidateOptions(), lambda x, y: (x * 2 + 100, y * 2 + 50))
    assert table.dtype == np.float32 and table.shape == (1, 5)
    assert table[0, 1] == pytest.approx(148, abs=0.2) and table[0, 2] == pytest.approx(114, abs=0.2)


def row(kp, x, y):
    return [kp, x, y, 0.5, 0]


def test_linker_follows_a_moving_point_and_bridges_a_short_gap():
    linker = PersonLinker(TrackletIds(), max_gap_frames=3, gate_box_frac=0.04)
    ids = [linker.link(f, np.array([row(5, 100 + 3 * f, 200)], np.float32), 400.0)[0] for f in (0, 1, 2, 4)]
    assert ids == [0, 0, 0, 0]                                 # frame 3 missing: the gate grows over the gap
    far = linker.link(5, np.array([row(5, 300, 200)], np.float32), 400.0)[0]
    assert far == 1                                            # beyond 4% of the box: a new tracklet


def test_linker_closes_tracklets_after_the_gap_and_separates_keypoints():
    linker = PersonLinker(TrackletIds(), max_gap_frames=3)
    linker.link(0, np.array([row(1, 10, 10), row(2, 10, 10)], np.float32), 400.0)
    out = linker.link(1, np.array([row(2, 11, 10), row(1, 11, 10)], np.float32), 400.0)
    np.testing.assert_array_equal(out, [1, 0])                 # same positions, each keypoint its own tracklet
    late = linker.link(9, np.array([row(1, 11, 10)], np.float32), 400.0)[0]
    assert late not in (0, 1)


def test_linker_keeps_two_close_candidates_apart():
    linker = PersonLinker(TrackletIds())
    both = np.array([row(3, 100, 100), row(3, 112, 100)], np.float32)
    first = linker.link(0, both, 400.0)
    second = linker.link(1, both[::-1] + [0, 1, 0, 0, 0], 400.0)
    np.testing.assert_array_equal(second, first[::-1])


def test_persons_of_one_camera_share_unique_ids():
    ids = TrackletIds()
    a, b = PersonLinker(ids), PersonLinker(ids)
    pa = a.link(0, np.array([row(0, 1, 1)], np.float32), 100.0)
    pb = b.link(0, np.array([row(0, 1, 1)], np.float32), 100.0)
    assert pa[0] != pb[0]


class CandidateEstimator:
    """One strong and one weak candidate of keypoint 0 per person; the strong one moves 2 px per frame."""

    name, version, input_size = "fake-pose", "1", (32, 24)
    candidates = CandidateOptions()

    def estimate_batch(self, items):
        return [[PoseResult(d.track_id, np.full((3, 3), 1.0, np.float32), candidates=np.array(
            [[0, 50 + 2 * float(img[0, 0, 0]), 40, 0.9, 0], [0, 100, 40, 0.2, 1]], np.float32)) for d in dets]
                for img, dets in items]

    def options(self):
        return {"model": self.name}


def candidate_rows(session, run_id, svid):
    return {frame: decode_kp_candidates_blob(bytes(blob)) for frame, blob in session.execute(
        "SELECT video_frame, keypoints FROM detection_keypoints WHERE detection_run_id=? AND shot_video_id=?"
        " AND region_type=? ORDER BY video_frame", (run_id, svid, KP_CANDIDATES_REGION))}


def test_loop_writes_linked_candidate_rows_next_to_the_body_row(session):  # noqa: F811
    run_id = new_run(session)
    make_loop(session, run_id, Boxes(), CandidateEstimator()).run(cam_jobs(0, 6))
    rows = candidate_rows(session, run_id, "v-cam-a")
    assert sorted(rows) == list(range(6))
    assert all(len(t) == 2 for t in rows.values())
    strong = {int(t[0, 5]) for t in rows.values()}
    weak = {int(t[1, 5]) for t in rows.values()}
    assert len(strong) == 1 and len(weak) == 1 and strong != weak      # two tracklets over the six frames
    assert session.execute(
        "SELECT COUNT(*) FROM detection_keypoints WHERE detection_run_id=? AND region_type='full_body'"
        " AND shot_video_id='v-cam-a'", (run_id,)).fetchone()[0] == 6
    other = candidate_rows(session, run_id, "v-cam-b")
    assert {int(t) for tab in other.values() for t in tab[:, 5]} == strong | weak     # ids restart per camera


def test_a_second_pass_into_the_run_continues_the_ids(session):  # noqa: F811
    run_id = new_run(session)
    make_loop(session, run_id, Boxes(), CandidateEstimator()).run(cam_jobs(0, 3))
    before = {int(t) for tab in candidate_rows(session, run_id, "v-cam-a").values() for t in tab[:, 5]}
    make_loop(session, run_id, Boxes(), CandidateEstimator()).run(cam_jobs(6, 9))
    later = {int(t) for f, tab in candidate_rows(session, run_id, "v-cam-a").items() if f >= 6 for t in tab[:, 5]}
    assert before.isdisjoint(later)


def test_runs_without_candidates_write_no_candidate_rows(session):  # noqa: F811
    class Plain(CandidateEstimator):
        candidates = None

        def estimate_batch(self, items):
            return [[PoseResult(d.track_id, np.full((3, 3), 1.0, np.float32)) for d in dets] for _, dets in items]

    run_id = new_run(session)
    make_loop(session, run_id, Boxes(), Plain()).run(cam_jobs(0, 3))
    assert candidate_rows(session, run_id, "v-cam-a") == {}


def test_run_config_records_the_options():
    from posetrak.detection.pipeline import run_config

    config = run_config(Boxes(), CandidateEstimator(), None, "thread", 6)
    assert config["candidates"] == {"max_per_keypoint": 6, "min_peak": 0.05, "nms_cells": 5,
                                    "link": {"max_gap_frames": 3, "gate_box_frac": 0.04}}
    assert CandidateOptions.from_config(config) == CandidateOptions()
    assert CandidateOptions.from_config({}) is None


def test_candidates_are_refused_for_rtmpose():
    from posetrak.detection.pose_model_choice import create_estimator

    with pytest.raises(ValueError, match="SOMA-77 runs only"):
        create_estimator("rtmpose-l-133kp", candidates=CandidateOptions())
