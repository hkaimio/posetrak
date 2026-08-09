"""Tests for app.setup.fiducial_markers (Phase 3: ArUco marker detection).

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 3.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import MarkerGroup, ObsPoint, marker_local_corners
from app.setup.fiducial_markers import (
    ARUCO_DICTIONARIES,
    ArucoDetector,
    FiducialDetection,
    MarkerCornerObs,
    merge_detections_into_groups,
)


def _render_marker_image(marker_id: int, dictionary: str = "DICT_4X4_50") -> np.ndarray:
    """A real ArUco marker rendered to a BGR image, with a white border so
    detection actually succeeds (a marker touching the image edge is not
    detected)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
    gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    padded = cv2.copyMakeBorder(gray, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# ArucoDetector — real cv2.aruco round-trip
# ---------------------------------------------------------------------------


def test_detects_a_real_marker():
    img = _render_marker_image(3)
    detector = ArucoDetector()
    detections = detector.detect(img, video_id="cam_A", frame_idx=42)

    assert len(detections) == 1
    det = detections[0]
    assert det.marker_type == "aruco"
    assert det.marker_id == "3"
    assert len(det.corners) == 4


def test_corner_video_id_and_frame_idx_stamped():
    img = _render_marker_image(1)
    detector = ArucoDetector()
    detections = detector.detect(img, video_id="cam_B", frame_idx=17)
    for corner in detections[0].corners:
        assert corner.video_id == "cam_B"
        assert corner.frame_idx == 17


def test_corner_order_matches_marker_local_corners_convention():
    """Real cv2.aruco.ArucoDetector corner order is top-left, top-right,
    bottom-right, bottom-left -- confirmed against marker_local_corners()'s
    documented (and now verified, not assumed) convention."""
    img = _render_marker_image(5)
    detections = ArucoDetector().detect(img)
    pts = [(c.px, c.py) for c in sorted(detections[0].corners, key=lambda c: c.corner_index)]

    top_left, top_right, bottom_right, bottom_left = pts
    assert top_left[0] < top_right[0]        # left is left of right
    assert top_left[1] < bottom_left[1]      # top is above bottom (image y grows down)
    assert bottom_right[0] > bottom_left[0]
    assert bottom_right[1] > top_right[1]


def test_no_markers_in_blank_image_returns_empty():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    assert ArucoDetector().detect(blank) == []


def test_multiple_markers_in_one_image():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES["DICT_4X4_50"])
    canvas = np.full((400, 400), 255, dtype=np.uint8)
    canvas[20:170, 20:170] = cv2.aruco.generateImageMarker(aruco_dict, 0, 150)
    canvas[220:370, 220:370] = cv2.aruco.generateImageMarker(aruco_dict, 1, 150)
    img = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    detections = ArucoDetector().detect(img)
    assert {d.marker_id for d in detections} == {"0", "1"}


# ---------------------------------------------------------------------------
# Size configuration -> corner_local_xyz
# ---------------------------------------------------------------------------


def test_no_size_configured_leaves_local_xyz_none():
    img = _render_marker_image(2)
    detections = ArucoDetector().detect(img)
    assert detections[0].corner_local_xyz is None


def test_default_size_populates_local_xyz():
    img = _render_marker_image(2)
    detector = ArucoDetector(default_size=0.1)
    detections = detector.detect(img)
    expected = marker_local_corners(0.1)
    np.testing.assert_allclose(np.array(detections[0].corner_local_xyz), expected)


def test_per_marker_size_override_wins_over_default():
    img = _render_marker_image(7)
    detector = ArucoDetector(default_size=0.1, size_by_id={"7": 0.25})
    detections = detector.detect(img)
    expected = marker_local_corners(0.25)
    np.testing.assert_allclose(np.array(detections[0].corner_local_xyz), expected)


def test_unlisted_marker_falls_back_to_default_size():
    img = _render_marker_image(9)
    detector = ArucoDetector(default_size=0.1, size_by_id={"7": 0.25})
    detections = detector.detect(img)
    expected = marker_local_corners(0.1)
    np.testing.assert_allclose(np.array(detections[0].corner_local_xyz), expected)


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError):
        ArucoDetector(dictionary="DICT_NOT_A_REAL_ONE")


# ---------------------------------------------------------------------------
# merge_detections_into_groups
# ---------------------------------------------------------------------------


def _detection(marker_id: str, video_id: str, frame_idx: int, offset: float = 0.0) -> FiducialDetection:
    corners = [
        MarkerCornerObs(
            marker_type="aruco", marker_id=marker_id, corner_index=i,
            video_id=video_id, frame_idx=frame_idx,
            px=10.0 * i + offset, py=20.0 * i + offset,
        )
        for i in range(4)
    ]
    return FiducialDetection(marker_type="aruco", marker_id=marker_id, corners=corners)


def test_merge_creates_new_group_for_unseen_marker():
    groups: dict[str, MarkerGroup] = {}
    merge_detections_into_groups([_detection("3", "cam_A", 10)], groups, size=None)

    assert set(groups) == {"3"}
    mg = groups["3"]
    assert mg.marker_id == "3"
    assert mg.size is None
    assert set(mg.obs) == {"cam_A"}
    assert mg.obs["cam_A"][0] == ObsPoint(frame_idx=10, px=0.0, py=0.0)
    assert mg.obs["cam_A"][3] == ObsPoint(frame_idx=10, px=30.0, py=60.0)


def test_merge_accumulates_across_cameras():
    groups: dict[str, MarkerGroup] = {}
    merge_detections_into_groups([_detection("3", "cam_A", 10)], groups, size=None)
    merge_detections_into_groups([_detection("3", "cam_B", 25)], groups, size=None)

    assert set(groups) == {"3"}
    assert set(groups["3"].obs) == {"cam_A", "cam_B"}
    assert groups["3"].obs["cam_B"][0].frame_idx == 25


def test_merge_overwrites_same_camera_on_redetect():
    groups: dict[str, MarkerGroup] = {}
    merge_detections_into_groups([_detection("3", "cam_A", 10)], groups, size=None)
    merge_detections_into_groups([_detection("3", "cam_A", 99, offset=5.0)], groups, size=None)

    assert groups["3"].obs["cam_A"][0] == ObsPoint(frame_idx=99, px=5.0, py=5.0)


def test_merge_updates_size():
    groups: dict[str, MarkerGroup] = {}
    merge_detections_into_groups([_detection("3", "cam_A", 10)], groups, size=None)
    assert groups["3"].size is None

    merge_detections_into_groups([_detection("3", "cam_A", 10)], groups, size=0.05)
    assert groups["3"].size == 0.05


def test_merge_different_markers_produce_different_groups():
    groups: dict[str, MarkerGroup] = {}
    merge_detections_into_groups(
        [_detection("3", "cam_A", 10), _detection("7", "cam_A", 10)], groups, size=None
    )
    assert set(groups) == {"3", "7"}
