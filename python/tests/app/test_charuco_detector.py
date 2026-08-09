"""Tests for CharucoDetector / anchor_from_charuco_board (Phase 4).

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 4 ("Coordinate-system anchoring
from a ChArUco board"), and status.md's Phase 4 notes for the documented
deviation from that section's original solvePnP-based phrasing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.setup.extrinsics_solver import CamCalibState, ObsPoint, run_calibration
from app.setup.fiducial_markers import CharucoBoardDetection, CharucoDetector, anchor_from_charuco_board

_REAL_BOARD_IMAGE = Path(__file__).parent.parent / "data" / "charuco_board_sample.png"
_REAL_BOARD_SMALL_IN_4K_IMAGE = (
    Path(__file__).parent.parent / "data" / "charuco_board_small_in_4k_frame.png"
)


def _render_board_image(
    squares_x=5, squares_y=7, square_length=0.04, marker_length=0.02,
    dictionary="DICT_4X4_50", size=(500, 700), margin=30,
) -> np.ndarray:
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)
    gray = board.generateImage(size, marginSize=margin)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _detector(**kwargs) -> CharucoDetector:
    defaults = dict(squares_x=5, squares_y=7, square_length=0.04, marker_length=0.02)
    defaults.update(kwargs)
    return CharucoDetector(**defaults)


# ---------------------------------------------------------------------------
# CharucoDetector.detect
# ---------------------------------------------------------------------------


def test_detects_a_real_board():
    img = _render_board_image()
    detection = _detector().detect(img, video_id="cam_A", frame_idx=3)

    assert detection is not None
    assert len(detection.corners) == 24  # (5-1) * (7-1) interior corners
    for c in detection.corners:
        assert c.video_id == "cam_A"
        assert c.frame_idx == 3


def test_corner_local_xyz_is_metric_and_matches_square_length():
    img = _render_board_image(square_length=0.05)
    detection = _detector(square_length=0.05).detect(img)
    by_id = {c.corner_id: c for c in detection.corners}

    # Corners 0 and 1 are adjacent along one row -- exactly one square apart.
    dist = np.linalg.norm(by_id[0].local_xyz - by_id[1].local_xyz)
    assert dist == pytest.approx(0.05, abs=1e-9)


def test_local_xyz_is_z_zero_plane():
    img = _render_board_image()
    detection = _detector().detect(img)
    for c in detection.corners:
        assert c.local_xyz[2] == 0.0


def test_no_board_in_blank_image_returns_none():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    assert _detector().detect(blank) is None


def test_board_corner_local_xyz_matches_detected_corner():
    img = _render_board_image()
    detector = _detector()
    detection = detector.detect(img)
    c = detection.corners[0]
    np.testing.assert_array_equal(detector.board_corner_local_xyz(c.corner_id), c.local_xyz)


def test_unknown_dictionary_raises():
    with pytest.raises(ValueError):
        CharucoDetector(dictionary="DICT_NOT_A_REAL_ONE")


def test_mismatched_board_geometry_fails_to_detect_cleanly():
    """A detector configured for the wrong board size doesn't find (enough
    of) the real board -- degrades to None rather than misreading garbage
    corner positions as if they belonged to a different-sized board."""
    img = _render_board_image(squares_x=5, squares_y=7, square_length=0.04)
    wrong_detector = _detector(squares_x=8, squares_y=8, square_length=0.09)
    detection = wrong_detector.detect(img)
    assert detection is None or len(detection.corners) < 24


# ---------------------------------------------------------------------------
# estimate_board_pose (diagnostic only, per module docstring)
# ---------------------------------------------------------------------------


def test_estimate_board_pose_returns_plausible_pose():
    img = _render_board_image()
    detector = _detector()
    detection = detector.detect(img)

    # A camera looking roughly along +Z at a board filling most of the frame.
    K = np.array([[800.0, 0.0, 250.0], [0.0, 800.0, 350.0], [0.0, 0.0, 1.0]])
    result = detector.estimate_board_pose(detection, K, np.zeros(4))

    assert result is not None
    R, t = result
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    # Rotation matrix must actually be a rotation (orthonormal, det=+1).
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_estimate_board_pose_too_few_corners_returns_none():
    tiny = CharucoBoardDetection(corners=[])
    K = np.eye(3)
    assert CharucoDetector().estimate_board_pose(tiny, K, np.zeros(4)) is None


# ---------------------------------------------------------------------------
# anchor_from_charuco_board
# ---------------------------------------------------------------------------


def _fake_detection(video_id: str, frame_idx: int, corner_ids: list[int], local_xyz_by_id: dict[int, np.ndarray]):
    from app.setup.fiducial_markers import CharucoCornerObs
    corners = [
        CharucoCornerObs(
            corner_id=cid, video_id=video_id, frame_idx=frame_idx,
            px=10.0 * cid, py=20.0 * cid, local_xyz=local_xyz_by_id[cid],
        )
        for cid in corner_ids
    ]
    return CharucoBoardDetection(corners=corners)


def test_anchor_produces_fixed_control_points():
    local_xyz = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.04, 0.0, 0.0])}
    det = _fake_detection("cam_A", 5, [0, 1], local_xyz)

    cps = anchor_from_charuco_board({"cam_A": det})

    assert {cp.name for cp in cps} == {"charuco_c0", "charuco_c1"}
    for cp in cps:
        assert cp.world_xyz is not None
    cp0 = next(cp for cp in cps if cp.name == "charuco_c0")
    np.testing.assert_array_equal(cp0.world_xyz, [0.0, 0.0, 0.0])
    assert cp0.obs["cam_A"] == ObsPoint(frame_idx=5, px=0.0, py=0.0)


def test_anchor_board_face_up_default_keeps_local_xyz():
    local_xyz = {0: np.array([0.1, 0.2, 0.0])}
    det = _fake_detection("cam_A", 0, [0], local_xyz)
    cps = anchor_from_charuco_board({"cam_A": det}, board_face_up=True)
    np.testing.assert_array_equal(cps[0].world_xyz, [0.1, 0.2, 0.0])


def test_anchor_board_face_down_flips_y_and_z():
    local_xyz = {0: np.array([0.1, 0.2, 0.3])}
    det = _fake_detection("cam_A", 0, [0], local_xyz)
    cps = anchor_from_charuco_board({"cam_A": det}, board_face_up=False)
    np.testing.assert_array_equal(cps[0].world_xyz, [0.1, -0.2, -0.3])


def test_anchor_merges_same_corner_across_cameras():
    local_xyz = {0: np.array([0.0, 0.0, 0.0])}
    det_a = _fake_detection("cam_A", 1, [0], local_xyz)
    det_b = _fake_detection("cam_B", 2, [0], local_xyz)

    cps = anchor_from_charuco_board({"cam_A": det_a, "cam_B": det_b})

    assert len(cps) == 1
    assert set(cps[0].obs) == {"cam_A", "cam_B"}
    assert cps[0].obs["cam_A"].frame_idx == 1
    assert cps[0].obs["cam_B"].frame_idx == 2


def test_anchor_with_no_detections_returns_empty():
    assert anchor_from_charuco_board({}) == []


def test_anchor_partial_corner_overlap_across_cameras():
    """Cameras seeing different (but overlapping) subsets of the board still
    merge correctly per corner_id."""
    local_xyz = {0: np.array([0.0, 0.0, 0.0]), 1: np.array([0.04, 0.0, 0.0]),
                 2: np.array([0.08, 0.0, 0.0])}
    det_a = _fake_detection("cam_A", 0, [0, 1], local_xyz)
    det_b = _fake_detection("cam_B", 0, [1, 2], local_xyz)

    cps = {cp.name: cp for cp in anchor_from_charuco_board({"cam_A": det_a, "cam_B": det_b})}

    assert set(cps) == {"charuco_c0", "charuco_c1", "charuco_c2"}
    assert set(cps["charuco_c0"].obs) == {"cam_A"}
    assert set(cps["charuco_c1"].obs) == {"cam_A", "cam_B"}
    assert set(cps["charuco_c2"].obs) == {"cam_B"}


# ---------------------------------------------------------------------------
# run_calibration integration: anchored board corners are ordinary fixed
# ControlPoints, so they flow through the *existing*, unmodified PnP-init +
# fixed-CP BA path -- no new solver code needed for Phase 4's BA
# integration. This proves that path end-to-end using anchor_from_charuco_
# board's output, without needing a real board image or real footage.
# ---------------------------------------------------------------------------


def test_anchored_board_corners_solve_unposed_cameras():
    board = _detector()
    real_board_img = _render_board_image()
    detection = board.detect(real_board_img)
    assert detection is not None and len(detection.corners) >= 8

    # A known camera pose to project the board's own (real, exact)
    # local_xyz through -- this is the ground truth run_calibration must
    # recover purely from the anchored corners' world_xyz + projected pixels.
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    rvec_true = np.array([0.05, -0.1, 0.02])
    R_true, _ = cv2.Rodrigues(rvec_true)
    t_true = np.array([0.05, 0.02, 1.2])

    obj_pts = np.array([c.local_xyz for c in detection.corners])
    proj, _ = cv2.projectPoints(obj_pts, rvec_true, t_true, K, np.zeros(4))
    pixels = proj.reshape(-1, 2)

    projected_detection = CharucoBoardDetection(corners=[
        type(c)(corner_id=c.corner_id, video_id="cam_A", frame_idx=0,
                 px=float(px), py=float(py), local_xyz=c.local_xyz)
        for c, (px, py) in zip(detection.corners, pixels)
    ])
    cps = anchor_from_charuco_board({"cam_A": projected_detection})
    assert len(cps) >= 8  # needs >= 4 for PnP init; comfortably more here

    state = CamCalibState(
        video_id="cam_A", label="cam_A", K=K, K_orig=K.copy(),
        dist=np.zeros((1, 4)), fisheye=False,
    )
    result = run_calibration([state], control_points=cps, cp_only=True)

    solved = result.cameras["cam_A"]
    assert solved.R is not None
    np.testing.assert_allclose(solved.R, R_true, atol=1e-4)
    np.testing.assert_allclose(solved.t.flatten(), t_true, atol=1e-3)


# ---------------------------------------------------------------------------
# Real-photo regression: a calib.io-generated board, photographed at an
# angle on a tiled floor, found not to detect at all during live UI testing
# (2026-08-09). Traced to two settings gotchas, neither a board-size/
# hardware problem -- see CharucoDetector's docstring for both. Locked in
# here so this exact board's correct settings are never lost, and so a
# future change to CharucoDetector can't silently reintroduce either
# failure mode without a test noticing.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _REAL_BOARD_IMAGE.exists(), reason="real board fixture image not present")
class TestRealPhotographedBoard:
    """DICT_4X4_50, calib.io board: 20mm squares, 15mm markers.

    OpenCV's squares_x/squares_y turned out to be swapped relative to how
    calib.io's generator labels the board (11 wide x 8 tall, not 8x11), and
    calib.io's boards need legacy_pattern=True -- OpenCV changed the
    ChArUco marker-placement convention in 4.7 and calib.io's generator
    predates that change. Both failures are silent: ArUco marker detection
    itself succeeds either way (so the image/lighting/settings don't look
    broken), only the chessboard-corner interpolation step quietly finds
    nothing.
    """

    CORRECT_KWARGS = dict(
        dictionary="DICT_4X4_50", squares_x=11, squares_y=8,
        square_length=0.02, marker_length=0.015, legacy_pattern=True,
    )

    def _load(self) -> np.ndarray:
        img = cv2.imread(str(_REAL_BOARD_IMAGE))
        assert img is not None, f"failed to load {_REAL_BOARD_IMAGE}"
        return img

    def test_correct_settings_detect_the_board(self):
        detector = CharucoDetector(**self.CORRECT_KWARGS)
        detection = detector.detect(self._load())
        assert detection is not None
        assert len(detection.corners) >= 8

    def test_default_legacy_pattern_false_fails_on_this_board(self):
        """Locks in the exact silent failure this diagnostic found: same
        geometry, only legacy_pattern differs."""
        kwargs = dict(self.CORRECT_KWARGS, legacy_pattern=False)
        detector = CharucoDetector(**kwargs)
        detection = detector.detect(self._load())
        assert detection is None

    def test_swapped_squares_xy_fails_even_with_legacy_pattern(self):
        """Locks in the other silent failure: right dictionary and legacy
        flag, wrong axis assignment."""
        kwargs = dict(self.CORRECT_KWARGS, squares_x=8, squares_y=11)
        detector = CharucoDetector(**kwargs)
        detection = detector.detect(self._load())
        assert detection is None

    def test_detected_corners_are_metric_and_z_zero(self):
        detector = CharucoDetector(**self.CORRECT_KWARGS)
        detection = detector.detect(self._load())
        for c in detection.corners:
            assert c.local_xyz[2] == 0.0
        # Board-local spacing must match the declared 20mm square size --
        # spot-check via two corners one row apart in the (11,8) layout
        # (corner ids increase along a row, wrapping every squares_x - 1).
        by_id = {c.corner_id: c for c in detection.corners}
        common_ids = sorted(by_id)
        # Any two adjacent ids that are one grid step apart along a row.
        for cid in common_ids:
            if cid + 1 in by_id and (cid % 10) != 9:  # stay within one row (10 corners/row for 11 squares wide)
                dist = np.linalg.norm(by_id[cid].local_xyz - by_id[cid + 1].local_xyz)
                assert dist == pytest.approx(0.02, abs=1e-6)  # local_xyz is float32
                break
        else:
            pytest.fail("no adjacent same-row corner pair found in this crop to spot-check spacing")


# ---------------------------------------------------------------------------
# Real-frame regression: the same board, same footage, but as it actually
# appears in a full 4K (3840px-tall) camera frame rather than a tight
# hand-crop -- found not to detect at all during a second live-testing
# round (2026-08-09), even with the axis/legacy-pattern settings from the
# first round already correct. Traced to cv2.aruco's
# minMarkerPerimeterRate default (0.03) rejecting markers this small
# relative to the frame -- a third, independent gotcha, not a regression
# of the first two. See CharucoDetector's docstring.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _REAL_BOARD_SMALL_IN_4K_IMAGE.exists(), reason="real 4K-frame fixture image not present"
)
class TestBoardSmallInFullFrame:
    """Same board/settings as TestRealPhotographedBoard, but the fixture
    here is a real 3840px-tall camera frame with the board occupying only
    a small fraction of it (cropped in width only, to keep the fixture
    file smaller, without touching the height that actually triggers the
    bug -- minMarkerPerimeterRate is relative to the image's larger
    dimension)."""

    CORRECT_KWARGS = dict(
        dictionary="DICT_4X4_50", squares_x=11, squares_y=8,
        square_length=0.02, marker_length=0.015, legacy_pattern=True,
    )

    def _load(self) -> np.ndarray:
        img = cv2.imread(str(_REAL_BOARD_SMALL_IN_4K_IMAGE))
        assert img is not None, f"failed to load {_REAL_BOARD_SMALL_IN_4K_IMAGE}"
        assert max(img.shape[:2]) > 3000  # the bug only reproduces at real camera-frame scale
        return img

    def test_default_min_marker_perimeter_rate_finds_nothing(self):
        """Locks in the exact silent failure this diagnostic found: same
        board, same correct axis/legacy-pattern settings as
        TestRealPhotographedBoard, only the image scale differs."""
        detector = CharucoDetector(**self.CORRECT_KWARGS)
        assert detector.detect(self._load()) is None

    def test_lowered_min_marker_perimeter_rate_detects_the_board(self):
        detector = CharucoDetector(**self.CORRECT_KWARGS, min_marker_perimeter_rate=0.01)
        detection = detector.detect(self._load())
        assert detection is not None
        assert len(detection.corners) >= 8

    def test_arucodetector_has_the_same_gotcha_and_fix(self):
        """The underlying cv2 setting affects plain ArUco detection too --
        this project's ArucoDetector needed the identical fix."""
        from app.setup.fiducial_markers import ArucoDetector

        img = self._load()
        default = ArucoDetector(dictionary="DICT_4X4_50")
        lowered = ArucoDetector(dictionary="DICT_4X4_50", min_marker_perimeter_rate=0.01)

        assert len(lowered.detect(img)) > len(default.detect(img))
