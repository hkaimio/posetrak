"""fiducial_markers.py — Marker detection framework for extrinsics calibration.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 3 ("Fiducial marker detection
framework"). Kept separate from ``extrinsics_solver.py``'s BA code so
marker-family-specific logic doesn't leak into the solver — adding a new
marker family (AprilTag, ...) later should only mean a new class in this
module, never a change to how the solver consumes detections.

``FiducialDetector.detect()`` is deliberately per-frame and stateless: it
answers only "what markers were seen where in *this* frame," with no claim
about whether a marker is fixed in the scene or moving (see the design doc's
"forward compatibility" note under section 3/6 and the corresponding
"Open questions" entry — a future moving-marker feature is a new consumer of
this same detection layer, not a rework of it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from app.setup.extrinsics_solver import ControlPoint, MarkerGroup, ObsPoint, marker_local_corners

# ArUco dictionary name -> cv2.aruco constant, exposed for UI population.
ARUCO_DICTIONARIES: dict[str, int] = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


@dataclass
class MarkerCornerObs:
    """One control point's observation for one corner of one marker in one camera."""
    marker_type: str      # "aruco", "charuco", "apriltag", ...
    marker_id: str        # dictionary-specific ID, or "<board>:<corner_id>" for ChArUco
    corner_index: int     # 0-3 for a quad marker; board-corner index for ChArUco
    video_id: str
    frame_idx: int
    px: float
    py: float


@dataclass
class FiducialDetection:
    """One marker's corners detected in one frame of one camera."""
    marker_type: str
    marker_id: str
    corners: list[MarkerCornerObs]                     # always 4 for a quad marker
    corner_local_xyz: list[np.ndarray] | None = None   # marker-local geometry, if size is known


class FiducialDetector(Protocol):
    def detect(self, image: np.ndarray, video_id: str, frame_idx: int) -> list[FiducialDetection]: ...


class ArucoDetector:
    """Detects plain (non-ChArUco-board) ArUco markers in an image.

    ``corner_local_xyz`` is populated (the (4, 3) marker-local offsets from
    ``extrinsics_solver.marker_local_corners()``, in the same corner order
    real ``cv2.aruco.ArucoDetector`` output uses -- verified directly, not
    assumed) when the marker's size is known via ``size_by_id`` or
    ``default_size``; otherwise it is left ``None`` and the marker's corners
    are treated as free, independently-triangulated correspondences (see
    ``MarkerGroup.as_control_points()``).
    """

    def __init__(
        self,
        dictionary: str = "DICT_4X4_50",
        default_size: float | None = None,
        size_by_id: dict[str, float] | None = None,
    ) -> None:
        if dictionary not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary!r}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
        params = cv2.aruco.DetectorParameters()
        self._cv_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.default_size = default_size
        self.size_by_id = dict(size_by_id or {})

    def size_for(self, marker_id: str) -> float | None:
        return self.size_by_id.get(marker_id, self.default_size)

    def detect(self, image: np.ndarray, video_id: str = "", frame_idx: int = 0) -> list[FiducialDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._cv_detector.detectMarkers(gray)
        detections: list[FiducialDetection] = []
        if ids is None:
            return detections
        for marker_corners, marker_id_arr in zip(corners, ids):
            marker_id = str(int(marker_id_arr[0]))
            pts = marker_corners.reshape(4, 2)
            corner_obs = [
                MarkerCornerObs(
                    marker_type="aruco", marker_id=marker_id, corner_index=i,
                    video_id=video_id, frame_idx=frame_idx,
                    px=float(x), py=float(y),
                )
                for i, (x, y) in enumerate(pts)
            ]
            size = self.size_for(marker_id)
            local_xyz = list(marker_local_corners(size)) if size else None
            detections.append(FiducialDetection(
                marker_type="aruco", marker_id=marker_id,
                corners=corner_obs, corner_local_xyz=local_xyz,
            ))
        return detections


@dataclass
class CharucoCornerObs:
    """One ChArUco board corner's observation in one camera/frame.

    Unlike an ArUco marker's 4 corners (a rigid quad, see
    ``MarkerCornerObs``), each ChArUco chessboard-intersection corner is its
    own independent point with a permanently fixed, exactly-known
    board-local position (``local_xyz``) -- there is no size ambiguity to
    resolve, since ``square_length`` already makes the whole board metric.
    """
    corner_id: int
    video_id: str
    frame_idx: int
    px: float
    py: float
    local_xyz: np.ndarray  # (3,) board-local coordinate; same for every observation of this corner_id


@dataclass
class CharucoBoardDetection:
    """One ChArUco board's corners detected in one frame of one camera."""
    corners: list[CharucoCornerObs]


class CharucoDetector:
    """Detects a ChArUco board (a grid of ArUco markers inside a
    checkerboard pattern) in an image.

    Every detected corner already has an exact, known board-local ``(x, y,
    0)`` (``board.getChessboardCorners()``) -- no size ambiguity, unlike a
    plain ArUco marker. See
    ``docs/roadmap/features/extrinsics-improvements/status.md``'s Phase 4
    notes for how this feeds the "set world origin/axes from the board"
    action (``fiducial_markers.anchor_from_charuco_board``) -- and for why
    that action does *not* actually need ``estimate_board_pose`` below,
    even though the design doc originally sketched it that way.

    **Two real gotchas found via a live test against a printed board**
    (see status.md's Phase 4 notes for the full story), both silent
    failures with no error, just zero corners detected:

    - ``squares_x``/``squares_y`` is OpenCV's own axis convention, which
      does not necessarily match how a third-party board generator labels
      "rows" vs. "columns" on the page. If detection finds 0 corners,
      try swapping the two values before suspecting anything else.
    - Boards generated by third-party tools that predate OpenCV 4.7's
      ChArUco marker-placement change (calib.io's generator among them)
      need ``legacy_pattern=True`` -- ``cv2.aruco.CharucoBoard`` silently
      assumes the *new* pattern otherwise, and marker detection succeeds
      (so it doesn't look broken) while chessboard-corner interpolation
      quietly finds nothing.
    """

    def __init__(
        self,
        dictionary: str = "DICT_4X4_50",
        squares_x: int = 5,
        squares_y: int = 7,
        square_length: float = 0.04,
        marker_length: float = 0.02,
        legacy_pattern: bool = False,
    ) -> None:
        if dictionary not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary!r}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
        self._board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, aruco_dict
        )
        self._board.setLegacyPattern(legacy_pattern)
        self._cv_detector = cv2.aruco.CharucoDetector(self._board)
        self.square_length = square_length

    def board_corner_local_xyz(self, corner_id: int) -> np.ndarray:
        return self._board.getChessboardCorners()[corner_id].copy()

    def detect(
        self, image: np.ndarray, video_id: str = "", frame_idx: int = 0
    ) -> CharucoBoardDetection | None:
        """Returns ``None`` if no board (or too few / collinear corners to
        be useful) is found -- not an error, the same "wrote nothing for
        that frame" graceful degradation as any other sparse detection in
        this project."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        charuco_corners, charuco_ids, _marker_corners, _marker_ids = self._cv_detector.detectBoard(gray)
        if charuco_ids is None or len(charuco_ids) < 4:
            return None
        if self._board.checkCharucoCornersCollinear(charuco_ids):
            return None

        board_corners = self._board.getChessboardCorners()
        corners = [
            CharucoCornerObs(
                corner_id=int(cid[0]), video_id=video_id, frame_idx=frame_idx,
                px=float(pt[0]), py=float(pt[1]), local_xyz=board_corners[int(cid[0])].copy(),
            )
            for pt, cid in zip(charuco_corners.reshape(-1, 2), charuco_ids)
        ]
        return CharucoBoardDetection(corners=corners)

    def estimate_board_pose(
        self, detection: CharucoBoardDetection, K: np.ndarray, dist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Solve the board's pose (board-local -> camera) in one camera's
        view via ``solvePnP``, given that camera's own intrinsics.

        Diagnostic use only (e.g. a future "does this detection look sane"
        preview) -- *not* on the critical path for
        ``anchor_from_charuco_board``'s world-coordinate assignment. See
        that function's docstring for why.
        """
        if len(detection.corners) < 4:
            return None
        obj_pts = np.array([c.local_xyz for c in detection.corners], dtype=np.float64)
        img_pts = np.array([[c.px, c.py] for c in detection.corners], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()


def anchor_from_charuco_board(
    detections_by_camera: dict[str, CharucoBoardDetection],
    *,
    board_face_up: bool = True,
) -> list[ControlPoint]:
    """Turn accumulated per-camera ChArUco corner detections into a set of
    fixed (``world_xyz``-set) ``ControlPoint``s -- one per physical corner
    ever detected, each with observations from every camera that saw it.

    **Deviates from the design doc's section 4 phrasing** ("solve the
    board's pose in that camera's frame via solvePnP ... maps through this
    one pose to a world xyz"): that describes computing the board's pose
    *relative to one reference camera* and using it as "world" -- but a
    camera's own world pose is exactly what calibration is trying to solve,
    so anchoring off one unsolved camera's frame doesn't actually give
    fixed world coordinates. The mechanism implemented here is simpler and
    doesn't have that problem: the board's own local coordinate frame
    (``CharucoDetector.board_corner_local_xyz`` -- already metric, since
    ``square_length`` is known) is used *directly* as the world frame, with
    only one user choice needed: whether the board's own +Z (pointing out
    of its printed face) should be world +Z ("board lying flat, face up")
    or flipped ("board mounted face-down"). This still delivers exactly the
    stated result -- scale, origin, and axes fixed together in one action --
    without needing any camera's intrinsics or an unsolved camera's pose at
    all. See ``docs/roadmap/features/extrinsics-improvements/status.md``'s
    Phase 4 notes for the full reasoning.

    *board_face_up=False* negates Y and Z (not Z alone) so the resulting
    frame stays right-handed.
    """
    by_corner_id: dict[int, ControlPoint] = {}
    for det in detections_by_camera.values():
        for c in det.corners:
            cp = by_corner_id.get(c.corner_id)
            if cp is None:
                world_xyz = c.local_xyz.copy()
                if not board_face_up:
                    world_xyz[1] *= -1
                    world_xyz[2] *= -1
                cp = ControlPoint(name=f"charuco_c{c.corner_id}", world_xyz=world_xyz)
                by_corner_id[c.corner_id] = cp
            cp.obs[c.video_id] = ObsPoint(frame_idx=c.frame_idx, px=c.px, py=c.py)
    return list(by_corner_id.values())


def merge_detections_into_groups(
    detections: list[FiducialDetection],
    groups: dict[str, MarkerGroup],
    size: float | None,
) -> None:
    """Fold one camera's detections into the accumulating marker-id -> MarkerGroup map.

    Mutates *groups* in place: creates a new ``MarkerGroup`` for a
    never-before-seen ``marker_id``, otherwise merges into the existing one.
    Re-detecting the same marker in the same camera (e.g. after scrubbing to
    a different frame) overwrites just that camera's corners for that
    marker, leaving every other camera's observations of it untouched --
    the same per-camera overwrite behavior Phase 2's ``_on_cam_click``
    already uses for manual control points.

    *size* updates the group's size unconditionally (last detector settings
    used win) -- there is only one size per physical marker, so there is no
    meaningful "merge" between two different size values, just "whichever
    was set most recently."
    """
    for det in detections:
        group = groups.get(det.marker_id)
        if group is None:
            group = MarkerGroup(marker_id=det.marker_id, size=size)
            groups[det.marker_id] = group
        else:
            group.size = size
        video_id = det.corners[0].video_id if det.corners else ""
        corners_by_index = {
            c.corner_index: ObsPoint(frame_idx=c.frame_idx, px=c.px, py=c.py)
            for c in det.corners
        }
        group.obs[video_id] = corners_by_index
