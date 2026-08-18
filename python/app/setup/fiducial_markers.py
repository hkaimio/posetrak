# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

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

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from app.setup.extrinsics_solver import ControlPoint, MarkerGroup, ObsPoint, marker_local_corners

_log = logging.getLogger(__name__)

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

    **A third real gotcha, found via a live test against a full 4K
    (3840-tall) camera frame** (see ``CharucoDetector``'s own docstring for
    the first two, and status.md's Phase 4 notes for the full story):
    ``cv2.aruco``'s ``minMarkerPerimeterRate`` default (0.03) rejects any
    marker whose perimeter is under 3% of the image's larger dimension --
    for a 3840px-tall frame that's ~115px, easily bigger than a
    calibration-board-sized marker photographed from across a room. This
    is a silent failure with no error, and it *looks* identical to the
    ChArUco settings gotchas (zero detections) despite being an unrelated
    cause -- a real frame in this project's own test data went from 3/28
    markers found to 28/28 by lowering this to 0.01 alone, no other
    setting changed. ``min_marker_perimeter_rate`` below defaults to
    ``None`` (OpenCV's own 0.03) for backward compatibility; the UI
    defaults its own spin box lower, since this project's real use case
    (a board or marker seen from across a room, not held up to the lens)
    hits this far more often than a typical close-up desk-calibration
    scenario cv2's own default was tuned for.
    """

    def __init__(
        self,
        dictionary: str = "DICT_4X4_50",
        default_size: float | None = None,
        size_by_id: dict[str, float] | None = None,
        min_marker_perimeter_rate: float | None = None,
    ) -> None:
        if dictionary not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary!r}")
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
        params = cv2.aruco.DetectorParameters()
        if min_marker_perimeter_rate is not None:
            params.minMarkerPerimeterRate = min_marker_perimeter_rate
        self._cv_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        self.dictionary = dictionary
        self.min_marker_perimeter_rate = min_marker_perimeter_rate
        self.default_size = default_size
        self.size_by_id = dict(size_by_id or {})

    def size_for(self, marker_id: str) -> float | None:
        return self.size_by_id.get(marker_id, self.default_size)

    def detect(self, image: np.ndarray, video_id: str = "", frame_idx: int = 0) -> list[FiducialDetection]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids, rejected = self._cv_detector.detectMarkers(gray)
        found_ids = [] if ids is None else sorted(int(i) for i in ids.ravel())
        duplicate_ids = sorted({i for i in found_ids if found_ids.count(i) > 1})
        _log.info(
            "ArucoDetector.detect video_id=%r frame_idx=%d image_shape=%s "
            "dictionary=%s min_marker_perimeter_rate=%s -> %d marker(s) found "
            "%s (%d rejected candidates)%s",
            video_id, frame_idx, gray.shape, self.dictionary,
            self.min_marker_perimeter_rate, len(found_ids), found_ids,
            0 if rejected is None else len(rejected),
            f" [DUPLICATE ids: {duplicate_ids} -- same id decoded from more than one "
            f"candidate quad; if min_marker_perimeter_rate is very low this is "
            f"often a false positive, not two real markers]" if duplicate_ids else "",
        )
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

    **A third gotcha, found next, against a full 4K camera frame rather
    than a tightly-cropped photo**: ``min_marker_perimeter_rate`` -- see
    ``ArucoDetector``'s docstring, the same underlying cv2 setting applies
    here since a ChArUco board's markers are ordinary ArUco markers.
    Default ``None`` uses OpenCV's own 0.03; a board photographed from
    across a room in a multi-thousand-pixel-tall frame usually needs this
    lowered (0.01 resolved the real case that motivated this).
    """

    def __init__(
        self,
        dictionary: str = "DICT_4X4_50",
        squares_x: int = 5,
        squares_y: int = 7,
        square_length: float = 0.04,
        marker_length: float = 0.02,
        legacy_pattern: bool = False,
        min_marker_perimeter_rate: float | None = None,
    ) -> None:
        if dictionary not in ARUCO_DICTIONARIES:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary!r}")
        # cv2.aruco.CharucoBoard's own constructor enforces these same
        # constraints, but does so via a C++ assertion that surfaces to
        # Python as an opaque `SystemError: ... returned a result with an
        # exception set` -- not the ValueError callers would normally
        # catch, and not a message a UI can show a user. Check up front so
        # a bad size (most commonly: square/marker length swapped or equal)
        # fails with something actionable instead.
        if squares_x <= 1 or squares_y <= 1:
            raise ValueError(
                f"squares_x and squares_y must each be > 1, got "
                f"({squares_x}, {squares_y})"
            )
        if marker_length <= 0:
            raise ValueError(f"marker_length must be > 0, got {marker_length!r}")
        if square_length <= marker_length:
            raise ValueError(
                f"square_length ({square_length!r}) must be greater than "
                f"marker_length ({marker_length!r}) -- got these two swapped?"
            )
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dictionary])
        self._board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y), square_length, marker_length, aruco_dict
        )
        self._board.setLegacyPattern(legacy_pattern)
        self._cv_detector = cv2.aruco.CharucoDetector(self._board)
        if min_marker_perimeter_rate is not None:
            params = cv2.aruco.DetectorParameters()
            params.minMarkerPerimeterRate = min_marker_perimeter_rate
            self._cv_detector.setDetectorParameters(params)
        self.dictionary = dictionary
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_length = square_length
        self.marker_length = marker_length
        self.legacy_pattern = legacy_pattern
        self.min_marker_perimeter_rate = min_marker_perimeter_rate

    def board_corner_local_xyz(self, corner_id: int) -> np.ndarray:
        return self._board.getChessboardCorners()[corner_id].copy()

    def expected_marker_ids(self) -> set[str]:
        """This board's own ArUco marker ids (as ``ArucoDetector``-style
        string ids), for callers that want to tell a board's own markers
        apart from unrelated ArUco markers sharing the same dictionary --
        see ``page_extrinsics.py``'s "Detect ArUco" handler, which excludes
        these so the board's ~44 sub-markers don't flood the plain-ArUco
        marker list every time the board is also in frame.
        """
        return {str(int(i)) for i in self._board.getIds()}

    def detect(
        self, image: np.ndarray, video_id: str = "", frame_idx: int = 0
    ) -> CharucoBoardDetection | None:
        """Returns ``None`` if no board (or too few / collinear corners to
        be useful) is found -- not an error, the same "wrote nothing for
        that frame" graceful degradation as any other sparse detection in
        this project.

        Every call logs the exact configuration and result at INFO level,
        and logs a WARNING with the found-vs-expected marker IDs when it's
        about to return ``None`` -- see the class docstring's "gotchas"
        list for what a mismatch there usually means. This is deliberately
        on the hot path (not gated behind a verbose flag): board detection
        happens a handful of times per calibration session, not per frame
        of video, so the log volume is negligible.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        charuco_corners, charuco_ids, marker_corners, marker_ids = self._cv_detector.detectBoard(gray)
        found_marker_ids = [] if marker_ids is None else sorted(int(i) for i in marker_ids.ravel())
        expected_ids = sorted(int(i) for i in self._board.getIds())
        n_corners = 0 if charuco_ids is None else len(charuco_ids)
        duplicate_ids = sorted({i for i in found_marker_ids if found_marker_ids.count(i) > 1})

        _log.info(
            "CharucoDetector.detect video_id=%r frame_idx=%d image_shape=%s "
            "dictionary=%s squares=(%d,%d) square_length=%.4f marker_length=%.4f "
            "legacy_pattern=%s min_marker_perimeter_rate=%s "
            "-> %d/%d expected ArUco marker(s) found %s, %d charuco corner(s)%s",
            video_id, frame_idx, gray.shape,
            self.dictionary, self.squares_x, self.squares_y,
            self.square_length, self.marker_length,
            self.legacy_pattern, self.min_marker_perimeter_rate,
            len(found_marker_ids), len(expected_ids), found_marker_ids, n_corners,
            f" [DUPLICATE ids: {duplicate_ids}]" if duplicate_ids else "",
        )

        if charuco_ids is None or len(charuco_ids) < 4:
            unexpected = sorted(set(found_marker_ids) - set(expected_ids))
            _log.warning(
                "CharucoDetector.detect: only %d charuco corner(s) (need >= 4) for "
                "video_id=%r frame_idx=%d. Found %d ArUco marker(s) %s of this "
                "board's %d expected ids %s%s%s. If markers are found but few/no "
                "corners come out: (a) legacy_pattern may still be wrong for this "
                "board -- a wrong pattern maps found marker ids to the wrong grid "
                "positions, so detectBoard() can't interpolate any corners even "
                "though the markers themselves decoded fine; (b) the found markers "
                "may be too few or too scattered/non-adjacent for interpolation "
                "(cv2.aruco.CharucoParameters.minMarkers, default 2, requires "
                "neighbouring markers around a corner); (c) if min_marker_perimeter_rate "
                "is already low, try RAISING it instead -- going too low is not "
                "always better, it can flood detection with false-positive/"
                "misdecoded markers (see the duplicate-id warning above, if any) "
                "that confuse corner interpolation as badly as finding too few "
                "markers does; there is usually a narrow working band, not a "
                "one-directional dial.",
                n_corners, video_id, frame_idx, len(found_marker_ids), found_marker_ids,
                len(expected_ids), expected_ids,
                f" (found {len(unexpected)} id(s) NOT belonging to this board: {unexpected}"
                f" -- likely a different marker/board sharing this dictionary, "
                f"not this board's own markers)" if unexpected else "",
                f" (DUPLICATE ids decoded more than once: {duplicate_ids} -- a strong "
                f"sign min_marker_perimeter_rate is set too low: the same physical "
                f"marker, or scene noise, is being picked up multiple times as "
                f"different candidate quads that all decode to the same id, which "
                f"confuses detectBoard()'s corner interpolation)" if duplicate_ids else "",
            )
            return None
        if self._board.checkCharucoCornersCollinear(charuco_ids):
            _log.warning(
                "CharucoDetector.detect: %d charuco corner(s) found for video_id=%r "
                "frame_idx=%d but they are collinear (degenerate) -- ids=%s",
                n_corners, video_id, frame_idx, sorted(int(i) for i in charuco_ids.ravel()),
            )
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
    dictionary: str | None = None,
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
    was set most recently." *dictionary*, if given, updates the group's
    dictionary the same way -- left at ``MarkerGroup``'s own default when
    omitted, for callers (existing tests, ``characterize_rig_from_video.py``)
    that don't care about it.
    """
    for det in detections:
        group = groups.get(det.marker_id)
        if group is None:
            group = MarkerGroup(
                marker_id=det.marker_id, size=size,
                **({"dictionary": dictionary} if dictionary is not None else {}),
            )
            groups[det.marker_id] = group
        else:
            group.size = size
            if dictionary is not None:
                group.dictionary = dictionary
        video_id = det.corners[0].video_id if det.corners else ""
        corners_by_index = {
            c.corner_index: ObsPoint(frame_idx=c.frame_idx, px=c.px, py=c.py)
            for c in det.corners
        }
        group.obs[video_id] = corners_by_index


# ---------------------------------------------------------------------------
# Portable non-planar calibration rig (Phase 8, design doc section 9 Tier A)
# ---------------------------------------------------------------------------
#
# Unlike CharucoDetector's single flat board, a rig's markers are not
# required to be coplanar -- genuine depth variation is the whole point (see
# status.md's sixth live-testing round): a non-planar marker set has no
# IPPE-style tilt/planar-pose ambiguity to resolve at all, which removes the
# entire class of failure that motivated section 9, rather than adding
# another layer of disambiguation on top of it.


@dataclass
class MarkerRigConfig:
    """Resolved per-marker corner geometry for a portable calibration rig.

    ``marker_corners``: marker_id -> (4, 3) rig-local xyz, in the same
    corner order (top-left, top-right, bottom-right, bottom-left, as seen
    facing the marker) real ``cv2.aruco`` detection uses -- see
    ``ArucoDetector``'s docstring. This is deliberately the *resolved* form
    only (no parametric shape descriptor here) -- see ``load_rig_config``
    for how a compact source format (e.g. the design doc's ``"box"``
    descriptor) expands into this.

    ``marker_dictionaries``: marker_id -> its ArUco dictionary, populated
    only by ``load_marker_body_yaml`` (design doc section 10 -- a body's
    markers may span more than one dictionary; each marker's own
    ``dictionary:`` field is what says which). Empty for configs loaded via
    ``load_rig_config``'s older single-dictionary JSON shape, in which case
    ``MarkerRigDetector`` falls back to its own ``dictionary`` constructor
    argument for every marker, unchanged from before this field existed.

    ``reflective_dots``: name -> body-local xyz, for ``type: reflective_dot``
    entries (section 10). Not matched against ``marker_corners`` (a dot has
    no decodable id) and not consumed by ``MarkerRigDetector``/
    ``anchor_from_marker_rig`` at all yet -- correspondence/tracking for
    dots is out of scope beyond storing their definition-time position (see
    the design doc's "Reflective dots" subsection and its Open Questions
    entry). Kept here so a marker body's dots survive the load/save
    round-trip for whenever that future feature is built.
    """
    rig_id: str
    marker_corners: dict[str, np.ndarray] = field(default_factory=dict)
    marker_dictionaries: dict[str, str] = field(default_factory=dict)
    reflective_dots: dict[str, np.ndarray] = field(default_factory=dict)


def load_rig_config(path: str) -> MarkerRigConfig:
    """Load a ``MarkerRigConfig`` from a rig-config JSON file.

    **Superseded by ``load_marker_body_yaml``** (design doc section 10) --
    kept only because the existing prototype scripts
    (``tools/characterize_rig_from_video.py``, ``tools/
    test_rig_anchor_capture1.py``) still read/write this JSON shape; new
    marker body definitions should be authored as section 10's YAML
    instead. Only the ``"explicit"`` shape (raw per-marker corner arrays,
    matching ``MarkerRigConfig`` directly) is implemented; the ``"box"``
    parametric shape was deliberately never built here for the same reason
    documented on ``load_marker_body_yaml``.
    """
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("v") != 1:
        raise ValueError(f"Unsupported rig config version: {payload.get('v')!r}")
    shape = payload.get("shape")
    if shape == "box":
        raise NotImplementedError(
            '"box" shape parametric expansion is not implemented yet (see '
            "load_rig_config's docstring for why) -- use \"explicit\" for now."
        )
    if shape != "explicit":
        raise ValueError(f"Unknown rig config shape: {shape!r}")
    marker_corners = {
        marker_id: np.array(corners, dtype=np.float64)
        for marker_id, corners in payload["marker_corners"].items()
    }
    return MarkerRigConfig(rig_id=payload.get("rig_id", Path(path).stem), marker_corners=marker_corners)


def _marker_corners_from_pose(
    center: np.ndarray, normal: np.ndarray, up: np.ndarray, size: float,
) -> np.ndarray:
    """Resolve one marker's (4, 3) corners from its center + orientation.

    ``normal`` becomes the marker's local Z axis (outward face normal);
    ``up`` is Gram-Schmidt-corrected against it (doesn't need to be exactly
    perpendicular) to give the local Y axis; the local X axis follows as
    ``ey x ez`` to keep (X, Y, Z) right-handed -- the same construction the
    box rig config's gravity-up reframing (status.md, 2026-08-12) already
    used. Corner order matches ``marker_local_corners()``'s convention
    (top-left, top-right, bottom-right, bottom-left).
    """
    ez = normal / np.linalg.norm(normal)
    ey = up - np.dot(up, ez) * ez
    ey = ey / np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    hs = size / 2.0
    return np.array([
        center - ex * hs + ey * hs,   # top-left
        center + ex * hs + ey * hs,   # top-right
        center + ex * hs - ey * hs,   # bottom-right
        center - ex * hs - ey * hs,   # bottom-left
    ], dtype=np.float64)


def load_marker_body_yaml(yaml_content: str, *, rig_id: str | None = None) -> MarkerRigConfig:
    """Load a ``MarkerRigConfig`` from a marker body definition YAML string
    (design doc section 10 -- the canonical, always-fully-resolved format;
    see that section for the full schema and why it deliberately has no
    parametric-shape parsing here, unlike ``load_rig_config``'s deferred
    ``"box"`` JSON idea).

    Each marker entry gives its geometry either as raw ``corners`` (a
    (4, 3) array, used as-is -- the provenance-preserving form for solved,
    not designed, geometry) or as ``center``/``normal``/``up``/``size``
    (resolved via ``_marker_corners_from_pose``). ``type: reflective_dot``
    entries are parsed into ``MarkerRigConfig.reflective_dots`` but never
    fed into ``marker_corners`` -- see that field's docstring for why.

    Parameters
    ----------
    yaml_content:
        The YAML document text (e.g. a ``marker_body_definitions.
        yaml_content`` DB column, or a file already read into memory).
    rig_id:
        Overrides the resolved ``MarkerRigConfig.rig_id``; defaults to the
        document's own top-level ``name:``.

    Raises
    ------
    ValueError
        If two markers share a ``name`` (must be unique within one body),
        two coded markers share an ``id`` regardless of dictionary (see
        the message for why -- ``MarkerRigDetector``'s bare-id lookup
        requires this), or a marker is missing a field its ``type``
        requires.
    NotImplementedError
        If any marker uses a ``{slot: "..."}`` id reference -- templating
        is reserved syntax (design doc section 10, "Templating"), not
        resolvable without a not-yet-built ``marker_body_instances`` slot
        mapping.
    """
    payload = yaml.safe_load(yaml_content) or {}
    resolved_rig_id = rig_id or payload.get("name", "marker_body")

    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    marker_corners: dict[str, np.ndarray] = {}
    marker_dictionaries: dict[str, str] = {}
    reflective_dots: dict[str, np.ndarray] = {}

    for entry in payload.get("markers") or []:
        name = entry.get("name")
        if not name:
            raise ValueError(f"marker body {resolved_rig_id!r}: every marker needs a 'name'")
        if name in seen_names:
            raise ValueError(
                f"marker body {resolved_rig_id!r}: duplicate marker name {name!r} -- "
                "'name' must be unique within one body"
            )
        seen_names.add(name)

        marker_type = entry.get("type")
        if not marker_type:
            raise ValueError(f"marker body {resolved_rig_id!r}: marker {name!r} has no 'type'")

        if marker_type == "reflective_dot":
            center = entry.get("center")
            if center is None:
                raise ValueError(
                    f"marker body {resolved_rig_id!r}: reflective_dot {name!r} needs 'center'"
                )
            reflective_dots[name] = np.array(center, dtype=np.float64)
            continue

        # Coded marker types (aruco, apriltag, ...): need dictionary + id + geometry.
        dictionary = entry.get("dictionary")
        if not dictionary:
            raise ValueError(
                f"marker body {resolved_rig_id!r}: marker {name!r} (type={marker_type!r}) "
                "needs a 'dictionary'"
            )
        raw_id = entry.get("id")
        if isinstance(raw_id, dict):
            raise NotImplementedError(
                f"marker body {resolved_rig_id!r}: marker {name!r} uses a slot reference "
                f"({raw_id!r}) -- templating is reserved syntax, not implemented yet "
                "(see the design doc's 'Templating' subsection). Use a literal id for now."
            )
        if not raw_id and raw_id != 0:
            raise ValueError(
                f"marker body {resolved_rig_id!r}: marker {name!r} (type={marker_type!r}) "
                "needs an 'id'"
            )
        marker_id = str(raw_id)
        if marker_id in seen_ids:
            raise ValueError(
                f"marker body {resolved_rig_id!r}: duplicate marker id {marker_id!r} -- "
                "MarkerRigDetector matches markers by bare id, so ids must be unique across "
                "every dictionary in one body, not just within a single dictionary. If this "
                "body genuinely needs the same id reused across two dictionaries, that needs "
                "a composite-key lookup this loader doesn't implement yet."
            )
        seen_ids.add(marker_id)

        if "corners" in entry:
            corners = np.array(entry["corners"], dtype=np.float64)
            if corners.shape != (4, 3):
                raise ValueError(
                    f"marker body {resolved_rig_id!r}: marker {name!r} 'corners' must be a "
                    f"(4, 3) array, got shape {corners.shape}"
                )
        else:
            center, normal, up, size = (
                entry.get("center"), entry.get("normal"), entry.get("up"), entry.get("size"),
            )
            if center is None or normal is None or up is None or size is None:
                raise ValueError(
                    f"marker body {resolved_rig_id!r}: marker {name!r} needs either 'corners' "
                    "or all of 'center'/'normal'/'up'/'size'"
                )
            corners = _marker_corners_from_pose(
                np.array(center, dtype=np.float64), np.array(normal, dtype=np.float64),
                np.array(up, dtype=np.float64), float(size),
            )

        marker_corners[marker_id] = corners
        marker_dictionaries[marker_id] = dictionary

    return MarkerRigConfig(
        rig_id=resolved_rig_id,
        marker_corners=marker_corners,
        marker_dictionaries=marker_dictionaries,
        reflective_dots=reflective_dots,
    )


def load_marker_body_yaml_file(path: str, *, rig_id: str | None = None) -> MarkerRigConfig:
    """Load a ``MarkerRigConfig`` from a marker body definition YAML file
    on disk. See ``load_marker_body_yaml`` for the format; this is a thin
    file-reading wrapper around it, mirroring ``import_marker_body``/
    ``import_marker_body_str``'s file/string pairing in
    ``manage_marker_body.py``.

    *rig_id*, if given, still takes priority over the YAML's own top-level
    ``name:`` (same as calling ``load_marker_body_yaml`` directly) -- the
    filename is only used as a last-resort fallback, when neither *rig_id*
    nor the YAML itself supplies one.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()
    config = load_marker_body_yaml(content, rig_id=rig_id)
    if rig_id is None and config.rig_id == "marker_body":
        config.rig_id = Path(path).stem
    return config


class MarkerRigDetector:
    """Detects a portable calibration rig's own markers in an image.

    Wraps ``ArucoDetector`` (raw marker detection) + a rig-local corner
    lookup (``MarkerRigConfig``) -- corners are deliberately reported with
    ``corner_local_xyz=None`` (see ``detect``'s docstring); rig-local
    coordinates are looked up from ``self.config``, not carried on the
    ``FiducialDetection`` itself, mirroring how ``CharucoDetector`` keeps
    board-local coordinates on the detector/board rather than duplicating
    them per detection.
    """

    def __init__(
        self,
        config: MarkerRigConfig,
        dictionary: str = "DICT_4X4_50",
        min_marker_perimeter_rate: float | None = None,
    ) -> None:
        self.config = config
        # A body's markers may span more than one ArUco dictionary
        # (config.marker_dictionaries, populated by load_marker_body_yaml --
        # see MarkerRigConfig's docstring); *dictionary* is only the
        # fallback for markers with no per-marker entry there, which is
        # every marker for a config loaded via the older load_rig_config
        # JSON shape -- unchanged behaviour for those.
        dictionaries_used = set(config.marker_dictionaries.values()) or {dictionary}
        self._aruco_by_dict = {
            dict_name: ArucoDetector(
                dictionary=dict_name, min_marker_perimeter_rate=min_marker_perimeter_rate,
            )
            for dict_name in dictionaries_used
        }
        # Markers with no explicit per-marker dictionary use the fallback,
        # so its detector must exist -- but only when some marker in this
        # config actually needs it (no per-marker entry, i.e. an older
        # load_rig_config JSON-shaped config). A fully-specified config
        # (every one of its own markers has an explicit dictionary entry
        # -- e.g. one reconstructed from saved scene markers, "Load
        # Markers…", see page_extrinsics.py's
        # _load_rig_config_from_scene_marker_group) never touches this
        # fallback and must not get an extra detector for it: adding one
        # anyway used to scan a dictionary this config's own markers don't
        # use at all (this class' *dictionary* default, "DICT_4X4_50",
        # regardless of what the caller's markers actually are), and
        # detect()'s membership filter matches by bare marker id with no
        # dictionary check -- so a marker from a completely different
        # rig/config in that spuriously-scanned dictionary, sharing a
        # numeric id purely by coincidence with one of this config's own
        # markers, would silently pass through as if it were one of them.
        # See status.md's 2026-08-15 "purple marker mixup" entry for the
        # real report this fixes (a physical rig's own DICT_4X4_50 marker
        # detected as if it were a loaded DICT_5X5_50 scene marker).
        needs_fallback = any(
            mid not in config.marker_dictionaries for mid in config.marker_corners
        )
        if needs_fallback and dictionary not in self._aruco_by_dict:
            self._aruco_by_dict[dictionary] = ArucoDetector(
                dictionary=dictionary, min_marker_perimeter_rate=min_marker_perimeter_rate,
            )

    def detect(self, image: np.ndarray, video_id: str = "", frame_idx: int = 0) -> list[FiducialDetection]:
        """Detect ArUco markers, keeping only ones that belong to this rig.

        Runs one ``ArucoDetector`` per distinct dictionary the config's
        markers actually use and merges the results -- a marker sharing one
        of those dictionaries but not part of the rig (e.g. one of the
        scattered Tier-B tags) is silently excluded here -- the same "only
        this instrument's own markers" filtering
        ``CharucoDetector.expected_marker_ids()`` already does for a
        ChArUco board, generalized to a rig. Duplicate ids across different
        dictionaries are rejected at load time (see
        ``load_marker_body_yaml``), so a bare-id membership check here is
        safe even when the rig spans more than one dictionary.
        """
        merged: list[FiducialDetection] = []
        for detector in self._aruco_by_dict.values():
            all_dets = detector.detect(image, video_id=video_id, frame_idx=frame_idx)
            merged.extend(d for d in all_dets if d.marker_id in self.config.marker_corners)
        return merged

    def estimate_rig_pose(
        self, detections: list[FiducialDetection], K: np.ndarray, dist: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Diagnostic-only pose estimate (rig-local -> camera) via a single
        multi-marker ``solvePnP`` across every corner of every detected rig
        marker in one camera's view.

        Not on ``anchor_from_marker_rig``'s critical path -- same reasoning
        as ``CharucoDetector.estimate_board_pose``, see that docstring.
        Useful as a quick per-camera sanity check (e.g. "does this camera's
        detection alone look geometrically sane") before running a full
        multi-camera solve.
        """
        obj_pts: list[np.ndarray] = []
        img_pts: list[list[float]] = []
        for det in detections:
            corners_local = self.config.marker_corners.get(det.marker_id)
            if corners_local is None:
                continue
            for c in det.corners:
                obj_pts.append(corners_local[c.corner_index])
                img_pts.append([c.px, c.py])
        if len(obj_pts) < 4:
            return None
        obj_arr = np.array(obj_pts, dtype=np.float64)
        img_arr = np.array(img_pts, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj_arr, img_arr, K, dist)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten()


def anchor_from_marker_rig(
    detections_by_camera: dict[str, list[FiducialDetection]],
    config: MarkerRigConfig,
) -> list[ControlPoint]:
    """Turn accumulated per-camera rig-marker detections into a set of fixed
    (``world_xyz``-set) ``ControlPoint``s -- one per physical marker corner
    ever detected, each with observations from every camera that saw it.

    Generalizes ``anchor_from_charuco_board``'s mechanism, not the design
    doc's originally-sketched "solvePnP in one reference camera's frame"
    approach: the rig's own local coordinate frame (already metric, from
    however ``config`` was built/measured) is used *directly* as the world
    frame, exactly like the ChArUco anchor's board-local-frame trick. This
    still needs no camera intrinsics or an unsolved camera's pose to anchor
    -- and unlike the ChArUco case, there is no face-up/face-down axis
    choice to make here either, since a non-planar rig's own local frame has
    no equivalent "which side is this facing" ambiguity in the first place.
    """
    by_key: dict[tuple[str, int], ControlPoint] = {}
    for detections in detections_by_camera.values():
        for det in detections:
            corners_local = config.marker_corners.get(det.marker_id)
            if corners_local is None:
                continue
            for c in det.corners:
                key = (det.marker_id, c.corner_index)
                cp = by_key.get(key)
                if cp is None:
                    cp = ControlPoint(
                        name=f"rig_{det.marker_id}_c{c.corner_index}",
                        world_xyz=corners_local[c.corner_index].copy(),
                    )
                    by_key[key] = cp
                cp.obs[c.video_id] = ObsPoint(frame_idx=c.frame_idx, px=c.px, py=c.py)
    return list(by_key.values())
