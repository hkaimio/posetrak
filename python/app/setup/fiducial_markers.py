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

from app.setup.extrinsics_solver import MarkerGroup, ObsPoint, marker_local_corners

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
