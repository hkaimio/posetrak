"""import_calib_h5.py — Import intrinsic calibration from an HDF5 file into the registry.

Reads calibration HDF5 files produced by ``calibrate_intrinsics.py`` and inserts
an ``intrinsics_calibrations`` row (with optional undistortion maps) into the
posetrak registry database.

Expected HDF5 layout
--------------------
::

    /intrinsics/
        matrix              float64 (3, 3)   — original K from calibrateCamera()
        matrix_undistorted  float64 (3, 3)   — optimal K (getOptimalNewCameraMatrix)
        distortions         float64 (N,)     — dist coefficients
        .attrs['size']                       — (width, height) tuple or array
        .attrs['model_type']                 — 'standard' | 'fisheye'
        .attrs['error']                      — RMS reprojection error (float)
    /undistortion_maps/
        mapx                float32 (H, W)   — cv2.remap map for X
        mapy                float32 (H, W)   — cv2.remap map for Y

All ``/undistortion_maps/`` datasets are optional (``--no-maps`` mode).
"""

from __future__ import annotations

import datetime
import struct
import zlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from posetrak.db.db import generate_id


@dataclass
class CalibH5ImportResult:
    """Result of an HDF5 calibration import.

    Attributes
    ----------
    intrinsics_id:
        ID of the newly created ``intrinsics_calibrations`` row.
    camera_name:
        Value of ``intrinsics.attrs['camera_name']`` from the HDF5, if present.
    """

    intrinsics_id: str
    camera_name: str = ""


def import_calib_h5(
    registry: sqlite3.Connection,
    h5_path: Path,
    camera_mode_id: str,
    *,
    camera_instance_id: str | None = None,
    store_maps: bool = True,
    calibration_tool: str = "calibrate_intrinsics",
    calibrated_at: str | None = None,
    notes: str = "",
) -> CalibH5ImportResult:
    """Import intrinsic calibration from an HDF5 file into the registry.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    h5_path:
        Path to the calibration ``.h5`` file.
    camera_mode_id:
        Registry ``camera_modes.id`` to associate with this calibration.
    camera_instance_id:
        Optional registry ``camera_instances.id``.  Stored in notes if provided,
        not as a direct column (intrinsics belong to a mode, not an instance).
    store_maps:
        If ``True`` (default), read and store the undistortion maps from
        ``/undistortion_maps/`` (compressed with zlib). Set to ``False`` to
        skip maps (saves ~3 MB per camera in the registry but requires
        recomputation for frame undistortion).
    calibration_tool:
        Name of the tool that produced the HDF5 (default: ``"calibrate_intrinsics"``).
    calibrated_at:
        ISO date string. Defaults to today.
    notes:
        Optional free-text notes stored with the row.

    Returns
    -------
    CalibH5ImportResult
        ID of the inserted row and camera name from the HDF5 (if present).

    Raises
    ------
    FileNotFoundError
        If *h5_path* does not exist.
    KeyError
        If required HDF5 datasets are missing.
    """
    try:
        import h5py  # type: ignore[import-untyped]
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "h5py and numpy are required for HDF5 import. "
            "Install with: uv add h5py"
        ) from exc

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    if calibrated_at is None:
        calibrated_at = datetime.date.today().isoformat()

    with h5py.File(h5_path, "r") as hf:
        intr = hf["intrinsics"]

        # Required datasets
        matrix_orig: np.ndarray = np.array(intr["matrix"], dtype=np.float64)          # (3,3)
        matrix_undist: np.ndarray = np.array(intr["matrix_undistorted"], dtype=np.float64)  # (3,3)
        dist_coeffs: np.ndarray = np.array(intr["distortions"], dtype=np.float64)

        # Attributes
        size = intr.attrs.get("size")  # (width, height)
        image_width: int | None = int(size[0]) if size is not None else None
        image_height: int | None = int(size[1]) if size is not None else None

        model_type_raw: str = str(intr.attrs.get("model_type", "standard"))
        distortion_model = "fisheye" if model_type_raw == "fisheye" else "radtan"
        rms_error: float | None = float(intr.attrs["error"]) if "error" in intr.attrs else None
        camera_name: str = str(intr.attrs.get("camera_name", ""))

        # Extract scalar K from the undistorted (optimal) matrix
        fx = float(matrix_undist[0, 0])
        fy = float(matrix_undist[1, 1])
        cx = float(matrix_undist[0, 2])
        cy = float(matrix_undist[1, 2])

        # Encode original K matrix as little-endian float64 blob (row-major, 9 elements)
        matrix_orig_blob: bytes = struct.pack("<9d", *matrix_orig.flatten())

        # Encode dist_coeffs as little-endian float64 blob
        n_dist = len(dist_coeffs)
        dist_blob: bytes = struct.pack(f"<{n_dist}d", *dist_coeffs)

        # Optional undistortion maps
        mapx_blob: bytes | None = None
        mapy_blob: bytes | None = None
        if store_maps and "undistortion_maps" in hf:
            maps = hf["undistortion_maps"]
            if "mapx" in maps and "mapy" in maps:
                mapx = np.array(maps["mapx"], dtype=np.float32)
                mapy = np.array(maps["mapy"], dtype=np.float32)
                mapx_blob = zlib.compress(mapx.tobytes(), level=6)
                mapy_blob = zlib.compress(mapy.tobytes(), level=6)

    # Build notes string with optional camera_instance reference
    full_notes = notes
    if camera_instance_id:
        prefix = f"camera_instance_id={camera_instance_id}"
        full_notes = f"{prefix}; {notes}" if notes else prefix

    intrinsics_id = generate_id()
    with registry:
        registry.execute(
            "INSERT INTO intrinsics_calibrations "
            "(id, camera_mode_id, calibrated_at, calibration_tool, distortion_model, "
            "fx, fy, cx, cy, dist_coeffs, rms_error, notes, "
            "image_width, image_height, matrix_original, undistort_mapx, undistort_mapy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                intrinsics_id, camera_mode_id, calibrated_at, calibration_tool,
                distortion_model, fx, fy, cx, cy, dist_blob, rms_error, full_notes,
                image_width, image_height, matrix_orig_blob, mapx_blob, mapy_blob,
            ),
        )

    return CalibH5ImportResult(intrinsics_id=intrinsics_id, camera_name=camera_name)
