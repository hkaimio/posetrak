"""import_extrinsics.py — Import extrinsic calibration from a Pose2Sim TOML file.

Reads the ``rotation`` (Rodrigues 3-vector) and ``translation`` fields from
each camera section of a Pose2Sim calibration TOML and stores them as
little-endian float64 blobs in the session database.

The Rodrigues vector is converted to a 3×3 rotation matrix using only NumPy
(no OpenCV dependency):

    θ = ‖rvec‖
    k = rvec / θ  (unit rotation axis)
    R = I + sin(θ)·K + (1−cos(θ))·K²

where K is the skew-symmetric cross-product matrix of k.
"""

from __future__ import annotations

import datetime
import sqlite3
import struct
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scripts.db.posetrak_db import generate_id


@dataclass
class ExtrinsicsImportResult:
    """Result of an extrinsic calibration import operation.

    Attributes
    ----------
    extrinsic_calibration_id:
        ID of the newly created ``extrinsic_calibrations`` row.
    camera_instance_ids:
        Mapping from TOML camera key (e.g. ``"cam1"``) to the registry
        ``camera_instances.id`` used for each extrinsic entry row.
    skipped:
        Set of TOML section keys that were skipped (not listed in
        *camera_instances* mapping).
    """

    extrinsic_calibration_id: str = ""
    camera_instance_ids: dict[str, str] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)


def _rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert a Rodrigues rotation vector to a 3×3 rotation matrix.

    Uses the Rodrigues formula without OpenCV:

        θ = ‖rvec‖
        if θ ≈ 0:  R = I
        else:
            k = rvec / θ
            K = skew-symmetric matrix of k
            R = I + sin(θ)·K + (1−cos(θ))·K²

    Parameters
    ----------
    rvec:
        1-D array of shape (3,) — the Rodrigues rotation vector.

    Returns
    -------
    np.ndarray
        Shape (3, 3) float64 rotation matrix.
    """
    rvec = np.asarray(rvec, dtype=np.float64).ravel()
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    k = rvec / theta
    kx, ky, kz = k
    K = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=np.float64,
    )
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R


def import_extrinsics(
    session: sqlite3.Connection,
    session_id: str,
    calib_path: Path,
    camera_instances: str | dict[str, str],
    *,
    method: str = "pose2sim",
    calibrated_at: str | None = None,
) -> ExtrinsicsImportResult:
    """Import extrinsic calibration data from a Pose2Sim TOML file.

    Creates one ``extrinsic_calibrations`` row and one ``extrinsic_entries``
    row per imported camera. All inserts are executed in a single transaction.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    session_id:
        ID of the ``mocap_sessions`` row this calibration belongs to.
    calib_path:
        Path to the Pose2Sim calibration ``.toml`` file.
    camera_instances:
        Camera instance assignment. Two forms:

        - **Homogeneous** (``str``): a single ``camera_instances.id`` UUID
          applied to every camera section in the TOML.
        - **Per-camera** (``dict[str, str]``): mapping from TOML section key
          (e.g. ``"cam1"``) to ``camera_instances.id``. Sections not listed
          are silently skipped.
    method:
        Name of the calibration method stored in ``extrinsic_calibrations``
        (default ``"pose2sim"``).
    calibrated_at:
        ISO-format date/datetime string. Defaults to today's date.

    Returns
    -------
    ExtrinsicsImportResult
        IDs of all rows created, plus the set of skipped section keys.
    """
    if calibrated_at is None:
        calibrated_at = datetime.date.today().isoformat()

    with calib_path.open("rb") as fh:
        raw: dict[str, object] = tomllib.load(fh)

    cam_keys = sorted(
        (k for k in raw if k.startswith("cam") and k != "metadata"),
        key=lambda k: int(k[3:]) if k[3:].isdigit() else float("inf"),
    )

    result = ExtrinsicsImportResult()
    calib_id = generate_id()
    result.extrinsic_calibration_id = calib_id

    rows_entries: list[tuple[str, str, bytes, bytes]] = []

    for cam_key in cam_keys:
        if isinstance(camera_instances, str):
            instance_id = camera_instances
        else:
            if cam_key not in camera_instances:
                result.skipped.add(cam_key)
                continue
            instance_id = camera_instances[cam_key]

        vals: dict[str, object] = raw[cam_key]  # type: ignore[assignment]
        rvec = np.array(vals["rotation"], dtype=np.float64)
        tvec = np.array(vals["translation"], dtype=np.float64).ravel()

        R = _rodrigues_to_matrix(rvec)
        R_flat = R.flatten()  # row-major 9 elements

        r_blob: bytes = struct.pack("<9d", *R_flat)
        t_blob: bytes = struct.pack("<3d", *tvec)

        rows_entries.append((calib_id, instance_id, r_blob, t_blob))
        result.camera_instance_ids[cam_key] = instance_id

    with session:
        session.execute(
            "INSERT INTO extrinsic_calibrations (id, session_id, calibrated_at, method) "
            "VALUES (?, ?, ?, ?)",
            (calib_id, session_id, calibrated_at, method),
        )
        session.executemany(
            "INSERT INTO extrinsic_entries "
            "(extrinsic_calibration_id, camera_instance_id, R, t) "
            "VALUES (?, ?, ?, ?)",
            rows_entries,
        )

    return result
