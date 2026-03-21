"""Import Pose2Sim camera calibration TOML into the posetrak registry database.

The Pose2Sim calibration TOML contains sections named ``cam1``, ``cam2``, …
(1-based). Each section provides intrinsic matrix, distortion coefficients,
and extrinsic rotation/translation vectors. This module imports only the
**intrinsic** parameters (Phase 1). Extrinsics are handled in Phase 2.

Camera IDs in the registry use 0-based integer strings: ``cam1`` → ``"0"``,
``cam2`` → ``"1"``, etc., matching the convention in ``calibrate_scale.py``.
"""

from __future__ import annotations

import datetime
import sqlite3
import struct
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from scripts.db.posetrak_db import generate_id


@dataclass
class CalibImportResult:
    """Result of a calibration TOML import operation.

    Attributes
    ----------
    camera_model_id:
        The registry ID of the shared camera model row created for this file.
    camera_instance_ids:
        Mapping from camera label (e.g. ``"Camera1"``) to the registry
        ``camera_instances.id`` for each imported camera.
    camera_mode_ids:
        Mapping from camera label to the registry ``camera_modes.id`` for each
        imported camera.
    intrinsics_ids:
        Mapping from camera label to the registry ``intrinsics_calibrations.id``
        for each imported camera.
    """

    camera_model_id: str
    camera_instance_ids: dict[str, str] = field(default_factory=dict)
    camera_mode_ids: dict[str, str] = field(default_factory=dict)
    intrinsics_ids: dict[str, str] = field(default_factory=dict)


def import_calib_toml(
    registry: sqlite3.Connection,
    calib_path: Path,
    *,
    width_px: int = 0,
    height_px: int = 0,
    nominal_fps: float = 0.0,
    codec: str = "",
    calibration_tool: str = "pose2sim",
    distortion_model: str = "radtan",
    calibrated_at: str | None = None,
    notes: str = "",
) -> CalibImportResult:
    """Import intrinsic calibration data from a Pose2Sim TOML file.

    One shared ``camera_models`` row is created for the entire file (since all
    cameras in a single calibration file are treated as an unnamed group).
    For each ``camN`` section a ``camera_instances``, ``camera_modes``, and
    ``intrinsics_calibrations`` row are inserted. All inserts are executed in a
    single transaction.

    Only intrinsics are imported (Phase 1). Extrinsic data (``rotation``,
    ``translation``) present in the TOML is silently ignored.

    The ``[metadata]`` section and any section whose key does not start with
    ``"cam"`` are skipped.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    calib_path:
        Path to the Pose2Sim calibration ``.toml`` file to import.
    width_px:
        Image width in pixels for the ``camera_modes`` row (0 = unknown).
    height_px:
        Image height in pixels for the ``camera_modes`` row (0 = unknown).
    nominal_fps:
        Nominal frame rate for the ``camera_modes`` row (0.0 = unknown).
    codec:
        Optional codec string stored in ``camera_modes``.
    calibration_tool:
        Name of the tool that produced the calibration (default ``"pose2sim"``).
    distortion_model:
        Distortion model identifier stored in ``intrinsics_calibrations``
        (default ``"radtan"``).
    calibrated_at:
        ISO-format date/datetime string for when the calibration was performed.
        Defaults to today's date (``datetime.date.today().isoformat()``).
    notes:
        Optional free-text notes stored in each ``intrinsics_calibrations`` row.

    Returns
    -------
    CalibImportResult
        IDs of all rows created in the registry.
    """
    if calibrated_at is None:
        calibrated_at = datetime.date.today().isoformat()

    with calib_path.open("rb") as fh:
        raw: dict[str, object] = tomllib.load(fh)

    # Collect camera sections: keys starting with "cam", excluding "metadata",
    # sorted lexicographically (cam1, cam10, cam2, … — use numeric sort on suffix).
    cam_keys = sorted(
        (k for k in raw if k.startswith("cam") and k != "metadata"),
        key=lambda k: int(k[3:]) if k[3:].isdigit() else float("inf"),
    )

    # --- Create one shared camera_model row for this calibration file ---
    model_id = generate_id()
    model_name = f"Imported from {calib_path.name}"

    result = CalibImportResult(camera_model_id=model_id)

    # Collect all rows before touching the DB so we fail fast on bad TOML.
    rows_instances: list[tuple[str, str, str, str]] = []  # (id, model_id, serial, label)
    rows_modes: list[tuple[str, str, int, int, float, str]] = []  # (id, model_id, w, h, fps, codec)
    rows_intrinsics: list[tuple[str, str, str, str, str, float, float, float, float, bytes, str]] = []

    for cam_key in cam_keys:
        vals: dict[str, object] = raw[cam_key]  # type: ignore[assignment]
        label: str = str(vals.get("name", cam_key))
        matrix: list[list[float]] = vals["matrix"]  # type: ignore[assignment]

        # Extract focal lengths and principal point from 3×3 camera matrix
        fx: float = float(matrix[0][0])
        fy: float = float(matrix[1][1])
        cx: float = float(matrix[0][2])
        cy: float = float(matrix[1][2])

        # Distortion coefficients — default to four zeros if absent
        dist_raw: list[float] | None = vals.get("distortions")  # type: ignore[assignment]
        dist: list[float] = [float(d) for d in dist_raw] if dist_raw is not None else [0.0, 0.0, 0.0, 0.0]
        dist_blob: bytes = struct.pack(f"<{len(dist)}d", *dist)

        instance_id = generate_id()
        mode_id = generate_id()
        intrinsics_id = generate_id()

        rows_instances.append((instance_id, model_id, "", label))
        rows_modes.append((mode_id, model_id, width_px, height_px, nominal_fps, codec))
        rows_intrinsics.append(
            (
                intrinsics_id,
                mode_id,
                calibrated_at,
                calibration_tool,
                distortion_model,
                fx,
                fy,
                cx,
                cy,
                dist_blob,
                notes,
            )
        )

        result.camera_instance_ids[label] = instance_id
        result.camera_mode_ids[label] = mode_id
        result.intrinsics_ids[label] = intrinsics_id

    # --- Insert everything in a single transaction ---
    with registry:
        registry.execute(
            "INSERT INTO camera_models (id, manufacturer, model_name, sensor_size) "
            "VALUES (?, ?, ?, ?)",
            (model_id, "", model_name, None),
        )
        registry.executemany(
            "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
            "VALUES (?, ?, ?, ?)",
            rows_instances,
        )
        registry.executemany(
            "INSERT INTO camera_modes "
            "(id, camera_model_id, width_px, height_px, nominal_fps, codec) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows_modes,
        )
        registry.executemany(
            "INSERT INTO intrinsics_calibrations "
            "(id, camera_mode_id, calibrated_at, calibration_tool, distortion_model, "
            "fx, fy, cx, cy, dist_coeffs, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows_intrinsics,
        )

    return result
