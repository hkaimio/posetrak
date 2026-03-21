"""Import Pose2Sim camera calibration TOML into the posetrak registry database.

The Pose2Sim calibration TOML contains sections named ``cam1``, ``cam2``, …
(1-based). Each section provides intrinsic matrix, distortion coefficients,
and extrinsic rotation/translation vectors. This module imports only the
**intrinsic** parameters (Phase 1). Extrinsics are handled in Phase 2.

Before calling :func:`import_calib_toml`, register the camera hardware in the
registry using :func:`~scripts.db.posetrak_db.create_camera_model` and
:func:`~scripts.db.posetrak_db.create_camera_mode`.  The importer then links
each imported camera to the supplied camera mode(s).
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
    camera_instance_ids:
        Mapping from camera label (e.g. ``"Camera1"``) to the registry
        ``camera_instances.id`` created for each imported camera.
    intrinsics_ids:
        Mapping from camera label to the registry ``intrinsics_calibrations.id``
        created for each imported camera.
    skipped:
        Set of TOML section keys (e.g. ``"cam3"``) that were present in the
        file but not listed in the per-camera mode mapping and therefore skipped.
    """

    camera_instance_ids: dict[str, str] = field(default_factory=dict)
    intrinsics_ids: dict[str, str] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)


def import_calib_toml(
    registry: sqlite3.Connection,
    calib_path: Path,
    camera_modes: str | dict[str, str],
    *,
    calibration_tool: str = "pose2sim",
    distortion_model: str = "radtan",
    calibrated_at: str | None = None,
    notes: str = "",
) -> CalibImportResult:
    """Import intrinsic calibration data from a Pose2Sim TOML file.

    The camera hardware (model + mode) must already exist in *registry* before
    calling this function. For each camera section that is imported, a
    ``camera_instances`` row and an ``intrinsics_calibrations`` row are
    inserted. All inserts are executed in a single transaction.

    Only intrinsics are imported (Phase 1). Extrinsic data (``rotation``,
    ``translation``) present in the TOML is silently ignored.

    The ``[metadata]`` section and any section whose key does not start with
    ``"cam"`` are always skipped.

    Parameters
    ----------
    registry:
        Open connection to a posetrak registry database.
    calib_path:
        Path to the Pose2Sim calibration ``.toml`` file to import.
    camera_modes:
        Camera mode assignment for cameras in the TOML. Two forms are accepted:

        - **Homogeneous** (``str``): a single ``camera_modes.id`` UUID applied
          to every camera in the file.
        - **Per-camera** (``dict[str, str]``): mapping from TOML section key
          (e.g. ``"cam1"``) to ``camera_modes.id``. Only cameras whose section
          key appears in the dict are imported; others are silently skipped.
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
        IDs of all rows created in the registry, plus the set of skipped
        section keys (only relevant for per-camera mode mappings).

    Raises
    ------
    ValueError
        If any camera mode ID in *camera_modes* does not exist in the registry.
    """
    if calibrated_at is None:
        calibrated_at = datetime.date.today().isoformat()

    # --- Validate all referenced mode IDs upfront and cache model IDs ---
    # mode_to_model: camera_mode_id → camera_model_id
    mode_to_model: dict[str, str] = {}

    if isinstance(camera_modes, str):
        unique_modes = {camera_modes}
    else:
        unique_modes = set(camera_modes.values())

    for mode_id in unique_modes:
        row = registry.execute(
            "SELECT camera_model_id FROM camera_modes WHERE id = ?",
            (mode_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"camera_mode_id '{mode_id}' not found in registry. "
                "Create the camera model and mode first with create_camera_model() "
                "and create_camera_mode()."
            )
        mode_to_model[mode_id] = row["camera_model_id"]

    # --- Parse TOML ---
    with calib_path.open("rb") as fh:
        raw: dict[str, object] = tomllib.load(fh)

    # Collect camera section keys sorted by numeric suffix.
    cam_keys = sorted(
        (k for k in raw if k.startswith("cam") and k != "metadata"),
        key=lambda k: int(k[3:]) if k[3:].isdigit() else float("inf"),
    )

    result = CalibImportResult()

    rows_instances: list[tuple[str, str, str, str]] = []
    rows_intrinsics: list[tuple[str, str, str, str, str, float, float, float, float, bytes, str]] = []

    for cam_key in cam_keys:
        # Resolve mode for this camera; skip if not in per-camera mapping.
        if isinstance(camera_modes, str):
            mode_id_for_cam = camera_modes
        else:
            if cam_key not in camera_modes:
                result.skipped.add(cam_key)
                continue
            mode_id_for_cam = camera_modes[cam_key]

        camera_model_id = mode_to_model[mode_id_for_cam]

        vals: dict[str, object] = raw[cam_key]  # type: ignore[assignment]
        label: str = str(vals.get("name", cam_key))
        matrix: list[list[float]] = vals["matrix"]  # type: ignore[assignment]

        fx: float = float(matrix[0][0])
        fy: float = float(matrix[1][1])
        cx: float = float(matrix[0][2])
        cy: float = float(matrix[1][2])

        dist_raw: list[float] | None = vals.get("distortions")  # type: ignore[assignment]
        dist: list[float] = (
            [float(d) for d in dist_raw] if dist_raw is not None else [0.0, 0.0, 0.0, 0.0]
        )
        dist_blob: bytes = struct.pack(f"<{len(dist)}d", *dist)

        instance_id = generate_id()
        intrinsics_id = generate_id()

        rows_instances.append((instance_id, camera_model_id, "", label))
        rows_intrinsics.append(
            (
                intrinsics_id,
                mode_id_for_cam,
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
        result.intrinsics_ids[label] = intrinsics_id

    with registry:
        registry.executemany(
            "INSERT INTO camera_instances (id, camera_model_id, serial_number, label) "
            "VALUES (?, ?, ?, ?)",
            rows_instances,
        )
        registry.executemany(
            "INSERT INTO intrinsics_calibrations "
            "(id, camera_mode_id, calibrated_at, calibration_tool, distortion_model, "
            "fx, fy, cx, cy, dist_coeffs, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows_intrinsics,
        )

    return result
