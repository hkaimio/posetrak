"""import_session_yaml.py — Import a capture project YAML into the session database.

Reads a project YAML file and creates the corresponding posetrak DB records:
one ``MocapSession``, one ``SessionCamera`` per camera, one ``Shot`` per scene
(with ``extrinsic_calibration_id = NULL``), one ``ShotVideo`` per camera per
scene, and one ``SyncConfig`` per shot with one ``SyncPoint`` per camera at the
rough sync anchor frame.

YAML format
-----------
::

    name: "my-session"            # used as session notes (required)
    location: "gym"               # optional
    recorded_at: "2024-03-15"     # optional ISO date; defaults to today

    cameras:
      cam1:                       # key used as camera label for DB lookup
        video_path: "/path/to/cam1.mp4"  # stored in shot_videos.file_path
        fps: 120.0
        sync_frame: 5678          # frame number at the common sync moment (timestamp_s=0)
        camera_instance_id: "uuid"          # optional; looked up by label if absent
        camera_mode_id: "uuid"              # optional; required if session DB lacks the mode
        intrinsics_calibration_id: "uuid"   # optional

    scenes:
      - label: "scene1"
        cameras:
          cam1:
            first_frame: 6001
            last_frame: 7200

Notes
-----
* ``shots.extrinsic_calibration_id`` is left NULL; set it later with
  ``posetrak-db extrinsics import --shot <id>``.
* Camera lookup by label queries ``camera_instances.label = cam_key`` in the
  registry; raises ``ValueError`` if no unique match is found.
"""

from __future__ import annotations

import datetime
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from posetrak.db.db import (
    add_session_camera,
    add_shot_video,
    create_mocap_session,
    create_shot,
    generate_id,
)


@dataclass
class SessionYamlImportResult:
    """Result of a YAML session import.

    Attributes
    ----------
    session_id:
        ID of the created ``mocap_sessions`` row.
    shot_ids:
        Mapping from scene label to the created ``shots.id``.
    sync_config_ids:
        Mapping from scene label to the created ``sync_configs.id``.
    camera_instance_ids:
        Mapping from YAML camera key to the ``camera_instances.id`` used.
    """

    session_id: str = ""
    shot_ids: dict[str, str] = field(default_factory=dict)
    sync_config_ids: dict[str, str] = field(default_factory=dict)
    camera_instance_ids: dict[str, str] = field(default_factory=dict)


def _resolve_camera_instance(registry: sqlite3.Connection, label: str) -> str:
    """Look up a camera instance by label in the registry.

    Raises ``ValueError`` if not found or ambiguous.
    """
    rows = registry.execute(
        "SELECT id FROM camera_instances WHERE label = ?", (label,)
    ).fetchall()
    if len(rows) == 0:
        raise ValueError(
            f"No camera_instances row with label={label!r} found in registry. "
            "Register the camera first with `posetrak-db camera-model add` and "
            "`posetrak-db camera-mode add`, or provide camera_instance_id in the YAML."
        )
    if len(rows) > 1:
        ids = ", ".join(r[0] for r in rows)
        raise ValueError(
            f"Ambiguous camera label {label!r}: matches {len(rows)} instances ({ids}). "
            "Provide camera_instance_id explicitly in the YAML."
        )
    return rows[0][0]


def _resolve_camera_mode(registry: sqlite3.Connection, camera_instance_id: str) -> str:
    """Return the camera_mode_id associated with a camera instance via camera_model.

    Raises ``ValueError`` if no unique mode is found for the instance's model.
    """
    inst = registry.execute(
        "SELECT camera_model_id FROM camera_instances WHERE id = ?",
        (camera_instance_id,),
    ).fetchone()
    if inst is None:
        raise ValueError(f"camera_instance {camera_instance_id!r} not found in registry")
    modes = registry.execute(
        "SELECT id FROM camera_modes WHERE camera_model_id = ?",
        (inst["camera_model_id"],),
    ).fetchall()
    if len(modes) == 0:
        raise ValueError(
            f"No camera_modes found for model {inst['camera_model_id']!r}. "
            "Provide camera_mode_id in the YAML."
        )
    if len(modes) > 1:
        raise ValueError(
            f"Ambiguous: {len(modes)} modes for camera model {inst['camera_model_id']!r}. "
            "Provide camera_mode_id explicitly in the YAML."
        )
    return modes[0][0]


def _latest_intrinsics(registry: sqlite3.Connection, camera_mode_id: str) -> str | None:
    """Return the most recently calibrated intrinsics for a camera mode, or None."""
    row = registry.execute(
        "SELECT id FROM intrinsics_calibrations "
        "WHERE camera_mode_id = ? ORDER BY calibrated_at DESC LIMIT 1",
        (camera_mode_id,),
    ).fetchone()
    return row["id"] if row else None


def import_session_yaml(
    session: sqlite3.Connection,
    registry: sqlite3.Connection,
    yaml_path: Path,
    *,
    session_label: str = "",
    dry_run: bool = False,
) -> SessionYamlImportResult:
    """Import a capture project YAML into the session database.

    Parameters
    ----------
    session:
        Open connection to a posetrak session database.
    registry:
        Open connection to the posetrak registry database. Used to resolve
        camera instances and copy registry rows into the session DB.
    yaml_path:
        Path to the project YAML file.
    session_label:
        If non-empty, overrides the ``name`` field from the YAML as the session
        notes string.
    dry_run:
        If ``True``, validate inputs and print what would be created without
        writing anything to the database.

    Returns
    -------
    SessionYamlImportResult
        IDs of all created rows.

    Raises
    ------
    ValueError
        If camera lookup fails or required YAML fields are missing.
    FileNotFoundError
        If *yaml_path* does not exist.
    """
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "pyyaml is required for YAML import. Install with: uv add pyyaml"
        ) from exc

    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with yaml_path.open(encoding="utf-8") as fh:
        doc: dict = yaml.safe_load(fh)

    name: str = session_label or str(doc.get("name", ""))
    location: str = str(doc.get("location", ""))
    recorded_at: str | None = str(doc["recorded_at"]) if "recorded_at" in doc else None
    cameras_raw: dict = doc.get("cameras", {})
    scenes_raw: list = doc.get("scenes", [])

    if not cameras_raw:
        raise ValueError("YAML must contain a non-empty 'cameras' section")
    if not scenes_raw:
        raise ValueError("YAML must contain a non-empty 'scenes' section")

    # ------------------------------------------------------------------
    # Resolve camera instance / mode / intrinsics IDs for each camera key
    # ------------------------------------------------------------------
    cam_instance_ids: dict[str, str] = {}
    cam_mode_ids: dict[str, str] = {}
    cam_intrinsics_ids: dict[str, str | None] = {}
    cam_sync_frames: dict[str, int] = {}
    cam_video_paths: dict[str, str] = {}
    cam_fps: dict[str, float] = {}

    for cam_key, cam_cfg in cameras_raw.items():
        instance_id: str = cam_cfg.get("camera_instance_id") or _resolve_camera_instance(
            registry, cam_key
        )
        mode_id: str = cam_cfg.get("camera_mode_id") or _resolve_camera_mode(
            registry, instance_id
        )
        intrinsics_id: str | None = cam_cfg.get("intrinsics_calibration_id") or _latest_intrinsics(
            registry, mode_id
        )
        if intrinsics_id is None:
            raise ValueError(
                f"No intrinsics_calibration found for camera {cam_key!r} "
                f"(camera_mode_id={mode_id!r}). "
                "Import calibration first with `posetrak-db calib import-h5` or provide "
                "intrinsics_calibration_id in the YAML."
            )

        cam_instance_ids[cam_key] = instance_id
        cam_mode_ids[cam_key] = mode_id
        cam_intrinsics_ids[cam_key] = intrinsics_id
        cam_sync_frames[cam_key] = int(cam_cfg["sync_frame"])
        cam_video_paths[cam_key] = str(cam_cfg["video_path"])
        cam_fps[cam_key] = float(cam_cfg["fps"])

    if dry_run:
        _print_dry_run(name, location, recorded_at, cam_instance_ids, scenes_raw)
        return SessionYamlImportResult()

    result = SessionYamlImportResult()
    recorded_at_str = recorded_at or datetime.date.today().isoformat()

    # ------------------------------------------------------------------
    # Create session
    # ------------------------------------------------------------------
    session_id = create_mocap_session(
        session, recorded_at=recorded_at_str, location=location, notes=name
    )
    result.session_id = session_id

    # ------------------------------------------------------------------
    # Add cameras to session (copies registry rows)
    # ------------------------------------------------------------------
    for cam_key in cameras_raw:
        add_session_camera(
            session,
            registry,
            session_id,
            cam_instance_ids[cam_key],
            cam_mode_ids[cam_key],
            cam_intrinsics_ids[cam_key],  # type: ignore[arg-type]
            label=cam_key,
        )
        result.camera_instance_ids[cam_key] = cam_instance_ids[cam_key]

    # ------------------------------------------------------------------
    # Create shots, videos, and sync configs
    # ------------------------------------------------------------------
    for scene in scenes_raw:
        scene_label: str = str(scene.get("label", ""))
        scene_cameras: dict = scene.get("cameras", {})

        # Shot with nullable extrinsic_calibration_id
        shot_id = create_shot(
            session,
            session_id,
            extrinsic_calibration_id=None,
            label=scene_label,
        )
        result.shot_ids[scene_label] = shot_id

        # ShotVideo rows
        shot_video_ids: dict[str, str] = {}
        for cam_key, frame_cfg in scene_cameras.items():
            if cam_key not in cam_instance_ids:
                continue
            sv_id = add_shot_video(
                session,
                shot_id,
                cam_instance_ids[cam_key],
                cam_video_paths[cam_key],
                int(frame_cfg["first_frame"]),
                int(frame_cfg["last_frame"]),
                cam_fps[cam_key],
            )
            shot_video_ids[cam_key] = sv_id

        # SyncConfig with one rough SyncPoint per camera
        sync_config_id = generate_id()
        with session:
            session.execute(
                "INSERT INTO sync_configs (id, shot_id, created_by, notes) "
                "VALUES (?, ?, ?, ?)",
                (
                    sync_config_id,
                    shot_id,
                    "yaml-import-rough",
                    "coarse sync from project.yaml sync_frame fields",
                ),
            )
            for cam_key, sv_id in shot_video_ids.items():
                session.execute(
                    "INSERT INTO sync_points "
                    "(sync_config_id, camera_instance_id, shot_video_id, "
                    "video_frame, timestamp_s) VALUES (?, ?, ?, ?, ?)",
                    (
                        sync_config_id,
                        cam_instance_ids[cam_key],
                        sv_id,
                        cam_sync_frames[cam_key],
                        0.0,
                    ),
                )
        result.sync_config_ids[scene_label] = sync_config_id

    return result


def _print_dry_run(
    name: str,
    location: str,
    recorded_at: str | None,
    cam_instance_ids: dict[str, str],
    scenes_raw: list,
) -> None:
    print(f"[dry-run] Would create MocapSession: name={name!r} location={location!r} recorded_at={recorded_at!r}")
    for cam_key, iid in cam_instance_ids.items():
        print(f"[dry-run]   SessionCamera: {cam_key} -> instance={iid}")
    for scene in scenes_raw:
        label = scene.get("label", "")
        cams = list(scene.get("cameras", {}).keys())
        print(f"[dry-run]   Shot: {label!r}  cameras={cams}")
        print(f"[dry-run]     SyncConfig (rough) + {len(cams)} SyncPoint(s)")
