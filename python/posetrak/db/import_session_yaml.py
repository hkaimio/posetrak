# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""import_session_yaml.py — Import a capture project YAML into the session database.

Reads a project YAML file and creates the corresponding posetrak DB records:
one ``MocapSession``, one ``SessionCamera`` per camera, one ``Shot`` per scene
(with ``extrinsic_calibration_id = NULL``), one ``ShotVideo`` per camera per
scene, and one ``SyncConfig`` per shot with one ``SyncPoint`` per camera at the
rough sync anchor frame.

YAML format
-----------
Two equivalent forms are accepted for the ``cameras`` section:

Dict form (key is the camera label)::

    cameras:
      cam1:
        video_path: "/path/to/cam1.mp4"
        fps: 120.0
        sync_frame: 5678
        camera_instance_id: "uuid"          # optional; looked up by label if absent
        camera_mode_id: "uuid"              # optional
        intrinsics_calibration_id: "uuid"   # optional

List form (as produced by sync_videos.py; ``path`` is accepted as alias for
``video_path``)::

    ref_camera: cam1
    cameras:
      - name: cam1
        path: "/path/to/cam1.mp4"
        fps: 120.0
        sync_frame: 5678

Two equivalent forms are accepted for the ``scenes`` section:

Explicit per-camera frames::

    scenes:
      - label: "scene1"
        cameras:
          cam1:
            first_frame: 6001
            last_frame: 7200

Ref-camera relative (``start_frame``/``end_frame`` are in the ref_camera's
frame coordinates; per-camera frames are derived via fps + sync offsets)::

    scenes:
      - name: "scene1"         # "label" is also accepted
        start_frame: 6001      # ref-camera frame number
        end_frame: 7200

Top-level fields::

    name: "my-session"            # used as session notes (optional)
    location: "gym"               # optional
    recorded_at: "2024-03-15"     # optional ISO date; defaults to today

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
from typing import Any

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


def _normalise_cameras(raw: Any) -> dict[str, dict]:
    """Accept both dict and list camera sections; return a uniform dict.

    Normalises field aliases:
    - ``path`` → ``video_path``
    - ``calib.intrinsics`` (UUID) → ``intrinsics_calibration_id``
    """
    if isinstance(raw, dict):
        entries = [{"name": k, **v} for k, v in raw.items()]
    else:
        entries = list(raw)

    result = {}
    for entry in entries:
        key = str(entry["name"])
        cam = {k: v for k, v in entry.items() if k != "name"}
        if "video_path" not in cam and "path" in cam:
            cam["video_path"] = cam.pop("path")
        # calib.intrinsics UUID → intrinsics_calibration_id
        calib = cam.pop("calib", None)
        if calib and "intrinsics" in calib and "intrinsics_calibration_id" not in cam:
            cam["intrinsics_calibration_id"] = calib["intrinsics"]
        result[key] = cam
    return result


def _normalise_scenes(
    raw: list,
    cameras: dict[str, dict],
    ref_camera: str | None,
) -> list[dict]:
    """Accept both explicit and ref-camera-relative scene formats.

    Explicit format already has ``cameras`` sub-dict with first/last frames.
    Relative format has ``start_frame``/``end_frame`` in ref-camera coordinates;
    per-camera frames are derived via sync offsets and fps.
    """
    result = []
    for scene in raw:
        label = str(scene.get("label") or scene.get("name", ""))
        if "cameras" in scene:
            result.append({"label": label, "cameras": scene["cameras"]})
            continue

        # Relative format — derive per-camera frames
        if "start_frame" not in scene or "end_frame" not in scene:
            raise ValueError(
                f"Scene {label!r}: must have either 'cameras' (explicit frames) "
                "or 'start_frame'/'end_frame' (ref-camera relative)."
            )
        if not ref_camera:
            raise ValueError(
                "YAML must specify 'ref_camera' when scenes use start_frame/end_frame."
            )
        if ref_camera not in cameras:
            raise ValueError(
                f"ref_camera {ref_camera!r} not found in cameras section."
            )
        ref_sync = int(cameras[ref_camera]["sync_frame"])
        ref_fps = float(cameras[ref_camera]["fps"])
        start_ref = int(scene["start_frame"])
        end_ref = int(scene["end_frame"])
        start_offset_s = (start_ref - ref_sync) / ref_fps
        end_offset_s = (end_ref - ref_sync) / ref_fps

        cam_frames: dict[str, dict] = {}
        for cam_key, cam_cfg in cameras.items():
            fps = float(cam_cfg["fps"])
            sync = int(cam_cfg["sync_frame"])
            cam_frames[cam_key] = {
                "first_frame": sync + int(round(start_offset_s * fps)),
                "last_frame": sync + int(round(end_offset_s * fps)),
            }
        result.append({"label": label, "cameras": cam_frames})
    return result


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


def _mode_from_intrinsics(registry: sqlite3.Connection, intrinsics_id: str) -> str:
    """Look up the camera_mode_id for a known intrinsics_calibration_id."""
    row = registry.execute(
        "SELECT camera_mode_id FROM intrinsics_calibrations WHERE id = ?",
        (intrinsics_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"intrinsics_calibration {intrinsics_id!r} not found in registry")
    return row["camera_mode_id"]


def _resolve_camera_mode(registry: sqlite3.Connection, camera_instance_id: str) -> str:
    """Return the camera_mode_id associated with a camera instance via camera_model.

    Raises ``ValueError`` if there is not exactly one mode for the model.
    Prefer calling ``_mode_from_intrinsics`` first when an intrinsics ID is known.
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
            "Provide camera_mode_id or intrinsics_calibration_id in the YAML."
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
    ref_camera: str | None = doc.get("ref_camera")
    cameras_raw: dict = _normalise_cameras(doc.get("cameras", {}))
    scenes_raw: list = _normalise_scenes(doc.get("scenes", []), cameras_raw, ref_camera)

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
        intrinsics_id: str | None = cam_cfg.get("intrinsics_calibration_id")
        mode_id: str = (
            cam_cfg.get("camera_mode_id")
            or (intrinsics_id and _mode_from_intrinsics(registry, intrinsics_id))
            or _resolve_camera_mode(registry, instance_id)
        )
        if not intrinsics_id:
            intrinsics_id = _latest_intrinsics(registry, mode_id)
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
