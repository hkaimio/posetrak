"""posetrak_db_cli.py — Command-line interface for the posetrak database layer.

Subcommands
-----------
init                  Create a new registry database.
camera-model-add      Register a camera hardware model.
camera-model-list     List registered camera models.
camera-mode-add       Register a capture mode (resolution/fps) for a camera model.
camera-mode-list      List registered camera modes.
import-calib          Import intrinsic calibration from a Pose2Sim TOML file.
set-project-root      Set the project root path stored in the registry settings.
info                  Print registry schema version and settings.
skeleton-import       Import a skeleton YAML file into the registry.
skeleton-list         List skeletons registered in the registry.
config-create         Create a tracker config snapshot from a TOML file.
config-edit           Derive a new tracker config by overriding fields.
config-list           List tracker configs registered in the registry.
session-create        Create a new mocap session in a session database.
session-camera-add    Link a camera to a session.
import-extrinsics     Import extrinsic calibration from a Pose2Sim TOML file.
shot-create           Create a new shot within a session.
shot-video-add        Add a video file record to a shot.
sync-import           Import camera sync anchors from a sync JSON file.
pose-import           Import 2-D pose observations from a pose directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.db.posetrak_db import (
    DEFAULT_REGISTRY_PATH,
    REGISTRY_SCHEMA_VERSION,
    add_session_camera,
    add_shot_video,
    create_camera_model,
    create_camera_mode,
    create_mocap_session,
    create_registry,
    create_session,
    create_shot,
    get_project_root,
    get_schema_version,
    list_camera_models,
    list_camera_modes,
    open_registry,
    open_session,
    set_project_root,
)
from scripts.db.import_calib_toml import import_calib_toml
from scripts.db.manage_skeleton import import_skeleton, list_skeletons
from scripts.db.manage_config import create_config_from_toml, edit_config, list_configs
from scripts.db.import_extrinsics import import_extrinsics
from scripts.db.import_sync_json import import_sync_json
from scripts.db.import_pose_json import import_pose_json


# ---------------------------------------------------------------------------
# Subcommand handlers — Phase 1
# ---------------------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> int:
    """Create a new registry database."""
    path = Path(args.registry)
    try:
        conn = create_registry(path)
        conn.close()
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Registry created: {path}")
    return 0


def _cmd_camera_model_add(args: argparse.Namespace) -> int:
    """Register a camera hardware model in the registry."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        model_id = create_camera_model(
            registry,
            manufacturer=args.manufacturer,
            model_name=args.model_name,
            sensor_size=args.sensor_size or None,
            notes=args.notes or None,
        )
    finally:
        registry.close()

    print(f"camera_model_id: {model_id}")
    return 0


def _cmd_camera_model_list(args: argparse.Namespace) -> int:
    """List camera models registered in the registry."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        rows = list_camera_models(registry)
    finally:
        registry.close()

    if not rows:
        print("No camera models registered.")
        return 0

    for row in rows:
        print(f"{row['id']}  {row['manufacturer'] or ''}  {row['model_name'] or ''}")
    return 0


def _cmd_camera_mode_add(args: argparse.Namespace) -> int:
    """Register a capture mode (resolution/fps) for a camera model."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        import sqlite3
        mode_id = create_camera_mode(
            registry,
            args.model_id,
            width_px=args.width,
            height_px=args.height,
            nominal_fps=args.fps,
            codec=args.codec or None,
            notes=args.notes or None,
        )
    except sqlite3.IntegrityError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        registry.close()
        return 1
    finally:
        registry.close()

    print(f"camera_mode_id: {mode_id}")
    return 0


def _cmd_camera_mode_list(args: argparse.Namespace) -> int:
    """List camera modes registered in the registry."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        rows = list_camera_modes(registry, camera_model_id=args.model_id or None)
    finally:
        registry.close()

    if not rows:
        print("No camera modes registered.")
        return 0

    for row in rows:
        fps = f"{row['nominal_fps']:.3g} fps" if row["nominal_fps"] else "fps unknown"
        res = (
            f"{row['width_px']}×{row['height_px']}"
            if row["width_px"] and row["height_px"]
            else "resolution unknown"
        )
        codec = f"  codec={row['codec']}" if row["codec"] else ""
        print(f"{row['id']}  model={row['camera_model_id']}  {res}  {fps}{codec}")
    return 0


def _parse_camera_modes(values: list[str]) -> str | dict[str, str]:
    """Parse ``--camera-mode`` argument values into a homogeneous UUID or per-camera dict.

    A single value without ``=`` is treated as a homogeneous UUID. One or more
    ``key=UUID`` values are parsed into a per-camera dict.

    Raises
    ------
    SystemExit
        If the values mix plain UUIDs and ``key=UUID`` pairs, or are otherwise malformed.
    """
    if len(values) == 1 and "=" not in values[0]:
        return values[0]  # homogeneous UUID

    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            print(
                f"Error: --camera-mode value {v!r} is ambiguous — "
                "mix of plain UUID and cam=UUID is not allowed. "
                "Use either a single UUID or one 'camN=UUID' per camera.",
                file=sys.stderr,
            )
            sys.exit(1)
        key, _, uuid = v.partition("=")
        if not key or not uuid:
            print(f"Error: malformed --camera-mode value {v!r}", file=sys.stderr)
            sys.exit(1)
        result[key] = uuid

    return result


def _parse_camera_instances(values: list[str] | None) -> str | dict[str, str] | None:
    """Parse ``--camera-instance`` argument values.

    Returns ``None`` if *values* is ``None`` or empty, a plain UUID string
    for a single un-keyed value, or a ``dict[str, str]`` for ``camN=UUID`` pairs.
    """
    if not values:
        return None
    if len(values) == 1 and "=" not in values[0]:
        return values[0]
    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            print(
                f"Error: --camera-instance value {v!r} is ambiguous.",
                file=sys.stderr,
            )
            sys.exit(1)
        key, _, uuid = v.partition("=")
        if not key or not uuid:
            print(f"Error: malformed --camera-instance value {v!r}", file=sys.stderr)
            sys.exit(1)
        result[key] = uuid
    return result


def _cmd_import_calib(args: argparse.Namespace) -> int:
    """Import intrinsic calibration from a Pose2Sim TOML file."""
    camera_modes = _parse_camera_modes(args.camera_mode)

    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        result = import_calib_toml(
            registry,
            Path(args.calib),
            camera_modes,
            calibration_tool=args.tool,
            distortion_model=args.distortion_model,
            notes=args.notes,
        )
    except (ValueError, Exception) as exc:  # noqa: BLE001
        print(f"Error importing calibration: {exc}", file=sys.stderr)
        registry.close()
        return 1
    finally:
        registry.close()

    n = len(result.camera_instance_ids)
    mode_desc = (
        camera_modes if isinstance(camera_modes, str) else "per-camera"
    )
    print(f"Imported {Path(args.calib).name}: {n} camera(s)  mode={mode_desc}")
    for label, iid in result.camera_instance_ids.items():
        intr_id = result.intrinsics_ids[label]
        print(f"  {label!r}  instance={iid}  intrinsics={intr_id}")
    if result.skipped:
        print(f"  skipped: {', '.join(sorted(result.skipped))}")
    return 0


def _cmd_set_project_root(args: argparse.Namespace) -> int:
    """Set the project root path in the registry settings."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        set_project_root(registry, Path(args.root).resolve())
    finally:
        registry.close()

    print(f"project_root set to: {Path(args.root).resolve()}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Print registry schema version and settings."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        version = get_schema_version(registry)
        project_root = get_project_root(registry)
        n_models = registry.execute("SELECT COUNT(*) FROM camera_models").fetchone()[0]
        n_modes = registry.execute("SELECT COUNT(*) FROM camera_modes").fetchone()[0]
        n_intrinsics = registry.execute(
            "SELECT COUNT(*) FROM intrinsics_calibrations"
        ).fetchone()[0]
    finally:
        registry.close()

    print(f"Registry: {args.registry}")
    print(f"  schema version : {version} (expected {REGISTRY_SCHEMA_VERSION})")
    print(f"  project_root   : {project_root}")
    print(f"  camera models  : {n_models}")
    print(f"  camera modes   : {n_modes}")
    print(f"  intrinsics     : {n_intrinsics}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand handlers — Phase 2: skeleton
# ---------------------------------------------------------------------------


def _cmd_skeleton_import(args: argparse.Namespace) -> int:
    """Import a skeleton YAML file into the registry."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        skeleton_id = import_skeleton(
            registry,
            Path(args.file),
            name=args.name or None,
            person_label=args.person_label or None,
            source=args.source or None,
            parent_id=args.parent_id or None,
            notes=args.notes or None,
        )
    except (FileNotFoundError, Exception) as exc:  # noqa: BLE001
        print(f"Error importing skeleton: {exc}", file=sys.stderr)
        registry.close()
        return 1
    finally:
        registry.close()

    print(f"skeleton_id: {skeleton_id}")
    return 0


def _cmd_skeleton_list(args: argparse.Namespace) -> int:
    """List skeletons registered in the registry."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        rows = list_skeletons(registry)
    finally:
        registry.close()

    if not rows:
        print("No skeletons registered.")
        return 0

    for row in rows:
        print(f"{row['id'][:16]}…  {row['name']}  created={row['created_at']}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand handlers — Phase 2: config
# ---------------------------------------------------------------------------


def _cmd_config_create(args: argparse.Namespace) -> int:
    """Create a tracker config snapshot from a TOML file."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        config_id = create_config_from_toml(
            registry,
            args.name,
            Path(args.from_toml),
            notes=args.notes or None,
        )
    except (FileNotFoundError, Exception) as exc:  # noqa: BLE001
        print(f"Error creating config: {exc}", file=sys.stderr)
        registry.close()
        return 1
    finally:
        registry.close()

    print(f"tracker_config_id: {config_id}")
    return 0


def _cmd_config_edit(args: argparse.Namespace) -> int:
    """Derive a new tracker config by overriding selected fields."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        new_id = edit_config(
            registry,
            args.id,
            alpha=args.alpha,
            beta=args.beta,
            kappa=args.kappa,
            process_noise_std=args.process_noise_std,
            measurement_noise_std=args.measurement_noise_std,
            outlier_threshold=args.outlier_threshold,
            tracker_fps=args.tracker_fps,
            ik_max_iterations=args.ik_max_iterations,
            ik_tolerance=args.ik_tolerance,
            init_position_std=args.init_position_std,
            init_orientation_std=args.init_orientation_std,
            init_joint_std=args.init_joint_std,
            init_velocity_std=args.init_velocity_std,
            min_cameras_for_init=args.min_cameras_for_init,
            notes=args.notes or None,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        registry.close()
        return 1
    finally:
        registry.close()

    print(f"new tracker_config_id: {new_id}")
    return 0


def _cmd_config_list(args: argparse.Namespace) -> int:
    """List tracker configs in the registry."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        rows = list_configs(registry, name=args.name or None)
    finally:
        registry.close()

    if not rows:
        print("No tracker configs registered.")
        return 0

    for row in rows:
        parent = f"  parent={row['parent_id']}" if row["parent_id"] else ""
        print(f"{row['id']}  {row['name']}  created={row['created_at']}{parent}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand handlers — Phase 2: session
# ---------------------------------------------------------------------------


def _cmd_session_create(args: argparse.Namespace) -> int:
    """Create a new mocap session in a session database."""
    sess_path = Path(args.session_db)
    try:
        if sess_path.exists():
            session_conn = open_session(sess_path)
        else:
            session_conn = create_session(sess_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        session_id = create_mocap_session(
            session_conn,
            recorded_at=args.date or None,
            location=args.location or "",
            notes=args.notes or "",
        )
    finally:
        session_conn.close()

    print(f"session_id: {session_id}")
    return 0


def _cmd_session_camera_add(args: argparse.Namespace) -> int:
    """Link a camera to a session."""
    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        import sqlite3
        add_session_camera(
            session_conn,
            args.session,
            args.camera_instance,
            args.camera_mode,
            args.intrinsics,
            label=args.label or "",
        )
    except sqlite3.IntegrityError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        session_conn.close()
        return 1
    finally:
        session_conn.close()

    print("session_camera added.")
    return 0


def _cmd_import_extrinsics(args: argparse.Namespace) -> int:
    """Import extrinsic calibration from a Pose2Sim TOML file."""
    cam_inst = _parse_camera_instances(args.camera_instance)
    if cam_inst is None:
        print("Error: at least one --camera-instance is required.", file=sys.stderr)
        return 1

    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        result = import_extrinsics(
            session_conn,
            args.session,
            Path(args.calib),
            cam_inst,
            method=args.method or "pose2sim",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error importing extrinsics: {exc}", file=sys.stderr)
        session_conn.close()
        return 1
    finally:
        session_conn.close()

    print(f"extrinsic_calibration_id: {result.extrinsic_calibration_id}")
    for cam_key, iid in result.camera_instance_ids.items():
        print(f"  {cam_key}  instance={iid}")
    if result.skipped:
        print(f"  skipped: {', '.join(sorted(result.skipped))}")
    return 0


def _cmd_shot_create(args: argparse.Namespace) -> int:
    """Create a new shot within a session."""
    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        shot_id = create_shot(
            session_conn,
            args.session,
            args.extrinsics,
            shot_number=args.number,
            label=args.label or "",
            notes=args.notes or "",
        )
    finally:
        session_conn.close()

    print(f"shot_id: {shot_id}")
    return 0


def _cmd_shot_video_add(args: argparse.Namespace) -> int:
    """Add a video file record to a shot."""
    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        video_id = add_shot_video(
            session_conn,
            args.shot,
            args.camera_instance,
            args.file,
            args.first_frame,
            args.last_frame,
            args.fps,
        )
    finally:
        session_conn.close()

    print(f"shot_video_id: {video_id}")
    return 0


def _cmd_sync_import(args: argparse.Namespace) -> int:
    """Import camera sync anchors from a sync JSON file."""
    cam_inst = _parse_camera_instances(args.camera_instance)
    if cam_inst is None:
        print("Error: at least one --camera-instance is required.", file=sys.stderr)
        return 1

    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        result = import_sync_json(
            session_conn,
            args.shot,
            Path(args.sync_json),
            cam_inst,
        )
    except (ValueError, Exception) as exc:  # noqa: BLE001
        print(f"Error importing sync: {exc}", file=sys.stderr)
        session_conn.close()
        return 1
    finally:
        session_conn.close()

    print(f"sync_config_id: {result.sync_config_id}")
    for cam_key, iid in result.camera_instance_ids.items():
        print(f"  {cam_key}  instance={iid}")
    if result.skipped:
        print(f"  skipped: {', '.join(sorted(result.skipped))}")
    return 0


def _cmd_pose_import(args: argparse.Namespace) -> int:
    """Import 2-D pose observations from a pose directory."""
    cam_inst = _parse_camera_instances(args.camera_instance)
    if cam_inst is None:
        print("Error: at least one --camera-instance is required.", file=sys.stderr)
        return 1

    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        result = import_pose_json(
            session_conn,
            args.shot,
            args.sync_config,
            Path(args.pose_dir),
            cam_inst,
            person_id=args.person_id,
            time_start=args.time_start,
            time_end=args.time_end,
            pose_model=args.pose_model or "",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error importing poses: {exc}", file=sys.stderr)
        session_conn.close()
        return 1
    finally:
        session_conn.close()

    print(f"sequence_id: {result.sequence_id}")
    print(f"n_observations: {result.n_observations}")
    if result.skipped_cameras:
        print(f"skipped cameras: {', '.join(sorted(result.skipped_cameras))}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser helpers
# ---------------------------------------------------------------------------


def _add_registry_arg(p: argparse.ArgumentParser) -> None:
    """Add the standard --registry argument with the default path."""
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        metavar="PATH",
        help=f"Path to the registry .db file (default: {DEFAULT_REGISTRY_PATH})",
    )


def _add_session_db_arg(p: argparse.ArgumentParser) -> None:
    """Add the standard --session-db argument."""
    p.add_argument(
        "--session-db",
        required=True,
        metavar="PATH",
        help="Path to the session .db file",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posetrak_db_cli",
        description="posetrak database management CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- init ---
    p_init = sub.add_parser("init", help="Create a new registry database")
    _add_registry_arg(p_init)

    # --- camera-model-add ---
    p_cma = sub.add_parser("camera-model-add", help="Register a camera hardware model")
    _add_registry_arg(p_cma)
    p_cma.add_argument("--manufacturer", default="", metavar="S")
    p_cma.add_argument("--model-name", default="", metavar="S")
    p_cma.add_argument("--sensor-size", default="", metavar="S")
    p_cma.add_argument("--notes", default="", metavar="S")

    # --- camera-model-list ---
    p_cml = sub.add_parser("camera-model-list", help="List registered camera models")
    _add_registry_arg(p_cml)

    # --- camera-mode-add ---
    p_coda = sub.add_parser(
        "camera-mode-add",
        help="Register a capture mode (resolution/fps) for a camera model",
    )
    _add_registry_arg(p_coda)
    p_coda.add_argument("--model-id", required=True, metavar="UUID",
                        help="ID of the parent camera_models row")
    p_coda.add_argument("--width", type=int, default=0, metavar="N",
                        help="Image width in pixels (default: 0 = unknown)")
    p_coda.add_argument("--height", type=int, default=0, metavar="N",
                        help="Image height in pixels (default: 0 = unknown)")
    p_coda.add_argument("--fps", type=float, default=0.0, metavar="F",
                        help="Nominal frames per second (default: 0 = unknown)")
    p_coda.add_argument("--codec", default="", metavar="S")
    p_coda.add_argument("--notes", default="", metavar="S")

    # --- camera-mode-list ---
    p_codl = sub.add_parser("camera-mode-list", help="List registered camera modes")
    _add_registry_arg(p_codl)
    p_codl.add_argument("--model-id", default="", metavar="UUID",
                        help="Filter by camera model ID")

    # --- import-calib ---
    p_ic = sub.add_parser(
        "import-calib",
        help="Import intrinsic calibration from a Pose2Sim TOML file",
    )
    _add_registry_arg(p_ic)
    p_ic.add_argument("--calib", required=True, metavar="TOML_PATH",
                      help="Path to the Pose2Sim calibration TOML file")
    p_ic.add_argument(
        "--camera-mode", required=True, action="append", metavar="SPEC", dest="camera_mode",
        help=(
            "Camera mode assignment. Two forms: "
            "(1) a single UUID applies to all cameras in the file; "
            "(2) one or more 'camN=UUID' pairs for per-camera assignment — "
            "cameras not listed are skipped. "
            "Example: --camera-mode cam1=<uuid> --camera-mode cam2=<uuid>"
        ),
    )
    p_ic.add_argument("--tool", default="pose2sim", metavar="S",
                      help="Calibration tool name (default: pose2sim)")
    p_ic.add_argument("--distortion-model", default="radtan", metavar="S",
                      help="Distortion model (default: radtan)")
    p_ic.add_argument("--notes", default="", metavar="S",
                      help="Free-text notes stored with each intrinsics row")

    # --- set-project-root ---
    p_spr = sub.add_parser("set-project-root",
                            help="Set the project_root setting in the registry")
    _add_registry_arg(p_spr)
    p_spr.add_argument("--root", required=True, metavar="DIR")

    # --- info ---
    p_info = sub.add_parser("info", help="Print registry info and settings")
    _add_registry_arg(p_info)

    # --- skeleton-import ---
    p_si = sub.add_parser("skeleton-import", help="Import a skeleton YAML into the registry")
    _add_registry_arg(p_si)
    p_si.add_argument("--file", required=True, metavar="YAML_PATH",
                      help="Path to the skeleton YAML file")
    p_si.add_argument("--name", default="", metavar="S", help="Human-readable name")
    p_si.add_argument("--person-label", default="", metavar="S")
    p_si.add_argument("--source", default="", metavar="S")
    p_si.add_argument("--parent-id", default="", metavar="SHA256",
                      help="Parent skeleton ID (for lineage)")
    p_si.add_argument("--notes", default="", metavar="S")

    # --- skeleton-list ---
    p_sl = sub.add_parser("skeleton-list", help="List registered skeletons")
    _add_registry_arg(p_sl)

    # --- config-create ---
    p_cc = sub.add_parser("config-create",
                          help="Create a tracker config snapshot from a TOML file")
    _add_registry_arg(p_cc)
    p_cc.add_argument("--name", required=True, metavar="S",
                      help="Name for this configuration snapshot")
    p_cc.add_argument("--from-toml", required=True, metavar="TOML_PATH",
                      help="Path to the posetrak TOML config file")
    p_cc.add_argument("--notes", default="", metavar="S")

    # --- config-edit ---
    p_ce = sub.add_parser("config-edit",
                          help="Derive a new tracker config by overriding fields")
    _add_registry_arg(p_ce)
    p_ce.add_argument("--id", required=True, metavar="UUID",
                      help="ID of the existing tracker_configs row to derive from")
    p_ce.add_argument("--alpha", type=float, default=None)
    p_ce.add_argument("--beta", type=float, default=None)
    p_ce.add_argument("--kappa", type=float, default=None)
    p_ce.add_argument("--process-noise-std", type=float, default=None,
                      dest="process_noise_std")
    p_ce.add_argument("--measurement-noise-std", type=float, default=None,
                      dest="measurement_noise_std")
    p_ce.add_argument("--outlier-threshold", type=float, default=None,
                      dest="outlier_threshold")
    p_ce.add_argument("--tracker-fps", type=float, default=None, dest="tracker_fps")
    p_ce.add_argument("--ik-max-iterations", type=int, default=None,
                      dest="ik_max_iterations")
    p_ce.add_argument("--ik-tolerance", type=float, default=None, dest="ik_tolerance")
    p_ce.add_argument("--init-position-std", type=float, default=None,
                      dest="init_position_std")
    p_ce.add_argument("--init-orientation-std", type=float, default=None,
                      dest="init_orientation_std")
    p_ce.add_argument("--init-joint-std", type=float, default=None, dest="init_joint_std")
    p_ce.add_argument("--init-velocity-std", type=float, default=None,
                      dest="init_velocity_std")
    p_ce.add_argument("--min-cameras-for-init", type=int, default=None,
                      dest="min_cameras_for_init")
    p_ce.add_argument("--notes", default="", metavar="S")

    # --- config-list ---
    p_cl = sub.add_parser("config-list", help="List tracker configs in the registry")
    _add_registry_arg(p_cl)
    p_cl.add_argument("--name", default="", metavar="S", help="Filter by name")

    # --- session-create ---
    p_sc = sub.add_parser("session-create", help="Create a new mocap session")
    _add_session_db_arg(p_sc)
    p_sc.add_argument("--date", default="", metavar="ISO_DATE",
                      help="Recording date (ISO format). Defaults to today.")
    p_sc.add_argument("--location", default="", metavar="S")
    p_sc.add_argument("--notes", default="", metavar="S")

    # --- session-camera-add ---
    p_sca = sub.add_parser("session-camera-add",
                           help="Link a camera to a session")
    _add_session_db_arg(p_sca)
    p_sca.add_argument("--session", required=True, metavar="UUID",
                       help="mocap_sessions.id")
    p_sca.add_argument("--camera-instance", required=True, metavar="UUID",
                       dest="camera_instance")
    p_sca.add_argument("--camera-mode", required=True, metavar="UUID",
                       dest="camera_mode")
    p_sca.add_argument("--intrinsics", required=True, metavar="UUID")
    p_sca.add_argument("--label", default="", metavar="S")

    # --- import-extrinsics ---
    p_ie = sub.add_parser("import-extrinsics",
                          help="Import extrinsic calibration from a Pose2Sim TOML")
    _add_session_db_arg(p_ie)
    p_ie.add_argument("--session", required=True, metavar="UUID",
                      help="mocap_sessions.id")
    p_ie.add_argument("--calib", required=True, metavar="TOML_PATH")
    p_ie.add_argument("--camera-instance", action="append", metavar="SPEC",
                      dest="camera_instance",
                      help="cam1=<uuid> pairs or single UUID")
    p_ie.add_argument("--method", default="pose2sim", metavar="S")

    # --- shot-create ---
    p_shot = sub.add_parser("shot-create", help="Create a new shot within a session")
    _add_session_db_arg(p_shot)
    p_shot.add_argument("--session", required=True, metavar="UUID")
    p_shot.add_argument("--extrinsics", required=True, metavar="UUID",
                        help="extrinsic_calibrations.id")
    p_shot.add_argument("--number", type=int, default=None, metavar="N")
    p_shot.add_argument("--label", default="", metavar="S")
    p_shot.add_argument("--notes", default="", metavar="S")

    # --- shot-video-add ---
    p_sv = sub.add_parser("shot-video-add", help="Add a video file to a shot")
    _add_session_db_arg(p_sv)
    p_sv.add_argument("--shot", required=True, metavar="UUID")
    p_sv.add_argument("--camera-instance", required=True, metavar="UUID",
                      dest="camera_instance")
    p_sv.add_argument("--file", required=True, metavar="PATH")
    p_sv.add_argument("--first-frame", required=True, type=int, metavar="N",
                      dest="first_frame")
    p_sv.add_argument("--last-frame", required=True, type=int, metavar="N",
                      dest="last_frame")
    p_sv.add_argument("--fps", required=True, type=float, metavar="F")

    # --- sync-import ---
    p_sync = sub.add_parser("sync-import",
                            help="Import camera sync anchors from a sync JSON file")
    _add_session_db_arg(p_sync)
    p_sync.add_argument("--shot", required=True, metavar="UUID")
    p_sync.add_argument("--sync-json", required=True, metavar="JSON_PATH",
                        dest="sync_json")
    p_sync.add_argument("--camera-instance", action="append", metavar="SPEC",
                        dest="camera_instance")

    # --- pose-import ---
    p_pose = sub.add_parser("pose-import",
                            help="Import 2-D pose observations from a pose directory")
    _add_session_db_arg(p_pose)
    p_pose.add_argument("--shot", required=True, metavar="UUID")
    p_pose.add_argument("--sync-config", required=True, metavar="UUID",
                        dest="sync_config")
    p_pose.add_argument("--pose-dir", required=True, metavar="DIR", dest="pose_dir")
    p_pose.add_argument("--camera-instance", action="append", metavar="SPEC",
                        dest="camera_instance")
    p_pose.add_argument("--person-id", type=int, default=0, dest="person_id")
    p_pose.add_argument("--time-start", type=float, default=None, dest="time_start")
    p_pose.add_argument("--time-end", type=float, default=None, dest="time_end")
    p_pose.add_argument("--pose-model", default="", dest="pose_model", metavar="S")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "init": _cmd_init,
        "camera-model-add": _cmd_camera_model_add,
        "camera-model-list": _cmd_camera_model_list,
        "camera-mode-add": _cmd_camera_mode_add,
        "camera-mode-list": _cmd_camera_mode_list,
        "import-calib": _cmd_import_calib,
        "set-project-root": _cmd_set_project_root,
        "info": _cmd_info,
        "skeleton-import": _cmd_skeleton_import,
        "skeleton-list": _cmd_skeleton_list,
        "config-create": _cmd_config_create,
        "config-edit": _cmd_config_edit,
        "config-list": _cmd_config_list,
        "session-create": _cmd_session_create,
        "session-camera-add": _cmd_session_camera_add,
        "import-extrinsics": _cmd_import_extrinsics,
        "shot-create": _cmd_shot_create,
        "shot-video-add": _cmd_shot_video_add,
        "sync-import": _cmd_sync_import,
        "pose-import": _cmd_pose_import,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))  # type: ignore[operator]


if __name__ == "__main__":
    main()
