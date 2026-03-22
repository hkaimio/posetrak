"""posetrak_db_cli.py — Command-line interface for the posetrak database layer.

Topics / Actions
----------------
registry  init      Create a new registry database.
registry  info      Print registry info and settings.
registry  set-root  Set the project_root path in the registry.

camera-model  add   Register a camera hardware model.
camera-model  list  List registered camera models.

camera-mode   add   Register a capture mode (resolution/fps) for a camera model.
camera-mode   list  List registered camera modes.

calib     import    Import intrinsic calibration from a Pose2Sim TOML file.

skeleton  import    Import a skeleton YAML file.
skeleton  list      List skeletons.

config    create    Create a tracker config snapshot from a TOML file.
config    edit      Derive a new tracker config by overriding fields.
config    list      List tracker configs.

session   create    Create a new mocap session in a session database.
session   list      List mocap sessions in a session database.
session   add-camera  Link a camera (with registry copy) to a session.

extrinsics  import  Import extrinsic calibration from a Pose2Sim TOML file.
extrinsics  list    List extrinsic calibrations in a session database.

shot      create    Create a new shot within a session.
shot      list      List shots in a session database.
shot      add-video Add a video file record to a shot.

sync      import    Import camera sync anchors from a sync JSON file.
sync      list      List sync configs in a session database.

pose      import    Import 2-D pose observations from a pose directory.
pose      list      List pose observation sequences in a session database.

tracking-run  list  List tracking runs in a session database.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from posetrak.db.db import (
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
    resolve_id_prefix,
    set_project_root,
)
from posetrak.db.import_calib_toml import import_calib_toml
from posetrak.db.manage_skeleton import (
    copy_skeleton_to_session,
    import_skeleton,
    list_skeletons,
)
from posetrak.db.manage_config import (
    copy_config_to_session,
    create_config_from_toml,
    edit_config,
    list_configs,
)
from posetrak.db.import_extrinsics import import_extrinsics
from posetrak.db.import_sync_json import import_sync_json
from posetrak.db.import_pose_json import import_pose_json


# ---------------------------------------------------------------------------
# ID resolution helper
# ---------------------------------------------------------------------------


def _resolve(conn, table: str, prefix: str | None) -> str | None:
    """Resolve a UUID prefix to a full ID; returns None if prefix is None."""
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def _add_registry_arg(p: argparse.ArgumentParser) -> None:
    """Add the standard --registry argument with the default path."""
    p.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY_PATH),
        metavar="PATH",
        help=f"Path to the registry .db file (default: {DEFAULT_REGISTRY_PATH})",
    )


def _add_session_db_arg(p: argparse.ArgumentParser, *, required: bool = False) -> None:
    """Add the standard --session-db argument."""
    p.add_argument(
        "--session-db",
        required=required,
        default=None,
        metavar="PATH",
        help="Path to the session .db file",
    )


# ---------------------------------------------------------------------------
# Camera mode / instance argument parsers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Command handlers — registry
# ---------------------------------------------------------------------------


def _cmd_registry_init(args: argparse.Namespace) -> int:
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


def _cmd_registry_info(args: argparse.Namespace) -> int:
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


def _cmd_registry_set_root(args: argparse.Namespace) -> int:
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


# ---------------------------------------------------------------------------
# Command handlers — camera-model
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Command handlers — camera-mode
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Command handlers — calib
# ---------------------------------------------------------------------------


def _cmd_calib_import(args: argparse.Namespace) -> int:
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


# ---------------------------------------------------------------------------
# Command handlers — skeleton
# ---------------------------------------------------------------------------


def _cmd_skeleton_import(args: argparse.Namespace) -> int:
    """Import a skeleton YAML file into the registry and/or a session DB."""
    yaml_path = Path(args.file)
    session_db_path = Path(args.session_db) if args.session_db else None
    use_global = args.global_registry

    if session_db_path is None and not use_global:
        print("Error: specify --session-db, --global, or both", file=sys.stderr)
        return 1

    skeleton_id = None

    if use_global:
        try:
            registry = open_registry(Path(args.registry))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening registry: {exc}", file=sys.stderr)
            return 1
        try:
            skeleton_id = import_skeleton(
                registry,
                yaml_path,
                name=args.name or None,
                person_label=args.person_label or None,
                source=args.source or None,
                parent_id=args.parent_id or None,
                notes=args.notes or None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error importing skeleton: {exc}", file=sys.stderr)
            registry.close()
            return 1
        finally:
            registry.close()

    if session_db_path is not None:
        try:
            session = open_session(session_db_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening session db: {exc}", file=sys.stderr)
            return 1
        try:
            skeleton_id = import_skeleton(
                session,
                yaml_path,
                name=args.name or None,
                person_label=args.person_label or None,
                source=args.source or None,
                parent_id=args.parent_id or None,
                notes=args.notes or None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error importing skeleton to session: {exc}", file=sys.stderr)
            session.close()
            return 1
        finally:
            session.close()

    print(f"skeleton_id: {skeleton_id}")
    return 0


def _cmd_skeleton_list(args: argparse.Namespace) -> int:
    """List skeletons — from session DB if provided, otherwise from registry."""
    if args.session_db:
        try:
            conn = open_session(Path(args.session_db))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening session db: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            conn = open_registry(Path(args.registry))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening registry: {exc}", file=sys.stderr)
            return 1

    try:
        rows = list_skeletons(conn)
    finally:
        conn.close()

    if not rows:
        print("No skeletons registered.")
        return 0

    print(f"{'id':<36}  {'name':<30}  created_at")
    print("-" * 85)
    for row in rows:
        print(f"{row['id']:<36}  {row['name']:<30}  {row['created_at']}")
    return 0


# ---------------------------------------------------------------------------
# Command handlers — config
# ---------------------------------------------------------------------------


def _cmd_config_create(args: argparse.Namespace) -> int:
    """Create a tracker config snapshot from a TOML file."""
    session_db_path = Path(args.session_db) if args.session_db else None
    use_global = args.global_registry

    if session_db_path is None and not use_global:
        print("Error: specify --session-db, --global, or both", file=sys.stderr)
        return 1

    config_id = None

    if use_global:
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
        except Exception as exc:  # noqa: BLE001
            print(f"Error creating config: {exc}", file=sys.stderr)
            registry.close()
            return 1
        finally:
            registry.close()

    if session_db_path is not None:
        try:
            session = open_session(session_db_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening session db: {exc}", file=sys.stderr)
            return 1
        try:
            config_id = create_config_from_toml(
                session,
                args.name,
                Path(args.from_toml),
                notes=args.notes or None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Error creating config in session: {exc}", file=sys.stderr)
            session.close()
            return 1
        finally:
            session.close()

    print(f"tracker_config_id: {config_id}")
    return 0


def _cmd_config_edit(args: argparse.Namespace) -> int:
    """Derive a new tracker config by overriding selected fields."""
    session_db_path = Path(args.session_db) if args.session_db else None
    use_global = args.global_registry

    if session_db_path is None and not use_global:
        print("Error: specify --session-db, --global, or both", file=sys.stderr)
        return 1

    new_id = None

    if use_global:
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
                notes=args.notes or None,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            registry.close()
            return 1
        finally:
            registry.close()

    if session_db_path is not None:
        try:
            session = open_session(session_db_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening session db: {exc}", file=sys.stderr)
            return 1
        try:
            new_id = edit_config(
                session,
                args.id,
                alpha=args.alpha,
                beta=args.beta,
                kappa=args.kappa,
                process_noise_std=args.process_noise_std,
                measurement_noise_std=args.measurement_noise_std,
                outlier_threshold=args.outlier_threshold,
                tracker_fps=args.tracker_fps,
                notes=args.notes or None,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            session.close()
            return 1
        finally:
            session.close()

    print(f"new tracker_config_id: {new_id}")
    return 0


def _cmd_config_list(args: argparse.Namespace) -> int:
    """List tracker configs — from session DB if provided, otherwise from registry."""
    if args.session_db:
        try:
            conn = open_session(Path(args.session_db))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening session db: {exc}", file=sys.stderr)
            return 1
    else:
        try:
            conn = open_registry(Path(args.registry))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error opening registry: {exc}", file=sys.stderr)
            return 1

    try:
        rows = list_configs(conn, name=args.name or None)
    finally:
        conn.close()

    if not rows:
        print("No tracker configs registered.")
        return 0

    for row in rows:
        parent = f"  parent={row['parent_id']}" if row["parent_id"] else ""
        print(f"{row['id']}  {row['name']}  created={row['created_at']}{parent}")
    return 0


# ---------------------------------------------------------------------------
# Command handlers — calib
# ---------------------------------------------------------------------------


def _cmd_calib_list(args: argparse.Namespace) -> int:
    """List camera instances and their intrinsics calibrations."""
    import sqlite3

    try:
        conn = open_registry(Path(args.registry)) if args.registry else None
        if conn is None and args.session_db:
            conn = open_session(Path(args.session_db))
        if conn is None:
            print("Error: provide --registry or --session-db", file=sys.stderr)
            return 1
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        rows = conn.execute(
            "SELECT ci.id, ci.label, ic.id, ic.calibrated_at, ic.fx, ic.fy, ic.cx, ic.cy"
            " FROM camera_instances ci"
            " JOIN intrinsics_calibrations ic ON ic.camera_mode_id IN"
            "   (SELECT id FROM camera_modes WHERE camera_model_id = ci.camera_model_id)"
            " ORDER BY ci.label, ic.calibrated_at"
        ).fetchall()
        if not rows:
            print("(no calibrations found)")
            return 0
        print(f"{'camera_key':<12}  {'instance_id':<36}  {'intrinsics_id':<36}  "
              f"{'calibrated_at':<10}  {'fx':>8}  {'fy':>8}")
        print("-" * 115)
        for r in rows:
            print(f"{r[1]:<12}  {r[0]:<36}  {r[2]:<36}  {r[3]:<10}  {r[4]:>8.1f}  {r[5]:>8.1f}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# Command handlers — session
# ---------------------------------------------------------------------------


def _cmd_session_list(args: argparse.Namespace) -> int:
    """List mocap sessions in a session database."""
    import sqlite3

    try:
        conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        rows = conn.execute(
            "SELECT id, recorded_at, location FROM mocap_sessions ORDER BY recorded_at"
        ).fetchall()
        if not rows:
            print("(no sessions)")
            return 0
        print(f"{'id':<36}  {'recorded_at':<12}  location")
        print("-" * 80)
        for r in rows:
            print(f"{r[0]:<36}  {r[1]:<12}  {r[2] or ''}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


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


def _cmd_session_add_camera(args: argparse.Namespace) -> int:
    """Link a camera to a session, copying registry rows into the session DB."""
    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        registry.close()
        return 1

    try:
        import sqlite3
        session_id = _resolve(session_conn, "mocap_sessions", args.session)
        camera_instance = _resolve(registry, "camera_instances", args.camera_instance)
        camera_mode = _resolve(registry, "camera_modes", args.camera_mode)
        intrinsics = _resolve(registry, "intrinsics_calibrations", args.intrinsics)
        add_session_camera(
            session_conn,
            registry,
            session_id,
            camera_instance,
            camera_mode,
            intrinsics,
            label=args.label or "",
        )
    except (sqlite3.IntegrityError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        session_conn.close()
        registry.close()
        return 1
    finally:
        session_conn.close()
        registry.close()

    print("session_camera added.")
    return 0


# ---------------------------------------------------------------------------
# Command handlers — extrinsics
# ---------------------------------------------------------------------------


def _cmd_extrinsics_import(args: argparse.Namespace) -> int:
    """Import extrinsic calibration from a Pose2Sim TOML file."""
    cam_inst = _parse_camera_instances(args.camera_instance)
    if cam_inst is None:
        print("Error: at least one --camera-instance is required.", file=sys.stderr)
        return 1

    try:
        registry = open_registry(Path(args.registry))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        registry.close()
        return 1

    try:
        session_id = _resolve(session_conn, "mocap_sessions", args.session)
        result = import_extrinsics(
            session_conn,
            session_id,
            Path(args.calib),
            cam_inst,
            registry=registry,
            method=args.method or "pose2sim",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error importing extrinsics: {exc}", file=sys.stderr)
        session_conn.close()
        registry.close()
        return 1
    finally:
        session_conn.close()
        registry.close()

    print(f"extrinsic_calibration_id: {result.extrinsic_calibration_id}")
    for cam_key, iid in result.camera_instance_ids.items():
        print(f"  {cam_key}  instance={iid}")
    if result.skipped:
        print(f"  skipped: {', '.join(sorted(result.skipped))}")
    return 0


# ---------------------------------------------------------------------------
# Command handlers — extrinsics (list)
# ---------------------------------------------------------------------------


def _cmd_extrinsics_list(args: argparse.Namespace) -> int:
    """List extrinsic calibrations in a session database."""
    import sqlite3

    try:
        conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        q = "SELECT id, session_id, calibrated_at, method FROM extrinsic_calibrations"
        params: list = []
        if args.session:
            q += " WHERE session_id = ?"
            params.append(args.session)
        q += " ORDER BY calibrated_at"
        rows = conn.execute(q, params).fetchall()
        if not rows:
            print("(no extrinsic calibrations)")
            return 0
        print(f"{'id':<36}  {'session_id':<36}  {'calibrated_at':<12}  method")
        print("-" * 100)
        for r in rows:
            print(f"{r[0]:<36}  {r[1]:<36}  {r[2]:<12}  {r[3] or ''}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# Command handlers — shot
# ---------------------------------------------------------------------------


def _cmd_shot_list(args: argparse.Namespace) -> int:
    """List shots in a session database."""
    import sqlite3

    try:
        conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        q = "SELECT id, session_id, shot_number, label, extrinsic_calibration_id FROM shots"
        params: list = []
        if args.session:
            q += " WHERE session_id = ?"
            params.append(args.session)
        q += " ORDER BY shot_number"
        rows = conn.execute(q, params).fetchall()
        if not rows:
            print("(no shots)")
            return 0
        print(f"{'id':<36}  {'#':>4}  {'label':<30}  {'extrinsics_id':<36}")
        print("-" * 115)
        for r in rows:
            print(f"{r[0]:<36}  {r[2]:>4}  {(r[3] or ''):<30}  {r[4]:<36}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


def _cmd_shot_create(args: argparse.Namespace) -> int:
    """Create a new shot within a session."""
    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        session_id = _resolve(session_conn, "mocap_sessions", args.session)
        extrinsics_id = _resolve(session_conn, "extrinsic_calibrations", args.extrinsics)
        shot_id = create_shot(
            session_conn,
            session_id,
            extrinsics_id,
            shot_number=args.number,
            label=args.label or "",
            notes=args.notes or "",
        )
    finally:
        session_conn.close()

    print(f"shot_id: {shot_id}")
    return 0


def _cmd_shot_add_video(args: argparse.Namespace) -> int:
    """Add a video file record to a shot."""
    try:
        session_conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening session db: {exc}", file=sys.stderr)
        return 1

    try:
        shot_id = _resolve(session_conn, "shots", args.shot)
        camera_instance = _resolve(session_conn, "camera_instances", args.camera_instance)
        video_id = add_shot_video(
            session_conn,
            shot_id,
            camera_instance,
            args.file,
            args.first_frame,
            args.last_frame,
            args.fps,
        )
    finally:
        session_conn.close()

    print(f"shot_video_id: {video_id}")
    return 0


# ---------------------------------------------------------------------------
# Command handlers — sync
# ---------------------------------------------------------------------------


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
        shot_id = _resolve(session_conn, "shots", args.shot)
        result = import_sync_json(
            session_conn,
            shot_id,
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


# ---------------------------------------------------------------------------
# Command handlers — sync (list)
# ---------------------------------------------------------------------------


def _cmd_sync_list(args: argparse.Namespace) -> int:
    """List sync configs in a session database."""
    import sqlite3

    try:
        conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        q = "SELECT id, shot_id FROM sync_configs"
        params: list = []
        if args.shot:
            q += " WHERE shot_id = ?"
            params.append(args.shot)
        rows = conn.execute(q, params).fetchall()
        if not rows:
            print("(no sync configs)")
            return 0
        print(f"{'id':<36}  {'shot_id':<36}")
        print("-" * 75)
        for r in rows:
            print(f"{r[0]:<36}  {r[1]:<36}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# Command handlers — pose (list)
# ---------------------------------------------------------------------------


def _cmd_pose_list(args: argparse.Namespace) -> int:
    """List pose observation sequences in a session database."""
    import sqlite3

    try:
        conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        q = ("SELECT id, shot_id, sync_config_id, time_start_s, time_end_s, pose_model"
             " FROM pose_observation_sequences")
        params: list = []
        if args.shot:
            q += " WHERE shot_id = ?"
            params.append(args.shot)
        q += " ORDER BY time_start_s"
        rows = conn.execute(q, params).fetchall()
        if not rows:
            print("(no pose sequences)")
            return 0
        print(f"{'id':<36}  {'t_start':>8}  {'t_end':>8}  {'pose_model':<24}  sync_config_id")
        print("-" * 120)
        for r in rows:
            print(f"{r[0]:<36}  {r[3]:>8.2f}  {r[4]:>8.2f}  {(r[5] or ''):<24}  {r[2]}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# Command handlers — pose
# ---------------------------------------------------------------------------


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
        shot_id = _resolve(session_conn, "shots", args.shot)
        sync_config_id = _resolve(session_conn, "sync_configs", args.sync_config)
        result = import_pose_json(
            session_conn,
            shot_id,
            sync_config_id,
            Path(args.pose_dir),
            cam_inst,
            person_ids=args.person_ids or None,
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
# Command handlers — tracking-run
# ---------------------------------------------------------------------------


def _cmd_tracking_run_list(args: argparse.Namespace) -> int:
    """List tracking runs in a session database."""
    import sqlite3
    try:
        conn = open_session(Path(args.session_db))
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        q = ("SELECT id, observation_sequence_id, skeleton_id, ran_at, posetrak_version"
             " FROM tracking_runs ORDER BY ran_at")
        rows = conn.execute(q).fetchall()
        if not rows:
            print("(no tracking runs)")
            return 0
        print(f"{'id':<36}  {'sequence_id':<36}  {'ran_at':<20}  version")
        print("-" * 105)
        for r in rows:
            print(f"{r[0]:<36}  {r[1]:<36}  {r[2]:<20}  {r[4] or ''}")
    except sqlite3.Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posetrak-db",
        description="posetrak database management CLI",
    )
    topics = parser.add_subparsers(dest="topic", required=True)

    # -------------------------------------------------------------------------
    # registry topic
    # -------------------------------------------------------------------------
    reg = topics.add_parser("registry", help="Manage the registry database")
    reg_actions = reg.add_subparsers(dest="action", required=True)

    p = reg_actions.add_parser("init", help="Create a new registry database")
    _add_registry_arg(p)

    p = reg_actions.add_parser("info", help="Print registry info and settings")
    _add_registry_arg(p)

    p = reg_actions.add_parser("set-root", help="Set project_root in registry")
    _add_registry_arg(p)
    p.add_argument("--root", required=True, metavar="DIR")

    # -------------------------------------------------------------------------
    # camera-model topic
    # -------------------------------------------------------------------------
    cm = topics.add_parser("camera-model", help="Manage camera hardware models")
    cm_actions = cm.add_subparsers(dest="action", required=True)

    p = cm_actions.add_parser("add", help="Register a camera hardware model")
    _add_registry_arg(p)
    p.add_argument("--manufacturer", default="", metavar="S")
    p.add_argument("--model-name", default="", metavar="S")
    p.add_argument("--sensor-size", default="", metavar="S")
    p.add_argument("--notes", default="", metavar="S")

    p = cm_actions.add_parser("list", help="List registered camera models")
    _add_registry_arg(p)

    # -------------------------------------------------------------------------
    # camera-mode topic
    # -------------------------------------------------------------------------
    cmo = topics.add_parser("camera-mode", help="Manage camera capture modes")
    cmo_actions = cmo.add_subparsers(dest="action", required=True)

    p = cmo_actions.add_parser("add", help="Register a capture mode for a camera model")
    _add_registry_arg(p)
    p.add_argument("--model-id", required=True, metavar="UUID",
                   help="ID of the parent camera_models row")
    p.add_argument("--width", type=int, default=0, metavar="N",
                   help="Image width in pixels (default: 0 = unknown)")
    p.add_argument("--height", type=int, default=0, metavar="N",
                   help="Image height in pixels (default: 0 = unknown)")
    p.add_argument("--fps", type=float, default=0.0, metavar="F",
                   help="Nominal frames per second (default: 0 = unknown)")
    p.add_argument("--codec", default="", metavar="S")
    p.add_argument("--notes", default="", metavar="S")

    p = cmo_actions.add_parser("list", help="List registered camera modes")
    _add_registry_arg(p)
    p.add_argument("--model-id", default="", metavar="UUID",
                   help="Filter by camera model ID")

    # -------------------------------------------------------------------------
    # calib topic
    # -------------------------------------------------------------------------
    cal = topics.add_parser("calib", help="Manage intrinsic calibrations")
    cal_actions = cal.add_subparsers(dest="action", required=True)

    p = cal_actions.add_parser("list", help="List camera instances and intrinsics calibrations")
    _add_registry_arg(p)
    _add_session_db_arg(p)

    p = cal_actions.add_parser("import",
                               help="Import intrinsic calibration from a Pose2Sim TOML file")
    _add_registry_arg(p)
    p.add_argument("--calib", required=True, metavar="TOML_PATH",
                   help="Path to the Pose2Sim calibration TOML file")
    p.add_argument(
        "--camera-mode", required=True, action="append", metavar="SPEC", dest="camera_mode",
        help=(
            "Camera mode assignment. Two forms: "
            "(1) a single UUID applies to all cameras in the file; "
            "(2) one or more 'camN=UUID' pairs for per-camera assignment. "
            "Example: --camera-mode cam1=<uuid> --camera-mode cam2=<uuid>"
        ),
    )
    p.add_argument("--tool", default="pose2sim", metavar="S",
                   help="Calibration tool name (default: pose2sim)")
    p.add_argument("--distortion-model", default="radtan", metavar="S",
                   help="Distortion model (default: radtan)")
    p.add_argument("--notes", default="", metavar="S")

    # -------------------------------------------------------------------------
    # skeleton topic
    # -------------------------------------------------------------------------
    sk = topics.add_parser("skeleton", help="Manage skeleton definitions")
    sk_actions = sk.add_subparsers(dest="action", required=True)

    p = sk_actions.add_parser("import", help="Import a skeleton YAML file")
    _add_registry_arg(p)
    _add_session_db_arg(p)
    p.add_argument("--global", dest="global_registry", action="store_true",
                   help="Also write to global registry")
    p.add_argument("--file", required=True, metavar="YAML_PATH",
                   help="Path to the skeleton YAML file")
    p.add_argument("--name", default="", metavar="S", help="Human-readable name")
    p.add_argument("--person-label", default="", metavar="S")
    p.add_argument("--source", default="", metavar="S")
    p.add_argument("--parent-id", default="", metavar="SHA256",
                   help="Parent skeleton ID (for lineage)")
    p.add_argument("--notes", default="", metavar="S")

    p = sk_actions.add_parser("list", help="List registered skeletons")
    _add_registry_arg(p)
    _add_session_db_arg(p)

    # -------------------------------------------------------------------------
    # config topic
    # -------------------------------------------------------------------------
    cfg = topics.add_parser("config", help="Manage tracker configurations")
    cfg_actions = cfg.add_subparsers(dest="action", required=True)

    p = cfg_actions.add_parser("create",
                               help="Create a tracker config snapshot from a TOML file")
    _add_registry_arg(p)
    _add_session_db_arg(p)
    p.add_argument("--global", dest="global_registry", action="store_true")
    p.add_argument("--name", required=True, metavar="S",
                   help="Name for this configuration snapshot")
    p.add_argument("--from-toml", required=True, metavar="TOML_PATH",
                   help="Path to the posetrak TOML config file")
    p.add_argument("--notes", default="", metavar="S")

    p = cfg_actions.add_parser("edit",
                               help="Derive a new tracker config by overriding fields")
    _add_registry_arg(p)
    _add_session_db_arg(p)
    p.add_argument("--global", dest="global_registry", action="store_true")
    p.add_argument("--id", required=True, metavar="UUID",
                   help="ID of the existing tracker_configs row to derive from")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--kappa", type=float, default=None)
    p.add_argument("--process-noise-std", type=float, default=None,
                   dest="process_noise_std")
    p.add_argument("--measurement-noise-std", type=float, default=None,
                   dest="measurement_noise_std")
    p.add_argument("--outlier-threshold", type=float, default=None,
                   dest="outlier_threshold")
    p.add_argument("--tracker-fps", type=float, default=None, dest="tracker_fps")
    p.add_argument("--notes", default="", metavar="S")

    p = cfg_actions.add_parser("list", help="List tracker configs")
    _add_registry_arg(p)
    _add_session_db_arg(p)
    p.add_argument("--name", default="", metavar="S", help="Filter by name")

    # -------------------------------------------------------------------------
    # session topic
    # -------------------------------------------------------------------------
    ses = topics.add_parser("session", help="Manage mocap sessions")
    ses_actions = ses.add_subparsers(dest="action", required=True)

    p = ses_actions.add_parser("list", help="List mocap sessions")
    _add_session_db_arg(p, required=True)

    p = ses_actions.add_parser("create", help="Create a new mocap session")
    _add_session_db_arg(p, required=True)
    p.add_argument("--date", default="", metavar="ISO_DATE",
                   help="Recording date (ISO format). Defaults to today.")
    p.add_argument("--location", default="", metavar="S")
    p.add_argument("--notes", default="", metavar="S")

    p = ses_actions.add_parser("add-camera",
                               help="Link a camera to a session (copies registry rows)")
    _add_registry_arg(p)
    _add_session_db_arg(p, required=True)
    p.add_argument("--session", required=True, metavar="UUID",
                   help="mocap_sessions.id")
    p.add_argument("--camera-instance", required=True, metavar="UUID",
                   dest="camera_instance")
    p.add_argument("--camera-mode", required=True, metavar="UUID",
                   dest="camera_mode")
    p.add_argument("--intrinsics", required=True, metavar="UUID")
    p.add_argument("--label", default="", metavar="S")

    # -------------------------------------------------------------------------
    # extrinsics topic
    # -------------------------------------------------------------------------
    ext = topics.add_parser("extrinsics", help="Manage extrinsic calibrations")
    ext_actions = ext.add_subparsers(dest="action", required=True)

    p = ext_actions.add_parser("list", help="List extrinsic calibrations")
    _add_session_db_arg(p, required=True)
    p.add_argument("--session", default=None, metavar="UUID",
                   help="Filter by mocap_sessions.id")

    p = ext_actions.add_parser("import",
                               help="Import extrinsic calibration from a Pose2Sim TOML file")
    _add_registry_arg(p)
    _add_session_db_arg(p, required=True)
    p.add_argument("--session", required=True, metavar="UUID",
                   help="mocap_sessions.id")
    p.add_argument("--calib", required=True, metavar="TOML_PATH")
    p.add_argument("--camera-instance", action="append", metavar="SPEC",
                   dest="camera_instance",
                   help="cam1=<uuid> pairs or single UUID")
    p.add_argument("--method", default="pose2sim", metavar="S")

    # -------------------------------------------------------------------------
    # shot topic
    # -------------------------------------------------------------------------
    shot = topics.add_parser("shot", help="Manage shots within a session")
    shot_actions = shot.add_subparsers(dest="action", required=True)

    p = shot_actions.add_parser("list", help="List shots")
    _add_session_db_arg(p, required=True)
    p.add_argument("--session", default=None, metavar="UUID",
                   help="Filter by mocap_sessions.id")

    p = shot_actions.add_parser("create", help="Create a new shot within a session")
    _add_session_db_arg(p, required=True)
    p.add_argument("--session", required=True, metavar="UUID")
    p.add_argument("--extrinsics", required=True, metavar="UUID",
                   help="extrinsic_calibrations.id")
    p.add_argument("--number", type=int, default=None, metavar="N")
    p.add_argument("--label", default="", metavar="S")
    p.add_argument("--notes", default="", metavar="S")

    p = shot_actions.add_parser("add-video", help="Add a video file to a shot")
    _add_session_db_arg(p, required=True)
    p.add_argument("--shot", required=True, metavar="UUID")
    p.add_argument("--camera-instance", required=True, metavar="UUID",
                   dest="camera_instance")
    p.add_argument("--file", required=True, metavar="PATH")
    p.add_argument("--first-frame", required=True, type=int, metavar="N",
                   dest="first_frame")
    p.add_argument("--last-frame", required=True, type=int, metavar="N",
                   dest="last_frame")
    p.add_argument("--fps", required=True, type=float, metavar="F")

    # -------------------------------------------------------------------------
    # sync topic
    # -------------------------------------------------------------------------
    sync = topics.add_parser("sync", help="Manage sync configurations")
    sync_actions = sync.add_subparsers(dest="action", required=True)

    p = sync_actions.add_parser("list", help="List sync configs")
    _add_session_db_arg(p, required=True)
    p.add_argument("--shot", default=None, metavar="UUID",
                   help="Filter by shots.id")

    p = sync_actions.add_parser("import",
                                help="Import camera sync anchors from a sync JSON file")
    _add_session_db_arg(p, required=True)
    p.add_argument("--shot", required=True, metavar="UUID")
    p.add_argument("--sync-json", required=True, metavar="JSON_PATH", dest="sync_json")
    p.add_argument("--camera-instance", action="append", metavar="SPEC",
                   dest="camera_instance")

    # -------------------------------------------------------------------------
    # pose topic
    # -------------------------------------------------------------------------
    pose = topics.add_parser("pose", help="Manage pose observations")
    pose_actions = pose.add_subparsers(dest="action", required=True)

    p = pose_actions.add_parser("list", help="List pose observation sequences")
    _add_session_db_arg(p, required=True)
    p.add_argument("--shot", default=None, metavar="UUID",
                   help="Filter by shots.id")

    p = pose_actions.add_parser("import",
                                help="Import 2-D pose observations from a pose directory")
    _add_session_db_arg(p, required=True)
    p.add_argument("--shot", required=True, metavar="UUID")
    p.add_argument("--sync-config", required=True, metavar="UUID", dest="sync_config")
    p.add_argument("--pose-dir", required=True, metavar="DIR", dest="pose_dir")
    p.add_argument("--camera-instance", action="append", metavar="SPEC",
                   dest="camera_instance")
    p.add_argument("--person-id", type=int, action="append", default=None,
                   dest="person_ids", metavar="N",
                   help="Import only this person ID (repeatable). Default: import all persons.")
    p.add_argument("--time-start", type=float, default=None, dest="time_start")
    p.add_argument("--time-end", type=float, default=None, dest="time_end")
    p.add_argument("--pose-model", default="", dest="pose_model", metavar="S")

    # -------------------------------------------------------------------------
    # tracking-run topic
    # -------------------------------------------------------------------------
    tr = topics.add_parser("tracking-run", help="Inspect tracking run results")
    tr_actions = tr.add_subparsers(dest="action", required=True)
    p = tr_actions.add_parser("list", help="List tracking runs")
    _add_session_db_arg(p, required=True)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        ("registry", "init"): _cmd_registry_init,
        ("registry", "info"): _cmd_registry_info,
        ("registry", "set-root"): _cmd_registry_set_root,
        ("camera-model", "add"): _cmd_camera_model_add,
        ("camera-model", "list"): _cmd_camera_model_list,
        ("camera-mode", "add"): _cmd_camera_mode_add,
        ("camera-mode", "list"): _cmd_camera_mode_list,
        ("calib", "import"): _cmd_calib_import,
        ("calib", "list"): _cmd_calib_list,
        ("skeleton", "import"): _cmd_skeleton_import,
        ("skeleton", "list"): _cmd_skeleton_list,
        ("config", "create"): _cmd_config_create,
        ("config", "edit"): _cmd_config_edit,
        ("config", "list"): _cmd_config_list,
        ("session", "list"): _cmd_session_list,
        ("session", "create"): _cmd_session_create,
        ("session", "add-camera"): _cmd_session_add_camera,
        ("extrinsics", "list"): _cmd_extrinsics_list,
        ("extrinsics", "import"): _cmd_extrinsics_import,
        ("shot", "list"): _cmd_shot_list,
        ("shot", "create"): _cmd_shot_create,
        ("shot", "add-video"): _cmd_shot_add_video,
        ("sync", "list"): _cmd_sync_list,
        ("sync", "import"): _cmd_sync_import,
        ("pose", "list"): _cmd_pose_list,
        ("pose", "import"): _cmd_pose_import,
        ("tracking-run", "list"): _cmd_tracking_run_list,
    }

    handler = handlers.get((args.topic, args.action))
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))  # type: ignore[operator]


if __name__ == "__main__":
    main()
