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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.db.posetrak_db import (
    REGISTRY_SCHEMA_VERSION,
    create_camera_model,
    create_camera_mode,
    create_registry,
    get_project_root,
    get_schema_version,
    list_camera_models,
    list_camera_modes,
    open_registry,
    set_project_root,
)
from scripts.db.import_calib_toml import import_calib_toml


# ---------------------------------------------------------------------------
# Subcommand handlers
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
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posetrak_db_cli",
        description="posetrak database management CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- init ---
    p_init = sub.add_parser("init", help="Create a new registry database")
    p_init.add_argument("--registry", required=True, metavar="PATH",
                        help="Path for the new registry .db file")

    # --- camera-model-add ---
    p_cma = sub.add_parser("camera-model-add", help="Register a camera hardware model")
    p_cma.add_argument("--registry", required=True, metavar="PATH")
    p_cma.add_argument("--manufacturer", default="", metavar="S")
    p_cma.add_argument("--model-name", default="", metavar="S")
    p_cma.add_argument("--sensor-size", default="", metavar="S")
    p_cma.add_argument("--notes", default="", metavar="S")

    # --- camera-model-list ---
    p_cml = sub.add_parser("camera-model-list", help="List registered camera models")
    p_cml.add_argument("--registry", required=True, metavar="PATH")

    # --- camera-mode-add ---
    p_coda = sub.add_parser(
        "camera-mode-add",
        help="Register a capture mode (resolution/fps) for a camera model",
    )
    p_coda.add_argument("--registry", required=True, metavar="PATH")
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
    p_codl.add_argument("--registry", required=True, metavar="PATH")
    p_codl.add_argument("--model-id", default="", metavar="UUID",
                        help="Filter by camera model ID")

    # --- import-calib ---
    p_ic = sub.add_parser(
        "import-calib",
        help="Import intrinsic calibration from a Pose2Sim TOML file",
    )
    p_ic.add_argument("--registry", required=True, metavar="PATH")
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
    p_spr.add_argument("--registry", required=True, metavar="PATH")
    p_spr.add_argument("--root", required=True, metavar="DIR")

    # --- info ---
    p_info = sub.add_parser("info", help="Print registry info and settings")
    p_info.add_argument("--registry", required=True, metavar="PATH")

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
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))  # type: ignore[operator]


if __name__ == "__main__":
    main()
