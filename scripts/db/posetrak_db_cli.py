"""posetrak_db_cli.py — Command-line interface for the posetrak database layer.

Subcommands
-----------
init              Create a new registry database.
import-calib      Import a Pose2Sim calibration TOML into the registry.
set-project-root  Set the project root path stored in the registry settings.
info              Print registry schema version and settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.db.posetrak_db import (
    REGISTRY_SCHEMA_VERSION,
    create_registry,
    get_project_root,
    get_schema_version,
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


def _cmd_import_calib(args: argparse.Namespace) -> int:
    """Import a Pose2Sim calibration TOML into the registry."""
    registry_path = Path(args.registry)
    calib_path = Path(args.calib)

    try:
        registry = open_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        result = import_calib_toml(
            registry,
            calib_path,
            width_px=args.width,
            height_px=args.height,
            nominal_fps=args.fps,
            codec=args.codec,
            calibration_tool=args.tool,
            distortion_model=args.distortion_model,
            notes=args.notes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error importing calibration: {exc}", file=sys.stderr)
        registry.close()
        return 1
    finally:
        registry.close()

    n_cameras = len(result.camera_instance_ids)
    print(f"Imported {calib_path.name}:")
    print(f"  camera_model_id : {result.camera_model_id}")
    print(f"  cameras imported: {n_cameras}")
    for label, iid in result.camera_instance_ids.items():
        mode_id = result.camera_mode_ids[label]
        intr_id = result.intrinsics_ids[label]
        print(f"    {label!r}")
        print(f"      instance_id  : {iid}")
        print(f"      mode_id      : {mode_id}")
        print(f"      intrinsics_id: {intr_id}")
    return 0


def _cmd_set_project_root(args: argparse.Namespace) -> int:
    """Set the project root path in the registry settings."""
    registry_path = Path(args.registry)
    root = Path(args.root).resolve()

    try:
        registry = open_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        set_project_root(registry, root)
    finally:
        registry.close()

    print(f"project_root set to: {root}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """Print registry schema version and settings."""
    registry_path = Path(args.registry)

    try:
        registry = open_registry(registry_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error opening registry: {exc}", file=sys.stderr)
        return 1

    try:
        version = get_schema_version(registry)
        project_root = get_project_root(registry)
        rows = registry.execute("SELECT key, value FROM settings").fetchall()
    finally:
        registry.close()

    print(f"Registry: {registry_path}")
    print(f"  schema version : {version} (expected {REGISTRY_SCHEMA_VERSION})")
    print(f"  project_root   : {project_root}")
    if rows:
        print("  settings:")
        for row in rows:
            print(f"    {row['key']} = {row['value']!r}")
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

    # init
    p_init = sub.add_parser("init", help="Create a new registry database")
    p_init.add_argument("--registry", required=True, metavar="PATH",
                        help="Path for the new registry .db file")

    # import-calib
    p_ic = sub.add_parser("import-calib", help="Import a Pose2Sim calibration TOML")
    p_ic.add_argument("--registry", required=True, metavar="PATH",
                      help="Path to the registry .db file")
    p_ic.add_argument("--calib", required=True, metavar="TOML_PATH",
                      help="Path to the Pose2Sim calibration TOML file")
    p_ic.add_argument("--width", type=int, default=0, metavar="N",
                      help="Image width in pixels (default: 0)")
    p_ic.add_argument("--height", type=int, default=0, metavar="N",
                      help="Image height in pixels (default: 0)")
    p_ic.add_argument("--fps", type=float, default=0.0, metavar="F",
                      help="Nominal frames per second (default: 0.0)")
    p_ic.add_argument("--codec", default="", metavar="S",
                      help="Codec identifier string (default: '')")
    p_ic.add_argument("--tool", default="pose2sim", metavar="S",
                      help="Calibration tool name (default: pose2sim)")
    p_ic.add_argument("--distortion-model", default="radtan", metavar="S",
                      help="Distortion model (default: radtan)")
    p_ic.add_argument("--notes", default="", metavar="S",
                      help="Free-text notes stored with each intrinsics row")

    # set-project-root
    p_spr = sub.add_parser("set-project-root",
                            help="Set the project_root setting in the registry")
    p_spr.add_argument("--registry", required=True, metavar="PATH",
                       help="Path to the registry .db file")
    p_spr.add_argument("--root", required=True, metavar="DIR",
                       help="Project root directory path to store")

    # info
    p_info = sub.add_parser("info", help="Print registry info and settings")
    p_info.add_argument("--registry", required=True, metavar="PATH",
                        help="Path to the registry .db file")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand handler."""
    parser = _build_parser()
    args = parser.parse_args()

    handlers: dict[str, object] = {
        "init": _cmd_init,
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
