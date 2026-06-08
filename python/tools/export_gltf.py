"""
Export posetrak tracking results to glTF 2.0 skeletal animation format — CLI entry point.

The export logic lives in posetrak.export.gltf.  Run via:

    uv run python/tools/export_gltf.py <tracking_dir> --skeleton <skel.yaml> --output take1.glb
    uv run python/tools/export_gltf.py --session-db session.db --run-id <uuid> --output take1.glb

To call programmatically:

    from posetrak.export.gltf import export_gltf

No extra packages required (uses only numpy and stdlib).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the posetrak package importable when run directly (not via uv / editable install)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.export.gltf import export_gltf  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export posetrak tracking results to glTF 2.0 skeletal animation format.")
    parser.add_argument("tracking_dir", type=Path, nargs="?", default=None,
                        help="Directory containing root_pose.csv and joint_angles.csv "
                             "(not required when using --session-db)")
    parser.add_argument("--skeleton", "-s", type=Path, default=None,
                        help="Skeleton YAML file (not required when using --session-db)")
    parser.add_argument("--session-db", type=str, default=None,
                        help="Path to session SQLite database")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Tracking run UUID (required with --session-db)")
    parser.add_argument("--person-id", type=int, default=0,
                        help="Person ID to export (default: 0, used with --session-db)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output file: .glb (binary, recommended) or .gltf (JSON). "
                             "Default: <tracking_dir>/tracking.glb or ./tracking.glb in DB mode")
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate (default: auto-detect from timestamps)")
    parser.add_argument("--units", choices=["m", "cm"], default="m",
                        help="Position units (default: m)")
    parser.add_argument("--coord", choices=["yup", "zup"], default="yup",
                        help="Target coordinate system for the root node. "
                             "yup: Y-up, Z-forward (Blender, Unity, Maya; default). "
                             "zup: Z-up, Y-backward (unchanged tracker frame).")
    parser.add_argument("--no-rest-frame", action="store_true",
                        help="Omit time-code-0 rest pose frame (not recommended)")
    parser.add_argument("--smoothed", action="store_true",
                        help="Use smoothed results (--session-db) or "
                             "smoothed_joint_angles.csv / smoothed_root_pose.csv (CSV mode)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="First tracking frame to export (1-based)")
    parser.add_argument("--end-frame", type=int, default=None,
                        help="Last tracking frame to export (1-based, inclusive)")
    args = parser.parse_args()

    if args.session_db is not None:
        if args.run_id is None:
            print("Error: --run-id is required when using --session-db", file=sys.stderr)
            sys.exit(1)
        output = args.output or Path("tracking.glb")
        print(f"Loading tracking run {args.run_id!r} from {args.session_db!r}")
    else:
        if args.tracking_dir is None:
            print("Error: tracking_dir is required when not using --session-db",
                  file=sys.stderr)
            sys.exit(1)
        if args.skeleton is None:
            print("Error: --skeleton is required when not using --session-db",
                  file=sys.stderr)
            sys.exit(1)
        output = args.output or (args.tracking_dir / "tracking.glb")
        print(f"Loading skeleton: {args.skeleton}")

    print(f"Writing: {output}")

    try:
        export_gltf(
            output,
            session_db=args.session_db,
            run_id=args.run_id,
            person_id=args.person_id,
            tracking_dir=args.tracking_dir,
            skeleton_path=args.skeleton,
            fps=args.fps,
            units=args.units,
            coord=args.coord,
            smoothed=args.smoothed,
            include_rest_frame=not args.no_rest_frame,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()
