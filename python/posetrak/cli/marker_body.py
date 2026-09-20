# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Marker body commands: import, list, show, export, to-skeleton, calibrate.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 10 ("Marker body definitions:
format and storage"). Mirrors posetrak/cli/skeleton.py's structure and
command shape directly -- marker_body_definitions follows the exact same
storage convention as skeletons (SHA-256-of-content id, YAML blob), so
there is no reason for its CLI to look any different.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click

from app.setup.fiducial_markers import load_marker_body_yaml
from posetrak.calibration.rigid_marker_body import CalibrationOptions, calibrate_rigid_marker_body
from posetrak.db.db import open_registry, open_session, resolve_id_prefix
from posetrak.db.manage_marker_body import import_marker_body, import_marker_body_str, list_marker_bodies
from posetrak.db.manage_skeleton import import_skeleton_str
from posetrak.skeleton.marker_body_to_skeleton import generate_prop_skeleton_yaml

from posetrak.cli._output import print_table, print_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_registry(obj: dict):
    path = obj["registry"]
    try:
        return open_registry(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening registry: {exc}") from exc


def _open_session(path: str):
    try:
        return open_session(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening session DB: {exc}") from exc


def _open_conn(obj: dict):
    """Return (conn, label) using session DB if provided, else registry."""
    session_path = obj.get("session")
    if session_path:
        return _open_session(session_path), "session"
    return _open_registry(obj), "registry"


def _resolve_prefix(conn, table: str, prefix: str | None) -> str | None:
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


# ---------------------------------------------------------------------------
# marker-body group
# ---------------------------------------------------------------------------


@click.group("marker-body")
def marker_body_group() -> None:
    """Manage marker body definitions (portable calibration rigs, reusable
    marker-cluster objects -- see extrinsics-improvements-design.md section 10)."""


@marker_body_group.command("import")
@click.option("--file", "yaml_file", required=True, metavar="YAML_PATH",
              help="Path to the marker body definition YAML file")
@click.option("--global", "global_registry", is_flag=True, default=False,
              help="Also write to global registry")
@click.option("--name", default="", metavar="S", help="Human-readable name")
@click.option("--source", default="", metavar="S")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def marker_body_import(
    obj: dict,
    yaml_file: str,
    global_registry: bool,
    name: str,
    source: str,
    notes: str,
) -> None:
    """Import a marker body definition YAML file into the registry and/or a session DB."""
    yaml_path = Path(yaml_file)
    session_path = obj.get("session")

    if session_path is None and not global_registry:
        raise click.UsageError("Specify --session, --global, or both.")

    registry = None
    session = None
    body_id = None

    try:
        if global_registry:
            registry = _open_registry(obj)
        if session_path:
            session = _open_session(session_path)

        if registry is not None:
            try:
                body_id = import_marker_body(
                    registry, yaml_path, name=name or None, source=source or None,
                    notes=notes or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(f"Error importing marker body: {exc}") from exc

        if session is not None:
            try:
                body_id = import_marker_body(
                    session, yaml_path, name=name or None, source=source or None,
                    notes=notes or None,
                )
            except Exception as exc:  # noqa: BLE001
                raise click.ClickException(
                    f"Error importing marker body to session: {exc}"
                ) from exc

    finally:
        if registry is not None:
            registry.close()
        if session is not None:
            session.close()

    click.echo(f"marker_body_id: {body_id}")


@marker_body_group.command("list")
@click.pass_obj
def marker_body_list(obj: dict) -> None:
    """List marker body definitions -- from session DB if provided, otherwise from registry."""
    conn, _ = _open_conn(obj)
    try:
        rows = list_marker_bodies(conn)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No marker body definitions registered.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["id", "name", "source", "created_at"],
        json_mode=obj["json_mode"],
    )


@marker_body_group.command("show")
@click.argument("marker_body_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def marker_body_show(obj: dict, marker_body_id: str) -> None:
    """Show metadata for a marker body definition."""
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "marker_body_definitions", marker_body_id)
        row = conn.execute(
            "SELECT id, name, source, created_at, notes "
            "FROM marker_body_definitions WHERE id = ?",
            (resolved,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise click.ClickException(f"Marker body not found: {marker_body_id}")

    print_record(
        {
            "id": row[0],
            "name": row[1] or "",
            "source": row[2] or "",
            "created_at": row[3] or "",
            "notes": row[4] or "",
        },
        json_mode=obj["json_mode"],
    )


@marker_body_group.command("export")
@click.argument("marker_body_id", metavar="ID_OR_PREFIX")
@click.option("--output", default="", metavar="PATH",
              help="Output file path (- for stdout, default: stdout)")
@click.pass_obj
def marker_body_export(obj: dict, marker_body_id: str, output: str) -> None:
    """Export a marker body definition's YAML to a file or stdout."""
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "marker_body_definitions", marker_body_id)
        row = conn.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (resolved,)
        ).fetchone()
    finally:
        conn.close()

    if row is None or row[0] is None:
        raise click.ClickException(f"Marker body '{marker_body_id}' has no YAML content")

    yaml_content: str = row[0]

    if output and output != "-":
        Path(output).write_text(yaml_content, encoding="utf-8")
        click.echo(f"Exported marker body to {output}")
    else:
        sys.stdout.write(yaml_content)


@marker_body_group.command("to-skeleton")
@click.argument("marker_body_id", metavar="ID_OR_PREFIX")
@click.option("--name", default="", metavar="S",
              help="Generated skeleton's name (default: the marker body's own name)")
@click.option("--output", default="", metavar="PATH",
              help="Also write the generated skeleton YAML to a file (- for stdout)")
@click.pass_obj
def marker_body_to_skeleton(obj: dict, marker_body_id: str, name: str, output: str) -> None:
    """Generate a prop tracking skeleton from a marker body definition and
    import it (marker-based-mocap design doc §5.3).

    One free-flyer root, markers only -- no articulated joints. Import is
    content-addressed and idempotent, same as `skeleton import`: running
    this again for the same marker body (and the same --name) returns the
    existing skeleton id rather than creating a duplicate row.
    """
    conn, _ = _open_conn(obj)
    try:
        resolved = _resolve_prefix(conn, "marker_body_definitions", marker_body_id)
        row = conn.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (resolved,)
        ).fetchone()
        if row is None or row[0] is None:
            raise click.ClickException(f"Marker body '{marker_body_id}' has no YAML content")

        config = load_marker_body_yaml(row[0])
        try:
            skeleton_yaml = generate_prop_skeleton_yaml(
                config, name=name or None, marker_body_definition_id=resolved,
            )
        except ValueError as exc:
            raise click.ClickException(f"Error generating skeleton: {exc}") from exc

        skeleton_id = import_skeleton_str(
            conn, skeleton_yaml,
            name=name or config.rig_id,
            source=f"marker_body:{resolved}",
        )
    finally:
        conn.close()

    if output:
        if output == "-":
            sys.stdout.write(skeleton_yaml)
        else:
            Path(output).write_text(skeleton_yaml, encoding="utf-8")
            click.echo(f"Wrote generated skeleton YAML to {output}")

    click.echo(f"skeleton_id: {skeleton_id}")


_CALIBRATION_DEFAULTS = CalibrationOptions(marker_ids=[], reference_id="", marker_size=1.0)


@marker_body_group.command("calibrate")
@click.option("--capture", required=True, metavar="ID", help="Capture to calibrate from (prefix accepted).")
@click.option("--time-start", type=float, required=True, help="Start of the footage to use, global seconds.")
@click.option("--time-end", type=float, required=True, help="End of the footage to use, global seconds.")
@click.option("--marker-size", type=float, required=True, help="Side length of the body's ArUco markers, metres.")
@click.option("--marker-ids", required=True, metavar="ID,ID,...", help="Every ArUco id on the body.")
@click.option("--reference-id", required=True, help="The marker whose frame becomes the body's frame.")
@click.option("--dictionary", default=_CALIBRATION_DEFAULTS.dictionary, show_default=True)
@click.option("--stride", type=int, default=_CALIBRATION_DEFAULTS.stride, show_default=True,
              help="Process every Nth decoded frame.")
@click.option("--min-cameras", type=int, default=_CALIBRATION_DEFAULTS.min_cameras, show_default=True,
              help="Cameras that must see a marker at once for its pose to be solved.")
@click.option("--camera", "cameras", multiple=True, metavar="LABEL",
              help="Use only this camera, by label; repeat for several. Needed when a marker id is "
                   "ambiguous in some camera's view, for instance a same-id tag on an unrelated prop.")
@click.option("--name", default=_CALIBRATION_DEFAULTS.name, show_default=True, help="Name of the marker body.")
@click.option("--output", default=None, metavar="PATH", help="Write the marker body YAML here.")
@click.option("--import", "import_to_session", is_flag=True, default=False,
              help="Import the marker body into the session (its ID is printed).")
@click.option("--detect-dots", is_flag=True, default=False,
              help="Also calibrate the body's reflective dots, from the same footage.")
@click.option("--dot-threshold", type=int, default=_CALIBRATION_DEFAULTS.dot_threshold, show_default=True,
              help="Brightness threshold, or the background residual with --dot-bg-subtract.")
@click.option("--dot-min-area", type=float, default=_CALIBRATION_DEFAULTS.dot_min_area, show_default=True)
@click.option("--dot-max-area", type=float, default=_CALIBRATION_DEFAULTS.dot_max_area, show_default=True)
@click.option("--dot-min-compactness", type=float, default=_CALIBRATION_DEFAULTS.dot_min_compactness,
              show_default=True)
@click.option("--dot-cluster-tolerance-m", type=float, default=_CALIBRATION_DEFAULTS.dot_cluster_tolerance_m,
              show_default=True, help="Largest distance for a dot sample to join an existing dot.")
@click.option("--dot-max-reprojection-px", type=float, default=_CALIBRATION_DEFAULTS.dot_max_reprojection_px,
              show_default=True, help="Reject a dot sample any of whose views reprojects farther than this.")
@click.option("--dot-bg-subtract", is_flag=True, default=False,
              help="Threshold the residual against each camera's median background. Needed when the dots "
                   "are dimmer than a bright feature of the scene. Costs an extra decode pass per camera.")
@click.option("--dot-bg-samples", type=int, default=_CALIBRATION_DEFAULTS.dot_bg_samples, show_default=True,
              help="Frames sampled to build the median background.")
@click.option("--dot-gate-radius-mult", type=float, default=_CALIBRATION_DEFAULTS.dot_gate_radius_mult,
              show_default=True,
              help="If above 0, keep only dot candidates within this many on-screen marker diagonals of a "
                   "visible marker, and none in a frame with no marker visible.")
@click.option("--dot-quad-exclude-scale", type=float, default=_CALIBRATION_DEFAULTS.dot_quad_exclude_scale,
              show_default=True,
              help="With --dot-gate-radius-mult, drop candidates inside a marker's own quad dilated by "
                   "this factor (its edges leak through background subtraction). 0 turns it off.")
@click.pass_obj
def marker_body_calibrate(
    obj: dict,
    capture: str,
    time_start: float,
    time_end: float,
    marker_size: float,
    marker_ids: str,
    reference_id: str,
    dictionary: str,
    stride: int,
    min_cameras: int,
    cameras: tuple[str, ...],
    name: str,
    output: str | None,
    import_to_session: bool,
    detect_dots: bool,
    dot_threshold: int,
    dot_min_area: float,
    dot_max_area: float,
    dot_min_compactness: float,
    dot_cluster_tolerance_m: float,
    dot_max_reprojection_px: float,
    dot_bg_subtract: bool,
    dot_bg_samples: int,
    dot_gate_radius_mult: float,
    dot_quad_exclude_scale: float,
) -> None:
    """Solve a rigid marker body from ordinary multi-camera footage.

    Where the body's ArUco markers, and with --detect-dots its reflective dots, sit
    relative to the reference marker follows from the footage alone: no turn-around
    video is needed, and no two markers have to be visible together from one camera.
    Every marker only has to be seen in the same instant as the reference marker
    somewhere in the footage, possibly by different cameras.

    The session is only read, unless --import is given.

    Example:

        posetrak -s session.db marker-body calibrate --capture <id> \\
            --time-start 34.4 --time-end 100.6 --marker-size 0.05 \\
            --marker-ids 2,3 --reference-id 2 --detect-dots --import
    """
    if not output and not import_to_session:
        raise click.UsageError("Give --output PATH, --import, or both: there is nowhere to put the result.")
    session_path = obj.get("session")
    if not session_path:
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    options = CalibrationOptions(
        marker_ids=[m.strip() for m in marker_ids.split(",") if m.strip()],
        reference_id=reference_id.strip(), marker_size=marker_size, dictionary=dictionary, stride=stride,
        min_cameras=min_cameras, name=name, camera_labels=list(cameras) or None, detect_dots=detect_dots,
        dot_threshold=dot_threshold, dot_min_area=dot_min_area, dot_max_area=dot_max_area,
        dot_min_compactness=dot_min_compactness, dot_cluster_tolerance_m=dot_cluster_tolerance_m,
        dot_max_reprojection_px=dot_max_reprojection_px, dot_bg_subtract=dot_bg_subtract,
        dot_bg_samples=dot_bg_samples, dot_gate_radius_mult=dot_gate_radius_mult,
        dot_quad_exclude_scale=dot_quad_exclude_scale,
    )
    try:
        options.validate()
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if not Path(session_path).is_file():
        raise click.ClickException(f"Error opening session DB: {session_path} does not exist")
    conn = sqlite3.connect(f"file:{session_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        capture_id = _resolve_prefix(conn, "captures", capture)
        try:
            result = calibrate_rigid_marker_body(
                conn, capture_id, time_start, time_end, options,
                log=lambda message: click.echo(message, err=True),
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if output:
        Path(output).write_text(result.yaml, encoding="utf-8")
        click.echo(f"Wrote {output}", err=True)
    if import_to_session:
        conn = _open_session(session_path)
        try:
            body_id = import_marker_body_str(
                conn, result.yaml, name=name, source=f"calibrated from capture {capture_id}",
            )
        finally:
            conn.close()
        click.echo(f"marker_body_definition_id: {body_id}")
