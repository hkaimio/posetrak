# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Session, capture, extrinsics, and sync commands."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import click

from posetrak.db.db import (
    add_capture_video,
    add_session_camera,
    create_capture,
    create_mocap_session,
    create_session,
    open_registry,
    open_session,
    resolve_id_prefix,
    set_capture_extrinsics,
)
from posetrak.db.import_extrinsics import import_extrinsics
from posetrak.db.import_session_yaml import import_session_yaml
from posetrak.db.import_sync_json import import_sync_json
from posetrak.db.trial_export import open_source_readonly

from posetrak.cli._camera import resolve_camera_instance

from posetrak.cli._output import print_table, print_record


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _open_session_required(obj: dict) -> sqlite3.Connection:
    """Open session DB from ctx.obj; fail clearly if not provided."""
    path = obj.get("session")
    if not path:
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    try:
        return open_session(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening session DB: {exc}") from exc


def _open_registry(obj: dict) -> sqlite3.Connection:
    path = obj["registry"]
    try:
        return open_registry(Path(path))
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening registry: {exc}") from exc


def _resolve(conn: sqlite3.Connection, table: str, prefix: str | None) -> str | None:
    if prefix is None:
        return None
    try:
        return resolve_id_prefix(conn, table, prefix)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _parse_camera_instances(
    values: tuple[str, ...] | None,
) -> str | dict[str, str] | None:
    if not values:
        return None
    values = list(values)
    if len(values) == 1 and "=" not in values[0]:
        return values[0]
    result: dict[str, str] = {}
    for v in values:
        if "=" not in v:
            raise click.UsageError(f"--camera-instance value {v!r} is ambiguous.")
        key, _, uuid = v.partition("=")
        if not key or not uuid:
            raise click.UsageError(f"Malformed --camera-instance value: {v!r}")
        result[key] = uuid
    return result


# ---------------------------------------------------------------------------
# session group
# ---------------------------------------------------------------------------


@click.group("session")
def session_group() -> None:
    """Manage mocap sessions."""


@session_group.command("list")
@click.pass_obj
def session_list(obj: dict) -> None:
    """List mocap sessions in a session database."""
    conn = _open_session_required(obj)
    try:
        rows = conn.execute(
            "SELECT id, recorded_at, location FROM mocap_sessions ORDER BY recorded_at"
        ).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("(no sessions)")
        return

    print_table(
        [{"id": r[0], "recorded_at": r[1], "location": r[2] or ""} for r in rows],
        columns=["id", "recorded_at", "location"],
        json_mode=obj["json_mode"],
    )


@session_group.command("create")
@click.option("--date", default="", metavar="ISO_DATE",
              help="Recording date (ISO format). Defaults to today.")
@click.option("--location", default="", metavar="S")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def session_create(obj: dict, date: str, location: str, notes: str) -> None:
    """Create a new mocap session in a session database."""
    path = obj.get("session")
    if not path:
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    sess_path = Path(path)
    try:
        if sess_path.exists():
            conn = open_session(sess_path)
        else:
            conn = create_session(sess_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening session db: {exc}") from exc

    try:
        session_id = create_mocap_session(
            conn,
            recorded_at=date or None,
            location=location or "",
            notes=notes or "",
        )
    finally:
        conn.close()

    click.echo(f"session_id: {session_id}")


@session_group.command("show")
@click.argument("session_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def session_show(obj: dict, session_id: str) -> None:
    """Show details for a mocap session."""
    conn = _open_session_required(obj)
    try:
        resolved = _resolve(conn, "mocap_sessions", session_id)
        row = conn.execute(
            "SELECT id, recorded_at, location, notes FROM mocap_sessions WHERE id = ?",
            (resolved,),
        ).fetchone()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if row is None:
        raise click.ClickException(f"Session not found: {session_id}")

    print_record(
        {"id": row[0], "recorded_at": row[1], "location": row[2] or "", "notes": row[3] or ""},
        json_mode=obj["json_mode"],
    )


@session_group.command("import-yaml")
@click.argument("yaml_file", metavar="YAML_FILE")
@click.option("--session-label", default="", metavar="S",
              help="Override the 'name' field from the YAML as the session notes")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print what would be created without writing to the database")
@click.pass_obj
def session_import_yaml(
    obj: dict, yaml_file: str, session_label: str, dry_run: bool
) -> None:
    """Import a capture project YAML into a session database."""
    registry = _open_registry(obj)

    path = obj.get("session")
    if not path:
        registry.close()
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    sess_path = Path(path)
    try:
        if sess_path.exists():
            session_conn = open_session(sess_path)
        else:
            session_conn = create_session(sess_path)
    except (FileNotFoundError, ValueError) as exc:
        registry.close()
        raise click.ClickException(f"Error opening session db: {exc}") from exc

    try:
        result = import_session_yaml(
            session_conn,
            registry,
            Path(yaml_file),
            session_label=session_label or "",
            dry_run=dry_run,
        )
    except Exception as exc:  # noqa: BLE001
        session_conn.close()
        registry.close()
        raise click.ClickException(f"Error importing session YAML: {exc}") from exc
    finally:
        session_conn.close()
        registry.close()

    if dry_run:
        return

    click.echo(f"session_id: {result.session_id}")
    for cam_key, iid in result.camera_instance_ids.items():
        click.echo(f"  camera {cam_key}: instance={iid}")
    for label, shot_id in result.shot_ids.items():
        sync_id = result.sync_config_ids.get(label, "")
        click.echo(f"  shot {label!r}: id={shot_id}  sync_config={sync_id}")


# ---------------------------------------------------------------------------
# session add-camera
# ---------------------------------------------------------------------------


def _resolve_mode_and_intrinsics(
    conn: sqlite3.Connection,
    camera_instance_id: str,
    camera_label: str,
    capture_id: str | None,
) -> tuple[str, str]:
    """Return (camera_mode_id, intrinsics_calibration_id) for *camera_instance_id*.

    *conn* may be a full session DB (with capture history) or a bare
    registry (camera_models/modes/instances/intrinsics only, no captures
    table at all). Resolution order:

    1. If *capture_id* is given, use exactly that capture's capture_videos
       row for this camera.
    2. Otherwise, if this camera has capture_videos history, look for a
       single distinct (mode, intrinsics) pair across every capture that
       used it — cameras are session-wide, but their calibration can change
       between captures (a lens getting bumped, refocused, recalibrated),
       so this only auto-resolves when there's genuinely one answer.
    3. Otherwise (a bare registry, or a camera never used in a capture),
       fall back to its camera_modes.default_intrinsics_calibration_id.
    """
    if capture_id is not None:
        resolved_capture = _resolve(conn, "captures", capture_id)
        row = conn.execute(
            "SELECT camera_mode_id, intrinsics_calibration_id FROM capture_videos "
            "WHERE shot_id = ? AND camera_instance_id = ?",
            (resolved_capture, camera_instance_id),
        ).fetchone()
        if row is None:
            raise click.ClickException(
                f"Camera {camera_label!r} was not used in capture {capture_id!r}."
            )
        if row["intrinsics_calibration_id"] is None:
            raise click.ClickException(
                f"Camera {camera_label!r} in capture {capture_id!r} has no "
                "intrinsics calibration recorded."
            )
        return row["camera_mode_id"], row["intrinsics_calibration_id"]

    try:
        pairs = conn.execute(
            "SELECT DISTINCT camera_mode_id, intrinsics_calibration_id, shot_id "
            "FROM capture_videos "
            "WHERE camera_instance_id = ? AND intrinsics_calibration_id IS NOT NULL",
            (camera_instance_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        pairs = []  # bare registry: no capture_videos table at all

    distinct = {(r["camera_mode_id"], r["intrinsics_calibration_id"]) for r in pairs}
    if len(distinct) == 1:
        return next(iter(distinct))
    if len(distinct) > 1:
        by_capture = ", ".join(f"{r['shot_id']}→{r['intrinsics_calibration_id']}" for r in pairs)
        raise click.ClickException(
            f"Camera {camera_label!r} has {len(distinct)} different intrinsics "
            f"calibrations across captures in the source ({by_capture}). "
            "Pass --capture to pick which one."
        )

    # No capture_videos history at all (bare registry, or camera never used
    # in a capture yet) — fall back to the mode's own recorded default.
    row = conn.execute(
        "SELECT cmo.id AS mode_id, cmo.default_intrinsics_calibration_id AS intr_id "
        "FROM camera_modes cmo "
        "JOIN camera_instances ci ON ci.camera_model_id = cmo.camera_model_id "
        "WHERE ci.id = ? AND cmo.default_intrinsics_calibration_id IS NOT NULL",
        (camera_instance_id,),
    ).fetchall()
    if len(row) == 1:
        return row[0]["mode_id"], row[0]["intr_id"]
    raise click.ClickException(
        f"Could not determine a camera mode/intrinsics calibration for "
        f"{camera_label!r} automatically. Pass --capture to pick one "
        "explicitly, or the source has no calibration for this camera at all."
    )


@session_group.command("add-camera")
@click.option("--from", "from_path", required=True, metavar="PATH",
              help="Source database (a session DB or a registry) to copy the "
                   "camera's registry rows from.")
@click.option("--camera", "cameras", multiple=True, required=True, metavar="LABEL",
              help="Camera to clone, by label or camera_instances.id prefix. Repeatable.")
@click.option("--capture", "capture_id", default=None, metavar="ID",
              help="Disambiguate which capture's mode/intrinsics to use, if a "
                   "camera was recalibrated between captures in the source.")
@click.option("--session", "mocap_session_id", default=None, metavar="UUID",
              help="mocap_sessions.id to add the camera(s) to. Only needed if "
                   "the target session DB already has more than one — a fresh "
                   "DB's sole session, or a newly created one, is used "
                   "automatically.")
@click.pass_obj
def session_add_camera(
    obj: dict,
    from_path: str,
    cameras: tuple[str, ...],
    capture_id: str | None,
    mocap_session_id: str | None,
) -> None:
    """Clone one or more cameras' registry rows into a session.

    Copies camera_models/camera_modes/camera_instances/intrinsics_calibrations
    for each --camera from the source DB (any session or registry) into the
    target session (--session, created if it doesn't exist yet), and links
    each one via session_cameras. Does not touch captures, videos, trials,
    or tracking data — this is camera setup only, e.g. to start a new
    session pre-configured with an existing session's camera intrinsics.

    Examples:

        posetrak --session tutorial1.db session add-camera \\
            --from ukemi-tommi-20260509.db --capture 6ca933a5 \\
            --camera insta_ace2_pro --camera gopro-11_mini_02 --camera pixel9
    """
    path = obj.get("session")
    if not path:
        raise click.UsageError(
            "A session DB path is required. Use --session PATH or set $POSETRAK_SESSION_DB."
        )
    sess_path = Path(path)
    try:
        session_conn = open_session(sess_path) if sess_path.exists() else create_session(sess_path)
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(f"Error opening session db: {exc}") from exc

    try:
        source_conn = open_source_readonly(Path(from_path))
    except Exception as exc:  # noqa: BLE001
        session_conn.close()
        raise click.ClickException(f"Error opening source {from_path!r}: {exc}") from exc

    try:
        if mocap_session_id is not None:
            session_id = _resolve(session_conn, "mocap_sessions", mocap_session_id)
        else:
            existing = session_conn.execute("SELECT id FROM mocap_sessions").fetchall()
            if len(existing) == 0:
                session_id = create_mocap_session(session_conn)
            elif len(existing) == 1:
                session_id = existing[0][0]
            else:
                raise click.ClickException(
                    f"{len(existing)} mocap_sessions rows exist in {sess_path} — "
                    "pass --session UUID to pick one."
                )

        for label_or_id in cameras:
            camera_instance_id = resolve_camera_instance(source_conn, label_or_id)
            camera_label = source_conn.execute(
                "SELECT label FROM camera_instances WHERE id = ?", (camera_instance_id,)
            ).fetchone()[0]
            camera_mode_id, intrinsics_id = _resolve_mode_and_intrinsics(
                source_conn, camera_instance_id, camera_label, capture_id
            )
            try:
                add_session_camera(
                    session_conn, source_conn, session_id,
                    camera_instance_id, camera_mode_id, intrinsics_id,
                    label=camera_label,
                )
            except sqlite3.IntegrityError as exc:
                raise click.ClickException(
                    f"Camera {camera_label!r} could not be added: {exc}"
                ) from exc
            click.echo(f"Added camera {camera_label!r} (instance={camera_instance_id})")
    finally:
        source_conn.close()
        session_conn.close()

    click.echo(f"session_id: {session_id}")


# ---------------------------------------------------------------------------
# capture group
# ---------------------------------------------------------------------------


@click.group("capture")
def capture_group() -> None:
    """Manage captures within a session."""


@capture_group.command("list")
@click.option("--session", default=None, metavar="UUID",
              help="Filter by mocap_sessions.id")
@click.pass_obj
def capture_list(obj: dict, session: str | None) -> None:
    """List captures in a session database."""
    conn = _open_session_required(obj)
    try:
        q = "SELECT id, session_id, capture_number, label, extrinsic_calibration_id FROM captures"
        params: list = []
        if session:
            q += " WHERE session_id = ?"
            params.append(session)
        q += " ORDER BY capture_number"
        rows = conn.execute(q, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("(no captures)")
        return

    print_table(
        [
            {
                "id": r[0],
                "session_id": r[1],
                "capture_number": str(r[2] or ""),
                "label": r[3] or "",
                "extrinsic_calibration_id": r[4] or "",
            }
            for r in rows
        ],
        columns=["id", "session_id", "capture_number", "label", "extrinsic_calibration_id"],
        json_mode=obj["json_mode"],
    )


@capture_group.command("show")
@click.argument("capture_id", metavar="ID_OR_PREFIX")
@click.pass_obj
def capture_show(obj: dict, capture_id: str) -> None:
    """Show details for one capture, including its video files."""
    conn = _open_session_required(obj)
    try:
        resolved = _resolve(conn, "captures", capture_id)
        row = conn.execute(
            "SELECT id, session_id, capture_number, label, extrinsic_calibration_id, notes "
            "FROM captures WHERE id = ?",
            (resolved,),
        ).fetchone()
        if row is None:
            raise click.ClickException(f"Capture not found: {capture_id}")

        video_rows = conn.execute(
            """
            SELECT cv.id, ci.label AS camera_label, cv.file_path, cv.actual_fps
            FROM capture_videos cv
            JOIN camera_instances ci ON ci.id = cv.camera_instance_id
            WHERE cv.shot_id = ?
            ORDER BY ci.label
            """,
            (resolved,),
        ).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    videos = [
        {
            "id":         r["id"],
            "camera":     r["camera_label"] or "",
            "file_path":  r["file_path"],
            "fps":        r["actual_fps"],
            "exists":     Path(r["file_path"]).exists(),
        }
        for r in video_rows
    ]

    json_mode: bool = obj["json_mode"]
    record = {
        "id": row[0],
        "session_id": row[1],
        "capture_number": row[2],
        "label": row[3] or "",
        "extrinsic_calibration_id": row[4] or "",
        "notes": row[5] or "",
    }

    if json_mode:
        record["videos"] = videos
        print_record(record, json_mode=True)
    else:
        print_record(record, json_mode=False)
        if videos:
            click.echo("")
            click.echo("Videos:")
            print_table(
                [
                    {
                        "id":        v["id"],
                        "camera":    v["camera"],
                        "file_path": v["file_path"],
                        "fps":       str(v["fps"]),
                        "exists":    "yes" if v["exists"] else "NO",
                    }
                    for v in videos
                ],
                columns=["id", "camera", "file_path", "fps", "exists"],
                json_mode=False,
            )


@capture_group.command("create")
@click.option("--session", required=True, metavar="UUID")
@click.option("--extrinsics", required=True, metavar="UUID",
              help="extrinsic_calibrations.id")
@click.option("--number", type=int, default=None, metavar="N")
@click.option("--label", default="", metavar="S")
@click.option("--notes", default="", metavar="S")
@click.pass_obj
def capture_create(
    obj: dict,
    session: str,
    extrinsics: str,
    number: int | None,
    label: str,
    notes: str,
) -> None:
    """Create a new capture within a session."""
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session)
        extrinsics_id = _resolve(conn, "extrinsic_calibrations", extrinsics)
        shot_id = create_capture(
            conn,
            session_id,
            extrinsics_id,
            capture_number=number,
            label=label or "",
            notes=notes or "",
        )
    finally:
        conn.close()

    click.echo(f"capture_id: {shot_id}")


@capture_group.command("add-video")
@click.option("--shot", required=True, metavar="UUID",
              help="captures.id")
@click.option("--camera-instance", required=True, metavar="UUID")
@click.option("--file", "file_path", required=True, metavar="PATH")
@click.option("--first-frame", required=True, type=int, metavar="N")
@click.option("--last-frame", required=True, type=int, metavar="N")
@click.option("--fps", required=True, type=float, metavar="F")
@click.pass_obj
def capture_add_video(
    obj: dict,
    shot: str,
    camera_instance: str,
    file_path: str,
    first_frame: int,
    last_frame: int,
    fps: float,
) -> None:
    """Add a video file record to a capture."""
    conn = _open_session_required(obj)
    try:
        shot_id = _resolve(conn, "captures", shot)
        camera_instance_id = _resolve(conn, "camera_instances", camera_instance)
        video_id = add_capture_video(
            conn,
            shot_id,
            camera_instance_id,
            file_path,
            first_frame,
            last_frame,
            fps,
        )
    finally:
        conn.close()

    click.echo(f"capture_video_id: {video_id}")


# ---------------------------------------------------------------------------
# extrinsics group
# ---------------------------------------------------------------------------


@click.group("extrinsics")
def extrinsics_group() -> None:
    """Manage extrinsic calibrations."""


@extrinsics_group.command("import")
@click.option("--session", required=True, metavar="UUID",
              help="mocap_sessions.id")
@click.option("--calib", required=True, metavar="TOML_PATH")
@click.option("--camera-instance", multiple=True, metavar="SPEC",
              help="cam1=<uuid> pairs or single UUID")
@click.option("--method", default="pose2sim", metavar="S")
@click.option("--shot", default=None, metavar="UUID",
              help="captures.id to link after import (sets extrinsic_calibration_id)")
@click.pass_obj
def extrinsics_import(
    obj: dict,
    session: str,
    calib: str,
    camera_instance: tuple[str, ...],
    method: str,
    shot: str | None,
) -> None:
    """Import extrinsic calibration from a Pose2Sim TOML file."""
    cam_inst = _parse_camera_instances(camera_instance)
    if cam_inst is None:
        raise click.UsageError("At least one --camera-instance is required.")

    registry = _open_registry(obj)
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session)
        result = import_extrinsics(
            conn,
            session_id,
            Path(calib),
            cam_inst,
            registry=registry,
            method=method or "pose2sim",
        )
        if shot:
            shot_id = _resolve(conn, "captures", shot)
            set_capture_extrinsics(conn, shot_id, result.extrinsic_calibration_id)
    except Exception as exc:  # noqa: BLE001
        conn.close()
        registry.close()
        raise click.ClickException(f"Error importing extrinsics: {exc}") from exc
    finally:
        conn.close()
        registry.close()

    click.echo(f"extrinsic_calibration_id: {result.extrinsic_calibration_id}")
    for cam_key, iid in result.camera_instance_ids.items():
        click.echo(f"  {cam_key}  instance={iid}")
    if result.skipped:
        click.echo(f"  skipped: {', '.join(sorted(result.skipped))}")


@extrinsics_group.command("list")
@click.option("--session", default=None, metavar="UUID",
              help="Filter by mocap_sessions.id")
@click.pass_obj
def extrinsics_list(obj: dict, session: str | None) -> None:
    """List extrinsic calibrations in a session database."""
    conn = _open_session_required(obj)
    try:
        q = "SELECT id, session_id, calibrated_at, method FROM extrinsic_calibrations"
        params: list = []
        if session:
            q += " WHERE session_id = ?"
            params.append(session)
        q += " ORDER BY calibrated_at"
        rows = conn.execute(q, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("(no extrinsic calibrations)")
        return

    print_table(
        [{"id": r[0], "session_id": r[1], "calibrated_at": r[2], "method": r[3] or ""} for r in rows],
        columns=["id", "session_id", "calibrated_at", "method"],
        json_mode=obj["json_mode"],
    )


# ---------------------------------------------------------------------------
# sync group
# ---------------------------------------------------------------------------


@click.group("sync")
def sync_group() -> None:
    """Manage sync configurations."""


@sync_group.command("import")
@click.option("--shot", required=True, metavar="UUID",
              help="captures.id")
@click.option("--sync-json", required=True, metavar="JSON_PATH")
@click.option("--camera-instance", multiple=True, metavar="SPEC")
@click.option("--notes", default="", metavar="S",
              help="Description of the sync method (e.g. 'LED detection', 'manual')")
@click.pass_obj
def sync_import(
    obj: dict,
    shot: str,
    sync_json: str,
    camera_instance: tuple[str, ...],
    notes: str,
) -> None:
    """Import camera sync anchors from a sync JSON file."""
    cam_inst = _parse_camera_instances(camera_instance)
    if cam_inst is None:
        raise click.UsageError("At least one --camera-instance is required.")

    conn = _open_session_required(obj)
    try:
        shot_id = _resolve(conn, "captures", shot)
        result = import_sync_json(
            conn,
            shot_id,
            Path(sync_json),
            cam_inst,
            notes=notes or "",
        )
    except Exception as exc:  # noqa: BLE001
        conn.close()
        raise click.ClickException(f"Error importing sync: {exc}") from exc
    finally:
        conn.close()

    click.echo(f"sync_config_id: {result.sync_config_id}")
    for cam_key, iid in result.camera_instance_ids.items():
        click.echo(f"  {cam_key}  instance={iid}")
    if result.skipped:
        click.echo(f"  skipped: {', '.join(sorted(result.skipped))}")


@sync_group.command("list")
@click.option("--shot", default=None, metavar="UUID",
              help="Filter by captures.id")
@click.pass_obj
def sync_list(obj: dict, shot: str | None) -> None:
    """List sync configs in a session database."""
    conn = _open_session_required(obj)
    try:
        q = "SELECT id, shot_id FROM sync_configs"
        params: list = []
        if shot:
            q += " WHERE shot_id = ?"
            params.append(shot)
        rows = conn.execute(q, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("(no sync configs)")
        return

    print_table(
        [{"id": r[0], "shot_id": r[1]} for r in rows],
        columns=["id", "shot_id"],
        json_mode=obj["json_mode"],
    )
