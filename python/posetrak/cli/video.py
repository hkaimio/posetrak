# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""video.py — Commands for managing capture video file paths."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import click

from posetrak.cli._output import fail, print_table
from posetrak.db.db import open_session, resolve_id_prefix


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("video")
def video_group() -> None:
    """Manage capture video file paths."""


def _open_session_required(ctx: click.Context) -> sqlite3.Connection:
    path = ctx.obj.get("session")
    if not path:
        fail("--session / POSETRAK_SESSION_DB is required.")
    try:
        return open_session(Path(path))
    except Exception as exc:
        fail(str(exc))


# ---------------------------------------------------------------------------
# video list
# ---------------------------------------------------------------------------


@video_group.command("list")
@click.option("--capture", "capture_id", default=None, metavar="ID",
              help="Filter to one capture (prefix accepted).")
@click.pass_context
def cmd_list(ctx: click.Context, capture_id: str | None) -> None:
    """List video files across all captures (or one capture).

    The 'exists' column checks whether the file is reachable from the current
    machine — useful for spotting stale paths after moving a database.

    Example:

        posetrak -s session.db video list
        posetrak -s session.db video list --capture 13af67f5
    """
    conn = _open_session_required(ctx)
    json_mode: bool = ctx.obj.get("json_mode", False)

    if capture_id is not None:
        try:
            capture_id = resolve_id_prefix(conn, "captures", capture_id)
        except ValueError as exc:
            conn.close()
            fail(str(exc))

    try:
        q = """
            SELECT
                cv.id,
                c.label      AS capture_label,
                ci.label     AS camera_label,
                cv.file_path,
                cv.actual_fps
            FROM capture_videos cv
            JOIN captures        c  ON c.id  = cv.shot_id
            JOIN camera_instances ci ON ci.id = cv.camera_instance_id
        """
        params: list = []
        if capture_id:
            q += " WHERE cv.shot_id = ?"
            params.append(capture_id)
        q += " ORDER BY c.capture_number, ci.label"
        rows = conn.execute(q, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        fail(str(exc))
    finally:
        conn.close()

    if not rows:
        click.echo("(no videos)")
        return

    records = []
    for r in rows:
        fp = r["file_path"]
        records.append({
            "id":            r["id"],
            "capture":       r["capture_label"] or "",
            "camera":        r["camera_label"] or "",
            "file_path":     fp,
            "fps":           str(r["actual_fps"]),
            "exists":        "yes" if Path(fp).exists() else "NO",
        })

    print_table(
        records,
        columns=["id", "capture", "camera", "file_path", "fps", "exists"],
        json_mode=json_mode,
    )


# ---------------------------------------------------------------------------
# video locate
# ---------------------------------------------------------------------------


@video_group.command("locate")
@click.argument("video_id", metavar="ID")
@click.argument("new_path",  metavar="NEW_PATH")
@click.pass_context
def cmd_locate(ctx: click.Context, video_id: str, new_path: str) -> None:
    """Set the file path for a single capture video.

    Useful when a video file has been renamed or moved individually.
    For bulk path prefix changes use 'video relocate'.

    Example:

        posetrak -s session.db video locate a3f1bc00 D:\\mocap\\cam1.mp4
    """
    conn = _open_session_required(ctx)

    try:
        resolved = resolve_id_prefix(conn, "capture_videos", video_id)
    except ValueError as exc:
        conn.close()
        fail(str(exc))

    row = conn.execute(
        "SELECT file_path FROM capture_videos WHERE id = ?", (resolved,)
    ).fetchone()
    if row is None:
        conn.close()
        fail(f"Video not found: {video_id}")

    old_path = row[0]
    if old_path == new_path:
        conn.close()
        click.echo("Path unchanged.")
        return

    conn.execute(
        "UPDATE capture_videos SET file_path = ? WHERE id = ?",
        (new_path, resolved),
    )
    conn.commit()
    conn.close()

    click.echo(f"Updated {resolved[:8]}…")
    click.echo(f"  was: {old_path}")
    click.echo(f"  now: {new_path}")


# ---------------------------------------------------------------------------
# video relocate
# ---------------------------------------------------------------------------


@video_group.command("relocate")
@click.option("--from", "old_prefix", required=True, metavar="OLD_PREFIX",
              help="Path prefix to replace (e.g. /mnt/d/mocap).")
@click.option("--to",   "new_prefix", required=True, metavar="NEW_PREFIX",
              help="Replacement prefix (e.g. D:\\mocap).")
@click.option("--capture", "capture_id", default=None, metavar="ID",
              help="Limit to one capture (prefix accepted).")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would change without writing.")
@click.pass_context
def cmd_relocate(
    ctx: click.Context,
    old_prefix: str,
    new_prefix: str,
    capture_id: str | None,
    dry_run: bool,
) -> None:
    """Bulk-replace a path prefix in all capture video records.

    The most common use case is updating stale paths after moving a session DB
    to a different machine or remounting a drive at a new location.

    The replacement is an exact string prefix match on file_path, so both
    forward and backward slash variants should be specified explicitly if
    needed.

    Examples:

        # Moved from WSL to Windows
        posetrak -s session.db video relocate \\
            --from /mnt/d/mocap --to D:\\mocap --dry-run

        # Change mount point
        posetrak -s session.db video relocate \\
            --from /old/mount --to /new/mount
    """
    conn = _open_session_required(ctx)
    json_mode: bool = ctx.obj.get("json_mode", False)

    if capture_id is not None:
        try:
            capture_id = resolve_id_prefix(conn, "captures", capture_id)
        except ValueError as exc:
            conn.close()
            fail(str(exc))

    try:
        q = "SELECT cv.id, cv.file_path FROM capture_videos cv"
        params: list = []
        if capture_id:
            q += " WHERE cv.shot_id = ?"
            params.append(capture_id)
        rows = conn.execute(q, params).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        fail(str(exc))

    # Normalise: strip trailing separators so the original path's separator
    # is always preserved in the suffix, regardless of whether the user
    # included a trailing slash on --from or --to.
    old_norm = old_prefix.rstrip("/\\")
    new_norm = new_prefix.rstrip("/\\")

    def _matches(fp: str) -> bool:
        return (
            fp == old_norm
            or fp.startswith(old_norm + "/")
            or fp.startswith(old_norm + "\\")
        )

    to_update = [(r["id"], r["file_path"]) for r in rows if _matches(r["file_path"])]

    if not to_update:
        conn.close()
        click.echo(f"No video paths start with {old_prefix!r}.")
        return

    records = []
    for vid_id, old_fp in to_update:
        suffix = old_fp[len(old_norm):]   # retains leading sep from original path
        new_fp = new_norm + suffix
        records.append({"id": vid_id, "old_path": old_fp, "new_path": new_fp})

    if dry_run:
        print_table(
            [{"id": r["id"], "old_path": r["old_path"], "new_path": r["new_path"]} for r in records],
            columns=["id", "old_path", "new_path"],
            json_mode=json_mode,
        )
        click.echo(f"\n{len(records)} path(s) would be updated.", err=True)
        conn.close()
        return

    for r in records:
        conn.execute(
            "UPDATE capture_videos SET file_path = ? WHERE id = ?",
            (r["new_path"], r["id"]),
        )
    conn.commit()
    conn.close()

    click.echo(f"Updated {len(records)} video path(s).")
    for r in records:
        click.echo(f"  {r['id'][:8]}…  {r['old_path']}")
        click.echo(f"           -> {r['new_path']}")


# ---------------------------------------------------------------------------
# video proxy — scrub proxy cache (sidecar database beside the session)
# ---------------------------------------------------------------------------


@video_group.group("proxy")
def proxy_group() -> None:
    """Manage the scrub proxy: small images of every Nth video frame that the
    viewer shows while dragging the time slider. Stored in <session>.cache.db,
    which is safe to delete."""


def _open_proxy(ctx: click.Context):
    from posetrak.detection.proxy_cache import ProxyCache, sidecar_path

    path = ctx.obj.get("session")
    if not path:
        fail("--session / POSETRAK_SESSION_DB is required.")
    return ProxyCache(sidecar_path(path))


@proxy_group.command("build")
@click.option("--capture", "capture_id", default=None, metavar="ID",
              help="Only this capture's videos (prefix accepted).")
@click.option("--step", default=None, type=int, metavar="N",
              help="Keep every Nth frame (default: 8).")
@click.option("--width", default=None, type=int, metavar="PX",
              help="Image width in pixels (default: 640).")
@click.pass_context
def cmd_proxy_build(ctx: click.Context, capture_id: str | None, step: int | None,
                    width: int | None) -> None:
    """Build the proxy for whole videos (the viewer builds the range it shows
    by itself; use this to prepare a capture ahead of time).

    Example:

        posetrak -s session.db video proxy build --capture 13af67f5
    """
    from posetrak.detection.proxy_cache import PROXY_STEP, PROXY_WIDTH

    conn = _open_session_required(ctx)
    try:
        if capture_id is not None:
            capture_id = resolve_id_prefix(conn, "captures", capture_id)
        q = "SELECT id, file_path FROM capture_videos"
        rows = conn.execute(q + (" WHERE shot_id = ?" if capture_id else ""),
                            (capture_id,) if capture_id else ()).fetchall()
    except (ValueError, sqlite3.Error) as exc:
        fail(str(exc))
    finally:
        conn.close()
    cache = _open_proxy(ctx)
    for row in rows:
        if not Path(row["file_path"]).exists():
            click.echo(f"{row['id'][:8]}  missing file, skipped: {row['file_path']}")
            continue
        n = cache.build(row["id"], row["file_path"], step=step or PROXY_STEP,
                        width=width or PROXY_WIDTH)
        click.echo(f"{row['id'][:8]}  {n} frames")
    cache.close()


@proxy_group.command("info")
@click.pass_context
def cmd_proxy_info(ctx: click.Context) -> None:
    """Show how many proxy frames are stored per video, and their size."""
    cache = _open_proxy(ctx)
    rows = cache.stats()
    cache.close()
    if not rows:
        click.echo("(empty)")
    for svid, frames, size in rows:
        click.echo(f"{svid[:8]}  {frames} frames  {size / 1e6:.1f} MB")


@proxy_group.command("clear")
@click.pass_context
def cmd_proxy_clear(ctx: click.Context) -> None:
    """Delete every proxy frame (the videos and session are not touched)."""
    cache = _open_proxy(ctx)
    cache.clear()
    cache.close()
    click.echo("cleared")
