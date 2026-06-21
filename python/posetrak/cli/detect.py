"""detect.py — CLI commands for running and listing detection runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from posetrak.cli._output import fail, print_jsonl, print_record, print_table
from posetrak.db.db import open_session, resolve_id_prefix
from posetrak.detection.backends_yolo import YOLOv11Detector
from posetrak.detection.backends_rtmpose import RTMPoseEstimator
from posetrak.detection.pipeline import DetectionPipeline


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("detect")
def detect_group() -> None:
    """Run and manage person-detection pipeline runs."""


# ---------------------------------------------------------------------------
# detect run
# ---------------------------------------------------------------------------


@detect_group.command("run")
@click.option("--capture", required=True, help="Capture ID (prefix accepted).")
@click.option("--sync", required=True, help="Sync config ID (prefix accepted).")
@click.option("--start", type=float, required=True, help="Start time in seconds (global).")
@click.option("--end", type=float, required=True, help="End time in seconds (global).")
@click.option("--detector", default="yolo11x", show_default=True, help="Detector model name.")
@click.option(
    "--pose-model",
    default="rtmpose-l-133kp",
    show_default=True,
    help="Pose estimation model name.",
)
@click.option(
    "--conf",
    default=0.3,
    show_default=True,
    type=float,
    help="Detector confidence threshold.",
)
@click.pass_context
def cmd_run(
    ctx: click.Context,
    capture: str,
    sync: str,
    start: float,
    end: float,
    detector: str,
    pose_model: str,
    conf: float,
) -> None:
    """Run person detection and pose estimation for a capture time range.

    Results are written to the session DB.  The detection run ID is printed
    to stdout on completion.

    Example:

        posetrak -s session.db detect run --capture <id> --sync <id> \\
            --start 12 --end 105
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'detect run'.")

    try:
        session = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    # Resolve ID prefixes
    try:
        capture_id = resolve_id_prefix(session, "captures", capture)
    except ValueError as exc:
        fail(str(exc))

    try:
        sync_id = resolve_id_prefix(session, "sync_configs", sync)
    except ValueError as exc:
        fail(str(exc))

    click.echo(f"Capture:     {capture_id}", err=True)
    click.echo(f"Sync config: {sync_id}", err=True)
    click.echo(f"Time range:  {start:.2f} – {end:.2f} s", err=True)
    click.echo(f"Detector:    {detector}  (conf={conf})", err=True)
    click.echo(f"Pose model:  {pose_model}", err=True)
    click.echo("", err=True)

    det = YOLOv11Detector(
        model_name=f"{detector}.pt",
        device=None,  # auto-detect
        conf=conf,
    )
    est = RTMPoseEstimator(
        model_name=pose_model,
        device=None,  # auto-detect
    )

    def on_progress(done: int, total: int, cam_id: str) -> None:
        click.echo(f"\r  {cam_id}  {done}/{total} frames", nl=False, err=True)

    pipeline = DetectionPipeline(
        session=session,
        shot_id=capture_id,
        sync_config_id=sync_id,
        time_start_s=start,
        time_end_s=end,
        detector=det,
        estimator=est,
    )

    result = pipeline.run(on_progress=on_progress)
    click.echo("", err=True)  # newline after progress

    # Print the run ID to stdout (bare, machine-readable)
    click.echo(result.detection_run_id)


# ---------------------------------------------------------------------------
# detect list
# ---------------------------------------------------------------------------


@detect_group.command("list")
@click.option("--capture", default=None, help="Filter by capture ID (prefix accepted).")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSONL.")
@click.pass_context
def cmd_list(ctx: click.Context, capture: str | None, output_json: bool) -> None:
    """List detection runs in the session."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'detect list'.")

    try:
        session = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    capture_id: str | None = None
    if capture is not None:
        try:
            capture_id = resolve_id_prefix(session, "captures", capture)
        except ValueError as exc:
            fail(str(exc))

    rows = _list_detection_runs(session, capture_id=capture_id)

    if not rows:
        sys.exit(0)

    if output_json:
        print_jsonl(rows)
        return

    columns = ["id", "capture_id", "detector", "pose_model", "status", "created_at"]
    print_table(rows, columns, json_mode=output_json)


# ---------------------------------------------------------------------------
# detect show
# ---------------------------------------------------------------------------


@detect_group.command("show")
@click.argument("run_id")
@click.option("--json", "output_json", is_flag=True, default=False, help="Output as JSON.")
@click.pass_context
def cmd_show(ctx: click.Context, run_id: str, output_json: bool) -> None:
    """Show details of a single detection run."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'detect show'.")

    try:
        session = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    try:
        full_id = resolve_id_prefix(session, "detection_runs", run_id)
    except ValueError as exc:
        fail(str(exc))

    row = session.execute(
        "SELECT id, shot_id AS capture_id, sync_config_id, "
        "       time_start_s, time_end_s, "
        "       detector_model AS detector, detector_version, "
        "       pose_model, pose_version, "
        "       detector_conf, pose_conf_threshold, "
        "       pose_input_width, pose_input_height, "
        "       status, created_at, completed_at "
        "FROM detection_runs WHERE id = ?",
        (full_id,),
    ).fetchone()

    if row is None:
        fail(f"Detection run '{full_id}' not found.")

    record = dict(row)

    if output_json:
        print(json.dumps(record))
        return

    print_record(record)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _list_detection_runs(
    session,
    capture_id: str | None = None,
) -> list[dict]:
    """Query detection_runs, optionally filtered by capture.

    Returns rows with keys: id, capture_id, detector, pose_model,
    status, created_at.
    """
    query = (
        "SELECT id, shot_id AS capture_id, "
        "       detector_model AS detector, pose_model, "
        "       status, created_at "
        "FROM detection_runs"
    )
    params: list = []
    if capture_id is not None:
        query += " WHERE shot_id = ?"
        params.append(capture_id)
    query += " ORDER BY created_at DESC"

    rows = session.execute(query, params).fetchall()
    return [dict(r) for r in rows]
