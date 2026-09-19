# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""detect.py — CLI commands for running and listing detection runs.

``detect run`` drives three kinds of run, chosen with ``--type``: person pose
detection (the default), coded-marker (ArUco) detection, and reflective-dot
detection. A marker run is stored as one ``detection_runs`` row with
``detector_type='aruco'`` whether or not it also carries dots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from click.core import ParameterSource

from posetrak.cli._output import fail, print_jsonl, print_record, print_table
from posetrak.db.db import open_session, resolve_id_prefix
from posetrak.db.manage_capture_object import list_capture_objects
from posetrak.detection.backends_rtmdet import YOLOXDetector
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


# Options that only make sense for one kind of run. ``detect run`` rejects them for the others
# rather than ignoring them silently.
_POSE_OPTIONS = ("detector", "pose_model", "conf")
_MARKER_OPTIONS = ("object_ref", "trial", "frame_step", "min_marker_perimeter_rate", "parallel", "max_workers")
_CODED_OPTIONS = ("marker_ids", "dictionary")
_DOT_OPTIONS = (
    "dot_threshold", "dot_threshold_by_camera", "dot_background_mode", "dot_bg_subtract",
    "dot_blacklist_frac", "dot_blacklist_frac_by_camera", "dot_blacklist_radius_px",
    "dot_max_saturation", "dot_max_saturation_by_camera", "dot_bg_sample_count",
)


@detect_group.command("run")
@click.option("--capture", required=True, help="Capture ID (prefix accepted).")
@click.option("--sync", required=True, help="Sync config ID (prefix accepted).")
@click.option("--start", type=float, required=True, help="Start time in seconds (global).")
@click.option("--end", type=float, required=True, help="End time in seconds (global).")
@click.option(
    "--type", "run_type", type=click.Choice(["pose", "aruco", "dots"]), default="pose", show_default=True,
    help="pose: person detection and pose estimation. aruco: coded markers, optionally with "
         "reflective dots on the cameras named by --dots-camera. dots: reflective dots only.",
)
@click.option("--detector", default="yolox-x", show_default=True, help="Person detector model (--type pose).")
@click.option("--pose-model", default="rtmpose-l-133kp", show_default=True,
              help="Pose estimation model (--type pose).")
@click.option("--conf", default=0.3, show_default=True, type=float,
              help="Person detector confidence threshold (--type pose).")
@click.option("--object", "object_ref", default=None,
              help="Bind a marker run to this capture object (name or ID prefix). Without it the "
                   "run is not bound to any object (--type aruco then needs --marker-ids).")
@click.option("--trial", default=None, help="Trial ID (prefix accepted) to record on a marker run.")
@click.option("--marker-ids", default=None,
              help="Comma-separated ids of every coded marker the scene may show, for an aruco run "
                   "that is not bound to an object. Fixes the corner-slot layout of the run: an id "
                   "left out is dropped for the whole run.")
@click.option("--dictionary", default="DICT_4X4_50", show_default=True,
              help="ArUco dictionary for an unbound aruco run. A bound run takes it from the marker body.")
@click.option("--min-marker-perimeter-rate", type=float, default=None,
              help="Smallest marker perimeter, as a fraction of the image, that is still detected.")
@click.option("--frame-step", type=int, default=1, show_default=True, help="Process every Nth frame.")
@click.option("--dots-camera", "dots_camera", multiple=True,
              help="Camera label to run reflective-dot detection on; repeat for several. Required "
                   "for --type dots. Only cameras with a ring light or reflective markers in view "
                   "are worth naming.")
@click.option("--dot-threshold", type=int, default=235, show_default=True,
              help="Brightness threshold: raw pixel value, or the background residual with "
                   "--dot-bg-subtract (use a much lower value then).")
@click.option("--dot-threshold-by-camera", multiple=True, metavar="LABEL=THRESHOLD",
              help="Per-camera --dot-threshold; cameras differ in the peak brightness of a real marker.")
@click.option("--dot-background-mode", type=click.Choice(["subtract", "blacklist"]), default="subtract",
              show_default=True,
              help="subtract: threshold the background-subtracted residual. blacklist: threshold raw "
                   "brightness and use the background only to veto spots that are bright with no "
                   "subject present (avoids fusing a marker into the subject's own limb).")
@click.option("--dot-bg-subtract", is_flag=True, default=False,
              help="Build a per-camera background model for 'subtract' mode ('blacklist' always builds one).")
@click.option("--dot-blacklist-frac", type=float, default=0.7, show_default=True,
              help="Glare veto looseness in 'blacklist' mode; higher vetoes fewer real markers.")
@click.option("--dot-blacklist-frac-by-camera", multiple=True, metavar="LABEL=FRAC",
              help="Per-camera --dot-blacklist-frac.")
@click.option("--dot-blacklist-radius-px", type=int, default=6, show_default=True,
              help="Radius of the glare veto in 'blacklist' mode.")
@click.option("--dot-max-saturation", type=float, default=255.0, show_default=True,
              help="Reject a candidate whose mean HSV saturation exceeds this (skin is chromatic, "
                   "dots are near-neutral); 255 turns the filter off.")
@click.option("--dot-max-saturation-by-camera", multiple=True, metavar="LABEL=MAXSAT",
              help="Per-camera --dot-max-saturation; some cameras render a marker colour-tinted.")
@click.option("--dot-bg-sample-count", type=int, default=40, show_default=True,
              help="Frames sampled to build each camera's background model.")
@click.option("--parallel", is_flag=True, default=False,
              help="Marker runs: process cameras concurrently, one process each. No cancellation "
                   "and per-camera rather than per-frame progress.")
@click.option("--max-workers", type=int, default=None, help="With --parallel: worker process cap.")
@click.pass_context
def cmd_run(
    ctx: click.Context,
    capture: str,
    sync: str,
    start: float,
    end: float,
    run_type: str,
    detector: str,
    pose_model: str,
    conf: float,
    object_ref: str | None,
    trial: str | None,
    marker_ids: str | None,
    dictionary: str,
    min_marker_perimeter_rate: float | None,
    frame_step: int,
    dots_camera: tuple[str, ...],
    dot_threshold: int,
    dot_threshold_by_camera: tuple[str, ...],
    dot_background_mode: str,
    dot_bg_subtract: bool,
    dot_blacklist_frac: float,
    dot_blacklist_frac_by_camera: tuple[str, ...],
    dot_blacklist_radius_px: int,
    dot_max_saturation: float,
    dot_max_saturation_by_camera: tuple[str, ...],
    dot_bg_sample_count: int,
    parallel: bool,
    max_workers: int | None,
) -> None:
    """Run detection for a capture time range.

    Results are written to the session DB.  The detection run ID is printed
    to stdout on completion.

    Person pose detection (default):

        posetrak -s session.db detect run --capture <id> --sync <id> \\
            --start 12 --end 105

    Coded markers of a capture object, plus reflective dots on two cameras:

        posetrak -s session.db detect run --type aruco --object sword \\
            --capture <id> --sync <id> --start 34 --end 100 \\
            --dots-camera gopro13_01 --dots-camera gopro13_02

    Reflective dots only, bound to no object (props and persons wearing dots):

        posetrak -s session.db detect run --type dots --capture <id> --sync <id> \\
            --start 34 --end 100 --dots-camera gopro13_01 --dot-background-mode blacklist
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'detect run'.")

    try:
        session = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    try:
        capture_id = resolve_id_prefix(session, "captures", capture)
        sync_id = resolve_id_prefix(session, "sync_configs", sync)
    except ValueError as exc:
        fail(str(exc))

    flags = {p.name: p.opts[0] for p in ctx.command.params}

    def reject(names: tuple[str, ...], reason: str) -> None:
        given = [
            flags[n] for n in names
            if ctx.get_parameter_source(n) in (ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT)
        ]
        if given:
            fail(f"{', '.join(given)}: {reason}")

    if run_type == "pose":
        reject(_MARKER_OPTIONS + _CODED_OPTIONS + _DOT_OPTIONS + ("dots_camera",),
               "only for --type aruco or dots.")
        _run_pose_detection(session, capture_id, sync_id, start, end, detector, pose_model, conf)
        return

    reject(_POSE_OPTIONS, "only for --type pose.")
    if run_type == "dots":
        reject(_CODED_OPTIONS, "only for --type aruco.")
    elif object_ref is not None:
        reject(_CODED_OPTIONS, "taken from the marker body when --object is given.")
    if not dots_camera:
        if run_type == "dots":
            fail("--type dots needs at least one --dots-camera.")
        reject(_DOT_OPTIONS, "only used with --dots-camera.")

    try:
        object_id = _resolve_capture_object(session, capture_id, object_ref) if object_ref else None
        trial_id = resolve_id_prefix(session, "trials", trial) if trial else None
        ids = [m.strip() for m in marker_ids.split(",") if m.strip()] if marker_ids else []
        if run_type == "aruco" and object_id is None and not ids:
            raise ValueError("--type aruco needs --object, or --marker-ids for a run bound to no object.")
        dot_cameras = {_camera_id(session, label, "--dots-camera") for label in dots_camera}
        settings = dict(
            min_marker_perimeter_rate=min_marker_perimeter_rate,
            frame_step=frame_step,
            trial_id=trial_id,
            detect_dots_for_cameras=dot_cameras,
            dot_bg_subtract=dot_bg_subtract,
            dot_threshold=dot_threshold,
            dot_threshold_by_camera=_per_camera(session, dot_threshold_by_camera, "--dot-threshold-by-camera", int),
            dot_background_mode=dot_background_mode,
            dot_blacklist_frac=dot_blacklist_frac,
            dot_blacklist_frac_by_camera=_per_camera(
                session, dot_blacklist_frac_by_camera, "--dot-blacklist-frac-by-camera", float),
            dot_blacklist_radius_px=dot_blacklist_radius_px,
            dot_max_saturation=dot_max_saturation,
            dot_max_saturation_by_camera=_per_camera(
                session, dot_max_saturation_by_camera, "--dot-max-saturation-by-camera", float),
            dot_bg_sample_count=dot_bg_sample_count,
            detect_coded=run_type == "aruco",
        )
        from posetrak.detection.marker_pipeline import MarkerDetectionPipeline, load_pipeline_for_capture_object

        if object_id is not None:
            pipeline = load_pipeline_for_capture_object(
                session, object_id, sync_id, start, end, **settings,
            )
        else:
            pipeline = MarkerDetectionPipeline(
                session, shot_id=capture_id, sync_config_id=sync_id, time_start_s=start, time_end_s=end,
                marker_ids=ids, dictionary=dictionary, **settings,
            )
        missing = dot_cameras - {c.camera_instance_id for c in pipeline.cameras}
        if missing:
            raise ValueError(
                f"--dots-camera: no video in this capture and time range for {sorted(m[:8] for m in missing)}."
            )
    except ValueError as exc:
        fail(str(exc))

    click.echo(f"Capture:     {capture_id}", err=True)
    click.echo(f"Sync config: {sync_id}", err=True)
    click.echo(f"Time range:  {start:.2f} – {end:.2f} s", err=True)
    click.echo(f"Run type:    {run_type}" + (f"  (object {object_ref})" if object_ref else ""), err=True)
    click.echo(f"Cameras:     {[c.label or c.camera_instance_id[:8] for c in pipeline.cameras]}", err=True)
    click.echo("", err=True)

    def on_progress(done: int, total: int, cam_label: str) -> None:
        click.echo(f"\r  {cam_label}  {done}/{total} frames", nl=False, err=True)

    def on_camera_done(n_done: int, n_total: int) -> None:
        click.echo(f"\nCamera {n_done}/{n_total} done", err=True)

    if parallel:
        result = pipeline.run_parallel(max_workers=max_workers, on_camera_done=on_camera_done)
    else:
        result = pipeline.run(on_progress=on_progress, on_camera_done=on_camera_done)
    click.echo("", err=True)

    click.echo(result.detection_run_id)
    if result.status != "complete":
        fail(f"Detection run {result.detection_run_id} ended with status '{result.status}'.")


def _run_pose_detection(
    session, capture_id: str, sync_id: str, start: float, end: float,
    detector: str, pose_model: str, conf: float,
) -> None:
    """Run person detection and pose estimation; print the run ID to stdout."""
    click.echo(f"Capture:     {capture_id}", err=True)
    click.echo(f"Sync config: {sync_id}", err=True)
    click.echo(f"Time range:  {start:.2f} – {end:.2f} s", err=True)
    click.echo(f"Detector:    {detector}  (conf={conf})", err=True)
    click.echo(f"Pose model:  {pose_model}", err=True)
    click.echo("", err=True)

    det = YOLOXDetector(
        model_name=detector,
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


@detect_group.command("import-2d")
@click.option("--capture", required=True, help="Capture ID (prefix accepted).")
@click.option("--sync", required=True, help="Sync config ID (prefix accepted).")
@click.option("--trial", default=None, help="Trial ID (prefix accepted) to record on the run.")
@click.option("--object", "object_ref", default=None,
              help="Bind the run to this capture object (name or ID prefix), so that "
                   "`sequence finalise-object` can make its sequence.")
@click.option(
    "--camera", "tracks", nargs=2, multiple=True, required=True, metavar="LABEL CSV",
    type=(str, click.Path(exists=True, dir_okay=False, path_type=Path)),
    help="A track file for the camera with this label. Repeat for several cameras, and for "
         "several tracks of one camera.",
)
@click.option("--source", default="external", show_default=True,
              help="The tool the tracks come from, recorded on the run (e.g. blender).")
@click.pass_context
def cmd_import_2d(
    ctx: click.Context,
    capture: str,
    sync: str,
    trial: str | None,
    object_ref: str | None,
    tracks: tuple[tuple[str, Path], ...],
    source: str,
) -> None:
    """Import 2D point tracks made in another tool as a detection run.

    Each CSV file is one track of one camera, with a header row and the columns
    video_frame, pixel_x and pixel_y (raw image pixels, origin top left): the
    output of python/tools/blender/blender_export_2d_tracks.py imports as it is.
    The points become anonymous dot candidates, so the run works wherever a dots
    run does. The run ID is printed to stdout.

    Example:

        posetrak -s session.db detect import-2d --capture <id> --sync <id> \
            --object ball --source blender \
            --camera gopro13_01 ball-gopro13_01.csv --camera gopro13_01 ball-gopro13_01-b.csv
    """
    from posetrak.detection.external_import import import_external_2d

    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'detect import-2d'.")
    try:
        session = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))
    try:
        capture_id = resolve_id_prefix(session, "captures", capture)
        sync_id = resolve_id_prefix(session, "sync_configs", sync)
        trial_id = resolve_id_prefix(session, "trials", trial) if trial else None
        object_id = _resolve_capture_object(session, capture_id, object_ref) if object_ref else None
        result = import_external_2d(
            session, capture_id, sync_id, list(tracks),
            trial_id=trial_id, capture_object_id=object_id, source=source,
        )
    except ValueError as exc:
        fail(str(exc))

    for label, n in sorted(result.rows_by_camera.items()):
        click.echo(f"  {label}: {n} frames", err=True)
    if result.skipped_no_timestamp:
        click.echo(f"Skipped {result.skipped_no_timestamp} points on frames without a sync timestamp.", err=True)
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


def _resolve_capture_object(session, capture_id: str, ref: str) -> str:
    """Resolve a capture object of *capture_id* by exact name, else by ID prefix."""
    objects = list_capture_objects(session, capture_id)
    matches = [o for o in objects if o["name"] == ref] or [o for o in objects if o["id"].startswith(ref)]
    if len(matches) == 1:
        return matches[0]["id"]
    names = ", ".join(sorted(o["name"] for o in objects)) or "none"
    if not matches:
        raise ValueError(f"--object: no capture object {ref!r} in this capture (objects: {names}).")
    raise ValueError(f"--object: {ref!r} matches several capture objects (objects: {names}).")


def _camera_id(session, label: str, flag: str) -> str:
    """Resolve a camera_instances label to its ID."""
    rows = session.execute("SELECT id FROM camera_instances WHERE label = ?", (label,)).fetchall()
    if len(rows) != 1:
        raise ValueError(f"{flag}: {'no' if not rows else 'several'} camera_instances with label {label!r}.")
    return rows[0]["id"]


def _per_camera(session, entries: tuple[str, ...], flag: str, cast) -> dict[str, object]:
    """Parse ``LABEL=VALUE`` entries into ``{camera_instance_id: cast(VALUE)}``."""
    out: dict[str, object] = {}
    for entry in entries:
        label, sep, value = entry.partition("=")
        if not sep:
            raise ValueError(f"{flag}: expected LABEL=VALUE, got {entry!r}.")
        camera_id = _camera_id(session, label, flag)
        try:
            out[camera_id] = cast(value)
        except ValueError:
            raise ValueError(f"{flag}: bad value in {entry!r}.") from None
    return out
