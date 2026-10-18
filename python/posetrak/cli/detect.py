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
from posetrak.detection.pipeline import DetectionPipeline
from posetrak.detection.pose_model_choice import POSE_MODELS, create_estimator, resolve_pose_model


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
# rather than ignoring them silently. --trial is shared by pose and marker runs, so it is not
# in either tuple.
_POSE_OPTIONS = ("detector", "pose_model", "conf", "segmentation", "from_run", "refine_hands", "accept_licences",
                 "decode")
_MARKER_OPTIONS = ("object_ref", "frame_step", "min_marker_perimeter_rate", "parallel", "max_workers")
_CODED_OPTIONS = ("marker_ids", "dictionary")
_DOT_OPTIONS = (
    "dot_threshold", "dot_threshold_by_camera", "dot_background_mode", "dot_bg_subtract",
    "dot_blacklist_frac", "dot_blacklist_frac_by_camera", "dot_blacklist_radius_px",
    "dot_max_saturation", "dot_max_saturation_by_camera", "dot_bg_sample_count",
)


@detect_group.command("run")
@click.option("--capture", required=True, help="Capture ID (prefix accepted).")
@click.option("--sync", required=True, help="Sync config ID (prefix accepted).")
@click.option("--start", type=float, default=None,
              help="Start time in seconds (global). Defaults to --trial's own start.")
@click.option("--end", type=float, default=None,
              help="End time in seconds (global). Defaults to --trial's own end.")
@click.option(
    "--type", "run_type", type=click.Choice(["pose", "aruco", "dots"]), default="pose", show_default=True,
    help="pose: person detection and pose estimation. aruco: coded markers, optionally with "
         "reflective dots on the cameras named by --dots-camera. dots: reflective dots only.",
)
@click.option("--detector", default="yolox-x", show_default=True, help="Person detector model (--type pose).")
@click.option("--pose-model", type=click.Choice(POSE_MODELS), default=POSE_MODELS[0], show_default=True,
              help="Pose estimation model (--type pose). The SOMA-77 weights are downloaded on first use "
                   "after you accept their licences; without them the run uses rtmpose-l-133kp and says why.")
@click.option("--conf", default=0.3, show_default=True, type=float,
              help="Person detector confidence threshold (--type pose).")
@click.option("--object", "object_ref", default=None,
              help="Bind a marker run to this capture object (name or ID prefix). Without it the "
                   "run is not bound to any object (--type aruco then needs --marker-ids).")
@click.option("--trial", default=None,
              help="Trial ID (prefix accepted) to record on the run. Also supplies --start/--end's "
                   "default when they are omitted.")
@click.option("--segmentation", default=None,
              help="Seg quality run ID (prefix accepted) to source person bboxes from instead of "
                   "running the YOLO detector (--type pose only). Masks give tighter crops and the "
                   "run is finalised automatically (segmentation labels are already stable "
                   "per-person identities, unlike YOLO tracks). See "
                   "docs/roadmap/features/segmentation-reuse/segmentation-reuse-design.md.")
@click.option("--from-run", "from_run", default=None,
              help="Detection run ID (prefix accepted) whose person boxes and track ids are reused instead of "
                   "running the YOLO detector (--type pose only): a new pose model on boxes already found. "
                   "The source run's track assignments are copied and the run finalised.")
@click.option("--refine-hands/--no-refine-hands", default=None,
              help="Re-detect each tracked wrist's hand in a tight crop on the frame already decoded and add "
                   "refined finger keypoints, for every pose model that names a wrist and an elbow. "
                   "Default: on with --segmentation for COCO-133 models, off otherwise (and always off for SOMA-77).")
@click.option("--accept-licences", "accept_licences", is_flag=True, default=False,
              help="Accept the NVIDIA Open Model License and the DINOv3 License for the SOMA-77 weights without "
                   "being asked (their texts are still named). Needed to download the weights without a terminal.")
@click.option("--decode", type=click.Choice(["thread", "process"]), default=None,
              help="Decode the camera videos on threads or in child processes (default: the pipeline's choice).")
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
    start: float | None,
    end: float | None,
    run_type: str,
    detector: str,
    pose_model: str,
    conf: float,
    object_ref: str | None,
    trial: str | None,
    segmentation: str | None,
    from_run: str | None,
    refine_hands: bool | None,
    accept_licences: bool,
    decode: str | None,
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

    Person pose sourced from an existing Cutie segmentation instead of YOLO, tied to a trial
    (--start/--end default to the trial's own range):

        posetrak -s session.db detect run --trial <id> --segmentation <seg_run_id> \\
            --capture <id> --sync <id>
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
        trial_id = resolve_id_prefix(session, "trials", trial) if trial else None
    except ValueError as exc:
        fail(str(exc))

    if trial_id is not None:
        trial_row = session.execute(
            "SELECT time_start_s, time_end_s FROM trials WHERE id = ?", (trial_id,)
        ).fetchone()
        if start is None:
            start = trial_row["time_start_s"]
        if end is None:
            end = trial_row["time_end_s"]
    if start is None or end is None:
        fail("--start/--end are required unless --trial supplies them.")

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
        if segmentation is not None and from_run is not None:
            fail("--segmentation and --from-run are alternative box sources; give one.")
        if segmentation is not None:
            reject(("detector", "conf"), "not used when --segmentation sources bboxes instead of YOLO.")
        if from_run is not None:
            reject(("detector", "conf"), "not used when --from-run reuses the boxes of a run.")
        _run_pose_detection(
            session, capture_id, sync_id, start, end, detector, pose_model, conf,
            trial_id=trial_id, segmentation=segmentation, from_run=from_run, refine_hands=refine_hands,
            accept_licences=accept_licences, decode=decode,
        )
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


def _licence_prompt(assume_yes: bool):
    """The callback that shows the licence texts of the SOMA-77 weights and asks for acceptance."""

    def accept(texts) -> bool:
        for licence, text in texts:
            click.echo(f"\n=== {licence.name}  ({licence.url}) ===", err=True)
            if assume_yes:
                click.echo("(text not repeated: --accept-licences)", err=True)
            else:
                click.echo_via_pager(text)
        if assume_yes:
            return True
        if not sys.stdin.isatty():
            click.echo("Not a terminal: pass --accept-licences to accept the licences above.", err=True)
            return False
        return click.confirm("Accept both licences and download the SOMA-77 weights (about 3.4 GB)?", err=True)

    return accept


def _run_pose_detection(
    session, capture_id: str, sync_id: str, start: float, end: float,
    detector: str, pose_model: str, conf: float,
    trial_id: str | None, segmentation: str | None, from_run: str | None, refine_hands: bool | None,
    accept_licences: bool, decode: str | None,
) -> None:
    """Run person detection and pose estimation; print the run ID to stdout."""
    click.echo(f"Capture:     {capture_id}", err=True)
    click.echo(f"Sync config: {sync_id}", err=True)
    click.echo(f"Time range:  {start:.2f} – {end:.2f} s", err=True)

    model, reason = resolve_pose_model(
        pose_model, accept=_licence_prompt(accept_licences),
        progress=lambda message: click.echo(message, err=True),
    )
    click.echo(f"Pose model:  {model}", err=True)
    if reason:
        click.echo(f"             not {pose_model}: {reason}", err=True)

    persons_ordered = source_run_id = None
    try:
        if segmentation is not None:
            from posetrak.db.manage_person import persons_ordered_for_seg_run
            from posetrak.detection.box_sources import MaskBoxSource

            seg_run_id = resolve_id_prefix(session, "seg_quality_runs", segmentation)
            persons_ordered = persons_ordered_for_seg_run(session, seg_run_id)
            box_source = MaskBoxSource(session, seg_run_id, persons_ordered)
            click.echo(f"Bbox source: segmentation {seg_run_id}", err=True)
        elif from_run is not None:
            from posetrak.detection.box_sources import RunBoxSource

            source_run_id = resolve_id_prefix(session, "detection_runs", from_run)
            box_source = RunBoxSource(session, source_run_id)
            click.echo(f"Bbox source: detection run {source_run_id}", err=True)
        else:
            from posetrak.detection.box_sources import YoloxBoxSource

            box_source = YoloxBoxSource(YOLOXDetector(model_name=detector, device=None, conf=conf))
            click.echo(f"Detector:    {detector}  (conf={conf})", err=True)
    except ValueError as exc:
        fail(str(exc))
    click.echo("", err=True)

    estimator = create_estimator(model, device=None)
    if refine_hands is None:
        # Validated on COCO-133 with mask boxes; SOMA-77 hands were not improved by it (see
        # docs/roadmap/features/hand-detection-refinement/hand-refinement-without-segmentation-design.md).
        refine_hands = segmentation is not None and "soma" not in model
    hand_pass = None
    if refine_hands:
        from posetrak.detection.hand_refinement import HandPass

        if HandPass.supports(model):
            hand_pass = HandPass(pose_model=model)
        else:
            click.echo(f"Hand refinement skipped: {model} names no wrist and elbow.", err=True)

    def on_progress(done: int, total: int, cam_id: str) -> None:
        click.echo(f"\r  {cam_id}  {done}/{total} frames", nl=False, err=True)

    pipeline = DetectionPipeline(
        session=session,
        shot_id=capture_id,
        sync_config_id=sync_id,
        time_start_s=start,
        time_end_s=end,
        box_source=box_source,
        estimator=estimator,
        hand_pass=hand_pass,
        decode_mode=decode,
        trial_id=trial_id,
    )
    if not pipeline.cameras:
        fail("No cameras with sync data for this time range.")

    result = pipeline.run(on_progress=on_progress)
    click.echo("", err=True)  # newline after progress

    if result.status == "complete":
        from app.pose.finalise import (
            auto_assign_and_finalise, conf_scale_for_model, finalise_from_source_run,
        )

        scale = conf_scale_for_model(model)
        if persons_ordered is not None:
            # A segmentation label is already a stable identity, so no stitching decision is left to make.
            auto_assign_and_finalise(
                session=session, detection_run_id=result.detection_run_id, shot_id=capture_id,
                sync_config_id=sync_id, persons_ordered=persons_ordered, pose_model=model, confidence_scale=scale,
            )
        elif source_run_id is not None:
            if not finalise_from_source_run(session, result.detection_run_id, source_run_id, model,
                                            confidence_scale=scale):
                click.echo("The source run has no track assignments; the run is not finalised.", err=True)

    # Print the run ID to stdout (bare, machine-readable)
    click.echo(result.detection_run_id)
    if result.status != "complete":
        fail(f"Detection run {result.detection_run_id} ended with status '{result.status}'.")


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
# detect assign
# ---------------------------------------------------------------------------


@detect_group.command("assign")
@click.argument("run_id")
@click.option("--persons", type=click.IntRange(min=1), default=None,
              help="Number of persons in the run. Default: the number of --subjects, else the number of persons "
                   "seen together for a fifth of the time.")
@click.option("--subjects", default=None, metavar="NAME[,NAME...]",
              help="Names of the capture's persons, given in order of first appearance to the persons no existing "
                   "assignment row names.")
@click.option("--fill-only", "fill_only", is_flag=True, default=False,
              help="The run already has assignment rows (from the stitcher): keep every one of them and assign only "
                   "the boxes they do not cover. The rows also name the persons: one row per person is enough.")
@click.option("--replace", "replace", is_flag=True, default=False,
              help="The run already has assignment rows: read them as seeds that name the persons, then replace "
                   "all of them.")
@click.option("--finalise", "do_finalise", is_flag=True, default=False,
              help="Also build the person sequences, as saving the assignments in the stitcher does.")
@click.option("--dry-run", is_flag=True, default=False, help="Print the report; write nothing.")
@click.option("--report", "report_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Where to write the report for the stitcher (default: next to the session database).")
@click.pass_context
def cmd_assign(ctx: click.Context, run_id: str, persons: int | None, subjects: str | None, fill_only: bool,
               replace: bool, do_finalise: bool, dry_run: bool, report_path: Path | None) -> None:
    """Assign the track fragments of a detection run to persons from the cameras' geometry.

    The boxes' trunk keypoints are matched across the calibrated cameras into 3D persons that are followed in
    time, and every box takes the person it projects to. The step writes the same assignment rows as the
    stitcher, and prints a report: the cameras' agreement (a bad calibration shows there), and the stretches to
    check, where persons are in contact or the person was carried over a gap by weak cues. The report is also
    written as JSON, next to the session database, for the stitcher. Where persons touch, segmentation is the
    better source of identity.

    A run that already has assignment rows is left alone unless --fill-only or --replace says what may happen
    to them.
    """
    from posetrak.detection.assign_run import (
        AssignError, assign_run, format_report, report_path as default_report_path, save_report, write_assignment,
    )

    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'detect assign'.")
    if fill_only and replace:
        fail("--fill-only and --replace exclude each other.")
    try:
        session = open_session(Path(session_path))
        full_id = resolve_id_prefix(session, "detection_runs", run_id)
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))
    names = [n.strip() for n in subjects.split(",") if n.strip()] if subjects else None
    if names and persons is not None and persons < len(names):
        fail(f"--persons {persons} is fewer than the {len(names)} --subjects.")
    existing = "fill" if fill_only else "replace" if replace else "refuse"
    try:
        result = assign_run(session, full_id, persons=persons, subjects=names, existing=existing)
    except AssignError as exc:
        fail(str(exc))
    labels = {r["id"]: r["label"] for r in session.execute(
        "SELECT cv.id, ci.label FROM capture_videos cv JOIN camera_instances ci ON ci.id = cv.camera_instance_id "
        "JOIN detection_runs dr ON dr.shot_id = cv.shot_id WHERE dr.id = ?", (full_id,))}
    click.echo(format_report(result, labels))
    if dry_run:
        return
    write_assignment(session, full_id, result, existing)
    out = report_path or default_report_path(session_path, full_id)
    save_report(result, out)
    click.echo(f"Wrote {len(result.rows)} assignment rows and the report {out}.", err=True)
    if do_finalise:
        from app.pose.finalise import TrackAssignment, conf_scale_for_model, finalise_to_db

        run = session.execute("SELECT shot_id, sync_config_id, pose_model FROM detection_runs WHERE id = ?",
                              (full_id,)).fetchone()
        rows = [*result.kept, *result.rows]
        try:
            finalise_to_db(
                session=session, detection_run_id=full_id, shot_id=run["shot_id"],
                sync_config_id=run["sync_config_id"],
                assignments=[TrackAssignment(r.svid, r.tid, r.person, r.first, r.last) for r in rows],
                pose_model=run["pose_model"], confidence_scale=conf_scale_for_model(run["pose_model"]))
        except RuntimeError as exc:
            fail(str(exc))
        click.echo("Built the person sequences.", err=True)


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
