# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""track.py — CLI commands for listing and exporting tracking runs."""
from __future__ import annotations

import datetime
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from posetrak.cli._output import fail, print_record, print_table
from posetrak.db.db import generate_id, open_session, resolve_id_prefix
from posetrak.db.manage_config import resolve_default_tracker_config
from posetrak.export.bvh import export_bvh
from posetrak.export.gltf import export_gltf
from posetrak.export.usd import export_usd
from posetrak.tracker.runner import (
    MultiPersonResult,
    PersonRunSpec,
    TrackerResult,
    default_binary_path,
    run_multi_person_tracker,
    run_tracker,
)


# ---------------------------------------------------------------------------
# Group
# ---------------------------------------------------------------------------


@click.group("track")
def track_group() -> None:
    """List and export tracking runs."""


# ---------------------------------------------------------------------------
# Config helpers for track run
# ---------------------------------------------------------------------------

# Defaults match the RunTrackerWidget UI defaults.
_CONFIG_DEFAULTS: dict = {
    "process_noise_std": 0.1,
    "process_noise_vel_std": 0.5,
    "velocity_half_life_s": 0.25,
    "measurement_noise_std": 60.0,
    "pose_noise_std": 0.0,
    "outlier_threshold": 4.0,
    "tracker_fps": 120.0,
    "use_relative_observations": 0,
    "relative_min_confidence": None,
    "cross_pair_max_px": None,
    "cross_pair_max_n": None,
    "velocity_mode_camera_ids": None,
}


def _get_sequence_cameras(conn: sqlite3.Connection, sync_config_id: str) -> list[str]:
    """Return ordered camera labels for a sync config."""
    rows = conn.execute(
        "SELECT ci.label"
        " FROM capture_videos sv"
        " JOIN captures sh ON sh.id = sv.shot_id"
        " JOIN sync_configs scfg ON scfg.shot_id = sh.id"
        " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
        " WHERE scfg.id = ?"
        " ORDER BY ci.label ASC",
        (sync_config_id,),
    ).fetchall()
    return [r["label"] for r in rows]


def _build_run_config(
    conn: sqlite3.Connection,
    *,
    base_config_id: str | None,
    process_noise_std: float | None,
    process_noise_vel_std: float | None,
    velocity_half_life_s: float | None,
    measurement_noise_std: float | None,
    pose_noise_std: float | None,
    outlier_threshold: float | None,
    tracker_fps: float | None,
    use_relative_obs: bool,
    relative_min_conf: float | None,
    cross_pair_radius: float | None,
    cross_pair_max_n: int | None,
    velocity_cam_labels: list[str] | None,
    sequence_cameras: list[str],
) -> str:
    """Insert a tracker_configs row into *conn* and return its ID.

    Starts from ``_CONFIG_DEFAULTS``, overlays the base config if provided,
    then overlays any non-None CLI arguments.
    """
    vals = dict(_CONFIG_DEFAULTS)

    if base_config_id is not None:
        row = conn.execute(
            "SELECT * FROM tracker_configs WHERE id = ?", (base_config_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"Config '{base_config_id}' not found in session DB.")
        for col in _CONFIG_DEFAULTS:
            db_val = row[col]
            if db_val is not None:
                vals[col] = db_val

    # Apply CLI overrides
    if process_noise_std is not None:
        vals["process_noise_std"] = process_noise_std
    if process_noise_vel_std is not None:
        vals["process_noise_vel_std"] = process_noise_vel_std
    if velocity_half_life_s is not None:
        vals["velocity_half_life_s"] = velocity_half_life_s
    if measurement_noise_std is not None:
        vals["measurement_noise_std"] = measurement_noise_std
    if pose_noise_std is not None:
        vals["pose_noise_std"] = pose_noise_std
    if outlier_threshold is not None:
        vals["outlier_threshold"] = outlier_threshold
    if tracker_fps is not None:
        vals["tracker_fps"] = tracker_fps
    if use_relative_obs:
        vals["use_relative_observations"] = 1
    if relative_min_conf is not None:
        vals["relative_min_confidence"] = relative_min_conf
    if cross_pair_radius is not None:
        vals["cross_pair_max_px"] = cross_pair_radius if cross_pair_radius > 0 else None
    if cross_pair_max_n is not None:
        vals["cross_pair_max_n"] = cross_pair_max_n

    if velocity_cam_labels is not None:
        label_to_idx = {label: i for i, label in enumerate(sequence_cameras)}
        unknown = [lb for lb in velocity_cam_labels if lb not in label_to_idx]
        if unknown:
            raise ValueError(
                f"Unknown velocity camera labels: {unknown}. "
                f"Available: {sequence_cameras}"
            )
        indices = sorted(label_to_idx[lb] for lb in velocity_cam_labels)
        vals["velocity_mode_camera_ids"] = json.dumps(indices) if indices else None

    config_id = generate_id()
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with conn:
        conn.execute(
            "INSERT INTO tracker_configs"
            " (id, name, parent_id, created_at,"
            "  process_noise_std, process_noise_vel_std, velocity_half_life_s,"
            "  measurement_noise_std, pose_noise_std, outlier_threshold, tracker_fps,"
            "  velocity_mode_camera_ids, use_relative_observations, relative_min_confidence,"
            "  cross_pair_max_px, cross_pair_max_n)"
            " VALUES (?, 'cli-run', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                config_id, base_config_id, now,
                vals["process_noise_std"],
                vals["process_noise_vel_std"],
                vals["velocity_half_life_s"],
                vals["measurement_noise_std"],
                vals["pose_noise_std"],
                vals["outlier_threshold"],
                vals["tracker_fps"],
                vals["velocity_mode_camera_ids"],
                vals["use_relative_observations"],
                vals["relative_min_confidence"],
                vals["cross_pair_max_px"],
                vals["cross_pair_max_n"],
            ),
        )
    return config_id


# ---------------------------------------------------------------------------
# track run
# ---------------------------------------------------------------------------


@track_group.command("run")
@click.option("--sequence", required=True, help="Pose observation sequence ID (prefix accepted).")
@click.option("--skeleton", required=True, help="Skeleton ID (prefix accepted).")
@click.option("--config", "base_config_id", default=None, help="Base tracker config ID (prefix accepted). Defaults used when omitted.")
@click.option("--output-dir", default=None, help="Output directory. Default: <session-dir>/posetrak_results/<capture>/<skeleton>/tracking")
@click.option("--person-id", default=0, show_default=True, type=int, help="Person index.")
@click.option("--start-time", default=None, type=float, help="Override sequence start time (s).")
@click.option("--end-time", default=None, type=float, help="Override sequence end time (s).")
@click.option("--no-smooth", is_flag=True, default=False, help="Disable RTS smoothing.")
@click.option("--binary", default=None, help="Path to posetrak-tracker binary.")
# Config parameter overrides
@click.option("--process-noise-std", default=None, type=float, help="Process noise std (rad/s²). Default 0.1.")
@click.option("--process-vel-noise-std", default=None, type=float, help="Velocity process noise std. Default 0.5.")
@click.option("--vel-half-life", default=None, type=float, help="Velocity decay half-life (s). Default 0.25.")
@click.option("--calib-noise-std", default=None, type=float, help="Calibration (extrinsic) noise std (px). Default 60.")
@click.option("--pose-noise-std", default=None, type=float, help="Pose estimation noise std (px). Default 0.")
@click.option("--outlier-threshold", default=None, type=float, help="Mahalanobis outlier threshold (σ). Default 4.")
@click.option("--tracker-fps", default=None, type=float, help="Target tracker frame rate (Hz). Default 120.")
@click.option("--use-relative-obs", is_flag=True, default=False, help="Enable child-minus-parent relative observations.")
@click.option("--relative-min-conf", default=None, type=float, help="Min keypoint confidence for relative pairs. Default 0.5.")
@click.option("--cross-pair-radius", default=None, type=float, help="Cross-pair spatial radius (px). 0 or omit = disabled.")
@click.option("--cross-pair-max-n", default=None, type=int, help="Max cross-pairs per frame per camera. Default 10.")
@click.option("--velocity-cameras", default=None, help="Comma-separated camera labels to run in velocity mode.")
@click.pass_context
def cmd_run(
    ctx: click.Context,
    sequence: str,
    skeleton: str,
    base_config_id: str | None,
    output_dir: str | None,
    person_id: int,
    start_time: float | None,
    end_time: float | None,
    no_smooth: bool,
    binary: str | None,
    process_noise_std: float | None,
    process_vel_noise_std: float | None,
    vel_half_life: float | None,
    calib_noise_std: float | None,
    pose_noise_std: float | None,
    outlier_threshold: float | None,
    tracker_fps: float | None,
    use_relative_obs: bool,
    relative_min_conf: float | None,
    cross_pair_radius: float | None,
    cross_pair_max_n: int | None,
    velocity_cameras: str | None,
) -> None:
    """Run the tracker binary on a pose observation sequence.

    Creates a tracker_configs row in the session DB from the supplied parameters
    (or from the base --config if given), then invokes the posetrak-tracker
    binary as a subprocess.  Prints tracking_run_id to stdout on success.

    Example:

        posetrak -s session.db track run \\
            --sequence <seq-id> --skeleton <skel-id> \\
            --calib-noise-std 40 --outlier-threshold 5
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track run'.")

    try:
        conn = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    # Resolve IDs
    try:
        sequence_id = resolve_id_prefix(conn, "pose_observation_sequences", sequence)
    except ValueError as exc:
        fail(str(exc))

    try:
        skeleton_id = resolve_id_prefix(conn, "skeletons", skeleton)
    except ValueError as exc:
        fail(str(exc))

    if base_config_id is not None:
        try:
            base_config_id = resolve_id_prefix(conn, "tracker_configs", base_config_id)
        except ValueError as exc:
            fail(str(exc))

    # Sequence info (capture label, sync_config_id, time range)
    seq_row = conn.execute(
        "SELECT pos.sync_config_id, sh.label, sh.capture_number,"
        "       pos.time_start_s, pos.time_end_s"
        " FROM pose_observation_sequences pos"
        " JOIN captures sh ON sh.id = pos.shot_id"
        " WHERE pos.id = ?",
        (sequence_id,),
    ).fetchone()
    if seq_row is None:
        fail(f"Sequence '{sequence_id}' not found.")

    # Cameras (for velocity-cameras resolution)
    seq_cameras: list[str] = []
    if seq_row["sync_config_id"]:
        seq_cameras = _get_sequence_cameras(conn, seq_row["sync_config_id"])

    vel_cam_labels: list[str] | None = None
    if velocity_cameras:
        vel_cam_labels = [c.strip() for c in velocity_cameras.split(",") if c.strip()]

    # Build config row
    try:
        config_id = _build_run_config(
            conn,
            base_config_id=base_config_id,
            process_noise_std=process_noise_std,
            process_noise_vel_std=process_vel_noise_std,
            velocity_half_life_s=vel_half_life,
            measurement_noise_std=calib_noise_std,
            pose_noise_std=pose_noise_std,
            outlier_threshold=outlier_threshold,
            tracker_fps=tracker_fps,
            use_relative_obs=use_relative_obs,
            relative_min_conf=relative_min_conf,
            cross_pair_radius=cross_pair_radius,
            cross_pair_max_n=cross_pair_max_n,
            velocity_cam_labels=vel_cam_labels,
            sequence_cameras=seq_cameras,
        )
    except ValueError as exc:
        fail(str(exc))

    # Default output dir
    if output_dir:
        out_path = Path(output_dir)
    else:
        capture_label = seq_row["label"] or f"capture{seq_row['capture_number']:03d}"
        skel_name_row = conn.execute(
            "SELECT name FROM skeletons WHERE id = ?", (skeleton_id,)
        ).fetchone()
        skel_name = (skel_name_row["name"] if skel_name_row else "skeleton").replace(" ", "_")
        out_path = Path(session_path).parent / "posetrak_results" / capture_label / skel_name / "tracking"

    out_path.mkdir(parents=True, exist_ok=True)
    conn.close()

    # Binary path
    binary_path = Path(binary) if binary else default_binary_path()
    if not binary_path.exists():
        fail(
            f"Tracker binary not found: {binary_path}\n"
            "Build with: meson setup optbuild --buildtype=release && meson compile -C optbuild"
        )

    click.echo(f"Sequence:   {sequence_id}", err=True)
    click.echo(f"Skeleton:   {skeleton_id}", err=True)
    click.echo(f"Config:     {config_id}", err=True)
    click.echo(f"Output dir: {out_path}", err=True)
    click.echo(f"Binary:     {binary_path}", err=True)
    click.echo("", err=True)

    result: TrackerResult = run_tracker(
        Path(session_path),
        sequence_id,
        skeleton_id,
        config_id,
        out_path,
        binary_path=binary_path,
        person_id=person_id,
        start_time=start_time if start_time is not None else seq_row["time_start_s"],
        end_time=end_time if end_time is not None else seq_row["time_end_s"],
        smooth=not no_smooth,
        on_progress=lambda line: click.echo(line, err=True),
    )

    if result.exit_code != 0:
        sys.exit(result.exit_code)

    if result.run_id:
        click.echo(result.run_id)
    else:
        click.echo("(no tracking_run_id emitted by binary)", err=True)


# ---------------------------------------------------------------------------
# track run-persons
# ---------------------------------------------------------------------------


@dataclass
class ResolvedPerson:
    name: str
    sequence_id: str
    skeleton_id: str
    time_start_s: float
    time_end_s: float


def resolve_trial_persons(
    conn: sqlite3.Connection, trial_id: str, names: list[str]
) -> list[ResolvedPerson]:
    """Resolve *names* -- this trial's capture's ``capture_persons`` -- to
    the (sequence, skeleton, time range) a ``--person`` 4-tuple needs.

    A scripted equivalent of ``RunTrackerWidget``'s capture-persons people
    table (config-improvements design doc, "Person model": E, CLI): each
    name must have exactly one detection run's observations in this trial
    to auto-select from (matching by ``capture_person_id``, falling back to
    an exact ``person_name`` match for pre-migration rows), and a
    ``default_skeleton_id`` set -- both ambiguity and a missing default are
    reported per-name rather than picked arbitrarily, since a script has no
    one to ask. Use the lower-level ``--person <seq> <skel> <config> <id>``
    flag directly for anything this can't resolve unambiguously.

    Raises
    ------
    ValueError
        If *trial_id* doesn't exist, a name isn't a defined person for this
        trial's capture, that person has no default skeleton, or that
        person's observations in this trial are missing or ambiguous.
    """
    trial_row = conn.execute(
        "SELECT capture_id FROM trials WHERE id = ?", (trial_id,)
    ).fetchone()
    if trial_row is None:
        raise ValueError(f"trial not found: {trial_id!r}")
    capture_id = trial_row["capture_id"]

    resolved: list[ResolvedPerson] = []
    for name in names:
        person_row = conn.execute(
            "SELECT id, default_skeleton_id FROM capture_persons"
            " WHERE capture_id = ? AND name = ?",
            (capture_id, name),
        ).fetchone()
        if person_row is None:
            raise ValueError(
                f"No person named {name!r} defined for this trial's capture "
                f"({capture_id!r}). Define one via CapturePanel's Persons "
                "section first."
            )
        if not person_row["default_skeleton_id"]:
            raise ValueError(
                f"Person {name!r} has no default skeleton set. Set one via "
                "CapturePanel's Persons section, or use --person directly."
            )

        seq_rows = conn.execute(
            "SELECT pos.id AS seq_id, pos.time_start_s, pos.time_end_s"
            " FROM detection_runs dr"
            " JOIN pose_observation_sequences pos ON pos.detection_run_id = dr.id"
            " JOIN sequence_persons sp ON sp.sequence_id = pos.id"
            " WHERE dr.trial_id = ? AND sp.person_id = 0"
            "   AND (sp.capture_person_id = ?"
            "        OR (sp.capture_person_id IS NULL AND sp.person_name = ?))"
            " ORDER BY dr.created_at DESC",
            (trial_id, person_row["id"], name),
        ).fetchall()
        if not seq_rows:
            raise ValueError(
                f"No detection-run observations for {name!r} in trial {trial_id!r}."
            )
        if len(seq_rows) > 1:
            raise ValueError(
                f"Ambiguous: {len(seq_rows)} detection runs have observations "
                f"for {name!r} in trial {trial_id!r}. Use --person <seq> <skel> "
                "<config> <id> directly to pick one."
            )
        seq_row = seq_rows[0]
        resolved.append(ResolvedPerson(
            name=name,
            sequence_id=seq_row["seq_id"],
            skeleton_id=person_row["default_skeleton_id"],
            time_start_s=seq_row["time_start_s"],
            time_end_s=seq_row["time_end_s"],
        ))
    return resolved


@track_group.command("run-persons")
@click.option("--trial", "trial_id", required=True, help="Trial ID (prefix accepted).")
@click.option(
    "--persons", required=True,
    help="Comma-separated capture_persons names to track together (e.g. Alice,Bob).",
)
@click.option(
    "--config", "base_config_id", default=None,
    help="Base tracker config ID (prefix accepted). Defaults to the trial's own "
         "resolved default (see 'Default tracker config' in the trial/capture panels).",
)
@click.option("--output-dir", default=None, help="Output directory. Default: <session-dir>/posetrak_results/<capture>/<trial>/tracking")
@click.option("--no-smooth", is_flag=True, default=False, help="Disable RTS smoothing.")
@click.option("--binary", default=None, help="Path to posetrak-tracker binary.")
@click.pass_context
def cmd_run_persons(
    ctx: click.Context,
    trial_id: str,
    persons: str,
    base_config_id: str | None,
    output_dir: str | None,
    no_smooth: bool,
    binary: str | None,
) -> None:
    """Run the tracker for named persons in a trial, resolved automatically.

    Higher-level alternative to 'track run': resolves each of --persons'
    comma-separated names against this trial's capture's capture_persons
    (see CapturePanel's Persons section) to a sequence/skeleton instead of
    requiring the caller to already know them. Additive to the existing
    --person 4-tuple mechanism (cli/track.cpp), not a replacement -- use
    'track run' directly for anything this can't resolve unambiguously.

    Example:

        posetrak -s session.db track run-persons \\
            --trial <trial-id> --persons Alice,Bob
    """
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track run-persons'.")

    try:
        conn = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    try:
        trial_id = resolve_id_prefix(conn, "trials", trial_id)
    except ValueError as exc:
        fail(str(exc))

    names = [n.strip() for n in persons.split(",") if n.strip()]
    if not names:
        fail("--persons must list at least one name.")

    try:
        resolved = resolve_trial_persons(conn, trial_id, names)
    except ValueError as exc:
        fail(str(exc))

    if base_config_id is not None:
        try:
            config_id = resolve_id_prefix(conn, "tracker_configs", base_config_id)
        except ValueError as exc:
            fail(str(exc))
    else:
        config_id = resolve_default_tracker_config(conn, trial_id=trial_id)

    # Default output dir
    if output_dir:
        out_path = Path(output_dir)
    else:
        trial_row = conn.execute(
            "SELECT cap.label AS capture_label, cap.capture_number, t.name AS trial_name"
            " FROM trials t JOIN captures cap ON cap.id = t.capture_id"
            " WHERE t.id = ?",
            (trial_id,),
        ).fetchone()
        capture_label = trial_row["capture_label"] or f"capture{trial_row['capture_number']:03d}"
        trial_label = (trial_row["trial_name"] or trial_id[:8]).replace(" ", "_")
        out_path = Path(session_path).parent / "posetrak_results" / capture_label / trial_label / "tracking"

    out_path.mkdir(parents=True, exist_ok=True)
    conn.close()

    binary_path = Path(binary) if binary else default_binary_path()
    if not binary_path.exists():
        fail(
            f"Tracker binary not found: {binary_path}\n"
            "Build with: meson setup optbuild --buildtype=release && meson compile -C optbuild"
        )

    person_specs = [
        PersonRunSpec(r.sequence_id, r.skeleton_id, config_id, 0) for r in resolved
    ]

    click.echo(f"Trial:      {trial_id}", err=True)
    for r in resolved:
        click.echo(f"  {r.name}: sequence={r.sequence_id} skeleton={r.skeleton_id}", err=True)
    click.echo(f"Config:     {config_id}", err=True)
    click.echo(f"Output dir: {out_path}", err=True)
    click.echo(f"Binary:     {binary_path}", err=True)
    click.echo("", err=True)

    result: MultiPersonResult = run_multi_person_tracker(
        Path(session_path),
        person_specs,
        out_path,
        binary_path=binary_path,
        start_time=resolved[0].time_start_s,
        end_time=resolved[0].time_end_s,
        smooth=not no_smooth,
        on_progress=lambda line: click.echo(line, err=True),
    )

    if result.exit_code != 0:
        sys.exit(result.exit_code)

    for name, run_id in zip(names, result.run_ids):
        click.echo(f"{name}: {run_id or '(no tracking_run_id emitted)'}")


# ---------------------------------------------------------------------------
# track list
# ---------------------------------------------------------------------------


@track_group.command("list")
@click.option("--sequence", default=None, help="Filter by observation_sequence_id (prefix accepted).")
@click.pass_context
def cmd_list(ctx: click.Context, sequence: str | None) -> None:
    """List tracking runs in the session."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track list'.")

    try:
        conn = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    sequence_id: str | None = None
    if sequence is not None:
        try:
            sequence_id = resolve_id_prefix(conn, "observation_sequences", sequence)
        except ValueError as exc:
            fail(str(exc))

    query = (
        "SELECT id, skeleton_id, ran_at, posetrak_version "
        "FROM tracking_runs"
    )
    params: list = []
    if sequence_id is not None:
        query += " WHERE observation_sequence_id = ?"
        params.append(sequence_id)
    query += " ORDER BY ran_at DESC"

    rows = [dict(r) for r in conn.execute(query, params).fetchall()]
    conn.close()

    if not rows:
        sys.exit(0)

    columns = ["id", "skeleton_id", "ran_at", "posetrak_version"]
    print_table(rows, columns, json_mode=ctx.obj["json_mode"])


# ---------------------------------------------------------------------------
# track show
# ---------------------------------------------------------------------------


@track_group.command("show")
@click.argument("run_id")
@click.pass_context
def cmd_show(ctx: click.Context, run_id: str) -> None:
    """Show details of a single tracking run."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track show'.")

    try:
        conn = open_session(Path(session_path))
    except (FileNotFoundError, ValueError) as exc:
        fail(str(exc))

    try:
        full_id = resolve_id_prefix(conn, "tracking_runs", run_id)
    except ValueError as exc:
        fail(str(exc))

    row = conn.execute(
        "SELECT id, observation_sequence_id, tracker_config_id, skeleton_id, "
        "       extrinsic_calibration_id, sync_config_id, ran_at, "
        "       posetrak_version, active_camera_ids, marker_names, notes "
        "FROM tracking_runs WHERE id = ?",
        (full_id,),
    ).fetchone()
    conn.close()

    if row is None:
        fail(f"Tracking run '{full_id}' not found.")

    print_record(dict(row), json_mode=ctx.obj["json_mode"])


# ---------------------------------------------------------------------------
# track export
# ---------------------------------------------------------------------------


@track_group.group("export")
def export_group() -> None:
    """Export tracking run data to an animation format."""


def _resolve_run(session_path: str, run_id: str) -> str:
    """Resolve a run_id prefix to a full ID."""
    conn = open_session(Path(session_path))
    try:
        return resolve_id_prefix(conn, "tracking_runs", run_id)
    except ValueError as exc:
        fail(str(exc))
    finally:
        conn.close()


_EXPORT_OPTIONS = [
    click.option("--smoothed", is_flag=True, default=False, help="Use RTS-smoothed output."),
    click.option("--fps", default=None, type=float, help="Override output frame rate."),
    click.option(
        "--units",
        default="m",
        show_default=True,
        type=click.Choice(["m", "cm"]),
        help="Output length units.",
    ),
    click.option(
        "--coord",
        default="yup",
        show_default=True,
        type=click.Choice(["yup", "zup"]),
        help="Output coordinate convention.",
    ),
    click.option("--person-id", default=0, show_default=True, type=int, help="Person index."),
]


def _add_export_options(cmd):
    for opt in reversed(_EXPORT_OPTIONS):
        cmd = opt(cmd)
    return cmd


@export_group.command("bvh")
@click.argument("run_id")
@click.argument("output")
@_add_export_options
@click.pass_context
def cmd_export_bvh(
    ctx: click.Context,
    run_id: str,
    output: str,
    smoothed: bool,
    fps: float | None,
    units: str,
    coord: str,
    person_id: int,
) -> None:
    """Export a tracking run to BVH format."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track export'.")

    full_id = _resolve_run(session_path, run_id)

    export_bvh(
        output,
        session_db=session_path,
        run_id=full_id,
        person_id=person_id,
        fps=fps,
        units=units,
        coord=coord,
        smoothed=smoothed,
    )
    click.echo(f"Wrote {output}", err=True)


@export_group.command("gltf")
@click.argument("run_id")
@click.argument("output")
@_add_export_options
@click.pass_context
def cmd_export_gltf(
    ctx: click.Context,
    run_id: str,
    output: str,
    smoothed: bool,
    fps: float | None,
    units: str,
    coord: str,
    person_id: int,
) -> None:
    """Export a tracking run to glTF format."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track export'.")

    full_id = _resolve_run(session_path, run_id)

    export_gltf(
        output,
        session_db=session_path,
        run_id=full_id,
        person_id=person_id,
        fps=fps,
        units=units,
        coord=coord,
        smoothed=smoothed,
    )
    click.echo(f"Wrote {output}", err=True)


@export_group.command("usd")
@click.argument("run_id")
@click.argument("output")
@_add_export_options
@click.pass_context
def cmd_export_usd(
    ctx: click.Context,
    run_id: str,
    output: str,
    smoothed: bool,
    fps: float | None,
    units: str,
    coord: str,
    person_id: int,
) -> None:
    """Export a tracking run to USD format."""
    session_path: str | None = ctx.obj.get("session")
    if session_path is None:
        fail("--session / POSETRAK_SESSION_DB is required for 'track export'.")

    full_id = _resolve_run(session_path, run_id)

    try:
        export_usd(
            output,
            session_db=session_path,
            run_id=full_id,
            person_id=person_id,
            fps=fps,
            units=units,
            coord=coord,
            smoothed=smoothed,
        )
    except ImportError:
        fail("usd-core package is required. Install with: uv sync --group usd")

    click.echo(f"Wrote {output}", err=True)
