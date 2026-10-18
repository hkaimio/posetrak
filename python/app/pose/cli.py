# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""cli.py — Command-line interface for the pose extraction pipeline."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from posetrak.db.db import open_session

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)


@click.group()
def main() -> None:
    """Pose extraction pipeline: run detection and pose estimation on a shot."""


@main.command()
@click.option("--session-db", required=True, type=click.Path(exists=True), help="Session DB path.")
@click.option("--shot", required=True, help="Shot ID (prefix accepted).")
@click.option("--sync", required=True, help="Sync config ID (prefix accepted).")
@click.option("--start", type=float, required=True, help="Processing start time in seconds (global).")
@click.option("--end", type=float, required=True, help="Processing end time in seconds (global).")
@click.option("--detector", default="yolox-x", show_default=True, help="Detector model name.")
@click.option("--pose-model", default="rtmpose-l-133kp", show_default=True, help="Pose model name.")
@click.option("--device", default="cuda", show_default=True, help="Inference device.")
@click.option("--detector-conf", default=0.3, show_default=True, help="Detector confidence threshold.")
@click.option(
    "--refine-hands/--no-refine-hands", default=True, show_default=True,
    help="Refine hand keypoints with a hand-specific model (rtmlib.Hand) on the frames already decoded. "
         "Only has an effect for 133-keypoint pose models.",
)
def run(session_db, shot, sync, start, end, detector, pose_model, device, detector_conf, refine_hands):
    """Run person detection and pose estimation for a shot time range.

    Results are written directly to the session DB; no intermediate files
    are created.  Run posetrak-pose list-runs to see stored detection runs.

    Example:
        posetrak-pose run --session-db session.db --shot <id> --sync <id> \\
            --start 12 --end 105
    """
    from posetrak.detection.backends_rtmdet import YOLOXDetector
    from posetrak.detection.box_sources import YoloxBoxSource
    from posetrak.detection.pipeline import DetectionPipeline
    from posetrak.detection.pose_model_choice import create_estimator

    session = open_session(Path(session_db))
    session.row_factory = __import__("sqlite3").Row

    # Resolve ID prefixes
    shot_id = _resolve_id(session, "captures", shot)
    sync_id = _resolve_id(session, "sync_configs", sync)

    click.echo(f"Shot:        {shot_id}")
    click.echo(f"Sync config: {sync_id}")
    click.echo(f"Time range:  {start:.2f} – {end:.2f} s")
    click.echo(f"Detector:    {detector}  (conf={detector_conf})")
    click.echo(f"Pose model:  {pose_model}")
    click.echo(f"Device:      {device}")
    click.echo()

    det = YOLOXDetector(model_name=detector, device=device, conf=detector_conf)
    est = create_estimator(pose_model, device=device)
    hand_pass = None
    if refine_hands and "133" in pose_model:
        from posetrak.detection.hand_refinement import HandPass

        hand_pass = HandPass()

    def on_progress(done: int, total: int, cam_id: str) -> None:
        pct = 100 * done / max(total, 1)
        click.echo(f"\r  {cam_id}: {done}/{total} frames ({pct:.0f}%)", nl=False)

    pipeline = DetectionPipeline(
        session=session,
        shot_id=shot_id,
        sync_config_id=sync_id,
        time_start_s=start,
        time_end_s=end,
        box_source=YoloxBoxSource(det),
        estimator=est,
        hand_pass=hand_pass,
    )

    click.echo("Running detection + pose estimation...")
    result = pipeline.run(on_progress=on_progress)
    click.echo()
    click.echo(f"Detection run ID: {result.detection_run_id}")
    click.echo(f"Cameras processed: {len(result.cameras_processed)}")
    click.echo(f"Frames processed:  {result.frames_processed}")
    click.echo(f"Status: {result.status}")


@main.command(name="list-runs")
@click.option("--session-db", required=True, type=click.Path(exists=True))
@click.option("--shot", required=True)
def list_runs(session_db, shot):
    """List detection runs for a shot."""
    from app.pose.db_cache import list_detection_runs

    session = open_session(Path(session_db))
    session.row_factory = __import__("sqlite3").Row
    shot_id = _resolve_id(session, "captures", shot)

    runs = list_detection_runs(session, shot_id)
    if not runs:
        click.echo("No detection runs found for this shot.")
        return

    for r in runs:
        click.echo(
            f"{r['id'][:8]}  {r['status']:8s}  {r['detector_model']} + {r['pose_model']}"
            f"  [{r['time_start_s']:.1f}–{r['time_end_s']:.1f}s]"
            f"  {r['created_at'][:19]}"
        )


@main.command(name="ui")
@click.option("--session-db", type=click.Path(), default=None, help="Session DB to open on launch.")
def cmd_ui(session_db):
    """Launch the pose extraction GUI."""
    import sys
    from PySide6.QtWidgets import QApplication
    from app.pose.main import PoseExtractionWindow
    app = QApplication(sys.argv)
    win = PoseExtractionWindow(session_db=session_db)
    win.show()
    sys.exit(app.exec())


def _resolve_id(session, table: str, prefix: str) -> str:
    """Resolve a full or prefix ID from a table."""
    row = session.execute(
        f"SELECT id FROM {table} WHERE id = ? OR id LIKE ?",
        (prefix, f"{prefix}%"),
    ).fetchone()
    if row is None:
        click.echo(f"Error: no row in {table!r} matching {prefix!r}", err=True)
        sys.exit(1)
    return row["id"]
