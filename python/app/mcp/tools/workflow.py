"""workflow.py — MCP tools for listing captures/detection-runs and triggering
detection/tracking pipelines.

Read tools (list_captures, list_detection_runs, get_capture_info) use the
server's read-only DB connection. Write tools (run_detection, run_tracking)
require the server to be started with --mcp-allow-write and open their own
writable connections.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# Read tools
# ---------------------------------------------------------------------------


def list_captures(conn: sqlite3.Connection) -> str:
    """List all captures with video count and sync/extrinsics status."""
    rows = conn.execute(
        """SELECT sh.id, sh.label, sh.capture_number,
                  COUNT(DISTINCT sv.id)  AS n_videos,
                  COUNT(DISTINCT sc.id)  AS n_syncs,
                  sh.extrinsic_calibration_id IS NOT NULL AS has_extrinsics
           FROM captures sh
           LEFT JOIN capture_videos sv ON sv.shot_id = sh.id
           LEFT JOIN sync_configs sc   ON sc.shot_id = sh.id
           GROUP BY sh.id
           ORDER BY sh.capture_number"""
    ).fetchall()

    if not rows:
        return "No captures found in this database."

    lines = ["Captures:\n"]
    for r in rows:
        label = r["label"] or f"capture{r['capture_number']:03d}"
        sync_mark = "✓" if r["n_syncs"] else "✗"
        ext_mark  = "✓" if r["has_extrinsics"] else "✗"
        lines.append(
            f"  {r['id']}\n"
            f"    label:      {label}\n"
            f"    videos:     {r['n_videos']}\n"
            f"    sync:       {sync_mark}  extrinsics: {ext_mark}\n"
        )
    return "".join(lines)


def list_detection_runs(conn: sqlite3.Connection) -> str:
    """List all detection runs with capture label and status."""
    rows = conn.execute(
        """SELECT dr.id, sh.label AS capture_label, sh.capture_number,
                  dr.detector_model, dr.pose_model, dr.status,
                  dr.time_start_s, dr.time_end_s, dr.created_at
           FROM detection_runs dr
           JOIN captures sh ON sh.id = dr.shot_id
           ORDER BY dr.created_at DESC"""
    ).fetchall()

    if not rows:
        return "No detection runs found in this database."

    lines = ["Detection runs (newest first):\n"]
    for r in rows:
        label = r["capture_label"] or f"capture{r['capture_number']:03d}"
        duration = (
            f"{r['time_start_s']:.1f}–{r['time_end_s']:.1f}s"
            if r["time_start_s"] is not None else "?"
        )
        lines.append(
            f"  {r['id']}\n"
            f"    capture:    {label}\n"
            f"    time:       {duration}\n"
            f"    detector:   {r['detector_model']}  pose: {r['pose_model']}\n"
            f"    status:     {r['status']}  created: {r['created_at']}\n"
        )
    return "".join(lines)


def get_capture_info(conn: sqlite3.Connection, capture_id: str) -> str:
    """Return detailed info about a single capture."""
    sh = conn.execute(
        "SELECT * FROM captures WHERE id = ?", (capture_id,)
    ).fetchone()
    if sh is None:
        return f"Capture not found: {capture_id}"

    videos = conn.execute(
        """SELECT sv.id, ci.label AS cam_label, sv.file_path,
                  sv.first_video_frame, sv.last_video_frame, sv.actual_fps
           FROM capture_videos sv
           JOIN camera_instances ci ON ci.id = sv.camera_instance_id
           WHERE sv.shot_id = ?
           ORDER BY ci.label""",
        (capture_id,),
    ).fetchall()

    syncs = conn.execute(
        "SELECT id, created_by FROM sync_configs WHERE shot_id = ?", (capture_id,)
    ).fetchall()

    lines = [
        f"Capture: {capture_id}",
        f"Label:   {sh['label'] or '(none)'}",
        f"Number:  {sh['capture_number']}",
        f"Extrinsics: {'yes — ' + sh['extrinsic_calibration_id'][:8] if sh['extrinsic_calibration_id'] else 'not set'}",
        "",
        f"Videos ({len(videos)}):",
    ]
    for v in videos:
        n_frames = (
            v["last_video_frame"] - v["first_video_frame"] + 1
            if v["last_video_frame"] is not None else "?"
        )
        lines.append(
            f"  {v['cam_label']:<30} {n_frames} frames  {v['actual_fps']:.1f} fps"
            f"\n    {v['file_path']}"
        )

    lines.append("")
    lines.append(f"Sync configs ({len(syncs)}):")
    for s in syncs:
        lines.append(f"  {s['id']}  (created_by: {s['created_by']})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Write tools
# ---------------------------------------------------------------------------


def run_detection(
    db_path: Path,
    capture_id: str,
    sync_id: str,
    start_s: float,
    end_s: float,
    detector_model: str = "yolox-x",
    pose_model: str = "rtmpose-l-133kp",
    conf: float = 0.3,
) -> str:
    """Run person detection + RTMPose estimation for a capture time range.

    Opens a writable connection to *db_path* and calls DetectionPipeline.run().
    Imports are deferred so the tool fails gracefully when detection dependencies
    are not installed.
    """
    try:
        from posetrak.detection.backends_rtmpose import RTMPoseEstimator
        from posetrak.detection.backends_rtmdet import YOLOXDetector
        from posetrak.detection.pipeline import DetectionPipeline
    except ImportError as exc:
        return f"Error: detection dependencies not available — {exc}"

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        det = YOLOXDetector(model_name=detector_model, device=None, conf=conf)
        est = RTMPoseEstimator(model_name=pose_model, device=None)

        pipeline = DetectionPipeline(
            session=conn,
            shot_id=capture_id,
            sync_config_id=sync_id,
            time_start_s=start_s,
            time_end_s=end_s,
            detector=det,
            estimator=est,
        )
        result = pipeline.run()
        return f"Detection complete. detection_run_id: {result.detection_run_id}"
    except Exception as exc:
        return f"Error: {exc}"
    finally:
        conn.close()


def run_tracking(
    db_path: Path,
    sequence_id: str,
    skeleton_id: str,
    config_id: str,
    output_dir: str,
    person_id: int = 0,
) -> str:
    """Invoke the posetrak-tracker binary for a pose observation sequence.

    Runs synchronously — this will block until tracking completes (potentially
    several minutes). Returns the tracking_run_id on success.
    """
    from posetrak.tracker.runner import TrackerResult, default_binary_path
    from posetrak.tracker.runner import run_tracker as _run_tracker

    binary = default_binary_path()
    if not binary.exists():
        return (
            f"Error: tracker binary not found at {binary}. "
            "Build with: meson setup optbuild --buildtype=release && meson compile -C optbuild"
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    result: TrackerResult = _run_tracker(
        db_path,
        sequence_id,
        skeleton_id,
        config_id,
        out_path,
        binary_path=binary,
        person_id=person_id,
        on_progress=lines.append,
    )

    if result.exit_code != 0:
        tail = "\n".join(lines[-20:])
        return f"Tracker exited with code {result.exit_code}.\n\nLast output:\n{tail}"

    return f"Tracking complete. tracking_run_id: {result.run_id}"
