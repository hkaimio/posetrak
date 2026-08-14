"""Marker-body-based extrinsics commands: anchor-rig, reanchor.

See docs/roadmap/features/extrinsics-improvements/
extrinsics-improvements-design.md, section 9 (Tier A/B) and section 10.
These commands replace the throwaway prototype scripts
(``tools/test_rig_anchor_capture1.py``, ``tools/test_reanchor_capture2.py``)
that validated the underlying mechanism against real footage (see
status.md, 2026-08-11/12) -- everything solver-facing here is already
implemented and tested in ``fiducial_markers.py``/``extrinsics_solver.py``;
this module is only the CLI plumbing (frame reading, camera/intrinsics
resolution, DB persistence) around that already-validated core.

Adds commands to the existing ``extrinsics`` group (defined in
``posetrak.cli.session``) rather than creating a new top-level group.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import click

from posetrak.detection.frame_source import iter_frames

from app.setup.extrinsics_solver import (
    CamCalibState,
    MarkerGroup,
    run_calibration,
    write_extrinsics_to_db,
)
from app.setup.fiducial_markers import (
    ArucoDetector,
    MarkerRigDetector,
    anchor_from_marker_rig,
    load_marker_body_yaml,
    merge_detections_into_groups,
)

from posetrak.db.manage_marker_body import (
    delete_scene_marker_body,
    list_scene_marker_bodies,
    list_scene_marker_group_names,
    upsert_scene_marker_body,
)
from posetrak.cli.session import extrinsics_group, _open_session_required, _resolve
from posetrak.cli._output import print_table
from posetrak.db.db import set_capture_extrinsics


# ---------------------------------------------------------------------------
# Camera spec parsing (shared by both commands)
# ---------------------------------------------------------------------------


@dataclass
class _CameraSpec:
    label: str
    camera_mode: str | None
    video_path: str
    frame_idx: int


def _parse_camera_spec(raw: str) -> _CameraSpec:
    # "|"-separated, not ":" -- a Windows drive letter ("D:/...") makes ":"
    # unusable as the field separator for video_path.
    parts = raw.split("|")
    if len(parts) != 4:
        raise click.UsageError(
            f"--camera must be 'label|camera_mode|video_path|frame_idx', got {raw!r}"
        )
    label, mode, video_path, frame_idx = parts
    try:
        idx = int(frame_idx)
    except ValueError:
        raise click.UsageError(f"--camera frame_idx must be an integer, got {frame_idx!r}")
    return _CameraSpec(label=label, camera_mode=mode or None, video_path=video_path, frame_idx=idx)


def _read_one_frame(path: str, frame_idx: int) -> np.ndarray:
    for _, img in iter_frames(path, frame_idx, frame_idx + 1):
        return img
    raise click.ClickException(f"Could not read frame {frame_idx} from {path}")


def _resolve_intrinsics(conn: sqlite3.Connection, camera_label: str, camera_mode: str | None) -> dict:
    """Resolve one camera_instances.label (+ optional camera_mode substring)
    to K/K_orig/dist/fisheye, from whichever DB *conn* is (session DBs
    mirror camera_modes/intrinsics_calibrations from the registry, so no
    separate registry connection is needed here).

    Deliberately does not silently pick a calibration when a camera model
    has more than one recording mode -- see fiducial_markers.load_rig_config
    era prototype (characterize_rig_from_video.py's _load_intrinsics) for
    the live bug this guards against (an ACE2 Pro's "MEGA mode" vs
    "4K 120 fps linear" modes are genuinely different FOV/distortion
    profiles, not just different resolutions).
    """
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        modes = conn.execute(
            """
            SELECT cm.id, cm.width_px, cm.height_px, cm.nominal_fps, cm.notes,
                   cm.default_intrinsics_calibration_id
            FROM camera_instances ci
            JOIN camera_modes cm ON cm.camera_model_id = ci.camera_model_id
            WHERE ci.label = ?
            """,
            (camera_label,),
        ).fetchall()
        if not modes:
            raise click.ClickException(
                f"No camera_modes found for camera_instances.label={camera_label!r}"
            )
        if camera_mode:
            needle = camera_mode.lower()
            modes = [m for m in modes if needle in (m["notes"] or "").lower()]
        if len(modes) != 1:
            lines = "\n".join(
                f"  id={m['id']}  {m['width_px']}x{m['height_px']}@{m['nominal_fps']}fps  "
                f"notes={m['notes']!r}"
                for m in modes
            )
            raise click.ClickException(
                f"{camera_label!r} has {len(modes)} matching camera_modes for "
                f"camera_mode={camera_mode!r} (need exactly 1). Candidates:\n{lines}"
            )
        mode = modes[0]
        calib_id = mode["default_intrinsics_calibration_id"]
        row = None
        if calib_id is not None:
            row = conn.execute(
                "SELECT * FROM intrinsics_calibrations WHERE id = ?", (calib_id,)
            ).fetchone()
        if row is None:
            row = conn.execute(
                "SELECT * FROM intrinsics_calibrations WHERE camera_mode_id = ? "
                "ORDER BY calibrated_at DESC LIMIT 1",
                (mode["id"],),
            ).fetchone()
        if row is None:
            raise click.ClickException(
                f"camera_modes.id={mode['id']} ({mode['notes']!r}) has no intrinsics_calibrations"
            )
    finally:
        conn.row_factory = old_factory

    fx, fy, cx, cy = row["fx"], row["fy"], row["cx"], row["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    K_orig = K.copy()
    if row["matrix_original"]:
        vals = struct.unpack("<9d", bytes(row["matrix_original"]))
        K_orig = np.array(vals).reshape(3, 3)
    if row["dist_coeffs"]:
        n = len(bytes(row["dist_coeffs"])) // 8
        dist = np.array(struct.unpack(f"<{n}d", bytes(row["dist_coeffs"]))).reshape(1, -1)
    else:
        dist = np.zeros((1, 4))
    return {"K": K, "K_orig": K_orig, "dist": dist, "fisheye": row["distortion_model"] == "fisheye"}


def _label_to_instance_id(conn: sqlite3.Connection) -> dict[str, str]:
    old_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT id, label FROM camera_instances").fetchall()
    finally:
        conn.row_factory = old_factory
    return {r["label"]: r["id"] for r in rows}


# ---------------------------------------------------------------------------
# anchor-rig
# ---------------------------------------------------------------------------


@extrinsics_group.command("anchor-rig")
@click.option("--session", "session_row", required=True, metavar="UUID", help="mocap_sessions.id")
@click.option("--marker-body", required=True, metavar="ID_OR_PREFIX",
              help="marker_body_definitions.id (or prefix) for the calibration rig")
@click.option("--camera", "cameras", multiple=True, required=True, metavar="SPEC",
              help="'label|camera_mode|video_path|frame_idx' (repeatable, one per camera)")
@click.option("--rig-dict", default="DICT_4X4_50", show_default=True)
@click.option("--scattered-dict", default="DICT_5X5_50", show_default=True,
              help="Dictionary for ordinary scattered scene tags (Tier B), if any")
@click.option("--tag-size", type=float, default=None, metavar="METRES",
              help="Physical side length of scattered tags. Without this, scattered tags "
                   "still help the solve as free correspondences but their poses cannot be "
                   "persisted to scene_marker_bodies (no known local geometry to solve for).")
@click.option("--min-marker-perimeter-rate", type=float, default=0.01, show_default=True)
@click.option("--method", default="rig-anchor", show_default=True)
@click.option("--capture", default=None, metavar="UUID",
              help="captures.id to link (sets extrinsic_calibration_id)")
@click.option("--name", "group_name", required=True, metavar="NAME",
              help="Name grouping this room's scene markers (e.g. 'room7') so a later "
                   "capture can pick it by name via 'reanchor --name' or the GUI's "
                   "\"Load Markers…\". Required -- matches the GUI's Save Markers flow, "
                   "which always asks for a name (see extrinsics-ux-redesign.md, UX Phase 5).")
@click.pass_obj
def extrinsics_anchor_rig(
    obj: dict,
    session_row: str,
    marker_body: str,
    cameras: tuple[str, ...],
    rig_dict: str,
    scattered_dict: str,
    tag_size: float | None,
    min_marker_perimeter_rate: float,
    method: str,
    capture: str | None,
    group_name: str,
) -> None:
    """Anchor a capture's cameras from a portable calibration rig.

    Detects the named marker body (a rig, section 10) across the given
    cameras/frames, uses it to fix the world frame (no per-camera solvePnP
    ambiguity -- a rig's markers are non-coplanar by design, see
    anchor_from_marker_rig), solves the cameras, and writes the result
    straight to extrinsic_calibrations/extrinsic_entries -- the same
    write path the GUI uses (write_extrinsics_to_db), not a TOML
    round-trip. Also upserts the rig's own anchor pose (identity, by
    construction) and any sized scattered tags into scene_marker_bodies.
    """
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session_row)
        body_id = _resolve(conn, "marker_body_definitions", marker_body)
        body_row = conn.execute(
            "SELECT yaml_content FROM marker_body_definitions WHERE id = ?", (body_id,)
        ).fetchone()
        if body_row is None:
            raise click.ClickException(f"Marker body not found: {marker_body}")
        rig_config = load_marker_body_yaml(body_row[0], rig_id=body_id)

        specs = [_parse_camera_spec(c) for c in cameras]
        label_to_instance = _label_to_instance_id(conn)

        rig_detector = MarkerRigDetector(
            rig_config, dictionary=rig_dict, min_marker_perimeter_rate=min_marker_perimeter_rate,
        )
        scattered_detector = ArucoDetector(
            dictionary=scattered_dict, min_marker_perimeter_rate=min_marker_perimeter_rate,
        )

        states: list[CamCalibState] = []
        rig_detections_by_camera: dict[str, list] = {}
        scattered_groups: dict[str, MarkerGroup] = {}

        for spec in specs:
            intr = _resolve_intrinsics(conn, spec.label, spec.camera_mode)
            img = _read_one_frame(spec.video_path, spec.frame_idx)
            states.append(CamCalibState(
                video_id=spec.label, label=spec.label,
                K=intr["K"], K_orig=intr["K_orig"], dist=intr["dist"], fisheye=intr["fisheye"],
                image=img,
            ))
            rig_dets = rig_detector.detect(img, video_id=spec.label, frame_idx=spec.frame_idx)
            rig_detections_by_camera[spec.label] = rig_dets
            scattered_dets = scattered_detector.detect(img, video_id=spec.label, frame_idx=spec.frame_idx)
            merge_detections_into_groups(
                scattered_dets, scattered_groups, size=tag_size, dictionary=scattered_dict
            )
            click.echo(f"{spec.label}: {len(rig_dets)} rig marker(s), "
                       f"{len(scattered_dets)} scattered tag detection(s)")

        cps = anchor_from_marker_rig(rig_detections_by_camera, rig_config)
        if not cps:
            raise click.ClickException(
                "Rig was not detected in any camera -- cannot anchor. Check --rig-dict and "
                "--min-marker-perimeter-rate, or that the rig is actually visible."
            )

        result = run_calibration(
            states, control_points=cps, marker_groups=list(scattered_groups.values()),
            cp_only=False,
        )
        if result.unsolved:
            click.echo(f"WARNING: {len(result.unsolved)} camera(s) unsolved: {result.unsolved}",
                       err=True)

        calib_id = write_extrinsics_to_db(result, conn, session_id, label_to_instance, method=method)

        if capture:
            capture_id = _resolve(conn, "captures", capture)
            set_capture_extrinsics(conn, capture_id, calib_id)

        # The rig defines the world frame for this solve, by construction --
        # its own anchor pose is the identity transform (see
        # anchor_from_marker_rig's docstring).
        upsert_scene_marker_body(
            conn, session_id, label=f"rig:{rig_config.rig_id}",
            R=np.eye(3), t=np.zeros(3), group_name=group_name,
            marker_body_definition_id=body_id, is_primary_anchor=True,
            source_extrinsic_calibration_id=calib_id,
        )

        n_tags_saved = 0
        if tag_size is not None:
            for marker_id, pose in result.marker_poses.items():
                R, _ = cv2.Rodrigues(pose.rvec)
                upsert_scene_marker_body(
                    conn, session_id, label=f"tag:{marker_id}", group_name=group_name,
                    R=R, t=pose.tvec,
                    marker_type="aruco", dictionary=scattered_dict, marker_id=marker_id,
                    marker_size=pose.size, source_extrinsic_calibration_id=calib_id,
                )
                n_tags_saved += 1
        elif scattered_groups:
            click.echo(
                f"NOTE: {len(scattered_groups)} scattered tag(s) detected but --tag-size not "
                f"given -- they helped the solve but were not persisted to scene_marker_bodies."
            )

    finally:
        conn.close()

    click.echo(f"\nextrinsic_calibration_id: {calib_id}")
    for s in result.cameras.values():
        if s.R is None:
            continue
        C = -s.R.T @ s.t.flatten()
        click.echo(f"  {s.label:20s}  ({C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f})")
    click.echo(f"scene_marker_bodies: 1 rig anchor + {n_tags_saved} tag(s) saved")


# ---------------------------------------------------------------------------
# reanchor
# ---------------------------------------------------------------------------


@extrinsics_group.command("reanchor")
@click.option("--session", "session_row", required=True, metavar="UUID", help="mocap_sessions.id")
@click.option("--camera", "cameras", multiple=True, required=True, metavar="SPEC",
              help="'label|camera_mode|video_path|frame_idx' (repeatable, one per camera)")
@click.option("--tag-dict", default="DICT_5X5_50", show_default=True)
@click.option("--min-marker-perimeter-rate", type=float, default=0.01, show_default=True)
@click.option("--method", default="reanchor", show_default=True)
@click.option("--capture", default=None, metavar="UUID",
              help="captures.id to link (sets extrinsic_calibration_id)")
@click.option("--name", "group_name", required=True, metavar="NAME",
              help="Re-anchor from this named group's markers (see 'anchor-rig --name' "
                   "and 'extrinsics scene-marker groups'). Required -- matches the GUI's "
                   "Load Markers… flow, which always picks a named config (see "
                   "extrinsics-ux-redesign.md, UX Phase 5).")
@click.pass_obj
def extrinsics_reanchor(
    obj: dict,
    session_row: str,
    cameras: tuple[str, ...],
    tag_dict: str,
    min_marker_perimeter_rate: float,
    method: str,
    capture: str | None,
    group_name: str,
) -> None:
    """Re-anchor a capture's cameras from previously-solved scattered tags.

    No physical rig needed -- reads this session's scene_marker_bodies
    rows with known geometry (dictionary/marker_id/marker_size set,
    i.e. tags a prior 'anchor-rig' or 'reanchor' run already solved and
    persisted) and reuses anchor_from_marker_rig completely unmodified:
    those tags' already-known world positions are just another
    MarkerRigConfig source (design doc section 9, Tier B).
    """
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session_row)

        old_factory = conn.row_factory
        conn.row_factory = sqlite3.Row
        try:
            tag_rows = conn.execute(
                "SELECT * FROM scene_marker_bodies WHERE session_id = ? "
                "AND marker_body_definition_id IS NULL AND dictionary = ? AND group_name = ? "
                "AND marker_id IS NOT NULL AND marker_size IS NOT NULL",
                (session_id, tag_dict, group_name),
            ).fetchall()
        finally:
            conn.row_factory = old_factory

        if not tag_rows:
            raise click.ClickException(
                f"No previously-solved scattered tags found for this session "
                f"(dictionary={tag_dict!r}, group={group_name!r}). Run 'anchor-rig' with "
                f"--tag-size first, or check 'extrinsics scene-marker groups' for the "
                f"right --name."
            )

        from app.setup.extrinsics_solver import marker_local_corners
        from app.setup.fiducial_markers import MarkerRigConfig

        marker_corners: dict[str, np.ndarray] = {}
        for row in tag_rows:
            R = np.array(struct.unpack("<9d", bytes(row["R"]))).reshape(3, 3)
            t = np.array(struct.unpack("<3d", bytes(row["t"])))
            local = marker_local_corners(row["marker_size"])
            marker_corners[row["marker_id"]] = (R @ local.T).T + t
        rig_config = MarkerRigConfig(rig_id="scene_tags", marker_corners=marker_corners)
        click.echo(f"Re-anchoring from {len(marker_corners)} previously-known tag(s): "
                   f"{sorted(marker_corners)}")

        specs = [_parse_camera_spec(c) for c in cameras]
        label_to_instance = _label_to_instance_id(conn)
        detector = ArucoDetector(
            dictionary=tag_dict, min_marker_perimeter_rate=min_marker_perimeter_rate,
        )

        states: list[CamCalibState] = []
        detections_by_camera: dict[str, list] = {}
        for spec in specs:
            intr = _resolve_intrinsics(conn, spec.label, spec.camera_mode)
            img = _read_one_frame(spec.video_path, spec.frame_idx)
            states.append(CamCalibState(
                video_id=spec.label, label=spec.label,
                K=intr["K"], K_orig=intr["K_orig"], dist=intr["dist"], fisheye=intr["fisheye"],
                image=img,
            ))
            all_dets = detector.detect(img, video_id=spec.label, frame_idx=spec.frame_idx)
            known_dets = [d for d in all_dets if d.marker_id in marker_corners]
            detections_by_camera[spec.label] = known_dets
            click.echo(f"{spec.label}: {len(known_dets)} known tag(s) seen "
                       f"(of {len(all_dets)} total {tag_dict} detections)")

        cps = anchor_from_marker_rig(detections_by_camera, rig_config)
        if not cps:
            raise click.ClickException("No known tag was detected in any camera -- cannot re-anchor.")

        result = run_calibration(states, control_points=cps, cp_only=False)
        if result.unsolved:
            click.echo(f"WARNING: {len(result.unsolved)} camera(s) unsolved: {result.unsolved}",
                       err=True)

        calib_id = write_extrinsics_to_db(result, conn, session_id, label_to_instance, method=method)

        if capture:
            capture_id = _resolve(conn, "captures", capture)
            set_capture_extrinsics(conn, capture_id, calib_id)

    finally:
        conn.close()

    click.echo(f"\nextrinsic_calibration_id: {calib_id}")
    for s in result.cameras.values():
        if s.R is None:
            continue
        C = -s.R.T @ s.t.flatten()
        click.echo(f"  {s.label:20s}  ({C[0]:+.3f}, {C[1]:+.3f}, {C[2]:+.3f})")


# ---------------------------------------------------------------------------
# scene-marker (view/prune scene_marker_bodies -- e.g. a portable rig's own
# anchor row once it's been physically removed, or a scattered tag whose
# position has moved -- see design doc section 9)
# ---------------------------------------------------------------------------


@extrinsics_group.group("scene-marker")
def scene_marker_group() -> None:
    """Inspect and prune this session's stored scene markers
    (scene_marker_bodies -- design doc section 9 Tier A/B anchors)."""


@scene_marker_group.command("list")
@click.option("--session", "session_row", required=True, metavar="UUID", help="mocap_sessions.id")
@click.pass_obj
def scene_marker_list(obj: dict, session_row: str) -> None:
    """List every scene marker stored for a session, across every group."""
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session_row)
        rows = list_scene_marker_bodies(conn, session_id)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No scene markers stored for this session.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=[
            "label", "group_name", "marker_body_definition_id", "marker_type", "dictionary",
            "marker_id", "marker_size", "is_primary_anchor", "updated_at",
        ],
        json_mode=obj["json_mode"],
    )


@scene_marker_group.command("groups")
@click.option("--session", "session_row", required=True, metavar="UUID", help="mocap_sessions.id")
@click.pass_obj
def scene_marker_groups(obj: dict, session_row: str) -> None:
    """List named scene-marker groups for a session (e.g. one per room),
    with a marker count and last-updated time -- for picking the right
    --name for 'reanchor'. 'anchor-rig'/'reanchor' both require --name
    (design doc, UX Phase 5); groups from before that requirement was
    added may still show an ungrouped ('') entry, prunable via
    'scene-marker delete'."""
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session_row)
        rows = list_scene_marker_group_names(conn, session_id)
    finally:
        conn.close()

    if not rows:
        if not obj["json_mode"]:
            click.echo("No named scene-marker groups stored for this session.")
        return

    print_table(
        [dict(r) for r in rows],
        columns=["group_name", "n_markers", "last_updated"],
        json_mode=obj["json_mode"],
    )


@scene_marker_group.command("delete")
@click.option("--session", "session_row", required=True, metavar="UUID", help="mocap_sessions.id")
@click.option("--group", "group_name", default=None, metavar="NAME",
              help="The marker's group (see 'scene-marker groups'). Omit for the "
                   "ungrouped default -- markers saved without --name.")
@click.argument("label", metavar="LABEL")
@click.pass_obj
def scene_marker_delete(obj: dict, session_row: str, group_name: str | None, label: str) -> None:
    """Delete one stored scene marker by its (group, label) (see
    'scene-marker list'/'scene-marker groups').

    For pruning stale entries -- e.g. a portable rig's own anchor row
    ("rig:<name>") once it's been physically removed from the scene, or a
    scattered tag ("tag:<id>") whose physical position has moved.
    """
    conn = _open_session_required(obj)
    try:
        session_id = _resolve(conn, "mocap_sessions", session_row)
        deleted = delete_scene_marker_body(conn, session_id, label, group_name=group_name)
    finally:
        conn.close()

    if not deleted:
        group_str = f" in group {group_name!r}" if group_name else ""
        raise click.ClickException(
            f"No scene marker found with label {label!r}{group_str} for this session."
        )
    click.echo(f"Deleted scene marker {label!r}.")
