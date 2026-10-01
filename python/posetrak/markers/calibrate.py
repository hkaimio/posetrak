# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate.py — find where a person's markers sit on the skeleton, from a calibration trial.

Per time window the dot tracklets on the person are grouped into physical markers
(grouping.py), each marker's 3D track is fitted to the bones the catalog allows
(bones.py), and a marker whose track is clearly rigid in one bone's frame votes for
that bone and offset. Votes from many windows are then clustered into candidate
markers, each a bone and an offset in the bone's frame.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from app.setup.extrinsics_solver import CamCalibState
from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
from posetrak.markers.bones import BodyPose, BoneFit, fit_bones, marker_observations, marker_track
from posetrak.markers.catalog import MarkerModule
from posetrak.markers.grouping import build_groups
from posetrak.markers.tracklets import SubjectMasks, collect_tracklets, filter_tracklets


def parent_joints(modules: list[MarkerModule]) -> list[str]:
    """The distinct joints the modules' markers ride on."""
    return sorted({slot.parent_joint for module in modules for slot in module.slots})


@dataclass
class CalibrationContext:
    conn: sqlite3.Connection
    shot_id: str
    detection_run_id: str
    states: dict[str, CamCalibState]
    sync_table: object
    svid_by_cam: dict[str, str]
    masks: SubjectMasks
    pose: BodyPose
    parents: list[str]


def open_context(
    conn: sqlite3.Connection, shot_id: str, detection_run_id: str, tracking_run_id: str, seg_run_id: str,
    mask_label: int, modules: list[MarkerModule], mask_margin_px: float = 40.0,
) -> CalibrationContext:
    """Everything the per-window steps need, for one capture."""
    states = load_camera_states(conn, shot_id)
    sync_table, svid_by_cam = load_sync_table(conn, shot_id)
    native = {
        r["id"]: (r["w"], r["h"]) for r in conn.execute(
            "SELECT cv.id, COALESCE(ic.image_width, cm.width_px) AS w, COALESCE(ic.image_height, cm.height_px) AS h "
            "FROM capture_videos cv LEFT JOIN camera_modes cm ON cm.id = cv.camera_mode_id "
            "LEFT JOIN intrinsics_calibrations ic ON ic.id = cv.intrinsics_calibration_id WHERE cv.shot_id = ?",
            (shot_id,),
        )
    }
    masks = SubjectMasks(conn, seg_run_id, mask_label, mask_margin_px, native)
    return CalibrationContext(
        conn, shot_id, detection_run_id, states, sync_table, svid_by_cam, masks,
        BodyPose(conn, tracking_run_id), parent_joints(modules),
    )


@dataclass
class GroupFit:
    """One marker (a group of tracklets) in one window, fitted to the allowed bones."""

    window: tuple[float, float]
    members: list[tuple[str, int]]
    n_samples: int
    fits: list[BoneFit] = field(default_factory=list)   # best (least spread) first

    def decisive(self, max_spread_m: float = 0.02, margin: float = 1.4) -> BoneFit | None:
        """The best bone if its spread is small and clearly below the runner-up's."""
        if not self.fits or self.fits[0].spread_m > max_spread_m:
            return None
        if len(self.fits) > 1 and self.fits[1].spread_m < margin * self.fits[0].spread_m:
            return None
        return self.fits[0]


def fit_window(ctx: CalibrationContext, t0: float, t1: float, *, min_samples: int = 30) -> list[GroupFit]:
    """Group the person's tracklets of [t0, t1] and fit each group to the allowed bones."""
    filtered = {}
    for cam_id, svid in ctx.svid_by_cam.items():
        if cam_id not in ctx.states:
            continue
        lo, hi = ctx.sync_table.lookup(t0, svid), ctx.sync_table.lookup(t1, svid)
        if lo is None or hi is None:
            continue
        tracklets = collect_tracklets(ctx.conn, ctx.detection_run_id, svid, lo, hi)
        filtered[cam_id], _ = filter_tracklets(tracklets, svid, ctx.masks)
    groups, _ = build_groups(filtered, ctx.states, ctx.sync_table, ctx.svid_by_cam, 5, 3.0, 6.0, 1.5)
    out = []
    for group in groups:
        observations = marker_observations(group["members"], filtered, ctx.sync_table, ctx.svid_by_cam)
        times, points = marker_track(observations, ctx.states)
        if len(times) < min_samples:
            continue
        out.append(GroupFit((t0, t1), group["members"], len(times),
                            fit_bones(times, points, ctx.pose, ctx.parents)))
    return out


@dataclass
class MarkerCandidate:
    """Votes for one marker: a bone and an offset in its frame."""

    joint: str
    offset: np.ndarray
    votes: int = 0
    samples: int = 0
    windows: set = field(default_factory=set)


def cluster_votes(group_fits: list[GroupFit], offset_tol_m: float = 0.03) -> list[MarkerCandidate]:
    """Cluster the decisive fits of many windows by bone and offset. Largest support first."""
    clusters: list[MarkerCandidate] = []
    for gf in group_fits:
        best = gf.decisive()
        if best is None:
            continue
        for cand in clusters:
            if cand.joint == best.joint and np.linalg.norm(cand.offset - best.offset) <= offset_tol_m:
                total = cand.samples + gf.n_samples
                cand.offset = (cand.offset * cand.samples + best.offset * gf.n_samples) / total
                cand.samples = total
                cand.votes += 1
                cand.windows.add(gf.window)
                break
        else:
            clusters.append(MarkerCandidate(best.joint, best.offset.copy(), 1, gf.n_samples, {gf.window}))
    return sorted(clusters, key=lambda c: (-len(c.windows), -c.samples))
