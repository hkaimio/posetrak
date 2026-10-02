# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""calibrate.py — find where a person's markers sit on the skeleton, from a calibration trial.

Per time window the dot tracklets on the person are grouped into physical markers
(grouping.py) and each marker's 3D track is kept. A track is then tested against
the slots of the catalog: it can be slot S if it is rigid in the frame of S's parent
bone (little spread) and its mean position there lies near where S is expected
along the bone (the slot's ``along``). Tests from many windows are clustered into
candidate markers, each a bone and an offset in the bone's frame.

The markerless skeleton is only good to about a centimetre and neighbouring bones
barely move against each other in many poses, so a track alone does not name its
bone; the expected place along the bone is what separates, say, a knee marker from
a thigh one. Slots of one bone that share the same place (knee_lat and knee_med)
are not told apart here: they come out as an unordered set for review.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np

from scipy.optimize import linear_sum_assignment

from app.setup.extrinsics_solver import CamCalibState
from posetrak.calibration.session_cameras import load_camera_states, load_sync_table
from posetrak.markers.bones import BodyPose, fit_bones, marker_observations, marker_track_with_cameras
from posetrak.markers.catalog import MarkerModule, Slot
from posetrak.markers.reference import bone_frame, rest_bones
from posetrak.markers.grouping import build_groups
from posetrak.markers.tracklets import SubjectMasks, collect_tracklets, filter_tracklets

MAX_SPREAD_M = 0.03      # a track this loose in a bone's frame is not rigid on that bone
MAX_ANCHOR_DIST_M = 0.12  # nor a slot this far from where it is expected, unless the slot gives its own reach
MAX_NOMINAL_DIST_M = 0.08  # the same when the slot has a full placement on the reference body
MIN_VISIBLE_COSINE = -0.2  # a camera this far behind the way a marker faces cannot have seen it
MAX_VIOLATIONS = 0.5       # a track seen mostly from behind its normal is not that slot


def slots_of(modules: list[MarkerModule]) -> list[Slot]:
    """Every marker of the modules that has a coarse place along its bone."""
    return [slot for module in modules for slot in module.slots if slot.along is not None]


def bone_length(pose: BodyPose, joint: str) -> float:
    """Length of the bone of *joint*: its tip offset, or for a joint without one (the pelvis)
    the offset of its first child."""
    joints = {j.name: j for j in pose.layout.joints}
    info = joints.get(joint)
    if info is not None and info.bone_tip_offset is not None:
        return float(np.linalg.norm(info.bone_tip_offset))
    for child in joints.values():
        if child.parent == joint:
            return float(np.linalg.norm(child.offset))
    raise KeyError(f"no bone length for joint {joint!r}")


def anchor(pose: BodyPose, slot: Slot) -> np.ndarray:
    """Where the slot is expected in its parent bone's frame (a bone runs along local y)."""
    return np.array([0.0, slot.along * bone_length(pose, slot.parent_joint), 0.0])


@dataclass
class GroupTrack:
    """The 3D track of one marker (a group of tracklets) in one window."""

    window: tuple[float, float]
    members: list[tuple[str, int]]
    times: np.ndarray
    points: np.ndarray
    cameras: list | None = None     # the cameras that saw the marker at each sample, if known


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
    slots: list[Slot]
    rest_bones: dict | None = None      # the skeleton's bones in its rest pose, upright (reference.rest_bones)
    rest_frames: dict | None = None


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
    pose = BodyPose(conn, tracking_run_id)
    bones, frames = rest_bones(pose.layout, pose.layout.decode_state_blob(pose._blobs[0]))
    return CalibrationContext(
        conn, shot_id, detection_run_id, states, sync_table, svid_by_cam, masks, pose, slots_of(modules),
        bones, frames,
    )


def window_tracks(ctx: CalibrationContext, t0: float, t1: float, *, min_samples: int = 30) -> list[GroupTrack]:
    """Group the person's tracklets of [t0, t1] and triangulate each group."""
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
        times, points, cameras = marker_track_with_cameras(observations, ctx.states)
        if len(times) >= min_samples:
            out.append(GroupTrack((t0, t1), group["members"], times, points, cameras))
    return out


@dataclass
class SlotTest:
    """A track tested against one slot: how rigid it is on the slot's bone and where it sits."""

    slot: str
    joint: str
    spread_m: float
    anchor_dist_m: float
    offset: np.ndarray
    samples: int

    @property
    def cost(self) -> float:
        return self.spread_m / 0.01 + self.anchor_dist_m / 0.05


def nominal_placement(ctx: CalibrationContext, slot: Slot) -> tuple[np.ndarray, np.ndarray | None]:
    """Where the slot is expected, and the way it faces, in its parent bone's frame.

    From the slot's full placement on the reference body when it has one (along, lateral,
    anterior and normal, see reference.py), otherwise just its place along the bone.
    """
    if slot.lateral is None or ctx.rest_bones is None:
        return anchor(ctx.pose, slot), None
    head, tail = ctx.rest_bones[slot.parent_joint]
    frame = bone_frame(head, tail, np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    rotation = ctx.rest_frames[slot.parent_joint][:3, :3]
    length = float(np.linalg.norm(tail - head))
    offset = rotation.T @ (frame @ np.array([slot.along * length, slot.lateral, slot.anterior]))
    normal = rotation.T @ (frame @ np.asarray(slot.normal, float)) if slot.normal else None
    return offset, normal


def visibility_violations(ctx: CalibrationContext, track: GroupTrack, joint: str, normal_local: np.ndarray) -> float:
    """Fraction of the sightings of the marker made from a camera behind the way it faces."""
    bad = total = 0
    for t, X, cams in zip(track.times, track.points, track.cameras):
        frames = ctx.pose.joint_frames(float(t))
        if frames is None:
            continue
        n = frames[joint][:3, :3] @ normal_local
        for cam in cams:
            state = ctx.states[cam]
            to_camera = -state.R.T @ state.t.ravel() - X
            total += 1
            if (n @ to_camera) / (np.linalg.norm(n) * np.linalg.norm(to_camera)) < MIN_VISIBLE_COSINE:
                bad += 1
    return bad / total if total else 0.0


def test_track(ctx: CalibrationContext, track: GroupTrack) -> list[SlotTest]:
    """The slots the track could be, best first."""
    by_joint: dict[str, object] = {}
    tests = []
    for slot in ctx.slots:
        if slot.parent_joint not in by_joint:
            fits = fit_bones(track.times, track.points, ctx.pose, [slot.parent_joint])
            by_joint[slot.parent_joint] = fits[0] if fits else None
        fit = by_joint[slot.parent_joint]
        if fit is None or fit.spread_m > MAX_SPREAD_M:
            continue
        expected, normal = nominal_placement(ctx, slot)
        dist = float(np.linalg.norm(fit.offset - expected))
        if dist > (MAX_NOMINAL_DIST_M if normal is not None else (slot.reach or MAX_ANCHOR_DIST_M)):
            continue
        violations = 0.0
        if normal is not None and track.cameras:
            violations = visibility_violations(ctx, track, slot.parent_joint, normal)
            if violations > MAX_VIOLATIONS:
                continue
        # being seen now and then from behind the normal costs a little, like a position error
        tests.append(SlotTest(slot.name, slot.parent_joint, fit.spread_m, dist + 0.1 * violations, fit.offset, fit.n))
    return sorted(tests, key=lambda t: t.cost)


@dataclass
class MarkerCandidate:
    """Votes for one marker: a bone and an offset in its frame, and the slots it could be."""

    joint: str
    offset: np.ndarray
    slots: set[str]
    votes: int = 0
    samples: int = 0
    windows: set = field(default_factory=set)


def cluster_tests(tests: list[tuple[GroupTrack, list[SlotTest]]], offset_tol_m: float = 0.03) -> list[MarkerCandidate]:
    """Cluster the best bone of each track, over many windows, by bone and offset.

    A track whose slot tests name two bones with equal cost (a forearm and hand marker
    near the wrist) votes for the cheaper one. Largest support first.
    """
    clusters: list[MarkerCandidate] = []
    for track, slot_tests in tests:
        if not slot_tests:
            continue
        best = slot_tests[0]
        slots = {t.slot for t in slot_tests if t.joint == best.joint and t.cost <= best.cost * 1.5 + 0.5}
        for cand in clusters:
            if cand.joint == best.joint and np.linalg.norm(cand.offset - best.offset) <= offset_tol_m:
                total = cand.samples + best.samples
                cand.offset = (cand.offset * cand.samples + best.offset * best.samples) / total
                cand.samples = total
                cand.votes += 1
                cand.windows.add(track.window)
                cand.slots &= slots or cand.slots
                break
        else:
            clusters.append(MarkerCandidate(best.joint, best.offset.copy(), slots, 1, best.samples, {track.window}))
    return sorted(clusters, key=lambda c: (-len(c.windows), -c.samples))


@dataclass
class Chain:
    """A marker's 3D track and the windows it comes from."""

    windows: list[tuple[float, float]]
    members: set[tuple[str, int]]
    times: np.ndarray
    points: np.ndarray


@dataclass
class ChainFit:
    chain: Chain
    tests: list[SlotTest]       # the slots the track could be, best first


@dataclass
class Placement:
    """A slot's calibrated place: a bone and an offset, and how sure it is."""

    slot: str
    joint: str
    offset: np.ndarray
    spread_m: float
    samples: int
    chains: int
    runner_up: str | None = None       # the next best slot for the same markers
    runner_up_margin: float = 0.0      # its extra cost; small means the two are hard to tell apart


def fit_tracks(ctx: CalibrationContext, tracks: list[GroupTrack]) -> list[ChainFit]:
    """Test every track against the slots."""
    return [
        ChainFit(Chain([t.window], set(t.members), t.times, t.points), test_track(ctx, t)) for t in tracks
    ]


def assign_slots(ctx: CalibrationContext, fits: list[ChainFit], offset_tol_m: float = 0.03,
                 unassigned_cost: float = 8.0, extra_cost=None) -> tuple[list[Placement], list[MarkerCandidate]]:
    """Give each slot at most one cluster of chains, and each cluster at most one slot.

    Chains are clustered by the bone and offset of their best slot; the clusters are then
    matched to slots by cost (rigidity on the slot's bone plus distance from its expected
    place), best overall, so that one marker cannot fill two slots and two markers on one
    bone take different slots. *extra_cost*, if given, is called as ``extra_cost(slot, joint,
    offset)`` and added to a cluster's cost for a slot (see naming.direction_penalty). Returns
    the placements and the clusters left without a slot.
    """
    clusters: list[tuple[MarkerCandidate, dict[str, SlotTest], list[float]]] = []
    for cf in fits:
        if not cf.tests:
            continue
        best = cf.tests[0]
        for cand, tests, spreads in clusters:
            if cand.joint == best.joint and np.linalg.norm(cand.offset - best.offset) <= offset_tol_m:
                total = cand.samples + best.samples
                cand.offset = (cand.offset * cand.samples + best.offset * best.samples) / total
                cand.samples = total
                cand.votes += 1
                cand.windows.update(cf.chain.windows)
                spreads.append(best.spread_m)
                for t in cf.tests:
                    if t.slot not in tests or t.cost < tests[t.slot].cost:
                        tests[t.slot] = t
                break
        else:
            clusters.append((MarkerCandidate(best.joint, best.offset.copy(), set(), 1, best.samples,
                                              set(cf.chain.windows)),
                             {t.slot: t for t in cf.tests}, [best.spread_m]))
    slot_names = [s.name for s in ctx.slots]
    n_c, n_s = len(clusters), len(slot_names)
    big = 1e6
    cost = np.full((n_c, n_s + n_c), big)
    for i, (cand, tests, _) in enumerate(clusters):
        for j, name in enumerate(slot_names):
            if name in tests:
                t = tests[name]
                extra = extra_cost(ctx.slots[j], t.joint, t.offset) if extra_cost else 0.0
                cost[i, j] = (t.cost + extra) / (1.0 + 0.3 * np.log10(max(cand.samples, 100) / 100.0))
        cost[i, n_s + i] = unassigned_cost            # left without a slot
    rows, cols = linear_sum_assignment(cost)
    placements, leftover = [], []
    for i, j in zip(rows, cols):
        cand, tests, spreads = clusters[i]
        if j >= n_s:
            leftover.append(cand)
            continue
        name = slot_names[j]
        others = sorted((t for t in tests.values() if t.slot != name), key=lambda t: t.cost)
        placements.append(Placement(
            name, tests[name].joint, tests[name].offset, tests[name].spread_m, cand.samples, cand.votes,
            others[0].slot if others else None,
            (others[0].cost - tests[name].cost) if others else 99.0,
        ))
    return sorted(placements, key=lambda p: p.slot), sorted(leftover, key=lambda c: -c.samples)
