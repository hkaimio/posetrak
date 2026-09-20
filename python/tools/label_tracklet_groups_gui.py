# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""label_tracklet_groups_gui.py — P-B, step B3 (redesign doc §4): manual
tracklet-group -> slot assignment. Consumes `build_tracklet_groups.py`'s
output (B2) and lets a human confirm, reject, split, or merge each group
-- a handful of actions for a whole clip, not per-frame labeling.

For each group, independently re-derives (not stored in B2's own JSON,
to keep that file small):
  * its fused 3D trajectory -- N-view triangulation at every instant
    where 2+ member cameras have a detection;
  * ranked slot suggestions -- mean 3D distance from that trajectory to
    every catalog slot's FK-predicted trajectory over the same instants,
    ascending (a suggestion, never authoritative -- B3's whole point is
    a human confirms it);
  * a couple of sample frames per member camera with the tracklet's own
    dot circled, so the operator can actually see what they're
    confirming rather than trust numbers alone.

Interaction:
    - group list (left), sorted by lifetime (longest first); each row
      shows member count, total distinct frames, flags from B2, and the
      top slot suggestion. "Show only unassigned" filters the list to
      groups with no recorded assignment/rejection yet.
    - selecting a group renders a grid (one row per member, one column
      per sampled instant) + a ranked-suggestion table. Sample instants
      are chosen to guarantee every member gets at least one real frame
      shown (its own median instant), plus an even spread across the
      group's whole time span (so an identity switch *within* one
      member's own tracklet is visible, not hidden between two samples)
      -- see `GroupAnalyzer.analyze()`. A cell whose member has no real
      detection near that instant says so plainly (with the actual time
      gap) rather than silently substituting a distant frame.
    - Every action below that changes the current group's own membership
      (Discard / Split / Assign-as-new-group) re-selects that *same*
      group afterward, with the acted-on members now gone, rather than
      jumping elsewhere after the list re-sorts (2026-09-11, Harri --
      a run of several such actions on one contaminated group was hard
      to follow when the view kept jumping away). Assign/Reject on the
      whole current group behave the same way, *except* when "Show only
      unassigned" is on and the group just became assigned/rejected --
      then it naturally drops out of the filtered view.
    - Assign: commit the chosen slot (combo box, defaults to the top
      suggestion; includes "unlabeled", and two "reject: ..." pseudo-slots
      that record a rejection with that specific reason instead of a slot)
    - Reject (unspecified): not a body marker, no specific reason recorded
      -- prefer the Slot combo's two "reject: ..." entries + Assign when
      the reason is known
    - Assign checked as new group: check one or more member rows, pick a
      slot (or a "reject: ..." reason) in the combo, then click this --
      splits the checked members into their own group (skipping the split
      step entirely if *every* member is checked) and immediately applies
      that slot/rejection. The one-tracklet-in-a-contaminated-group case
      Harri asked for (2026-09-11): previously required Split, reselect
      the new group in the list, then Assign, as three separate steps.
    - Discard checked (noise) / Split checked into new group: the same
      one-step mechanism, fixed to "reject: noise" / not finalized
      (just moved to a fresh group_id) respectively -- "this subset is
      noise" and "this subset is a different real marker, decide later"
    - Split member's tracklet (scrub)...: for a single checked member
      whose own tracklet drifted onto a *different* physical marker
      mid-track (a real limitation in the per-camera dot-tracklet linker
      as it stood before 2026-09-11 -- it linked by a fixed pixel-distance
      gate with no identity/appearance check, so it could hop between two
      real markers that pass near each other; fixed in `MotionGatedLinker`,
      see status.md, but old detection runs made with the earlier linker
      can still show this) --
      opens `_ScrubSplitDialog`, which steps frame-by-frame through that
      member's own real frames (not just the main grid's coarse
      sample-time columns) showing the actual cropped image at each step,
      seeded at an auto-suggested transition frame from per-frame FK-slot
      classification (`GroupAnalyzer.classify_member_frames`). Also
      reachable by right-clicking any shown thumbnail ("Scrub & split
      near here...", seeded at that exact frame; "Split before this
      frame" commits immediately without scrubbing, for when the shown
      frame is already precise enough). Produces two sub-range members of
      the *same* tracklet_id (frames before/after the cut), which can
      then be split into different groups.
    - Merge into...: fold this group's members into another group_id
      (the linker/B2 missed a link, e.g. only ever co-visible from
      non-overlapping camera pairs)

Output: tracklet_group_assignments.json --
    {group_id: {"slot": name | null, "rejected": bool,
                "reject_reason": "noise" | "other_subject_or_prop" (optional,
                                  only present when rejected and a reason
                                  was recorded -- absent, not null, otherwise),
                "members": [[camera_label, tracklet_id], ...]}}
A member is normally a 2-element [camera_label, tracklet_id] pair (the
whole tracklet); a manual mid-tracklet split produces a 4-element
[camera_label, tracklet_id, frame_lo, frame_hi] entry instead, meaning
"this tracklet restricted to video frames in [frame_lo, frame_hi)"
(either bound may be null, meaning unbounded on that side).

Usage:
    python tools/label_tracklet_groups_gui.py \\
        --groups scratch/dot_ground_truth/tracklet_groups.json \\
        --output scratch/dot_ground_truth/tracklet_group_assignments.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.setup.extrinsics_solver import _proj_matrix, _undistort_pts  # noqa: E402
from posetrak.detection.frame_source import iter_frames  # noqa: E402
from posetrak.markers.catalog import catalog_module  # noqa: E402
from tools.build_tracklet_groups import FKPredictor, collect_tracklets  # noqa: E402
from tools.calibrate_rigid_marker_body import load_camera_states, load_sync_table  # noqa: E402
from tools.prototype_fk_marker_prediction import _DEFAULT_TRIAL  # noqa: E402
from tools.prototype_multi_camera_fusion import triangulate_multiview  # noqa: E402

_THUMB_W, _THUMB_H = 340, 260
# A member's frame shown for a "t=..." column must be within this many
# seconds of the real requested instant, or it isn't the same moment --
# see GroupAnalyzer.sample_image(). Also used to merge near-duplicate
# candidate sample instants in analyze()'s time-selection.
_MAX_SAMPLE_GAP_S = 0.15
# Sample-instant selection (found 2026-09-11: the old version only ever
# sampled instants where >=2 members' triangulation succeeded, so a
# member with no such overlap silently never got a frame shown at all --
# indistinguishable from a bug to a reviewer). Now: one instant at each
# member's own median real time (guarantees every member is shown at
# least once), plus an even spread across the group's whole real time
# span (surfaces an identity switch *within* one member's own tracklet),
# merged and capped to this many rendered columns.
_SPREAD_SAMPLES = 5
_MAX_SAMPLE_TIMES = 8
_UNLABELED = "unlabeled"
# Pseudo-slot combo entries -- picking one of these and clicking Assign (or
# "Assign checked as new group") records a rejection with a specific reason
# instead of a real slot. Distinguishing these (2026-09-11, Harri) matters:
# a marker on another subject or a prop is a real, correctly-behaving
# detection just not for this capture's tracked person -- worth keeping
# separate from actual noise (glare, a spurious blob) for any future
# quality accounting or reuse, rather than dumping both into one bucket.
_REJECT_NOISE = "reject: noise"
_REJECT_OTHER_SUBJECT = "reject: other subject / prop"
_REJECT_REASONS = {_REJECT_NOISE: "noise", _REJECT_OTHER_SUBJECT: "other_subject_or_prop"}

# Real anatomical slot -> P-A trial probe name, from eval_fk_prediction.py's
# empirical majority-vote run (status.md, 2026-09-11). _DEFAULT_TRIAL's own
# keys are unlabelled exploratory probes (a_hip_R, k_Xp_R, ...); this
# translates so suggestions are offered under names the assignment combo box
# (and a human) actually recognises. Several real slots share one probe
# (ankle_lat_L/ankle_med_L both -> a_ankle_L) -- an honest reflection that
# the trial catalog doesn't yet cleanly separate every same-joint offset,
# not a bug here; it means those slots' suggestions won't distinguish
# medial from lateral until the catalog is refined with real offsets.
_SLOT_TO_PROBE = {
    "ankle_lat_L": "a_ankle_L", "ankle_med_L": "a_ankle_L",
    "ankle_lat_R": "a_ankle_R", "ankle_med_R": "a_ankle_R",
    "heel_L": "n_Zp_L", "heel_R": "n_Zp_R",
    "hip_L": "a_hip_L", "hip_R": "a_hip_R",
    "knee_front_L": "k_Zm_L", "knee_front_R": "k_Zm_R",
    "knee_lat_L": "a_knee_L", "knee_med_R": "a_knee_R",
    "knee_lat_R": "k_Xm_R", "knee_med_L": "k_Xp_L",
    "toe_L": "a_toe_L", "toe_R": "a_toe_R",
}


def _parse_member(entry: list) -> tuple[str, int, int | None, int | None]:
    """A group's own [camera_label, tracklet_id] pairs, or a mid-tracklet
    split's [camera_label, tracklet_id, frame_lo, frame_hi] -- normalize
    both to (label, tid, lo, hi), lo/hi None meaning unbounded/whole
    tracklet."""
    if len(entry) == 2:
        label, tid = entry
        return label, tid, None, None
    label, tid, lo, hi = entry
    return label, tid, lo, hi


def _member_json(label: str, tid: int, lo: int | None, hi: int | None) -> list:
    if lo is None and hi is None:
        return [label, tid]
    return [label, tid, lo, hi]


def _suggest_split_frame(classified: list[tuple[int, float, str | None, float]]) -> int | None:
    """Given one member's own per-frame slot classification (see
    `GroupAnalyzer.classify_member_frames`), find the single frame that
    best separates it into two differing-identity halves: the cut index
    maximizing (frames before agreeing with the before-half's own
    majority slot) + (frames after agreeing with the after-half's own
    majority slot), only counted when those two majority slots actually
    differ. Returns None if fewer than 10 frames or no such split found
    (e.g. the tracklet consistently classifies as one slot throughout)."""
    labels = [c[2] for c in classified]
    frames = [c[0] for c in classified]
    n = len(labels)
    if n < 10:
        return None
    best_i, best_score = None, -1
    margin = max(3, n // 20)
    for i in range(margin, n - margin):
        left = [l for l in labels[:i] if l is not None]
        right = [l for l in labels[i:] if l is not None]
        if not left or not right:
            continue
        left_mode, left_n = Counter(left).most_common(1)[0]
        right_mode, right_n = Counter(right).most_common(1)[0]
        if left_mode == right_mode:
            continue
        score = left_n + right_n
        if score > best_score:
            best_score, best_i = score, i
    return frames[best_i] if best_i is not None else None


def _bgr_to_qpixmap(img: np.ndarray, mark: tuple[float, float] | None = None) -> QtGui.QPixmap:
    out = img
    if mark is not None:
        out = img.copy()
        p = (int(round(mark[0])), int(round(mark[1])))
        # Circle only (2026-09-12, Harri): the cross's own lines passed
        # straight through the center, hiding exactly the pixel content
        # (a real dot vs. noise) the reviewer needs to see there. A ring
        # around the point marks it without covering it.
        cv2.circle(out, p, 14, (0, 255, 0), 2)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    qim = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qim)


class GroupAnalyzer:
    """Re-derives everything B3 needs for one group from the DB -- not
    stored in B2's own (deliberately small) JSON output."""

    def __init__(self, conn, states, sync_table, svid_by_cam, label_to_cam, fkp: FKPredictor,
                 cache_dir: Path | None = None):
        self.conn = conn
        self.states = states
        self.sync_table = sync_table
        self.svid_by_cam = svid_by_cam
        self.label_to_cam = label_to_cam
        self.fkp = fkp
        self.cache_dir = cache_dir
        self._member_frames_cache: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}

    def member_frames(
        self, cam_id: str, tid: int, meta: dict, lo_frame: int | None = None, hi_frame: int | None = None,
    ) -> dict[int, tuple[float, float]]:
        """The tracklet's own frames, optionally restricted to
        [lo_frame, hi_frame) -- a manual mid-tracklet split (see
        `_MainWindow._split_member_tracklet`). The underlying per-tracklet
        DB read is cached on the *whole* tracklet id, so requesting
        several different sub-ranges of the same tracklet costs one DB
        read, not one per range."""
        key = (cam_id, tid)
        if key not in self._member_frames_cache:
            svid = self.svid_by_cam[cam_id]
            lo = self.sync_table.lookup(meta["start_time"], svid)
            hi = self.sync_table.lookup(meta["end_time"], svid)
            all_tracklets = collect_tracklets(self.conn, meta["detection_run"], svid, lo, hi)
            self._member_frames_cache[key] = all_tracklets.get(tid, {})
        frames = self._member_frames_cache[key]
        if lo_frame is not None or hi_frame is not None:
            frames = {
                vf: xy for vf, xy in frames.items()
                if (lo_frame is None or vf >= lo_frame) and (hi_frame is None or vf < hi_frame)
            }
        return frames

    def analyze(self, group: dict, meta: dict) -> dict:
        members = [
            (self.label_to_cam[label], tid, lo, hi)
            for label, tid, lo, hi in (_parse_member(e) for e in group["members"])
        ]
        frames_by_member = {m: self.member_frames(m[0], m[1], meta, m[2], m[3]) for m in members}

        # anchor on whichever member has the most frames, walk its own frame
        # timeline, and at each instant gather whichever OTHER members also
        # have a detection at that same global time (via the sync table) --
        # exactly the "shared frames" logic build_tracklet_groups.py itself uses.
        anchor = max(members, key=lambda m: len(frames_by_member[m]))
        anchor_svid = self.svid_by_cam[anchor[0]]

        times: list[float] = []
        fused_xyz: list[np.ndarray] = []
        for vf, xy in frames_by_member[anchor].items():
            t = self.sync_table.frame_to_global_time(vf, anchor_svid)
            if t is None:
                continue
            views_P, views_pt = [], []
            for m, frames in frames_by_member.items():
                cam_id = m[0]
                if m == anchor:
                    fidx = vf
                else:
                    fidx = self.sync_table.lookup(t, self.svid_by_cam[cam_id])
                    if fidx is None or fidx not in frames:
                        continue
                px, py = frames[fidx]
                pt = _undistort_pts(np.array([[px, py]]), self.states[cam_id])[0]
                views_P.append(_proj_matrix(self.states[cam_id]))
                views_pt.append((float(pt[0]), float(pt[1])))
            if len(views_P) >= 2:
                xyz = triangulate_multiview(views_P, views_pt)
                if xyz is not None:
                    times.append(t)
                    fused_xyz.append(xyz)

        suggestions: list[tuple[str, float]] = []
        if fused_xyz:
            per_slot_dist: dict[str, list[float]] = {}
            for t, xyz in zip(times, fused_xyz):
                world = self.fkp.predicted_world(t)
                for slot_name, probe_name in _SLOT_TO_PROBE.items():
                    pos = world.get(probe_name)
                    if pos is not None:
                        per_slot_dist.setdefault(slot_name, []).append(float(np.linalg.norm(pos - xyz)))
            suggestions = sorted(
                ((name, float(np.mean(d))) for name, d in per_slot_dist.items() if len(d) >= max(3, 0.5 * len(times))),
                key=lambda x: x[1],
            )

        # Sample-instant selection: one at every member's own median real
        # time (guarantees each member is shown at least once, even one
        # that never overlaps enough to triangulate with anything else),
        # plus an even spread across the group's whole real time span (so
        # an identity switch *within* one member's own tracklet -- e.g.
        # heel_L drifting onto heel_R mid-track -- lands in a rendered
        # column instead of being averaged away between two samples).
        member_times: dict[tuple, list[float]] = {}
        for m, frames in frames_by_member.items():
            svid = self.svid_by_cam[m[0]]
            ts = sorted(t for vf in frames for t in [self.sync_table.frame_to_global_time(vf, svid)] if t is not None)
            if ts:
                member_times[m] = ts

        sample_times: list[float] = []
        if member_times:
            candidates = {ts[len(ts) // 2] for ts in member_times.values()}
            glo = min(ts[0] for ts in member_times.values())
            ghi = max(ts[-1] for ts in member_times.values())
            candidates.update(glo + frac * (ghi - glo) for frac in np.linspace(0, 1, _SPREAD_SAMPLES))
            for t in sorted(candidates):
                if not sample_times or t - sample_times[-1] > _MAX_SAMPLE_GAP_S:
                    sample_times.append(t)
            if len(sample_times) > _MAX_SAMPLE_TIMES:
                idx = np.linspace(0, len(sample_times) - 1, _MAX_SAMPLE_TIMES).astype(int)
                sample_times = sorted({sample_times[i] for i in idx})

        return {
            "members": members, "frames_by_member": frames_by_member,
            "n_frames_total": sum(len(f) for f in frames_by_member.values()),
            "suggestions": suggestions, "sample_times": sample_times,
        }

    def classify_member_frames(
        self, cam_id: str, tid: int, frames: dict[int, tuple[float, float]],
    ) -> list[tuple[int, float, str | None, float]]:
        """For each of this member's OWN frames, the real slot whose
        FK-predicted 2D projection (same camera, no triangulation needed)
        is nearest to the tracklet's actual pixel position. Used to
        locate exactly where a tracklet's identity drifts from one
        physical marker to another (2026-09-11: the pre-fix per-camera
        dot-tracklet linker linked by a fixed pixel-distance gate with no
        appearance/identity check, so it could hop between two real
        markers that pass near each other -- confirmed root cause of a
        heel_L -> heel_R drift Harri found in gopro13_02#287; fixed in
        `MotionGatedLinker`, but useful on detection runs made before
        that fix).

        Returns [(video_frame, global_time, best_slot_or_None, dist_px),
        ...], sorted by frame."""
        svid = self.svid_by_cam[cam_id]
        out = []
        for vf in sorted(frames):
            t = self.sync_table.frame_to_global_time(vf, svid)
            if t is None:
                continue
            px, py = frames[vf]
            proj = self.fkp.predicted_in_camera(t, cam_id)
            best_slot, best_d = None, None
            for slot_name, probe_name in _SLOT_TO_PROBE.items():
                pos = proj.get(probe_name)
                if pos is None:
                    continue
                d = math.hypot(float(pos[0]) - px, float(pos[1]) - py)
                if best_d is None or d < best_d:
                    best_slot, best_d = slot_name, d
            out.append((vf, t, best_slot, best_d))
        return out

    def _cache_path(self, svid: str, tid: int, fidx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / svid / f"{tid}_{fidx}.png"

    def frame_image(self, cam_id: str, tid: int, fidx: int, frames: dict) -> QtGui.QPixmap | None:
        """Decode+crop+mark one *specific* frame this tracklet actually has
        a detection on (`fidx` must be a key of `frames`) -- no time-lookup
        or gap logic, unlike `sample_image`. Used directly by the scrub-
        and-split dialog, which steps through a tracklet's own real frames
        one at a time rather than through synced-instant columns. Decoded
        crops are cached to disk (`self.cache_dir`, keyed by camera +
        tracklet + frame) since re-decoding a video seek on every group
        switch (or every scrub step) was the dominant cost of browsing."""
        svid = self.svid_by_cam[cam_id]
        cache_path = self._cache_path(svid, tid, fidx)
        if cache_path is not None and cache_path.exists():
            pm = QtGui.QPixmap(str(cache_path))
            if not pm.isNull():
                return pm

        file_path = self.conn.execute(
            "SELECT file_path FROM capture_videos WHERE id = ?", (svid,)
        ).fetchone()[0]
        img = None
        for _, dec in iter_frames(file_path, fidx, fidx + 1):
            img = dec
            break
        if img is None:
            return None
        px, py = frames[fidx]
        x0, y0 = max(0, int(px - _THUMB_W // 2)), max(0, int(py - _THUMB_H // 2))
        crop = img[y0:y0 + _THUMB_H, x0:x0 + _THUMB_W]
        pm = _bgr_to_qpixmap(crop, mark=(px - x0, py - y0))
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pm.save(str(cache_path), "PNG")
        return pm

    def sample_image(
        self, cam_id: str, tid: int, t: float, meta: dict, frames: dict
    ) -> tuple[QtGui.QPixmap | None, float | None, int | None]:
        """Returns (pixmap, gap_s, fidx). gap_s is the real time distance
        between the requested instant `t` and the frame actually shown --
        None means an (approximately) exact match, otherwise the caller
        should make the mismatch visible rather than silently rendering a
        different moment as if it were `t` (found 2026-09-11: a member
        whose own tracklet doesn't span `t` at all -- e.g. two temporal
        fragments of one physical marker sitting in the same group -- used
        to fall back to its single nearest frame with no distance bound,
        which for a multi-second gap showed a visibly different real
        moment, e.g. a different footedness, right in the "same instant"
        column). fidx is the actual video frame shown (None if nothing
        was), returned so the caller can offer "split before this frame"."""
        svid = self.svid_by_cam[cam_id]
        fidx = self.sync_table.lookup(t, svid)
        if fidx is None or not frames:
            return None, None, None
        # nearest frame this tracklet actually has a detection on, near fidx
        gap_s = None
        if fidx not in frames:
            fidx = min(frames, key=lambda f: abs(f - fidx))
            actual_t = self.sync_table.frame_to_global_time(fidx, svid)
            if actual_t is None:
                return None, None, None
            gap_s = abs(actual_t - t)
            if gap_s > _MAX_SAMPLE_GAP_S:
                return None, gap_s, None
        pm = self.frame_image(cam_id, tid, fidx, frames)
        if pm is None:
            return None, gap_s, None
        return pm, gap_s, fidx


class _ThumbLabel(QtWidgets.QLabel):
    """A rendered sample-frame cell that knows which (member, frame) it
    shows, so it can offer a split starting point on right-click. "Split
    before this frame" commits immediately at the exact frame shown;
    "Scrub & split near here..." opens `_ScrubSplitDialog` centered on it
    -- the grid's own columns are coarse (a handful of synced instants),
    so a real identity-switch frame usually isn't one of them exactly."""

    def __init__(self, cam_id: str, tid: int, lo: int | None, hi: int | None, fidx: int,
                 on_split, on_scrub) -> None:
        super().__init__()
        self._member = (cam_id, tid, lo, hi)
        self._fidx = fidx
        self._on_split = on_split
        self._on_scrub = on_scrub

    def contextMenuEvent(self, ev: QtGui.QContextMenuEvent) -> None:
        menu = QtWidgets.QMenu(self)
        act_split = menu.addAction(f"Split before frame {self._fidx}")
        act_scrub = menu.addAction("Scrub & split near here...")
        chosen = menu.exec(ev.globalPos())
        if chosen is act_split:
            self._on_split(*self._member, self._fidx)
        elif chosen is act_scrub:
            self._on_scrub(*self._member, self._fidx)


class _ScrubSplitDialog(QtWidgets.QDialog):
    """Steps frame-by-frame through one member's own tracklet (not just
    the main grid's coarse sample-time columns) so the operator can *see*
    where an identity switch happens before splitting there, rather than
    guessing a frame number blind (2026-09-11 feedback: the main grid
    shows timestamps, not frame numbers, and a typed-number dialog gave no
    way to actually look at the frames in between)."""

    def __init__(self, analyzer: GroupAnalyzer, sync_table, cam_id: str, tid: int,
                 frames: dict[int, tuple[float, float]], start_fidx: int | None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.analyzer = analyzer
        self.sync_table = sync_table
        self.cam_id = cam_id
        self.tid = tid
        self.frames = frames
        self.frame_list = sorted(frames)
        self.svid = analyzer.svid_by_cam[cam_id]
        self.result_cut_frame: int | None = None

        self.setWindowTitle(f"Scrub & split -- tracklet #{tid}")
        layout = QtWidgets.QVBoxLayout(self)

        self.img_label = QtWidgets.QLabel()
        self.img_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_label.setMinimumSize(_THUMB_W, _THUMB_H)
        layout.addWidget(self.img_label)

        self.info_label = QtWidgets.QLabel()
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.frame_list) - 1))
        self.slider.valueChanged.connect(self._render)
        layout.addWidget(self.slider)

        nav = QtWidgets.QHBoxLayout()
        for text, delta in [("<< 10", -10), ("< 1", -1), ("1 >", 1), ("10 >>", 10)]:
            btn = QtWidgets.QPushButton(text)
            btn.clicked.connect(lambda _checked=False, d=delta: self.slider.setValue(
                max(0, min(self.slider.maximum(), self.slider.value() + d))))
            nav.addWidget(btn)
        layout.addLayout(nav)

        btns = QtWidgets.QHBoxLayout()
        split_btn = QtWidgets.QPushButton("Split before this frame")
        split_btn.clicked.connect(self._accept_split)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addWidget(split_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

        start_idx = 0
        if self.frame_list:
            if start_fidx is not None:
                start_idx = min(range(len(self.frame_list)), key=lambda i: abs(self.frame_list[i] - start_fidx))
            else:
                start_idx = len(self.frame_list) // 2
        self.slider.setValue(start_idx)
        self._render(start_idx)

    def _render(self, idx: int) -> None:
        if not self.frame_list:
            self.info_label.setText("(no frames)")
            return
        fidx = self.frame_list[idx]
        t = self.sync_table.frame_to_global_time(fidx, self.svid)
        pm = self.analyzer.frame_image(self.cam_id, self.tid, fidx, self.frames)
        if pm is not None:
            self.img_label.setPixmap(pm)
        else:
            self.img_label.setText("(decode failed)")
        t_str = f"{t:.3f}s" if t is not None else "?"
        self.info_label.setText(
            f"video_frame {fidx}   |   t = {t_str}   |   {idx + 1}/{len(self.frame_list)} in this tracklet"
        )

    def _accept_split(self) -> None:
        if not self.frame_list:
            return
        self.result_cut_frame = self.frame_list[self.slider.value()]
        self.accept()


class _MainWindow(QtWidgets.QMainWindow):
    def __init__(self, groups_path: Path, out_path: Path):
        super().__init__()
        doc = json.loads(groups_path.read_text())
        self.meta = doc["meta"]
        self.groups: list[dict] = doc["groups"]
        # Optional {"label#tid": "was: <slot>"} hints (2026-09-12,
        # audit_cross_slot_consistency.py's conflict-review groups) --
        # shown alongside each member's checkbox so a conflict between
        # two previously-different slot assignments is visible at a
        # glance, without needing a separate reference file open.
        self.member_hints: dict[str, str] = doc.get("member_hints", {})
        self.out_path = out_path
        self.assignments: dict[str, dict] = {}
        if out_path.exists():
            self.assignments = json.loads(out_path.read_text())

        conn = sqlite3.connect(f"file:{self.meta['session']}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        self.conn = conn
        self.states = load_camera_states(conn, self.meta["shot_id"])
        self.sync_table, self.svid_by_cam = load_sync_table(conn, self.meta["shot_id"])
        self.label_to_cam = {r["label"]: r["id"] for r in conn.execute("SELECT id, label FROM camera_instances")}
        self.fkp = FKPredictor(conn, self.meta["tracking_run"], self.states, _DEFAULT_TRIAL)
        # On-disk crop cache: re-decoding a video seek for every thumbnail on
        # every group switch was the dominant cost of browsing groups
        # (found 2026-09-11, Harri). Lives next to the groups file so it
        # survives across GUI relaunches on the same run.
        cache_dir = groups_path.parent / "thumb_cache"
        self.analyzer = GroupAnalyzer(
            conn, self.states, self.sync_table, self.svid_by_cam, self.label_to_cam, self.fkp, cache_dir,
        )

        # Real anatomical slot names -- distinct from _DEFAULT_TRIAL's own
        # keys (a_hip_R, k_Xp_R, ...), which are P-A's unlabelled FK probes,
        # not real slots.
        self.slot_names = catalog_module("leg").slot_names()

        self._build_ui()
        self._populate_list()

    # ---- UI scaffolding ----
    def _build_ui(self) -> None:
        self.setWindowTitle("label_tracklet_groups_gui")
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        left = QtWidgets.QVBoxLayout()
        self.filter_unassigned_check = QtWidgets.QCheckBox("Show only unassigned")
        self.filter_unassigned_check.stateChanged.connect(self._on_filter_toggled)
        left.addWidget(self.filter_unassigned_check)
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setFixedWidth(420)
        self.list_widget.currentRowChanged.connect(self._on_select)
        left.addWidget(self.list_widget)
        layout.addLayout(left)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right)

        self.thumbs_layout = QtWidgets.QGridLayout()
        self.thumbs_layout.setSpacing(4)
        thumbs_widget = QtWidgets.QWidget()
        thumbs_widget.setLayout(self.thumbs_layout)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(thumbs_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(3 * (_THUMB_H + 50) + 40)
        right.addWidget(scroll)

        self.suggestion_table = QtWidgets.QTableWidget(0, 2)
        self.suggestion_table.setHorizontalHeaderLabels(["slot", "mean dist (m)"])
        self.suggestion_table.setFixedHeight(160)
        self.suggestion_table.itemDoubleClicked.connect(self._pick_suggestion)
        right.addWidget(self.suggestion_table)

        form = QtWidgets.QHBoxLayout()
        self.slot_combo = QtWidgets.QComboBox()
        self.slot_combo.addItems([_UNLABELED] + self.slot_names + [_REJECT_NOISE, _REJECT_OTHER_SUBJECT])
        form.addWidget(QtWidgets.QLabel("Slot:"))
        form.addWidget(self.slot_combo)
        assign_btn = QtWidgets.QPushButton("Assign")
        assign_btn.clicked.connect(self._assign)
        form.addWidget(assign_btn)
        reject_btn = QtWidgets.QPushButton("Reject (unspecified)")
        reject_btn.clicked.connect(self._reject)
        form.addWidget(reject_btn)
        right.addLayout(form)

        split_row = QtWidgets.QHBoxLayout()
        split_row.addWidget(QtWidgets.QLabel("Check members above, then:"))
        assign_new_btn = QtWidgets.QPushButton("Assign checked as new group")
        assign_new_btn.setToolTip("Cuts checked members into their own group and immediately "
                                   "applies the Slot selection above -- a real slot, or one of "
                                   "the two 'reject:' entries -- in one step.")
        assign_new_btn.clicked.connect(self._assign_checked_as_new_group)
        split_row.addWidget(assign_new_btn)
        accept_hint_btn = QtWidgets.QPushButton("Keep checked members' own \"was: ...\" slot")
        accept_hint_btn.setToolTip("For conflict-review groups (member_hints): confirms each checked "
                                    "member's own previous slot as correct, individually -- unlike "
                                    "Assign-as-new-group, checked members can keep DIFFERENT slots.")
        accept_hint_btn.clicked.connect(self._accept_hints_for_checked)
        split_row.addWidget(accept_hint_btn)
        discard_btn = QtWidgets.QPushButton("Discard checked (noise)")
        discard_btn.clicked.connect(self._discard_selected)
        split_row.addWidget(discard_btn)
        split_btn = QtWidgets.QPushButton("Split checked into new group")
        split_btn.clicked.connect(self._split_selected)
        split_row.addWidget(split_btn)
        split_frame_btn = QtWidgets.QPushButton("Split checked member's tracklet (scrub)...")
        split_frame_btn.clicked.connect(self._split_member_at_frame)
        split_row.addWidget(split_frame_btn)
        self.merge_target = QtWidgets.QLineEdit()
        self.merge_target.setPlaceholderText("target group_id")
        merge_btn = QtWidgets.QPushButton("Merge whole group into...")
        merge_btn.clicked.connect(self._merge)
        split_row.addWidget(self.merge_target)
        split_row.addWidget(merge_btn)
        right.addLayout(split_row)

        self.status_label = QtWidgets.QLabel()
        right.addWidget(self.status_label)
        right.addStretch(1)

        self.statusBar()
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+S"), self, self._save)

    def _row_text(self, g: dict) -> str:
        a = self.assignments.get(g["group_id"])
        state = ""
        if a:
            if a.get("rejected"):
                reason = a.get("reject_reason")
                state = f"  [REJECTED: {reason}]" if reason else "  [REJECTED]"
            else:
                state = f"  -> {a.get('slot')}"
        flags = ("  [CONTRADICTION]" if g["flagged_contradiction"] else "") + ("  [ambiguous]" if g["ambiguous"] else "")
        return f"{g['group_id']}  ({len(g['members'])} members){flags}{state}"

    def _populate_list(self, keep_group_id: str | None = None) -> None:
        """Rebuilds the group list. `keep_group_id`, when given, re-selects
        that group afterward (2026-09-11, Harri: re-sorting after an action
        used to jump the view to an arbitrary group, making a run of
        several splits/discards on the same contaminated group hard to
        follow) -- if it no longer exists or is filtered out (e.g. it was
        just assigned and "Show only unassigned" is on), falls back to the
        top of the list rather than leaving the panel stuck on stale data."""
        only_unassigned = self.filter_unassigned_check.isChecked()
        all_sorted = sorted(self.groups, key=lambda g: -len(g["members"]))
        self._sorted_groups = (
            [g for g in all_sorted if g["group_id"] not in self.assignments] if only_unassigned else all_sorted
        )

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for g in self._sorted_groups:
            self.list_widget.addItem(self._row_text(g))
        self.list_widget.blockSignals(False)
        self._refresh_status()

        idx = None
        if keep_group_id is not None:
            idx = next((i for i, g in enumerate(self._sorted_groups) if g["group_id"] == keep_group_id), None)
        if idx is None and self._sorted_groups:
            idx = 0
        if idx is not None:
            self.list_widget.setCurrentRow(idx)  # signals unblocked again -- fires _on_select
        else:
            self._on_select(-1)  # list is empty -- clear the panel explicitly

    def _on_filter_toggled(self, _state: int) -> None:
        g = self._current_group()
        self._populate_list(keep_group_id=g["group_id"] if g else None)

    def _refresh_status(self) -> None:
        n_done = sum(1 for g in self.groups if g["group_id"] in self.assignments)
        self.status_label.setText(f"{n_done}/{len(self.groups)} groups reviewed  "
                                   f"(saved: {self.out_path})")

    def _current_group(self) -> dict | None:
        row = self.list_widget.currentRow()
        if row < 0 or row >= len(self._sorted_groups):
            return None
        return self._sorted_groups[row]

    # ---- selection / rendering ----
    def _on_select(self, row: int) -> None:
        for i in reversed(range(self.thumbs_layout.count())):
            w = self.thumbs_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.suggestion_table.setRowCount(0)
        self._member_checks: list[tuple[tuple[str, int], QtWidgets.QCheckBox]] = []
        g = self._current_group()
        if g is None:
            return

        info = self.analyzer.analyze(g, self.meta)
        # one row per member (camera, tracklet[, sub-range]), one column per
        # shared sample time -- so the same instant lines up across cameras
        # and a mismatched marker (a different physical thing in one row) is
        # obvious at a glance.
        for col, t in enumerate(info["sample_times"]):
            self.thumbs_layout.addWidget(QtWidgets.QLabel(f"t={t:.2f}s"), 0, col + 1)
        for r, (cam_id, tid, lo, hi) in enumerate(info["members"], start=1):
            label = next(k for k, v in self.label_to_cam.items() if v == cam_id)
            range_suffix = f"[{lo or ''}:{hi or ''}]" if lo is not None or hi is not None else ""
            hint = self.member_hints.get(f"{label}#{tid}", "")
            hint_suffix = f"  ({hint})" if hint else ""
            check = QtWidgets.QCheckBox(f"{label}#{tid}{range_suffix}{hint_suffix}")
            self._member_checks.append(((cam_id, tid, lo, hi), check))
            self.thumbs_layout.addWidget(check, r, 0)
            frames = info["frames_by_member"][(cam_id, tid, lo, hi)]
            for c, t in enumerate(info["sample_times"]):
                pm, gap_s, fidx = self.analyzer.sample_image(cam_id, tid, t, self.meta, frames)
                if pm is not None:
                    lbl_img = _ThumbLabel(cam_id, tid, lo, hi, fidx, self._split_member_tracklet,
                                          self._open_scrub_split)
                    lbl_img.setPixmap(pm)
                else:
                    lbl_img = QtWidgets.QLabel()
                    if gap_s is not None:
                        # this tracklet has no detection anywhere near this
                        # instant -- do not fake alignment by showing a
                        # distant frame under a misleading timestamp.
                        lbl_img.setText(f"(no frame here\n±{gap_s:.1f}s away)")
                    else:
                        lbl_img.setText("(no frame)")
                self.thumbs_layout.addWidget(lbl_img, r, c + 1)

        for name, dist in info["suggestions"][:8]:
            r = self.suggestion_table.rowCount()
            self.suggestion_table.insertRow(r)
            self.suggestion_table.setItem(r, 0, QtWidgets.QTableWidgetItem(name))
            self.suggestion_table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{dist:.3f}"))

        saved = self.assignments.get(g["group_id"])
        if saved and saved.get("slot"):
            self.slot_combo.setCurrentText(saved["slot"])
        elif info["suggestions"]:
            top = info["suggestions"][0][0]
            if top in self.slot_names:
                self.slot_combo.setCurrentText(top)

    def _pick_suggestion(self, item: QtWidgets.QTableWidgetItem) -> None:
        name = self.suggestion_table.item(item.row(), 0).text()
        if name in self.slot_names:
            self.slot_combo.setCurrentText(name)

    # ---- actions ----
    def _assign(self) -> None:
        g = self._current_group()
        if g is None:
            return
        text = self.slot_combo.currentText()
        if text in _REJECT_REASONS:
            self.assignments[g["group_id"]] = {
                "slot": None, "rejected": True, "reject_reason": _REJECT_REASONS[text], "members": g["members"],
            }
        else:
            self.assignments[g["group_id"]] = {"slot": text, "rejected": False, "members": g["members"]}
        self._save()
        self._populate_list(keep_group_id=g["group_id"])

    def _reject(self) -> None:
        """Reject with no specific reason -- use the Slot combo's two
        'reject:' entries + Assign instead when the reason (noise vs. a
        real marker on another subject/prop) is worth recording."""
        g = self._current_group()
        if g is None:
            return
        self.assignments[g["group_id"]] = {"slot": None, "rejected": True, "members": g["members"]}
        self._save()
        self._populate_list(keep_group_id=g["group_id"])

    def _checked_members(self) -> list[list]:
        """Checked members in the grid, in the group's own JSON member shape
        (2-element [label, tid], or 4-element with a frame sub-range)."""
        out = []
        for (cam_id, tid, lo, hi), check in getattr(self, "_member_checks", []):
            if check.isChecked():
                label = next(k for k, v in self.label_to_cam.items() if v == cam_id)
                out.append(_member_json(label, tid, lo, hi))
        return out

    def _next_group_id(self) -> str:
        existing = {g["group_id"] for g in self.groups}
        n = 0
        while f"group_{n}" in existing:
            n += 1
        return f"group_{n}"

    def _finalize_members(self, g: dict, members: list, slot: str | None, rejected: bool,
                          reject_reason: str | None) -> str:
        """Cut `members` out of group `g` into their own new group (or
        reuse `g` directly if every one of its own members is in
        `members`, i.e. there's nothing left to split it *from*) and
        record the assignment/rejection. Returns the target group_id.
        Caller owns `_save()`/`_populate_list()` -- factored out of
        `_finalize_checked` (2026-09-12) so a batch action (accepting
        several members' own differing hints in one go) can make many of
        these mutations before refreshing the view once, instead of once
        per member."""
        if len(members) == len(g["members"]):
            target_id = g["group_id"]
        else:
            g["members"] = [m for m in g["members"] if m not in members]
            target_id = self._next_group_id()
            self.groups.append({"group_id": target_id, "members": members,
                                "flagged_contradiction": False, "ambiguous": False})
        entry = {"slot": slot, "rejected": rejected, "members": members}
        if reject_reason is not None:
            entry["reject_reason"] = reject_reason
        self.assignments[target_id] = entry
        return target_id

    def _finalize_checked(self, slot: str | None, rejected: bool, reject_reason: str | None) -> None:
        """Cut the checked members out into their own group (or reuse the
        current group if every member is checked, i.e. there's nothing to
        split it *from*) and immediately record its assignment/rejection
        in one step -- the "pick one tracklet out of a contaminated group
        and assign it" fast path Harri asked for (2026-09-11), instead of
        split-then-reselect-in-the-list-then-assign as three separate
        actions. The view stays on the *original* group afterward (see
        `_populate_list`'s `keep_group_id`), not the newly finalized one --
        that's the group still being worked on."""
        g = self._current_group()
        checked = self._checked_members()
        if g is None or not checked:
            return
        self._finalize_members(g, checked, slot, rejected, reject_reason)
        self._save()
        self._populate_list(keep_group_id=g["group_id"])

    def _accept_hints_for_checked(self) -> None:
        """Confirms each checked member's own "was: ..." hint (see
        `member_hints`, 2026-09-12) as correct -- unlike
        `_assign_checked_as_new_group`, which applies one Slot combo
        value to every checked member, this finalizes each member
        *individually* to whatever slot its own hint says, since a
        conflict group's checked members can (and often do) have
        different previous slots that are each already right. The fast
        path Harri asked for when reviewing a conflict group: "select the
        correct ones and keep that slot for those" without retyping the
        slot per member."""
        g = self._current_group()
        checked_pairs = [(m, c) for m, c in getattr(self, "_member_checks", []) if c.isChecked()]
        if g is None or not checked_pairs:
            return
        resolved, missing = [], []
        for (cam_id, tid, lo, hi), _ in checked_pairs:
            label = next(k for k, v in self.label_to_cam.items() if v == cam_id)
            hint = self.member_hints.get(f"{label}#{tid}", "")
            if hint.startswith("was: "):
                resolved.append((_member_json(label, tid, lo, hi), hint[len("was: "):]))
            else:
                missing.append(f"{label}#{tid}")
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Accept hint",
                "No 'was: ...' hint for: " + ", ".join(missing) + " -- uncheck these or use Assign instead.")
            return
        # group members sharing the same hinted slot into one new group
        # each, rather than one singleton group per member.
        by_slot: dict[str, list] = {}
        for member_json, slot in resolved:
            by_slot.setdefault(slot, []).append(member_json)
        for slot, members in by_slot.items():
            self._finalize_members(g, members, slot, False, None)
        self._save()
        self._populate_list(keep_group_id=g["group_id"])

    def _discard_selected(self) -> None:
        """Marks the checked members as noise -- a specific case of
        `_finalize_checked` (2026-09-11): previously this just deleted
        them with no record at all; now it leaves the same audit trail as
        any other rejection, distinguishing "noise" from "a real marker on
        another subject/prop" (the Slot combo's `reject: other subject /
        prop` entry + "Assign checked as new group", for that case)."""
        self._finalize_checked(slot=None, rejected=True, reject_reason="noise")

    def _assign_checked_as_new_group(self) -> None:
        """Applies the current Slot combo selection -- a real slot, or one
        of the two 'reject:' pseudo-entries -- to the checked members,
        splitting them off first if they're not the whole group."""
        text = self.slot_combo.currentText()
        if text in _REJECT_REASONS:
            self._finalize_checked(slot=None, rejected=True, reject_reason=_REJECT_REASONS[text])
        else:
            self._finalize_checked(slot=text, rejected=False, reject_reason=None)

    def _split_selected(self) -> None:
        g = self._current_group()
        checked = self._checked_members()
        if g is None or not checked:
            return
        if len(checked) == len(g["members"]):
            QtWidgets.QMessageBox.warning(self, "Split", "every member is checked -- nothing left in the "
                                          "original group to split it from")
            return
        g["members"] = [m for m in g["members"] if m not in checked]
        self.groups.append({"group_id": self._next_group_id(), "members": checked,
                            "flagged_contradiction": False, "ambiguous": False})
        self._populate_list(keep_group_id=g["group_id"])

    def _split_member_tracklet(self, cam_id: str, tid: int, lo: int | None, hi: int | None, cut_fidx: int) -> None:
        """Cut one member's own tracklet in two at `cut_fidx`, in place in
        the current group -- frames before the cut keep [lo, cut_fidx),
        frames from the cut on become [cut_fidx, hi). Use "Split checked
        into new group" afterwards to route one half to a different
        group_id/slot."""
        g = self._current_group()
        if g is None:
            return
        label = next(k for k, v in self.label_to_cam.items() if v == cam_id)
        target = _member_json(label, tid, lo, hi)
        if target not in g["members"]:
            return
        g["members"] = [m for m in g["members"] if m != target]
        g["members"].append(_member_json(label, tid, lo, cut_fidx))
        g["members"].append(_member_json(label, tid, cut_fidx, hi))
        self._on_select(self.list_widget.currentRow())

    def _split_member_at_frame(self) -> None:
        """Split exactly one checked member's own tracklet -- opens the
        scrub dialog seeded at an auto-suggested frame (per-frame FK-slot
        classification, see GroupAnalyzer.classify_member_frames /
        _suggest_split_frame) rather than asking for a frame number blind."""
        checked = [(m, c) for m, c in getattr(self, "_member_checks", []) if c.isChecked()]
        if len(checked) != 1:
            QtWidgets.QMessageBox.information(self, "Split tracklet", "Check exactly one member row first.")
            return
        (cam_id, tid, lo, hi), _ = checked[0]
        self._open_scrub_split(cam_id, tid, lo, hi, start_fidx=None)

    def _open_scrub_split(self, cam_id: str, tid: int, lo: int | None, hi: int | None,
                          start_fidx: int | None) -> None:
        """Open the scrub-and-split dialog for one member. `start_fidx`
        seeds the scrub position -- an exact frame when opened from a
        thumbnail's right-click, or None (falls back to an auto-suggested
        transition frame, or the tracklet's midpoint) from the button."""
        g = self._current_group()
        if g is None:
            return
        frames = self.analyzer.member_frames(cam_id, tid, self.meta, lo, hi)
        if len(frames) < 2:
            QtWidgets.QMessageBox.information(self, "Split tracklet", "This member has too few frames to split.")
            return
        if start_fidx is None:
            classified = self.analyzer.classify_member_frames(cam_id, tid, frames)
            start_fidx = _suggest_split_frame(classified)
        dlg = _ScrubSplitDialog(self.analyzer, self.sync_table, cam_id, tid, frames, start_fidx, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted and dlg.result_cut_frame is not None:
            self._split_member_tracklet(cam_id, tid, lo, hi, dlg.result_cut_frame)

    def _merge(self) -> None:
        g = self._current_group()
        target_id = self.merge_target.text().strip()
        if g is None or not target_id:
            return
        target = next((x for x in self.groups if x["group_id"] == target_id), None)
        if target is None or target is g:
            QtWidgets.QMessageBox.warning(self, "Merge", f"no such group_id: {target_id!r}")
            return
        target["members"] = target["members"] + [m for m in g["members"] if m not in target["members"]]
        self.groups = [x for x in self.groups if x["group_id"] != g["group_id"]]
        self._populate_list(keep_group_id=target["group_id"])

    def _save(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(self.assignments, indent=2))
        self._refresh_status()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self._save()
        ev.accept()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", required=True, help="build_tracklet_groups.py's --output file.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = _MainWindow(Path(args.groups), Path(args.output))
    win.resize(1500, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
