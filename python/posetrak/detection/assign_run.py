# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""assign_run.py — `detect assign`: write the track-to-person assignment of a detection run.

Reads the boxes and trunk keypoints of a run, runs the geometric association of ``track_assignment.py`` and writes
``detection_track_assignments`` rows, the same rows the stitcher writes by hand. Design:
docs/roadmap/features/track-person-assignment/track-person-assignment-design.md.

Rows already on the run are handled by *existing*: ``refuse`` (the default) stops, ``fill`` keeps every row and
assigns only the boxes no row covers, ``replace`` deletes them first. In the last two the rows are also seeds: a
person track that overlaps the boxes of a row takes the row's person name, and a person named by a row is the
same person throughout the run.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.setup.db_context import SyncPoint, SyncTable
from app.setup.extrinsics_solver import _proj_matrix, _undistort_pts
from posetrak.calibration.session_cameras import load_camera_states
from posetrak.db.manage_person import find_or_create_person, list_persons
from posetrak.detection import track_assignment as ta
from posetrak.detection.pose_models import spec_for_pose_model


class AssignError(Exception):
    """The run cannot be assigned (unknown pose model, existing rows, missing calibration, ...)."""


@dataclass
class Row:
    """One ``detection_track_assignments`` row."""

    svid: str          # capture_videos.id
    tid: int           # detector track id
    person: str
    first: int         # first video frame
    last: int          # last video frame, inclusive


@dataclass
class Check:
    """A stretch to check by hand or with segmentation."""

    kind: str                  # 'contact' | 'handover' | 'unevidenced' | 'split' | 'unassigned'
    t0: float                  # global time, seconds
    t1: float
    persons: list[str]
    # (camera video id, first frame, last frame) of the boxes involved, for the box-based kinds
    frames: list[tuple[str, int, int]] = field(default_factory=list)


@dataclass
class AssignResult:
    rows: list[Row]                          # rows to write (only the new ones in fill mode)
    kept: list[Row]                          # existing rows left in place
    checks: list[Check]
    persons: list[dict]
    cameras: dict[str, dict]
    unassigned: list[tuple[str, int, int, int]]    # (camera video, track, first, last) of boxes without a person
    report: dict

    def flagged(self, svid: str, frame: int, t: float, person: str | None) -> bool:
        """Whether a box (camera video, frame, global time, assigned person) lies in a stretch listed for checking."""
        for c in self.checks:
            if c.kind in ("contact", "handover"):
                if c.t0 <= t <= c.t1 and (person is None or person in c.persons):
                    return True
            elif any(s == svid and a <= frame <= b for s, a, b in c.frames):
                return True
        return False


def _subtract(first: int, last: int, taken: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out = [(first, last)]
    for a, b in taken:
        nxt = []
        for x, y in out:
            if b < x or a > y:
                nxt.append((x, y))
                continue
            if x < a:
                nxt.append((x, a - 1))
            if b < y:
                nxt.append((b + 1, y))
        out = nxt
    return out


def load_sync(conn: sqlite3.Connection, sync_id: str) -> SyncTable:
    """The sync table of one sync configuration."""
    rows = conn.execute(
        "SELECT sp.shot_video_id, sp.video_frame, sp.timestamp_s, cv.actual_fps FROM sync_points sp "
        "JOIN capture_videos cv ON cv.id = sp.shot_video_id WHERE sp.sync_config_id = ?", (sync_id,)).fetchall()
    pts = [SyncPoint(r["shot_video_id"], r["shot_video_id"], int(r["video_frame"]), float(r["timestamp_s"]))
           for r in rows]
    return SyncTable(pts, {r["shot_video_id"]: float(r["actual_fps"]) for r in rows})


def _load_observations(conn, run, spec, states, prm):
    """Undistorted trunk observations of the run, with the pieces of every camera's detector fragments.

    Returns ``obs`` ({camera video: {(frame, piece): Obs}}), ``cam_of`` (video -> camera instance), ``tid_of``
    ({(video, piece, frame): detector track id}) and ``piece_frames`` ({(video, piece): sorted frames}, every box
    of the piece, also those without trunk observations).
    """
    trunk, body = spec.trunk_indices(), spec.body_indices()
    cam_of = {r["id"]: r["camera_instance_id"] for r in conn.execute(
        "SELECT id, camera_instance_id FROM capture_videos WHERE shot_id = ?", (run["shot_id"],))}
    heights = {(r[0], r[1], r[2]): r[3] for r in conn.execute(
        "SELECT shot_video_id, video_frame, track_id, bbox_h FROM person_detections "
        "WHERE detection_run_id = ? AND region_type = 'full_body' AND bbox_h IS NOT NULL", (run["id"],))}
    frames: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))   # video -> frame -> boxes
    trunk_of: dict[tuple[str, int, int], tuple] = {}                              # (video, frame, tid) -> trunk
    for r in conn.execute(
            "SELECT shot_video_id, video_frame, track_id, keypoints FROM detection_keypoints "
            "WHERE detection_run_id = ? AND region_type = 'full_body'", (run["id"],)):
        svid = r[0]
        if cam_of.get(svid) not in states:
            continue
        kp = np.frombuffer(bytes(r[3]), dtype="<f4").reshape(-1, 3)
        h = heights.get((svid, r[1], r[2]))
        if h is None:       # no stored box: the extent of the confident keypoints
            conf = kp[kp[:, 2] >= run["pose_conf_threshold"]]
            h = float(np.ptp(conf[:, 1])) if len(conf) > 1 else 0.0
        frames[svid][r[1]].append((r[2], kp[body], h))
        ok = kp[trunk, 2] >= run["pose_conf_threshold"]
        if ok.sum() >= prm.min_joints and h > 0:
            trunk_of[(svid, r[1], r[2])] = (kp[trunk, :2].astype(np.float64), ok, h)
    obs: dict[str, dict] = {}
    tid_of: dict[tuple[str, int, int], int] = {}
    piece_frames: dict[tuple[str, int], list[int]] = defaultdict(list)
    for svid, fr in frames.items():
        piece = ta.make_pieces(fr, prm)
        for (f, tid), pc in piece.items():
            tid_of[(svid, pc, f)] = tid
            piece_frames[(svid, pc)].append(f)
        keys = [(f, tid) for (f, tid) in piece if (svid, f, tid) in trunk_of]
        if not keys:
            continue
        und = _undistort_pts(np.concatenate([trunk_of[(svid, f, tid)][0] for f, tid in keys]),
                             states[cam_of[svid]]).reshape(len(keys), -1, 2)
        obs[svid] = {(f, piece[(f, tid)]): ta.Obs(svid, piece[(f, tid)], f, und[i], *trunk_of[(svid, f, tid)][1:])
                     for i, (f, tid) in enumerate(keys)}
    for v in piece_frames.values():
        v.sort()
    return obs, cam_of, tid_of, piece_frames


def assign_run(
    conn: sqlite3.Connection, run_id: str, *, persons: int | None = None, subjects: list[str] | None = None,
    existing: str = "refuse", prm: ta.AssignParams | None = None, trace: dict | None = None,
) -> AssignResult:
    """Compute the assignment of *run_id*; nothing is written (see :func:`write_assignment`).

    *persons* is the number of persons (default: the subjects given, else the number seen together for a fifth of
    the time); *subjects* are capture person names, given to the persons in order of first appearance after the
    persons named by seed rows; *existing* is ``refuse``, ``fill`` or ``replace`` (module docstring). A dict given
    as *trace* receives the intermediate results (tracks, their labels and boxes) for inspection.
    """
    prm = prm or ta.AssignParams()
    run = conn.execute("SELECT * FROM detection_runs WHERE id = ?", (run_id,)).fetchone()
    if run is None:
        raise AssignError(f"detection run {run_id!r} not found")
    spec = spec_for_pose_model(run["pose_model"])
    if spec is None or not spec.trunk:
        raise AssignError(f"pose model {run['pose_model']!r} has no registered trunk keypoints")
    prior = [Row(r[0], r[1], r[2], r[3], r[4]) for r in conn.execute(
        "SELECT shot_video_id, track_id, person_name, first_frame, last_frame FROM detection_track_assignments "
        "WHERE detection_run_id = ? ORDER BY first_frame, shot_video_id", (run_id,))]
    if prior and existing == "refuse":
        raise AssignError(
            f"the run has {len(prior)} track assignment rows already (hand-made or earlier); use --fill-only to "
            "keep them and assign the rest, or --replace to rewrite them")
    try:
        states = load_camera_states(conn, run["shot_id"])
    except ValueError as exc:
        raise AssignError(str(exc)) from exc
    sync = load_sync(conn, run["sync_config_id"])
    obs_by_cam, cam_of, tid_of, piece_frames = _load_observations(conn, run, spec, states, prm)
    if len(obs_by_cam) < 2:
        raise AssignError("fewer than two calibrated cameras have trunk keypoints in the run")
    P = {svid: _proj_matrix(states[cam_of[svid]]) for svid in obs_by_cam}
    fps = {r["id"]: float(r["actual_fps"]) for r in conn.execute(
        "SELECT id, actual_fps FROM capture_videos WHERE shot_id = ?", (run["shot_id"],))}
    dt = 1.0 / min(fps[s] for s in obs_by_cam)
    t0 = float(run["time_start_s"])

    # --- instants of a common grid; per camera the frame closest to each grid time
    by_frame: dict[tuple[str, int], list[ta.Obs]] = defaultdict(list)
    for svid, d in obs_by_cam.items():
        for (f, _), o in d.items():
            by_frame[(svid, f)].append(o)
    frame_t: dict[tuple[str, int], float] = {}
    best: dict[tuple[int, str], tuple[float, int]] = {}
    for svid, f in by_frame:
        t = sync.frame_to_global_time(f, svid)
        if t is None:
            continue
        frame_t[(svid, f)] = t
        i = round((t - t0) / dt)
        off = abs(t - (t0 + i * dt))
        if (i, svid) not in best or off < best[(i, svid)][0]:
            best[(i, svid)] = (off, f)
    inst: dict[int, list[ta.Obs]] = defaultdict(list)
    for (i, svid), (_, f) in best.items():
        inst[i].extend(by_frame[(svid, f)])
    n_inst = max(inst) + 1 if inst else 0
    obs_at = [inst.get(i, []) for i in range(n_inst)]

    # --- hypotheses and tracks
    hyps = [ta.cluster_instant(o, P, prm) if len({x.cam for x in o}) > 1 else [] for o in obs_at]
    tracks, which = ta.link_hypotheses(hyps, dt, prm)
    win = max(1, round(prm.stitch_window_s / dt))
    ends: dict[int, tuple[Counter, Counter]] = {t.id: (Counter(), Counter()) for t in tracks}
    boxes_of: dict[int, list[tuple[str, int, int]]] = defaultdict(list)
    for i, hs in enumerate(hyps):
        for h, t in zip(hs, which[i]):
            if t is None:
                continue
            fs = [(obs_at[i][m].cam, obs_at[i][m].tid) for m in h.members]
            if i < tracks[t].first() + win:
                ends[t][0].update(fs)
            if i > tracks[t].last() - win:
                ends[t][1].update(fs)
            boxes_of[t].extend((obs_at[i][m].cam, obs_at[i][m].tid, obs_at[i][m].frame) for m in h.members)

    # --- seeds: the persons the rows already there name, and the tracks they pin down
    seed_names: list[str] = []
    seed_at: dict[tuple[str, int], list[tuple[int, int, str]]] = defaultdict(list)
    for r in prior:
        if r.person not in seed_names:
            seed_names.append(r.person)
        seed_at[(r.svid, r.tid)].append((r.first, r.last, r.person))
    anchors: dict[int, int] = {}
    for t, boxes in boxes_of.items():
        votes = Counter(name for cam, pc, f in boxes
                        for a, b, name in seed_at.get((cam, tid_of[(cam, pc, f)]), ()) if a <= f <= b)
        top = votes.most_common(2)
        if top and top[0][1] >= prm.anchor_min_boxes and (len(top) == 1 or top[0][1] >= 2 * top[1][1]):
            anchors[t] = seed_names.index(top[0][0])

    # --- label the tracks, then every box
    wanted = persons if persons is not None else (len(subjects or []) or ta.count_persons(tracks, prm))
    n_persons = max(wanted, len(seed_names))
    label = ta.label_tracks(tracks, ends, dt, n_persons, prm, anchors)
    if trace is not None:
        trace.update(tracks=tracks, label=label, boxes_of=boxes_of, ends=ends, anchors=anchors, dt=dt, tid_of=tid_of)
    pos: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    for t in tracks:
        if label.get(t.id) is not None:
            pos[label[t.id]].update(t.pos)
    kept = sorted(pos, key=lambda c: min(pos[c]))
    got, track_pos = ta.refine_persons(obs_at, {c: pos[c] for c in kept}, dt, P, prm)
    kept = [c for c in kept if track_pos.get(c)] or kept
    votes_of: dict[tuple[str, int], dict[int, int | None]] = defaultdict(dict)
    for i, assigned in got.items():
        for k, person in assigned.items():
            o = obs_at[i][k]
            votes_of[(o.cam, o.tid)][o.frame] = person

    # --- names
    existing_names = [p["name"] for p in list_persons(conn, run["shot_id"])]
    free = [n for n in (subjects or []) if n not in seed_names]
    if not subjects and not seed_names and len(kept) == 1 and len(existing_names) == 1:
        free = list(existing_names)                      # a single person in a capture with a single person
    name_of: dict[int, str] = {}
    k = 0
    for c in kept:
        if c < len(seed_names):
            name_of[c] = seed_names[c]
        elif free:
            name_of[c] = free.pop(0)
        else:
            k += 1
            while f"Person {k}" in name_of.values() or f"Person {k}" in seed_names:
                k += 1
            name_of[c] = f"Person {k}"

    # --- pieces -> boxes -> rows
    covered: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    person_cov: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    if existing == "fill":
        for r in prior:
            covered[(r.svid, r.tid)].append((r.first, r.last))
            person_cov[(r.svid, r.person)].append((r.first, r.last))
    checks: list[Check] = []
    person_at: dict[tuple[str, int, int], str | None] = {}      # (video, detector track, frame) -> person name

    def span_t(svid: str, a: int, b: int) -> tuple[float, float]:
        tt = [frame_t[(svid, f)] for f in (a, b) if (svid, f) in frame_t]
        return (min(tt), max(tt)) if tt else (0.0, 0.0)

    for (svid, pc), frames in sorted(piece_frames.items()):
        votes = {f: (p if p in name_of else None) for f, p in votes_of.get((svid, pc), {}).items()}
        segs = ta.label_fragment(frames, votes, prm.min_run)
        for s_ in segs:
            name = name_of[s_.person] if s_.person is not None else None
            for f in frames:
                if s_.first <= f <= s_.last:
                    person_at[(svid, tid_of[(svid, pc, f)], f)] = name
            if name is None:
                continue
            long_blind = [(a, b) for a, b in s_.blind if (b - a + 1) / fps[svid] > prm.max_unevidenced_s]
            if long_blind:
                checks.append(Check("unevidenced", *span_t(svid, long_blind[0][0], long_blind[-1][1]), [name],
                                    [(svid, a, b) for a, b in long_blind]))
        m = int(prm.check_margin_s * fps[svid])
        for s1, s2 in zip(segs, segs[1:]):
            if s1.person is not None and s2.person is not None:
                checks.append(Check("split", *span_t(svid, s1.last, s2.first),
                                    [name_of[s1.person], name_of[s2.person]], [(svid, s1.last - m, s2.first + m)]))
    # a row per run of one person on one detector track
    by_tid: dict[tuple[str, int], list[tuple[int, str | None]]] = defaultdict(list)
    for (svid, tid, f), name in person_at.items():
        by_tid[(svid, tid)].append((f, name))
    rows: list[Row] = []
    unassigned: list[tuple[str, int, int, int]] = []
    for (svid, tid), seq in sorted(by_tid.items()):
        seq.sort()
        run_: list[int] = []
        cur: str | None = None
        for f, name in [*seq, (None, "end")]:
            if run_ and name != cur:
                if cur is None:
                    unassigned.append((svid, tid, run_[0], run_[-1]))
                    checks.append(Check("unassigned", *span_t(svid, run_[0], run_[-1]), [],
                                        [(svid, run_[0], run_[-1])]))
                else:
                    for a, b in _subtract(run_[0], run_[-1], covered[(svid, tid)] + person_cov[(svid, cur)]):
                        rows.append(Row(svid, tid, cur, a, b))
                run_ = []
            if f is not None:
                run_.append(f)
                cur = name

    # --- contacts and hand-overs
    con = ta.contacts(track_pos, n_inst, dt, prm)
    span: list[list] = []
    for i in sorted(con):
        key = frozenset(con[i])
        if span and span[-1][0] == key and i - span[-1][2] <= 1 + int(prm.check_margin_s / dt):
            span[-1][2] = i
        else:
            span.append([key, i, i])
    for key, a, b in span:
        checks.append(Check("contact", t0 + a * dt - prm.check_margin_s, t0 + b * dt + prm.check_margin_s,
                            sorted(name_of[c] for c in key if c in name_of)))
    for c in kept:
        ts = sorted((t for t in tracks if label.get(t.id) == c), key=lambda t: t.first())
        for a, b in zip(ts, ts[1:]):
            # The person is carried from one track to the next by weak cues, unless a seed pins both. With another
            # person near at either end, the identity of the whole next track rests on that decision.
            if b.first() > a.last() and not (a.id in anchors and b.id in anchors):
                near = any(
                    (x := ta.interpolate(track_pos.get(o, {}), i, dt, prm.interp_gap_s)) is not None
                    and np.linalg.norm(x - y) < prm.handover_risk_m
                    for o in kept if o != c for i, y in ((a.last(), a.pos[a.last()]), (b.first(), b.pos[b.first()])))
                checks.append(Check("handover", t0 + a.last() * dt, t0 + (b.last() if near else b.first()) * dt,
                                    [name_of[c]]))

    persons_info = [{"name": name_of[c], "instants": len(track_pos.get(c, ())), "first_s": t0 + min(pos[c]) * dt,
                     "last_s": t0 + max(pos[c]) * dt, "seeded": c < len(seed_names)} for c in kept]
    cam_err: dict[str, list[float]] = defaultdict(list)
    cam_boxes: Counter = Counter()
    cam_clustered: Counter = Counter()
    for i, hs in enumerate(hyps):
        for o in obs_at[i]:
            cam_boxes[o.cam] += 1
        for h in hs:
            for m, e in zip(h.members, h.err):
                cam_err[obs_at[i][m].cam].append(e)
                cam_clustered[obs_at[i][m].cam] += 1
    cameras = {svid: {"boxes": cam_boxes[svid], "clustered_fraction": cam_clustered[svid] / cam_boxes[svid],
                      "median_rel_reproj_err": float(np.median(cam_err[svid])) if cam_err[svid] else None}
               for svid in cam_boxes}
    report = {
        "run": run_id, "persons": persons_info, "cameras": cameras,
        "checks": [{"kind": c.kind, "t0": c.t0, "t1": c.t1, "persons": c.persons, "frames": c.frames}
                   for c in checks],
        "unassigned": [{"camera": s, "track": t, "first": a, "last": b} for s, t, a, b in unassigned],
    }
    return AssignResult(rows, prior if existing == "fill" else [], checks, persons_info, cameras, unassigned, report)


def write_assignment(conn: sqlite3.Connection, run_id: str, result: AssignResult, existing: str) -> None:
    """Write *result* to ``detection_track_assignments`` (replacing the run's rows first in ``replace`` mode)."""
    run = conn.execute("SELECT shot_id FROM detection_runs WHERE id = ?", (run_id,)).fetchone()
    with conn:
        if existing == "replace":
            conn.execute("DELETE FROM detection_track_assignments WHERE detection_run_id = ?", (run_id,))
        ids = {n: find_or_create_person(conn, run["shot_id"], n) for n in {r.person for r in result.rows}}
        conn.executemany(
            "INSERT INTO detection_track_assignments (detection_run_id, shot_video_id, track_id, person_name, "
            "capture_person_id, subject_id, first_frame, last_frame) VALUES (?,?,?,?,?,?,?,?)",
            [(run_id, r.svid, r.tid, r.person, ids[r.person], ids[r.person], r.first, r.last) for r in result.rows])


def format_report(result: AssignResult, cam_label: dict[str, str]) -> str:
    """The report as text; *cam_label* maps a camera video id to its label."""
    out = [f"Assigned {len(result.rows)} track segments to {len(result.persons)} person(s)."]
    for p in result.persons:
        out.append(f"  {p['name']}: {p['first_s']:.2f}-{p['last_s']:.2f} s, located in 3D at {p['instants']} instants"
                   + (" (named by a seed)" if p["seeded"] else ""))
    out.append("Cameras (clustered: share of boxes that agree with another camera; err: median trunk reprojection "
               "error / box height; a camera far above the others has a calibration or sync problem):")
    fractions = [c["clustered_fraction"] for c in result.cameras.values()]
    errors = [c["median_rel_reproj_err"] for c in result.cameras.values() if c["median_rel_reproj_err"] is not None]
    for svid, c in result.cameras.items():
        e = c["median_rel_reproj_err"]
        # a camera that agrees much less than the others: its calibration or its synchronisation is suspect
        odd = len(fractions) > 2 and (c["clustered_fraction"] < 0.6 * float(np.median(fractions))
                                      or (e is not None and e > 1.5 * float(np.median(errors))))
        out.append(f"  {cam_label.get(svid, svid[:8])}: {c['boxes']} boxes, clustered {c['clustered_fraction']:.0%}"
                   + (f", err {e:.3f}" if e is not None else "") + ("   <- calibration or sync?" if odd else ""))
    by_kind = Counter(c.kind for c in result.checks)
    out.append("To check: " + (", ".join(f"{n} {k}" for k, n in sorted(by_kind.items())) or "nothing"))
    for c in sorted(result.checks, key=lambda c: c.t0)[:30]:
        out.append(f"  {c.kind:11s} {c.t0:8.2f}-{c.t1:8.2f} s  {', '.join(c.persons)}")
    if len(result.checks) > 30:
        out.append(f"  ... {len(result.checks) - 30} more in the report file")
    return "\n".join(out)


def report_path(db_path: str | Path, run_id: str) -> Path:
    """Where the report of the assignment of *run_id* is kept: next to the session database."""
    db = Path(db_path)
    return db.with_name(f"{db.stem}.assign-{run_id[:8]}.json")


def save_report(result: AssignResult, path: str | Path) -> None:
    """Write the report as JSON, for the stitcher."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=1)


def load_report(db_path: str | Path, run_id: str) -> dict | None:
    """The report written by the last `detect assign` of *run_id*, or None."""
    path = report_path(db_path, run_id)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
