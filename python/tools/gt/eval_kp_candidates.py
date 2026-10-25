# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""eval_kp_candidates.py — score keypoint picks and tracker runs on the frozen evaluation set.

The evaluation set (ranges, hard windows, events, reference runs) is named in ``kp_eval_set.toml``;
its data lives outside the repository. Usage:

    export POSETRAK_KP_EVAL_ROOT=/path/to/kp_eval_set_v1
    export POSETRAK_KP_EVAL_DB=/path/to/session.db
    python python/tools/gt/eval_kp_candidates.py verify
    python python/tools/gt/eval_kp_candidates.py picks [--pick-sequence 1=SEQ --pick-sequence 2=SEQ ...]
    python python/tools/gt/eval_kp_candidates.py tracker --runs proposal_range3
    python python/tools/gt/eval_kp_candidates.py tracker --run 1=<run id> --run 2=<run id>
    python python/tools/gt/eval_kp_candidates.py events --runs proposal
    python python/tools/gt/eval_kp_candidates.py track-picks --range range1 --run 1=<run id> --run 3=<run id>

``picks`` reports, per range and subject, for the picks of the strongest maximum (or of a stored
pose sequence), the within-3%-of-box-height share, the same-candidate share, the share without a
pick and the false picks on not-visible labels, inside the hard windows and outside them.
``tracker`` is the label-fed comparison: body-joint distance of runs to the reference run.
``events`` is the failure-event comparison of two run groups.
``window-picks`` compares the candidates a tracking run chose over a time window (``tracking_kp_picks``) with the
prototype's picks for the same settings, on the frames whose crop is the same.
``track-picks`` traces the candidate each used observation of a tracking run came from and scores it against the
labels of a range, next to the strongest maximum of the same keypoint-frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tomllib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kp_gt  # noqa: E402

SET_FILE = Path(__file__).with_name("kp_eval_set.toml")
# Tracker distances are scored from this time to the end of the range, the span the design documents' figures refer to.
TRACKER_FROM_S = 52.0


def load_set() -> dict:
    return tomllib.loads(SET_FILE.read_text())


def eval_root(args) -> Path:
    root = args.root or os.environ.get("POSETRAK_KP_EVAL_ROOT")
    if not root:
        sys.exit("give --root or set POSETRAK_KP_EVAL_ROOT (directory of the frozen evaluation set)")
    return Path(root)


def open_db(args) -> sqlite3.Connection:
    path = args.db or os.environ.get("POSETRAK_KP_EVAL_DB")
    if not path:
        sys.exit("give --db or set POSETRAK_KP_EVAL_DB (session database with the tracking runs)")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def cmd_verify(args) -> int:
    root = eval_root(args)
    sums = json.loads((root / load_set()["evaluation_set"]["checksums"]).read_text())
    bad = [name for name, digest in sums.items() if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest]
    print(f"{len(sums) - len(bad)} of {len(sums)} files match their checksum")
    for name in bad:
        print("  differs:", name)
    return 1 if bad else 0


def cmd_picks(args) -> int:
    cfg, root = load_set(), eval_root(args)
    es = cfg["evaluation_set"]
    hard = [(w["t0"], w["t1"]) for w in cfg["hard_windows"]]
    sequences = dict(item.split("=", 1) for item in args.pick_sequence)
    conn = open_db(args) if sequences else None
    print(f"{'range/subject':16s} {'window':8s} {'n':>7s} {'within 3%':>10s} {'same cand.':>11s} {'no pick':>8s} {'false pick':>11s}")
    for name, rng in cfg["ranges"].items():
        for subject in rng["subjects"]:
            d = root / rng["dir"]
            cand = kp_gt.read_candidates(d / f"candidates_subject{subject}.npz")
            labels = kp_gt.read_labels(d / f"labels_subject{subject}.npz", d / f"units_subject{subject}.json", rng["units"])
            picks = (kp_gt.read_sequence_picks(conn, sequences[str(subject)], 0, cand.cameras)
                     if sequences else kp_gt.strongest_picks(cand))
            skip = rng["skip_cameras"].get(str(subject), [])
            outside = _outside_windows(rng, hard)
            for window, windows in (("hard", hard), ("outside", outside)):
                m = kp_gt.pick_metrics(picks, labels, cand, t0=rng["t0"], t1=rng["t1"], tolerance=es["tolerance_box_height"],
                                       same_px=es["same_candidate_px"], skip_cameras=skip, windows=windows)
                print(f"{name + '/' + str(subject):16s} {window:8s} {m['n_visible']:7d} {100 * m['within_tol']:9.1f}% "
                      f"{100 * m['same_candidate']:10.1f}% {100 * m['no_pick']:7.1f}% {100 * m['false_pick']:10.1f}%")
    return 0


def _outside_windows(rng, hard) -> list[tuple[float, float]]:
    """The spans of the range that lie outside the hard windows."""
    spans, start = [], rng["t0"]
    for a, b in sorted(hard):
        if b <= start or a >= rng["t1"]:
            continue
        if a > start:
            spans.append((start, a))
        start = max(start, b)
    if start < rng["t1"]:
        spans.append((start, rng["t1"]))
    return spans


def same_crop_frames(conn, run: str, frames: np.ndarray, videos: list[str], subject: int, tol_px: float):
    """Split the prototype's frames into those whose crop the detection run *run* shares and those it does not.

    ``frames`` is the prototype file's frame table (camera, frame, time, crop centre x, y, crop side), ``videos`` the
    shot video ids of its cameras. The run stores the person's box; the crop is 1.6 times its longer side. Frames
    without a box in the run are in neither set. Returns two sets of (prototype camera index, frame).
    """
    same_crop, other_crop = set(), set()
    for cam, frame, _, cx, cy, side in frames:
        video = videos[int(cam)]
        box = conn.execute("SELECT bbox_x, bbox_y, bbox_w, bbox_h FROM person_detections WHERE detection_run_id = ? AND "
                           "shot_video_id = ? AND video_frame = ? AND track_id = ? AND region_type = 'full_body'",
                           (run, video, int(frame), subject)).fetchone()
        if box is None:
            continue
        near = max(abs(box[0] - cx), abs(box[1] - cy), abs(max(box[2], box[3]) * 1.6 - side)) <= tol_px
        (same_crop if near else other_crop).add((int(cam), int(frame)))
    return same_crop, other_crop


def cmd_compare(args) -> int:
    """Candidates of a detection run against the prototype's candidate file of the same subject and range."""
    cfg, root, conn = load_set(), eval_root(args), open_db(args)
    rng = cfg["ranges"][args.range]
    subject = int(args.subject)
    proto_path = root / rng["dir"] / f"candidates_subject{subject}.npz"
    proto = kp_gt.read_candidates(proto_path)
    proto_videos = [str(v) for v in np.load(proto_path, allow_pickle=True)["svids"]]
    run = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ?", (args.detection_run + "%",)).fetchone()[0]
    new = kp_gt.read_candidates(conn, run_id=run, track_ids=[subject])
    camera_map = {i: new.cameras.index(v) for i, v in enumerate(proto_videos) if v in new.cameras}
    # Frames whose crop is the same in both: the prototype stores the crop (centre, side) of each frame, the run its boxes.
    z = np.load(proto_path, allow_pickle=True)["frames"]
    # The frozen candidate file was made with the heatmap cell index at the cell's left edge; the decoder now places a
    # position at the cell centre. --cell-shift moves the file's positions by that share of a cell (a cell is a
    # sixty-fourth of the crop side) so that only real differences remain.
    if args.cell_shift:
        side = {(int(c), int(f)): s for c, f, _, _, _, s in z}
        shift = np.array([side.get((int(c), int(f)), 0.0) for c, f in zip(proto.camera, proto.frame)]) / 64 * args.cell_shift
        proto.x, proto.y = proto.x + shift, proto.y + shift
    same_crop, other_crop = same_crop_frames(conn, run, z, proto_videos, subject, args.crop_tol_px)
    print(f"{'frames':22s} {'prototype':>10s} {'run':>10s} {'prototype reproduced':>21s} {'run reproduced':>15s}")
    for label, frames in (("all common frames", None), ("same crop", same_crop), ("different crop", other_crop)):
        if frames is not None and not frames:
            continue
        r = kp_gt.reproduction(proto, new, camera_map, args.tol_px, frames=frames)
        print(f"{label:22s} {r['n_reference']:10d} {r['n_other']:10d} {100 * r['reference_reproduced']:20.1f}% "
              f"{100 * r['other_reproduced']:14.1f}%")
    print(f"crops within {args.crop_tol_px} px of the prototype's: {len(same_crop)} frames, other: {len(other_crop)} frames; "
          f"tolerance {args.tol_px} px")
    return 0


def read_window_picks(conn, run: str, person_id: int = 0) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``tracking_kp_picks`` of a run: camera instance id -> (video frames, tracklet ids, ranks).

    The two arrays are [n_frames, n_keypoints]; -1 is no pick, -2 a pick without a tracklet (rank is valid).
    """
    out = {}
    for cam, n, k, frames, tracklets, ranks in conn.execute(
            "SELECT camera_instance_id, n_frames, n_keypoints, video_frames, tracklet_ids, ranks FROM tracking_kp_picks "
            "WHERE run_id = ? AND person_id = ?", (run, person_id)):
        out[cam] = (np.frombuffer(frames, "<i4"), np.frombuffer(tracklets, "<i4").reshape(n, k),
                    np.frombuffer(ranks, "i1").reshape(n, k))
    return out


def cmd_window_picks(args) -> int:
    """The candidates a run chose over a time window against the prototype's picks, on the frames with the same crop.

    ``--prototype-picks`` is an npz made by running the prototype's cross-camera choice with the same settings and the
    same neighbour limits: ``cams`` (camera instance ids), ``camera``, ``frame``, ``keypoint`` (one entry per
    keypoint-frame the prototype decided), ``x``, ``y`` (the position of the candidate it picked in the prototype
    file's convention, NaN for no pick) and ``rank``. A keypoint-frame is the same when both picked a candidate within
    ``--tol-px`` of each other (after ``--cell-shift``) or neither picked one. The rest are classified by what the
    two candidate sets allow: the prototype's pick has no counterpart among the run's candidates (a data
    difference), the run's pick has none among the prototype's, or both exist and the choices differ.
    """
    cfg, root, conn = load_set(), eval_root(args), open_db(args)
    rng = cfg["ranges"][args.range]
    subject = int(args.subject)
    hard = [(w["t0"], w["t1"]) for w in cfg["hard_windows"]]
    proto_path = Path(args.prototype_candidates) if args.prototype_candidates else root / rng["dir"] / f"candidates_subject{subject}.npz"
    z = np.load(proto_path, allow_pickle=True)
    proto_videos = [str(v) for v in z["svids"]]
    proto_cams = [str(c) for c in z["cams"]]
    frames_table = z["frames"]
    time_of = {(int(c), int(f)): float(t) for c, f, t, *_ in frames_table}
    side_of = {(int(c), int(f)): float(side) for c, f, _, _, _, side in frames_table}
    run = conn.execute("SELECT id FROM detection_runs WHERE id LIKE ?", (args.detection_run + "%",)).fetchone()[0]
    tracking = conn.execute("SELECT id FROM tracking_runs WHERE id LIKE ?", (args.run + "%",)).fetchone()[0]
    same_crop, _ = same_crop_frames(conn, run, frames_table, proto_videos, subject, args.crop_tol_px)

    picks = np.load(args.prototype_picks, allow_pickle=True)
    pcam, pframe, pkp = (picks[k].astype(int) for k in ("camera", "frame", "keypoint"))
    px, py = picks["x"].astype(float), picks["y"].astype(float)
    prank = picks["rank"].astype(int)
    if [str(c) for c in picks["cams"]] != proto_cams:
        sys.exit("the prototype picks were made with a different camera list than the candidate file")
    shift = np.array([side_of.get((c, f), 0.0) for c, f in zip(pcam, pframe)]) / 64 * args.cell_shift
    px, py = px + shift, py + shift
    proto_pick = {(int(c), int(f), int(k)): (x, y, r) for c, f, k, x, y, r in zip(pcam, pframe, pkp, px, py, prank)}

    # The prototype's own candidates, to tell a data difference from a different choice.
    proto_cand = kp_gt.read_candidates(proto_path)
    cs = np.array([side_of.get((int(c), int(f)), 0.0) for c, f in zip(proto_cand.camera, proto_cand.frame)]) / 64 * args.cell_shift
    proto_by = {}
    for c, f, k, x, y in zip(proto_cand.camera, proto_cand.frame, proto_cand.keypoint, proto_cand.x + cs, proto_cand.y + cs):
        proto_by.setdefault((int(c), int(f), int(k)), []).append((float(x), float(y)))

    # The run's candidates and picks.
    rows = kp_gt.read_sequence_candidates(conn, args.sequence, 0)
    cand_by = {}
    for cam_id, items in rows.items():
        if cam_id not in proto_cams:
            continue
        c = proto_cams.index(cam_id)
        for frame, _, table in items:
            for kp, x, y, peak, rank, tracklet in table:
                cand_by.setdefault((c, frame, int(kp)), []).append((float(x), float(y), int(rank), int(tracklet)))
    run_picks = read_window_picks(conn, tracking)

    def near(pos, points):
        return any(np.hypot(pos[0] - q[0], pos[1] - q[1]) <= args.tol_px for q in points)

    t0, t1 = rng["t0"], rng["t1"]
    tallies: dict[str, dict[str, int]] = {}

    def add(window: str, kind: str):
        tallies.setdefault(window, {})
        tallies[window][kind] = tallies[window].get(kind, 0) + 1

    for cam_id, (frames, tracklets, ranks) in run_picks.items():
        if cam_id not in proto_cams:
            continue
        c = proto_cams.index(cam_id)
        for i, frame in enumerate(frames):
            frame = int(frame)
            if (c, frame) not in same_crop:
                continue
            t = time_of.get((c, frame), float("nan"))
            if not t0 <= t <= t1:
                continue
            windows = ["all"]
            if t0 + args.interior_s <= t <= t1 - args.interior_s:
                windows.append("interior")
            windows.append("hard" if any(a <= t <= b for a, b in hard) else "outside")
            for kp in range(tracklets.shape[1]):
                if (c, frame, kp) not in proto_pick:
                    continue  # the prototype did not decide this keypoint-frame (a keypoint it does not track)
                x, y, _ = proto_pick[(c, frame, kp)]
                proto_has = not np.isnan(x)
                trk, rank = int(tracklets[i, kp]), int(ranks[i, kp])
                run_pos = None
                if trk != -1:
                    for cx, cy, crank, ctrk in cand_by.get((c, frame, kp), []):
                        if crank == rank and (trk == -2 or ctrk == trk):
                            run_pos = (cx, cy)
                            break
                run_has = run_pos is not None
                if not proto_has and not run_has:
                    kind = "neither picked"
                elif proto_has and run_has:
                    if np.hypot(x - run_pos[0], y - run_pos[1]) <= args.tol_px:
                        kind = "same candidate"
                    elif not near((x, y), cand_by.get((c, frame, kp), [])):
                        kind = "differs: prototype's candidate is not in the run's"
                    elif not near(run_pos, proto_by.get((c, frame, kp), [])):
                        kind = "differs: run's candidate is not in the prototype's"
                    else:
                        kind = "differs: different choice"
                elif proto_has:
                    kind = "prototype picked, run did not"
                else:
                    kind = "run picked, prototype did not"
                for w in windows:
                    add(w, kind)

    kinds = ["same candidate", "neither picked", "differs: different choice", "differs: prototype's candidate is not in the run's",
             "differs: run's candidate is not in the prototype's", "prototype picked, run did not", "run picked, prototype did not"]
    print(f"{len(same_crop)} frames with the same crop; picks of {args.run} against {args.prototype_picks}")
    print(f"{'':54s}" + "".join(f"{w:>11s}" for w in ("all", "interior", "hard", "outside")))
    for kind in kinds:
        print(f"{kind:54s}" + "".join(f"{tallies.get(w, {}).get(kind, 0):11d}" for w in ("all", "interior", "hard", "outside")))
    print(f"{'same, of keypoint-frames with a pick in either':54s}" + "".join(
        f"{100 * _share(tallies.get(w, {})):10.2f}%" for w in ("all", "interior", "hard", "outside")))
    print(f"{'same, counting neither picked':54s}" + "".join(
        f"{100 * _share(tallies.get(w, {}), count_neither=True):10.2f}%" for w in ("all", "interior", "hard", "outside")))
    return 0


def _share(t: dict[str, int], count_neither: bool = False) -> float:
    total = sum(t.values()) - (0 if count_neither else t.get("neither picked", 0))
    same = t.get("same candidate", 0) + (t.get("neither picked", 0) if count_neither else 0)
    return same / total if total else float("nan")


def cmd_tracker(args) -> int:
    cfg, conn = load_set(), open_db(args)
    ranges = cfg["ranges"]["range3"]
    hard = [(w["t0"], w["t1"]) for w in cfg["hard_windows"]]
    ref = cfg["tracker"]["reference"]
    print(f"{'subject':8s} {'run':10s} {'window':8s} {'n':>8s} {'mean cm':>8s} {'median':>8s} {'p90':>8s}")
    runs = dict(r.split("=", 1) for r in args.run) if args.run else cfg["tracker"][args.runs]
    for subject, run in runs.items():
        for window, spans in (("range", [(TRACKER_FROM_S, ranges["t1"])]), ("hard", hard)):
            r = kp_gt.tracker_comparison(conn, run, ref[subject], spans)
            print(f"{subject:8s} {run:10s} {window:8s} {r['n']:8d} {r['mean']:8.1f} {r['median']:8.1f} {r['p90']:8.1f}")
    return 0


def cmd_track_picks(args) -> int:
    """Which candidate a tracking run used per keypoint-frame, scored against the labels of a range.

    Counts the share of labelled keypoint-frames, among those the run took a candidate for, where the used
    candidate lies within ``tolerance_box_height`` of the label, next to the strongest maximum of the same
    keypoint-frames.
    """
    cfg, root, conn = load_set(), eval_root(args), open_db(args)
    es, rng = cfg["evaluation_set"], cfg["ranges"][args.range]
    print(f"{'subject':8s} {'run':10s} {'labels':>7s} {'tracker':>8s} {'strongest':>10s} {'observations traced':>20s}")
    for item in args.run:
        subject, run = item.split("=", 1)
        d = root / rng["dir"]
        cand = kp_gt.read_candidates(d / f"candidates_subject{subject}.npz")
        labels = kp_gt.read_labels(d / f"labels_subject{subject}.npz", d / f"units_subject{subject}.json", rng["units"])
        tracker, strongest, counts = kp_gt.tracker_picks(conn, run, cand.cameras)
        skip = rng["skip_cameras"].get(str(subject), [])
        share = {}
        for name, picks in (("tracker", tracker), ("strongest", strongest)):
            m = kp_gt.pick_metrics(picks, labels, cand, t0=rng["t0"], t1=rng["t1"], tolerance=es["tolerance_box_height"],
                                   same_px=es["same_candidate_px"], skip_cameras=skip)
            n = max(m["n_visible"] * (1.0 - m["no_pick"]), 1.0)
            share[name] = (m["within_tol"] * m["n_visible"] / n, n)
        print(f"{subject:8s} {run:10s} {share['tracker'][1]:7.0f} {100 * share['tracker'][0]:7.1f}% "
              f"{100 * share['strongest'][0]:9.1f}% {counts['matched']:10d} of {counts['used']}")
    return 0


def cmd_events(args) -> int:
    cfg, root, conn = load_set(), eval_root(args), open_db(args)
    es = cfg["evaluation_set"]
    events = json.loads((root / es["events"]).read_text())
    base, other = cfg["tracker"]["baseline"], cfg["tracker"][args.runs]
    dist, dist_any = {}, {}
    print(f"{'event':14s} {'subject':8s} {'window':11s} {'part':6s} {'baseline vs ' + args.runs + ' (cm): part':>34s} {'any joint':>10s}")
    for e in events:
        s = str(e["subject"])
        if s not in other or e["t0"] < 43.5:
            continue
        dist[e["id"]] = kp_gt.event_distance(conn, base[s], other[s], e)
        dist_any[e["id"]] = kp_gt.event_distance(conn, base[s], other[s], {**e, "part": "whole"})
        print(f"{e['id']:14s} {s:8s} {e['t0']:4.1f}-{e['t1']:4.1f} {e['part']:6s} {dist[e['id']]:34.0f} {dist_any[e['id']]:10.0f}")
    for label, d in (("part", dist), ("any joint", dist_any)):
        flagged = kp_gt.events_flagged({k: v for k, v in d.items() if not np.isnan(v)}, es["failure_flag_cm"])
        print(f"flagged at {es['failure_flag_cm']:.0f} cm ({label}): {len(flagged)} of {len(d)} scorable events")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="directory of the frozen evaluation set (default $POSETRAK_KP_EVAL_ROOT)")
    ap.add_argument("--db", help="session database with the tracking runs (default $POSETRAK_KP_EVAL_DB)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    q = sub.add_parser("compare")
    q.add_argument("--detection-run", required=True, help="detection run (id prefix) with kp_candidates rows")
    q.add_argument("--subject", required=True, help="subject of the evaluation set; also the run's track id")
    q.add_argument("--range", default="range3")
    q.add_argument("--tol-px", type=float, default=0.5)
    q.add_argument("--crop-tol-px", type=float, default=1.0)
    q.add_argument("--cell-shift", type=float, default=0.5,
                   help="move the prototype file's positions by this share of a heatmap cell (the file has the cell "
                        "index at the cell's left edge, the decoder uses the cell centre); 0 compares them as stored")
    q = sub.add_parser("window-picks")
    q.add_argument("--detection-run", required=True, help="detection run (id prefix) the tracked sequence was made from")
    q.add_argument("--sequence", required=True, help="the pose observation sequence (id) the run tracked")
    q.add_argument("--run", required=True, help="tracking run (id prefix) that chose candidates over a time window")
    q.add_argument("--subject", required=True, help="subject of the evaluation set; also the detection run's track id")
    q.add_argument("--prototype-picks", required=True, help="npz of the prototype's picks (see the subcommand's help)")
    q.add_argument("--prototype-candidates", default=None,
                   help="candidate file the prototype's picks were made from, if not the evaluation set's file of the "
                        "subject (a file made from the run's own candidates compares the choice alone)")
    q.add_argument("--range", default="range3")
    q.add_argument("--tol-px", type=float, default=0.5)
    q.add_argument("--crop-tol-px", type=float, default=1.0)
    q.add_argument("--cell-shift", type=float, default=0.5, help="as for compare")
    q.add_argument("--interior-s", type=float, default=1.0,
                   help="the 'interior' column leaves out this many seconds at each end of the range, where the run "
                        "and the prototype do not average over the same window")
    p = sub.add_parser("picks")
    p.add_argument("--pick-sequence", action="append", default=[], metavar="SUBJECT=SEQUENCE",
                   help="score the picks of this pose observation sequence (person 0) for the subject instead of the "
                        "strongest maxima; give one per subject")
    q = sub.add_parser("track-picks")
    q.add_argument("--range", default="range1")
    q.add_argument("--run", action="append", required=True, metavar="SUBJECT=RUN",
                   help="tracking run (id prefix) of a subject of the range")
    for name in ("tracker", "events"):
        q = sub.add_parser(name)
        if name == "tracker":
            q.add_argument("--run", action="append", default=[], metavar="SUBJECT=RUN",
                           help="score these runs (an id prefix per subject) instead of a group of the set; "
                                "for runs of a work copy that the set does not name")
        q.add_argument("--runs", default="proposal_range3" if name == "tracker" else "proposal",
                       help="group of [tracker.*] runs in kp_eval_set.toml to score")
    args = ap.parse_args(argv)
    return {"verify": cmd_verify, "picks": cmd_picks, "tracker": cmd_tracker, "events": cmd_events,
            "compare": cmd_compare, "track-picks": cmd_track_picks, "window-picks": cmd_window_picks}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
