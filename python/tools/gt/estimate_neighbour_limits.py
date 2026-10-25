# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Estimate the skeleton-neighbour distance limits of a tracked sequence from the picks that agree across cameras.

The shipped table of limits (``posetrak.detection.soma77_neighbour_limits``) comes from operator labels. The
alternative is to estimate the same percentile in the run itself, from the candidates the multi-camera consensus
flagged (``posetrak-tracker track --kp-choice-dump``): the 2D distance between the two keypoints of a pair in the
same camera and frame, in box heights, 1.15 times its 98th percentile, for the pairs with more than 50 samples.

    python python/tools/gt/estimate_neighbour_limits.py --db session.db --sequence SEQ --dump dump.csv \\
        --model-out estimate_model.csv

prints the estimate against the shipped table and, with ``--model-out``, writes a choice model file (the
estimate for the pairs it covers, the shipped limit for the rest, and the keypoint classes) that
``track run --kp-choice-model`` takes.
"""
from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kp_gt  # noqa: E402

from posetrak.detection.kp_choice_model import soma77_keypoint_class  # noqa: E402
from posetrak.detection.pose_models import SOMA77  # noqa: E402
from posetrak.detection.soma77_neighbour_limits import SOMA77_NEIGHBOUR_LIMITS  # noqa: E402

BOX_HEIGHT_PER_CROP_SCALE = 160.0
MARGIN = 1.15
PERCENTILE = 98.0
MIN_SAMPLES = 50


def estimate(db: str, sequence: str, dump: str, person: int = 0) -> dict[tuple[str, str], tuple[float, int]]:
    """``{(child, parent): (limit, samples)}`` from the consensus-flagged candidates of the dump."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    seq = conn.execute("SELECT id FROM pose_observation_sequences WHERE id LIKE ?", (sequence + "%",)).fetchone()[0]
    agreed: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    with open(dump, newline="") as f:
        for row in csv.DictReader(f):
            if row["consistent"] == "1" and row["removed"] == "0":
                agreed.setdefault((row["camera_instance_id"], int(row["video_frame"]), int(row["keypoint"])), []).append(
                    (int(row["rank"]), int(row["tracklet"])))
    rows = {(cam, frame): (kp_gt.decode_kp_candidates_blob(bytes(blob)), float(scale))
            for cam, frame, blob, scale in conn.execute(
                "SELECT camera_instance_id, video_frame, kp_blob, COALESCE(noise_scale, 1.0) FROM pose_observations "
                "WHERE sequence_id = ? AND person_id = ? AND source = 'kp_candidates'", (seq, person))}
    pairs = [(c, p) for c, p in enumerate(SOMA77.parents) if p >= 0]
    distances: dict[tuple[int, int], list[float]] = {pair: [] for pair in pairs}
    for (cam, frame), (table, scale) in rows.items():
        position = {}
        for kp, x, y, _, rank, tracklet in table:
            if (int(rank), int(tracklet)) in agreed.get((cam, frame, int(kp)), ()):
                position[int(kp)] = (float(x), float(y))
        height = scale * BOX_HEIGHT_PER_CROP_SCALE
        for child, parent in pairs:
            if child in position and parent in position:
                a, b = position[child], position[parent]
                distances[(child, parent)].append(float(np.hypot(a[0] - b[0], a[1] - b[1])) / height)
    names = SOMA77.keypoint_names
    return {(names[c], names[p]): (round(MARGIN * float(np.percentile(v, PERCENTILE)), 4), len(v))
            for (c, p), v in distances.items() if len(v) > MIN_SAMPLES}


def write_model(path: Path, limits: dict[tuple[str, str], float]) -> None:
    source = SOMA77.source
    lines = [f"# choice model of the {source} pose model with limits estimated in the run"]
    lines += [f"pair,{source}:{a},{source}:{b},{v}" for (a, b), v in limits.items()]
    lines += [f"class,{source}:{n},{soma77_keypoint_class(n)}" for n in SOMA77.keypoint_names]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--dump", required=True, help="CSV written by posetrak-tracker track --kp-choice-dump")
    ap.add_argument("--person", type=int, default=0)
    ap.add_argument("--model-out", default=None)
    args = ap.parse_args()
    est = estimate(args.db, args.sequence, args.dump, args.person)
    print(f"{'pair':36s} {'shipped':>8s} {'estimate':>9s} {'samples':>8s} {'ratio':>6s}")
    ratios = []
    for pair, (limit, n) in est.items():
        shipped = SOMA77_NEIGHBOUR_LIMITS.get(pair)
        ratio = limit / shipped if shipped else float("nan")
        ratios.append(ratio)
        print(f"{pair[0] + ' - ' + pair[1]:36s} {shipped if shipped else float('nan'):8.3f} {limit:9.3f} {n:8d} {ratio:6.2f}")
    only_shipped = [p for p in SOMA77_NEIGHBOUR_LIMITS if p not in est]
    print(f"\n{len(est)} pairs estimated, {len(SOMA77_NEIGHBOUR_LIMITS)} in the shipped table ({len(only_shipped)} of them without an estimate)")
    r = np.array([x for x in ratios if x == x])
    if len(r):
        print(f"estimate / shipped on the {len(r)} pairs in both: median {np.median(r):.2f}, range {r.min():.2f}..{r.max():.2f}")
    if args.model_out:
        merged = {**SOMA77_NEIGHBOUR_LIMITS, **{p: v for p, (v, _) in est.items() if p in SOMA77_NEIGHBOUR_LIMITS}}
        write_model(Path(args.model_out), merged)
        print("wrote", args.model_out)


if __name__ == "__main__":
    main()
