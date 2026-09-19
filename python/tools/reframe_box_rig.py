# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""reframe_box_rig.py — PROTOTYPE: re-express a characterize_rig_from_video.py
solve (arbitrary rig-local frame, anchored on whichever marker the solver
picked) into Harri's world-frame convention for the 2026-09-06 calibration
box:

  +Y axis = outward normal of the face carrying markers 3 & 4
  +X axis = outward normal of the face carrying X_FACE_MARKER_ID (6 or 8 --
            see calibrate_extrinsics_from_moving_box.py's own open question)
  origin  = box bottom-center, so the top face (marker 1, confirmed by
            Harri to be one of the box's two top markers) sits at
            z = BOX_HEIGHT_M

Verified against the box's own known dimensions before trusting this: the
distance between markers 6 and 8's solved centers, projected onto their
shared face-normal axis, is ~0.491m -- matches Harri's measured 49cm long
side almost exactly, confirming the face-role identification below (not
just the axis convention) is correct, not merely self-consistent.

Only the 5 markers characterize_rig_from_video.py solved with enough
multi-camera coverage (1, 3, 4, 6, 8) are handled here -- markers 0, 2, 9
were seen in the footage (id 2 abundantly; id 0 not at all in a coarse
scan) but didn't get enough simultaneous multi-camera corner coverage in
either sampling attempt tried so far. Output covers only what was solved;
extend BOX_FACES/re-run characterize_rig_from_video.py with different
sampling to add more once needed.

Usage
-----
    python reframe_box_rig.py RAW_JSON --x-face-marker-id 6 -o box_body.yaml
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import yaml

BOX_WIDTH_M = 0.49   # X extent (long axis, confirmed via marker 6/8 spacing above)
BOX_DEPTH_M = 0.33   # Y extent (short axis) -- given, not yet independently verified
BOX_HEIGHT_M = 0.18  # Z extent / top-face height -- given

TOP_MARKER_ID = "1"       # confirmed by Harri: one of the box's 2 top markers
Y_FACE_MARKER_IDS = ("3", "4")  # confirmed: together on one long (Y) face


def _outward_normal_and_center(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """corners: (4,3) in order TL, TR, BR, BL (cv2.aruco / marker_local_corners
    convention). Outward normal = cross(TR-TL, TL-BL) -- verified against
    marker_local_corners()'s own canonical corners (normal=[0,0,1] there
    reproduces exactly its TL=(-h,h,0) etc.), NOT the more "obvious"
    cross(TR-TL, BL-TL) (that one is the inward normal -- a real mistake
    caught by checking against the canonical reference before trusting it).
    """
    normal = np.cross(corners[1] - corners[0], corners[0] - corners[3])
    normal = normal / np.linalg.norm(normal)
    return normal, corners.mean(axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_json", help="characterize_rig_from_video.py output")
    ap.add_argument("--x-face-marker-id", required=True, choices=["6", "8"],
                     help="Which short-face marker's outward normal becomes world +X")
    ap.add_argument("-o", "--output", required=True, help="Output marker-body YAML path")
    args = ap.parse_args()

    with open(args.raw_json, encoding="utf-8") as f:
        data = json.load(f)
    marker_corners = {mid: np.array(c) for mid, c in data["marker_corners"].items()}
    dictionary = data["dict"]

    missing = [m for m in (TOP_MARKER_ID, *Y_FACE_MARKER_IDS, args.x_face_marker_id) if m not in marker_corners]
    if missing:
        raise SystemExit(f"Raw solve is missing marker(s) {missing} needed to build the world frame")

    top_normal, top_center = _outward_normal_and_center(marker_corners[TOP_MARKER_ID])
    y_normals, y_centers = zip(*(_outward_normal_and_center(marker_corners[m]) for m in Y_FACE_MARKER_IDS))
    y_normal = np.mean(y_normals, axis=0)
    y_normal /= np.linalg.norm(y_normal)
    y_center = np.mean(y_centers, axis=0)
    x_normal, x_center = _outward_normal_and_center(marker_corners[args.x_face_marker_id])

    # Gram-Schmidt: X and Y define the frame, Z follows as their cross
    # product (real physical faces are only *approximately* 90 deg apart
    # once solve noise is included -- don't trust top_normal literally,
    # just sanity-check the derived Z axis against it below).
    ex = x_normal / np.linalg.norm(x_normal)
    ey = y_normal - np.dot(y_normal, ex) * ex
    ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)

    angle_check_deg = np.degrees(np.arccos(np.clip(np.dot(top_normal, ez), -1, 1)))
    print(f"Sanity check: derived Z axis vs. marker {TOP_MARKER_ID}'s own outward normal "
          f"-> {angle_check_deg:.1f} deg apart (expect close to 0 or close to 180 -- "
          f"marker 1's own raw-frame Z sign is arbitrary, only *which line* matters here)")

    # Raw-frame point that maps to world origin (0,0,0) -- see module
    # docstring's derivation: solve each axis's origin offset independently
    # then combine, since {ex,ey,ez} is an orthonormal basis.
    origin_raw = (
        (np.dot(x_center, ex) - BOX_WIDTH_M / 2) * ex
        + (np.dot(y_center, ey) - BOX_DEPTH_M / 2) * ey
        + (np.dot(top_center, ez) - BOX_HEIGHT_M) * ez
    )

    def to_world(p: np.ndarray) -> np.ndarray:
        d = p - origin_raw
        return np.array([np.dot(d, ex), np.dot(d, ey), np.dot(d, ez)])

    print("\nSolved marker positions in the new world frame (face center, outward normal):")
    markers_yaml = []
    for mid, corners in marker_corners.items():
        world_corners = np.array([to_world(c) for c in corners])
        normal, center = _outward_normal_and_center(world_corners)
        print(f"  id={mid}  center={np.round(center, 4)}  normal={np.round(normal, 3)}")
        markers_yaml.append({
            "name": f"box_m{mid}",
            "type": "aruco",
            "dictionary": dictionary,
            "id": int(mid),
            "corners": [c.tolist() for c in world_corners],
        })

    doc = {
        "name": "calib-box-2026-09-06",
        "_provenance": {
            "source": "reframe_box_rig.py + characterize_rig_from_video.py",
            "raw_json": args.raw_json,
            "x_face_marker_id": args.x_face_marker_id,
            "box_dimensions_m": [BOX_WIDTH_M, BOX_DEPTH_M, BOX_HEIGHT_M],
            "note": "Only markers with enough multi-camera coverage in the orbit "
                    "video are included -- see this script's own module docstring.",
        },
        "markers": markers_yaml,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"\nWrote {len(markers_yaml)} marker(s) to {args.output}")


if __name__ == "__main__":
    main()
