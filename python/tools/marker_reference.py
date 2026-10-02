# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""marker_reference.py — exchange marker placements with Blender (see blender/blender_marker_reference.py).

Make the document a Blender scene is built from, starting from a calibrated attachment set::

    python tools/marker_reference.py make --session session.db --tracking-run RUN \\
        --attachment-set catalog/body.calibrated.yaml --output reference.json

Write the placements exported from Blender into the catalog's marker modules::

    python tools/marker_reference.py apply placements.json [--catalog-dir catalog/modules]

A marker written once for both sides (``mirror: true``) takes the right-hand placement; the
left-hand one is only compared with its mirror image, and a difference of more than 2 cm is
reported.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.db.skeleton_layout import SkeletonLayout  # noqa: E402
from posetrak.markers.bones import BodyPose  # noqa: E402
from posetrak.markers.catalog import catalog_dir, load_marker_module  # noqa: E402
from posetrak.markers.reference import (  # noqa: E402
    apply_placements, bone_lengths, placements_from_attachment_set, reference_document, rescale_along, rest_bones,
)

MIRROR_TOLERANCE_M = 0.02


def make(args) -> None:
    conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    run = conn.execute("SELECT id FROM tracking_runs WHERE id LIKE ?", (args.tracking_run + "%",)).fetchone()
    pose = BodyPose(conn, run["id"])
    bones, frames = rest_bones(pose.layout, pose.layout.decode_state_blob(pose._blobs[0]))
    parents = {j.name: j.parent for j in pose.layout.joints}
    parents["hips"] = None
    attachment = yaml.safe_load(Path(args.attachment_set).read_text(encoding="utf-8"))
    markers = placements_from_attachment_set(attachment, bones, frames)
    Path(args.output).write_text(json.dumps(reference_document(bones, parents, markers), indent=1), encoding="utf-8")
    print(f"wrote {args.output}: {len(bones)} bones, {len(markers)} markers")


def apply(args) -> None:
    placements = {m["name"]: m for m in json.loads(Path(args.placements).read_text(encoding="utf-8"))["markers"]}
    if args.session:
        conn = sqlite3.connect(f"file:{args.session}?mode=ro", uri=True)
        yaml_text = conn.execute("SELECT yaml_content FROM skeletons WHERE id LIKE ?", (args.skeleton + "%",)).fetchone()[0]
        lengths = bone_lengths(SkeletonLayout(yaml_text))
        placements = {n: rescale_along(p, lengths[p["parent_joint"]]) for n, p in placements.items()}
    directory = Path(args.catalog_dir) if args.catalog_dir else catalog_dir()
    total, missing = 0, set(placements)
    for path in sorted(directory.glob("*.marker-module.yaml")):
        text = path.read_text(encoding="utf-8")
        module = load_marker_module(text, source=str(path))
        new, changed = apply_placements(text, placements)
        if changed:
            path.write_text(new, encoding="utf-8")
        total += len(changed)
        names = {s.name for s in module.slots}
        missing -= names
        for slot in module.slots:                                   # compare each left twin with its mirror image
            if slot.name.endswith("_L") and slot.name[:-2] + "_R" in placements and slot.name in placements:
                right, left = placements[slot.name[:-2] + "_R"], placements[slot.name]
                gap = np.linalg.norm([left["lateral"] + right["lateral"], left["anterior"] - right["anterior"]])
                if gap > MIRROR_TOLERANCE_M:
                    print(f"  {slot.name}: differs from the mirror image of {slot.name[:-2]}_R by {100 * gap:.1f} cm")
    print(f"updated {total} markers in {directory}")
    if missing:
        print("placements without a catalog marker:", ", ".join(sorted(missing)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("make")
    p.add_argument("--session", required=True)
    p.add_argument("--tracking-run", required=True)
    p.add_argument("--attachment-set", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=make)
    p = sub.add_parser("apply")
    p.add_argument("placements")
    p.add_argument("--session", default=None, help="With --skeleton: re-express 'along' on this skeleton's bone lengths "
                   "(needed when the placements carry the armature's bone_length).")
    p.add_argument("--skeleton", default=None, help="Skeleton id (prefix) in the session.")
    p.add_argument("--catalog-dir", default=None)
    p.set_defaults(func=apply)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
