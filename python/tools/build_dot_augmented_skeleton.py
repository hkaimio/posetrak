# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""build_dot_augmented_skeleton.py -- Step 1 of the real-tracker
integration (status.md, 2026-09-12): converts a calibrated marker
attachment set (fit_calibrated_attachment_set.py's B4/B5 output) into
real skeleton `markers:` entries, appended to a copy of the base
skeleton the calibration was fit against, and registers the result as a
new skeleton row.

No schema change needed on the C++/data side -- confirmed directly in
`skeleton_loader.cpp`: a marker's `track`/`landmark`/`normal` fields are
already parsed today, and `input_tracks:` entries accept any `type`
string (the loader never validates it against a fixed set), so
`unlabeled_points` -- the exact type `Tracker::predict_dot_slot_
predictions()` filters on -- is already a legal skeleton, no new C++
needed for the *data* side (only for the general/articulated
MarkerPrediction computation itself, built the same day in tracker.cpp/
ukf.cpp).

The base skeleton's own joints and existing (openpose-keypoint) markers
are carried through unchanged -- this produces a skeleton that can still
track normally from pose keypoints *and* now also has named dot slots,
not a replacement for the original.

Usage:
    python tools/build_dot_augmented_skeleton.py \\
        --session /path/to/session.db \\
        --base-skeleton-id 0d08a889b32acc13298ecba139ac585792f4221679b3238d0be40c356c945f04 \\
        --calibrated-attachment-set /path/to/leg.calibrated.yaml \\
        --output /path/to/reallusion-no-waist.dot-augmented.yaml
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from posetrak.db.manage_skeleton import import_skeleton_str  # noqa: E402
from posetrak.markers.catalog import check_module_fits_skeleton  # noqa: E402

_DOTS_TRACK_ID = "dots"


def build_dot_augmented_skeleton(base_yaml_content: str, calibrated_doc: dict) -> str:
    """Returns the merged skeleton YAML content as a string.

    Raises MarkerSetError if the attachment set was not written for the base skeleton's
    topology, or names joints the skeleton does not have.
    """
    check_module_fits_skeleton(calibrated_doc, base_yaml_content)
    skeleton = yaml.safe_load(base_yaml_content)

    input_tracks = skeleton.setdefault("input_tracks", [])
    if not any(t.get("id") == _DOTS_TRACK_ID for t in input_tracks):
        input_tracks.append({"id": _DOTS_TRACK_ID, "type": "unlabeled_points"})

    markers = skeleton.setdefault("markers", [])
    existing_names = {m["name"] for m in markers}
    for m in calibrated_doc["markers"]:
        if m["name"] in existing_names:
            raise ValueError(
                f"base skeleton already has a marker named '{m['name']}' -- "
                "refusing to silently shadow it"
            )
        markers.append({
            "name": m["name"],
            "parent": m["parent_joint"],
            "offset": m["offset"],
            "normal": m["normal"],
            "track": _DOTS_TRACK_ID,
            "landmark": m["name"],
        })

    # REUSE-IgnoreStart -- this builds the *output* file's own SPDX header as a
    # string; written this way (concatenated pieces) so `reuse lint` doesn't
    # mistake it for a badly-formed header on *this* .py file.
    spdx_tag = "SPDX" "-License-Identifier"
    spdx_header = (
        "# SPDX-FileCopyrightText: 2026 Harri Kaimio\n"
        "#\n"
        f"# {spdx_tag}: Apache-2.0\n\n"
    )
    # REUSE-IgnoreEnd
    return spdx_header + yaml.safe_dump(skeleton, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--base-skeleton-id", required=True)
    ap.add_argument("--calibrated-attachment-set", required=True)
    ap.add_argument("--output", required=True, help="Also written to this file for inspection/version control.")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    conn = sqlite3.connect(args.session)  # writable -- registering a new skeleton row
    conn.row_factory = sqlite3.Row

    base_row = conn.execute(
        "SELECT yaml_content, name FROM skeletons WHERE id = ?", (args.base_skeleton_id,)
    ).fetchone()
    if base_row is None:
        raise SystemExit(f"no skeleton with id {args.base_skeleton_id!r} in {args.session}")

    calibrated_doc = yaml.safe_load(Path(args.calibrated_attachment_set).read_text())
    merged_yaml = build_dot_augmented_skeleton(base_row["yaml_content"], calibrated_doc)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(merged_yaml)

    name = args.name or f"{base_row['name']} + dot slots ({Path(args.calibrated_attachment_set).stem})"
    new_id = import_skeleton_str(
        conn, merged_yaml, name=name, parent_id=args.base_skeleton_id,
        source=f"build_dot_augmented_skeleton.py from {args.calibrated_attachment_set}",
        notes=f"Base skeleton {args.base_skeleton_id} + {len(calibrated_doc['markers'])} calibrated "
              f"dot-slot markers ({calibrated_doc['calibration'].get('method', 'B4/B5')}, "
              f"{calibrated_doc['calibration'].get('date', 'undated')}).",
    )
    conn.commit()

    n_markers = len(yaml.safe_load(merged_yaml)["markers"])
    print(f"base skeleton: {args.base_skeleton_id} ({base_row['name']!r})")
    print(f"added {len(calibrated_doc['markers'])} dot-slot markers -> {n_markers} markers total")
    print(f"wrote {args.output}")
    print(f"registered new skeleton_id: {new_id}")


if __name__ == "__main__":
    main()
