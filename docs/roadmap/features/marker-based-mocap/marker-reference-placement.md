# Marker placement on a reference body

The catalog's marker modules name the slots of a marker set and the bone each rides on.
Where a marker sits on that bone, and which way it faces, is geometry: it is easier to
place on a model of a body than to write down. This note describes how the placements are
made in Blender and brought back into the catalog, and what they are used for.

## What a placement is

Per marker, in a frame that depends only on the skeleton's rest pose (so it carries over to
any skeleton with the same topology, scaled or not):

| field | meaning |
|---|---|
| `along` | position along the parent bone as a fraction of its length, joint = 0, far end = 1 |
| `lateral`, `anterior` | metres off the bone along the two directions below |
| `normal` | outward surface normal as (long, lateral, anterior) components |

The frame of a bone comes from its direction `d` (joint to far end) and the body axes
(right, front, up; from the shoulders and the hips-to-head direction). *Anterior* is the front
direction with its component along the bone removed (for a bone that points forward, the foot,
up is used instead: dorsal); *lateral* is `d x anterior`. No bone roll is involved, so a
Character Creator armature and posetrak's skeleton give the same numbers for the same point on
the body. A marker mirrored across the body negates `lateral` and the lateral part of `normal`,
which is what `mirror: true` in a module does.

Code: `posetrak/markers/reference.py` (frames, conversion, writing into modules) and
`python/tools/blender/blender_marker_reference.py` (the Blender side).

## Workflow

1. `python tools/marker_reference.py make --session ... --tracking-run ... --attachment-set ...
   --output reference.json` writes the rest-pose bones and the current placements.
2. In Blender (5.2 tested), with the character's mesh in the scene and the armature (the
   one in the file, or a template armature aligned to the character):
   `blender --python blender_marker_reference.py -- build reference.json`, then
   `-- snap <mesh object>` moves every marker onto the nearest surface point and points its
   arrow along the surface normal. Adjust by hand and snap again.
3. `-- export placements.json`, then
   `python tools/marker_reference.py apply placements.json` rewrites `along`, `lateral`,
   `anterior` and `normal` of the markers in `catalog/modules/*.marker-module.yaml`, keeping
   the files' comments. Left twins are compared with the mirror image of the right-hand
   placement and a difference above 2 cm is reported.

The bones' positions come from the armature's *current pose*, not its rest pose, so an
armature that was posed or animated onto the model (a BVH of a tracking run, aligned to the
character) works even when its rest pose is not a standing one. The body's right, front and up
are worked out from the bones, so a scene's axes do not matter (a Blender scene facing -Y with the
subject's right towards -X is fine).

The round trip through Blender reproduces a placement to better than a millimetre. Placements
made from a calibration also give a check against the model: the calibrated markers of the first
full-body calibration lay a median of 1.6 cm from the skin of a Character Creator model of the
performer, and all but a handful within 4 cm.

## Use

- The expected place of a marker is a point on the body, so one small tolerance replaces the
  per-marker `reach` that was needed while only `along` was known.
- A marker can only be seen from the side its normal faces. A calibration track whose cameras
  are on the wrong side cannot be that slot, which separates front from back and left from right
  markers, and the tracker already uses a marker's normal to cull predictions that face away
  from a camera.
- On another person the reference is scaled: `along` follows the bone, and `lateral` and
  `anterior` are multiplied by the ratio of that person's bone length to the reference's.
