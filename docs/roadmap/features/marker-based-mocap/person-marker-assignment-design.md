# Person marker assignment — design

**Status**: Design proposal (2026-09-09). Refines and details
[marker-mocap-design.md](marker-mocap-design.md)'s phase 5 (identified body
markers) transitioning into phase 6 (anonymous dots on people) — that
document's §3 architecture, §5.1/§5.2 skeleton mechanisms, and §6.3 person-marker
UX are the basis this builds on, not a replacement for them.

## 0. Motivation — what changed since the base design was written

The base design's phase 5/6 split and §5.2's structure/scale/attachment-set
layering were written before any real person-marker capture existed. A real
capture (2026-09-06/08, see [status.md](status.md)) surfaced several concrete
facts the base design didn't yet have to account for:

- The physical rig puts **more than one marker per joint** (this capture: 2
  hip, 3 per knee — medial/lateral/front — ×2, 2 per ankle — medial/lateral —
  ×2, plus heel and big-toe ×2 = 16 leg markers), not the 1:1
  marker↔pose-keypoint mapping earlier examples implied.
- **Per-camera visibility is sparse and angle-dependent** — a retroreflective
  marker only throws a strong return toward a camera near its own light
  source's axis (confirmed, status.md 2026-09-06), and a marker on the far
  side of a limb from a given camera is often simply not visible at all, not
  just dim. Any assignment scheme has to treat "not visible from this camera
  right now" as the normal case, not an edge case.
- A naive per-keypoint independent nearest-candidate search — the obvious
  first thing to try — has a severe, structural failure mode: the same real
  candidate gets "matched" by several different keypoint names at once
  (confirmed at scale: 97.3% of frames with any match had this). A real
  assignment needs a proper one-to-one solve, not N independent searches.
- Background-subtraction-based detection has its own structural problems
  (fixed at the detector level, `background_mode='blacklist'`,
  see status.md) — this document assumes that fix is in place and focuses on
  what happens *after* detection produces a reasonably clean candidate
  stream.

## 1. Marker layout catalog

A **catalog** describes what markers exist and their rough anatomical role,
with no geometry — the person-marker analog of a `marker_body_definitions`
row's naming, without the rigid-body geometry a person doesn't have.

**Modular, not flat.** A catalog is composed of **modules** — a named,
parameterized marker group for one body region, written once and
mirrored:

```yaml
module: lower_leg
side_param: side   # instantiated once per {L, R}
markers:
  - name: knee_{side}_medial
    joint: knee.{side}
  - name: knee_{side}_lateral
    joint: knee.{side}
  - name: knee_{side}_front
    joint: knee.{side}
  - name: ankle_{side}_medial
    joint: ankle.{side}
  - name: ankle_{side}_lateral
    joint: ankle.{side}
  - name: heel_{side}
    joint: ankle.{side}
  - name: toe_{side}
    joint: ankle.{side}
```

A capture's actual marker plan is a **selection of modules matching the
study**, not a fixed universal set — gait analysis might use only
`lower_leg` + `pelvis`; an aikido wrist-lock study needs `hand` + `forearm`
at high density and can skip legs entirely. Different modules can carry
different implied accuracy/density for the same body without either being
wrong.

**The catalog is a convenience, not a constraint.** The final attachment set
must support markers beyond any catalog module — a session-specific ad hoc
marker (something taped on for one particular question) attaches directly
by `{name, joint, offset}` with no catalog entry at all. The catalog exists
to make seeding/naming fast for the common case, never to gate what's
representable.

**Storage**: proposed as plain versioned YAML files (a small library of
standard modules), not a new DB table, for now — the shape of "module" isn't
validated by real use yet, and this project's own precedent (§5.2's own
discussion) is to avoid committing storage machinery before a mechanism has
proven out. Revisit as a `marker_layout_catalogs` table (content-addressed,
like `marker_body_definitions`) once more than one real study has used this.

## 2. Attachment set: marker normal and backface culling

Two markers can sit close together in body-local space while facing opposite
directions (medial vs. lateral knee markers, a few cm apart, but visible
from disjoint sets of cameras at any given limb orientation). Distance-only
gating can't tell them apart; a camera on the medial side has no way to
reject the lateral marker as an assignment candidate purely on 2D pixel
proximity.

**Proposal**: each attachment-set entry gains a `normal: [x, y, z]` (parent
joint's local frame, same convention as `offset`) — the marker's
approximate outward-facing direction. At assignment time, compute the
world-frame normal from the joint's current estimated orientation, and gate
out any (marker, camera) pair where

```
dot(normal_world, unit(camera_pos - marker_pos_world)) < cos(max_angle)
```

before the candidate ever enters the cost matrix — cheap backface culling,
the same test 3D graphics uses to skip triangles facing away from the
viewer. `max_angle` wants some slack (not a hard 90°) since a marker can
still throw a visible return somewhat off its ideal normal.

**Deliberately not doing full occlusion analysis in v1**: ray-casting
against approximate body geometry (self-occlusion by another limb,
occlusion by another performer) would catch more real cases but is
significantly more complex to get right (needs a body-shape proxy, not just
joint positions) — backface culling via normal handles what's likely the
dominant error class (wrong-side-of-limb assignment) far more cheaply.
Flagged as a possible later refinement, not committed.

## 3. Assignment algorithm

Architecture stays what the base design already specifies (§3, algorithms
doc §3): a predicted-projection-gated cost-matrix assignment, reusing
`resolve_dot_assignment()`'s existing multi-camera Mahalanobis machinery —
already built and validated for props. What's genuinely new for a
non-rigid body is the **bootstrap**, since §4.1's rigid-template
registration (constant pairwise distances) doesn't apply to an articulated
person.

**Two stages:**

- **Stage A (bootstrap)** — before any attachment set exists, gate raw
  candidates against *pose keypoints* directly: loose, per-frame,
  one-to-one (Hungarian/`linear_sum_assignment` over the full
  keypoints×candidates cost matrix — already prototyped and validated this
  session to eliminate the multi-claim failure mode). For catalog markers
  with no directly-corresponding pose keypoint (the 3-per-knee, 2-per-ankle
  case), approximate an initial position via a guessed offset along the
  limb's own estimated local axes (e.g. hip→knee and knee→ankle vectors
  give an approximate sideways direction to offset "medial"/"lateral" from
  the knee keypoint). Noisy by construction — good enough to seed
  calibration, not trusted as final data.
- **Stage B (steady-state)** — once a calibration pass (§5 below) produces
  a real attachment set (real offsets *and* normals), switch to the tight,
  already-existing `resolve_dot_assignment()` gate — real FK-predicted
  marker projections plus the backface-culling pre-filter from §2 — for the
  rest of the trial. Same C++ mechanism props already use; no new solver.

**Scope for v1**: stays per-camera/per-body, matching the base design's own
incremental precedent (single-body scope in phase 2 before phase 4's
cross-subject generalization) — full cross-camera 3D-consistency fusion is
a later phase (§6), not attempted until single-camera-scoped assignment
proves insufficient.

**Review granularity**: both automatic proposal and manual correction
operate at the **tracklet** level (`DotTrackletLinker`, already built) as
the primary unit, not per-frame — a tracklet gets one proposed name and one
confidence/cost score; review is by exception (sort by lowest confidence),
matching R2.3's "drop, don't guess" philosophy at the UI level too.

## 4. Editing/correction UI

**Extend the existing keypoint timeline widget** (`keypoint_timeline_widget.py`)
rather than building a new panel — this is materially the same interaction
shape as reviewing pose-keypoint edits, just at tracklet instead of
per-frame granularity:

- One timeline row per marker name, rows grouped by body part/joint
  (mirroring the catalog's own module grouping — all `knee_L_*` rows
  together, etc.).
- Each raw detection shown as a small color-coded rectangle on its
  candidate's row/time position; a tracklet shown as an outline rectangle
  grouping its member detections (or, if visually simpler, one rectangle
  spanning the tracklet's own start-end range) — so a tracklet's temporal
  extent and identity are visible at a glance without opening it.
- Automatic assignments pre-fill a proposed marker row per tracklet, color
  intensity/opacity indicating confidence, so low-confidence tracklets are
  visually obvious to review first.
- Right-click (or equivalent) actions on a tracklet: **reassign** to a
  different marker name, **split** at a chosen frame (the escape hatch for
  a tracklet-linker mistake that silently carries one identity across what
  was actually two different physical dots).

**Manual candidate addition** (not just relabeling) for the case the
detector missed a marker entirely — the fast-swing sword-capture case from
earlier this session, where the true streak never crossed threshold at
all. Hoped-for scope: **no need to reposition existing detections**, only to
add missing ones. Interaction: click the streak's start, drag to its end —
this directly produces a `BlobCandidate`-shaped entry (position = midpoint,
`major_axis_px`/`minor_axis_px`/`dir_x`/`dir_y` computed from the drag
vector, exactly the fields the automatic streak detector already
populates) so a manually-added candidate flows through assignment and
tracking completely unmodified, no special-casing anywhere downstream.

## 5. Skeleton scaling

Keep scale (person-scale skeleton) and marker offsets (attachment set) as
**separate stored artifacts** per §5.2's own reuse argument (a marker
session's scale improvement should still reach markerless sessions) — but
produce both from **one joint calibration solve**: bone lengths + marker
offsets + marker normals + per-frame joint angles, jointly, over a
diverse-pose frame set (the bundle-adjustment approach sketched earlier
this session), then split the solve's output into the two stored artifacts
afterward.

Frame selection for that solve is fed by Stage A's bootstrap-labeled
tracklets, spanning a deliberately diverse range of poses (a single static
pose under-constrains offsets along some axes — already noted earlier this
session). No iterative re-calibration loop in v1 (re-running calibration
later with Stage B's better-assigned data) — one good pass should be
enough; revisit only if it proves insufficient in practice.

## 6. Phased prototyping/implementation plan

Each phase has its own validation check, matching the base design's own
phasing convention (§7). Ordered so each phase's output is real input to
the next, not built speculatively ahead of need.

| Phase | Delivers | Validation |
|---|---|---|
| **P1** (next) | Python prototype extending this session's validated one-to-one assignment from the 10 pose-keypoint-anchored slots to the full real catalog (16 leg markers for this rig), with simple anatomically-derived offset guesses for the multi-marker joints. Single camera, 2D only, no normal-culling yet. | Run against the real detection run + pose sequence already on disk; hand-check a sample of proposed assignments (does `knee_L_medial` land near a plausible medial-side candidate, not the front one?) the same way this session's ground-truth spot-checks worked. |
| **P2** | Add marker-normal-based backface culling (§2) to the same Python prototype, using limb-axis vectors from pose keypoints as a stand-in for real segment orientation. | Confirm it measurably reduces the class of error P1 can't catch alone — a medial-side marker wrongly proposed in a camera that geometrically can't see that side of the limb. |
| **P3** | Joint bundle-adjustment calibration prototype (§5) over a diverse-pose frame set drawn from P1/P2's bootstrap-labeled tracklets. | Recovered bone lengths plausible against known anatomy; reprojection residuals small across the calibration window's diverse poses, not just the frame(s) used to seed it. |
| **P4** | Port validated P1–P3 logic into the real pipeline: attachment-set loader + `input_tracks` binding (§5.1), `resolve_dot_assignment()` extended to consume person attachment sets (offsets + normals) the same way it already consumes prop marker bodies. | Real tracked run on the person-marker capture; tracked% and reprojection error compared against the markerless-only baseline, same two-sided check used throughout this project. |
| **P5** | Tracklet review UI (§4) as a `keypoint_timeline_widget.py` extension — reassign/split interactions, confidence-sorted review-by-exception. | A reviewer can find and fix a deliberately-seeded bad assignment through the UI alone, without touching the DB directly. |
| **P6** | Manual candidate addition (§4, streak click-drag) in the same UI. | A manually-added streak candidate is indistinguishable, downstream, from an automatically-detected one — same schema, same assignment/tracking code path. |
| **P7** (conditional, not committed) | Cross-camera/multi-view 3D-consistency fusion, generalizing P1–P4's per-camera scope. | Only pursued if P1–P4's per-camera-scoped assignment proves insufficient in practice — not built speculatively ahead of that evidence. |

Phase ordering rationale: P1 before P2 so the basic multi-marker assignment
is validated before adding a geometric filter on top of it (isolates which
mechanism is responsible for which improvement, matching this session's own
diagnostic discipline); P3 depends on P1/P2's labeled tracklets as its input
data, so it can't come first; P4 is the production port, deliberately after
every algorithmic piece is prototype-validated, matching this whole
project's established prototype→validate→port pattern; P5/P6 (UI) come
after the algorithm they're reviewing actually exists and produces
non-trivial output worth reviewing; P7 is explicitly conditional rather than
scheduled, per the base design's own "generalize once needed, not
speculatively" precedent for cross-subject scope.
