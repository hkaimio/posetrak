# Decisions about whole dot tracklets

The tracker assigns the dots of a frame to marker slots one frame at a time, from its own
prediction of where each marker is. A dot tracklet (one detector-linked dot followed through a
camera's frames) carries more evidence than any single frame: its path over a second is much
easier to match to one marker than its position in one frame. This note describes how decisions
about whole tracklets reach the tracker, where they come from, and what they were measured to do.

## Where the decisions come from

- **Review decisions.** The dot-assignment review (`app/ui/dot_review.py`) stores, per trial,
  rules for a tracklet over a range of its video frames: `assign` (the tracklet is this marker),
  `forbid` (it is not this marker), `reject` (it is not a marker) and `other` (it is a marker of
  another subject). A later rule overrides an earlier one. Table `dot_assignment_edits`.
- **Automatic proposals.** `posetrak.markers.proposals` assigns each tracklet to the slot whose
  predicted path, from the stored poses of an earlier tracking run, is closest to it, and tags
  the assignment `high`, `medium` or `review` from the distance, the margin to the next slot and
  whether a second camera's tracklet of the same slot triangulates with it. Table
  `dot_assignment_proposals`.

## How the tracker uses them

`posetrak track run --dot-detection-run RUN [--dot-edits-trial TRIAL] [--dot-proposals]` writes the
decisions as a CSV (`camera_label,tracklet_id,first_frame,last_frame,kind,marker_name`) and
passes it to `posetrak-tracker track --tracklet-assignments`. A high-confidence proposal is written
as `assign`, any other proposal as `suggest`, and the review decisions after them so that they win.

In the shared dot assignment (`dot_assignment.hpp`) a `TrackletAssignmentModifier` acts on every
(candidate, slot) pair of a candidate that has a tracklet id, for the video frame the candidate was
detected in:

| rule in force | effect |
|---|---|
| `assign` marker M | the pair with M costs `1/dot_assigned_factor` (default 1000) of its Mahalanobis cost, so it passes the gate whatever this frame's prediction error is; every other marker is excluded for the candidate |
| `reject`, `other` | the candidate is excluded from every marker |
| `forbid` marker M | the pair with M is excluded |
| `suggest` marker M | the pair with M costs `1/dot_suggested_factor` (default 3) of its cost |

An `assign` or `suggest` of a marker that a later `forbid` rules out does not count. A candidate
without a tracklet id, or whose tracklet has no rule covering its frame, is assigned as before.

## What it was measured to do

Scored against human-reviewed ground truth on footage the marker placements were not fitted on
(rolling falls, 14 s, four cameras, 45 markers), by whether the tracker used the labelled dot for the
labelled marker in a frame (a dot counts when the tracker's used observation lies within 1.5 px of
any of the same tracklet's detections within three video frames of the labelled one):

| tracker input | precision | recall |
|---|---|---|
| per-frame assignment only | 72.5% | 52.2% |
| per-frame assignment with the gate raised from 9.21 to 25 | 70.7% | 64.4% |
| automatic proposals (high confidence followed, others suggested) | 91.9% | 80.0% |
| the review decisions themselves | 100.0% | 98.7% |

Where the per-frame assignment fails is not obscure: of the labelled tracklet-frames, about 60%
won their marker, 24% were costed but gated out (squared Mahalanobis cost around 16 against a gate
of 9.21, because the filter's pose uncertainty is too small for fast motion) and 15% went to another
marker. A tracklet decision removes both failures. Recall below 100% with the review decisions is the
dots the tracker never received as candidates in that step, and the pairs the one-dot-per-slot rule
does not allow when two tracklets are given the same marker at once.

Comparisons of tracker runs against a ground truth must match a dot by its tracklet over neighbouring
frames, not by its exact position: the tracker runs at twice the video frame rate and takes the
detection of the neighbouring frame in half of its steps, which at ten pixels of movement per frame
puts the used dot outside any tight positional match and understates recall during fast motion.
