# Tracklet editing for keypoint candidates

Extends the keypoint editor ([keypoint-editing-design.md](keypoint-editing-design.md)) for
sequences that carry keypoint candidates
([keypoint-candidates-design.md](../keypoint-candidates/keypoint-candidates-design.md)). The unit
of editing becomes the tracklet: one keypoint's maxima linked through one camera's frames.

## 1. Why tracklets

- Per-keypoint, per-frame editing (`pose_observation_edits`) does not scale to the failures that
  matter. Correcting a failure window of a fall took about 1800 keypoint-frame edits per person in a
  simulation on the labelled trial; grouping the wrong picks into continuous segments per camera and
  keypoint gave 452 and 245 segments for two persons, and grouping segments of one camera that overlap
  in time and lie within two joints in the skeleton gave 149 and 122 groups merged over cameras, with
  half of the wrong frames in the 9 and 5 largest groups.
- Correcting single keyframes barely helps (the filter does not hold the correction and the wrong
  picks after it pull the skeleton back); correcting whole failure windows reaches the label-fed
  level. A correction has to cover a time range.
- The identity errors happen along smooth tracklets (a head tracklet continues onto the neighbour's
  head), so a tracklet must be splittable at a frame.

## 2. Operations and their storage

Tracklet decisions use the dot decision table, `dot_assignment_edits`, which is keyed by
(trial, detection run, subject, camera, tracklet id, frame range) and is read by the tracker as
`TrackletAssignments` rules (`tracklet_assignment.hpp`). Keypoint tracklet ids are unique per
detection run and camera, so the table addresses them as it is.

| Operation | Stored as | Tracker effect |
|---|---|---|
| Reject: the tracklet cannot be used | `reject`, whole tracklet or a range | candidate excluded from every slot |
| Belongs to another person | `other` for this subject; optionally `assign` for the other subject | excluded here; assigned there |
| Not this keypoint of this person | `forbid` with the slot | pair excluded |
| Use this tracklet for this slot | `assign` with the slot | pair cost divided by `dot_assigned_factor`, so it passes the assignment gate; other slots excluded |
| Force: use it even against the filter | `force` with the slot (new kind) | as `assign`, and the resulting observation is exempt from the update's outlier rejection, with its noise capped the way trusted keypoint edits are (`edited_kp_noise_std`) |
| Split at frame `f` | `split` with `first_frame = f` (new kind) | none; the editor shows two pieces, and later rules can address either |
| Clear | delete the user's rules of the tracklet | back to automatic |

A range rule is how "split, then decide one piece" is stored: the split row only makes the boundary
visible and selectable. A later rule wins, as today.

### Groups

An operation applied to a group of tracklets writes one row per tracklet, all with the same
`edit_group_id` (new nullable column). The editor shows the group as one item, and undo removes it as
one. The tracker does not read the group id.

## 3. Stages

### First stage: needs no schema change

Works with the candidate storage and the existing `dot_assignment_edits` kinds:

- Tracklet lanes in the timeline (`KeypointTimelineWidget`): per camera, per keypoint, one bar per
  tracklet over its frame range, coloured by whether the last tracking run used it, and by the latest
  rule. Unused high-peak tracklets and used tracklets that the tracker gated out are marked.
- In the crop grid (`PersonCropGridWidget`): all candidates of the current frame as small markers, the
  used one filled, each with a short trail of its tracklet (the existing trail overlay drawn per
  tracklet); clicking a candidate selects its tracklet.
- Reject, other person, forbid, assign on the selected tracklet, for the whole tracklet or the
  timeline's selected frame range.
- Re-run tracking on the trial with the rules (existing `--dot-edits-trial` path).

### Second stage: force, split, groups

- `force` and `split` kinds and `edit_group_id` (one rebuild of `dot_assignment_edits`).
- Group selection by rectangle in the timeline (as the spike's labeller did) and by "same time
  range": tracklets of one camera whose start or end lies within a tolerance of the selected
  tracklet's start or end and whose keypoints are within two joints in the skeleton. This is the case
  of head and shoulder tracklets that switch to another person at the same moment; one operation
  splits all of them at that frame and assigns the pieces.
- Cross-camera grouping: tracklets of other cameras whose rays meet the selected tracklet's over
  their shared frames (`posetrak/markers/grouping.py`, the dot grouping test) are offered as the same
  physical keypoint, so a decision can be applied in all cameras at once. The spike's simulation of
  cross-camera propagation halved the clicks needed to remove half of the wrong frames.
- Suggested groups: segments where the last tracking run and its time-reversed or strongest-peak
  variant differ by more than 40 cm (the spike's failure flag, 13 of 15 failure events at about 19
  flagged windows per minute of person footage) are listed as review windows, with the tracklets
  used inside them grouped as above.

### Later, separate feature

Re-tracking only the edited time range (iterative tracking with state snapshots and a windowed
smoother) would shorten the edit-track loop substantially. It is a large tracker change and is planned
separately; until then each check re-tracks the trial.

## 4. Visualisation

- Candidates: unfilled markers in the keypoint's colour, size by peak; the used candidate filled.
- Tracklets: trails over a short window around the current frame; a selected tracklet's whole trail.
- Rules: a rejected tracklet drawn crossed; an assigned or forced one with a white border (the dot
  review's convention); a split boundary as a tick on the bar.
- The labeller prototype (`python/tools/prototypes/kp_labeling/labeler.py`) is the reference for
  interaction: per-camera timelines, frame-span edits with modifier-key scopes, rectangle selection.
  Its frame cache is superseded by the application's preview caching.

## 5. Acceptance

- First stage: on the labelled trial, the operator removes the wrong picks of one failure window with
  tracklet operations only, and the re-tracked result in that window reaches the label-fed run within
  the spike's figure for corrected windows (3.3 / 1.7 cm in the hard windows).
- Second stage: the head-and-shoulder switch case is corrected by one group operation.
- The number of operations needed to remove half of the wrong frames of a failure window is reported
  against the spike's simulated figures.
