# Track to person assignment without segmentation

## 1. Problem

A YOLOX detection run gives each camera hundreds of track fragments: the detector's tracker starts a
new track id whenever a person is lost for a frame (`backends_rtmdet.py`), so no track id covers a
person. Worse, the tracker follows boxes by overlap, so where two persons overlap it swaps their ids
and one fragment carries one person before and another after. On the labelled five-camera trial only
82% of the boxes of a fragment belong to the person most of its boxes belong to.

Today the fragments are assigned to persons by hand in the pose extraction tool's stitcher
(`app/pose/stitcher.py`, `assignment.py`), which writes `detection_track_assignments`; finalisation
builds one sequence per person from those rows. The SOMA prototype avoids the problem with a
one-person mode (the most confident box per frame).

A segmentation run solves identity per pixel and is the best source where persons overlap, but it is
slow and needs interactive seeding. An automatic step is needed for footage without segmentation and
to seed segmentation faster.

The spike also found a related defect in the segmentation route: the per-camera assignment rows had
gaps where the track identity was uncertain, although boxes and keypoints existed, so every tracker
run of the spike lacked all observations of a person for some intervals. An automatic assignment that
covers every frame with a box fixes this for both routes.

## 2. Design: cross-camera association of fragments

`posetrak detect assign RUN [--persons N] [--subjects NAME,...] [--fill-only | --replace] [--finalise]`
(`posetrak/detection/assign_run.py`, algorithm in `track_assignment.py`) reads the boxes and body
keypoints of a run, writes ordinary `detection_track_assignments` rows and prints a report.
Everything downstream (finalisation, the stitcher as a review and correction tool) stays as it is.
The trunk keypoints come from the pose model registry (`PoseModelSpec.trunk`: shoulders and hips for
COCO-133; hips, spine, neck and shoulder joints for SOMA-77), so RTMPose and SOMA runs work alike.

1. **Pieces.** Each camera's fragments are cut where the detector swapped identities. A person's
   skeleton moves a few hundredths of its height between frames, so boxes of consecutive frames are
   matched by the median distance of their body keypoints (Hungarian, gate `piece_gate` in box
   heights); a box that matches nothing starts a new piece, and a box with too few confident
   keypoints continues its detector track. On the trial this raises the share of boxes in the
   dominant person of their fragment from 82% to 96%. Everything after this step uses pieces in place
   of detector track ids; rows are written back as ranges of detector ids.
2. **3D person hypotheses per instant.** On a grid at the slowest camera's rate, the trunk
   keypoints of every box are triangulated across cameras; two boxes of different cameras are
   consistent when the triangulated trunk is person-sized and reprojects within `max_rel_err` of the
   box height in both. Consistent boxes are clustered greedily by agreement, one box per camera, and a
   cluster is cut down until every box fits the joint estimate. Epipolar consistency alone is weak:
   two persons of similar height pass it for 15-20% of box pairs, even when they are apart, so no
   pair is trusted without the continuity of step 3.
3. **Tracks.** Hypotheses are linked over time (constant velocity, Hungarian, a gate in metres) only
   where a hypothesis is made largely of the pieces the track was made of, and no other person is
   within `link_clear_m`. A track therefore ends at every contact between persons, and tracks are
   nearly pure: on the trial 97% of the boxes in tracks belong to the person most of their track's
   boxes belong to.
4. **Labelling the tracks.** A person is a set of tracks. Tracks that do not overlap in time and are
   at most `merge_gap_s` apart are tied by the pieces that have boxes at the end of one and the start
   of the other (the detector kept the person through the contact) and, weaker, by the distance
   between the ends; tracks that overlap in time are one person if in the same place and two if not.
   The labelling with the most tie weight inside persons and no cannot-link inside is found by
   iterated local search from random starts; a track may also take no person, so a bystander does not
   join the subject. The number of persons is `--persons`, else the number of `--subjects`, else the
   number seen together for a fifth of the time that anyone is seen.
5. **Boxes.** Every box of every camera is matched to the persons by projecting their 3D positions
   (interpolated across gaps up to `interp_gap_s`) onto it; the persons are triangulated again from
   the boxes they received, three rounds. A box that two persons project near equally is left out. A
   piece takes the person of its voted boxes, unvoted boxes take the person of the nearest voted box
   of the piece, and a piece whose boxes switch person is split in the middle of the unvoted
   stretch between the switching boxes. Each box now has a person or none; rows are runs of one
   person on one detector track. Boxes with no person (detections of bystanders, ghosts) get no
   row and are listed.
6. **Names.** Persons are named after the persons that existing rows name (seeds, section 3), then
   after `--subjects` in order of first appearance, else `Person 1`, `Person 2`. A run with one
   person in a capture that has exactly one capture person takes that person's name, so
   single-person footage needs no manual step. Rows carry `capture_person_id` and `subject_id` of the
   named capture person, created if missing, as finalisation does. `--finalise` then builds the
   sequences.

The step uses only boxes and body keypoints of the run and the capture's calibration.

## 3. Existing assignments and seeds

A run may already have rows, from the stitcher or from an earlier `detect assign`. Following the
rule that an automated write does not silently override a human edit:

- By default `detect assign` does not touch a run that has any row and says how to proceed.
- `--fill-only` keeps every row and assigns only the boxes no row covers (and not to a person that
  already has a row on the same camera and frames).
- `--replace` deletes all rows after reading them as seeds.

In both modes the rows are seeds: a track that has boxes inside the frames of a row takes the row's
person, and a person named by a row is the same person throughout the run. One hand-made row per
person therefore names the persons (and the rows after a contact pin the identity where the cues
conflict, section 5). A row that conflicts with the geometry is not changed in `--fill-only`.

## 4. Output

- Rows in `detection_track_assignments`; no schema change.
- A text report: persons found, per camera the share of boxes that agree with another camera and the
  median trunk reprojection error as a fraction of the box height (a camera far above the others
  has a calibration or synchronisation problem, which is how calibration quality limits the step
  visibly), and the stretches to check: `contact` (two persons closer than `contact_m`), `handover`
  (a person carried from one track to the next across a gap by weak cues, unless seeds pin both),
  `split` (a piece changes person), `unevidenced` (a long stretch of a piece with no box of its own
  in any hypothesis) and `unassigned` (boxes with no person).
- The same report as JSON next to the session database (`<db stem>.assign-<run id prefix>.json`),
  which the stitcher lists as a "To check" selector that moves its time cursor.

## 5. Relation to segmentation

- Where persons are apart the automatic assignment is enough; where they overlap or touch,
  segmentation is the recommended source of boxes and identity, and a mask-sourced detection run
  needs no assignment step.
- The automatic step's output seeds segmentation: its per-person boxes at chosen frames can initialise
  the segmentation model's object prompts instead of clicks, and its "contact" report proposes
  the segmentation split points (`capture_segmentation_hints`). Building that seeding is part of the
  segmentation UI feature.
- With keypoint candidates the person a candidate belongs to matters less: candidates of overlapping
  crops are merged and the tracker decides ownership (keypoint-candidates design, section 3). The
  assignment step still decides which crops form a person's sequence.

## 6. Acceptance and measurements

The step is for footage where persons are mostly apart: one person, or several persons who move
among each other or among bystanders without holding on to each other. Contacts are listed for the
stitcher or for segmentation, not resolved.

- Single-person footage needs no manual step from detection to tracking.
- With several persons apart, the assignment gives each person long stretches that are nearly pure,
  and the report lists the places where a person's label changes, so that the manual work in the
  stitcher is joining stretches at the listed places rather than assigning fragments.
- Sequences built from the automatic assignment have no interval without observations where a box of
  the person exists in two or more cameras, outside listed stretches.

The original criterion, 98% agreement where persons are a body width apart on the labelled
five-camera trial of section "Measurements", is not met on that trial. The trial is mostly contact
(persons in a throwing technique), which is the case the step does not claim to solve; its figures
are kept as the record of that case.

`python/tools/eval_track_assignment.py` scores a run against a segmentation-sourced reference run
(overlap matching, apart = boxes disjoint horizontally, person names paired by the best
assignment, which one seed click per person gives too).

### Measurements

Trial: five cameras, three persons, 38.08-66.44 s (spike report, section 3). The assigned run is the
only run of the whole trial that has detector-track fragments: 83 fragments from a YOLO11x person
detector and RTMPose-133 (no YOLOX run of the trial exists; its tracker produces the fragments and
identity swaps described in section 1). The reference is the segmentation-sourced run of the trial.
One camera (`insta_ace2_pro`) is left out of the scoring: the spike report names its segmentation and
calibration as unreliable, and its reference labels disagree with the hand-made assignment far more
than those of the other cameras do (82% against 93-100% where persons are apart). 22302 box-frames
of the four other cameras are matched to the reference.

| Agreement with the reference | box-frames | persons apart | persons not apart |
|---|---|---|---|
| `detect assign`, no seeds | 83.3% | 88.7% | 77.4% |
| `detect assign --fill-only`, ten hand-made rows of one camera as seeds (scored on the other three cameras) | 88.2% | 95.8% | 80.0% |
| assignment made by hand in the stitcher | 97.0% | 98.2% | 95.7% |

Without seeds, persons apart, per person: 98.9%, 66.9%, 98.8%; per camera (labels as in the
report): pixel7 91.2%, gopro-11_mini_02 71.6%, pixel9 92.2%, gopro-11_mini_01 99.3%. The report of the
same run shows why the cameras differ: pixel9 and gopro-11_mini_02 agree with another camera for 67%
and 74% of their boxes, the other cameras for 89-95%, and gopro-11_mini_02 is the camera with the
lowest agreement with the reference. The agreement criterion (98%) is therefore not met automatically;
with one camera's rows as seeds the other cameras reach 95.8%, and the hand-made assignment itself
reaches 98.2%.

The listed stretches (contacts, hand-overs, splits, unassigned) contain 69% of the disagreeing
box-frames (61% of those where persons are apart) without seeds, and 99.9% with seeds: the 90%
criterion is met only with seeds. Without seeds the listed stretches cover 70% of all box-frames
of this trial, which is mostly contact, so the list says little about where the errors are.

Where the automatic run goes wrong is one decision: the persons of two tracks 9 cm apart at a
contact are exchanged for ten seconds, in which one person is labelled as the other and the other
person has no sequence (the stretches the reference sees in two or more cameras without a box of
the person's sequence: one 10 s stretch of one person, and a few sub-second ones of the others). With
the right person for every track (what a seed on every track would give) the same boxes reach 98.1%
where persons are apart (99.8% for the best and 95.4% for the worst person) and 93% of the
disagreements are listed, so the box-level steps meet the criterion and the identity of the tracks across
contacts is what is missing. Single decisions of this kind decide the figure: ten variations of the thresholds move the automatic agreement of
persons apart between 54% and 91%.

Single-person footage needs no manual step: on a six-camera run of one person, `detect assign RUN
--finalise` names the person after the capture's only person, writes 15 rows and builds a sequence
with 17065 observations, against 17062 of the sequence made from the operator's rows; a three-camera
YOLOX run gives 4257 against 4256. The report lists one hand-over and the boxes of bystanders as
unassigned, and marks the camera that agrees with the others for 35% of its boxes.

Several persons among bystanders: six cameras at 120 fps, 97 s, five
persons: two who move about the room and three bystanders who mostly stay in place, one of them seen
by three cameras only. YOLOX with SOMA-77 (116 min of detection), scored against the
segmentation-sourced run of the capture, which labels the two moving persons; 73099 box-frames are
matched to it.

| `detect assign` | persons apart | the subject (in view throughout) | the second moving person |
|---|---|---|---|
| `--persons 2` | 45.6% | 42.0% | 91.7% |
| `--persons 4` | 90.3% | 90.2%; 99.8% after the first 16 s | 91.7%; 100% after the first 16 s |
| `--persons 5` (the true count) | 85.1% | 90.2%; 99.8% after the first 16 s | 19.9% (three labels, see below) |

With fewer persons than are present, the bystanders take the labels of the moving persons, and the
automatic count (persons seen together for a fifth of the time) found three: with bystanders the
count has to be given. With the true count, the labels the run gives are pure stretches: the second
moving person is one label to 42.5 s, a second to 101.3 s and a third after that, and the subject
one label to 42.5 s and another after, the two moving persons exchanging labels at 42.5 s, where they
pass close to each other. The scoring pairs each reference person with one label, so the three
stretches score as 19.9% although nearly every box in each of them belongs to the same person. Both
places are in the report (`handover` of the two labels at 42.46 s, `split` between the two labels
at 101.28 s), as are 99% of the disagreeing boxes. The other errors are in the first 16 s, where the
fifth person and a bystander take boxes of the moving persons, and in an exchange of the two
bystanders' labels midway. Fixing the run in the stitcher is joining three or four stretches at
listed places.

`detect assign` takes 26 s on the 28 s five-camera trial and 4 s on 10 s of one person.

## 7. Limits

- Identity across a contact rests on the detector's fragments and on kinematics, both of which fail
  where persons are tangled for seconds: the fragment evidence for the right and the wrong pairing
  can be equally strong, and the labelling that keeps the most evidence is then not the true one.
  A wrong decision at one contact relabels everything after it, so the figures depend on a few
  decisions. This is what `contact` and `handover` stretches are listed for, and what seeds and
  segmentation are for.
- The number of persons found automatically counts the persons seen together; bystanders seen by
  few cameras or only part of the time are missed, and with too few persons a bystander takes a
  moving person's label. With bystanders `--persons` has to be given.
- A camera with a poor calibration or a poor synchronisation agrees with the others less; it shows
  in the report's per-camera figures and drags the cameras it pairs with.
- Body proportions measured in 3D (shoulder width, torso, limb lengths) did not tell two of the
  three persons of the trial apart (their medians differed by less than their spread over a
  tracklet), so they are not used.
