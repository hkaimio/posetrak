# Track to person assignment without segmentation

## 1. Problem

A YOLOX detection run gives each camera hundreds of short track fragments: the detector's tracker
starts a new track id whenever a person is lost for a frame (`backends_rtmdet.py`), so no track id
covers a person. Today the fragments are assigned to persons by hand in the pose extraction tool's
stitcher (`app/pose/stitcher.py`, `assignment.py`), which writes `detection_track_assignments`;
finalisation builds one sequence per person from those rows. The SOMA prototype avoids the problem
with a one-person mode (the most confident box per frame).

A segmentation run solves identity per pixel and is the best source where persons overlap, but it is
slow and needs interactive seeding. An automatic step is needed for footage without segmentation and
to seed segmentation faster.

The spike also found a related defect in the segmentation route: the per-camera assignment rows had
gaps where the track identity was uncertain, although boxes and keypoints existed, so every tracker
run of the spike lacked all observations of a person for some intervals. An automatic assignment that
covers every frame with a box fixes this for both routes.

## 2. Design: cross-camera association of fragments

A Python step after detection, `posetrak detect assign RUN [--persons N | --subjects ...]`, that
writes ordinary `detection_track_assignments` rows. Everything downstream (finalisation, the stitcher
as a review and correction tool) stays as it is.

1. **3D person hypotheses per instant.** For each global time (the tracker rate or the slowest camera
   rate), triangulate the fragments' trunk keypoints (hips, shoulders; for SOMA also spine and neck)
   pairwise across cameras with the calibrated cameras; a pair is consistent when the reprojection
   error in both is below a fraction of the box height. Cluster consistent fragment observations into
   persons per instant (greedy by agreement count, one fragment per camera per person).
2. **Link over time.** Constant-velocity linking of the 3D person hypotheses (Hungarian per instant,
   gate in metres), giving person tracks in 3D.
3. **Label fragments.** Each fragment gets the person track it was assigned to in most of its frames;
   a fragment that switches persons inside (a detector identity switch) is split at the switch frame,
   which the assignment table already supports (`first_frame`, `last_frame` per row).
4. **Name persons.** With `--persons N` the N longest person tracks are kept and named in order of
   first appearance; with the capture's subjects (`capture_subjects`) and one seed click per person in
   the stitcher, the tracks get their subject ids. Single-person footage needs no seed.
5. **Report.** Frames where a fragment could not be assigned or two persons' 3D tracks come within a
   body width are listed: these are where segmentation or a manual check pays.

The step uses only boxes and body keypoints of the run, so it works for RTMPose and SOMA runs alike.
It reuses the triangulation helpers of the extrinsics solver and the cross-camera grouping test of
`posetrak/markers/grouping.py` (rays of two cameras meeting over a shared lifetime).

## 3. Relation to segmentation

- Where persons are apart the automatic assignment is enough; where they overlap or touch,
  segmentation is the recommended source of boxes and identity, and a mask-sourced detection run
  needs no assignment step.
- The automatic step's output seeds segmentation: its per-person boxes at chosen frames can initialise
  the segmentation model's object prompts instead of clicks, and its "close contact" report proposes
  the segmentation split points (`capture_segmentation_hints`). Building that seeding is part of the
  segmentation UI feature.
- With keypoint candidates the person a candidate belongs to matters less: candidates of overlapping
  crops are merged and the tracker decides ownership (keypoint-candidates design, section 3). The
  assignment step still decides which crops form a person's sequence.

## 4. Acceptance

- On the labelled five-camera trial, YOLOX fragments assigned automatically agree with the
  segmentation-derived identity on at least 98% of box-frames where persons are more than a body
  width apart, and the report flags at least 90% of the frames where they disagree.
- Single-person footage needs no manual step from detection to tracking.
- Sequences built from the automatic assignment have no interval without observations where a box of
  the person exists in two or more cameras.
