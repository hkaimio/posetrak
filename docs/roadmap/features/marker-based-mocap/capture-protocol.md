# Capture protocol for reflective-marker capture

posetrak is an offline tracker, and captures are often made at a remote location with at
most a laptop or tablet. Nothing in the capture session can ingest videos, detect dots and
track to check the data while the session is running. What a capture must contain is
therefore guaranteed by the setup, by a short routine at the start of every take and by
checks that need nothing but the cameras' own screens. This note says what those are and why.

## What a reference capture showed

A capture with four usable cameras around one person wearing 45 reflective markers (legs, arms,
torso and head) was reviewed tracklet by tracklet to produce a ground truth. Per marker, the
number of cameras that saw it for at least 30% of a 23 s clip:

| cameras | markers |
|---|---|
| 0 | 5 |
| 1 | 22 |
| 2 | 15 |
| 3 | 2 |

A position needs two cameras; checking it needs three. The markers seen by one camera or none
(back, sacrum, right hip, the top of the right shoulder) could not be triangulated, so neither
their placement on the body nor their assignments could be verified, and the automatic
assignment left most of their dots unassigned. The markers on the legs, feet and the side of the
body facing two or more cameras were assigned with 98% precision at the confidence level that
needs no review.

The causes were all in the capture:

- one camera behind the person failed during the recording, which removed the only view of the back;
- one camera applied local tone mapping, which dimmed the small bright dots until the detector
  found almost none;
- the person faced roughly one way for the whole clip, so each marker faced one or two cameras.

Other effects that affect the choice of markers and cameras:

- markers 8-10 cm apart on the same bone region were confused (hip and anterior superior iliac
  spine, index and pinky knuckle, wrist radial and ulnar side, ankle and toe) whenever the
  estimated body pose was off by a few centimetres, which it is during fast movement. Pairs
  nearer than the pose accuracy need more than one camera to be told apart.
- the width of the pelvis measured by triangulating two marker dots was 9 cm larger than on the
  reference body the placements were first made on; markers attached by hand are not where a
  mirrored reference model puts them.

## Requirements

1. **Cameras.** At least six working cameras, spread around the person at roughly equal angles
   and at two heights, so that every marker is within the viewing cone of at least three. A marker
   is visible within about 80 degrees of its outward normal. Four cameras cannot give three views
   of every marker of a 45-marker set. Record with a spare camera running when possible; a camera
   that fails mid-take should cost one view, not a side of the body.
2. **Camera settings.** Fixed exposure with a short shutter, so that the dots are small bright
   spots and the rest of the picture is not blown out. No HDR, local tone mapping, auto
   exposure or beautification mode on any camera. The same frame rate on all cameras. The sync
   event used by the sync step (light flash or clap) at the start of every take.
3. **Camera readiness check.** Before the first take, for each camera, record 10 s of the person
   in the marker suit standing still and turning once, and look at the clip on the camera's own
   screen (or a phone). Pass if, from a metre or two away, each marker facing the camera is a
   clear bright spot; fail if the dots are faint or merged with the background. Repeat after any
   change of exposure settings and after a battery or card change.
4. **Marker set.** Markers that are named differently but sit close (under about 10 cm) should
   differ in arrangement, not only in position, wherever the body allows: different spacing
   between the left and right pair, or one marker of a pair on a different surface. Put two
   markers where one is seen only from the front or the back, such as the shoulder (one in
   front, one behind). Add at least two markers on the back so that they are seen by the cameras
   behind the person. Take photographs of the finished marker placement from the front, back
   and both sides before the first take; they are the reference for the calibration.

## Routine for every take

1. Sync event, visible to all cameras.
2. **T-pose, about 3 s**, facing the centre of the camera ring. This gives the tracker a clean
   starting pose and a clean view of both arms.
3. **Slow turn in place, about 10 s**, one full revolution at walking-pace turning speed,
   arms slightly away from the body. Each marker faces each camera for part of the turn, so the
   offline calibration can see every marker from every side even if the take itself does not.
4. The take.
5. T-pose again, about 3 s, so that drift in the pose can be seen.

The turn and the two T-poses are in every take, not only the first: the calibration of a
marker's place on its bone is better the more of its visible range it has seen.

## After the capture (offline)

The steps are the same for any capture and need only the recorded videos:

1. Ingest the videos, sync them and detect the dots and the person's mask.
2. Run the markerless pose tracker on the turn and T-pose part of each take.
3. Run the automatic dot-to-slot assignment on that part and open the review tool; confirm or
   correct the low-confidence tracklets. A review of a 25 s clip with 750 tracklets is a matter
   of an hour.
4. Freeze the decisions as a ground truth and refit the marker placements from them. A marker
   that the ground truth cannot place (seen by fewer than two cameras) keeps its reference
   placement and is marked as unverified in the attachment set.
5. Report, per marker, the number of cameras that saw it and for how long. A marker seen by
   fewer than two cameras for at least a few seconds of the turn is a finding for the next
   capture's setup, not a tracking problem to tune away.

A capture is good enough to develop and evaluate the assignment and tracking algorithms when,
for the markers that matter to the work, the high-confidence assignments of a held-out take
reach about 95% precision and the unassigned remainder is below about 15% of the marker
frames. In the reference capture the markers with two or more cameras met this; the others did
not, and no tuning of the algorithms recovered them.

## Checklist

Before the session
- [ ] six or more cameras placed around the person, two heights, all framing the whole body
- [ ] exposure fixed and short, HDR / tone mapping / beautification off, same fps everywhere
- [ ] 10 s readiness clip per camera, dots checked on the camera's screen
- [ ] markers attached; photographs from front, back and both sides
- [ ] extra markers and spare tape; spare camera ready

For every take
- [ ] sync event
- [ ] T-pose 3 s
- [ ] slow full turn 10 s
- [ ] take
- [ ] T-pose 3 s

After the session
- [ ] ingest, detect, markerless pose of the turn
- [ ] review and freeze the ground truth, refit the placements
- [ ] per-marker camera coverage reported; markers with fewer than two cameras noted for the next setup
