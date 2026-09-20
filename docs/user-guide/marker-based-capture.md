# Marker-based capture from the command line

Posetrak can track more than a person's body. With **markers** it also tracks rigid
props such as a sword, a pen or a ball, and it can follow markers worn on a person.
This page walks through the command-line workflow for each case that is supported today.

Two kinds of marker are used:

- **ArUco markers** are printed square tags. Each has a number, so a detection says
  which marker it is, and its four corners give the marker's full position and
  orientation.
- **Reflective dots** are small retroreflective spots, usually lit by a ring light on
  the camera. A detected dot has no identity: the tracker works out which dot is which
  from where each should be.

!!! note "Command syntax"
    Every command is `posetrak [--session PATH] <group> <command> ...`. Give the
    session database once with `-s`/`--session`, or set `POSETRAK_SESSION_DB`. Long
    commands below are split with `\`; in PowerShell write a backtick instead, or put
    the command on one line. IDs are printed in full and may be shortened to any unique
    prefix. Options that repeat (`--dots-camera`, `--camera`, `--object`) are given once
    per value.

## What you need first

The session must already hold a **capture** with its videos, the cameras' **intrinsics**
and **extrinsics**, and a **sync** configuration. Those steps are the same as for any
capture and are described in [Your first capture](first-capture.md),
[Camera intrinsics](camera-intrinsics.md),
[Extrinsics calibration](extrinsics-calibration.md) and
[Synchronizing videos](synchronizing-videos.md). Marker tracking uses the same cameras
and calibration as body tracking.

You also need the tracker program `posetrak-tracker` (see [Setup](../setup.md)). Pass
`--binary PATH` to the tracking commands if it is not where Posetrak looks for it.

## Which workflow do you need?

| You want to track | Follow | Status |
|---|---|---|
| A rigid prop with one or more ArUco markers, with or without reflective dots | [A](#a-a-rigid-prop-with-aruco-markers) | Complete on the command line |
| A prop that has one reflective dot and no ArUco marker (a ball) | [B](#b-a-prop-with-only-a-reflective-dot) | Complete, needs a starting position |
| A prop with several reflective dots and no ArUco marker | not supported | [Add one ArUco marker](#limits) |
| Points you tracked by hand in another program (Blender) | [B](#b-a-prop-with-only-a-reflective-dot) | Complete |
| A person wearing reflective dots | [C](#c-a-person-wearing-reflective-dots) | Tracking is on the command line; preparing the dot set is still done with scripts |
| A person and a prop together | [D](#d-tracking-several-subjects-together) | Complete |

## The pieces

- A **marker body** is the geometry of a prop: where its ArUco markers and dots are, in
  one marker's own coordinate frame. You write it by hand, or solve it from video
  ([Solving a marker body from video](#solving-a-marker-body-from-video)).
- A **capture object** is one physical prop in one capture. It names a marker body.
  A name is unique within a capture.
- A **detection run** is one pass of a detector over the videos: coded markers
  (`aruco`), reflective dots (`dots`) or imported tracks (`external_2d`).
- An **observation sequence** is what the tracker reads for one subject. Making one from
  a detection run is called *finalising* it.
- A **tracking skeleton** is what the tracker moves. A prop's skeleton is one free-floating
  body carrying its markers; it is generated from the marker body.
- A **tracking run** is one execution of the tracker.

## A. A rigid prop with ArUco markers

Print the markers, glue them on the prop, and measure or solve where they sit. The
markers must be large enough to be recognised in the cameras that should see them. A prop
whose markers face different directions is tracked through all of them, so put markers on
every side that the cameras will see.

### A1. Describe the prop

You need the marker body of the prop as a YAML file. There are three ways.

**Write it by hand.** Choose one marker's frame as the prop's frame and give every marker
a `center`, the direction its face points (`normal`), a direction along its top edge
(`up`) and its `size` (side length of the printed square, metres). Reflective dots need
only a `center`.

```yaml
name: bokken
units: meters
markers:
  - name: hilt
    type: aruco
    dictionary: DICT_4X4_50
    id: "3"
    size: 0.05
    center: [0.0, 0.0, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: tip
    type: aruco
    dictionary: DICT_4X4_50
    id: "7"
    size: 0.03
    center: [0.0, 0.9, 0.0]
    normal: [0.0, 0.0, 1.0]
    up: [0.0, 1.0, 0.0]
  - name: dot1
    type: reflective_dot
    center: [0.05, 0.4, 0.01]
```

**Solve it from several fixed cameras** when the prop is in your normal capture:

```bash
posetrak -s session.db marker-body calibrate --capture <capture> \
    --time-start 34.4 --time-end 100.6 --marker-size 0.05 \
    --marker-ids 3,7 --reference-id 3 --detect-dots --output bokken.yaml
```

**Solve it from one moving camera** when you can film the prop separately. This is often
easier and more accurate than the multi-camera method. See
[Solving a marker body from video](#solving-a-marker-body-from-video) for both, and for
how to film.

### A2. Register the prop

Import the body, tell the capture about the prop, and generate the prop's tracking
skeleton:

```bash
posetrak -s session.db marker-body import --file bokken.yaml --name bokken
# marker_body_id: <body>

posetrak -s session.db capture object add --capture <capture> --name bokken --marker-body <body>
# capture_object_id: <object>

posetrak -s session.db marker-body to-skeleton <body> --name bokken
# skeleton_id: <skeleton>
```

`marker-body import` and `capture object add` accept ID prefixes. `capture object list
--capture <capture>` shows the objects. The skeleton generation is safe to repeat: the
same body and name give the same skeleton back.

`--marker-body` on `capture object add` also finds a body that is in the registry but not
yet in the session, and copies it in.

### A3. Create the trial

Tracking works on a **trial**: a named time range of a capture. Create it before you detect,
because a prop's observation sequence belongs to a trial through its detection run, and
`track run-persons` only finds sequences on the trial it is given.

```bash
posetrak -s session.db trial create --capture <capture> --name "Swing 1" --start 34 --end 100
# trial_id: <trial>
```

### A4. Detect the markers

```bash
posetrak -s session.db detect run --type aruco --object bokken --trial <trial> \
    --capture <capture> --sync <sync> --start 34 --end 100
# prints the detection run id
```

The run reads every camera's video between the two times. The ArUco dictionary and the
marker ids come from the object's marker body. Progress goes to the terminal; the run id
is printed to stdout when it ends.

To also detect the prop's reflective dots in the same pass, name the cameras that should
look for them:

```bash
posetrak -s session.db detect run --type aruco --object bokken --trial <trial> \
    --capture <capture> --sync <sync> --start 34 --end 100 \
    --dots-camera gopro13_01 --dots-camera gopro13_02
```

Name only cameras that really see the dots: those with a ring light next to the lens, or
that look at strongly reflective markers. Other settings are described in
[Detecting dots](#detecting-dots). Add `--parallel` to process the cameras in parallel.

A detection run is never changed afterwards. To try different settings, run detection
again.

### A5. Make the observation sequence

```bash
posetrak -s session.db sequence finalise-object --detection-run <run>
# sequence_id: <sequence>
```

This copies the coded-marker corners and any dot candidates of the run into one sequence
for the object. Finalising the same run again replaces its sequence, unless that sequence
has already been tracked or edited.

### A6. Track

```bash
posetrak -s session.db track run-persons --trial <trial> --object bokken=<skeleton>
# prints the tracking run
```

The tracker looks near the start of the range (within about two seconds) for a frame in
which enough markers are seen by enough cameras to fix the prop's pose, starts there,
follows the prop, and smooths the result afterwards (`--no-smooth` turns that off). Results
are stored in the session and also written as CSV files, one folder per subject, under
`<session folder>/posetrak_results/<capture>/<trial>/tracking`. When it ends it reports how
many steps were tracked and how many were lost.

By default the tracked range is the range of the sequence. `--start-time` and `--end-time`
narrow it. `--config` selects a tracker configuration other than the trial's default.

You can also give a sequence directly, which is all that is needed for one subject:

```bash
posetrak -s session.db track run --sequence <sequence> --skeleton <skeleton>
```

### A7. Look at the result

```bash
posetrak -s session.db track list
posetrak -s session.db track show <run>
posetrak -s session.db track export gltf <run> prop.glb --smoothed
```

`track export` also writes `bvh` and `usd`. For a run with several subjects, `--person-id N`
selects the subject by its position in the run.

### Several props, one detection pass

Reading the videos is the slow part, so detect all props together and finalise each one
from the shared run. Detect once for every marker id, bound to no object:

```bash
posetrak -s session.db detect run --type aruco --marker-ids 2,3,16,17 --trial <trial> \
    --capture <capture> --sync <sync> --start 34 --end 100
# prints the shared run
```

Then make each prop's sequence from it:

```bash
posetrak -s session.db sequence finalise-object --detection-run <shared> --object pen
posetrak -s session.db sequence finalise-object --detection-run <shared> --object pad
```

Each prop gets a detection run of its own, derived from the shared one (and so on the same
trial), holding only its markers in the order its marker body gives. The shared run is not changed. Add
`--with-dots` if the prop's body has reflective dots and the shared run detected them.
If a marker id is meant for one prop only, the other props' bodies must not list it:
markers that are not in a prop's body never reach its sequence.

## B. A prop with only a reflective dot

A ball with one reflective dot has no marker that says where it is and which way it faces,
so the tracker cannot find it by itself: **you give the starting position**. A prop with
several dots and no ArUco marker is not supported, because the dots alone do not give the
prop's orientation. Put one ArUco marker on it and follow [A](#a-a-rigid-prop-with-aruco-markers).

### B1. Register the ball

The marker body has one dot:

```yaml
name: ball
units: meters
markers:
  - name: dot0
    type: reflective_dot
    center: [0.0, 0.0, 0.0]
```

```bash
posetrak -s session.db marker-body import --file ball.yaml --name ball
posetrak -s session.db capture object add --capture <capture> --name ball --marker-body <body>
posetrak -s session.db marker-body to-skeleton <body> --name ball
```

### B2. Get the dot observations

Either detect the dots automatically, bound to the ball:

```bash
posetrak -s session.db detect run --type dots --object ball --trial <trial> \
    --capture <capture> --sync <sync> --start 56 --end 70 \
    --dots-camera gopro13_01 --dots-camera gopro13_02 --dot-background-mode blacklist
```

or import points that you tracked by hand in another program. Blender's Movie Clip
Editor works well when the automatic detector struggles. Each file is one track of one
camera and needs the columns `video_frame`, `pixel_x` and `pixel_y` (raw image pixels,
origin at the top left). `python/tools/blender/blender_export_2d_tracks.py` writes them in
this form.

```bash
posetrak -s session.db detect import-2d --capture <capture> --sync <sync> \
    --object ball --trial <trial> --source blender \
    --camera gopro13_01 ball-gopro13_01.csv \
    --camera gopro13_01 ball-gopro13_01-b.csv \
    --camera gopro-11_mini_01 ball-gopro11.csv
```

A camera may have several files, for instance when you restarted the track after the ball
was hidden. Each file is kept as a separate track. A point outside the video's own frame
numbers is rejected: it means the export's frame numbering does not match the video.

### B3. Combine sources

A sequence holds one dot row per camera and frame, so different runs can share a
sequence only on different cameras. For example: hand-tracked points on three cameras and
the automatic detector on a fourth:

```bash
posetrak -s session.db sequence finalise-object --detection-run <imported-run>
posetrak -s session.db sequence add-dots --detection-run <automatic-run> \
    --sequence <sequence> --camera gopro13_02
```

`add-dots` refuses a camera the sequence already has dots for. `--replace` swaps exactly
those cameras for the new run's, and is refused once the sequence has been tracked.

### B4. Track with a starting position

Create the trial before B2 (see [A3](#a3-create-the-trial)), give the same `--trial` to the
detection or the import, and finalise the run with `sequence finalise-object`. Then:

```bash
posetrak -s session.db track run-persons --trial <trial> \
    --object ball=<skeleton>@0.0,-0.97,1.1
```

The position `@X,Y,Z` is the ball's position in the world frame of the calibration, in
metres, at the start of the tracked range. The tracker starts from it, so it should be where
the ball really is. Every object that is only dots needs its own position. Without it the
command says so and stops.

## C. A person wearing reflective dots

Dots worn on a person are tracked together with the body. The tracker predicts where each
dot should appear from the body's pose and matches detections to those predictions.

Three things are needed, and the first two are the same as in a body-only capture:

1. **The person's observation sequence from body detection**, made and stitched as usual
   (see [Tutorial: tracking a capture](tutorial1.md)).
2. **A skeleton that has the dots as markers.** This is the person's skeleton plus a set
   of dot slots with fitted positions on the bones.
3. **The dot detections in the person's sequence.**

Step 3 is on the command line. Detect the dots over the whole span, then add them to the
person's sequence:

```bash
posetrak -s session.db detect run --type dots --capture <capture> --sync <sync> \
    --start 30 --end 130 --dots-camera gopro13_01 --dots-camera gopro13_02 \
    --dot-background-mode blacklist --dot-threshold-by-camera gopro13_02=200
# prints the run

posetrak -s session.db sequence add-dots --detection-run <run> --sequence <person-sequence>
```

The dots run must use the same sync configuration as the person's sequence, or the frames
would map to the wrong times; the command refuses a mismatch.

Then track with the dot skeleton:

```bash
posetrak -s session.db track run --sequence <person-sequence> --skeleton <dot-skeleton>
```

`--dot-background-mode blacklist` is the mode to start with for markers worn on a moving
person; the default `subtract` can merge a dot into the limb it sits on. Cameras differ in
how bright a real dot appears, so give the worse cameras their own threshold.

!!! warning "Step 2 is not a command-line workflow yet"
    Preparing the dot skeleton means labelling which detected dots are which slot, fitting
    the dots' positions on the bones, and merging them into the skeleton. Today this is done
    with the scripts in `python/tools/` (`build_tracklet_groups.py`,
    `label_tracklet_groups_gui.py`, `fit_calibrated_attachment_set.py`,
    `build_dot_augmented_skeleton.py`). The slot names and joints of a leg set are in
    `catalog/modules/leg.marker-module.yaml`, and the tools refuse a skeleton the set does
    not fit. A `marker-set` command group is planned for this.

## D. Tracking several subjects together

Persons and props of a trial are tracked in one run, so that the dots they have in common
are shared correctly: every detected dot goes to at most one subject.

```bash
posetrak -s session.db track run-persons --trial <trial> \
    --persons Alice \
    --object sword=<sword-skeleton> \
    --object ball=<ball-skeleton>@0.0,-0.97,1.1
```

- `--persons` lists persons of the capture by name. Each person's default skeleton and
  observation sequence are used. Use `track run` if a person needs another skeleton.
- Each `--object` is a capture object's name with its skeleton, and a starting position if
  it is only dots. Repeat `--object` for several objects. Every subject's detections must be
  recorded on the trial: for objects, through `--trial` on `detect run` or `detect import-2d`.
- All subjects are stepped together over one time range. The default is where every
  subject's sequence overlaps; `--start-time` and `--end-time` set it explicitly.
- `--config` gives one tracker configuration to every subject, which is what subjects that
  share dot detections need: they must agree on the dot settings.

## Solving a marker body from video

The marker body's geometry follows from footage. Both methods write the same file, in the
frame of the reference marker (a marker you choose), and `--output` writes it while
`--import` also adds it to the session.

### From several fixed cameras: `marker-body calibrate`

Use footage from your normal, already calibrated cameras. The prop moves in front of them.
Every marker of the body must be seen in the same instant as the reference marker
somewhere in the footage, by two or more cameras, though not necessarily by the same
cameras.

```bash
posetrak -s session.db marker-body calibrate --capture <capture> \
    --time-start 98 --time-end 113 --marker-size 0.095 \
    --marker-ids 2,3 --reference-id 2 --output pen.yaml
```

`--detect-dots` also solves the reflective dots. `--camera LABEL` (repeated) restricts the
cameras, which you need when a marker id also occurs on something else in some camera's
view. A marker that is never seen together with the reference is left out, and the command
says so.

### From one moving camera: `marker-body calibrate-video`

Film the prop with one camera that you move around it, while the prop and everything
around it stay still. This method often needs less equipment and gives a better result.

Give it the video, the camera, and a capture that used the same camera in the same mode,
which supplies the camera's intrinsics:

```bash
posetrak -s session.db marker-body calibrate-video --video orbit.mp4 \
    --camera insta_ace2_pro --intrinsics-capture <capture> \
    --body-markers 2,3:0.06 --body-marker-size 0.095 --reference-id 2 \
    --anchor-markers 0,1 --anchor-marker-size 0.19 \
    --detect-dots --output prop.yaml
```

- `--body-markers` are the markers on the prop, each with its side length in metres
  (`ID:SIZE`, or a default from `--body-marker-size`). Sizes may differ.
- `--anchor-markers` are further markers **around** the prop, not part of it. They only
  track the camera while the prop's own markers face away, so they can be larger and
  need no measured position. They must stand still.
- `--reference-id` is the prop marker whose frame becomes the prop's frame.
- `--dictionary` (default `DICT_4X4_50`) applies to all markers. Sizes fix the scale of
  the result, so measure the printed sizes.

Two kinds of target work:

1. **A prop with one or a few markers, plus anchors around it.** Put the anchors so that
   every side of the prop is filmed together with some anchor, and that the anchors
   themselves are seen together in turn. A chain of frames that see two markers each
   connects them all to the reference.
2. **A prop with markers on every side** (a box). No anchors are needed, because the
   prop's own markers keep the camera tracked.

To film: move slowly and steadily all the way around, keep the markers sharp, and cover every
side of the prop, at more than one height if you can. A frame counts once it shows two known
markers. `--stride` (default 6) samples every sixth frame; `--first-frame` and
`--last-frame` narrow the video.

The command reports how many frames it used, the reprojection error per marker (a healthy
result is around one pixel), and any prop marker that no frames connect to the reference.
Solving the dots also lists how many views back each one. Dots are only kept if they lie
within `--dot-max-distance-m` of the reference marker (default 1 m; raise it for a long
prop), if they are not on a marker's printed area, and if their views agree on a single
point.

!!! tip "The prop's origin and axes"
    The result is in the frame of the reference marker. To put the origin and axes elsewhere
    (the handle, say, with an axis along the blade), apply one rigid transform to every marker
    corner and dot in the YAML before importing it. There is no command for this yet.

## Detecting dots

Dots are found as small bright blobs. These options belong to `detect run --type aruco` and
`--type dots`, and are recorded with the run:

| Option | Meaning |
|---|---|
| `--dots-camera LABEL` | Cameras that look for dots. Repeat for several. Required for `--type dots`. |
| `--dot-threshold N` | Brightness a dot must reach (default 235). |
| `--dot-threshold-by-camera LABEL=N` | The same, per camera, for cameras that render dots dimmer. |
| `--dot-background-mode subtract\|blacklist` | `subtract` thresholds the difference from an empty-scene background. `blacklist` thresholds the raw brightness and uses the background only to reject glare. Use `blacklist` for worn markers. |
| `--dot-blacklist-frac X`, `--dot-blacklist-frac-by-camera LABEL=X` | Looseness of the glare rejection. Higher rejects fewer real dots. |
| `--dot-max-saturation X`, `--dot-max-saturation-by-camera LABEL=X` | Reject colourful candidates (skin, fabric). 255 turns it off. |
| `--dot-bg-sample-count N` | Frames sampled to build the background. |
| `--frame-step N` | Process every Nth frame. |

Options that belong to another kind of run are rejected rather than ignored. Cameras
render the same marker differently, so it pays to look at a few frames and set per-camera
values rather than one for all.

## When something goes wrong

| Message or symptom | What it means | What to do |
|---|---|---|
| `--type aruco needs --object, or --marker-ids ...` | An unbound marker run has to say which ids to look for. | Add `--object NAME` or `--marker-ids 2,3`. |
| `no video in this capture and time range for ...` | A `--dots-camera` has no video in the range. | Check the label with `capture show`, and the times. |
| `... has none of the marker ids of object ...` | The shared run was not told to look for this prop's markers. | Detect again with `--marker-ids` that include them. |
| `... already has run ... derived from this one` | The prop was already finalised from this shared run. | Use the run the message names. |
| `sequence ... already has N dot rows for these cameras` | The sequence already has dots for that camera. | Choose other cameras with `--camera`, or `--replace`. |
| `No finalised sequence for object '...' in trial '...'` | The object's detection run was made without this trial. | Detect again with `--trial <trial>`, then `sequence finalise-object`. |
| `Subject '...' has only anonymous-dot markers ... seed position` | A dots-only prop needs a starting position. | Add `@X,Y,Z` to the `--object`. |
| `... has N anonymous-dot markers and no coded marker` | A prop with several dots and no ArUco marker cannot be started. | Add an ArUco marker. |
| `marker '3': never seen together with the reference, left out` (`calibrate`), `body marker '3': not linked to the reference by any shared frame, left out` (`calibrate-video`) | The footage never puts that marker in the same instant or frame as the reference, directly or through anchors. | Film so that it is seen with the reference, or with markers that are. |
| `the reference marker ... was never seen by 2 or more cameras at once` (`calibrate`), `... never seen in a frame together with another known marker` (`calibrate-video`) | The reference marker's pose could not be solved. | Check the marker id, the dictionary and the sizes, and that the markers are visible. |
| Reprojection error of several pixels | Some marker sizes are wrong, a marker is not flat or not still, or the intrinsics do not match the video. | Re-measure the printed sizes; check the mode of the camera. |
| The tracker loses the prop | Not enough markers in view, or a marker id is repeated on another object. | Look at the detections; use more markers; give the ids per object. |

## Limits

- A prop of only reflective dots is supported when it has a single dot. With several dots it
  needs at least one ArUco marker.
- The single-camera calibration needs the prop and its surroundings to stay still. Solving
  the origin and axes of the prop is left to you.
- Dots worn on a person still need their slot set prepared with scripts.
- The colour patches around dots, which could help tell dots apart, are not used.
- Importing points as **labelled** markers (naming each track after a marker) is not available;
  imported tracks are anonymous dots.
- The graphical tools do not yet expose every option described here.

## Quick reference

| Task | Command |
|---|---|
| Solve a body, fixed cameras | `marker-body calibrate` |
| Solve a body, one moving camera | `marker-body calibrate-video` |
| Import / list / show a body | `marker-body import`, `list`, `show` |
| Prop skeleton from a body | `marker-body to-skeleton` |
| Add / list / rename / remove props | `capture object add`, `list`, `rename`, `rm` |
| Detect markers or dots | `detect run --type aruco` or `--type dots` |
| Import tracks from another program | `detect import-2d` |
| Sequence of a prop | `sequence finalise-object` |
| Dots into an existing sequence | `sequence add-dots` |
| Track one or several subjects | `track run`, `track run-persons` |
| Results | `track list`, `track show`, `track export gltf\|bvh\|usd` |
