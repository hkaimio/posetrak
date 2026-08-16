# Your first capture

Main steps in creating a motion capture with Posetrak. This is meant as
an overview/checklist — the linked pages cover each step's details.

<!-- TRIM CANDIDATE (whole doc): once the linked detail pages below are
filled in, most of the explanatory sub-bullets here can shrink to a
one-liner + link, so this page reads as a checklist a new user can
follow start-to-finish without hitting a wall of detail. Specific
candidates are marked inline below. -->

## Equipment needed

- Cameras.
  - In theory 2 might work, but 3 or more is recommended. More is better, and needed for more complex scenes.
  - Can be any cameras that can record video — they don't need to be identical models (although it makes things simpler).
  - Do not use autofocus or image stabilization.
  - Must be mounted on a tripod or other support that prevents them from moving.

<!-- TRIM CANDIDATE: the autofocus/stabilization/tripod bullets are really
"how to get a good intrinsics calibration and stable extrinsics" advice.
Could shrink to "cameras must stay put and use a fixed focus" here, with
the why moved to camera-intrinsics.md / extrinsics-calibration.md. -->

### Optional

- Calibration rig & fiducial markers. Not mandatory, but they help a lot with [extrinsics calibration](extrinsics-calibration.md).
- Synchronization light. Any blinking LED (e.g. a headlamp) — helps find frames captured at the exact same moment by different cameras. See [synchronizing videos](synchronizing-videos.md).

<!-- TRIM CANDIDATE: now that both have detail pages, these two bullets
could each drop to a single clause ("a calibration rig helps, see
extrinsics-calibration.md") rather than explaining why. -->

## Before your first capture

- Calibrate [camera intrinsics](camera-intrinsics.md).
- Motion capture is complex and it's easy to make mistakes. Practice before you go capture an important performance!

## Capture

- Set up the cameras on all sides of the area where the performance will happen. Ideally the cameras should be at roughly equal intervals, and their fields of view should together cover the whole area.
- If you have fiducial markers, attach them to walls, the floor, etc. so that each camera sees at least a few of them.
- If you have a calibration rig, put it on the floor in the center, where you'd like the coordinate origin to be.
- Position the sync light so that all cameras can see it (and so it's not usually occluded by performers).

- Turn the cameras on.
- After capturing the calibration rig on video, remove it.
- I usually add a "clapper board" sync signal at the beginning and end of each video to help with synchronizing — e.g. clap my hands above my head so they're visible in all cameras.
- It's good practice to start and end the capture with performers standing in an A or T pose. Not mandatory, but it helps with skeleton calibration and ensures the tracking algorithm gets the initial pose right.

<!-- TRIM CANDIDATE: the sync-light and clapper-board bullets duplicate
technique detail that now lives in synchronizing-videos.md ("Setting up
sync points"). Could shrink to "set up your sync signal (see
Synchronizing videos)" and let that page own the how/why. -->

## Processing

- Copy the videos to your computer. Use a systematic naming convention for the files (I create a directory for each capture session, then put the videos there with names like `<date>-<capture name>-<camera name>-<camera mode>-<resolution>-<frame rate>.mp4`).

<!-- TRIM CANDIDATE: this is a personal-workflow aside, not a Posetrak
feature — candidate to cut entirely, or move to a short "organizing
capture files" tip if one gets written, rather than living in the main
checklist. -->

- Create a new capture in the Posetrak UI. Import the videos and select the right camera for each.
- [Synchronize the videos](synchronizing-videos.md). For each video you need to set at least 2 frames that are captured at the same moment as a frame in another video. If you used the hand-clapping technique or the blinking light, this should be easy.
- [Calibrate extrinsics](extrinsics-calibration.md), i.e. figure out the exact 3D position and orientation of each camera.

<!-- TRIM CANDIDATE: both bullets above can drop to "see [page]" one-liners
once those pages are complete — the "for each video you need..." and
"i.e. figure out..." clauses are exactly what the linked pages now
explain in full. -->

- Select the range of the capture you want to track — in Posetrak this is called a trial.
  Set the start & end frames in the UI and click "New trial".
- [Analyze poses](analyzing-poses.md) from each video.
  - Posetrak supports 2 ways of doing this:
    - Detect persons at the same time the poses are analyzed. Usually works well for simple scenes, but gets confused if multiple persons constantly cross paths.
    - For these more complex cases, Posetrak can first run a segmentation: it analyses the video using an advanced video segmentation algorithm plus some user assistance, and stores exactly which pixels in each frame belong to each performer. Requires an additional step, but is much more robust for multi-person scenes.

<!-- TRIM CANDIDATE: the two sub-bullets comparing detection approaches
duplicate analyzing-poses.md's "Direct detection" / "Segmentation-
assisted detection" sections almost one-for-one. Once that page is
filled in, shrink this to "two ways — direct detection (simple scenes)
or segmentation-assisted (multi-person/crossing paths), see [Analyzing
poses](analyzing-poses.md)". -->

- Run the tracker. This is the key step that calculates joint angles from the data.

## Analysis & fixing

- If the tracking looks good enough, you're done. However, it often takes more work to get good results.
- Adjust skeleton.
  - If the skeleton dimensions (length of legs, arms & spine) don't match the performer, the tracker can't find an exact match to their poses. You might need to adjust the skeleton manually.
- Adjust tracker parameters.
  - If the tracker doesn't follow movements correctly (guide TBD).
- Edit keypoints.
  - Sometimes Posetrak doesn't detect the poses correctly, or doesn't detect some body parts of the person at all in some camera frames. Usually this isn't a problem if the frame is analyzed correctly from at least some cameras, but as a last resort Posetrak has tools to manually edit the detected poses.

See [tracking & troubleshooting](tracking-troubleshooting.md) for all three of the above.

<!-- TRIM CANDIDATE: this whole section is really "what to do when
tracking doesn't work", now covered (or stubbed) in
tracking-troubleshooting.md. Once that page has real content, this
section could shrink to 3 one-line pointers ("skeleton not matching the
performer?", "tracker losing movements?", "a few frames detected wrong?")
each linking to the matching subsection there, instead of restating the
explanation here too. -->

## Export results

- The motion capture results are exported as a BVH file that can be imported into almost all 3D animation applications.
