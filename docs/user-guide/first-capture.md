# Your first capture

Main steps in creating a motion capture with Posetrak.

## Equipment needed

- Cameras.
  - In theory 2 might work, but 3 or more is recommended. More is better, and needed for more complex scenes.
  - Can be any cameras that can record video — they don't need to be identical models (although it makes things simpler).
  - Do not use autofocus or image stabilization.
  - Must be mounted on a tripod or other support that prevents them from moving.

### Optional

- Calibration rig & fiducial markers. Not mandatory, but they help a lot with extrinsics calibration.
- Synchronization light. Any blinking LED (e.g. a headlamp) — helps find frames captured at the exact same moment by different cameras.

## Before your first capture

- Calibrate camera intrinsics.
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

## Processing

- Copy the videos to your computer. Use a systematic naming convention for the files (I create a directory for each capture session, then put the videos there with names like `<date>-<capture name>-<camera name>-<camera mode>-<resolution>-<frame rate>.mp4`).
- Create a new capture in the Posetrak UI. Import the videos and select the right camera for each.
- Synchronize the videos. For each video you need to set at least 2 frames that are captured at the same moment as a frame in another video. If you used the hand-clapping technique or the blinking light, this should be easy.
- Calibrate extrinsics, i.e. figure out the exact 3D position and orientation of each camera.
- Select the range of the capture you want to track — in Posetrak this is called a trial.
  Set the start & end frames in the UI and click "New trial".
- Analyze poses from each video.
  - Posetrak supports 2 ways of doing this:
    - Detect persons at the same time the poses are analyzed. Usually works well for simple scenes, but gets confused if multiple persons constantly cross paths.
    - For these more complex cases, Posetrak can first run a segmentation: it analyses the video using an advanced video segmentation algorithm plus some user assistance, and stores exactly which pixels in each frame belong to each performer. Requires an additional step, but is much more robust for multi-person scenes.
- Run the tracker. This is the key step that calculates joint angles from the data.

## Analysis & fixing

- If the tracking looks good enough, you're done. However, it often takes more work to get good results.
- Adjust skeleton.
  - If the skeleton dimensions (length of legs, arms & spine) don't match the performer, the tracker can't find an exact match to their poses. You might need to adjust the skeleton manually.
- Adjust tracker parameters.
  - If the tracker doesn't follow movements correctly (guide TBD).
- Edit keypoints.
  - Sometimes Posetrak doesn't detect the poses correctly, or doesn't detect some body parts of the person at all in some camera frames. Usually this isn't a problem if the frame is analyzed correctly from at least some cameras, but as a last resort Posetrak has tools to manually edit the detected poses.

## Export results

- The motion capture results are exported as a BVH file that can be imported into almost all 3D animation applications.
