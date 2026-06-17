# Posetrak feature: Keypoint editing

Design brief

## Problem statement

Sometimes pose detection fails for a period of frames. Posetrak's UKF-based skeleton solver handles individual outliers and missing detections well, but if there is a period of several frames with no correct detections its confidence and accuracy decreases, or it can even converge to a wrong state based on incorrect keypoint detections.

Two typical situations where this happens are:
* The person being tracked is in an unusual pose and the pose detector fails to recognize keypoints correctly. The result is either misplaced keypoints or no keypoints at all (depending on the confidence threshold set in the tracker).
* The pose detector gets confused by two persons who are close to each other. In these situations it often correctly detects the location of e.g. an ankle or wrist but fails to assign them to the correct person.

If these errors occur in individual frames or in a few cameras, Posetrak usually rejects the incorrect detections as outliers. But if there is a longer period where these occur in most of the cameras, tracking fails. Manual pose editing aims to help in these situations — the target is not to make all detection series flawless.

### Operations in manual pose editing
* Mark a keypoint in a frame or frame range as an outlier (or inlier)
* Move a keypoint in a frame or frame range to a new location

### Data model considerations
* Does the edited detection a) become a new detection series, b) modify the existing detection series, or c) become a new version of the existing detection series? Preference is c.
* Are the edited keypoints saved instantly to the database or is there a separate "save" operation? The latter is somewhat against the general Posetrak UI logic. One possibility (assuming c from the previous point) would be that edits are saved immediately to the database as the new "latest version", and later it is possible to mark the current state as a frozen version to which one can revert. Running the tracker should always create a frozen version for reproducibility.

### UI

* This should be part of the detection UI page.
* Keypoints are already displayed with color coding (green = active, gray = outlier in the currently selected run). This is a good starting point.
* When a keypoint is selected, its trail is displayed (e.g. 10 previous & next locations — previous ones in red, future ones in blue, with lines connecting them). This should be visible in all camera views.
* The selected keypoint can be moved to a new location by dragging it with the mouse.
* Keyboard shortcuts — using the WASD idiom (goal is that all editing COULD be done from keyboard too):
  * `a` / `d` — move to previous/next frame
  * `Shift+A` / `Shift+D` — extend selection to the keypoint in the previous/next frame
  * Cursor keys — move the selected keypoint. `Shift` + cursor → larger step
  * `Space` — toggle keypoint inlier/outlier
  * Future addition: selecting multiple keypoints, e.g.:
    * `s` — select first child of the currently selected keypoint
    * `Shift+S` — select current keypoint and all in the hierarchy below it
    * `w` — select parent of the current keypoint
    * `z` / `x` — previous / next sibling
    * etc.

### Other considerations

* How to handle situations where a keypoint is not detected at all for a frame? This is the typical case — the user wants to manually place the keypoint where the pose estimator failed, but there is no existing keypoint to move. Suggestion: always draw the trail all the way to the nearest known keypoint position (even if it is further in the past/future than the normal trail length) and interpolate missing detections in the UI at even intervals along the trail (these must not be saved to the database unless the user moves them or marks them as inliers).
* Currently the database caches person bounding boxes only for frames where a person was detected. In many cases we need to edit frames where detection failed. Suggested solution: cache the person bounding box also for these frames (see cropping logic below).
* Current cropping logic is based on the person bounding box detected for the frame, but if detection fails this may be too tight. Suggestion: select cached image cropping so that the cropped region is extended to encompass all detections in the N previous and future frames.
