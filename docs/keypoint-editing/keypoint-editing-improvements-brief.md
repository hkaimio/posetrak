The keypoint editing feature is quite useful but still clumsy to use.
- The cached images are often too narrowly cropped.
- Interpolating a time range is time consuming.
- Estimating results is difficult, as the user needs to rerun the tracker for the whole trial.

## Improvement ideas

## Timeline view
- Add a timeline view. This would be similar to Blender's timeline view or Cascadeur's timeline:
  - Each keypoint is represented by a small rectangle. Color indicates state: green - original, inside segmentation; yellow - original, outside person segment; blue - edited; grey - disabled.
  - One row per keypoint. Tree structure - body parts can be collapsed. The body part rows show a summary status of child keypoints (e.g. split coloring based on the status of children in that frame).
  - Possible to select by dragging with the mouse (area of keypoints in a time range); ctrl-click to add/remove.
  - Same keyboard shortcuts work as in the image view (space disables/enables, i interpolates region, etc.).
  - For interpolation, also support interpolation through multiple keyframes. Basic workflow: user disables keypoints in a time range, then adds/enables keypoints for a few frames in the range (incl. start & end frames). Interpolation is then done through the enabled keyframes.
    - How to indicate whether the user wants this or the existing behavior? Simply overwriting everything in the time range with interpolation is also a valid use case.

## Partial tracking

Support rerunning the tracker for the modified time range.
- Option to store checkpoints of the UKF tracker state, e.g. every 1 s, including covariance matrices etc.
- After editing keypoints, the user can test the impact of the changes by running a short tracking pass from the checkpoint preceding the start of the edits.
- How to handle RTS smoothing? Should the checkpoints store the smoothed state?
- Need a "temporary tracking" concept - these test tracking runs are ephemeral and should be removed eventually.
- How to handle visualization: the UI should show the original tracking up to the test tracking start frame, then switch to the test run.

## Keypoint/camera/frame specific measurement errors

- Keypoint measurement errors seem to vary a lot between keypoints. Fingers are quite accurate, but hip keypoint detections in particular can have large errors.
- Add a default per keypoint (maybe a multiplier applied to the value set in the tracking config).
- Add an option to adjust this in the editor, e.g. show the keypoint stddev as a circle/ellipse that can be grown/shrunk with keyboard shortcuts. This should also work for a whole time range and support interpolation.
