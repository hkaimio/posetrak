# Initial brief of productizing the GEM-X/SOMA detector in Posetrak

## What worked in proto spike

- The SOMA keypoint set is clearly better starting point for tracker. It should be makde the default just due to this reason
- The idea of moltiple candidates from detections & trackign time assignment is useful but not universally proven better

### GEM-X tracker

- IMHO, make it default but keep ViTPose too
- Add supprot for hand refinement pass
- Should work both with segmentation mask and without
- The tracker performance needs improvement. First thin is to tune the pipeline so that GPU is fed new frames so that it stays at 100%

### Multiple detections

- Implement support in database. WHether it is on should depend on detection model
- Tracking time should have options to switch on/off the strict mask adherence (with the tracklet inside mask in a  time window criteria) & cross-camera corrobation (we should find a more descriptive name for this!)
- Should utilize data from all person's detections if they overlap. SHould have mechanism that detects same tracklets/keypoint detections from multiple detection rounds so that
- Assignment should consider also anonymous marker based tracking - the framework should be common although details may differ

### keypoint editing

- Keypoint editing (and visualization) needs changes due to tracklets & multiple candidates
- Tracklets should be the "first class citizen" in keypoint editor
- Should support also splitting tracklets & time range based opertations
- Multiple candidates: reject a tracklet (cannot be used)
- Force tracklet (override tracking time selection and even mahal gate sothat the tracklet is always used)
- All tracklet opertaion should be available also for groups of similar tracklets (clustered by their time range & behavior, ref the case where head & shoulder tracklets switched to different person at same time - this shold be correctable witha single operation, Posetrak detecting which trackelts to split/exclude)
- Tracklet clustering should also consider cross-camera similarities (if tracklets triangulate to sensible 3D trajectory)
- OVerall, I found the UI in the prototype labeling tool nice to use - develop posetrak keypoint editor to same direction
- Visualization: how to visualize the multiple keypoints
- iterative tracking would significantly improve keypoint editing
  - after edits, rerun tracker just for the changed time range
  - requires storing snapshots of tracking state so that it can be restarted
  - Also smoothign needs to be time window based
  - In the end, full tracking run of course needed for accurate results

### Seegmentation & Mask usage

- Masks are not immutable - if they are modified, also detections may need update (depends on whether the bbox changes too much) and previous tracking runs cannot be reproduced. Some kind of versioning needed for those.
- Segmentation UI is good enough to use but would benefit from improvements - thsi should be a separate feature

## What did not work

## Other findigns worth productizing

- Extrinsics calibration from calib target video (expose in UI)
- Possibly extrinsics calibration from keyponts detectons (at least as a recipe for command line use/AI)
- Video caching for fast scrubbing - this has been improved a lot, and the proto labeling tools is even better than the improved Posetrak


## Other problems

- Default skeleton joint issues
- db locking - several runs failed silently when using both command line and GUI
