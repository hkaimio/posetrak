# Keypoint editing — user guide

## When to use this feature

Posetrak is designed around a multi-camera setup and an Unscented Kalman Filter (UKF) that naturally rejects noisy or missing detections.  With enough cameras, you rarely need to touch individual keypoints: the filter cross-validates each camera against the others and simply ignores implausible measurements.

Manual keypoint editing is for the cases where that assumption breaks down — when detection fails badly enough, across enough cameras, for long enough, that the filter can no longer maintain a reliable state.  Two situations that come up repeatedly in practice:

- **Unusual poses** — the person bends into a position the pose detector has not seen often (floor work, overhead reach, extreme lateral flexion).  The detector either misses keypoints entirely or fires them at wrong locations, and the errors are consistent enough across views that the filter cannot reject them.

- **Person proximity** — two people cross or overlap.  The detector finds the correct pixel locations for ankles or wrists, but assigns them to the wrong person.  If this happens in most cameras simultaneously, the outlier rejection fails.

**The goal is not to make every keypoint in every camera correct.**  It is narrower: give the filter enough correct signal to hold onto the right skeleton state through the difficult period.  In most cases this means fixing the one or two most informative cameras for the worst frames, not correcting all cameras exhaustively.  Once the filter has good anchors at the start and end of a bad stretch, interpolation can fill the middle automatically.

Edits are stored separately from the original detection data and are applied as an overlay when the tracker reads observations.  You can always re-run detection without losing your edits, and you can re-run the tracker at any time to pick up new edits.

---

## Entering edit mode

1. In the main window, navigate to a person inside a pose observation sequence in the session tree.
2. The person panel opens showing the multi-camera crop grid.
3. Click **Edit** (or press **E** when the grid is focused) to enter edit mode.  The cursor changes and keypoint dots become interactive.

Edit mode is per-person and per-sequence.  Edits from a previous tracker run are preserved when you re-run; if you re-run the stitch step and create a new sequence ID the edits do not carry over.

---

## Navigating frames

| Key | Action |
|---|---|
| `D` | Next frame |
| `A` | Previous frame |

Frame navigation works whether or not edit mode is active.  The crop grid loads directly from the JPEG cache — no video file is read while scrubbing.

Frames where the person was not detected (gaps in the detection run) display a wider crop synthesised from nearby frames.  These *ghost frames* are the ones you most often need to edit.

---

## Selecting a keypoint

Click any coloured dot in any camera cell to select that keypoint.  A **trail** appears in all camera cells showing the past (red) and future (blue) positions of that keypoint across nearby frames.  Dotted trail positions on ghost frames are linearly interpolated from the nearest real detections and give you a visual sense of where the keypoint should be.

### Extending the selection

| Interaction | Result |
|---|---|
| Ctrl+click a dot | Add that keypoint to the selection (or remove it if already selected) |
| Drag on empty area | Rubber-band: selects all dots inside the rectangle |
| Right-click on cell | Context menu with named groups (Face, Left arm, Right hand, etc.) |
| `Esc` | Clear selection |

When multiple keypoints are selected, editing operations (nudge, Space toggle, interpolation) apply to all of them simultaneously.  The trail is shown for the *primary* keypoint (the last one you explicitly clicked) to avoid visual clutter.

---

## Moving a keypoint

### Mouse

Drag any selected keypoint dot to a new position.  The edit is written on mouse release.

On a ghost frame with no existing detection, clicking anywhere on the cell places the primary selected keypoint at that location.

### Keyboard nudge

| Key | Move |
|---|---|
| `←` `→` `↑` `↓` | ±1 px (full-frame coordinates) |
| `Shift` + arrow | ±10 px |

Nudge applies to all selected keypoints.

---

## Marking keypoints as outliers / inliers

Press `Space` to toggle the selected keypoint(s) between inlier and outlier at the current frame.  Outlier keypoints are drawn in grey and the tracker ignores them.  Inlier keypoints are drawn in their normal colour and the tracker uses them.

Use this to suppress a badly detected keypoint that you cannot easily move to the right position, or to reinstate a keypoint that was incorrectly auto-rejected.

---

## Working with a frame range

For longer problem periods, working frame by frame is slow.  Frame range mode lets you operate on a span of frames at once.

### Setting a range

| Key | Action |
|---|---|
| `Shift+D` | Anchor the range at the current frame, extend one step right, advance |
| `Shift+A` | Anchor the range at the current frame, extend one step left, retreat |
| Repeat | Keep pressing to extend the range further in that direction |
| `A` or `D` (no Shift) | Clear the range and step normally |
| `Esc` | Clear the range |

Selected range frames are indicated by white rings on the keypoint trail positions.

### Range operations

With a range active:

- **Space** — marks the selected keypoint(s) as outliers (or inliers) across every frame in the range.  This is the quickest way to suppress a burst of bad detections.

- **I** — linear interpolation (see next section).

---

## Interpolating across a gap

The most common workflow for a detection gap:

1. Find a frame just before the gap where the keypoint position is still correct.  This becomes the left anchor.
2. Find a frame just after the gap where detection has recovered.  This becomes the right anchor.
3. Select the keypoint(s) you want to fix.
4. Set a frame range that spans from the left anchor to the right anchor (use `Shift+D` / `Shift+A`).
5. Press **I**.

Posetrak uses the first and last frames of the range as anchors and writes linearly interpolated positions for all inner frames.  The anchor frames themselves are not modified.  Any keypoint where either anchor has zero confidence (outlier or missing) is skipped.

After interpolation the range is cleared and the display refreshes.

**Tip**: you do not need to anchor exactly on a real detection.  If neither boundary frame has a good detection, first place the keypoint manually on those frames (ghost-frame click or nudge), then set the range and press I.

---

## Copy and paste

| Key | Action |
|---|---|
| `Ctrl+C` | Copy all selected keypoints at the current frame to the clipboard |
| `Ctrl+V` | Paste the clipboard into the current frame |

Paste writes only the copied slots; other keypoints in the frame are unchanged.  Pasting onto a ghost frame creates a new edit row for that frame.

A typical use: find a frame where the pose detector got most of the body right but missed a few keypoints.  Copy a good nearby frame, paste it here to fill the missing slots, then nudge the individual keypoints to their correct positions.

---

## Practical workflow for a problem segment

A reasonable approach for a multi-frame tracking failure:

1. **Locate the segment** — scrub through the sequence and find where the skeleton diverges.  Look for frames where most camera cells show clearly wrong keypoints or empty ghost crops.

2. **Identify the cameras that matter** — choose one or two cameras with a clear view of the body part that the filter is losing track of.  Do not try to fix every camera.

3. **Mark bulk outliers first** — select the keypoints causing the most damage, set a range over the worst frames, and press `Space` to suppress them.  Re-run the tracker and see how much this alone helps.

4. **Add anchors at the boundaries** — scrub to the first frame of the gap and place the keypoint where the person actually is.  Do the same at the last frame of the gap.  Use ghost-frame click and nudge; Ctrl+C/Ctrl+V from a nearby good frame is often faster.

5. **Interpolate** — select the relevant keypoints, set the range, press I.

6. **Re-run the tracker** — edits are picked up automatically on the next tracker run.  Check whether the segment now tracks correctly before refining further.

---

## Keyboard reference

| Key | Action |
|---|---|
| `A` / `D` | Previous / next frame; clears active range |
| `Shift+A` / `Shift+D` | Extend frame range left / right |
| `Esc` | Clear keypoint selection and frame range |
| `←` `→` `↑` `↓` | Nudge selected keypoint(s) ±1 px |
| `Shift` + arrow | Nudge ±10 px |
| `Space` | Toggle outlier/inlier on selected keypoint(s) at current frame or range |
| `I` | Interpolate selected keypoint(s) across the active range |
| `Ctrl+C` | Copy selected keypoint(s) |
| `Ctrl+V` | Paste clipboard into current frame |
