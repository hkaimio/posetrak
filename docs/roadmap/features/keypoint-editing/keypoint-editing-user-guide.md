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
2. The person panel opens showing the multi-camera crop grid, with the keypoint timeline docked below it.
3. Click **Edit** (or press **E** when the grid is focused) to enter edit mode.  The cursor changes and keypoint dots become interactive.

Edit mode is per-person and per-sequence.  Edits from a previous tracker run are preserved when you re-run; if you re-run the stitch step and create a new sequence ID the edits do not carry over.

---

## Navigating frames

| Key | Action |
|---|---|
| `D` | Next frame |
| `A` | Previous frame |

Frame navigation works whether or not edit mode is active.  The crop grid loads directly from the JPEG cache — no video file is read while scrubbing.

You can also click or drag directly on the keypoint timeline's ruler (see below) to jump to any point in the trial — this is usually faster than stepping frame by frame once you know roughly where the problem is.

Frames where the person was not detected (gaps in the detection run) display a wider crop synthesised from nearby frames.  These *ghost frames* are the ones you most often need to edit.

---

## The keypoint timeline

Below the crop grid, a dope-sheet style timeline shows every keypoint's status across the whole trial at a glance, and doubles as the scrub control. It's the fastest way to find *where* a trial needs manual correction, without stepping through it frame by frame.

It starts **collapsed** to a single ruler strip — scrubbing works without expanding it. Click the small ▸ arrow at the left of the ruler to expand it and see the per-keypoint rows; click it again (now ▾) to collapse.

### Scrubbing

Click or drag anywhere on the ruler (the row with tick marks) to move the playhead. This works whether or not edit mode is active, and whether or not the timeline is expanded. The ruler's timestamps match the current-time label above the crop grid — both show the same point on the capture's global clock, not time relative to the trial start.

### Zoom and pan

- **Ctrl+scroll** over the timeline, or the **−** / **+** buttons, zoom in or out around the current playhead position.
- **Fit** resets the view to the whole trial.
- Once zoomed in, a horizontal scrollbar appears below the rows for panning.
- At high enough zoom (roughly past 6 px per frame), individual frame cells get a small gap between them so you can see exactly where one frame ends and the next begins.

### Reading the rows

Each row is one keypoint, or a collapsible group of keypoints (click the ▶/▼ arrow on a group row to expand or collapse it). A row's colour at a given time shows that keypoint's status **for the camera currently shown in the tab strip**:

| Colour | Meaning |
|---|---|
| Green | Original detection, inside the person's segmentation mask |
| Yellow | Original detection, outside the segmentation mask (often unreliable) |
| Blue | Edited — moved, re-enabled, or frozen as a keyframe |
| Grey | Disabled (marked outlier), or no usable detection at all |

A thin bar under each keypoint row shows what fraction of the *other* cameras already have a good detection for that keypoint at that time. A nearly-full bar means the gap you're looking at in the currently shown camera probably isn't worth fixing by hand — the filter likely already has enough signal from elsewhere.

The timeline automatically switches to whichever camera you last selected a keypoint in (in either the crop grid or the timeline itself), so you don't need to manually keep the two in sync. Zoom, pan, and expand/collapse state are preserved across edits, frame changes, and camera switches.

### Selecting and editing from the timeline

Selection follows the same underlying rules as the crop grid:

| Interaction | Result |
|---|---|
| Click a row | Clear the current selection (this also moves the playhead — see *Scrubbing* above) |
| Drag over rows/time | Select every visible keypoint touched by the drag, replacing the current selection, and set the active frame range to match |
| Ctrl+drag | Add the touched keypoints to the selection instead of replacing it |
| Ctrl+click a single keypoint's cell | Toggle a keyframe at that exact frame — see *Multi-keyframe interpolation* below |

A translucent highlight shows the current selection and the active frame range (snapped to whole frames, since that's what actually gets selected and interpolated).

### Hiding keypoints

Each row has a small eye icon at the right edge of its label. Click it to hide that keypoint — or, on a group row, every keypoint in the group at once. Hidden keypoints are dimmed in the timeline and disappear from the crop grid entirely: they cannot be selected, dragged, clicked, rubber-band selected, or interpolated until you click the eye icon again to show them.

This is mainly useful for decluttering a rubber-band selection or a "Select all" when you only care about part of the body — hide everything else first, then select freely without worrying about accidentally grabbing something you didn't mean to touch.

Clicking a group's eye icon when it's only *partially* hidden hides the rest, rather than doing nothing — the icon reads as "showing" until every keypoint in the group is hidden.

Visibility is a viewing preference for the current editing session only; it is not saved with the sequence, and resets the next time you open the editor.

---

## Selecting a keypoint

Click any coloured dot in any camera cell to select that keypoint.  A **trail** appears in all camera cells showing the past (red) and future (blue) positions of that keypoint across nearby frames.  Dotted trail positions on ghost frames are linearly interpolated from the nearest real detections and give you a visual sense of where the keypoint should be.

You can select keypoints from the crop grid or from the timeline (see above) — both operate on the same underlying selection, so you can mix the two freely.

### Extending the selection

| Interaction | Result |
|---|---|
| Ctrl+click a dot | Add that keypoint to the selection (or remove it if already selected) |
| Drag on empty area | Rubber-band: selects all dots inside the rectangle |
| Right-click on cell | Context menu with named groups (Face, Face (detail), Left arm, Right hand, etc.) |
| `Esc` | Clear selection |

When multiple keypoints are selected, editing operations (nudge, Space toggle, interpolation) apply to all of them simultaneously.  The trail is shown for the *primary* keypoint (the last one you explicitly clicked) to avoid visual clutter.

"Face" only covers the keypoints the default skeleton actually attaches markers to (nose and both ears). The remaining face landmarks (eyes plus the 68 detailed contour/eyebrow/lip points from the whole-body pose model) are grouped separately under "Face (detail)" — pull that in only if you specifically need the fine-grained face landmarks.

Hidden keypoints (see *Hiding keypoints* above) are excluded from all of these — they never appear in a rubber-band selection, a named-group selection, or "Select all".

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

| Key / interaction | Action |
|---|---|
| `Shift+D` | Anchor the range at the current frame, extend one step right, advance |
| `Shift+A` | Anchor the range at the current frame, extend one step left, retreat |
| Repeat | Keep pressing to extend the range further in that direction |
| Drag on the timeline | Set the range directly to whatever frames the drag covers (see *The keypoint timeline* above) |
| `A` or `D` (no Shift) | Clear the range and step normally |
| `Esc` | Clear the range |

Selected range frames are indicated by white rings on the keypoint trail positions in the crop grid, and by a translucent highlight on the timeline.

### Range operations

With a range active:

- **Space** — marks the selected keypoint(s) as outliers (or inliers) across every frame in the range.  This is the quickest way to suppress a burst of bad detections.

- **I** — interpolation (see next section).

---

## Interpolating across a gap

### Simple case: one gap, two anchors

The most common workflow for a single detection gap:

1. Find a frame just before the gap where the keypoint position is still correct.  This becomes the left anchor.
2. Find a frame just after the gap where detection has recovered.  This becomes the right anchor.
3. Select the keypoint(s) you want to fix.
4. Set a frame range that spans from the left anchor to the right anchor (`Shift+D` / `Shift+A`, or drag on the timeline).
5. Press **I**.

Posetrak uses the first and last frames of the range as anchors and writes linearly interpolated positions for all inner frames.  The anchor frames themselves are not modified.  Any keypoint where either anchor has zero confidence (outlier or missing) is skipped.

After interpolation the range is cleared and the display refreshes.

**Tip**: you do not need to anchor exactly on a real detection.  If neither boundary frame has a good detection, first place the keypoint manually on those frames (ghost-frame click or nudge), then set the range and press I.

### Multi-keyframe interpolation

If a range contains more than one bad patch, or you want to keep a few known-good frames in the middle instead of overwriting them with a single straight line, mark those frames as **keyframes** before pressing I. Any of the following turns a frame into a keyframe for the selected keypoint:

- Move the keypoint to its correct position on that frame, or
- Press `Space` to re-enable it if it's currently marked as an outlier, or
- `Ctrl`+click its cell on the timeline to freeze it at its current position without changing where it is.

Every frame in the active range that's been turned into a keyframe like this becomes an anchor.  Interpolation then works piecewise between however many anchors you've placed, instead of always drawing one straight line between the range's two ends — a keypoint moved at frames 1, 10, and 20 interpolates as two independent segments (1→10, then 10→20), and the keyframed frames themselves are left untouched.

If you don't place any interior keyframes, this behaves exactly like the simple two-anchor case above: **an original detection that merely happens to sit inside the range is never treated as a keyframe**, even if it's currently a confident inlier.  So "select a wide range covering a few bad frames among otherwise-good ones, press I" still overwrites everything in between with a single straight line — it does not accidentally preserve the bad frames just because they weren't disabled first.

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

1. **Locate the segment** — expand the keypoint timeline and scan for a stretch of grey/yellow rows, or scrub through the sequence and find where the skeleton diverges.  Look for frames where most camera cells show clearly wrong keypoints or empty ghost crops.  The per-keypoint inlier-count bar on the timeline is a quick way to tell whether a gap in one camera actually matters — if most other cameras are still green at that time, it probably doesn't.

2. **Identify the cameras that matter** — choose one or two cameras with a clear view of the body part that the filter is losing track of.  Switching the crop-grid selection to a camera also switches the timeline to match, so you can keep working in one place.  Do not try to fix every camera.

3. **Mark bulk outliers first** — select the keypoints causing the most damage, set a range over the worst frames, and press `Space` to suppress them.  Re-run the tracker and see how much this alone helps.

4. **Add anchors (and optional interior keyframes) at the boundaries** — scrub to the first frame of the gap and place the keypoint where the person actually is.  Do the same at the last frame of the gap.  Use ghost-frame click and nudge; Ctrl+C/Ctrl+V from a nearby good frame is often faster.  If there's a reliable frame in the middle of the gap worth keeping, `Ctrl`+click it on the timeline to keyframe it too.

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
| `I` | Interpolate selected keypoint(s) across the active range (see *Multi-keyframe interpolation*) |
| `Ctrl+C` | Copy selected keypoint(s) |
| `Ctrl+V` | Paste clipboard into current frame |

## Timeline mouse reference

| Interaction | Action |
|---|---|
| Click/drag on the ruler | Move the playhead |
| Click a row | Clear selection (and move the playhead) |
| Drag over rows/time | Select touched keypoints + set frame range |
| Ctrl+drag over rows/time | Add touched keypoints to the selection |
| Ctrl+click a keypoint's cell | Toggle a keyframe at that frame |
| Click a row's eye icon | Hide/show that keypoint (or group) |
| Click a group row's ▶/▼ arrow | Expand/collapse the group |
| Click the ▸/▾ arrow on the ruler | Expand/collapse the whole timeline |
| Ctrl+scroll | Zoom around the playhead |
