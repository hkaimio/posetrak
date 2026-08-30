# Segmenting multi-person scenes

<div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;margin:1.5em 0;">
<iframe src="https://www.youtube.com/embed/wr1fi7PVvnw"
        style="position:absolute;top:0;left:0;width:100%;height:100%;border:0;"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>
</div>

Posetrak estimates a person's pose from a video frame in two steps:
first a detector finds the people in the frame, then a separate pass
crops out just the area around each person and looks for pose
keypoints in that crop.

This works well when tracking a single person. With several people in
frame, though, Posetrak can easily lose track of which detection
belongs to which person from one frame to the next — you'd then have
to manually stitch each person's timeline together from the raw
detections afterwards, which is slow, especially when people move
around a lot in the camera views.

For these harder scenes, Posetrak has an advanced segmentation tool,
similar to what video editors use to separate an object from its
background. It creates a mask for each person and remembers frames
it has already processed, so it can follow people through a scene
far more reliably than the plain person detector.

## Installing the segmentation tool

The segmentation tool needs an NVIDIA GPU and additional software
components that take several gigabytes of disk space, so it isn't
installed by default. Select it separately when running the Posetrak
installer.

## Starting a segmentation

Segmentation must be done *before* running pose detection for a trial.
Select the trial you want to track and click **Create segmentation**.

## The segmentation window

The segmentation window shows video from one camera at a time — switch
which camera is shown using the dropdown above the video.

Below the video is a timeline with three color-coded stripes:

- **Top (amber)** — the time range covered by the trial you're
  tracking. If you're going to track several trials from the same
  captured video set, you can segment all of them in one pass here;
  otherwise just work within this range.
- **Middle (blue)** — the time range currently selected for
  segmentation, set with the **Mark Start**/**Mark End** buttons.
  Defaults to the whole trial. The segmentation algorithm only runs
  over this range.
- **Bottom** — already-segmented frames in green, frames queued for
  segmentation but not yet processed in gold.

## Creating your first masks

Scrub to a frame where everyone you want to track is clearly visible.
Select a person's name in the area below the video, then click that
person in the video — a colored mask should appear over them. If the
mask isn't quite right, click again (on the person, or on a spot that
shouldn't be included, to correct it). Repeat for every person you
want to track.

## Running the segmentation

Click **Segment Range from Seed**. The masks you just created are the
seed — this runs the segmentation algorithm to extend them to every
frame in the selected (blue) time range, propagating both backward and
forward from the seed frame.

The job doesn't start immediately — it's added to the **Job Queue**
panel on the right. Segmentation takes real time, and you need to do
it for every camera, so the efficient way to work is to set seed
frames for all cameras first, queuing a job for each, and only then
let your computer work through the whole queue while you take a break.

When you click **Segment Range from Seed**, notice the bottom stripe
of the timeline turns gold for the frames you just queued — that's
your visual confirmation the job is queued, not yet run.

When you've set up seed frames for everything you want processed,
click **Run Queue** to actually process all the queued jobs.

## Using the segmentation for pose detection

Once you're happy with the results, close the segmentation window, go
to the trial page, and start pose detection (**Run detection…**). In
the **Bbox source** dropdown, select the segmentation you just created
instead of the default **YOLO detection** option.

## Handling difficult footage: splitting into segments

For longer sequences where people cross paths closely and repeatedly,
even the segmentation algorithm can make mistakes — usually right at
the moment two people's masks would otherwise merge or swap. Rather
than segmenting a whole difficult sequence as one continuous pass,
split it into shorter, easier segments and give each one its own seed
frame.

Identify the moments in the video that look potentially difficult, go
to that frame, and click **Mark Split** (✂). Repeat for every difficult
moment in the sequence.

!!! note "Split points are per camera"
    Split points aren't shared across cameras — you mark them
    separately for each one shown in the camera dropdown. This
    actually fits the problem well: whether a moment is genuinely hard
    to segment depends on the camera's own viewing angle (two people
    can visibly overlap from one camera while staying clearly
    separated in another's view), so each camera's split points can
    legitimately land on different frames.

Once the whole video is split into segments, work through them one at
a time: find a good seed frame within a segment, click **Snap Marks to
Segment** (⇤) to set Mark Start/Mark End to the nearest split points
around it, select the people in that frame and place their seed masks
as before, then queue a segmentation job for just that sub-range.

If part of a video ends up incorrectly masked after running the queue,
you don't need to redo the whole thing: set Mark Start/Mark End around
just the incorrect time range, fix up a seed frame within it (or use
the paint/erase tools — see below — to correct a mask directly), and
run segmentation again for that part only.

## Not covered here: manual touch-ups

The segmentation window also has **Paint**, **Erase**, and **Zoom**
tools (alongside the click-based **Select** tool used above) for fixing
a mask by hand on a single frame — useful when segmentation leaves a
stray mislabeled patch behind, for example where two people were in
contact. That's its own topic and isn't covered in this recording.

## See also

- [Your first capture](first-capture.md)
- [Analyzing poses](analyzing-poses.md)
- [Tracking & troubleshooting](tracking-troubleshooting.md)
