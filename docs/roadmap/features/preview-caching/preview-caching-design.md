# Preview frame caching for tracking runs, including props — design discussion

Status: proposal for discussion. Nothing here is implemented.

## 1. How caching works today

There are three separate mechanisms, all built around *people*.

**1. Persistent person crops (`frame_cache_entries`).** A row is a small JPEG
of one person's detection box (plus a 20 % margin), at most 240 px tall,
quality 75, keyed by camera video, frame, track, region and detection run. It
is written while detecting (`DetectionBatchWriter`) and afterwards by
`CropBackfillWorker` for any detected frame that has no row yet. The schema
also lists `full_frame` and `thumb` types; nothing writes them. On the sword
session the table holds 213,830 `person_crop` rows and 9,973 `ghost_crop`
rows, inside a 7.6 GB session database.

**2. The backfill worker (`CropBackfillWorker`, one thread per open panel).**
It queues every detected frame lacking a row, in frame order, and moves a
frame to the front when the user scrubs to it. A frame with *no* detection
takes the "ghost" path: it crops the union of nearby detection boxes, or, if
there are none at all, shrinks the whole frame to 240 px height. The result
is kept in memory and also saved as a `ghost_crop` row. Every frame is
fetched with `cv2.VideoCapture.set(POS_FRAMES)` followed by `read()`, one
frame and one camera at a time.

**3. The wide-crop cache (`WideCropExtractWorker`, edit mode only).** It walks
each video sequentially in 0.4 s epochs, builds each person's padded window,
merges overlapping people into one shared crop, and keeps JPEGs (quality 90,
long edge at most 1200 px) in an in-memory index, shared between open panels
of the same detection run. It is never persisted.

**What the viewer does with them.** `PersonCropGridWidget._load_frame` looks
up, in order: the wide cache (edit mode), the backfill's in-memory result, a
`person_crop` row within ±3 frames, and otherwise shows the "generating…"
placeholder and asks the backfill worker to prioritise that frame. Overlays
(detected keypoints, tracked skeleton, segmentation mask) are drawn afterwards
in full-frame coordinates using each image's stored source rectangle.

**Props take a different route.** The object review widget
(`ObjectCropGridWidget`) has no cache: one `FrameReader` thread per camera
decodes the full frame on demand, latest request wins. The tracking-run panel
for an object run, however, reuses `PersonCropGridWidget`. An object sequence
has no person tracks or boxes, so every frame goes through the ghost path,
which yields a 240 px-tall thumbnail of the whole frame (too small to see a
reflective dot) produced one camera at a time.

## 2. Why scrubbing a prop run is slow

Measured on a 4K, 120 fps HEVC file of the sword session:

- A random seek plus decode costs 0.3–0.6 s per frame; reading the next frame
  sequentially costs about 10 ms.
- Five cameras are handled by one worker in sequence, so landing on a new
  frame costs roughly 2–3 s before all cameras have updated.
- Resizing to 640×360 and JPEG-encoding takes about 3 ms; a full 4K JPEG at
  quality 90 is about 830 KB and takes 11 ms.

So the cost is random-access decoding, done serially, with nothing to show
in the meantime. Secondary problems: the cache is keyed by person detection
(no box, no proper crop), the ghost thumbnail is too small for props, the
crops live in the main session database, and the person and object paths do
not share code.

## 3. Requirements

1. Scrubbing gives immediate feedback from cached, reduced-resolution images.
2. When scrubbing stops, the true full-resolution frame (or a full-resolution
   crop of the region of interest) appears, and neighbouring frames are
   prefetched so single-frame stepping stays quick.
3. Works for any subject: people, props, and frames with no detection.
4. Keeps working when the original video files are unavailable (network or
   offline); the display then says it is showing a proxy.
5. Cache can be inspected, pruned and rebuilt from the command line and the UI.
6. The cache must not make the session database bigger or harder to back up.

## 4. Proposal

Two image tiers, a region-of-interest provider, and per-camera decoders.

### 4.1 Tiers

**Scrub proxy.** A reduced whole-frame image for (nearly) every frame, made
once by a background job. Example sizes for a 640×360 JPEG at quality 70:
31 KB each, about 3 ms to make. Stored in a **sidecar cache database**
(`<session>.cache.db`, same table shape as `frame_cache_entries`), not in the
session database. The sidecar is disposable: deleting it loses nothing, and
it is excluded from trial export. Frame step is configurable, so the size is a
trade-off:

| Range (5 cameras) | every frame | every 2nd | every 4th |
|---|---|---|---|
| Sword trial, 6.7k frames each | 1.0 GB | 0.5 GB | 0.26 GB |
| Whole capture, 42k frames each | 6.5 GB | 3.3 GB | 1.6 GB |

Building costs about 13 ms per frame (sequential decode plus resize and
encode): roughly 1.5 minutes per camera for the sword trial, with the cameras
running in parallel. The build is scoped to a trial or range, not the whole
capture by default.

**Full-resolution settle.** After about 150 ms without scrubbing, each camera's
persistent decoder fetches the exact frame at full resolution and crops it to
the region of interest. While the user frame-steps, the decoder reads ahead
sequentially (about 10 ms per frame) and fills an in-memory LRU of ready
images. Crops may also be persisted in the sidecar as an `roi_crop` type.

### 4.2 Region of interest provider

The wide-crop cache already answers "what rectangle should this person's view
show". Generalise it to `roi_for(subject, camera, frame) -> rectangle | None`,
tried in order:

1. A person detection box (today's behaviour).
2. The tracking run's projected markers for that frame (the display code
   already unions these into the view rectangle).
3. The bounding box of the subject's label in the segmentation mask (uses the
   subject registry's permanent label).
4. The object detection's keypoint or dot-candidate box.
5. The whole frame.

For a prop the person-detection step simply yields nothing, so the same code
serves both cases.

### 4.3 Decoding

One persistent decoder thread per camera (as `FrameReader` does today, but
for every camera in parallel and shared by all views of the capture),
preferring: proxy for scrubbing, exact decode when settling, sequential read
for stepping and prefetch. Five cameras settling in parallel takes about 0.5 s
instead of 2–3 s. Without the source video the settle step falls back to the
proxy, upscaled and labelled.

### 4.4 Tooling

- CLI: `posetrak cache status`, `posetrak cache build --trial X [--step N]`,
  `posetrak cache prune [--max-size]`, with `--tier proxy|roi`.
- UI: a cache section in the capture and trial panels (size, build for this
  trial, prune), and a low-priority automatic build of the active trial when a
  tracking run is opened.
- Size-capped LRU pruning of `roi_crop` entries; the proxy is only removed
  explicitly.

## 5. Order of work (each step useful alone)

1. **Parallel persistent decoders and debounce.** Per-camera decoders with
   read-ahead, a settle debounce, and a larger placeholder for props than the
   240 px ghost thumbnail. No schema change. Removes most of the waiting.
2. **Region of interest for objects.** The provider above, used by the
   tracking-run panel so a prop run shows a useful crop and the segmentation
   overlay.
3. **Sidecar proxy cache.** Build job, sidecar database, offline fallback and
   the CLI.
4. **Management UI and unification.** Cache section in the UI, automatic
   pruning, and moving the person path onto the shared decoder and provider.

## 6. Alternatives considered

- **Keep crops in `frame_cache_entries` in the session database.** Simplest,
  but it makes a 7.6 GB database larger and couples cache lifetime to data
  that must not be lost.
- **A proxy video file per camera** (all-intra MJPEG through OpenCV, no
  ffmpeg needed). Comparable size and instant seeking, but harder to prune
  per range and to keep alongside the session; per-frame rows in a sidecar
  database prune and rebuild trivially.
- **A keyframe-aware decoder library (PyAV).** Fixes seek cost for the full
  resolution tier but adds a dependency, and does nothing for offline use.
- **Only sparse scrub thumbnails, no proxy for stepping.** Cheapest, but every
  step then waits on a 0.3–0.6 s seek unless read-ahead is working.

## 7. Open questions

1. Sidecar cache database next to the session database, or a per-user cache
   folder shared across sessions?
2. Default proxy: 640×360 every 2nd frame, or 960×540 every 4th? Dots are tiny
   at 640 px wide; the settle step covers detail, so the proxy only needs to
   convey motion.
3. What should the offline experience promise: scrub only, or also stepping
   at proxy resolution?
4. `ObjectCropGridWidget` (marker correction) needs exact full-resolution
   frames and will use only the settle path; should it share the decoders, or
   stay separate?
5. The 9,973 existing `ghost_crop` rows: leave them, or drop them when the
   proxy exists?
