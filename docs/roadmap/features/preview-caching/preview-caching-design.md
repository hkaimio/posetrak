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

**What is and is not persisted.** Full-resolution frames are never stored in
the session database. A whole 4K frame is about 830 KB as a JPEG, so a
capture would need roughly 175 GB (42k frames, 5 cameras): the video file
stays the store for full resolution, and re-decoding an exact frame takes
about 0.5 s. Only the proxy (and, optionally, crops of frames you have looked
at) is persisted, in the sidecar. Offline therefore means proxy-only, unless
you "pin" a range: a native-resolution crop of a prop's region is about
113 KB, so pinning the 6.7k-frame sword trial for 5 cameras costs about
3.8 GB.

**Proxy step follows the slider, not the frame rate.** A 56 s trial on a
slider of about 800 px is roughly 8 frames per pixel, so a slider drag cannot
address individual frames anyway. A proxy every 8th frame is enough for
dragging (about 0.13 GB for the sword trial and 5 cameras); every frame of
detail comes from the settle and read-ahead path. Per-frame stepping needs
its own controls (section 4.6).

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

### 4.3 Decoding: PyAV, not OpenCV

The persistent decoders should use PyAV (`av`, already a declared dependency,
and already what detection uses through `posetrak/detection/frame_source.py`),
not `cv2.VideoCapture` as the preview code does today. Measured on the same
4K, 120 fps HEVC file (keyframe every 120 frames):

| | OpenCV | PyAV, threaded | PyAV, single thread |
|---|---|---|---|
| Random seek to an exact frame | 0.27–0.62 s | 0.12–0.47 s | 0.28–2.6 s |
| Sequential frame, decode only | 3.0 ms (`grab`) | 3.1 ms | 22 ms |
| Sequential frame, decoded to BGR | 9.9 ms | 5.5 ms | 26 ms |

What PyAV adds beyond speed:

- **Frame-exact and consistent with detection.** Seeking by timestamp landed
  on the requested frame every time; the number of frames decoded from the
  keyframe (12 to 115) matches the position inside the interval. The frame
  index is derived from the timestamp exactly as detection does it, so a
  preview frame is the frame a detection or tracking result refers to. OpenCV
  counts frames itself, which can disagree on variable-frame-rate phone
  footage, and the code already has a defensive rescale for the two decoders
  disagreeing on resolution.
- **Keyframe index for free.** Reading the packet headers without decoding
  found all 352 keyframes of a 42k-frame file in 1.9 s. That gives the exact
  block boundaries the backward policy in section 4.4 needs, instead of
  assuming the interval. The index can be stored in the sidecar per video.
- **Control over conversion and threads.** Only frames we keep are converted
  to BGR, and thread count can be set per decoder.
- **Rotation handled in one place.** Detection already corrects portrait
  phone footage (stream rotation tag or display matrix); the preview would use
  the same code instead of relying on OpenCV's auto-rotation.

What PyAV does not change: a frame is still only reachable from its keyframe,
so the 0.2–0.5 s random-seek cost and the block policy for backward stepping
remain. It also removes nothing for offline use.

**Caveat.** `frame_source.py` documents that closing a container while
threaded decoding is still mid-stream can hang forever (seen with a stack dump
of a worker stuck in the parser), and therefore decodes single-threaded there.
Single-threaded decode is 5–7 times slower here (22 ms per frame), too slow
for the preview. A quick test of 8 close-after-50-frames cycles with threaded
decoding did not hang, but that is not proof. So the decoders must be built so
a hang cannot block the interface: each decoder runs on its own thread, is
closed from that thread with a watchdog, and is abandoned (not joined) if it
does not finish in a few seconds. To be validated with a longer soak test
before relying on threaded decode.

### 4.4 Memory cache and prefetch

**What the memory cache holds.** For each camera and frame, the region of
interest plus a margin, at native resolution, kept as JPEG bytes (about
113 KB for a 1500×800 region at quality 90) and decoded only when displayed
(about 5–10 ms). Holding decoded images would cost 3.6 MB each, more than 30
times as much. It is neither the whole frame (25 MB raw) nor exactly the
displayed window: the displayed window is cut from the cached region, and the
existing black-fill covers the case where an overlay needs a little more. The
region is computed per frame, so a fast-moving prop does not need the next
frame's crop to overlap this one's.

**Keyframe spacing decides the policy.** This footage has a keyframe every
1.001 s, 120 frames. A seek lands on the previous keyframe and decodes
forward, which is the 0.3–0.6 s measured earlier. Forward reading is cheap,
60 frames in about 0.5 s; reading backwards is not, because a frame can only
be reached from its keyframe. Measured: a seek plus 60 sequential reads takes
0.9 s.

- **Forward:** keep about 60 frames (0.5 s of footage) ready ahead of the
  current frame; each camera's decoder fills them in about 0.5 s, all
  cameras in parallel.
- **Backward:** decode in blocks, not per frame. When the user steps back, or
  gets within about 15 frames of the start of what is cached, decode the whole
  previous keyframe interval (120 frames, about 1.2 s) and keep it. Stepping
  back through that second is then free.
- **Direction bias:** after two steps in one direction, spend about two thirds
  of the budget in that direction.
- **Budget:** about 256 MB in total, evicting by distance from the current
  frame. At 113 KB × 5 cameras that is about 450 frame-times, 3.7 s of
  120 fps footage. A jump larger than the window drops the old window and
  starts a new one at the target after the settle delay.
- The keyframe interval differs between cameras and captures (a 30 fps clip
  may have 30–60 frames per interval), so the block size is read from the file,
  not assumed.

### 4.5 Tooling

- CLI: `posetrak cache status`, `posetrak cache build --trial X [--step N]`,
  `posetrak cache prune [--max-size]`, with `--tier proxy|roi`.
- UI: a cache section in the capture and trial panels (size, build for this
  trial, prune), and a low-priority automatic build of the active trial when a
  tracking run is opened.
- Size-capped LRU pruning of `roi_crop` entries; the proxy is only removed
  explicitly.

### 4.6 Per-frame stepping and playback controls

The timeline slider is the only control today, so no user action is a single
frame step and the cache has never had a stepping case to serve. Adding one is
part of this work, because stepping is what read-ahead exists for:

- Step buttons and keys: previous/next frame (arrow keys), 10 frames
  (Shift+arrow), and jump to the previous/next tracker step or keyframe.
- Play/pause at a chosen rate, using sequential decode (about 100 frames per
  second per camera at 4K, so real-time 120 fps is borderline and playback
  should drop frames to 30 fps display).
- Slider drag stays the "scrub" case: served from the proxy, with the exact
  frame after the settle delay.

## 5. Order of work (each step useful alone)

1. **Parallel persistent decoders, debounce and stepping controls.**
   Per-camera decoders with read-ahead and backward block decoding, a settle
   debounce, a larger placeholder for props than the 240 px ghost thumbnail,
   and the per-frame step and play controls. No schema change. Removes most of
   the waiting.
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
- **Keep OpenCV (`VideoCapture`) for decoding.** Works, but it is slower per
  frame, gives no keyframe information, indexes frames by its own count rather
  than by timestamp, and is a different decoder from the one detection uses;
  see section 4.3.
- **Only sparse scrub thumbnails, no proxy for stepping.** Cheapest, but every
  step then waits on a 0.3–0.6 s seek unless read-ahead is working.

## 7. Open questions

1. Sidecar cache database next to the session database, or a per-user cache
   folder shared across sessions?
2. Default proxy: 640×360 every 8th frame (matches slider granularity, about
   0.13 GB per trial), or a denser or larger one? Dots are tiny at 640 px wide;
   the settle step covers detail, so the proxy only needs to convey motion.
3. What should the offline experience promise: scrub only, or also stepping
   at proxy resolution?
4. `ObjectCropGridWidget` (marker correction) needs exact full-resolution
   frames and will use only the settle path; should it share the decoders, or
   stay separate?
5. The 9,973 existing `ghost_crop` rows: leave them, or drop them when the
   proxy exists?
