# Detection pipeline: one pass, pluggable box source, SOMA-77 as a native pose model

## 1. Problem

Person pose detection has three code paths today, each with its own frame loop:

| Path | Entry | Boxes | Frame loop |
|---|---|---|---|
| YOLOX + RTMPose | `posetrak detect run`, `DetectionPipeline` (`posetrak/detection/pipeline.py`) | YOLOX with its tracker, per frame | PyAV, single-threaded decode, one camera after another |
| Segmentation | `detect run --segmentation`, GUI Cutie panel; `PoseWorker` (`app/pose/pose_worker.py`) | Tight box of each mask label | `cv2.VideoCapture`, per-frame PNG decode of the mask, resize to native size |
| SOMA-77 prototype | `python/tools/prototypes/soma77_extract.py` | Copied from an existing detection run | PyAV, batches of eight frames |

The SOMA model therefore needs a source run, which for YOLOX footage means a full RTMPose pass
only to get boxes (about 35 minutes per trial in the spike). The hand refinement pass decodes the
video a second time (`HandRefinementPipeline`, `iter_frames`).

## 2. Design

One `detect run` command and one frame loop, with two plug points:

```
BoxSource ──boxes per (camera, frame)──▶ PoseEstimator ──keypoints (+ candidates)──▶ writer
   ├─ YoloxBoxSource       (detector + its tracker; track ids are fragments)
   ├─ MaskBoxSource        (seg_quality_run + revision; track id = mask label = subject label)
   └─ RunBoxSource         (person_detections of an existing run, with its track assignments)
```

- A box source (`posetrak/detection/box_sources.py`) is called in two places so that each part runs where it is
  cheap: `prepare(camera, video_frame, image)` on the decode thread of the camera (CPU only: the mask source
  decodes the mask and finds its boxes, the YOLOX source resizes the image to the model input; it must be
  thread-safe), and `detect(works)` on the inference thread, the frames of one camera in order (the YOLOX model
  call and its tracker). `start_camera` runs once per camera before its frames (tracker reset, boxes of a run
  loaded) and `wants` lets a source say that a frame cannot have boxes, so that the loop does not hand its image
  on. The mask source attaches the person's grown mask to each `PersonDetection` (`mask`, at a working resolution
  of 480 pixels across) so that the estimator can gate heatmaps with it.
- `PoseEstimator` (existing protocol in `backends.py`) gets a batched form,
  `estimate_batch(items) -> list[list[PoseResult]]`, where an item is a frame with its detections, the frames
  of any cameras. RTMPose and SOMA-77 both implement it; the single-frame `estimate` stays for callers that need
  it. `options()` returns every option that changes the keypoints, for `config_json`. A `PoseResult` may carry
  its own `noise_scale` (SOMA-77 writes the width of the model window in frame pixels per model input pixel; the
  writer's default is the box width over the model input width).
- SOMA-77 (`backends_soma77.py`) is registered as estimator `soma77-vitpose-h`, with the options the
  spike validated: flip test, crop enlargement 1.6, parabolic refinement, decoding at cell centres
  (index + 0.5), heatmap gating by the person's mask when the box source provides masks. Optional
  candidate output (several maxima per keypoint) is specified in
  [keypoint-candidates](../keypoint-candidates/keypoint-candidates-design.md).
- The run records its inputs: `detection_runs.seg_quality_run_id` and `seg_revision` when masks
  provided boxes or gated heatmaps, `source_detection_run_id` when boxes were copied from a run, and
  every estimator option in `config_json` (see the architecture and schema summary).
- A mask-sourced run is finalised automatically, as today. A YOLOX-sourced run goes through track
  to person assignment ([track-person-assignment](../track-person-assignment/track-person-assignment-design.md)).
- Hand refinement runs inside the same loop on the frame already decoded, instead of a second pass
  ([hand refinement without segmentation](../hand-detection-refinement/hand-refinement-without-segmentation-design.md)).
- `PoseWorker` and the GUI dialogs call the same loop; `pose_worker.py`'s own loop is removed.

### Feeding the GPU

The loop is split into three stages connected by bounded queues:

1. **Decode** — one producer per camera video, yielding frames of the range (`frame_readers.py`: a thread per
   camera, or a child process per camera that hands the frames over in shared memory; section 3 has the soak
   test that chooses). Each producer also prepares the frame for the box source, so mask decoding runs here too.
2. **Infer** — one thread that takes the frames of all producers, asks the box source for their boxes and runs
   the estimator on the crops of all persons of up to six frames at once, in model calls of a fixed size
   (16 crops for SOMA, the size measured at 12 ms per crop, the flipped copies counted), then the hand pass.
3. **Write** — the thread that called the loop writes the rows in batches (the database connection belongs to
   it) and encodes the preview crops; the loop reports progress from there.

The GPU then waits only when all producers together cannot keep up, which the profiling below
shows or rules out.

YOLOX is the exception to batching across cameras: the shipped YOLOX ONNX graphs have batch size 1 baked
into their reshape nodes (changing the batch dimension of the input makes the first reshape fail), so the YOLOX
box source makes one model call per frame. Its image resizing moves to the decode threads and all its model calls
are made by the single inference thread, in the order the frames arrive. A batched YOLOX needs a re-exported
model.

The hand pass has the same limit: the hand detector (RTMDet-nano) has batch size 1 and runs per hand crop; the
hand pose model takes any batch and runs once for all detected hands of the batch.

Decoding is single-threaded per video on purpose: closing a container with threaded decoding before
the end of the file can hang in FFmpeg (`frame_source.py`). Parallelism comes from decoding several
cameras at once, not from threaded decoding of one; hardware decoding is an option to measure.

### Variable frame rate

The pose model need not run on every frame when the persons are far apart. Per global time `t`,
the loop decides from the boxes of all cameras whether `t` is a **full-rate** instant:

- any two persons' enlarged boxes overlap in any camera, or
- any person's box moves faster than a threshold (box heights per second), or
- the instant lies within a margin of a full-rate instant.

Other instants are processed at every `N`th frame (default 1, i.e. off). Every camera skips the
same instants. The box source still runs on every frame (YOLOX tracking needs it); only the pose
model is skipped. Skipped frames have no rows; the tracker predicts through them. The same rule can
later drive the segmentation propagation rate, which is part of the segmentation feature.

This only pays when the pose model, not decoding, dominates the time; it is built after the
profiling and only if it does.

## 3. Profiling plan

Where the time goes is not known. GPU utilisation readings so far are single samples (about 96%
during the SOMA pass, 45-50% during the RTMPose pass). Code reading gives hypotheses only:

- 4K video decoded single-threaded, one camera at a time (`_iter_frames_av`, `thread_type = "NONE"`).
- A JPEG preview crop encoded per detection inside the loop (`DetectionBatchWriter.add_frame`).
- The segmentation route decodes a PNG mask per frame, resizes it to native size with nearest
  neighbour and scans it with `np.where` per label (`pose_worker.py`).
- RTMPose runs one frame's crops per call (no batching across frames); SOMA runs the flip test,
  doubling the model work.

Measurement on a fixed clip (10 s of the five-camera labelled trial), optimized environment, nothing
else running:

1. Per-stage wall time per frame, from timers around decode, box source, crop preparation, model
   call, decoding of the output, crop encoding and database write, enabled by an environment
   variable (the same pattern as `POSETRAK_PROFILE_FRAME_STEP` in the tracker).
2. A sampling profile of the whole process (`py-spy record --native`) to catch time outside the
   timers (GIL, conversion to `ndarray`, colour conversion).
3. GPU utilisation sampled once a second for the whole run (`nvidia-smi dmon -s u`), reported as
   mean and distribution, not single readings.
4. Routes: YOLOX + RTMPose-133; segmentation boxes + RTMPose-133; segmentation boxes + SOMA-77
   with and without flip test; RTMPose with hand refinement on and off.
5. Decode alone (frames read and dropped) per camera, with PyAV software decode and with hardware
   decode if available, as the floor any pipeline change can reach.

The result is a table of milliseconds per frame per stage and route; the pipeline work starts from
the largest entry.

### Instrumentation

`POSETRAK_PROFILE_DETECTION=1` switches on per-stage timing in the detection loop
(`posetrak/detection/stage_timing.py`); any other value is the path of a JSON file that receives the numbers
in addition to the table printed at exit. Stages: `decode`, `box_source`, `crop_preprocess`, `inference`,
`heatmap_decode`, `mask_gate`, `hand_model`, `preview_encode`, `db_write`; the hand pass reports its own
stages under `hand_pass/`. Times are exclusive (a nested stage is not counted in its parent). With the
variable unset the decorated functions and iterators are the original objects (a unit test checks this).
The SOMA-77 prototype route (`soma77_extract.py`) has no hooks of its own; its stages are measured from
outside with `python/tools/prototypes/profile_detection_routes.py`, which wraps the video container, the mask
reader and the connection and runs the prototype code unchanged.

### Results

Material: the five-camera, three-person trial, 10 s (46-56 s of the capture timeline), 6,000 camera-frames
(three 4K cameras at 120 fps, two FHD phone cameras at 120 fps), person masks of an edited segmentation run,
RTX 4080 SUPER, PyAV software decode, `onnxruntime-gpu` 1.23. Every run wrote its detection run into a copy
of the session database. Build: the commit that adds the instrumentation (no other change to the loop). Seconds are
totals over the clip; GPU utilisation is `nvidia-smi` sampled once a second (mean / 10th / 90th
percentile). "Masks + SOMA" is the prototype route (boxes copied from an existing segmentation-sourced run,
heatmaps gated by the person mask, batches of eight frames), for one of the three persons; the other
routes handle all three persons of every frame.

| Route | wall s | decode | box source | crop + preprocess | inference | heatmap decode + gate | hand pass | preview encode | db write | outside stages | GPU % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Decode alone (PyAV) | 124 | 121 | - | - | - | - | - | - | - | 3 | 2 / 1 / 3 |
| YOLOX + RTMPose-133 | 395 | 135 | 143 | 17 | 90 | 2 | - | 3 | 1 | 4 | 42 / 30 / 51 |
| Masks + RTMPose-133, no hand pass | 437 | 47 | 149 | 19 | 212 | 2 | - | 3 | 1 | 4 | 44 / 31 / 58 |
| Masks + RTMPose-133, hand pass | 918 | 45 | 147 | 19 | 206 | 2 | 485 | 3 | 1 | 9 | 39 / 28 / 52 |
| Masks + SOMA-77, flip test, one person | 339 | 124 | 40 | 5 | 144 | 5 | - | - | 0 | 21 | 40 / 0 / 96 |
| Masks + SOMA-77, no flip test, one person | 280 | 134 | 41 | 5 | 73 | 5 | - | - | 0 | 21 | 22 / 0 / 69 |

Commands (shell, `PYTHONUTF8=1`, the variable set to a JSON path, GPU sampler `nvidia-smi
--query-gpu=utilization.gpu,... -l 1` alongside):

- `profile_detection_routes.py decode SESSION SHOT 46 56`
- `posetrak --session SESSION detect run --capture C --sync S --start 46 --end 56 --detector yolox-x --pose-model rtmpose-l-133kp`
- the same with `--segmentation RUN --no-refine-hands` and `--segmentation RUN --refine-hands`
- `profile_detection_routes.py soma SESSION SOURCE_RUN gem-x/onnx/vitpose.onnx 46 56 --mask-run RUN --mask-label 1`, and with
  `--no-flip` (inference of the unflipped crop only; the keypoints it writes are not meant to be used)

Per-unit costs behind the table:

| Unit | Cost |
|---|---|
| Decode of one 4K camera-frame (PyAV, software) | 27-31 ms (36.8 s for 1,199 frames of the slowest camera); FHD phone camera 9 ms |
| Decode of the same frames with `cv2.VideoCapture` (segmentation route) | 7.5-7.8 ms mean over all five cameras |
| YOLOX box source | 23.8 ms per frame |
| Mask box source in the RTMPose route (PNG decode, nearest resize to native, scan per label) | 24.9 ms per frame, three persons |
| Mask decode and grow in the SOMA route | 6.7 ms per frame, one person |
| RTMPose-133 inference | 5.7 ms per crop in the YOLOX route, 11.5-11.9 ms per crop in the mask route |
| SOMA-77 inference | 192 ms per batch of 16 crops (eight frames with their flipped copies): 12 ms per crop; 97 ms per eight crops without flip |
| Hand pass | 19.5 ms per body crop (both hands), plus a second decode of the whole clip (22.6 ms per frame) |
| Preview JPEG encode | 0.16 ms per crop |

Sampling profile of the slowest route (masks + RTMPose-133 with the hand pass; `py-spy record --native -r 20
-f raw --pid <process>`, 21,021 samples of the process; the run took 1,105 s with the profiler attached
against 918 s without): 51% of the samples are inside the ONNX Runtime `run` call (body and hand models), 46%
inside the hand pass's `detect_hand_in_crop`, 19% in the body pose `estimate`, and 13% in the mask handling
`_bboxes_from_mask` (nearest-neighbour resize of the mask to the native resolution and the scan per label)
against 4% for reading and decoding the stored PNG mask. The leaf functions are dominated by waiting for
the GPU: `NtGdiDdDDISubmitCommandToHwQueue` (command submission to the GPU queue) 22% and the CUDA runtime's
synchronisation 9%, then libavcodec 11.5% (the hand pass's second decode) and OpenCV 7%. No time outside the
timed stages stands out; the main thread spends about a third of its time waiting for the GPU while the GPU
itself is on average 39% busy, which is the signature of a serial loop in which decode, mask handling and
inference never overlap.

What this says about the hypotheses above:

- **Decode is a large and serial cost, the model is not the only bound.** Decoding alone takes 124 s for the
  10 s of capture (12 times real time, camera after camera). In the SOMA route it is 37-48% of the wall time
  and the GPU is idle while it runs: the utilisation is bimodal (10th percentile 0%, 90th percentile 96%,
  mean 40%), not a steady 96%; the earlier single readings of 96% are consistent with samples taken during the inference bursts.
- **Once decoding is overlapped the model dominates.** The slowest camera decodes in 37 s; SOMA inference for
  one person is 144 s with the flip test and 73 s without. A decode thread per camera would hide the decode
  (projection, not measured), and the bound would become inference (144 s for one person, about 430 s for
  three), which is also where a variable frame rate and the flip test would pay.
- **The prototype route decodes the video once per person and per pass.** Three persons need three passes
  (about 3 x 339 s), after a source pass for the boxes (395 s from YOLOX, 437 s from masks). A one-pass
  pipeline with overlapped decode would be bounded by inference, about 150 s per person (projection), so
  the acceptance target (twice the prototype's throughput) leaves a wide margin, including the source pass
  the prototype needs.
- **The YOLOX box source is as expensive as decoding** (36% of the route) and larger than the RTMPose model
  itself. The mask box source of the RTMPose route is also costly (34%): PNG decode, a nearest-neighbour resize
  of the mask to 4K and a scan per label. Both are cheaper in the SOMA route (mask decode 12%).
- **Preview encoding is not a problem** (0.7%), and nothing is lost by keeping it in the loop; it can move to
  the writer thread without effect on the numbers.
- **The hand pass doubles the RTMPose mask route** (918 s against 437 s): the hand model and a second decode
  of the clip. Inside the frame loop the second decode disappears (136 s); the hand model remains
  (349 s, 38% of the route) and is a candidate for batching.
- **Explained (GPU clock):** RTMPose inference was twice as expensive per crop in the segmentation route
  (11.5-11.9 ms) as in the YOLOX route (5.7 ms) for the same model, device and options. A stand-alone loop of the
  estimator with `cv2.VideoCapture` reads and a Qt core application gave 6.8 ms per crop, so neither explained it.
  The unified loop's measurement (results below) shows the same gap, 7.8 against 11.3 ms per crop, and the GPU
  clock accounts for it: 2757 MHz while the model is busy in the YOLOX route, 985 MHz in the segmentation route,
  where the GPU is idle between short bursts and clocks down. The cost is a property of a lightly loaded GPU, not
  of the code path, and it disappears when the GPU is kept busy, as it is in the SOMA routes.

### Decode producers: threads or processes

`python/tools/prototypes/decode_soak.py` takes any set of camera videos (here the five videos of the profiling
clip), opens them at once, decodes random
windows of 10 s in each and closes them, over and over, in one mode, with a watchdog that counts a step (open
until the first frame, decode, close) that makes no progress for 60 s as a hang. In process mode the reader's own
stall timeout (60 s without a frame) kills the child, which is counted as a recovered stall; a hung decode
thread cannot be killed, so it is abandoned and counted as a leaked thread.

| Run | Mode | Length | Windows | Frames/s (all videos) | Hangs open / decode / close | Recovered stalls | Errors |
|---|---|---|---|---|---|---|---|
| Smoke test, 2 s windows | thread | 38 s | 83 | 178 | 0 / 0 / 0 | - | 0 |
| Smoke test, 2 s windows | process | 34 s | 62 | 154 | 0 / 0 / 0 | 0 | 0 |
| Soak, 10 s windows | thread | 30 min | pending (operator) | | | - | |
| Soak, 10 s windows | process | 30 min | pending (operator) | | | | |

Default: **threads** until the soak run says otherwise. Neither mode hung in the smoke test, threads decode
about 15% faster in it (the process mode copies each 4K frame out of shared memory), and threads are what the
loop was measured with. A hang in the soak run's thread mode, or any leaked thread, moves the default to
processes (`--decode process`, `DetectionPipeline(decode_mode="process")`); the one constant to change is
`DEFAULT_DECODE_MODE` in `pipeline.py`.

### Smoke test of the unified loop

One second of two captures of the real-session copy used for the migration check (not the profiling clip: four
cameras, 4K and FHD, 120 fps; 484 camera-frames), SOMA-77 with the flip test, `onnxruntime-gpu`, RTX 4080 SUPER,
the machine in ordinary use (browser, desktop; not the quiet machine of the measurements above), decode on threads.
The stage times are summed over the threads of the loop.

| Route | wall s | decode (sum of 4 threads) | box source | crop + preprocess | inference | heatmap decode + gate | preview encode | GPU % mean (range) | GPU memory peak above idle |
|---|---|---|---|---|---|---|---|---|---|
| YOLOX + SOMA-77, 374 persons | 21.4 | 21.8 | 12.3 | 0.4 | 8.4 (22.6 ms per person) | 0.3 | 0.1 | 74 (65-93) | 6.4 GB |
| Masks + SOMA-77, 480 persons | 13.4 | 22.0 | 3.5 | 0.9 | 11.6 (24.2 ms per person) | 0.5 | 0.1 | 83 (76-94) | 5.4 GB |

Reading: the model costs 11-12 ms per crop (flip included), as in the single-person prototype; with masks the
inference thread is the bound and the GPU is busy 83% of the time on average over a 13 s run that includes loading
the model, so the 85% criterion is within reach on the clip; with YOLOX the serial batch-1 YOLOX model calls
(about 12 ms per frame on the inference thread) take a share of the inference thread and the GPU is waiting for
them, which a re-exported batched YOLOX would remove. Peak GPU memory is YOLOX-x, ViT-H with 16-crop batches
and the CUDA context together; the decode threads hold no GPU memory.

### Results of the unified loop

Same material, hardware and commands as above (idle machine, work-copy database migrated to session version 66,
`POSETRAK_PROFILE_DETECTION` set, `nvidia-smi` once a second), run on the finished loop with decode on threads.
The stage times are summed over the threads of the loop, so they add up to more than the wall time; "outside
stages" is not meaningful and is left out. GPU memory is the peak above the idle level of the desktop.

| Route | wall s | decode | box source | crop + preprocess | inference | heatmap decode + gate | hand pass | preview encode | db write | GPU % mean | GPU memory GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOX + SOMA-77, all persons (15,960 crops) | 586 | 211 | 158 | 19 | 392 | 18 | - | 4 | 2 | 79 | 6.5 |
| Masks + SOMA-77, all persons (17,850 crops) | 506 | 196 | 64 | 28 | 449 | 20 | - | 4 | 2 | 82 | 5.4 |
| Run boxes + SOMA-77 (boxes of run `b05e51a9`, 17,337 crops) | 484 | 208 | - | 30 | 429 | 19 | - | 4 | 2 | 80 | 5.5 |
| YOLOX + RTMPose-133 | 311 | 217 | 159 | 31 | 125 | 1 | - | 4 | 2 | 39 | 3.5 |
| Masks + RTMPose-133, no hand pass | 247 | 209 | 65 | 43 | 202 | 2 | - | 5 | 2 | 31 | 1.3 |
| Masks + RTMPose-133, hand pass | 677 | 209 | 65 | 99 | 315 | 4 | 253 | 5 | 2 | 29 | 1.9 |

Findings:

- **Throughput.** The masks + SOMA-77 route takes 506 s for the three persons of the clip. The prototype route is
  1,454 s (source pass 437 s plus 3 x 339 s), so the loop is 2.9 times faster; the criterion of 2 times holds.
- **GPU use.** 79-82% on the SOMA routes, short of 85%. Inference is 90% of the inference thread's work (about
  25 ms per person, the flip test included) and decode (196-211 s over four threads, the slowest camera about 60 s)
  is below the wall time, so the model is the bound and the GPU is idle only where the inference thread does
  other work: crop preparation (1.2-1.7 ms per crop) and, with YOLOX, the serial model calls (13 ms per frame).
  Reaching 85% needs that work off the inference thread (crop preparation in the decode threads, a batched YOLOX).
- **Hand pass.** 677 s against 918 s before (no second decode). The hand model is 253 s (253 ms per batch of hand
  crops, hand detector per crop).
- **RTMPose cost gap.** It remains: 7.8 ms per crop in the YOLOX route and 11.3 ms in the mask route (5.7 and
  11.7 before). The GPU clock explains it: while the model is busy the SM clock averages 2757 MHz in the YOLOX route
  and 985 MHz in the mask route (1,377 MHz with the hand pass), because in the mask route the GPU is idle
  between short bursts (31% utilisation) and clocks down. The same model call costs twice as much at the lower
  clock. It is a property of a lightly loaded GPU, not of the code path; it disappears when the GPU is kept busy,
  as the SOMA routes do (2745 MHz).

## 4. Model weights and licence

The GEM-X model weights are not in the repository and must not be: the repository is Apache-2.0 and
MIT and REUSE-clean, and the weights are under other terms.

What the published terms say (read from the Hugging Face model card of `nvidia/GEM-X`, the NVIDIA
Open Model License Agreement and the DINOv3 License Agreement):

- The `nvidia/GEM-X` repository holds both the 3D model and the 2D model used here
  (`vitpose.pth`, `onnx/vitpose.onnx` with `.data`). It is not gated. Its card says the source code is
  under Apache-2.0 and "use of the associated model is governed by the NVIDIA Open Model License
  Agreement". The `LICENSE` file in the repository is the Apache-2.0 text; the card is what
  distinguishes code from weights.
- The NVIDIA Open Model License allows commercial use and redistribution. Redistributing the model
  requires giving recipients a copy of the agreement and a "Notice" text file containing "Licensed by
  NVIDIA Corporation under the NVIDIA Open Model License". Rights terminate if a guardrail of the model
  is bypassed without a substitute, or if the licensee starts copyright or patent litigation over the
  model. Use must be consistent with NVIDIA's Trustworthy AI terms. Derivative models belong to their
  maker.
- The model card's use-case restrictions forbid processing video of individuals without legal
  authorisation and, where required, consent, synthetic impersonation, unauthorised surveillance, and
  uses violating privacy or biometric law.
- The 2D model is a DINOv3 ViT-H backbone with a heatmap head (GEM-X's `vitpose_extractor.py` builds it
  as `Dinov3_ViTPose_huge_metrosim_256x192` from the `facebookresearch/dinov3` code). The model card
  does not say whether the backbone was initialised from Meta's DINOv3 weights. NVIDIA's listing of
  the corresponding NIM names the DINOv3 licence among its terms. If it applies, the DINOv3 License
  allows use, modification and redistribution of the materials and derivative works only under that
  agreement, with a copy of it and "Built with DINOv3" displayed in the product's documentation or
  interface; it prohibits military, ITAR-controlled, nuclear, espionage and weapons uses, and Meta may
  change its terms.

What follows for posetrak:

- The weights are downloaded on first use from the Hugging Face repository into the user's model
  cache, the way RTMPose checkpoints already are, after the user has been shown both licence texts
  and accepted them. The accepted licence version is stored with the downloaded file.
  Implementation (`posetrak/detection/model_weights.py`): the model directory is `~/.posetrak/models`
  (`POSETRAK_MODEL_DIR`), the files are `gem-x/onnx/vitpose.onnx` and its `.data` file (about 3.4 GB), fetched with
  `huggingface_hub`. The two licence texts are bundled with Posetrak as package data
  (`posetrak/data/licences/`: `nvidia-open-model-license.txt`, version released October 24, 2025, and
  `dinov3-license.txt`, August 14, 2025), copied unchanged from the vendors' pages under a short header that
  names the licence, the source URL and the version; they are third-party texts in the REUSE data (vendor as
  copyright holder, their own `LicenseRef` file in `LICENSES/`). Showing and accepting them needs no network
  access; only the download does. The "version" of an accepted licence is the SHA-256 of the bundled text
  (line ends normalised to LF), stored in `licence-acceptance.json` beside the
  weights with the notice lines; a record whose hash differs from the text bundled in the running release counts as
  not accepted, so a release that ships a changed text asks again; the weights are not used
  without a valid record. `POSETRAK_SOMA77_MODEL` names an
  `onnx/vitpose.onnx` the user licensed on their own and bypasses download and record. The acceptance is asked
  in the detection dialogs and the segmentation panel (a dialog with a tab per licence and a progress dialog for
  the download), and by `detect run` (the texts in a pager and a confirmation, or `--accept-licences`).
- The installer does not bundle the weights; bundling would require the notice file and both
  agreements in the installer, and waits for the answer on the DINOv3 terms.
- The about box (Help, About PoseTrak) and the user documentation (`docs/user-guide/pose-models.md`) carry the
  NVIDIA notice line and "Built with DINOv3".
- SOMA-77 is the recommended model: the detection dialogs and `detect run` preselect it and offer
  the download with the licences. A user who declines the licences, or has no weights, gets
  RTMPose-133 (with ViTPose where configured), and the pipeline says why. Both stay selectable.
- Whether the DINOv3 terms apply to `vitpose.pth` is a question for NVIDIA; until answered they are
  assumed to apply.

### Open item: licence question to NVIDIA

Status: drafted, not sent. The answer is recorded here, with the date it was received and who gave it,
and decides whether the installer may bundle the weights (plan owner decision 1) and which notices the
about box carries.

> Subject: Licence terms of `vitpose.pth` / `onnx/vitpose.onnx` in `nvidia/GEM-X`
>
> We are an open-source motion-capture project (Apache-2.0) that would like to use the 2D pose model
> from the `nvidia/GEM-X` Hugging Face repository (`vitpose.pth`, and the ONNX export
> `onnx/vitpose.onnx` with its `.data` file) as an optional pose detector. The model card says that use
> of the model is governed by the NVIDIA Open Model License Agreement. GEM-X's `vitpose_extractor.py`
> builds the network as `Dinov3_ViTPose_huge_metrosim_256x192` from the `facebookresearch/dinov3`
> code, and the listing of the corresponding NIM names the DINOv3 License Agreement among its terms.
> We would like to ask:
>
> 1. Was the backbone of `vitpose.pth` initialised from Meta's DINOv3 weights? If it was, do the terms of
>    the DINOv3 License Agreement apply to `vitpose.pth` and to the ONNX export, in addition to the
>    NVIDIA Open Model License Agreement, for (a) use and (b) redistribution?
> 2. If both apply, which notices must accompany a copy of the weights that our software downloads
>    from `nvidia/GEM-X` onto the user's machine at the user's request (the NVIDIA notice line, "Built
>    with DINOv3", copies of both agreements), and where must they be shown (about box, documentation,
>    download dialog)?
> 3. May we include the weights in an installer or other package that we distribute, provided those
>    notices and agreements are included, or should users always download them from your repository?
> 4. Do any of the use restrictions of either agreement go beyond those in the model card's use-case
>    section (consent for processing video of individuals, unauthorised surveillance, biometric law)?
>    Our software is used for body motion capture of consenting performers in studio and sport settings;
>    it does not identify people.
> 5. Is a new release of the model planned that would change the licence terms or the file layout?
>
> Please reply to the project maintainer address. We will keep your answer in the repository's design
> documentation so that downstream users can rely on it.

Sources to attach: the model card of `nvidia/GEM-X`, the NVIDIA Open Model License Agreement and the
DINOv3 License Agreement at the versions read when the draft was written.

## 5. Acceptance

- One `detect run` produces a SOMA-77 run from YOLOX boxes, from masks and from an existing run,
  without a prior RTMPose pass.
- On the profiling clip, the SOMA route's mean GPU utilisation is at least 85%, or the profile shows
  which non-GPU stage bounds it and that stage is at its decode floor.
- Throughput of the SOMA route from masks is at least twice that of the prototype route
  (RTMPose source run plus `soma77_extract.py`) on the same clip.
- Keypoints of the native SOMA run equal those of the prototype on the same boxes (same options)
  to within 0.5 px, apart from the documented decode change of half a cell.

### Status of the acceptance criteria

- One `detect run` makes a SOMA-77 run from YOLOX boxes (default), from masks (`--segmentation`) and from an
  existing run (`--from-run`), each without a prior RTMPose pass: built, covered by unit tests with fake models, and
  run on the profiling clip (routes 1-3 of "Results of the unified loop", all exit 0).
- Throughput of at least twice the prototype route's: met, 2.9 times (506 s against 1,454 s).
- GPU utilisation of at least 85% on the profiling clip: not met, 79-82%. Inference is the bound, not a non-GPU
  stage at its decode floor; the findings after the results table say what would move it.
- Parity with the prototype: see the status document.
