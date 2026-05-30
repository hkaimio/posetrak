# Segmentation-Based Keypoint Noise Weighting — Integration Design

**Date:** 2026-05-30
**Status:** Proposal
**Prerequisites:** `per-frame-measurement-noise-design.md` (per-observation UKF noise vector)
**Background:** `seg-pose-experiment-notes.md`

## Problem

In capture sessions with close person-to-person contact (throws, ukemi, partner drills) the
pose estimator (RTMPose-133) sometimes attaches keypoints to the wrong body.  These mis-detections
are spatially close to the correct person's silhouette and therefore survive the UKF Mahalanobis
gate, causing slow drift or filter divergence in hand/foot keypoints.

## Approach

Run instance segmentation (SAM2) on each frame in parallel with RTMPose.  After pose estimation,
for each of the 133 keypoints check whether the detected pixel coordinate lies inside the SAM2
mask for that person.  Keypoints that fall outside the mask receive an inflated noise value in the
UKF observation noise matrix R.  This does **not** discard any observation — it softly
down-weights suspect keypoints so the UKF naturally ignores them if they contradict other cameras
or the motion prior.

Key principle: **pose estimation runs on the original, unmasked image**.  Pre-hoc masking was
tested and found to degrade accuracy even on easy frames (see
`seg-pose-experiment-notes.md`).  Segmentation is used only as a post-hoc quality signal.

## RTMPose note

The pipeline uses **RTMPose-133** (wholebody, via rtmlib) rather than YOLO-pose because hand
keypoints are required.  YOLO is used only for person detection (bounding boxes).  SAM2
initialisation uses the YOLO bounding box for the person, not a pose-model output.  This is
unaffected by the RTMPose choice.

---

## 1. Segmentation model: SAM2

**Why SAM2 over YOLO-seg / SAM3:**

- SAM2 has a temporal memory bank that propagates each person's segmentation identity across
  frames.  Once person A and person B are correctly separated in a clean frame, SAM2 can maintain
  that separation through brief occlusions.  YOLO-seg and SAM3 are stateless per-frame and fail
  on exactly the contact frames we care about.
- SAM3 adds open-vocabulary text prompting but does not have the temporal memory architecture.
- SAM2 is available in Ultralytics as `sam2.1_l.pt` (or smaller `sam2.1_b.pt`).

**Relationship to YOLO tracking**

YOLO remains the **primary person tracker** throughout the pipeline — nothing changes there.  The
stitcher assigns YOLO track IDs to persons and produces contiguous *tracking segments* (time
ranges).  SAM2's sole responsibility is providing a per-frame silhouette mask for keypoints that
YOLO has already detected and RTMPose has already processed.

**Initialisation and re-initialisation**

The natural unit of SAM2 initialisation is the **tracking segment** from the stitcher.  SAM2 is
initialised from the YOLO bounding box at the start of each segment and tracks forward until the
segment ends.  Gaps and re-entries are already managed by the stitcher, so a new segment always
triggers a SAM2 re-init from the fresh YOLO bbox — no extra gap-detection logic is needed.

Within a segment a periodic re-init every ~30–60 frames is recommended to prevent mask drift.
An additional trigger is mask degeneration: if the SAM2 mask area falls below e.g. 20 % of the
running-average area the tracker has probably lost the person, and a re-init from the current
YOLO bbox is warranted before the next few frames are processed.

For multi-person scenes, each person is tracked as a separate SAM2 "object" with its own memory.

---

## 2. Python pipeline changes

### 2a. New module: `python/pipeline/pose/segmentation.py`

Responsible for:

1. Loading SAM2 and running it on a video clip.
2. For a given frame and person, returning the binary mask (or soft probability mask) at full
   frame resolution.
3. Given a set of keypoints `(N, 2)` and a mask `(H, W)`, returning a per-keypoint
   `in_mask_score` vector of shape `(N,)` with values in `[0, 1]`.

Key functions:

```python
class SAM2Segmentor:
    """Tracks persons across a video clip using SAM2."""

    def __init__(self, model_name: str = "sam2.1_l.pt", device: str = "cuda"):
        ...

    def initialize(
        self,
        video_path: str,
        start_frame: int,
        persons: dict[str, np.ndarray],   # person_id -> bbox xyxy at start_frame
    ) -> None:
        """Prompt SAM2 with one bbox per person at start_frame."""

    def get_mask(self, frame_idx: int, person_id: str) -> np.ndarray:
        """(H, W) float32 probability mask for person at frame_idx."""

    def get_keypoint_scores(
        self,
        frame_idx: int,
        person_id: str,
        keypoints_xy: np.ndarray,   # (N, 2) pixel coords, original video resolution
        erosion_px: int = 5,
    ) -> np.ndarray:
        """Per-keypoint in-mask score (N,).  Erodes mask by erosion_px before query."""
```

The erosion step removes the silhouette boundary region where both segmentation uncertainty and
projection error peak simultaneously.  A keypoint on the exact boundary of a mask should not be
confidently labelled as "in" or "out".

### 2b. Changes to `poseanalysis.py` and `pose_extraction.py`

- `VideoData` (or a new `VideoDataWithSeg` subclass) stores the SAM2 segmentor alongside the
  RTMPose results.
- After RTMPose keypoints are extracted for person P at frame F in camera C, call
  `segmentor.get_keypoint_scores(F, P, kpts_xy)` to get `in_mask_scores` of shape `(133,)`.
- Store these scores alongside the existing keypoint data.

The segmentation step can run lazily (on demand when exporting) or eagerly during the RTMPose
pass.  Eager is simpler for the first prototype.

---

## 3. Database schema

### 3a. New table: `keypoint_obs_quality`

The table is intentionally **source-agnostic**: the `source` column distinguishes SAM2 mask
scores from optical-flow scores or any future signal.  Each source stores one float32 per
keypoint as a compact BLOB.

```sql
CREATE TABLE keypoint_obs_quality (
    sequence_id  TEXT    NOT NULL,
    camera_id    INTEGER NOT NULL,
    video_frame  INTEGER NOT NULL,
    person_id    TEXT    NOT NULL,
    source       TEXT    NOT NULL,   -- 'sam2', 'optical_flow', ...
    -- 133 little-endian float32 values in COCO-Wholebody order.
    -- Score in [0, 1]: 1.0 = high quality / in-mask.
    -- Sentinel -1.0 = data not available for this keypoint.
    scores_blob  BLOB    NOT NULL,
    PRIMARY KEY (sequence_id, camera_id, video_frame, person_id, source),
    FOREIGN KEY (sequence_id) REFERENCES pose_observation_sequences(id)
);
```

Scores are **real-valued float32**, not binary.  This matters at mask boundaries (SAM2 returns a
probability map, not a hard 0/1) and for optical flow (score encodes deviation magnitude, not
just a flag).  Both signals live on [0, 1] and drive the same inflation formula.

The -1.0 sentinel covers frames where a source has no data (SAM2 lost tracking, no previous
frame for optical flow, etc.).  The C++ reader treats -1.0 as "ignore this source for this
keypoint".

Each source can carry its own inflation factor in the tracker config (see §7), allowing them to
be tuned independently.  The C++ reader combines multiple sources per keypoint using `min()` by
default (most pessimistic quality signal wins).

### 3b. Relationship to `yolo_detections`

The `yolo_detections` table proposed in `per-frame-measurement-noise-design.md` stores per-frame
bounding boxes used to derive `bbox_scale_factor`.  `keypoint_obs_quality` is a parallel table at
the same (sequence_id, camera_id, video_frame) granularity.  Both are populated in the export step.

### 3c. Migration

```sql
CREATE TABLE IF NOT EXISTS keypoint_obs_quality (
    sequence_id  TEXT    NOT NULL,
    camera_id    INTEGER NOT NULL,
    video_frame  INTEGER NOT NULL,
    person_id    TEXT    NOT NULL,
    source       TEXT    NOT NULL,
    scores_blob  BLOB    NOT NULL,
    PRIMARY KEY (sequence_id, camera_id, video_frame, person_id, source),
    FOREIGN KEY (sequence_id) REFERENCES pose_observation_sequences(id)
);
```

Existing sessions without quality data have no rows — the C++ reader falls back gracefully
(see §5 below).

---

## 4. C++ `Observation` struct changes

Add an `obs_quality` field alongside the existing `confidence`:

```cpp
struct Observation {
    int camera_id;
    int marker_id;
    int frame_idx;
    double timestamp;

    Eigen::Vector2d position;
    Eigen::Vector2d position_distorted;
    double confidence;          ///< RTMPose keypoint confidence [0, 1]

    // NEW
    double obs_quality = 1.0;  ///< Combined quality score from keypoint_obs_quality table.
                                ///< [0, 1]: 1.0 = full confidence (no inflation).
                                ///< -1.0   = data not available (no inflation, backward-compat).
                                ///< Populated by session_reader from min() across all sources.

    /// @brief Effective measurement noise std for the UKF.
    ///
    /// Combines three independent factors:
    ///   1. base_noise:         intrinsic detector precision in detector-space pixels
    ///   2. confidence:         RTMPose keypoint confidence
    ///   3. obs_quality:        quality score from segmentation / optical flow
    ///
    /// @param base_noise        Base noise (detector-space px), ~5–10 for RTMPose-384
    /// @param quality_inflation Multiplier applied when obs_quality < quality_threshold
    /// @param quality_threshold Score below which inflation is applied (default 0.5)
    double measurement_noise_std(
        double base_noise = 5.0,
        double quality_inflation = 10.0,
        double quality_threshold = 0.5) const
    {
        double noise = base_noise / std::max(confidence, 0.1);
        if (obs_quality >= 0.0 && obs_quality < quality_threshold)
            noise *= quality_inflation;
        return noise;
    }
};
```

The -1.0 sentinel means "no quality data available" — the noise computation ignores it,
preserving exact backward compatibility.

---

## 5. C++ session reader changes (`session_reader.cpp`)

At load time, after populating keypoints from the existing observation tables, join
`keypoint_obs_quality` to fill `obs_quality`.  Multiple sources (e.g. `sam2` and
`optical_flow`) are combined using `min()` — the most pessimistic quality estimate wins.

The mapping from `marker_id` to the 0-based RTMPose-133 blob index reuses the existing
infrastructure that already handles this mapping during observation loading.  No new lookup
table is needed.

```cpp
// Pseudocode — adapt to actual query helper patterns

// Collect quality scores per source: frame -> marker_133_idx -> score
using ScoreMap = std::unordered_map<int, std::vector<float>>;
std::unordered_map<std::string, ScoreMap> source_scores;

auto q_stmt = db.prepare(
    "SELECT source, video_frame, scores_blob "
    "FROM keypoint_obs_quality "
    "WHERE sequence_id = ? AND camera_id = ? AND person_id = ?");
q_stmt.bind(sequence_id, camera_id, person_id);

while (q_stmt.step()) {
    std::string source = q_stmt.column_text(0);
    int frame          = q_stmt.column_int(1);
    auto scores        = decode_float_blob(q_stmt.column_blob(2), 133);
    source_scores[source][frame] = std::move(scores);
}

// Fill Observation::obs_quality as min() across all sources
for (auto& obs : observations) {
    int kp_idx = marker_id_to_rtmpose_133_idx(obs.marker_id);  // existing mapping
    if (kp_idx < 0) continue;

    float combined = 1.0f;
    for (auto const& [source, frame_map] : source_scores) {
        auto it = frame_map.find(obs.frame_idx);
        if (it != frame_map.end()) {
            float s = it->second[kp_idx];
            if (s >= 0.0f)                // skip sentinel
                combined = std::min(combined, s);
        }
    }
    obs.obs_quality = combined;
}
```

When `keypoint_obs_quality` has no rows the map is empty and `obs_quality` stays 1.0 for all
observations — full backward compatibility.

---

## 6. UKF integration

This change is an extension of `per-frame-measurement-noise-design.md`.  Both are needed for the
full benefit; however, they can be implemented independently:

**Phase A (this design):** Add `obs_quality` to `Observation`; update
`measurement_noise_std()` to use it.  No change to the UKF itself — the existing scalar noise
path already calls `obs.measurement_noise_std(base_noise)` per observation.

**Phase B (per-frame-noise design):** Change `UnscentedKalmanFilter::update()` to accept a
per-observation noise vector instead of a single scalar.  At that point
`Tracker::track_frame()` builds the vector by calling `obs.measurement_noise_std(...)` for each
active observation, and both `bbox_scale_factor` and `obs_quality` contribute automatically.

Phases A and B are independent and can be shipped in either order.

### Inflation factor tuning

Recommended starting values and their rationale:

| Parameter           | Default | Notes |
|---------------------|---------|-------|
| `quality_inflation` | 10×     | Inflated variance 100×: only a near-perfect prediction can accept this obs |
| `quality_threshold` | 0.5     | Score must be ≥ 0.5 to avoid inflation |
| `erosion_px`        | 5 px    | Remove silhouette boundary pixels before SAM2 mask query |

A Mahalanobis rejection threshold of 4.0 rejects observations > 4 sigma.  With 10× inflation
a keypoint must be within 40 sigma (in un-inflated units) to pass — effectively ignored unless
the state prediction already points to that exact pixel.  This is the correct behaviour for an
out-of-mask keypoint.

---

## 7. Tracker config parameter

Add an optional `[tracking.obs_quality]` section to the TOML config:

```toml
[tracking.obs_quality]
quality_inflation  = 10.0   # noise multiplier for low-quality keypoints
quality_threshold  = 0.5    # score below which inflation applies
```

When this section is absent the defaults in `Observation::measurement_noise_std()` apply.
When no quality data is present in the DB, `obs_quality = 1.0` for all observations and the
section has no effect.

---

## 8. Fallback: optical flow as a lighter alternative

Before implementing SAM2 (which requires GPU time and DB storage), a simpler proxy can be used:

1. Compute Lucas-Kanade optical flow for each keypoint between frames N-1 and N.
2. Predicted position at N = detected position at N-1 + flow vector.
3. Compute a quality score: `score = exp(-d / d_max)` where `d` is the distance between the
   detected position and the flow-predicted position.

This catches the "neighbouring body part drifts in" failure because the intruding limb has a
different optical flow trajectory from the tracked person's limbs.  It does not require any
segmentation model.  Implementation complexity: ~100 lines in Python, no C++ changes beyond
reading the `obs_quality` field.

Being source-agnostic, `keypoint_obs_quality` stores optical-flow scores under
`source = 'optical_flow'` alongside SAM2 scores.  The C++ reader combines them with `min()`.

---

## 9. Implementation order

1. **`segmentation.py`**: SAM2 wrapper; test on `test.mp4` to verify mask quality on contact frames.
2. **DB migration**: create `keypoint_obs_quality` table.
3. **`pose_extraction.py`**: populate `keypoint_obs_quality` with `source='sam2'` during export.
4. **`Observation` struct**: add `obs_quality` field; update `measurement_noise_std()`.
5. **`session_reader.cpp`**: join `keypoint_obs_quality` and fill `obs_quality` at load time.
6. **Tracker config**: expose `quality_inflation` and `quality_threshold` parameters.
7. **Validation**: re-run on the ukemi session; compare `tracking_stats.csv` with and without
   quality weighting on the contact frames.

Steps 1–3 (Python-only) can be prototyped without touching the C++ codebase.
Steps 4–6 require the per-observation noise vector from `per-frame-measurement-noise-design.md`
for full effect, but the Phase A path (scalar noise modified per observation) works immediately.
