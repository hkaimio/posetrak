# Independent assessment of the SOMA-77 pose-model spike

This was a successful feasibility and diagnostic spike, but it does not yet establish GEM-X/SOMA-77 as a better replacement for ViTPose-L. Its strongest result is evidence that Posetrak's remaining failures involve the relationship between detections, skeleton geometry, camera observations and temporal inference. Changing the detector alone is unlikely to resolve them.

This assessment is based on the [spike report](soma77-spike-report.md), [original plan](soma77-pose-spike-plan.md), supporting notes, and selected labelling and evaluation scripts. The experiments were not rerun, and the rendered videos were not independently reviewed. Supporting notes and data scripts referenced below are local artifacts outside the project repository.

## 1. Keep SOMA available, but qualify the accuracy conclusion

The integration, reproducible model failures, tracker bug fixes and candidate-inspection tools are concrete successes. Keeping several peaks also demonstrably helps in some difficult windows.

However, "about as accurate as ViTPose-L" understates the rolling-fall results: SOMA has higher reported errors for **every listed body group in both time windows** against the dots-only reference. The reference is imperfect, so this does not prove general inferiority—but it provides no quantitative case for replacement. Conversely, the visible improvement in torso and head motion deserves investigation rather than dismissal as subjective. See [report §5.2](soma77-spike-report.md#52-accuracy-against-vitpose-l-single-person-reference-capture).

These may be different objectives:

- Joint-position accuracy measures where the skeleton is.
- Segment orientation, torso articulation and head direction measure how convincingly it moves.
- Correction time measures whether the output is useful.

SOMA could improve the latter two while worsening some position errors. Measure those benefits explicitly through segment-angle errors, contact preservation and blinded retargeted comparisons.

A **hybrid experiment—SOMA trunk/head with ViTPose limbs—is more justified than wholesale replacement**, provided their landmark definitions and uncertainty are reconciled.

## 2. Candidate coverage is less independent than it appears

This is the most consequential evaluation issue identified in this review.

The labeller largely constructs reference positions by accepting or selecting model-generated tracklets, including candidates from another person's crop. Its export retains those candidate coordinates. That is useful human-reviewed association data, but it is not independent localization ground truth. See `base_resolved()` and `export()` in [labeler.py](../../../../python/tools/prototypes/kp_labeling/labeler.py).

Consequently:

- "A candidate exists near 99% of labels" is partly built into how labels were created.
- Comparisons favour the extraction on which the labels were selected.
- Human review can establish whose joint a candidate represents without establishing that its position is accurate.
- More than 100,000 keypoint-frame samples still represent only a few people and motion episodes; they are highly correlated.

Preserve this dataset as an **association benchmark**, then add a small independent localization benchmark: manually placed points with candidates hidden, an explicit "visible but no correct candidate" category, and duplicate annotation on a subset.

Also test whether the 3%-of-box-height threshold behaves fairly during rolls. A changing silhouette height changes the tolerance even if the underlying pixel error stays constant.

## 3. Diagnose the label-fed tracker's residual before expanding association

Even with reviewed inputs, the tracker misses the labels by roughly 5% of box height in one evaluation; elsewhere only about a quarter of projections fall within the 3% acceptance radius. The system cannot adequately explain inputs it is treating as its best available reference. See [report §§5.4–5.5](soma77-spike-report.md#54-tracker-level).

Several explanations remain entangled:

- Camera calibration or timing errors.
- Subject proportions and transferred marker offsets.
- Pose-dependent landmark definitions.
- Filter dynamics, observation weights and rejection.
- Inconsistency or incompleteness in the reference itself.

The label-fed result is therefore a **controlled comparison run, not a 3D oracle**. Closeness to it measures sensitivity to input changes, not necessarily accuracy.

The next experiment should separate these causes:

1. Triangulate independently labelled, visible joints without a skeleton.
2. Fit the skeleton independently per frame, without temporal filtering.
3. Run the existing temporal tracker on the same observations.

Poor triangulation implicates cameras, timing or labels. Good triangulation but poor skeleton fitting implicates geometry and landmark mapping. Good independent fits but poor tracking implicate dynamics, weighting or rejection.

That experiment would identify which subsystem deserves the next substantial investment.

## 4. Move calibration and synchronization up the roadmap

The supporting notes record agreement tolerances expanded from 12 pixels to as much as 60 pixels for problematic cameras. This is a recorded observation, not proof of its cause, but it is too large to treat merely as background noise. Source: `C:/Users/HarriKaimio/agent-memory/memory/projects/posetrak-candidate-keypoints-test-plan.md`, observation dated 2026-10-06.

Fast movements make temporal alignment especially relevant. Illustratively, a point moving at 5 m/s travels 5 cm in 10 ms. A small timing error can therefore resemble exactly the detector or tracker failure seen during throws. Joint treatment of camera timing and moving-point reconstruction is an established research direction. See [Spatiotemporal Bundle Adjustment for Dynamic 3D Reconstruction](https://openaccess.thecvf.com/content_cvpr_2016/papers/Vo_Spatiotemporal_Bundle_Adjustment_CVPR_2016_paper.pdf).

Run camera-subset and small time-offset sweeps, then examine residuals against image position and motion speed.

The statement that performance is limited by the worst camera also deserves challenge. A robust system should be able to exclude or down-weight a bad camera. **A useful acceptance test is that adding a poor camera should not materially worsen the result.**

## 5. Visibility matters, but invisibility is not equivalent to error

An occluded joint still has a position. A detector may infer it accurately from context, or confidently invent the wrong position. Counting every invisible pick as false conflates these cases. This is particularly awkward for internal joint centres, which are never directly visible in the same sense as a surface marker.

The useful distinction is between:

- Correct-person localization with calibrated uncertainty.
- Wrong-person association.
- Unsupported prediction.
- No useful measurement.

The detailed ablations support a more nuanced conclusion than the report: manually removing invisible picks improved agreement with the label-fed run, but correcting visible positions also helped; different windows needed different interventions. Source: `C:/Users/HarriKaimio/agent-memory/memory/projects/posetrak-candidate-keypoints-test-plan.md`, observation dated 2026-10-10 under "Where the proposal's errors hurt."

Geometry-based visibility is worth testing, but it is not established as necessary or sufficient. If the body pose is already wrong, visibility inferred from it can suppress the evidence needed for recovery.

Train or calibrate the decision around **whether an observation helps reconstruction**, rather than visibility alone.

## 6. Audit correlated evidence

Several SOMA trunk points may reflect one shared learned body-shape assumption. Several cameras can reproduce that assumption. Position, velocity and parent-relative observations may also derive from the same underlying detections.

If the tracker treats these as independent measurements, it can become overconfident without receiving equivalent independent information. The report does not establish whether this happens; it warrants an audit.

Cheap tests include:

- Sparse, well-separated landmarks versus all 77.
- Position-only observations versus added derived measurements.
- Capped information per body region.
- Residual correlation across joints, cameras and time.

This could explain why adding more anatomically attractive landmarks changes appearance without consistently improving accuracy.

## 7. Preserve alternative hypotheses through temporal inference

Multi-hypothesis association is promising, but putting candidates inside a UKF is not sufficient by itself.

Two plausible wrists belonging to different people constitute distinct explanations. Averaging them can produce a wrist belonging to nobody. The design must preserve alternative identity/pose hypotheses long enough for later evidence to distinguish them.

Likewise, two genuine wrists can project almost onto the same pixel during a grab. Exclusivity should concern ownership of an observation hypothesis, not a blanket prohibition on nearby projected joints.

First prototype **short-window optimization over pose and association**, with:

- Several discrete association alternatives.
- An explicit "none of these candidates" option.
- Robust observation costs.
- Frames before and after the failure.
- Soft contact constraints where supported.

Hybrid discrete/continuous association is a concrete existing formulation; it need not be reduced immediately to one Gaussian update. See [GTSAM's hybrid data-association example](https://borglab.github.io/gtsam/hybriddataassociationtutorial/).

A second worthwhile alternative is retaining heatmap evidence until multi-view fusion. Extracting maxima discards distribution shape and weaker spatial support before the cameras can resolve ambiguity. Volumetric multi-view fusion provides an established example of this approach. See [Learnable Triangulation of Human Pose](https://arxiv.org/abs/1905.05754).

## 8. Put motion generation after reconstruction and recovery experiments

The problematic intervals are short and surrounded by usable observations. First test backward tracking, local reinitialization, short-window fitting and simple changes to the motion model.

A learned motion prior could produce a convincing throw that differs from the recorded throw. That may be acceptable for animation repair, but it must be evaluated separately from faithful capture. Partner contact makes this distinction especially important.

There is also an information limit: if a movement is genuinely hidden from all useful views, no association method can recover it uniquely. Targeted user constraints may then be more valuable than another automatic heuristic.

## Recommended experiment order

| Priority | Next experiment | Decision it enables |
|---|---|---|
| 1 | Freeze inputs, labels, configurations and event definitions; add a small independent annotation set | Establish which gains are real |
| 2 | Triangulation → per-frame skeleton fit → temporal tracker, including camera/timing ablations | Locate the dominant error source |
| 3 | Subject-specific mapping, sparse-landmark and correlated-observation tests | Determine whether the tracker can use SOMA effectively |
| 4 | Hybrid detector experiment and short-window association/recovery | Test complementary evidence and temporal recovery |
| 5 | Failure triage and targeted correction | Measure reduction in operator effort |
| 6 | Learned gap filling | Address genuinely underconstrained intervals |

For triage, the existing 12-of-15 event result is encouraging but incomplete: `D:/mocap/vanhaa-spike/event_scores.py` measures maximum disagreement inside known failure windows. It does not establish false-alarm rates across an entire capture. Evaluate failures caught, false alerts per minute, time spent reviewing, and total correction time on a previously unseen clip.

**Retain SOMA as an experimental option, prioritize measurement consistency and short-window recovery, and postpone a production detector replacement.** The next milestone should demonstrate fewer failed motion episodes and less correction work on an independent capture—not merely a higher percentage of correctly selected keypoints.
