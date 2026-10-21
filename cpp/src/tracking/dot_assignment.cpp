// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/dot_assignment.hpp"

#include "posetrak/tracking/assignment.hpp"
#include <algorithm>
#include <cmath>
#include <utility>

namespace posetrak {

namespace {
// Cost of an excluded pair: far above any gate.
constexpr double kExcludedCost = 1e9;

// Shared by TrackletContinuityModifier (which turns this into a cost discount)
// and ReacquisitionGateModifier (which treats it as independent identity
// evidence regardless of the tracklet multiplier): does *candidate* continue
// the tracklet that resolved into *slot* on the previous frame?
bool tracklet_continues(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                        PrevDotTrackletIds const& prev_tracklet_ids) {
    if (candidate.tracklet_id < 0)
        return false;
    auto const subject_it = prev_tracklet_ids.find(slot.subject_id);
    if (subject_it == prev_tracklet_ids.end())
        return false;
    auto const camera_it = subject_it->second.find(slot.camera_id);
    if (camera_it == subject_it->second.end())
        return false;
    auto const marker_it = camera_it->second.find(slot.marker_id);
    return marker_it != camera_it->second.end() && marker_it->second == candidate.tracklet_id;
}

// Distance from point b to point a's epipolar line under fundamental matrix F
// (F such that F * [a;1] is a's epipolar line in b's camera). Large when F
// (or a, or b) is degenerate (a zero line direction) -- callers only use this
// where F comes from real, distinct cameras, so that is not guarded against.
double epipolar_distance(Eigen::Matrix3d const& F, Eigen::Vector2d const& a,
                         Eigen::Vector2d const& b) {
    Eigen::Vector3d const line = F * Eigen::Vector3d(a.x(), a.y(), 1.0);
    Eigen::Vector3d const point(b.x(), b.y(), 1.0);
    return std::abs(line.dot(point)) / std::hypot(line.x(), line.y());
}

// Shared by ReacquisitionGateModifier (independent evidence for a gapped slot)
// and CrossViewCorroborationModifier (the cost discount itself): is *candidate*,
// seen in *slot.camera_id*, corroborated by some other camera's near-prediction
// candidate for the same slot (DotAssignmentContext::slot_evidence)?
bool corroborated(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                  DotAssignmentContext const& context) {
    if (context.dot_cross_view_corroboration_px <= 0.0 || context.camera_fundamentals.empty())
        return false;
    auto const evidence_subject = context.slot_evidence.find(slot.subject_id);
    if (evidence_subject == context.slot_evidence.end())
        return false;
    auto const evidence_marker = evidence_subject->second.find(slot.marker_id);
    if (evidence_marker == evidence_subject->second.end())
        return false;
    for (auto const& [other_camera_id, positions] : evidence_marker->second) {
        if (other_camera_id == slot.camera_id)
            continue;  // corroboration needs an *other* camera's own candidate
        auto const f_it = context.camera_fundamentals.find(
            (std::int64_t{slot.camera_id} << 32) | static_cast<std::uint32_t>(other_camera_id));
        if (f_it == context.camera_fundamentals.end())
            continue;  // no calibrated pair for these two cameras
        for (auto const& other_pos : positions) {
            if (epipolar_distance(f_it->second, candidate.position, other_pos) <=
                context.dot_cross_view_corroboration_px)
                return true;
        }
    }
    return false;
}
}  // namespace

CostSupport KeypointClassModifier::apply(DotSlotRef const& slot,
                                         UnlabeledCandidate const& candidate,
                                         DotAssignmentContext const& context) const {
    int slot_keypoint = -1;
    if (auto const subject_it = context.subject_marker_keypoint.find(slot.subject_id);
        subject_it != context.subject_marker_keypoint.end()) {
        if (auto const marker_it = subject_it->second.find(slot.marker_id);
            marker_it != subject_it->second.end())
            slot_keypoint = marker_it->second;
    }
    return {1.0, candidate.keypoint_idx != slot_keypoint};
}

CostSupport TrackletContinuityModifier::apply(DotSlotRef const& slot,
                                              UnlabeledCandidate const& candidate,
                                              DotAssignmentContext const& context) const {
    if (!tracklet_continues(slot, candidate, context.prev_tracklet_ids))
        return {};
    return {candidate.keypoint_idx >= 0 ? context.kp_candidate_tracklet_multiplier
                                        : context.dot_tracklet_gate_multiplier,
            false};
}

CostSupport CandidateOwnershipModifier::apply(DotSlotRef const& slot,
                                              UnlabeledCandidate const& candidate,
                                              DotAssignmentContext const&) const {
    // subject_mask has 64 bits; a subject id beyond them owns nothing it can be denied.
    bool const owned = slot.subject_id >= 64 || ((candidate.subject_mask >> slot.subject_id) & 1U);
    return {1.0, !owned};
}

CostSupport TrackletAssignmentModifier::apply(DotSlotRef const& slot,
                                              UnlabeledCandidate const& candidate,
                                              DotAssignmentContext const& context) const {
    if (context.tracklet_assignments == nullptr || candidate.tracklet_id < 0)
        return {};
    auto const verdict = context.tracklet_assignments->verdict(
        candidate.camera_id, candidate.tracklet_id, candidate.frame_idx);
    switch (verdict.kind) {
        case TrackletVerdict::Kind::EXCLUDED:
            return {1.0, true};
        case TrackletVerdict::Kind::ASSIGNED:
            if (verdict.assigned_marker == slot.marker_id)
                return {context.dot_assigned_factor, false};
            return {1.0, true};
        case TrackletVerdict::Kind::NONE:
            break;
    }
    for (int const forbidden : verdict.forbidden_markers) {
        if (forbidden == slot.marker_id)
            return {1.0, true};
    }
    if (verdict.suggested_marker == slot.marker_id)
        return {context.dot_suggested_factor, false};
    return {};
}

CostSupport ReacquisitionGateModifier::apply(DotSlotRef const& slot,
                                             UnlabeledCandidate const& candidate,
                                             DotAssignmentContext const& context) const {
    auto const subject_it = context.prev_resolved_frame.find(slot.subject_id);
    if (subject_it == context.prev_resolved_frame.end())
        return {};  // never resolved -- nothing established to protect
    auto const camera_it = subject_it->second.find(slot.camera_id);
    if (camera_it == subject_it->second.end())
        return {};
    auto const marker_it = camera_it->second.find(slot.marker_id);
    if (marker_it == camera_it->second.end())
        return {};
    int const gap = context.frame_idx - marker_it->second;
    if (gap < context.dot_reacquire_gap_frames)
        return {};  // still within the coasting window, not a reacquisition

    if (tracklet_continues(slot, candidate, context.prev_tracklet_ids))
        return {};
    if (corroborated(slot, candidate, context))
        return {};
    auto const evidence_subject = context.slot_evidence.find(slot.subject_id);
    if (evidence_subject != context.slot_evidence.end()) {
        auto const evidence_marker = evidence_subject->second.find(slot.marker_id);
        if (evidence_marker != evidence_subject->second.end() &&
            evidence_marker->second.size() >= 2)
            return {};
    }
    return {1.0, true};
}

CostSupport CrossViewCorroborationModifier::apply(DotSlotRef const& slot,
                                                  UnlabeledCandidate const& candidate,
                                                  DotAssignmentContext const& context) const {
    if (!corroborated(slot, candidate, context))
        return {};
    return {context.dot_corroboration_gate_multiplier, false};
}

namespace {
// Shared by SegmentationMaskModifier::apply() and the winning-pair accumulator
// update below: this candidate's running in-mask fraction for *slot*'s
// subject, or 1.0 (fully trusted) if there's no history yet -- see
// SegmentationMaskModifier's own doc comment for why "no history" must not
// read as "untrusted".
double tracklet_mask_fraction(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                              DotAssignmentContext const& context) {
    if (candidate.tracklet_id < 0 || context.tracklet_mask_state == nullptr)
        return 1.0;
    auto const subject_it = context.tracklet_mask_state->find(slot.subject_id);
    if (subject_it == context.tracklet_mask_state->end())
        return 1.0;
    auto const camera_it = subject_it->second.find(slot.camera_id);
    if (camera_it == subject_it->second.end())
        return 1.0;
    auto const tracklet_it = camera_it->second.find(candidate.tracklet_id);
    if (tracklet_it == camera_it->second.end())
        return 1.0;
    // A handful of samples must back the fraction before it can exclude or
    // penalise anything: with one sample it would be 0 or 1, letting a single
    // bad frame (mask lag, blur) condemn an established tracklet.
    constexpr int kMinTrackletSamples = 5;
    auto const frac = tracklet_it->second.fraction(kMinTrackletSamples);
    return frac.value_or(1.0);
}
}  // namespace

CostSupport SegmentationMaskModifier::apply(DotSlotRef const& slot,
                                            UnlabeledCandidate const& candidate,
                                            DotAssignmentContext const& context) const {
    auto const subject_it = context.seg_mask_subject.find(slot.subject_id);
    if (subject_it == context.seg_mask_subject.end() || subject_it->second.mask_label <= 0)
        return {};

    double const fraction = tracklet_mask_fraction(slot, candidate, context);

    auto const mask_it = context.seg_masks_by_camera.find(slot.camera_id);
    if (mask_it != context.seg_masks_by_camera.end()) {
        // Far from the subject in this frame: not its dot, whatever the tracklet's
        // history -- a young tracklet (a flickering background light gets a new id
        // every few frames) has none to judge it by. Only when the subject is in
        // the mask at all; a frame where segmentation lost it says nothing.
        if (context.dot_seg_mask_hard_distance_px > 0.0 &&
            mask_it->second.has_label(subject_it->second.mask_label) &&
            !mask_label_present_near(mask_it->second, candidate.position_distorted,
                                     subject_it->second.mask_label,
                                     context.dot_seg_mask_hard_distance_px))
            return {1.0, true};
        bool const in_mask_now =
            mask_label_present_near(mask_it->second, candidate.position_distorted,
                                    subject_it->second.mask_label, context.dot_seg_mask_dilate_px);
        if (!in_mask_now && fraction < context.dot_seg_mask_min_tracklet_fraction)
            return {1.0, true};
    }

    double const scale = context.dot_seg_mask_cost_scale_at_zero_fraction;
    double const factor = 1.0 / (1.0 + (scale - 1.0) * (1.0 - fraction));
    return {factor, false};
}

std::vector<std::unique_ptr<CostModifier>>
default_cost_modifiers(DotAssignmentContext const& context) {
    std::vector<std::unique_ptr<CostModifier>> modifiers;
    if (!context.subject_marker_keypoint.empty())
        modifiers.push_back(std::make_unique<KeypointClassModifier>());
    if ((context.dot_tracklet_gate_multiplier > 1.0 ||
         (context.kp_candidate_tracklet_multiplier > 1.0 &&
          !context.subject_marker_keypoint.empty())) &&
        !context.prev_tracklet_ids.empty())
        modifiers.push_back(std::make_unique<TrackletContinuityModifier>());
    modifiers.push_back(std::make_unique<CandidateOwnershipModifier>());
    if (context.dot_cross_view_corroboration_px > 0.0 &&
        context.dot_corroboration_gate_multiplier > 1.0 && !context.camera_fundamentals.empty())
        modifiers.push_back(std::make_unique<CrossViewCorroborationModifier>());
    if (context.dot_reacquire_gap_frames > 0)
        modifiers.push_back(std::make_unique<ReacquisitionGateModifier>());
    if (context.dot_seg_mask_enabled && !context.seg_mask_subject.empty() &&
        context.tracklet_mask_state != nullptr)
        modifiers.push_back(std::make_unique<SegmentationMaskModifier>());
    if (context.tracklet_assignments != nullptr && !context.tracklet_assignments->empty())
        modifiers.push_back(std::make_unique<TrackletAssignmentModifier>());
    return modifiers;
}

namespace {
// Shared evidence pre-pass for ReacquisitionGateModifier and
// CrossViewCorroborationModifier (design plan §2.2/§2.3: "build it once per
// step in a pre-pass", not once per mechanism): for every (subject, marker)
// slot, the candidate positions found within *max_px* of each camera's own
// prediction for the slot. A plain pixel radius, not the assignment gate's
// Mahalanobis metric -- this is corroboration evidence, not a substitute for
// the gate. Only built when at least one of the two mechanisms is on, since
// it costs an extra pass over every (camera, slot, candidate) triple that the
// rest of resolve_dot_assignment() does not otherwise need.
SlotEvidence build_slot_evidence(
    std::vector<SubjectDotPredictions> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    double max_px) {
    SlotEvidence evidence;
    for (auto const& [camera_id, candidates] : candidates_by_camera) {
        for (auto const& subject : subjects) {
            auto const cam_it = subject.predictions_by_camera.find(camera_id);
            if (cam_it == subject.predictions_by_camera.end())
                continue;
            for (auto const& [marker_id, prediction] : cam_it->second) {
                std::vector<Eigen::Vector2d> near;
                for (auto const& candidate : candidates) {
                    if ((candidate.position - prediction.position).norm() <= max_px)
                        near.push_back(candidate.position);
                }
                if (!near.empty())
                    evidence[subject.subject_id][marker_id][camera_id] = std::move(near);
            }
        }
    }
    return evidence;
}
}  // namespace

CameraPairFundamental build_camera_fundamentals(std::unordered_map<int, Camera> const& cameras) {
    CameraPairFundamental result;
    for (auto const& [id_a, cam_a] : cameras) {
        Eigen::Matrix3d const Ra = cam_a.orientation().toRotationMatrix();
        Eigen::Vector3d const ta = -Ra * cam_a.position();
        Eigen::Matrix3d Ka = Eigen::Matrix3d::Identity();
        Ka(0, 0) = cam_a.intrinsics().fx;
        Ka(1, 1) = cam_a.intrinsics().fy;
        Ka(0, 2) = cam_a.intrinsics().cx;
        Ka(1, 2) = cam_a.intrinsics().cy;
        for (auto const& [id_b, cam_b] : cameras) {
            if (id_a == id_b)
                continue;
            Eigen::Matrix3d const Rb = cam_b.orientation().toRotationMatrix();
            Eigen::Vector3d const tb = -Rb * cam_b.position();
            Eigen::Matrix3d Kb = Eigen::Matrix3d::Identity();
            Kb(0, 0) = cam_b.intrinsics().fx;
            Kb(1, 1) = cam_b.intrinsics().fy;
            Kb(0, 2) = cam_b.intrinsics().cx;
            Kb(1, 2) = cam_b.intrinsics().cy;

            // Relative pose of B with respect to A, then the essential and
            // fundamental matrices such that F * [x_a; 1] is x_a's epipolar
            // line in B's undistorted pixels (Hartley & Zisserman §9.6.1).
            Eigen::Matrix3d const R_rel = Rb * Ra.transpose();
            Eigen::Vector3d const t_rel = tb - R_rel * ta;
            Eigen::Matrix3d t_cross;
            t_cross << 0.0, -t_rel.z(), t_rel.y(), t_rel.z(), 0.0, -t_rel.x(), -t_rel.y(),
                t_rel.x(), 0.0;
            Eigen::Matrix3d const E = t_cross * R_rel;
            Eigen::Matrix3d const F = Kb.inverse().transpose() * E * Ka.inverse();

            result[(std::int64_t{id_a} << 32) | static_cast<std::uint32_t>(id_b)] = F;
        }
    }
    return result;
}

std::unordered_map<int, SubjectDotAssignment> resolve_dot_assignment(
    std::vector<SubjectDotPredictions> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    DotAssignmentContext const& context, AssignmentTrace* trace) {
    std::unordered_map<int, SubjectDotAssignment> result;
    bool const tracing = trace != nullptr && trace->wants(context.frame_idx);

    double const gate_mahalanobis = context.gate_mahalanobis;
    int const frame_idx = context.frame_idx;
    double const timestamp = context.timestamp;
    double const calib_noise_std = context.calib_noise_std;
    StreakVelocityConfig const& streak_config = context.streak_config;
    PrevDotPositions const& prev_positions = context.prev_positions;

    // Only touched when streak_config.enabled -- a throwaway local is fine when the
    // caller doesn't want k to persist across calls.
    std::unordered_map<int, StreakKAccumulator> local_streak_k_state;
    std::unordered_map<int, StreakKAccumulator>& streak_k =
        context.streak_k_state ? *context.streak_k_state : local_streak_k_state;

    auto const modifiers = default_cost_modifiers(context);

    // The reacquisition gate's and cross-view corroboration's shared evidence
    // needs every camera's candidates already gathered, so it is computed once
    // here rather than by the caller, and folded into a local copy of the
    // context the modifiers below read instead of *context* itself.
    DotAssignmentContext effective_context = context;
    if (context.dot_reacquire_gap_frames > 0 || context.dot_cross_view_corroboration_px > 0.0) {
        effective_context.slot_evidence =
            build_slot_evidence(subjects, candidates_by_camera, context.dot_reacquire_max_px);
    }

    // One column per (subject, marker) slot with a prediction for this camera.
    struct Column {
        int subject_id;
        int marker_id;
        MarkerPrediction const* prediction;
    };

    for (auto const& [camera_id, candidates] : candidates_by_camera) {
        if (candidates.empty())
            continue;

        std::vector<Column> columns;
        for (auto const& subject : subjects) {
            auto cam_it = subject.predictions_by_camera.find(camera_id);
            if (cam_it == subject.predictions_by_camera.end())
                continue;
            for (auto const& [marker_id, prediction] : cam_it->second) {
                columns.push_back(Column{subject.subject_id, marker_id, &prediction});
            }
        }
        if (columns.empty())
            continue;

        int const n_rows = static_cast<int>(candidates.size());
        int const n_cols = static_cast<int>(columns.size());
        std::vector<double> cost(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols));
        // Likelihood terms of keypoint candidates (ln det S and the peak), added to the cost of the
        // pairs that pass the gate after the matrix is complete; zero for a dot.
        std::vector<double> extra(cost.size(), 0.0);
        double min_extra = 0.0;
        double max_extra = 0.0;
        size_t const trace_start = tracing ? trace->entries.size() : 0;
        for (int r = 0; r < n_rows; ++r) {
            UnlabeledCandidate const& row_cand = candidates[static_cast<size_t>(r)];
            Eigen::Vector2d const cand_pos = row_cand.position;
            for (int c = 0; c < n_cols; ++c) {
                Column const& col = columns[static_cast<size_t>(c)];
                MarkerPrediction const& pred = *col.prediction;
                Eigen::Vector2d const diff = cand_pos - pred.position;
                Eigen::Matrix2d covariance = pred.covariance;
                double nll_extra = 0.0;
                if (row_cand.keypoint_idx >= 0) {
                    // A keypoint's detection error is not part of the state's uncertainty: add it
                    // the way the filter's innovation covariance does for a fixed observation.
                    double const noise =
                        (context.pose_noise_std * row_cand.crop_scale + context.calib_noise_std) /
                        std::max(row_cand.confidence, 0.1);
                    covariance += Eigen::Matrix2d::Identity() * (noise * noise);
                    // The negative log-likelihood of the detection: a wider covariance (a weak
                    // maximum) pays for its width, a higher peak is rewarded.
                    nll_extra = std::log(covariance.determinant()) -
                                context.kp_candidate_peak_weight *
                                    std::log(std::max(row_cand.confidence, 0.01));
                }
                double const raw_mahal_sq = diff.transpose() * covariance.inverse() * diff;
                double mahal_sq = raw_mahal_sq;
                char const* excluded_by = "";

                DotSlotRef const slot{col.subject_id, camera_id, col.marker_id, &pred};
                for (auto const& modifier : modifiers) {
                    CostSupport const support = modifier->apply(slot, row_cand, effective_context);
                    if (support.exclude) {
                        mahal_sq = kExcludedCost;
                        excluded_by = modifier->name();
                        break;
                    }
                    if (support.factor != 1.0)
                        mahal_sq /= support.factor;
                }

                if (tracing) {
                    trace->entries.push_back(AssignmentTraceEntry{
                        camera_id,
                        r,
                        row_cand.tracklet_id,
                        cand_pos.x(),
                        cand_pos.y(),
                        col.subject_id,
                        col.marker_id,
                        raw_mahal_sq,
                        mahal_sq,
                        excluded_by,
                        false,
                    });
                }

                // Flatten every hopeless (but not modifier-excluded) pairing to one shared
                // cost before it reaches the solver. Without this, solve_assignment()'s
                // padded-square formulation forces every real column (marker slot) to take
                // *some* real row (candidate) -- there is no dummy-row equivalent of a
                // dummy column, so a marker with genuinely no valid candidate this frame
                // (out of view, occluded by the prop's own geometry) still has to be paired
                // with whichever real candidate is least-bad for it. Minimising total cost
                // over the whole matrix then lets the solver trade a *different*, perfectly
                // good (gate-passing) pairing away in exchange for shaving a few points off
                // that already-doomed pairing -- pure waste, since a pairing above the gate
                // is dropped by the caller's own filter regardless of its exact cost, but it
                // can steal the row a nearby marker's real, correct match needed (confirmed
                // on real data: two markers with candidates matching to within 1e-4 lost
                // their own slots this way to a third marker that had no valid candidate on
                // that camera at all). Clamping removes the incentive: every hopeless pairing
                // now costs the same regardless of which specific row it lands on, so the
                // solver has nothing left to gain by disturbing a pairing that already beats
                // the gate on its own merits.
                if (mahal_sq > gate_mahalanobis && mahal_sq < kExcludedCost) {
                    mahal_sq = gate_mahalanobis + 1.0;
                }
                size_t const cell =
                    static_cast<size_t>(r) * static_cast<size_t>(n_cols) + static_cast<size_t>(c);
                cost[cell] = mahal_sq;
                if (nll_extra != 0.0 && mahal_sq <= gate_mahalanobis) {
                    extra[cell] = nll_extra;
                    min_extra = std::min(min_extra, nll_extra);
                    max_extra = std::max(max_extra, nll_extra);
                }
            }
        }

        // The gate above tested the distance alone. The extra terms can be negative, and the solver
        // takes non-negative costs below one gate: shift every pair that passed by the smallest
        // extra and widen the gate by the spread, so the pairs that passed stay below the gate and
        // the hopeless and excluded ones above it. Without keypoint candidates both are zero and
        // the matrix is untouched.
        double const solver_gate = gate_mahalanobis + (max_extra - min_extra);
        if (solver_gate != gate_mahalanobis) {
            for (size_t i = 0; i < cost.size(); ++i) {
                if (cost[i] <= gate_mahalanobis)
                    cost[i] += extra[i] - min_extra;
                else if (cost[i] < kExcludedCost)
                    cost[i] = solver_gate + 1.0;
            }
        }

        // Per-camera trust (TrackerConfig::dot_camera_noise_scale): looked up once per
        // camera, since it does not vary per candidate or slot.
        double const camera_noise_scale = [&] {
            auto const it = context.dot_camera_noise_scale.find(camera_id);
            return it != context.dot_camera_noise_scale.end() ? it->second : 1.0;
        }();

        auto pairs = solve_assignment(cost, n_rows, n_cols, solver_gate);
        if (tracing) {
            // Entries for this camera sit at [trace_start, trace_start + n_rows*n_cols)
            // in the same row-major (r, c) order the loop above pushed them in, so a
            // winning pair's entry is a direct index, no search needed.
            for (auto const& pair : pairs) {
                size_t const idx = trace_start +
                                   static_cast<size_t>(pair.row) * static_cast<size_t>(n_cols) +
                                   static_cast<size_t>(pair.col);
                trace->entries[idx].in_final_assignment = true;
            }
        }
        for (auto const& pair : pairs) {
            Column const& col = columns[static_cast<size_t>(pair.col)];
            UnlabeledCandidate const& cand = candidates[static_cast<size_t>(pair.row)];

            Observation obs;
            obs.camera_id = camera_id;
            obs.marker_id = col.marker_id;
            obs.frame_idx = frame_idx;
            obs.timestamp = timestamp;
            obs.position = cand.position;
            obs.position_distorted = cand.position_distorted;
            obs.confidence = cand.confidence;
            obs.tracklet_id = cand.tracklet_id;
            // A keypoint candidate is a heatmap maximum of its crop: its noise scales with the crop
            // like the fixed observation of the same keypoint (set again below for a dot).
            obs.crop_scale = cand.keypoint_idx >= 0 ? cand.crop_scale : 0.0;

            // Segmentation-mask trust bookkeeping (segmentation-mask-dot-assignment-
            // design.md §3): record whether the candidate that actually won this
            // slot was inside its subject's own mask label this frame, so a later
            // step's SegmentationMaskModifier can read this tracklet's *recent*
            // trustworthiness. Updated for the winning pair only -- a losing
            // candidate's position says nothing about which subject, if any, it
            // truly belongs to. Silently skipped (not a std::optional early-return)
            // when segmentation isn't configured, no mask decoded this frame, or the
            // candidate has no tracklet id to key the accumulator by.
            if (context.tracklet_mask_state != nullptr && cand.tracklet_id >= 0) {
                auto const seg_it = context.seg_mask_subject.find(col.subject_id);
                if (seg_it != context.seg_mask_subject.end() && seg_it->second.mask_label > 0) {
                    auto const mask_it = context.seg_masks_by_camera.find(camera_id);
                    if (mask_it != context.seg_masks_by_camera.end()) {
                        bool const in_mask = mask_label_present_near(
                            mask_it->second, cand.position_distorted, seg_it->second.mask_label,
                            context.dot_seg_mask_dilate_px);
                        (*context.tracklet_mask_state)[col.subject_id][camera_id][cand.tracklet_id]
                            .add(in_mask, context.dot_seg_mask_tracklet_window);
                    }
                }
            }

            // Same reasoning as the ArUco corner and dot-detector write paths'
            // own noise_scale=0.0 convention: a dot candidate's centroid comes
            // from thresholding the full-resolution frame directly, not a
            // fixed-input-resolution network, so there is no crop-scaled
            // detection-algorithm error to describe -- calibration error (ec)
            // alone should dominate Observation::measurement_noise_std(). (A keypoint candidate
            // keeps the crop scale set above.)

            // Motion-blur streak (dot_blob_detector.py's elongated-blob
            // acceptance path): its centroid is genuinely less precise than
            // a round dot's, roughly in proportion to how far the dot moved
            // during the exposure. A first cut, not yet modeling *direction*
            // -- the real uncertainty is larger along the blur than across
            // it, but Observation::measurement_noise_std() is a single
            // scalar (a known gap; streak-velocity-design.md covers the
            // velocity-from-streak and blinking-LED sub-frame-timing ideas).
            // Treats
            // the elongation beyond the round-dot footprint as spread
            // uniformly over the exposure window (stddev = width/sqrt(12)).
            // Left at 0.0 (the default -- normal formula applies) for a
            // round dot, where major_axis == minor_axis.
            double const elongation = cand.major_axis - cand.minor_axis;
            if (elongation > 1.0) {
                obs.noise_std_override = calib_noise_std + elongation / std::sqrt(12.0);
            }
            // Per-camera trust: scales whatever noise this observation would otherwise
            // get (the streak-inflated value above, or the base calibration noise), not
            // a cost -- a distrusted camera should still win its candidates on
            // geometry, only be weighed less once assigned. 1.0 (default, no entry for
            // this camera) leaves noise_std_override exactly as set above.
            if (camera_noise_scale != 1.0) {
                double const base =
                    obs.noise_std_override > 0.0 ? obs.noise_std_override : calib_noise_std;
                obs.noise_std_override = base * camera_noise_scale;
            }

            result[col.subject_id].resolved.push_back(obs);

            // Streak-derived velocity observation (streak-velocity-design.md §3/§4) --
            // an optional, orthogonal extra Observation alongside the POSITION one
            // above, not a replacement for it. See resolve_dot_assignment()'s own doc
            // comment (dot_assignment.hpp) for the documented "can't help a dot
            // reacquired after being lost entirely" scope limit this real-previous-
            // position requirement implies.
            if (streak_config.enabled && elongation >= streak_config.min_elongation_px &&
                (cand.dir_x != 0.0 || cand.dir_y != 0.0)) {
                auto subj_it = prev_positions.find(col.subject_id);
                if (subj_it != prev_positions.end()) {
                    auto cam_it2 = subj_it->second.find(camera_id);
                    if (cam_it2 != subj_it->second.end()) {
                        auto marker_it = cam_it2->second.find(col.marker_id);
                        if (marker_it != cam_it2->second.end()) {
                            Eigen::Vector2d const disp = cand.position - marker_it->second;
                            double const disp_px = disp.norm();
                            if (disp_px >= streak_config.min_displacement_px) {
                                StreakKAccumulator& acc = streak_k[camera_id];
                                acc.add(elongation, disp_px, streak_config.k_window);
                                if (auto k = acc.k(streak_config.k_min_samples);
                                    k.has_value() && *k > 0.0) {
                                    // The streak's axis is undirected (dot_blob_detector.py's
                                    // own canonicalization) -- resolve the sign against this
                                    // frame's real, measured displacement direction.
                                    Eigen::Vector2d streak_dir(cand.dir_x, cand.dir_y);
                                    if (streak_dir.dot(disp) < 0.0) {
                                        streak_dir = -streak_dir;
                                    }
                                    // Rescale the sub-exposure streak displacement up to a
                                    // full-frame-equivalent one (design doc §1: v =
                                    // streak_length / (k * frame_time), and this is
                                    // v * frame_time = streak_length / k, in the streak's
                                    // own resolved direction).
                                    Eigen::Vector2d const frame_equiv_disp =
                                        streak_dir * (elongation / *k);

                                    Observation vel_obs;
                                    vel_obs.camera_id = camera_id;
                                    vel_obs.marker_id = col.marker_id;
                                    vel_obs.frame_idx = frame_idx;
                                    vel_obs.timestamp = timestamp;
                                    vel_obs.confidence = cand.confidence;
                                    vel_obs.mode = MeasurementMode::VELOCITY;
                                    // position - prev_position must equal frame_equiv_disp --
                                    // see UnscentedKalmanFilter::observations_to_vector().
                                    vel_obs.position = cand.position;
                                    vel_obs.prev_position = cand.position - frame_equiv_disp;
                                    vel_obs.crop_scale = 0.0;
                                    vel_obs.noise_std_override = streak_config.velocity_noise_std;
                                    if (camera_noise_scale != 1.0) {
                                        vel_obs.noise_std_override *= camera_noise_scale;
                                    }

                                    result[col.subject_id].resolved.push_back(vel_obs);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return result;
}

void append_unique_candidates(std::vector<UnlabeledCandidate>& dest,
                              std::vector<UnlabeledCandidate> const& src, std::uint64_t owner_bit) {
    size_t const n_existing = dest.size();
    for (auto const& c : src) {
        UnlabeledCandidate* duplicate_of = nullptr;
        for (size_t i = 0; i < n_existing && duplicate_of == nullptr; ++i) {
            auto& d = dest[i];
            if (c.keypoint_idx >= 0 || d.keypoint_idx >= 0) {
                if (d.keypoint_idx == c.keypoint_idx && d.frame_idx == c.frame_idx &&
                    (d.position_distorted - c.position_distorted).norm() <=
                        std::min(d.merge_radius_px, c.merge_radius_px))
                    duplicate_of = &d;
                continue;
            }
            if (d.frame_idx == c.frame_idx && d.timestamp == c.timestamp &&
                d.tracklet_id == c.tracklet_id &&
                d.position_distorted.x() == c.position_distorted.x() &&
                d.position_distorted.y() == c.position_distorted.y()) {
                duplicate_of = &d;
            }
        }
        if (duplicate_of != nullptr) {
            duplicate_of->subject_mask |= owner_bit;
        } else {
            dest.push_back(c);
            dest.back().subject_mask = owner_bit;
        }
    }
}

std::string find_dot_config_disagreement(std::vector<TrackerConfig const*> const& configs) {
    if (configs.size() < 2)
        return {};
    TrackerConfig const& a = *configs.front();
    for (size_t i = 1; i < configs.size(); ++i) {
        TrackerConfig const& b = *configs[i];
#define POSETRAK_CHECK_DOT_FIELD(field) \
    if (a.field != b.field)             \
        return #field;
        POSETRAK_CHECK_DOT_FIELD(dot_assignment_gate_mahalanobis)
        POSETRAK_CHECK_DOT_FIELD(dot_tracklet_gate_multiplier)
        POSETRAK_CHECK_DOT_FIELD(calib_noise_std)
        POSETRAK_CHECK_DOT_FIELD(dot_streak_velocity_enabled)
        POSETRAK_CHECK_DOT_FIELD(dot_streak_k_window)
        POSETRAK_CHECK_DOT_FIELD(dot_streak_k_min_samples)
        POSETRAK_CHECK_DOT_FIELD(dot_streak_min_displacement_px)
        POSETRAK_CHECK_DOT_FIELD(dot_streak_min_elongation_px)
        POSETRAK_CHECK_DOT_FIELD(dot_streak_velocity_noise_std)
        POSETRAK_CHECK_DOT_FIELD(dot_reacquire_gap_frames)
        POSETRAK_CHECK_DOT_FIELD(dot_reacquire_max_px)
        POSETRAK_CHECK_DOT_FIELD(dot_cross_view_corroboration_px)
        POSETRAK_CHECK_DOT_FIELD(dot_corroboration_gate_multiplier)
        POSETRAK_CHECK_DOT_FIELD(dot_camera_noise_scale)
        POSETRAK_CHECK_DOT_FIELD(dot_seg_mask_enabled)
        POSETRAK_CHECK_DOT_FIELD(dot_seg_mask_dilate_px)
        POSETRAK_CHECK_DOT_FIELD(dot_seg_mask_tracklet_window)
        POSETRAK_CHECK_DOT_FIELD(dot_seg_mask_min_tracklet_fraction)
        POSETRAK_CHECK_DOT_FIELD(dot_seg_mask_cost_scale_at_zero_fraction)
        POSETRAK_CHECK_DOT_FIELD(dot_seg_mask_hard_distance_px)
        POSETRAK_CHECK_DOT_FIELD(pose_noise_std)
        POSETRAK_CHECK_DOT_FIELD(kp_candidate_peak_weight)
        POSETRAK_CHECK_DOT_FIELD(kp_candidate_tracklet_multiplier)
#undef POSETRAK_CHECK_DOT_FIELD
    }
    return {};
}

std::unordered_map<int, SubjectDotAssignment> resolve_shared_dot_assignment(
    std::vector<DotAssignmentSubject> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    TrackerConfig const& config, int frame_idx, double timestamp, AssignmentTrace* trace,
    std::unordered_map<int, DecodedSegMask> const& seg_masks_by_camera,
    TrackletMaskState* tracklet_mask_state, TrackletAssignments const* tracklet_assignments) {
    std::vector<SubjectDotPredictions> predictions;
    predictions.reserve(subjects.size());
    PrevDotPositions prev_positions;
    PrevDotTrackletIds prev_tracklet_ids;
    PrevDotResolvedFrame prev_resolved_frame;
    std::unordered_map<int, SegMaskSubjectConfig> seg_mask_subject;
    for (auto const& subject : subjects) {
        if (subject.seg_mask.mask_label > 0)
            seg_mask_subject[subject.subject_id] = subject.seg_mask;
    }
    // Cameras with at least one candidate this step -- computed once, shared
    // by every subject, rather than re-filtered inside each subject's own
    // loop.
    std::vector<int> cameras_with_candidates;
    for (auto const& [camera_id, candidates] : candidates_by_camera) {
        if (!candidates.empty())
            cameras_with_candidates.push_back(camera_id);
    }

    for (auto const& subject : subjects) {
        SubjectDotPredictions sp;
        sp.subject_id = subject.subject_id;
        // Batched across every camera in one call:
        // calling predict_dot_slot_predictions() once per camera repeated
        // sigma-point generation and the per-sigma-point FK sweep once per
        // camera, even though neither depends on camera_id at all -- see
        // Tracker::predict_dot_slot_predictions_all_cameras()'s own doc
        // comment.
        sp.predictions_by_camera =
            subject.tracker->predict_dot_slot_predictions_all_cameras(cameras_with_candidates);
        predictions.push_back(std::move(sp));
        prev_positions[subject.subject_id] = subject.tracker->prev_observations();
        prev_tracklet_ids[subject.subject_id] = subject.tracker->prev_dot_tracklet_ids();
        prev_resolved_frame[subject.subject_id] = subject.tracker->prev_dot_resolved_frame();
    }

    StreakVelocityConfig streak_config;
    streak_config.enabled = config.dot_streak_velocity_enabled;
    streak_config.k_window = config.dot_streak_k_window;
    streak_config.k_min_samples = config.dot_streak_k_min_samples;
    streak_config.min_displacement_px = config.dot_streak_min_displacement_px;
    streak_config.min_elongation_px = config.dot_streak_min_elongation_px;
    streak_config.velocity_noise_std = config.dot_streak_velocity_noise_std;

    // See this function's own doc comment (dot_assignment.hpp) for why the first
    // subject's accumulator map is used rather than a true per-camera-shared one.
    std::unordered_map<int, StreakKAccumulator>* streak_k_state =
        subjects.empty() ? nullptr : &subjects.front().tracker->streak_k_accumulators();

    DotAssignmentContext context;
    context.gate_mahalanobis = config.dot_assignment_gate_mahalanobis;
    context.frame_idx = frame_idx;
    context.timestamp = timestamp;
    context.calib_noise_std = config.calib_noise_std;
    context.pose_noise_std = config.pose_noise_std;
    for (auto const& subject : subjects) {
        if (!subject.marker_keypoint.empty())
            context.subject_marker_keypoint[subject.subject_id] = subject.marker_keypoint;
    }
    context.streak_config = streak_config;
    context.prev_positions = std::move(prev_positions);
    context.streak_k_state = streak_k_state;
    context.dot_tracklet_gate_multiplier = config.dot_tracklet_gate_multiplier;
    context.kp_candidate_peak_weight = config.kp_candidate_peak_weight;
    context.kp_candidate_tracklet_multiplier = config.kp_candidate_tracklet_multiplier;
    context.prev_tracklet_ids = std::move(prev_tracklet_ids);
    context.dot_reacquire_gap_frames = config.dot_reacquire_gap_frames;
    context.dot_reacquire_max_px = config.dot_reacquire_max_px;
    context.prev_resolved_frame = std::move(prev_resolved_frame);
    context.dot_cross_view_corroboration_px = config.dot_cross_view_corroboration_px;
    context.dot_corroboration_gate_multiplier = config.dot_corroboration_gate_multiplier;
    context.dot_camera_noise_scale = config.dot_camera_noise_scale;
    context.dot_seg_mask_enabled = config.dot_seg_mask_enabled;
    context.dot_seg_mask_dilate_px = config.dot_seg_mask_dilate_px;
    context.dot_seg_mask_hard_distance_px = config.dot_seg_mask_hard_distance_px;
    context.dot_seg_mask_tracklet_window = config.dot_seg_mask_tracklet_window;
    context.dot_seg_mask_min_tracklet_fraction = config.dot_seg_mask_min_tracklet_fraction;
    context.dot_seg_mask_cost_scale_at_zero_fraction =
        config.dot_seg_mask_cost_scale_at_zero_fraction;
    context.seg_mask_subject = std::move(seg_mask_subject);
    context.seg_masks_by_camera = seg_masks_by_camera;
    context.tracklet_mask_state = tracklet_mask_state;
    context.tracklet_assignments = tracklet_assignments;
    // Fundamental matrices are pure geometry from the calibrated cameras, so any
    // subject's own Tracker gives the same answer -- same "first subject" precedent
    // as streak_k_state above (every dot-bearing subject here is assumed built
    // against the same camera set). Skipped entirely when corroboration is off, to
    // avoid the O(cameras^2) matrix work on every step for a run that never uses it.
    if (config.dot_cross_view_corroboration_px > 0.0 && !subjects.empty()) {
        context.camera_fundamentals =
            build_camera_fundamentals(subjects.front().tracker->cameras());
    }
    return resolve_dot_assignment(predictions, candidates_by_camera, context, trace);
}

}  // namespace posetrak
