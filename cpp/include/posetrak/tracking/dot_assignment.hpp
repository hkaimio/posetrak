// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file dot_assignment.hpp
 * @brief The shared dot-assignment phase -- see
 * docs/roadmap/features/marker-based-mocap/dot-assignment-architecture-design.md
 * §5.2/§7.1 for the design this implements.
 *
 * Resolves anonymous reflective-dot candidates to named marker slots across
 * every tracked subject *at once*, so a candidate can only ever be claimed by
 * one subject, never assigned to two independently. This has to run as one
 * combined phase rather than one independent resolution per subject: two
 * subjects' own local solves would each pick whichever candidate fits their
 * own prediction best, with neither aware the other might want the same one.
 *
 * Split into a pure core (resolve_dot_assignment(), no Tracker/skeleton/
 * camera access -- directly testable against fabricated predictions and
 * candidates) and a thin Tracker-calling wrapper
 * (resolve_shared_dot_assignment()), mirroring this codebase's existing
 * update_contact_pairs()/build_cross_person_anchors() vs.
 * MultiPersonTracker::update_contact_gate()/build_anchor_observations()
 * split for the structurally analogous cross-person case
 * (multi_person_tracker.hpp).
 *
 * Not yet wired into either real per-frame loop (run_track_from_db()'s raw
 * loop, MultiPersonTracker::run()) -- that's separate, later work (design
 * doc §9/§11): both callers need a predict-all/resolve/update-all three-pass
 * shape instead of their current one-call-per-subject step, which is real
 * plumbing, not something this file's own scope covers.
 */
#pragma once

#include "posetrak/core/config.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/db/session_reader.hpp"
#include "posetrak/tracking/marker_prediction.hpp"
#include "posetrak/tracking/tracker.hpp"
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief One subject's resolved dot observations for this frame -- the
/// per-subject share of resolve_dot_assignment()'s / resolve_shared_dot_assignment()'s
/// combined result.
struct SubjectDotAssignment {
    std::vector<Observation> resolved;
};

/// @brief One dot-bearing subject's already-computed MarkerPrediction seam
/// output for this frame: camera_id -> marker_id (skeleton().markers() index)
/// -> MarkerPrediction, e.g. calling Tracker::predict_dot_slot_predictions()
/// once per camera and gathering the results here. No Tracker/skeleton
/// access of its own -- this is what makes resolve_dot_assignment() directly
/// testable against fabricated data.
struct SubjectDotPredictions {
    int subject_id;
    std::unordered_map<int, std::unordered_map<int, MarkerPrediction>> predictions_by_camera;
};

/// @brief Pure core of the shared dot-assignment phase (design doc §5.2/§7.1):
/// one combined Hungarian solve per camera, columns = the union of every
/// participating subject's dot-slot predictions for that camera, rows = that
/// camera's candidate list -- so a candidate can only ever go to one subject,
/// never both, and the assignment is globally optimal across every subject at
/// once rather than order-dependent (design doc §5.3's joint-vs-sequential
/// decision).
///
/// Cost is squared Mahalanobis distance,
/// `(candidate - predicted)^T * Cov_pixel^-1 * (candidate - predicted)`,
/// gated against *gate_mahalanobis* via solve_assignment() (assignment.hpp) --
/// a pairing above the gate is dropped, not forced ("ambiguity policy: drop,
/// don't guess", marker-detection-analysis.md). A slot or candidate absent
/// from every returned Observation was left unmatched by the gate, not a bug.
///
/// @param subjects Every dot-bearing subject's predictions for this frame,
///        gathered by the caller (see resolve_shared_dot_assignment() for
///        the Tracker-calling version of that gathering step).
/// @param candidates_by_camera This frame's anonymous dot candidates, keyed
///        by camera_id (design doc §5.4: assumed already a single
///        de-duplicated pool per camera -- the scene-wide-detection/de-dup
///        bridge a second real dot-bearing subject would need is explicitly
///        out of scope here, see that section).
/// @param gate_mahalanobis Squared-Mahalanobis-distance gate
///        (TrackerConfig::dot_assignment_gate_mahalanobis).
/// @param frame_idx Frame index to stamp onto every resolved Observation.
/// @param timestamp Timestamp to stamp onto every resolved Observation.
/// @param calib_noise_std Base calibration noise (TrackerConfig::calib_noise_std)
///        a resolved Observation's noise_std_override is inflated from when its
///        candidate is a motion-blur streak (major_axis notably exceeds
///        minor_axis -- see dot_blob_detector.py's elongated-blob acceptance
///        path) rather than a round dot. Defaulted so existing callers/tests
///        constructing candidates with major_axis == minor_axis (a round dot)
///        are unaffected either way.
/// @return subject_id -> SubjectDotAssignment, for every subject that had at
///         least one resolved Observation. A subject with nothing resolved
///         this frame (no predictions, or every candidate gated out) is
///         simply absent from the map, not present with an empty vector.
std::unordered_map<int, SubjectDotAssignment> resolve_dot_assignment(
    std::vector<SubjectDotPredictions> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    double gate_mahalanobis, int frame_idx, double timestamp, double calib_noise_std = 5.0);

/// @brief One dot-bearing subject as resolve_shared_dot_assignment() needs
/// it: an id to key the result map by, plus the Tracker to query
/// predictions from.
struct DotAssignmentSubject {
    int subject_id;
    /// Must have already had predict_step() called for this frame --
    /// resolve_shared_dot_assignment() only queries
    /// predict_dot_slot_predictions(), it never advances the filter itself
    /// (design doc §5.2: every subject predicts first, in a separate pass
    /// the caller drives, before this function runs).
    Tracker* tracker;
};

/// @brief Thin Tracker-calling wrapper around resolve_dot_assignment(): for
/// every subject, calls predict_dot_slot_predictions() once per camera_id
/// key present in *candidates_by_camera*, gathers the results into
/// SubjectDotPredictions, and delegates. See resolve_dot_assignment() for
/// the actual resolution logic and its own doc comment for every parameter
/// this forwards unchanged.
///
/// @param subjects Every dot-bearing subject participating this frame.
/// @note Every camera_id key in *candidates_by_camera* is assumed valid for
///       every subject's own Tracker (i.e. every subject was built against
///       the same camera map) -- Tracker::predict_dot_slot_predictions()
///       throws for a camera_id its own Tracker doesn't know about, and
///       that throw is not caught here; a real caller spanning subjects
///       with genuinely different camera coverage needs to account for that
///       itself (not a case this round's real captures produce).
std::unordered_map<int, SubjectDotAssignment> resolve_shared_dot_assignment(
    std::vector<DotAssignmentSubject> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    TrackerConfig const& config, int frame_idx, double timestamp);

}  // namespace posetrak
