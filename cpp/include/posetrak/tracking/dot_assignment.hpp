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
 * Wired into run_track_from_db()'s single-subject loop (cpp/cli/track.cpp);
 * MultiPersonTracker::run() still needs the same predict-all/resolve/
 * update-all three-pass wiring (design doc §9/§11) -- real plumbing, not
 * something this file's own scope covers.
 */
#pragma once

#include "posetrak/core/config.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/db/session_reader.hpp"
#include "posetrak/tracking/marker_prediction.hpp"
#include "posetrak/tracking/streak_k_accumulator.hpp"
#include "posetrak/tracking/tracker.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief Everything resolve_dot_assignment() needs to also emit a streak-derived
/// VELOCITY observation for a streaked candidate (streak-velocity-design.md §4) --
/// bundled since it's an optional, orthogonal extra step on top of the core
/// assignment. A default-constructed value (enabled=false) skips it entirely, so
/// every existing caller/test is unaffected.
struct StreakVelocityConfig {
    bool enabled = false;
    int k_window = 200;                ///< Rolling sample window per camera.
    int k_min_samples = 20;            ///< Minimum samples before k is trusted.
    double min_displacement_px = 3.0;  ///< "Only dots with actual movement" gate.
    double min_elongation_px = 1.0;    ///< Matches the streak-noise-inflation gate below.
    double velocity_noise_std = 10.0;  ///< noise_std_override for the emitted observation.
};

/// @brief This frame's previous-frame resolved dot positions, keyed
/// subject_id -> camera_id -> marker_id -> undistorted pixel position -- one
/// subject's own Tracker::prev_observations() gathered per subject (subject-scoped
/// rather than a flat camera/marker map) so two subjects' unrelated marker_id
/// numbering can never collide. Only used by the streak-velocity extension (empty
/// is the correct default when it's disabled).
using PrevDotPositions =
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, Eigen::Vector2d>>>;

/// @brief This frame's previous-frame resolved dot tracklet ids, keyed
/// subject_id -> camera_id -> marker_id -> tracklet_id (db::DotCandidate::
/// tracklet_id, dot_tracklet.MotionGatedLinker) -- same shape and same
/// stale-until-overwritten lifetime as PrevDotPositions above (gathered
/// from one subject's own Tracker::prev_dot_tracklet_ids() per subject_id).
/// Used by the gate-relaxation mechanism below: a candidate whose own
/// tracklet_id matches the id that resolved into a given (subject, camera,
/// marker) slot *last* frame is treated as continuing an already-
/// established track, not judged from a cold start every frame. Empty (the
/// default) disables relaxation entirely -- every existing caller/test is
/// unaffected.
using PrevDotTrackletIds =
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, int>>>;

/// @brief The frame_idx each (subject, camera, marker) slot was last resolved
/// on, same shape and stale-until-overwritten lifetime as PrevDotTrackletIds
/// above (gathered from one subject's own Tracker::prev_dot_resolved_frame()
/// per subject_id). Feeds ReacquisitionGateModifier: absent from the map means
/// the slot has never resolved, so there is no established track a bad
/// reacquisition could discontinue, and the gate does not apply.
using PrevDotResolvedFrame =
    std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, int>>>;

/// @brief Per (subject_id, marker_id) slot, the candidate positions found
/// within DotAssignmentContext::dot_reacquire_max_px of that camera's own
/// prediction for the slot this step, keyed by camera_id -- the shared
/// cross-camera evidence ReacquisitionGateModifier and
/// CrossViewCorroborationModifier both read (design plan §2.2/§2.3 call for
/// one pre-pass, not one per mechanism). ReacquisitionGateModifier only needs
/// which cameras are present (the map's key count); CrossViewCorroborationModifier
/// needs the actual positions, to test each one's epipolar distance. Computed
/// once per step by resolve_dot_assignment() itself (it needs every camera's
/// candidates, not just the one being costed), so a caller leaves this at its
/// default; it is not meant to be filled in by hand.
using SlotEvidence = std::unordered_map<
    int, std::unordered_map<int, std::unordered_map<int, std::vector<Eigen::Vector2d>>>>;

/// @brief The fundamental matrix from one camera to another, for every ordered
/// pair of cameras CrossViewCorroborationModifier might compare -- keyed by
/// `(int64_t(from_camera_id) << 32) | uint32_t(to_camera_id)`, F such that for a
/// point x in *from_camera_id*'s undistorted pixels, `F * [x;1]` is its epipolar
/// line in *to_camera_id*. Computed once per call by resolve_shared_dot_assignment()
/// from the calibrated cameras (the pure core never touches a Camera object) --
/// see build_camera_fundamentals(). Empty when corroboration is off.
using CameraPairFundamental = std::unordered_map<std::int64_t, Eigen::Matrix3d>;

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

/// @brief Everything a call of resolve_dot_assignment() reads besides the
/// subjects' predictions and the candidates.
///
/// Holds the frame being resolved, the gate, and the state of the previous
/// frame that the cost modifiers and the streak-velocity extension look up. The
/// defaults leave every optional mechanism off, so a default-constructed value
/// resolves by squared Mahalanobis distance alone.
struct DotAssignmentContext {
    /// Squared-Mahalanobis gate (TrackerConfig::dot_assignment_gate_mahalanobis,
    /// chi-squared 99% for 2 DOF). A pairing above it is dropped, not forced
    /// ("ambiguity policy: drop, don't guess", marker-detection-analysis.md).
    double gate_mahalanobis = 9.21;
    int frame_idx = 0;       ///< Stamped onto every resolved Observation.
    double timestamp = 0.0;  ///< Stamped onto every resolved Observation.
    /// Base calibration noise (TrackerConfig::calib_noise_std) a resolved
    /// Observation's noise_std_override is inflated from when its candidate is a
    /// motion-blur streak (major_axis notably exceeds minor_axis -- see
    /// dot_blob_detector.py's elongated-blob acceptance path). A round dot keeps
    /// the normal noise formula.
    double calib_noise_std = 5.0;
    /// Streak-derived velocity extension (streak-velocity-design.md §4).
    /// Default-constructed (enabled=false) skips it. When enabled, a resolved
    /// streaked candidate also gets a VELOCITY-mode Observation for the same slot,
    /// but only once prev_positions holds a real previous-frame position for that
    /// (subject, camera, marker): that admits the sample into the camera's k
    /// estimate and resolves the streak axis's 180-degree sign ambiguity against a
    /// measured direction. A dot reacquired after being lost entirely gets no
    /// streak-velocity boost on the frame it reappears (design doc §4).
    StreakVelocityConfig streak_config;
    /// Previous-frame resolved positions; empty is correct when streak_config is
    /// disabled.
    PrevDotPositions prev_positions;
    /// Per-camera k accumulator, updated in place as samples are admitted.
    /// nullptr uses a call-local one that is discarded.
    std::unordered_map<int, StreakKAccumulator>* streak_k_state = nullptr;
    /// Tracklet gate relaxation (TrackerConfig::dot_tracklet_gate_multiplier), see
    /// TrackletContinuityModifier. 1.0 turns it off.
    double dot_tracklet_gate_multiplier = 1.0;
    /// Previous-frame resolved tracklet ids; empty disables the relaxation.
    PrevDotTrackletIds prev_tracklet_ids;
    /// Reacquisition gate (TrackerConfig::dot_reacquire_gap_frames), see
    /// ReacquisitionGateModifier. 0 turns it off.
    int dot_reacquire_gap_frames = 0;
    /// Radius (TrackerConfig::dot_reacquire_max_px) for the gate's cross-camera
    /// "near the prediction" evidence -- shared with CrossViewCorroborationModifier's
    /// own evidence-gathering pass (see SlotEvidence).
    double dot_reacquire_max_px = 0.0;
    /// The frame_idx each slot was last resolved on; empty means the gate never
    /// applies (nothing has resolved yet to protect).
    PrevDotResolvedFrame prev_resolved_frame;
    /// Cross-camera candidate evidence for each slot this step (shared by
    /// ReacquisitionGateModifier and CrossViewCorroborationModifier). Computed
    /// internally by resolve_dot_assignment(); left at its default by every caller.
    SlotEvidence slot_evidence;
    /// Epipolar tolerance, in pixels, for CrossViewCorroborationModifier (TrackerConfig::
    /// dot_cross_view_corroboration_px). 0 turns it off.
    double dot_cross_view_corroboration_px = 0.0;
    /// Cost discount for a corroborated pair (TrackerConfig::
    /// dot_corroboration_gate_multiplier), same divide-the-cost convention as
    /// dot_tracklet_gate_multiplier. 1.0 is a no-op.
    double dot_corroboration_gate_multiplier = 1.0;
    /// Fundamental matrices for the participating cameras, computed by
    /// resolve_shared_dot_assignment() from the calibrated cameras. Empty
    /// disables corroboration regardless of the settings above (there is
    /// nothing to test candidates against) -- left at its default by every
    /// caller that does not have real cameras to offer (e.g. a unit test can
    /// still fill it in by hand against fabricated matrices).
    CameraPairFundamental camera_fundamentals;
    /// Per-camera factor (TrackerConfig::dot_camera_noise_scale, keyed by camera_id)
    /// on the measurement noise of a resolved dot observation from that camera --
    /// applied directly to Observation::noise_std_override, not a cost modifier (a
    /// per-camera trust is not usually a reason a candidate should *lose* the
    /// assignment, only a reason the filter should weigh its measurement
    /// differently once assigned). A camera absent from the map, or the map being
    /// empty, is a 1.0 no-op.
    std::unordered_map<int, double> dot_camera_noise_scale;
};

/// @brief One (subject, camera, marker) slot as a cost modifier sees it.
struct DotSlotRef {
    int subject_id;
    int camera_id;
    int marker_id;
    MarkerPrediction const* prediction;  ///< This step's prediction for the slot.
};

/// @brief What a CostModifier says about one candidate-slot pair.
struct CostSupport {
    /// The pair's squared Mahalanobis cost is divided by this: 1 leaves it, above
    /// 1 makes the pair cheaper, below 1 dearer. It divides rather than multiplies
    /// so that a factor equal to a config value reproduces that value's own
    /// division exactly.
    double factor = 1.0;
    /// The pair must not be chosen; its cost only has to lose to the solver's
    /// padding cost, which is just above the gate.
    bool exclude = false;
};

/// @brief One piece of evidence that changes the cost of pairing a candidate with
/// a slot.
///
/// A modifier is a pure function of the slot, the candidate and the context, so it
/// is testable on its own. resolve_dot_assignment() applies the modifiers in
/// order; an exclusion stops the chain for that pair.
class CostModifier {
   public:
    virtual ~CostModifier() = default;

    /// @param slot The slot being costed.
    /// @param candidate The candidate it might be paired with.
    /// @param context The frame's context.
    /// @return The modifier's support for the pair.
    virtual CostSupport apply(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                              DotAssignmentContext const& context) const = 0;
};

/// @brief Makes a pair cheaper when the candidate continues the tracklet that
/// resolved into the slot on the previous frame.
///
/// The factor is DotAssignmentContext::dot_tracklet_gate_multiplier. The gate
/// itself stays one scalar; only a pairing with independent identity evidence
/// behind it gets cheaper, rather than the gate loosening for everything (real
/// data showed only about 4-6% of raw candidates during a fast swing survive the
/// plain gate, with none rejected at the later UKF outlier check, so the
/// attrition is entirely there). Applies to candidates with a tracklet id.
class TrackletContinuityModifier final : public CostModifier {
   public:
    CostSupport apply(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                      DotAssignmentContext const& context) const override;
};

/// @brief Excludes a pair when the candidate is not in the subject's own sequence
/// (UnlabeledCandidate::subject_mask has no bit for the subject).
class CandidateOwnershipModifier final : public CostModifier {
   public:
    CostSupport apply(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                      DotAssignmentContext const& context) const override;
};

/// @brief Excludes a pair when the slot has gone DotAssignmentContext::
/// dot_reacquire_gap_frames-or-more steps unresolved and the candidate has none
/// of the three admitted kinds of independent evidence: it continues the
/// slot's last resolved tracklet (the same test TrackletContinuityModifier
/// uses, applied here regardless of the tracklet multiplier); the slot has a
/// near-prediction candidate in at least two cameras (DotAssignmentContext::
/// slot_evidence); or the candidate is itself epipolar-corroborated (the same
/// test CrossViewCorroborationModifier uses, applied here regardless of
/// dot_corroboration_gate_multiplier -- corroboration is real cross-camera
/// identity evidence for this exact candidate, the same kind the tracklet test
/// is, so it belongs in this "or" list once it exists, the two positional
/// checks above being cruder stand-ins for it). Otherwise a no-op -- including
/// for a slot that has never resolved, which has no established track to
/// protect.
///
/// This is a "never reseed from a bad guess after a gap, verify consistency
/// before trusting a resumed detection" rule, applied at assignment time
/// instead of after the fact: a slot left unresolved coasts on the process
/// model rather than risk locking onto a confidently wrong candidate.
class ReacquisitionGateModifier final : public CostModifier {
   public:
    CostSupport apply(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                      DotAssignmentContext const& context) const override;
};

/// @brief Makes a pair cheaper when the candidate is corroborated by another
/// camera: some candidate near another camera's own prediction for the same
/// slot (DotAssignmentContext::slot_evidence) lies within
/// dot_cross_view_corroboration_px of this candidate's epipolar line in that
/// camera (DotAssignmentContext::camera_fundamentals). The factor is
/// dot_corroboration_gate_multiplier; never below 1 (a lone camera with every
/// other one occluded must not be penalised, only a corroborated one
/// rewarded). A no-op when camera_fundamentals is empty (no cameras to
/// corroborate against) or dot_cross_view_corroboration_px is 0.
///
/// A candidate inside the assignment gate is already close to the true one, so
/// its epipolar line is close to the true candidate's own -- this test only
/// separates a confidently wrong candidate from a real one when
/// dot_cross_view_corroboration_px is tighter than the wrong candidate's own
/// offset, and that tolerance cannot be tighter than the camera pair's own
/// calibration/sync accuracy. Measured against a real, confidently-wrong
/// candidate on one capture: real correspondences between well-behaved camera
/// pairs residual under about 20 px, a fast-moving one (sync-sensitive) up to
/// about 60 px, while the wrong candidates were typically hundreds of pixels
/// off epipolar -- separable in practice, with a small unavoidable tail where
/// a wrong candidate happens to sit near the true one's epipolar ray (the
/// geometry any single other camera's epipolar line cannot resolve).
///
/// Not yet safe for a body with closely spaced dot slots: the same evidence
/// (DotAssignmentContext::slot_evidence) that finds a corroborating candidate
/// in another camera is gathered per marker but not marker-exclusive, so a
/// real candidate for one marker can be found as if it corroborated a
/// different, nearby marker. Backface culling on the marker's own outward
/// normal, extended from the rigid-body prediction path to the articulated
/// one, is the fix: two markers close together but facing opposite directions
/// are visible from disjoint camera sets, so the wrong marker would never earn
/// a prediction (and so never contaminate the evidence) in a camera that
/// cannot see its side to begin with.
class CrossViewCorroborationModifier final : public CostModifier {
   public:
    CostSupport apply(DotSlotRef const& slot, UnlabeledCandidate const& candidate,
                      DotAssignmentContext const& context) const override;
};

/// @brief The modifiers a context asks for, in the order they apply: tracklet
/// continuity (only when its multiplier is above 1 and there are previous
/// tracklet ids), then candidate ownership, then cross-view corroboration
/// (only when dot_cross_view_corroboration_px and dot_corroboration_gate_
/// multiplier are both set to have an effect and camera_fundamentals is not
/// empty), then the reacquisition gate (only when dot_reacquire_gap_frames is
/// above 0).
///
/// @param context The frame's context.
/// @return Owning pointers to the modifiers.
std::vector<std::unique_ptr<CostModifier>>
default_cost_modifiers(DotAssignmentContext const& context);

/// @brief Fundamental matrices for every ordered pair of *cameras* (design doc
/// §2.3) -- see CameraPairFundamental for the convention and key encoding.
/// Called by resolve_shared_dot_assignment(); a unit test wanting
/// DotAssignmentContext::camera_fundamentals can call it directly against its
/// own fabricated Camera objects, since this is pure geometry with no
/// Tracker/skeleton access.
///
/// @param cameras Every camera the participating subjects were built against.
/// @return Fundamental matrix for each ordered pair (both directions).
CameraPairFundamental build_camera_fundamentals(std::unordered_map<int, Camera> const& cameras);

/// @brief Pure core of the shared dot-assignment phase (design doc §5.2/§7.1):
/// one combined Hungarian solve per camera, columns = the union of every
/// participating subject's dot-slot predictions for that camera, rows = that
/// camera's candidate list -- so a candidate can only ever go to one subject,
/// never both, and the assignment is globally optimal across every subject at
/// once rather than order-dependent (design doc §5.3's joint-vs-sequential
/// decision).
///
/// The cost of a pair is the squared Mahalanobis distance,
/// `(candidate - predicted)^T * Cov_pixel^-1 * (candidate - predicted)`, changed by
/// the CostModifiers of default_cost_modifiers(), and gated against
/// DotAssignmentContext::gate_mahalanobis via solve_assignment() (assignment.hpp).
/// A slot or candidate absent from every returned Observation was left unmatched
/// by the gate, not a bug.
///
/// @param subjects Every dot-bearing subject's predictions for this frame,
///        gathered by the caller (see resolve_shared_dot_assignment() for the
///        Tracker-calling version of that gathering step).
/// @param candidates_by_camera This frame's anonymous dot candidates, keyed by
///        camera_id: one pool per camera, with the owners of each candidate in its
///        subject_mask (see append_unique_candidates()).
/// @param context The gate, frame stamp and previous-frame state.
/// @return subject_id -> SubjectDotAssignment, for every subject that had at
///         least one resolved Observation. A subject with nothing resolved this
///         frame (no predictions, or every candidate gated out) is absent from
///         the map, not present with an empty vector.
std::unordered_map<int, SubjectDotAssignment> resolve_dot_assignment(
    std::vector<SubjectDotPredictions> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    DotAssignmentContext const& context);

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

/// @brief Merges one subject's candidates into the shared pool of a camera.
///
/// Several subjects' sequences may each hold a copy of the same detection
/// run's dot rows. Merged naively, one physical dot becomes several
/// candidates and the joint assignment can hand it to each subject in turn.
/// Copies of one detection are byte-identical, so two candidates are the same
/// detection when frame, timestamp, distorted position and tracklet id all
/// match. A duplicate is not added again; its owner is added to the existing
/// candidate's subject_mask. Every other candidate is added owned by this
/// subject alone.
///
/// The lists are one camera's candidates for one tracker step (about one video
/// frame), not a whole sequence. The comparison is O(existing x incoming): for
/// three subjects sharing 340 candidates per camera it measured 0.07 ms per
/// camera per step, and 0.3 ms at 700, against 18 ms and 175 ms for the
/// assignment solver that follows. Index the existing entries by a hash of the
/// identifying fields if the per-camera count ever grows far beyond that.
///
/// @param dest Pool to merge into. Only the entries present before the call are
///        compared, so one subject's own list is never merged with itself.
/// @param src The subject's candidates for this camera and step.
/// @param owner_bit The subject's bit, `1 << subject id`, set in the subject_mask
///        of every candidate it holds.
void append_unique_candidates(std::vector<UnlabeledCandidate>& dest,
                              std::vector<UnlabeledCandidate> const& src, std::uint64_t owner_bit);

/// @brief Finds a TrackerConfig setting on which the configs of subjects that
/// share one dot assignment disagree.
///
/// The shared solve takes its settings from one config for every subject, so a
/// disagreement would silently favour whichever subject came first.
///
/// @param configs The tracker configs of the dot-bearing subjects.
/// @return The name of the first setting the shared solve reads on which two
///         configs differ, or an empty string when they all agree.
std::string find_dot_config_disagreement(std::vector<TrackerConfig const*> const& configs);

/// @brief Thin Tracker-calling wrapper around resolve_dot_assignment(): for
/// every subject, calls predict_dot_slot_predictions() once per camera_id
/// key present in *candidates_by_camera*, gathers the results into
/// SubjectDotPredictions, and delegates. See resolve_dot_assignment() for
/// the actual resolution logic and its own doc comment for every parameter
/// this forwards unchanged.
///
/// Also builds resolve_dot_assignment()'s streak-velocity inputs from
/// *config*'s dot_streak_* fields: gathers each subject's own
/// Tracker::prev_observations() into a PrevDotPositions, and forwards the
/// first subject's Tracker::streak_k_accumulators() as the mutable k state --
/// k is a property of the camera (exposure/frame-rate), not of any one
/// subject, so it should in principle be shared across every subject using
/// that camera; simplified to the first subject's own accumulator map since
/// no real capture this round has more than one dot-bearing subject sharing a
/// camera (same simplification precedent as this function's own multi-
/// subject/camera-coverage note below, for a different concern).
///
/// Also gathers each subject's own Tracker::prev_dot_tracklet_ids() into a
/// PrevDotTrackletIds and forwards *config*.dot_tracklet_gate_multiplier --
/// the tracklet gate-relaxation mechanism (see resolve_dot_assignment()'s own
/// doc comment).
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
