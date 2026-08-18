// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file multi_person_tracker.hpp
 * @brief Per-person tracking pipeline shared by the single- and multi-person CLI paths,
 * plus the MultiPersonTracker orchestrator that drives several people through it at once.
 *
 * Stage 1 of the cross-person relative observations plan (see
 * docs/roadmap/features/error-improvements/phase5-cross-person-plan.md) established the
 * "for frame: for person" orchestrator with no cross-person coupling -- output had to be
 * bitwise-identical to running each person through today's single-person `track` command
 * separately, which is why the per-person pipeline (build/step/finalize a PersonContext)
 * lives here as one shared implementation instead of being duplicated.
 *
 * Stage 2 adds the coupling itself: three-level contact gating (bounding-box pre-gate,
 * marker-pair distance gate with hysteresis, per-camera candidate cap) plus runtime
 * cross-person PAIR_DIFF anchor construction, per the plan's "measurement model" and
 * "contact gating" sections. update_contact_pairs() and build_cross_person_anchors() are
 * pure functions (no Tracker/DB access) precisely so the gating and anchor-construction
 * logic can be unit-tested directly against synthetic marker positions/observations.
 */
#pragma once

#include "posetrak/core/camera.hpp"
#include "posetrak/core/config.hpp"
#include "posetrak/core/observation.hpp"
#include "posetrak/core/skeleton.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/core/state.hpp"
#include "posetrak/db/result_writer.hpp"
#include "posetrak/io/statistics_tracker.hpp"
#include "posetrak/io/tracking_export.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/tracking/tracker.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace posetrak {

// ---------------------------------------------------------------------------
// CSV export helpers (moved from cli/track.cpp so the library-level tracking
// pipeline below can use them; the CLI still calls these directly for the
// legacy TOML `run_track()` path).
// ---------------------------------------------------------------------------

/// @brief Append one row of {actual, predicted, residual} per observation to *file*.
void export_predicted_observations(std::ofstream& file, int frame_idx, double timestamp,
                                   std::vector<Observation> const& observations, State const& state,
                                   ForwardKinematics* fk,
                                   std::unordered_map<int, Camera> const& cameras,
                                   Skeleton const& skeleton);

/// @brief Append one full state-vector row (root pose + joint angles/velocities) to *file*.
void export_state_vector(std::ofstream& file, int frame_idx, double timestamp, State const& state,
                         SkeletonLayout const& layout);

/// @brief Generate the CSV header line matching export_state_vector()'s column order.
std::string generate_state_header(SkeletonLayout const& layout);

/// @brief Append one smoothed-joint-angles row (joint_angles.csv format) to *file*.
void write_smoothed_joint_angles_frame(std::ofstream& file, int frame, double timestamp,
                                       State const& state, SkeletonLayout const& layout);

/// @brief Append one smoothed-root-pose row (root_pose.csv format) to *file*.
void write_smoothed_root_pose_frame(std::ofstream& file, int frame, double timestamp,
                                    State const& state);

// ---------------------------------------------------------------------------
// Per-person tracking pipeline
// ---------------------------------------------------------------------------

/// @brief Identifies one person's tracking inputs within a session DB.
struct PersonSpec {
    std::string sequence_id;
    std::string skeleton_id;
    std::string config_id;
    int person_id = 0;
    std::filesystem::path output_dir;
};

/// @brief Options shared by every person built within one CLI invocation
/// (single- or multi-person).
struct BuildPersonContextOptions {
    std::string db_path;
    double min_confidence = 0.1;
    std::vector<std::string> active_joint_groups;
    double override_start_time = std::numeric_limits<double>::quiet_NaN();
    double override_end_time = std::numeric_limits<double>::quiet_NaN();
    bool debug_output = false;
    bool debug_init = false;
    bool smooth_output = false;
    bool quiet = true;
};

/// @brief Owns everything needed to track one person through a sequence and record
/// the results: the loaded skeleton/cameras/config/observations, the Tracker itself,
/// and the exporters/writers. Built by build_person_context(), advanced one frame at
/// a time by step_person_context_frame0()/step_person_context(), and closed out by
/// finalize_person_context() -- mirroring run_track_from_db()'s previous inline
/// setup/loop/teardown exactly, so behavior is unchanged.
///
/// Non-copyable, non-movable (holds ResultWriter/TrackingExporter/ofstream members
/// with reference/no-move semantics) -- always held via std::unique_ptr.
struct PersonContext {
    PersonSpec spec;

    // Loaded once in build_person_context().
    Skeleton skeleton;
    std::map<std::string, Camera> cameras_by_name;
    std::unordered_map<int, Camera> cameras_by_id;
    TrackerConfig tracker_config;
    double tracker_fps = 100.0;
    ObservationSet observations;
    std::string session_id;
    std::string extrinsic_calibration_id;
    std::string sync_config_id;

    std::shared_ptr<const SkeletonLayout> layout;

    /// Full-skeleton-width layout, built only when this person's tracker_config
    /// has hierarchical-solver child stages (tracker_config_stages rows) --
    /// nullptr for ordinary monolithic runs. In hierarchical mode *layout* may
    /// be a strict subset (e.g. "main" only), scoped to what this person's own
    /// Tracker actually solves, but tracking_results rows must stay
    /// full-skeleton-width so run_hierarchical_child_stages()'s merge
    /// (SkeletonLayout::build_index_map_from(child_layout), which requires the
    /// receiving layout to be a superset of the child's joints) can reach
    /// every stage's DOFs, not just the ones *layout* itself covers. Compact
    /// states are expanded to this width via
    /// hierarchical_solver.hpp's expand_state_to_full_layout() before being
    /// handed to ResultWriter -- see step_person_context()/finalize_person_context().
    std::shared_ptr<const SkeletonLayout> full_layout;

    std::unique_ptr<Tracker> tracker;
    ForwardKinematics* fk = nullptr;  ///< Owned by *tracker*.

    std::unique_ptr<ResultWriter> result_writer;
    std::unique_ptr<TrackingExporter> exporter;
    std::unique_ptr<StatisticsTracker> stats_tracker;
    std::ofstream pred_obs_file;
    std::ofstream state_vec_file;

    double start_time = 0.0;
    double end_time = 0.0;
    double dt = 0.0;
    int num_steps = 0;

    int frames_tracked = 0;
    int frames_lost = 0;
    std::chrono::steady_clock::time_point track_start_time;

    /// True iff step_person_context_frame0()'s track_frame() call actually ran and
    /// succeeded (non-empty observations, not tracking_lost) -- meaning it pushed an
    /// entry to Tracker's RTS smoother cache that has no filtered-row (is_smoothed=0)
    /// counterpart, since that frame's result is deliberately never written to
    /// tracking_results/state_vectors.csv (see step_person_context_frame0()'s doc
    /// comment). finalize_person_context() uses this to skip that leading cache
    /// entry when writing smoothed output, so smoothed tracker_step N lines up with
    /// filtered tracker_step N instead of being off by one.
    bool frame0_tracked = false;

    /// Populated by finalize_person_context() when smooth_output is true, already
    /// trimmed of the frame0_tracked leading entry (i.e. smoothed_frames[i]
    /// corresponds to tracker_step (i+1), matching the filtered rows exactly).
    /// Empty when smooth_output is false. A hierarchical solver's child stages
    /// (run_hierarchical_child_stages(), called after finalize_person_context())
    /// stream this as their freeflyer joint's trajectory.
    std::vector<SmoothedFrame> smoothed_frames;
};

/// @brief Load a person's skeleton/cameras/config/observations from the session DB,
/// construct and initialize their Tracker, and open their output CSVs/DB writer.
/// Throws std::runtime_error on failure (no observations, bad time range, etc.) --
/// same failure conditions as today's run_track_from_db() setup.
std::unique_ptr<PersonContext>
build_person_context(PersonSpec const& spec, BuildPersonContextOptions const& opts, bool verbose);

/// @brief Process the first post-initialization frame (step 0). Unlike every later
/// step, this one does not export/write anything -- matching run_track_from_db()'s
/// existing (long-standing) behavior of only tracking, not recording, that frame.
void step_person_context_frame0(PersonContext& ctx);

/// @brief Process tracker step *step* (1-based, i.e. the second tracked frame
/// onward): predict+update, then export to CSV and write to the session DB.
/// *quiet* only affects the periodic console progress print (every 10 steps).
/// *extra_observations* (Stage 2: cross-person anchor observations built by
/// MultiPersonTracker) are appended to this frame's own detections before the
/// empty-frame check and the track_frame() call, so a person with no detections
/// of their own this frame can still be tracked via anchors alone.
void step_person_context(PersonContext& ctx, int step, bool verbose, bool quiet,
                         std::vector<Observation> const& extra_observations = {});

/// @brief Close exporters, run RTS smoothing if *smooth_output*, flush the result
/// writer, and write final statistics/summary files.
void finalize_person_context(PersonContext& ctx, bool smooth_output, bool quiet, bool verbose);

// ---------------------------------------------------------------------------
// Stage 2: contact gating + cross-person anchor construction
//
// Both of the below are pure functions -- no Tracker/FK/DB access -- operating
// on plain marker-position maps and Observation lists, so the gating and
// anchor-construction logic is directly unit-testable without a session DB.
// ---------------------------------------------------------------------------

/// @brief Identifies one cross-person marker pair for contact gating. Canonical
/// form has person_a < person_b (the pair is inherently unordered at the
/// gating level -- see phase5-cross-person-plan.md's "measurement model"
/// section for why anchor *observations* are nonetheless directional).
struct ContactMarkerPair {
    int person_a;
    int marker_a;
    int person_b;
    int marker_b;

    bool operator<(ContactMarkerPair const& o) const {
        return std::tie(person_a, marker_a, person_b, marker_b) <
               std::tie(o.person_a, o.marker_a, o.person_b, o.marker_b);
    }
};

/// @brief One person's inputs to the contact gate for one frame.
struct PersonGatingInput {
    /// 0 = cross-person coupling disabled for this person (matches
    /// TrackerConfig::cross_person_max_world_mm's "0 = off" convention).
    double cross_person_max_world_mm = 0.0;
    /// name -> current 3D world position (meters).
    std::map<std::string, Eigen::Vector3d> marker_world_positions;
    /// name -> marker index (skeleton.markers() position), for building
    /// ContactMarkerPair keys.
    std::unordered_map<std::string, int> marker_name_to_id;
};

/// @brief Update the active cross-person contact marker-pair set in place, given
/// each person's current marker positions. Implements contact-gating levels 1
/// (bounding-box pre-gate, O(N^2) box checks) and 2 (marker-pair 3D distance,
/// O(M^2) only for boxes that intersect) from the plan's "Contact gating"
/// section, including the enter-at-T/exit-at-1.2*T hysteresis that keeps the
/// active set from flickering frame to frame.
///
/// *active_pairs* maps each currently-active pair to its last-computed 3D
/// distance (meters) -- reused by build_cross_person_anchors() for
/// closest-first candidate sorting without recomputing FK.
///
/// A pair between persons i<j is only ever considered when both
/// persons[i].cross_person_max_world_mm and persons[j]'s are > 0; the
/// effective threshold for that pair is the smaller (more conservative) of
/// the two.
void update_contact_pairs(std::vector<PersonGatingInput> const& persons,
                          std::map<ContactMarkerPair, double>& active_pairs);

/// @brief Build cross-person PAIR_DIFF anchor Observations for *my_idx*'s markers
/// that are in *active_pairs* with *other_idx*, using *other_idx*'s anchor-state
/// marker positions (either its current-frame posterior or a velocity-
/// extrapolated frame-(t-1) posterior -- the caller decides which and passes
/// the corresponding FK output). Implements contact-gating level 3 (per-camera
/// candidate cap, closest-first using the cached distances from
/// update_contact_pairs()) and the anchor construction/noise composition from
/// the plan's "measurement model" section.
///
/// Both *my_frame_obs* and *other_frame_obs* are the two people's own raw
/// per-camera detections for this frame (same source as track_frame()'s own
/// input) -- a candidate marker pair only produces an anchor observation for a
/// given camera when both people have a detection there meeting the
/// confidence floor. *anchor_noise_std_floor* is Stage 2's placeholder for
/// sigma_anchor: looked up per (camera, other marker) in
/// *anchor_noise_std_by_camera_marker* -- Stage 3's Tracker::marker_projection_std(),
/// the real per-marker Jacobian-based projected-uncertainty value -- floored and
/// mildly inflated by *anchor_noise_std_floor* as its own term in the noise
/// composition, guarding against the decentralized-fusion "data incest" failure
/// mode the plan flags. Falls back to the floor alone (as Stage 2 did before
/// Stage 3) for any (camera, marker) missing from the lookup, e.g. because the
/// projection failed or a caller hasn't wired Stage 3 in yet.
std::vector<Observation> build_cross_person_anchors(
    int my_idx, int other_idx, std::map<ContactMarkerPair, double> const& active_pairs,
    std::vector<Observation> const& my_frame_obs, std::vector<Observation> const& other_frame_obs,
    std::map<std::string, Eigen::Vector3d> const& other_anchor_marker_positions,
    Skeleton const& other_skeleton, std::unordered_map<int, Camera> const& cameras,
    double my_min_confidence, double other_min_confidence, int max_n, double my_pose_noise_std,
    double other_pose_noise_std, double anchor_noise_std_floor, int frame_idx, double timestamp,
    std::unordered_map<int, std::unordered_map<int, double>> const&
        anchor_noise_std_by_camera_marker = {});

// ---------------------------------------------------------------------------
// Multi-person orchestrator
// ---------------------------------------------------------------------------

/// @brief Owns N persons' PersonContexts and drives them through a shared tracking
/// loop, one call to run() tracking all of them to completion.
///
/// Stage 1 (see phase5-cross-person-plan.md) established the loop shape: "for
/// frame: for person: step person", so Stage 2 could insert contact-gating and
/// anchor injection without restructuring it. Stage 2 does exactly that: after
/// every person completes a frame, update_contact_pairs() re-evaluates the
/// active contact set from their frame-t FK positions; immediately before each
/// person's next track_frame() call, build_cross_person_anchors() builds that
/// person's anchor observations from the active set. The person processing
/// order rotates every frame so the "who goes first sees only stale anchors"
/// asymmetry doesn't consistently favor the same person.
class MultiPersonTracker {
   public:
    MultiPersonTracker(std::vector<PersonSpec> const& specs, BuildPersonContextOptions const& opts,
                       bool verbose);

    /// @brief Run every person to completion (interleaved by step index).
    void run();

    std::vector<std::unique_ptr<PersonContext>> const& persons() const { return persons_; }

   private:
    /// @brief Re-evaluate the active contact set from every (still-running)
    /// person's current FK marker positions -- gate levels 1-2.
    void update_contact_gate();

    /// @brief Build person *idx*'s cross-person anchor observations for *step*
    /// from the currently-active contact set -- gate level 3 + construction.
    /// *processed_this_frame[j]* selects, for each other person j the active
    /// set references, whether to anchor to their current-frame posterior
    /// (already stepped this frame) or a velocity-extrapolated frame-(t-1)
    /// posterior (not yet stepped).
    std::vector<Observation>
    build_anchor_observations(int idx, int step, std::vector<char> const& processed_this_frame);

    std::vector<std::unique_ptr<PersonContext>> persons_;
    BuildPersonContextOptions opts_;
    bool verbose_ = false;

    /// Active cross-person contact marker pairs -> last-computed 3D distance
    /// (meters); persists across frames for update_contact_pairs()'s hysteresis.
    std::map<ContactMarkerPair, double> active_contact_pairs_;
    /// Per-person marker name -> id, built once at construction (skeletons/markers
    /// don't change during a run).
    std::vector<std::unordered_map<std::string, int>> marker_name_to_id_;
};

}  // namespace posetrak
