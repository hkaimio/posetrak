/**
 * @file multi_person_tracker.hpp
 * @brief Per-person tracking pipeline shared by the single- and multi-person CLI paths,
 * plus the MultiPersonTracker orchestrator that drives several people through it at once.
 *
 * This is Stage 1 of the cross-person relative observations plan (see
 * docs/roadmap/features/error-improvements/phase5-cross-person-plan.md): before any
 * cross-person coupling exists, the multi-person path must produce output that is
 * bitwise-identical to running each person through today's single-person `track`
 * command separately. The only way to guarantee that is to have one implementation of
 * "load a person, track them frame by frame, write results" that both paths call --
 * not two parallel implementations that can drift apart. That shared implementation
 * (build/step/finalize a PersonContext) used to be inlined in cli/track.cpp's
 * run_track_from_db(); it now lives here so MultiPersonTracker can reuse it.
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
void step_person_context(PersonContext& ctx, int step, bool verbose, bool quiet);

/// @brief Close exporters, run RTS smoothing if *smooth_output*, flush the result
/// writer, and write final statistics/summary files.
void finalize_person_context(PersonContext& ctx, bool smooth_output, bool quiet, bool verbose);

// ---------------------------------------------------------------------------
// Multi-person orchestrator
// ---------------------------------------------------------------------------

/// @brief Owns N persons' PersonContexts and drives them through a shared tracking
/// loop, one call to run() tracking all of them to completion.
///
/// Stage 1 (see phase5-cross-person-plan.md): persons are fully independent here --
/// no cross-person coupling exists yet. The loop is nonetheless structured as
/// "for frame: for person: step person" (not "for person: run to completion") so
/// Stage 2 can insert contact-gating/anchor-injection logic once per frame, after
/// all persons complete that frame, without restructuring the loop.
class MultiPersonTracker {
   public:
    MultiPersonTracker(std::vector<PersonSpec> const& specs, BuildPersonContextOptions const& opts,
                       bool verbose);

    /// @brief Run every person to completion (interleaved by step index).
    void run();

    std::vector<std::unique_ptr<PersonContext>> const& persons() const { return persons_; }

   private:
    std::vector<std::unique_ptr<PersonContext>> persons_;
    BuildPersonContextOptions opts_;
    bool verbose_ = false;
};

}  // namespace posetrak
