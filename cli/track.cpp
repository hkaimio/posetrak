#include <CLI/CLI.hpp>
#include <fmt/core.h>

#include "fmt/base.h"
#include "posetrak/calibration/scale_calibration.hpp"
#include "posetrak/core/config.hpp"
#include "posetrak/core/skeleton_layout.hpp"
#include "posetrak/db/result_writer.hpp"
#include "posetrak/db/session_reader.hpp"
#include "posetrak/io/camera_loader.hpp"
#include "posetrak/io/observation_loader.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "posetrak/io/statistics_tracker.hpp"
#include "posetrak/io/sync_loader.hpp"
#include "posetrak/io/tracking_export.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include "posetrak/kinematics/triangulation.hpp"
#include "posetrak/tracking/hierarchical_solver.hpp"
#include "posetrak/tracking/multi_person_tracker.hpp"
#include "posetrak/tracking/tracker.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

using namespace posetrak;

// ---------------------------------------------------------------------------
// 'scale' subcommand: post-process a completed calibration run.
// Reads state_vectors.csv from the config output directory, checks per-group
// convergence, then writes a calibrated skeleton YAML with updated offsets.
// ---------------------------------------------------------------------------
static int run_scale(std::string const& config_path, std::string scale_output_yaml, bool quiet) {
    try {
        if (!quiet) {
            fmt::print("Loading configuration: {}\n", config_path);
        }
        auto config = TrackerAppConfig::load(config_path);

        auto csv_path = (config.output_dir / "state_vectors.csv").string();
        if (!quiet) {
            fmt::print("Reading calibration data from: {}\n", csv_path);
        }

        ScaleCalibrationOptions opts;
        auto results = check_scale_convergence(csv_path, opts);

        // Print results table
        int converged_count = 0;
        fmt::print("\nScale calibration convergence (last {} frames):\n\n", opts.window_frames);
        fmt::print("  {:<24}  {:>8}  {:>8}  {}\n", "Group", "Scale", "Std", "Status");
        fmt::print("  {:<24}  {:>8}  {:>8}  {}\n", std::string(24, '-'), std::string(8, '-'),
                   std::string(8, '-'), "--------");
        for (auto const& r : results) {
            fmt::print("  {:<24}  {:>8.4f}  {:>8.4f}  {}\n", r.name, r.final_scale, r.scale_std,
                       r.converged ? "converged" : "NOT converged");
            if (r.converged)
                ++converged_count;
        }
        fmt::print("\n  {}/{} groups converged.\n\n", converged_count,
                   static_cast<int>(results.size()));

        if (scale_output_yaml.empty()) {
            scale_output_yaml = (config.output_dir / "calibrated.yaml").string();
        }

        if (!quiet) {
            fmt::print("Writing calibrated skeleton to: {}\n", scale_output_yaml);
        }
        write_calibrated_yaml(config.skeleton_path.string(), scale_output_yaml, results);
        fmt::print("Done. Calibrated skeleton: {}\n", scale_output_yaml);
        return 0;

    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }
}

// export_predicted_observations, export_state_vector, generate_state_header,
// write_smoothed_joint_angles_frame, write_smoothed_root_pose_frame now live in
// posetrak/tracking/multi_person_tracker.hpp -- shared with the per-person tracking
// pipeline (build_person_context/step_person_context/finalize_person_context) so the
// single- and multi-person CLI paths use one implementation instead of two that can
// drift apart.

// Helper: Load Python tracker state from CSV (for validation/comparison)
std::optional<State> load_python_state(std::string const& csv_path, Skeleton const& skeleton,
                                       int frame = 0) {
    fmt::print("\n=== Loading Python Tracker State from CSV: {} ===\n", csv_path);
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        fmt::print("Warning: Could not open Python state CSV: {}\n", csv_path);
        return std::nullopt;
    }

    std::string line;
    // Read header
    if (!std::getline(file, line)) {
        return std::nullopt;
    }

    // Read frames until we find the target frame
    int current_frame = -1;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string token;

        // Parse frame number (tracker_frame_idx - first column)
        if (!std::getline(ss, token, ',')) {
            continue;
        }
        current_frame = std::stoi(token);

        if (current_frame != frame) {
            continue;
        }

        // Skip timestamp
        if (!std::getline(ss, token, ',')) {
            continue;
        }

        // Found target frame, parse state
        // Format: ,tracker_frame_idx,timestamp,root_position_x,root_position_y,root_position_z,
        //         root_quaternion_w,root_quaternion_x,root_quaternion_y,root_quaternion_z,
        //         ...velocities..., joint_angles...

        // Root position
        Eigen::Vector3d root_position;
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(ss, token, ',')) {
                return std::nullopt;
            }
            root_position[i] = std::stod(token);
        }

        // Root orientation (quaternion: w,x,y,z)
        Eigen::Quaterniond root_orientation;
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.w() = std::stod(token);
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.x() = std::stod(token);
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.y() = std::stod(token);
        if (!std::getline(ss, token, ','))
            return std::nullopt;
        root_orientation.z() = std::stod(token);

        // Root velocity (x,y,z)
        Eigen::Vector3d root_velocity;
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(ss, token, ',')) {
                return std::nullopt;
            }
            root_velocity[i] = std::stod(token);
        }

        // Root angular velocity (x,y,z)
        Eigen::Vector3d root_angular_velocity;
        for (int i = 0; i < 3; ++i) {
            if (!std::getline(ss, token, ',')) {
                return std::nullopt;
            }
            root_angular_velocity[i] = std::stod(token);
        }

        // Joint angles and velocities (interleaved per joint)
        // Format: angle_0, angle_1, angle_2, velocity_0, velocity_1, velocity_2 for each joint
        // NOTE: CSV does NOT include the root/hips joint - it only has body joints
        int num_dof = skeleton.total_dof_count();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);
        Eigen::VectorXd joint_velocities = Eigen::VectorXd::Zero(num_dof);

        int dof_idx = 0;
        for (auto const& joint : skeleton.joints()) {
            // Skip root joint - CSV doesn't include it (root is handled separately above)
            // Note: total_dof_count() already excludes root, so don't increment dof_idx
            if (!joint.parent_index.has_value()) {
                continue;
            }

            int num_joint_dof = joint.dof;  // CSV has ALL DoFs

            // Read angles for this joint
            for (int i = 0; i < num_joint_dof; ++i) {
                if (!std::getline(ss, token, ',')) {
                    break;
                }
                joint_angles[dof_idx + i] = std::stod(token);
            }

            // Read velocities for this joint
            for (int i = 0; i < num_joint_dof; ++i) {
                if (!std::getline(ss, token, ',')) {
                    break;
                }
                joint_velocities[dof_idx + i] = std::stod(token);
            }

            dof_idx += num_joint_dof;
        }

        return State(root_position, root_orientation, joint_angles, root_velocity,
                     root_angular_velocity, joint_velocities);
    }

    return std::nullopt;
}

// Helper: Triangulate markers from first frame and compare with Python
void validate_camera_model(std::vector<Observation> const& observations,
                           std::unordered_map<int, Camera> const& cameras, Skeleton const& skeleton,
                           std::string const& python_csv_path) {
    fmt::print("\n=== Camera Model Validation ===\n");

    // Load Python marker positions for frame 0
    std::map<std::string, Eigen::Vector3d> python_positions;
    if (!python_csv_path.empty()) {
        std::ifstream file(python_csv_path);
        if (file.is_open()) {
            std::string line;
            // Read header
            std::getline(file, line);

            // Read frame 0 data
            while (std::getline(file, line)) {
                std::istringstream ss(line);
                std::string token;

                // Skip index column
                if (!std::getline(ss, token, ','))
                    continue;

                // Parse frame number
                if (!std::getline(ss, token, ','))
                    continue;
                int frame = std::stoi(token);
                if (frame != 0)
                    continue;

                // Skip timestamp
                if (!std::getline(ss, token, ','))
                    continue;

                // Skip person_id
                if (!std::getline(ss, token, ','))
                    continue;

                // Parse marker name
                std::string marker_name;
                if (!std::getline(ss, token, ','))
                    continue;
                marker_name = token;

                // Parse x, y, z
                Eigen::Vector3d pos;
                for (int i = 0; i < 3; ++i) {
                    if (!std::getline(ss, token, ','))
                        break;
                    pos[i] = std::stod(token);
                }

                python_positions[marker_name] = pos;
            }
        }
    }

    // Group observations by marker
    std::map<int, std::vector<Observation>> obs_by_marker;
    for (auto const& obs : observations) {
        obs_by_marker[obs.marker_id].push_back(obs);
    }

    // Triangulate each marker
    Triangulator triangulator(Triangulator::Method::DLT);
    int num_triangulated = 0;
    double total_error = 0.0;
    int num_compared = 0;

    for (auto const& [marker_id, marker_obs] : obs_by_marker) {
        if (marker_obs.size() < 2) {
            continue;  // Need at least 2 cameras
        }

        // Get marker name
        if (marker_id >= static_cast<int>(skeleton.markers().size())) {
            continue;
        }
        std::string marker_name = skeleton.markers()[marker_id].name;

        // Prepare for triangulation
        std::vector<Eigen::Vector2d> pixel_coords;
        std::vector<Camera const*> marker_cameras;
        std::vector<double> confidences;

        for (auto const& obs : marker_obs) {
            auto it = cameras.find(obs.camera_id);
            if (it == cameras.end()) {
                continue;
            }
            pixel_coords.push_back(obs.position);
            marker_cameras.push_back(&it->second);
            confidences.push_back(obs.confidence);
        }

        // Triangulate
        auto result = triangulator.triangulate(pixel_coords, marker_cameras, confidences);
        if (result.success) {
            num_triangulated++;
            Eigen::Vector3d cpp_pos = result.position;

            // Compare with Python if available
            auto it = python_positions.find(marker_name);
            if (it != python_positions.end()) {
                Eigen::Vector3d python_pos = it->second;
                double error = (cpp_pos - python_pos).norm();
                total_error += error;
                num_compared++;

                fmt::print(
                    "  {}: C++ ({:.3f}, {:.3f}, {:.3f}) vs Python ({:.3f}, {:.3f}, "
                    "{:.3f}) - error: {:.4f}m\n",
                    marker_name, cpp_pos.x(), cpp_pos.y(), cpp_pos.z(), python_pos.x(),
                    python_pos.y(), python_pos.z(), error);
            } else {
                fmt::print("  {}: C++ ({:.3f}, {:.3f}, {:.3f}) - no Python comparison\n",
                           marker_name, cpp_pos.x(), cpp_pos.y(), cpp_pos.z());
            }
        }
    }

    fmt::print("\nTriangulation summary:\n");
    fmt::print("  Triangulated markers: {}\n", num_triangulated);
    if (num_compared > 0) {
        fmt::print("  Compared with Python: {}\n", num_compared);
        fmt::print("  Mean position error: {:.4f}m\n", total_error / num_compared);
    }
    fmt::print("==============================\n\n");
}

// ---------------------------------------------------------------------------
// 'track' subcommand: run the UKF tracker on a TOML config.
// ---------------------------------------------------------------------------
static int run_track(std::string const& config_path, bool verbose, bool quiet, bool smooth_output,
                     bool calibrate, bool debug_output) {
    try {
        // Load configuration
        if (!quiet) {
            fmt::print("Loading configuration: {}\n", config_path);
        }
        auto config = TrackerAppConfig::load(config_path);
        config.validate();

        // Apply CLI overrides after config load
        if (calibrate) {
            config.calibration_mode = true;
        }
        if (debug_output) {
            config.export_debug = true;
        }

        // Load skeleton
        if (!quiet) {
            fmt::print("Loading skeleton: {}\n", config.skeleton_path.string());
        }
        auto skeleton = load_skeleton_from_yaml(config.skeleton_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} joints\n", skeleton.joints().size());
        }

        // active_joint_groups are used when constructing the SkeletonLayout:
        // Tracker reads config.active_joint_groups to build from_groups() layout.

        // Load cameras
        if (!quiet) {
            fmt::print("Loading cameras: {}\n", config.cameras_path.string());
        }
        auto cameras_by_name = load_cameras_from_toml(config.cameras_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} cameras\n", cameras_by_name.size());
        }

        // Load sync (optional)
        if (config.sync_path) {
            if (!quiet) {
                fmt::print("Loading sync: {}\n", config.sync_path->string());
            }
            auto sync_data = load_sync_metadata(config.sync_path->string());
            // Apply sync to cameras (modifies cameras_by_name in-place)
            apply_sync_metadata(cameras_by_name, sync_data, false);
        }

        // Load observations (camera IDs will be correct from the start)
        if (!quiet) {
            fmt::print("Loading observations: {}\n", config.observations_dir.string());
        }

        // Determine end time
        double end_time = config.end_time;
        if (end_time < 0.0) {
            // Load all available data - we'll determine end_time after loading
            end_time = std::numeric_limits<double>::max();
        }

        auto observations_set =
            load_openpose_sequence(config.observations_dir.string(), cameras_by_name, skeleton,
                                   config.start_time, end_time, 0.1, config.person_id);

        if (observations_set.empty()) {
            throw std::runtime_error("No observations found in time range");
        }

        // Auto-detect end time if not specified
        if (config.end_time < 0.0) {
            end_time = observations_set.max_time();
        }

        // Calculate number of tracker steps
        double dt = 1.0 / config.tracker_fps;
        int num_steps = static_cast<int>((end_time - config.start_time) / dt);

        if (num_steps <= 0) {
            throw std::runtime_error(
                "No time steps to process - check start_time, end_time, and tracker_fps");
        }

        if (!quiet) {
            fmt::print("  Time range: [{:.3f}, {:.3f}) seconds\n", config.start_time, end_time);
            fmt::print("  Tracker sample rate: {:.1f} Hz (dt = {:.6f} s)\n", config.tracker_fps,
                       dt);
            fmt::print("  Will process {} time steps\n", num_steps);
        }

        // Create tracker
        if (!quiet) {
            fmt::print("\nInitializing tracker...\n");
        }

        // Convert cameras to ID-keyed map for Tracker
        std::unordered_map<int, Camera> cameras;
        for (auto const& [name, cam] : cameras_by_name) {
            cameras.emplace(cam.id(), cam);
        }

        auto tracker_config = config.to_tracker_config();
        auto skeleton_ptr = std::make_shared<const Skeleton>(skeleton);
        Tracker tracker(skeleton_ptr, cameras, tracker_config);

        if (smooth_output) {
            tracker.enable_smoothing(true);
            if (!quiet) {
                fmt::print("RTS smoothing enabled: caching forward-pass data.\n");
            }
        }

        // Validate camera model by triangulating first frame
        double t_first_window = config.start_time + dt;
        auto first_frame_obs = observations_set.get_all_in_range(config.start_time, t_first_window);

        // Initialization priority:
        //  1. Explicit python_state_path in config → load that state directly (useful for
        //  debugging)
        //  2. Otherwise → run IK on first frame's triangulated observations (the normal path)
        //  3. If IK initialization fails → fall back to rest pose with a warning
        bool initialized = false;

        if (config.python_state_path.has_value()) {
            auto python_state =
                load_python_state(config.python_state_path.value().string(), skeleton, 0);
            if (python_state.has_value()) {
                if (!quiet) {
                    fmt::print("  Initializing from Python tracker state: {}\n",
                               config.python_state_path.value().string());
                    auto const& s = python_state.value();
                    fmt::print("    Root position: ({:.3f}, {:.3f}, {:.3f})\n",
                               s.root_position().x(), s.root_position().y(), s.root_position().z());
                }
                tracker.initialize_from_state(python_state.value(), config.start_time);
                initialized = true;
            } else {
                fmt::print(
                    "  WARNING: python_state_path '{}' could not be loaded, "
                    "falling back to IK initialization\n",
                    config.python_state_path.value().string());
            }
        }

        if (!initialized) {
            if (!quiet) {
                fmt::print("  Initializing from first-frame observations via IK...\n");
            }
            initialized = tracker.initialize(first_frame_obs, config.start_time);
            if (initialized) {
                if (!quiet) {
                    fmt::print("  IK initialization successful\n");
                }
            } else {
                fmt::print("  WARNING: IK initialization failed, falling back to rest pose\n");
                tracker.initialize_from_rest_pose(config.start_time);
                initialized = true;
            }
        }

        if (!quiet) {
            fmt::print("  Initialization complete\n\n");
        }

        // Get FK from tracker (matches the layout being used)
        ForwardKinematics* fk = tracker.get_fk();
        if (!fk) {
            throw std::runtime_error("Failed to get FK from tracker");
        }

        // Create exporters
        std::unique_ptr<TrackingExporter> exporter;
        std::unique_ptr<StatisticsTracker> stats_tracker;

        // Create layout for state export (matches tracker's layout).
        // Must be created before TrackingExporter since the exporter references it.
        auto layout = config.active_joint_groups.empty()
                          ? SkeletonLayout::from_full_skeleton(skeleton_ptr)
                          : SkeletonLayout::from_groups(skeleton_ptr, config.active_joint_groups);

        if (config.export_tracking_results) {
            exporter =
                std::make_unique<TrackingExporter>(config.output_dir, skeleton, *layout, cameras);
            exporter->open();
        }

        if (config.export_statistics) {
            stats_tracker = std::make_unique<StatisticsTracker>();
        }

        // Create diagnostic export files
        std::ofstream pred_obs_file;
        std::ofstream state_vec_file;
        if (config.export_tracking_results) {
            pred_obs_file.open(config.output_dir / "predicted_observations.csv");
            pred_obs_file << "frame,camera,marker,obs_u,obs_v,pred_u,pred_v,res_u,res_v,res_norm\n";

            state_vec_file.open(config.output_dir / "state_vectors.csv");
            state_vec_file << generate_state_header(*layout) << "\n";
        }

        // Enable UKF debug mode if requested via config or --debug flag
        if (config.export_debug) {
            if (auto* ukf = tracker.get_ukf()) {
                ukf->enable_debug(true, (config.output_dir / "debug").string());
            }
        }

        // Track sequence
        if (!quiet) {
            fmt::print("Tracking:\n");
        }

        auto start_time = std::chrono::steady_clock::now();
        int frames_tracked = 0;
        int frames_lost = 0;
        // True iff the frame-0 update below actually ran and succeeded, meaning it
        // pushed an entry to Tracker's RTS smoother cache with no filtered-row
        // counterpart (see the comment below and the smoothing block further down).
        bool frame0_tracked = false;

        // Note: State vectors will start at frame 1 (step 1 posterior) to align with
        // tracking_results.csv Initialization state (frame 0) is not exported to maintain alignment

        // Process first update (correct the initialization with first observations)
        {
            auto frame_0_obs = observations_set.get_all_in_range(config.start_time, t_first_window);
            if (!frame_0_obs.empty()) {
                if (auto* ukf = tracker.get_ukf()) {
                    ukf->set_frame_number(0);
                }

                // Debug: count unique marker-camera pairs
                std::set<std::pair<int, int>> unique_pairs;
                std::map<int, int> camera_counts;
                for (const auto& obs : frame_0_obs) {
                    unique_pairs.insert({obs.marker_id, obs.camera_id});
                    camera_counts[obs.camera_id]++;
                }

                if (!quiet) {
                    fmt::print(
                        "  Step 0: Updating initialization with {} observations ({} unique "
                        "marker-camera pairs) in time [{:.6f}, {:.6f})\n",
                        frame_0_obs.size(), unique_pairs.size(), config.start_time, t_first_window);
                    fmt::print("    Camera distribution:");
                    for (const auto& [cam_id, count] : camera_counts) {
                        fmt::print(" cam{}={}", cam_id, count);
                    }
                    fmt::print("\n");
                }

                // Use window midpoint as effective timestamp
                double t_effective = config.start_time + dt * 0.5;
                auto result = tracker.track_frame(frame_0_obs, t_effective);

                // Note: Step 0 posterior not exported - tracking_results starts at frame 1 (step 1)

                if (result.tracking_lost) {
                    fmt::print(stderr, "Warning: Tracking lost on first update\n");
                } else {
                    frames_tracked++;
                    frame0_tracked = true;
                }
            } else {
                if (!quiet) {
                    fmt::print(
                        "  No observations in time [{:.6f}, {:.6f}), skipping first update\n",
                        config.start_time, t_first_window);
                }
            }
        }

        // Main tracking loop - process remaining time steps
        for (int step = 1; step < num_steps; ++step) {
            double t_start =
                config.start_time + step * dt - dt / 2.0;  // Center the window around the step time
            double t_end = t_start + dt;

            // Set frame number for UKF debug
            if (auto* ukf = tracker.get_ukf()) {
                ukf->set_frame_number(step);
            }

            // Get all observations in this time window
            auto frame_obs = observations_set.get_all_in_range(t_start, t_end);

            if (frame_obs.empty()) {
                if (verbose) {
                    fmt::print("  Step {}: t=[{:.3f}, {:.3f}): No observations, skipping\n", step,
                               t_start, t_end);
                }
                continue;
            }

            // Use window midpoint as effective timestamp
            double t_effective = t_start + dt / 2.0;

            // Export predicted observations BEFORE tracking update (uses prior state)
            if (pred_obs_file.is_open()) {
                export_predicted_observations(pred_obs_file, step + 1, t_effective, frame_obs,
                                              tracker.state(), fk, cameras, skeleton);
            }

            // DEBUG: Export step 1 prior state for comparison with Python frame 1
            if (step == 1) {
                auto debug_dir = config.output_dir / "debug";
                std::filesystem::create_directories(debug_dir);
                auto prior_state_path = debug_dir / "step_0001_prior_state.csv";
                std::ofstream prior_file(prior_state_path);
                if (prior_file.is_open()) {
                    // Write header (same as state_vectors.csv)
                    prior_file << generate_state_header(*layout) << "\n";
                    // Write prior state (before update)
                    export_state_vector(prior_file, 2, t_effective, tracker.state(), *layout);
                    prior_file.close();
                    if (!quiet) {
                        fmt::print("  [DEBUG] Exported step 1 prior state to {}\n",
                                   prior_state_path.string());
                    }
                }
            }

            // Track using observations in window
            auto result = tracker.track_frame(frame_obs, t_effective);

            if (result.tracking_lost) {
                frames_lost++;
                if (verbose) {
                    fmt::print("  Step {}: t=[{:.3f}, {:.3f}): Tracking LOST ({} obs)\n", step,
                               t_start, t_end, result.update_info.num_observations);
                }
            } else {
                frames_tracked++;
                if (true) {
                    double sum_error = 0.0;
                    double max_error = 0.0;
                    uint32_t count = 0;
                    for (auto const& obs_result : result.update_info.observations) {
                        if (!obs_result.is_outlier) {
                            double error = obs_result.innovation.norm();
                            sum_error += error;
                            max_error = std::max(max_error, error);
                            count++;
                        }
                    }

                    fmt::print(
                        "  Step {}: t=[{:.3f}, {:.3f}): {} inliers, {} outliers mean reproj error "
                        "{:.4f} max {:.4f}\n",
                        step, t_start, t_end, result.update_info.num_inliers,
                        result.update_info.num_outliers, count > 0 ? sum_error / count : 0.0,
                        max_error);
                }
            }

            // Export state vector AFTER tracking update (posterior state)
            if (state_vec_file.is_open()) {
                export_state_vector(state_vec_file, step, t_effective, result.state, *layout);
            }

            // Export
            if (exporter) {
                // Compute marker positions using FK
                auto marker_positions_3d_map = fk->compute(result.state);

                // Convert to std::map (exporter expects std::map, not unordered_map)
                std::map<std::string, Eigen::Vector3d> marker_positions_3d(
                    marker_positions_3d_map.begin(), marker_positions_3d_map.end());

                exporter->write_frame(step, t_effective, result.state, marker_positions_3d,
                                      frame_obs, result.update_info);
            }

            if (stats_tracker) {
                stats_tracker->add_frame_stats(
                    step, t_effective, result.update_info, result.covariance, result.tracking_lost,
                    result.predict_ms, result.update_ms, result.p_sigma_gen_ms,
                    result.p_propagate_ms, result.p_mean_cov_ms, result.p_rts_ms, result.u_fk1_ms,
                    result.u_s_ms, result.u_outlier_ms, result.u_fk2_ms, result.u_inlier_ms,
                    result.u_kalman_ms, result.u_cov_update_ms);
            }

            // Progress indicator
            if (!quiet && !verbose && step % 10 == 0) {
                double percent = 100.0 * step / num_steps;
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
                double steps_per_sec = step / elapsed_sec;
                int eta_sec = static_cast<int>((num_steps - step) / steps_per_sec);

                fmt::print("  Progress: {}/{} ({:.1f}%) | {:.1f} steps/s | ETA: {}s\r", step,
                           num_steps, percent, steps_per_sec, eta_sec);
                std::cout.flush();
            }
        }

        if (!quiet && !verbose) {
            fmt::print("\n");
        }

        // Close exporters
        if (exporter) {
            exporter->close();
        }

        // Close diagnostic files
        if (pred_obs_file.is_open()) {
            pred_obs_file.close();
        }
        if (state_vec_file.is_open()) {
            state_vec_file.close();
        }

        // RTS smoother backward pass
        if (smooth_output) {
            if (!quiet) {
                fmt::print("Running RTS backward smoother...\n");
            }
            auto smoothed = tracker.smooth();

            // smoothed[0] is the frame-0 update's untracked warm-up result when it
            // actually ran (frame0_tracked) -- it has no filtered-row counterpart,
            // since that frame's posterior is deliberately never exported (see the
            // "Note" above the frame-0 update block). Skip it here so smoothed frame N
            // lines up with tracking_results.csv's frame N instead of being off by one
            // (a real, previously-shipped bug: every smoothed row was mislabeled).
            auto const smoothed_begin =
                frame0_tracked && !smoothed.empty() ? smoothed.begin() + 1 : smoothed.begin();

            // smoothed_state_vectors.csv  (diagnostic, same format as state_vectors.csv)
            {
                auto path = config.output_dir / "smoothed_state_vectors.csv";
                std::ofstream f(path);
                f << generate_state_header(*layout) << "\n";
                int step = 1;
                for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                    export_state_vector(f, step++, it->timestamp, it->state, *layout);
                }
            }

            // smoothed_joint_angles.csv  (same format as joint_angles.csv — BVH input)
            {
                auto path = config.output_dir / "smoothed_joint_angles.csv";
                std::ofstream f(path);
                f << "frame,timestamp,joint_name,angle_x,angle_y,angle_z,"
                     "velocity_x,velocity_y,velocity_z\n";
                int step = 1;
                for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                    write_smoothed_joint_angles_frame(f, step++, it->timestamp, it->state, *layout);
                }
            }

            // smoothed_root_pose.csv  (same format as root_pose.csv)
            {
                auto path = config.output_dir / "smoothed_root_pose.csv";
                std::ofstream f(path);
                f << "frame,timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,"
                     "vel_x,vel_y,vel_z,omega_x,omega_y,omega_z\n";
                int step = 1;
                for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                    write_smoothed_root_pose_frame(f, step++, it->timestamp, it->state);
                }
            }

            if (!quiet) {
                fmt::print(
                    "  Smoothed {} frames \u2192 smoothed_joint_angles.csv, "
                    "smoothed_root_pose.csv, smoothed_state_vectors.csv\n",
                    smoothed.size());
            }
        }

        // Write statistics
        if (stats_tracker) {
            stats_tracker->write_frame_stats(config.output_dir / "tracking_stats.csv");

            nlohmann::json metadata;
            metadata["sequence_name"] = config.observations_dir.filename().string();
            metadata["skeleton_file"] = config.skeleton_path.filename().string();
            metadata["num_cameras"] = cameras.size();
            metadata["num_markers"] = skeleton.markers().size();
            metadata["start_time"] = config.start_time;
            metadata["end_time"] = end_time;
            metadata["num_steps"] = num_steps;
            metadata["tracker_fps"] = config.tracker_fps;
            metadata["config"] = {
                {"process_noise_std", config.process_noise_std},
                {"pose_noise_std", config.pose_noise_std},
                {"calib_noise_std", config.calib_noise_std},
                {"outlier_threshold", config.outlier_threshold},
                {"ukf_alpha", config.ukf_alpha},
            };

            stats_tracker->write_summary_stats(config.output_dir / "overall_stats.json", metadata);
        }

        // Final summary
        auto tracking_end_time = std::chrono::steady_clock::now();
        auto total_elapsed = tracking_end_time - start_time;
        double total_sec =
            std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count() / 1000.0;
        double avg_steps_per_sec = frames_tracked / total_sec;

        if (!quiet) {
            fmt::print("\nTracking complete!\n");
            fmt::print("  Tracked: {}/{} steps ({:.1f}%)\n", frames_tracked, num_steps,
                       100.0 * frames_tracked / num_steps);
            fmt::print("  Lost: {} steps\n", frames_lost);
            fmt::print("  Average rate: {:.1f} steps/s\n", avg_steps_per_sec);
            fmt::print("  Total time: {:.1f}s\n", total_sec);

            if (stats_tracker) {
                fmt::print("  Mean reprojection error: {:.2f}px\n",
                           stats_tracker->mean_reprojection_error());
                fmt::print("  Outlier rate: {:.1f}%\n", stats_tracker->outlier_rate() * 100.0);
            }

            if (auto* ukf = tracker.get_ukf()) {
                fmt::print("  PSD eigensolver fired: {}/{} frames\n", ukf->psd_fix_count(),
                           frames_tracked);
            }

            fmt::print("\nResults exported to: {}\n", config.output_dir.string());
            if (config.export_tracking_results) {
                fmt::print("  - tracking_results.csv\n");
                fmt::print("  - joint_angles.csv\n");
                fmt::print("  - root_pose.csv\n");
                fmt::print("  - marker_projections.csv\n");
                fmt::print("  - observations.csv\n");
                fmt::print("  - predicted_observations.csv (diagnostic)\n");
                fmt::print("  - state_vectors.csv (diagnostic)\n");
            }
            if (config.export_statistics) {
                fmt::print("  - tracking_stats.csv\n");
                fmt::print("  - overall_stats.json\n");
            }
        }

        return 0;

    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }
}

// ---------------------------------------------------------------------------
// 'track --session-db' subcommand: run the tracker from a session DB.
// ---------------------------------------------------------------------------
static int run_track_from_db(std::string const& db_path, std::string const& sequence_id,
                             std::string const& skeleton_id, std::string const& config_id,
                             std::string const& output_dir, bool verbose, bool quiet,
                             bool smooth_output, bool debug_output, bool debug_init,
                             double min_confidence, int person_id,
                             std::vector<std::string> const& active_joint_groups,
                             double override_start_time = std::numeric_limits<double>::quiet_NaN(),
                             double override_end_time = std::numeric_limits<double>::quiet_NaN()) {
    try {
        PersonSpec spec;
        spec.sequence_id = sequence_id;
        spec.skeleton_id = skeleton_id;
        spec.config_id = config_id;
        spec.person_id = person_id;
        spec.output_dir = output_dir;

        BuildPersonContextOptions opts;
        opts.db_path = db_path;
        opts.min_confidence = min_confidence;
        opts.active_joint_groups = active_joint_groups;
        opts.override_start_time = override_start_time;
        opts.override_end_time = override_end_time;
        opts.debug_output = debug_output;
        opts.debug_init = debug_init;
        opts.smooth_output = smooth_output;
        opts.quiet = quiet;

        auto ctx = build_person_context(spec, opts, verbose);

        step_person_context_frame0(*ctx);
        for (int step = 1; step < ctx->num_steps; ++step) {
            step_person_context(*ctx, step, verbose, quiet);
        }

        finalize_person_context(*ctx, smooth_output, quiet, verbose);

        // Hierarchical solver child stages (existence-based toggle: a tracker_config_id
        // with tracker_config_stages rows runs hierarchically -- see
        // docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md).
        {
            SessionReader stage_reader(db_path);
            auto stages = stage_reader.load_tracker_config_stages(config_id);
            if (!stages.empty()) {
                run_hierarchical_child_stages(*ctx, stages, ctx->smoothed_frames, db_path, verbose,
                                              quiet);
            }
        }

        return 0;

    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }
}

// ---------------------------------------------------------------------------
// 'track --person ...' multi-person DB mode: tracks N people through the same
// session DB in one process, interleaved frame-by-frame. Stage 1 of the cross-person
// relative observations plan -- see
// docs/roadmap/features/error-improvements/phase5-cross-person-plan.md. No
// cross-person coupling yet: output must match running each person through
// run_track_from_db() separately (each person's CSVs land in
// <output_dir>/person_<index>/).
// ---------------------------------------------------------------------------
static int run_multi_person_track_from_db(std::string const& db_path,
                                          std::vector<PersonSpec> const& specs,
                                          std::string const& output_dir, bool verbose, bool quiet,
                                          bool smooth_output, bool debug_output, bool debug_init,
                                          double min_confidence,
                                          std::vector<std::string> const& active_joint_groups,
                                          double override_start_time, double override_end_time) {
    try {
        BuildPersonContextOptions opts;
        opts.db_path = db_path;
        opts.min_confidence = min_confidence;
        opts.active_joint_groups = active_joint_groups;
        opts.override_start_time = override_start_time;
        opts.override_end_time = override_end_time;
        opts.debug_output = debug_output;
        opts.debug_init = debug_init;
        opts.smooth_output = smooth_output;
        opts.quiet = quiet;

        std::vector<PersonSpec> resolved_specs = specs;
        for (size_t i = 0; i < resolved_specs.size(); ++i) {
            resolved_specs[i].output_dir =
                std::filesystem::path(output_dir) / ("person_" + std::to_string(i));
        }

        MultiPersonTracker tracker(resolved_specs, opts, verbose);
        tracker.run();

        // Hierarchical solver child stages, per person (existence-based toggle -- see
        // docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md).
        // unique_ptr<PersonContext> const& in persons() only const-qualifies the
        // pointer itself, not the pointee -- *persons()[i] is a mutable PersonContext&.
        {
            SessionReader stage_reader(db_path);
            for (auto const& person_ctx : tracker.persons()) {
                auto stages = stage_reader.load_tracker_config_stages(person_ctx->spec.config_id);
                if (!stages.empty()) {
                    run_hierarchical_child_stages(*person_ctx, stages, person_ctx->smoothed_frames,
                                                  db_path, verbose, quiet);
                }
            }
        }

        return 0;

    } catch (std::exception const& e) {
        fmt::print(stderr, "Error: {}\n", e.what());
        return 1;
    }
}

// ---------------------------------------------------------------------------
// main: thin subcommand dispatcher
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    CLI::App app{"Posetrak - Motion Capture Tracker"};
    app.require_subcommand(1);

    // ---- 'track' subcommand ---------------------------------------------
    auto* track_cmd = app.add_subcommand("track", "Run the UKF tracker on a capture sequence.");
    std::string track_config;
    bool verbose = false;
    bool quiet = false;
    bool smooth_output = false;
    bool calibrate = false;
    bool debug_output = false;
    bool debug_init = false;
    // DB mode arguments
    std::string db_path;
    std::string db_sequence_id;
    std::string db_skeleton_id;
    std::string db_config_id;
    std::string db_output_dir = "tracking_output";
    double db_min_confidence = 0.1;
    int db_person_id = 0;
    std::vector<std::string> db_active_joint_groups;
    double db_start_time = std::numeric_limits<double>::quiet_NaN();
    double db_end_time = std::numeric_limits<double>::quiet_NaN();
    // Multi-person mode: repeated 4-value groups (sequence, skeleton, tracker_config,
    // person_id), flattened by CLI11's take_all() into one vector.
    std::vector<std::string> db_person_specs;

    track_cmd->add_option("config", track_config, "Configuration file (TOML)")
        ->expected(0, 1)
        ->check(CLI::ExistingFile);
    track_cmd->add_flag("-v,--verbose", verbose, "Verbose output (show per-frame statistics)");
    track_cmd->add_flag("-q,--quiet", quiet, "Quiet mode (only show errors)");
    track_cmd->add_flag("--smooth", smooth_output,
                        "Run RTS backward smoother after forward pass and export "
                        "smoothed_joint_angles.csv, smoothed_root_pose.csv, "
                        "smoothed_state_vectors.csv");
    track_cmd->add_flag("--calibrate", calibrate,
                        "Enable calibration mode: prismatic (bone-length) DOFs receive "
                        "small process noise so the UKF can update bone lengths from "
                        "marker residuals");
    track_cmd->add_flag("--debug", debug_output,
                        "Enable UKF debug output (overrides export_debug in config file). "
                        "Writes per-frame diagnostics to <output_dir>/debug/");
    track_cmd->add_flag("--debug-init", debug_init,
                        "Print per-marker 3D prior/posterior errors for the first tracked frame. "
                        "Shows FK vs triangulated 3D for both the UKF prior and posterior state.");
    // DB mode options
    track_cmd->add_option("--session-db", db_path, "Session DB file (enables DB mode)");
    track_cmd->add_option("--sequence", db_sequence_id,
                          "Pose observation sequence ID (required with --session-db)");
    track_cmd->add_option("--skeleton", db_skeleton_id, "Skeleton ID");
    track_cmd->add_option("--tracker-config", db_config_id, "Tracker config ID");
    track_cmd->add_option("--output-dir", db_output_dir,
                          "Output directory for DB mode (default: tracking_output)");
    track_cmd->add_option("--min-confidence", db_min_confidence,
                          "Min keypoint confidence for DB mode (default: 0.1)");
    track_cmd->add_option("--person-id", db_person_id,
                          "Person index in pose observations (default: 0)");
    track_cmd->add_option("--joint-groups", db_active_joint_groups,
                          "Active joint groups for DB mode (empty = all)");
    track_cmd->add_option("--start-time", db_start_time,
                          "Override sequence start time in seconds (default: auto-detect "
                          "first frame where min_cameras_for_init cameras are active)");
    track_cmd->add_option("--end-time", db_end_time,
                          "Override sequence end time in seconds (default: from sequence record)");
    track_cmd
        ->add_option("--person", db_person_specs,
                     "Track an additional person: --person <sequence> <skeleton> "
                     "<tracker_config> <person_id>. Repeat for each person; when given, "
                     "--sequence/--skeleton/--tracker-config/--person-id are ignored and "
                     "each person's output is written to <output-dir>/person_<index>/. "
                     "No cross-person coupling yet (Stage 1).")
        ->expected(4)
        ->take_all();

    // ---- 'scale' subcommand ---------------------------------------------
    auto* scale_cmd = app.add_subcommand(
        "scale",
        "Post-process a calibration run: check per-scale-group convergence and\n"
        "write a calibrated skeleton YAML with offsets absorbed from the final\n"
        "scale factors.  Reads state_vectors.csv from the config output directory.");
    std::string scale_config;
    std::string scale_output;
    bool scale_quiet = false;
    scale_cmd->add_option("config", scale_config, "Configuration file (TOML)")
        ->required()
        ->check(CLI::ExistingFile);
    scale_cmd->add_option("-o,--output", scale_output,
                          "Output path for calibrated skeleton YAML "
                          "(default: <output_dir>/calibrated.yaml)");
    scale_cmd->add_flag("-q,--quiet", scale_quiet, "Quiet mode (only show errors)");

    CLI11_PARSE(app, argc, argv);

    if (*track_cmd) {
        if (!db_person_specs.empty()) {
            // Multi-person mode: db_person_specs is a flat vector, 4 values per --person.
            if (db_path.empty()) {
                fmt::print(stderr, "Error: --person requires --session-db\n");
                return 1;
            }
            if (db_person_specs.size() % 4 != 0) {
                fmt::print(stderr, "Error: --person expects exactly 4 values per occurrence\n");
                return 1;
            }
            std::vector<PersonSpec> specs;
            for (size_t i = 0; i < db_person_specs.size(); i += 4) {
                PersonSpec spec;
                spec.sequence_id = db_person_specs[i];
                spec.skeleton_id = db_person_specs[i + 1];
                spec.config_id = db_person_specs[i + 2];
                spec.person_id = std::stoi(db_person_specs[i + 3]);
                specs.push_back(spec);
            }
            return run_multi_person_track_from_db(
                db_path, specs, db_output_dir, verbose, quiet, smooth_output, debug_output,
                debug_init, db_min_confidence, db_active_joint_groups, db_start_time, db_end_time);
        } else if (!db_path.empty()) {
            // DB mode: validate required DB args
            if (db_sequence_id.empty() || db_skeleton_id.empty() || db_config_id.empty()) {
                fmt::print(stderr,
                           "Error: --session-db requires --sequence, --skeleton, "
                           "and --tracker-config\n");
                return 1;
            }
            return run_track_from_db(db_path, db_sequence_id, db_skeleton_id, db_config_id,
                                     db_output_dir, verbose, quiet, smooth_output, debug_output,
                                     debug_init, db_min_confidence, db_person_id,
                                     db_active_joint_groups, db_start_time, db_end_time);
        } else {
            if (track_config.empty()) {
                fmt::print(stderr, "Error: config file required when not using --session-db\n");
                return 1;
            }
            return run_track(track_config, verbose, quiet, smooth_output, calibrate, debug_output);
        }
    }
    if (*scale_cmd)
        return run_scale(scale_config, scale_output, scale_quiet);
    return 1;
}
