#include "posetrak/tracking/multi_person_tracker.hpp"

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "posetrak/db/session_reader.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace posetrak {

// ---------------------------------------------------------------------------
// CSV export helpers (moved verbatim from cli/track.cpp)
// ---------------------------------------------------------------------------

void export_predicted_observations(std::ofstream& file, int frame_idx, double timestamp,
                                   std::vector<Observation> const& observations, State const& state,
                                   ForwardKinematics* fk,
                                   std::unordered_map<int, Camera> const& cameras,
                                   Skeleton const& skeleton) {
    (void)timestamp;
    // Compute 3D marker positions from current state
    auto marker_positions_3d = fk->compute(state);

    // For each observation, project and compute residual
    for (auto const& obs : observations) {
        // Find marker name from marker_id (marker_id is the index in skeleton.markers())
        if (obs.marker_id < 0 || obs.marker_id >= static_cast<int>(skeleton.markers().size()))
            continue;

        std::string marker_name = skeleton.markers()[obs.marker_id].name;

        // Get 3D position
        auto it = marker_positions_3d.find(marker_name);
        if (it == marker_positions_3d.end())
            continue;

        Eigen::Vector3d pos_3d = it->second;

        // Find camera
        auto cam_it = cameras.find(obs.camera_id);
        if (cam_it == cameras.end())
            continue;

        auto const& camera = cam_it->second;

        // Project to camera
        auto predicted_opt = camera.project(pos_3d);
        if (!predicted_opt.has_value()) {
            // Skip failed projections
            continue;
        }
        Eigen::Vector2d predicted = *predicted_opt;

        // Compute residual
        double residual_u = obs.position.x() - predicted.x();
        double residual_v = obs.position.y() - predicted.y();
        double residual_norm = std::sqrt(residual_u * residual_u + residual_v * residual_v);

        // Write: frame,camera,marker,obs_u,obs_v,pred_u,pred_v,res_u,res_v,res_norm
        file << frame_idx << "," << camera.name() << "," << obs.position.x() << ","
             << obs.position.y() << "," << predicted.x() << "," << predicted.y() << ","
             << residual_u << "," << residual_v << "," << residual_norm << "\n";
    }
}

void export_state_vector(std::ofstream& file, int frame_idx, double timestamp, State const& state,
                         SkeletonLayout const& layout) {
    // Format matches Python: tracker_frame_idx,timestamp,root_position_x/y/z,
    // root_quaternion_w/x/y/z, root_velocity_x/y/z, root_angular_velocity_x/y/z,
    // joint_<name>_angle_<n>, joint_<name>_velocity_<n>

    file << frame_idx << "," << timestamp << ",";

    // Root position
    file << state.root_position().x() << "," << state.root_position().y() << ","
         << state.root_position().z() << ",";

    // Root orientation (quaternion: w,x,y,z)
    file << state.root_orientation().w() << "," << state.root_orientation().x() << ","
         << state.root_orientation().y() << "," << state.root_orientation().z() << ",";

    // Root velocity
    file << state.root_velocity().x() << "," << state.root_velocity().y() << ","
         << state.root_velocity().z() << ",";

    // Root angular velocity
    file << state.root_angular_velocity().x() << "," << state.root_angular_velocity().y() << ","
         << state.root_angular_velocity().z();

    // Joint angles and velocities (in layout order, using precomputed indices).
    // PRISMATIC leaders output the scale factor once (followers share the same slot, skip).
    for (auto const& desc : layout.joints()) {
        if (desc.is_scale_follower)
            continue;  // same state_index as leader, skip duplicate
        for (int i = 0; i < static_cast<int>(desc.storage_dof_count); ++i) {
            file << "," << state.joint_angles()[desc.state_index + i];
        }
        for (int i = 0; i < static_cast<int>(desc.storage_dof_count); ++i) {
            file << "," << state.joint_velocities()[desc.state_index + i];
        }
    }

    file << "\n";
}

std::string generate_state_header(SkeletonLayout const& layout) {
    std::string header = "tracker_frame_idx,timestamp,";
    header += "root_position_x,root_position_y,root_position_z,";
    header += "root_quaternion_w,root_quaternion_x,root_quaternion_y,root_quaternion_z,";
    header += "root_velocity_x,root_velocity_y,root_velocity_z,";
    header += "root_angular_velocity_x,root_angular_velocity_y,root_angular_velocity_z";

    // Joint angles and velocities (in layout order).
    // PRISMATIC leaders use the scale-group name; followers are skipped (same slot).
    for (auto const& desc : layout.joints()) {
        if (desc.is_scale_follower)
            continue;
        if (desc.type == JointType::PRISMATIC && !desc.scale_group.empty()) {
            // Scale-group leader: one column named by the group, no DOF index needed
            header += ",scale_group_" + desc.scale_group;
            header += ",scale_group_" + desc.scale_group + "_velocity";
        } else {
            // Normal joint: angle_0 / angle_1 / angle_2 + matching velocity columns
            for (int i = 0; i < static_cast<int>(desc.storage_dof_count); ++i) {
                header += ",joint_" + desc.name + "_angle_" + std::to_string(i);
            }
            for (int i = 0; i < static_cast<int>(desc.storage_dof_count); ++i) {
                header += ",joint_" + desc.name + "_velocity_" + std::to_string(i);
            }
        }
    }

    return header;
}

// Uses the SkeletonLayout (not skeleton.joints()) so that only active-group
// joints are written in the correct state-vector order.  Iterating skeleton.joints()
// would count non-active joints (e.g. heel.02.L/R) as if they have state DOFs,
// shifting all subsequent joints to wrong state indices.
void write_smoothed_joint_angles_frame(std::ofstream& file, int frame, double timestamp,
                                       State const& state, SkeletonLayout const& layout) {
    auto const& angles = state.joint_angles();
    auto const& vels = state.joint_velocities();
    for (auto const& desc : layout.joints()) {
        if (desc.is_scale_follower)
            continue;
        if (desc.type == JointType::SPHERICAL) {
            Eigen::Vector3d a = angles.segment<3>(desc.state_index);
            Eigen::Vector3d v = Eigen::Vector3d::Zero();
            if (desc.state_index + 2 < static_cast<uint32_t>(vels.size())) {
                v = vels.segment<3>(desc.state_index);
            }
            file << fmt::format("{},{},{},{},{},{},{},{},{}\n", frame, timestamp, desc.name, a.x(),
                                a.y(), a.z(), v.x(), v.y(), v.z());
        } else if (desc.type == JointType::REVOLUTE || desc.type == JointType::PRISMATIC) {
            double a = angles(desc.state_index);
            double v = (desc.state_index < static_cast<uint32_t>(vels.size()))
                           ? vels(desc.state_index)
                           : 0.0;
            file << fmt::format("{},{},{},{},{},{},{},{},{}\n", frame, timestamp, desc.name, a, 0.0,
                                0.0, v, 0.0, 0.0);
        }
    }
}

void write_smoothed_root_pose_frame(std::ofstream& file, int frame, double timestamp,
                                    State const& state) {
    auto const& p = state.root_position();
    Eigen::Quaterniond q = state.root_orientation().normalized();
    auto const& lv = state.root_velocity();
    auto const& av = state.root_angular_velocity();
    file << fmt::format("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n", frame, timestamp, p.x(),
                        p.y(), p.z(), q.w(), q.x(), q.y(), q.z(), lv.x(), lv.y(), lv.z(), av.x(),
                        av.y(), av.z());
}

// ---------------------------------------------------------------------------
// Per-person tracking pipeline
// ---------------------------------------------------------------------------

std::unique_ptr<PersonContext>
build_person_context(PersonSpec const& spec, BuildPersonContextOptions const& opts, bool verbose) {
    bool const quiet = opts.quiet;
    std::string const& db_path = opts.db_path;

    if (!quiet) {
        fmt::print("Opening session DB: {}\n", db_path);
    }
    SessionReader reader(db_path);

    // Resolve any prefix IDs to full UUIDs
    std::string full_sequence_id =
        reader.resolve_id("pose_observation_sequences", spec.sequence_id);
    std::string full_skeleton_id = reader.resolve_id("skeletons", spec.skeleton_id);
    std::string full_config_id = reader.resolve_id("tracker_configs", spec.config_id);

    auto ctx = std::make_unique<PersonContext>();
    ctx->spec = spec;

    // Load skeleton
    if (!quiet) {
        fmt::print("Loading skeleton '{}' from DB\n", full_skeleton_id);
    }
    std::string yaml_content = reader.load_skeleton_yaml(full_skeleton_id);
    ctx->skeleton = load_skeleton_from_yaml_string(yaml_content);
    if (!quiet) {
        fmt::print("  Loaded {} joints\n", ctx->skeleton.joints().size());
    }

    // Load tracker config
    if (!quiet) {
        fmt::print("Loading tracker config '{}' from DB\n", full_config_id);
    }
    auto db_cfg = reader.load_tracker_config(full_config_id);
    ctx->tracker_config = db_cfg.tracker;
    ctx->tracker_fps = db_cfg.tracker_fps;

    // Apply active_joint_groups override from CLI
    if (!opts.active_joint_groups.empty()) {
        ctx->tracker_config.active_joint_groups = opts.active_joint_groups;
    }
    if (opts.debug_init) {
        ctx->tracker_config.debug_init_frames = 1;
    }

    // Load sequence info
    auto seq_info = reader.load_sequence_info(full_sequence_id);

    // Load cameras (derives session/extrinsics/sync from the sequence record)
    if (!quiet) {
        fmt::print("Loading cameras for sequence '{}'\n", full_sequence_id);
    }
    ctx->cameras_by_name = reader.load_cameras_for_sequence(full_sequence_id);
    if (!quiet) {
        fmt::print("  Loaded {} cameras\n", ctx->cameras_by_name.size());
    }

    // Load sequence metadata (session_id, extrinsic_calibration_id, sync_config_id)
    auto seq_meta = reader.load_sequence_metadata(full_sequence_id);
    ctx->session_id = seq_meta.session_id;
    ctx->extrinsic_calibration_id = seq_meta.extrinsic_calibration_id;
    ctx->sync_config_id = seq_meta.sync_config_id;

    // Load observations
    if (!quiet) {
        fmt::print("Loading observations from sequence '{}'\n", full_sequence_id);
    }
    ctx->observations = reader.load_observations(
        full_sequence_id, ctx->cameras_by_name, ctx->skeleton, opts.min_confidence, spec.person_id,
        ctx->tracker_config.use_relative_observations, ctx->tracker_config.relative_min_confidence,
        ctx->tracker_config.pose_noise_std, ctx->tracker_config.cross_pair_max_px,
        ctx->tracker_config.cross_pair_max_n, ctx->tracker_config.edited_kp_noise_std);

    if (ctx->observations.empty()) {
        throw std::runtime_error("No observations found in sequence");
    }

    if (!quiet) {
        fmt::print("  Loaded {} observations across {} cameras\n",
                   ctx->observations.total_observations(), ctx->observations.camera_count());
    }

    // Determine effective end time
    double end_time =
        std::isnan(opts.override_end_time) ? seq_info.time_end_s : opts.override_end_time;
    if (end_time < 0.0) {
        end_time = ctx->observations.max_time();
    }

    // Determine effective start time.
    // If not explicitly overridden, auto-detect the first tracker step at which at least
    // min_cameras_for_init cameras have observations — so initialization has enough views.
    double dt = 1.0 / ctx->tracker_fps;
    double start_time;
    if (!std::isnan(opts.override_start_time)) {
        start_time = opts.override_start_time;
    } else {
        // Collect per-camera first-observation times and find the Nth smallest,
        // where N = min_cameras_for_init.
        int n_needed = ctx->tracker_config.min_cameras_for_init;
        std::vector<double> cam_starts;
        for (auto const& [name, seq] : ctx->observations.sequences()) {
            if (!seq.empty())
                cam_starts.push_back(seq.min_time());
        }
        std::sort(cam_starts.begin(), cam_starts.end());

        double seq_start = seq_info.time_start_s;
        if (static_cast<int>(cam_starts.size()) >= n_needed) {
            // Nth smallest start time: earliest point where N cameras have at least one
            // observation.  Find which tracker step window [seq_start + k*dt, seq_start +
            // (k+1)*dt) contains that observation using floor(), not ceil().  ceil() would
            // advance start_time by a full extra step for cameras whose first observation
            // falls anywhere inside the first window (e.g. 0.5 ms after seq_start), even
            // though those observations ARE available for initialization at seq_start.
            double t_n = cam_starts[static_cast<size_t>(n_needed - 1)];
            if (t_n > seq_start) {
                int steps_ahead = static_cast<int>(std::floor((t_n - seq_start) / dt));
                start_time = seq_start + steps_ahead * dt;
                if (!quiet && start_time > seq_start) {
                    fmt::print(
                        "  Auto-detected start time: {:.4f}s (sequence start {:.4f}s; "
                        "waiting for {} cameras)\n",
                        start_time, seq_start, n_needed);
                }
            } else {
                start_time = seq_start;
            }
        } else {
            start_time = seq_start;
        }
    }

    // Calculate tracker steps
    int num_steps = static_cast<int>((end_time - start_time) / dt);
    if (num_steps <= 0) {
        throw std::runtime_error(
            "No time steps to process - check time_start_s, time_end_s, and tracker_fps");
    }

    if (!quiet) {
        fmt::print("  Time range: [{:.3f}, {:.3f}) seconds\n", start_time, end_time);
        fmt::print("  Tracker sample rate: {:.1f} Hz (dt = {:.6f} s)\n", ctx->tracker_fps, dt);
        fmt::print("  Will process {} time steps\n", num_steps);
    }

    ctx->start_time = start_time;
    ctx->end_time = end_time;
    ctx->dt = dt;
    ctx->num_steps = num_steps;

    // Create output directory
    std::filesystem::create_directories(spec.output_dir);

    // Convert cameras to ID-keyed map for Tracker
    for (auto const& [name, cam] : ctx->cameras_by_name) {
        ctx->cameras_by_id.emplace(cam.id(), cam);
    }

    // Create result writer (writes tracking results to the session DB alongside CSV output)
    ctx->result_writer =
        std::make_unique<ResultWriter>(db_path, full_sequence_id, full_skeleton_id, full_config_id,
                                       seq_meta.extrinsic_calibration_id, seq_meta.sync_config_id,
                                       spec.person_id, ctx->cameras_by_name, ctx->skeleton);

    // Create tracker
    if (!quiet) {
        fmt::print("\nInitializing tracker...\n");
    }
    auto skeleton_ptr = std::make_shared<const Skeleton>(ctx->skeleton);
    ctx->tracker = std::make_unique<Tracker>(skeleton_ptr, ctx->cameras_by_id, ctx->tracker_config);

    if (opts.smooth_output) {
        ctx->tracker->enable_smoothing(true);
        if (!quiet) {
            fmt::print("RTS smoothing enabled: caching forward-pass data.\n");
        }
    }

    // Initialize from first-frame observations
    double t_first_window = start_time + dt;
    auto first_frame_obs = ctx->observations.get_all_in_range(start_time, t_first_window);
    bool initialized = ctx->tracker->initialize(first_frame_obs, start_time);
    if (initialized) {
        if (!quiet) {
            fmt::print("  IK initialization successful\n");
        }
    } else {
        fmt::print("  WARNING: IK initialization failed, falling back to rest pose\n");
        ctx->tracker->initialize_from_rest_pose(start_time);
    }

    if (!quiet) {
        fmt::print("  Initialization complete\n\n");
    }

    // Get FK from tracker
    ctx->fk = ctx->tracker->get_fk();
    if (!ctx->fk) {
        throw std::runtime_error("Failed to get FK from tracker");
    }

    // Create exporters
    ctx->layout =
        ctx->tracker_config.active_joint_groups.empty()
            ? SkeletonLayout::from_full_skeleton(skeleton_ptr)
            : SkeletonLayout::from_groups(skeleton_ptr, ctx->tracker_config.active_joint_groups);

    ctx->exporter = std::make_unique<TrackingExporter>(spec.output_dir, ctx->skeleton, *ctx->layout,
                                                       ctx->cameras_by_id);
    ctx->exporter->open();

    ctx->stats_tracker = std::make_unique<StatisticsTracker>();

    ctx->pred_obs_file.open(spec.output_dir / "predicted_observations.csv");
    ctx->pred_obs_file << "frame,camera,marker,obs_u,obs_v,pred_u,pred_v,res_u,res_v,res_norm\n";

    ctx->state_vec_file.open(spec.output_dir / "state_vectors.csv");
    ctx->state_vec_file << generate_state_header(*ctx->layout) << "\n";

    // Enable UKF debug output if requested
    if (opts.debug_output) {
        auto debug_dir = spec.output_dir / "debug" / ctx->result_writer->run_id();
        std::filesystem::create_directories(debug_dir);
        if (auto* ukf = ctx->tracker->get_ukf()) {
            ukf->enable_debug(true, debug_dir.string());
        }
        if (!quiet) {
            fmt::print("Debug output enabled: {}\n", debug_dir.string());
        }
    }

    if (!quiet) {
        fmt::print("Tracking:\n");
    }
    ctx->track_start_time = std::chrono::steady_clock::now();

    (void)verbose;
    return ctx;
}

void step_person_context_frame0(PersonContext& ctx) {
    double t_first_window = ctx.start_time + ctx.dt;
    auto frame_0_obs = ctx.observations.get_all_in_range(ctx.start_time, t_first_window);
    if (!frame_0_obs.empty()) {
        if (auto* ukf = ctx.tracker->get_ukf()) {
            ukf->set_frame_number(0);
        }
        double t_effective = ctx.start_time + ctx.dt * 0.5;
        auto result = ctx.tracker->track_frame(frame_0_obs, t_effective);
        if (result.tracking_lost) {
            fmt::print(stderr, "Warning: Tracking lost on first update\n");
        } else {
            ctx.frames_tracked++;
        }
    }
}

void step_person_context(PersonContext& ctx, int step, bool verbose, bool quiet) {
    double t_start = ctx.start_time + step * ctx.dt - ctx.dt / 2.0;
    double t_end = t_start + ctx.dt;

    if (auto* ukf = ctx.tracker->get_ukf()) {
        ukf->set_frame_number(step);
    }

    auto frame_obs = ctx.observations.get_all_in_range(t_start, t_end);

    if (frame_obs.empty()) {
        if (verbose) {
            fmt::print("  Step {}: t=[{:.3f}, {:.3f}): No observations, skipping\n", step, t_start,
                       t_end);
        }
        return;
    }

    double t_effective = t_start + ctx.dt / 2.0;

    if (ctx.pred_obs_file.is_open()) {
        export_predicted_observations(ctx.pred_obs_file, step + 1, t_effective, frame_obs,
                                      ctx.tracker->state(), ctx.fk, ctx.cameras_by_id,
                                      ctx.skeleton);
    }

    auto result = ctx.tracker->track_frame(frame_obs, t_effective);

    if (result.tracking_lost) {
        ctx.frames_lost++;
        if (verbose) {
            fmt::print("  Step {}: t=[{:.3f}, {:.3f}): Tracking LOST\n", step, t_start, t_end);
        }
    } else {
        ctx.frames_tracked++;
        if (verbose) {
            fmt::print("  Step {}: t=[{:.3f}, {:.3f}): {} inliers, {} outliers\n", step, t_start,
                       t_end, result.update_info.num_inliers, result.update_info.num_outliers);
        }
    }

    if (ctx.state_vec_file.is_open()) {
        export_state_vector(ctx.state_vec_file, step, t_effective, result.state, *ctx.layout);
    }

    {
        auto marker_positions_3d_map = ctx.fk->compute(result.state);
        std::map<std::string, Eigen::Vector3d> marker_positions_3d(marker_positions_3d_map.begin(),
                                                                   marker_positions_3d_map.end());
        ctx.exporter->write_frame(step, t_effective, result.state, marker_positions_3d, frame_obs,
                                  result.update_info);
    }

    double cov_cond = 0.0;
    if (result.covariance.size() > 0) {
        Eigen::VectorXd diag = result.covariance.diagonal();
        double dmax = diag.maxCoeff();
        double dmin = diag.minCoeff();
        if (dmin > 0.0)
            cov_cond = dmax / dmin;
    }
    ctx.result_writer->write_frame(step, t_effective, result.state.to_error_vector(),
                                   result.covariance, result.tracking_lost,
                                   result.update_info.num_inliers, cov_cond, result.update_info.nis,
                                   result.update_info.nis_dof);
    if (!result.update_info.observations.empty())
        ctx.result_writer->write_obs_results(step, result.update_info.observations);

    ctx.stats_tracker->add_frame_stats(
        step, t_effective, result.update_info, result.covariance, result.tracking_lost,
        result.predict_ms, result.update_ms, result.p_sigma_gen_ms, result.p_propagate_ms,
        result.p_mean_cov_ms, result.p_rts_ms, result.u_fk1_ms, result.u_s_ms, result.u_outlier_ms,
        result.u_fk2_ms, result.u_inlier_ms, result.u_kalman_ms, result.u_cov_update_ms);

    if (!quiet && !verbose && step % 10 == 0) {
        double percent = 100.0 * step / ctx.num_steps;
        auto elapsed = std::chrono::steady_clock::now() - ctx.track_start_time;
        double elapsed_sec =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
        double steps_per_sec = step / elapsed_sec;
        int eta_sec = static_cast<int>((ctx.num_steps - step) / steps_per_sec);
        fmt::print("  Progress: {}/{} ({:.1f}%) | {:.1f} steps/s | ETA: {}s\r", step, ctx.num_steps,
                   percent, steps_per_sec, eta_sec);
        std::cout.flush();
    }
}

void finalize_person_context(PersonContext& ctx, bool smooth_output, bool quiet, bool verbose) {
    if (!quiet && !verbose) {
        fmt::print("\n");
    }

    ctx.exporter->close();
    ctx.pred_obs_file.close();
    ctx.state_vec_file.close();

    // RTS smoother
    if (smooth_output) {
        if (!quiet) {
            fmt::print("Running RTS backward smoother...\n");
        }
        auto smoothed = ctx.tracker->smooth();

        {
            auto path = ctx.spec.output_dir / "smoothed_state_vectors.csv";
            std::ofstream f(path);
            f << generate_state_header(*ctx.layout) << "\n";
            int step = 1;
            for (auto const& sf : smoothed) {
                export_state_vector(f, step++, sf.timestamp, sf.state, *ctx.layout);
            }
        }
        {
            auto path = ctx.spec.output_dir / "smoothed_joint_angles.csv";
            std::ofstream f(path);
            f << "frame,timestamp,joint_name,angle_x,angle_y,angle_z,"
                 "velocity_x,velocity_y,velocity_z\n";
            int step = 1;
            for (auto const& sf : smoothed) {
                write_smoothed_joint_angles_frame(f, step++, sf.timestamp, sf.state, *ctx.layout);
            }
        }
        {
            auto path = ctx.spec.output_dir / "smoothed_root_pose.csv";
            std::ofstream f(path);
            f << "frame,timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,"
                 "vel_x,vel_y,vel_z,omega_x,omega_y,omega_z\n";
            int step = 1;
            for (auto const& sf : smoothed) {
                write_smoothed_root_pose_frame(f, step++, sf.timestamp, sf.state);
            }
        }

        // Write smoothed frames to DB
        {
            int step_idx = 1;
            for (auto const& sf : smoothed) {
                ctx.result_writer->write_smoothed_frame(step_idx++, sf.timestamp,
                                                        sf.state.to_error_vector(), sf.covariance);
            }
        }

        if (!quiet) {
            fmt::print("  Smoothed {} frames\n", smoothed.size());
        }
    }

    // Flush result writer and report run ID (always printed for machine parsing)
    ctx.result_writer->flush();
    fmt::print("tracking_run_id: {}\n", ctx.result_writer->run_id());

    // Write statistics
    ctx.stats_tracker->write_frame_stats(ctx.spec.output_dir / "tracking_stats.csv");

    nlohmann::json metadata;
    metadata["sequence_id"] = ctx.spec.sequence_id;
    metadata["skeleton_id"] = ctx.spec.skeleton_id;
    metadata["num_cameras"] = ctx.cameras_by_id.size();
    metadata["num_markers"] = ctx.skeleton.markers().size();
    metadata["start_time"] = ctx.start_time;
    metadata["end_time"] = ctx.end_time;
    metadata["num_steps"] = ctx.num_steps;
    metadata["tracker_fps"] = ctx.tracker_fps;
    ctx.stats_tracker->write_summary_stats(ctx.spec.output_dir / "overall_stats.json", metadata);

    // Final summary
    auto tracking_end = std::chrono::steady_clock::now();
    auto total_elapsed = tracking_end - ctx.track_start_time;
    double total_sec =
        std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count() / 1000.0;

    if (!quiet) {
        fmt::print("\nTracking complete!\n");
        fmt::print("  Tracked: {}/{} steps ({:.1f}%)\n", ctx.frames_tracked, ctx.num_steps,
                   100.0 * ctx.frames_tracked / ctx.num_steps);
        fmt::print("  Lost: {} steps\n", ctx.frames_lost);
        fmt::print("  Average rate: {:.1f} steps/s\n", ctx.frames_tracked / total_sec);
        fmt::print("  Total time: {:.1f}s\n", total_sec);
        if (auto* ukf = ctx.tracker->get_ukf()) {
            fmt::print("  PSD eigensolver fired: {}/{} frames\n", ukf->psd_fix_count(),
                       ctx.frames_tracked);
        }
        fmt::print("\nResults exported to: {}\n", ctx.spec.output_dir.string());
    }
}

// ---------------------------------------------------------------------------
// MultiPersonTracker
// ---------------------------------------------------------------------------

MultiPersonTracker::MultiPersonTracker(std::vector<PersonSpec> const& specs,
                                       BuildPersonContextOptions const& opts, bool verbose)
    : opts_(opts), verbose_(verbose) {
    persons_.reserve(specs.size());
    for (auto const& spec : specs) {
        persons_.push_back(build_person_context(spec, opts_, verbose_));
    }
}

void MultiPersonTracker::run() {
    for (auto& ctx : persons_) {
        step_person_context_frame0(*ctx);
    }

    int max_steps = 0;
    for (auto const& ctx : persons_) {
        max_steps = std::max(max_steps, ctx->num_steps);
    }

    // "for frame: for person" so Stage 2 can insert contact-gating/anchor-injection
    // logic once per frame, after every person completes that frame, without
    // restructuring this loop.
    for (int step = 1; step < max_steps; ++step) {
        for (auto& ctx : persons_) {
            if (step < ctx->num_steps) {
                step_person_context(*ctx, step, verbose_, opts_.quiet);
            }
        }
    }

    for (auto& ctx : persons_) {
        finalize_person_context(*ctx, opts_.smooth_output, opts_.quiet, verbose_);
    }
}

}  // namespace posetrak
