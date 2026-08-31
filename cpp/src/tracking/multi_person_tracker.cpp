// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/multi_person_tracker.hpp"

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "posetrak/db/session_reader.hpp"
#include "posetrak/filters/process_model.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "posetrak/tracking/hierarchical_solver.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace posetrak {

namespace {

/// @brief Wrap a covariance diagonal back into a diagonal MatrixXd, matching
/// what ResultWriter::write_frame()/write_smoothed_frame() expect (they only
/// ever read .diagonal() themselves, but keep the MatrixXd signature since
/// most callers pass a real UKF posterior covariance, not a diagonal-only one).
Eigen::MatrixXd diag_to_covariance_matrix(Eigen::VectorXd const& diag) {
    Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(diag.size(), diag.size());
    cov.diagonal() = diag;
    return cov;
}

}  // namespace

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

    // Initialize from first-frame observations. The auto-detect above (waiting for
    // min_cameras_for_init cameras to have *ever* seen anything) is a coarse heuristic --
    // it doesn't guarantee that exact window has enough *simultaneous* multi-camera
    // coverage for triangulation to actually succeed, and an explicit --start-time
    // override skips it entirely. Real multi-camera captures have sparse,
    // independently-timed per-camera detections (a marker-based-mocap object
    // especially -- see marker-mocap-design.md status.md's 2026-08-30 entry), so
    // search forward across a window rather than trying start_time once.
    constexpr double kInitSearchWindowS = 2.0;  // TODO: promote to a TrackerConfig field
    // if per-capture tuning turns out to matter (mirrors TrackerAppConfig's TOML-path
    // init_search_window_s, config.hpp).
    double const search_end = std::min(end_time, start_time + kInitSearchWindowS);
    double init_timestamp = start_time;
    bool initialized = false;
    for (double t = start_time; t < search_end; t += dt) {
        auto obs = ctx->observations.get_all_in_range(t, t + dt);
        if (ctx->tracker->initialize(obs, t)) {
            initialized = true;
            init_timestamp = t;
            break;
        }
    }

    if (initialized) {
        if (init_timestamp > start_time) {
            fmt::print(
                "  IK initialization successful at t={:.3f}s (searched forward {:.3f}s from "
                "the requested start time {:.3f}s -- no valid init window there)\n",
                init_timestamp, init_timestamp - start_time, start_time);
            start_time = init_timestamp;
            num_steps = static_cast<int>((end_time - start_time) / dt);
            if (num_steps <= 0) {
                throw std::runtime_error(
                    "No time steps left to process after the initialization search shifted "
                    "start_time forward -- check end_time");
            }
            ctx->start_time = start_time;
            ctx->num_steps = num_steps;
        } else if (!quiet) {
            fmt::print("  IK initialization successful\n");
        }
    } else if (ctx->skeleton.is_rigid_body()) {
        throw std::runtime_error(fmt::format(
            "Rigid-body initialization failed across the entire search window [{:.3f}, "
            "{:.3f}) -- no window there had enough camera coverage (>= 3 triangulated "
            "markers, >= {} cameras, and a non-collinear layout) for a valid Kabsch/Umeyama "
            "fit. A rest-pose fallback is meaningless for a free-floating prop (unlike an "
            "articulated skeleton), so refusing to proceed rather than track from a "
            "silently wrong pose. Try a later --start-time.",
            start_time, search_end, ctx->tracker_config.min_cameras_for_init));
    } else {
        fmt::print(
            "  WARNING: IK initialization failed across the search window [{:.3f}, {:.3f}), "
            "falling back to rest pose\n",
            start_time, search_end);
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

    // Existence-based hierarchical-mode toggle (see hierarchical_solver.hpp): a
    // tracker_config with any tracker_config_stages rows needs its DB rows kept
    // full-skeleton-width even though *ctx->layout* above may be group-scoped,
    // so a child stage can later merge into DOFs *ctx->layout* doesn't cover.
    if (!reader.load_tracker_config_stages(full_config_id).empty()) {
        ctx->full_layout = SkeletonLayout::from_full_skeleton(skeleton_ptr);
    }

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
            // Successful, non-lost track_frame() calls always push to the smoother
            // cache when smoothing is enabled -- see PersonContext::frame0_tracked.
            ctx.frame0_tracked = true;
        }
    }
}

void step_person_context(PersonContext& ctx, int step, bool verbose, bool quiet,
                         std::vector<Observation> const& extra_observations) {
    double t_start = ctx.start_time + step * ctx.dt - ctx.dt / 2.0;
    double t_end = t_start + ctx.dt;

    if (auto* ukf = ctx.tracker->get_ukf()) {
        ukf->set_frame_number(step);
    }

    auto frame_obs = ctx.observations.get_all_in_range(t_start, t_end);
    frame_obs.insert(frame_obs.end(), extra_observations.begin(), extra_observations.end());

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
    // Hierarchical mode: expand this person's group-scoped state AND covariance
    // diagonal to full-skeleton width before writing -- see PersonContext::
    // full_layout's doc comment. Child-owned DOF ranges get a placeholder
    // variance (matching state's rest-pose-default convention) until a child
    // stage's own merge (run_hierarchical_child_stages()) overwrites them with
    // real values -- see the design doc's cov_diag semantics note.
    Eigen::VectorXd const state_vec =
        ctx.full_layout ? expand_state_to_full_layout(result.state, *ctx.layout, *ctx.full_layout)
                              .to_error_vector()
                        : result.state.to_error_vector();
    Eigen::MatrixXd const cov_for_write =
        ctx.full_layout && result.covariance.size() > 0
            ? diag_to_covariance_matrix(expand_cov_diag_to_full_layout(
                  result.covariance.diagonal(), *ctx.layout, *ctx.full_layout,
                  ctx.tracker_config.init_joint_std * ctx.tracker_config.init_joint_std,
                  ctx.tracker_config.init_velocity_std * ctx.tracker_config.init_velocity_std))
            : result.covariance;
    ctx.result_writer->write_frame(step, t_effective, state_vec, cov_for_write,
                                   result.tracking_lost, result.update_info.num_inliers, cov_cond,
                                   result.update_info.nis, result.update_info.nis_dof);
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

        // smoothed[0] is step_person_context_frame0()'s untracked warm-up result when
        // that frame actually ran (ctx.frame0_tracked) -- it has no filtered-row
        // (is_smoothed=0) counterpart, since that frame's result is deliberately never
        // written to tracking_results/state_vectors.csv. Skip it here so smoothed
        // tracker_step N lines up with filtered tracker_step N instead of being off by
        // one (a real, previously-shipped bug: every smoothed row was mislabeled).
        auto const smoothed_begin =
            ctx.frame0_tracked && !smoothed.empty() ? smoothed.begin() + 1 : smoothed.begin();

        {
            auto path = ctx.spec.output_dir / "smoothed_state_vectors.csv";
            std::ofstream f(path);
            f << generate_state_header(*ctx.layout) << "\n";
            int step = 1;
            for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                export_state_vector(f, step++, it->timestamp, it->state, *ctx.layout);
            }
        }
        {
            auto path = ctx.spec.output_dir / "smoothed_joint_angles.csv";
            std::ofstream f(path);
            f << "frame,timestamp,joint_name,angle_x,angle_y,angle_z,"
                 "velocity_x,velocity_y,velocity_z\n";
            int step = 1;
            for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                write_smoothed_joint_angles_frame(f, step++, it->timestamp, it->state, *ctx.layout);
            }
        }
        {
            auto path = ctx.spec.output_dir / "smoothed_root_pose.csv";
            std::ofstream f(path);
            f << "frame,timestamp,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,"
                 "vel_x,vel_y,vel_z,omega_x,omega_y,omega_z\n";
            int step = 1;
            for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                write_smoothed_root_pose_frame(f, step++, it->timestamp, it->state);
            }
        }

        // Write smoothed frames to DB (same full-skeleton expansion as the forward
        // pass above -- see PersonContext::full_layout's doc comment).
        {
            int step_idx = 1;
            for (auto it = smoothed_begin; it != smoothed.end(); ++it) {
                Eigen::VectorXd const state_vec =
                    ctx.full_layout
                        ? expand_state_to_full_layout(it->state, *ctx.layout, *ctx.full_layout)
                              .to_error_vector()
                        : it->state.to_error_vector();
                Eigen::MatrixXd const cov_for_write =
                    ctx.full_layout && it->covariance.size() > 0
                        ? diag_to_covariance_matrix(expand_cov_diag_to_full_layout(
                              it->covariance.diagonal(), *ctx.layout, *ctx.full_layout,
                              ctx.tracker_config.init_joint_std * ctx.tracker_config.init_joint_std,
                              ctx.tracker_config.init_velocity_std *
                                  ctx.tracker_config.init_velocity_std))
                        : it->covariance;
                ctx.result_writer->write_smoothed_frame(step_idx++, it->timestamp, state_vec,
                                                        cov_for_write);
            }
        }

        ctx.smoothed_frames.assign(smoothed_begin, smoothed.end());

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
// Stage 2: contact gating + cross-person anchor construction (pure functions)
// ---------------------------------------------------------------------------

namespace {

struct Box {
    Eigen::Vector3d lo = Eigen::Vector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::Vector3d hi = Eigen::Vector3d::Constant(-std::numeric_limits<double>::infinity());
};

Box compute_box(std::map<std::string, Eigen::Vector3d> const& positions) {
    Box box;
    for (auto const& [name, pos] : positions) {
        (void)name;
        box.lo = box.lo.cwiseMin(pos);
        box.hi = box.hi.cwiseMax(pos);
    }
    return box;
}

/// True if two AABBs are within *margin* of each other (an inflate-and-overlap
/// test applied per axis).
bool boxes_within(Box const& a, Box const& b, double margin) {
    for (int axis = 0; axis < 3; ++axis) {
        if (a.lo[axis] > b.hi[axis] + margin)
            return false;
        if (b.lo[axis] > a.hi[axis] + margin)
            return false;
    }
    return true;
}

}  // namespace

void update_contact_pairs(std::vector<PersonGatingInput> const& persons,
                          std::map<ContactMarkerPair, double>& active_pairs) {
    size_t const n = persons.size();
    std::vector<Box> boxes(n);
    for (size_t i = 0; i < n; ++i) {
        if (persons[i].cross_person_max_world_mm > 0.0) {
            boxes[i] = compute_box(persons[i].marker_world_positions);
        }
    }

    for (size_t i = 0; i < n; ++i) {
        if (persons[i].cross_person_max_world_mm <= 0.0)
            continue;
        for (size_t j = i + 1; j < n; ++j) {
            if (persons[j].cross_person_max_world_mm <= 0.0)
                continue;

            double const threshold_m = std::min(persons[i].cross_person_max_world_mm,
                                                persons[j].cross_person_max_world_mm) /
                                       1000.0;
            double const exit_threshold_m = threshold_m * 1.2;

            // Use the exit (not enter) threshold as the bbox margin: gate 1 must
            // never be stricter than gate 2's hysteresis, or it would prematurely
            // drop pairs the marker-distance gate still wants to keep active.
            if (!boxes_within(boxes[i], boxes[j], exit_threshold_m)) {
                // Out of range even with margin: drop every active pair between i and j.
                for (auto it = active_pairs.begin(); it != active_pairs.end();) {
                    if (it->first.person_a == static_cast<int>(i) &&
                        it->first.person_b == static_cast<int>(j)) {
                        it = active_pairs.erase(it);
                    } else {
                        ++it;
                    }
                }
                continue;
            }

            for (auto const& [name_a, pos_a] : persons[i].marker_world_positions) {
                auto id_a_it = persons[i].marker_name_to_id.find(name_a);
                if (id_a_it == persons[i].marker_name_to_id.end())
                    continue;
                for (auto const& [name_b, pos_b] : persons[j].marker_world_positions) {
                    auto id_b_it = persons[j].marker_name_to_id.find(name_b);
                    if (id_b_it == persons[j].marker_name_to_id.end())
                        continue;

                    double const dist = (pos_a - pos_b).norm();
                    ContactMarkerPair key{static_cast<int>(i), id_a_it->second, static_cast<int>(j),
                                          id_b_it->second};
                    auto existing = active_pairs.find(key);
                    if (existing != active_pairs.end()) {
                        if (dist > exit_threshold_m) {
                            active_pairs.erase(existing);
                        } else {
                            existing->second = dist;
                        }
                    } else if (dist < threshold_m) {
                        active_pairs.emplace(key, dist);
                    }
                }
            }
        }
    }
}

std::vector<Observation> build_cross_person_anchors(
    int my_idx, int other_idx, std::map<ContactMarkerPair, double> const& active_pairs,
    std::vector<Observation> const& my_frame_obs, std::vector<Observation> const& other_frame_obs,
    std::map<std::string, Eigen::Vector3d> const& other_anchor_marker_positions,
    Skeleton const& other_skeleton, std::unordered_map<int, Camera> const& cameras,
    double my_min_confidence, double other_min_confidence, int max_n, double my_pose_noise_std,
    double other_pose_noise_std, double anchor_noise_std_floor, int frame_idx, double timestamp,
    std::unordered_map<int, std::unordered_map<int, double>> const&
        anchor_noise_std_by_camera_marker) {
    std::vector<Observation> result;

    // Marker id -> detections this frame (a marker can have one detection per
    // camera), for both people.
    std::unordered_map<int, std::vector<Observation const*>> my_by_marker;
    for (auto const& o : my_frame_obs)
        my_by_marker[o.marker_id].push_back(&o);
    std::unordered_map<int, std::vector<Observation const*>> other_by_marker;
    for (auto const& o : other_frame_obs)
        other_by_marker[o.marker_id].push_back(&o);

    // Which (my_marker, other_marker) pairs are active between these two people,
    // in canonical (min_idx, max_idx) form.
    std::vector<std::pair<int, int>> marker_pairs;  // (my_marker, other_marker)
    for (auto const& [pair, dist] : active_pairs) {
        (void)dist;
        if (pair.person_a == my_idx && pair.person_b == other_idx) {
            marker_pairs.emplace_back(pair.marker_a, pair.marker_b);
        } else if (pair.person_b == my_idx && pair.person_a == other_idx) {
            marker_pairs.emplace_back(pair.marker_b, pair.marker_a);
        }
    }

    struct Candidate {
        int camera_id;
        Observation const* mine;
        Observation const* other;
        int other_marker;
        double dist3d;
    };
    std::unordered_map<int, std::vector<Candidate>> by_camera;

    for (auto const& [my_marker, other_marker] : marker_pairs) {
        auto mit = my_by_marker.find(my_marker);
        auto oit = other_by_marker.find(other_marker);
        if (mit == my_by_marker.end() || oit == other_by_marker.end())
            continue;

        int const key_a = std::min(my_idx, other_idx);
        int const key_b = std::max(my_idx, other_idx);
        int const marker_key_a = (my_idx < other_idx) ? my_marker : other_marker;
        int const marker_key_b = (my_idx < other_idx) ? other_marker : my_marker;
        ContactMarkerPair key{key_a, marker_key_a, key_b, marker_key_b};
        auto dist_it = active_pairs.find(key);
        double const dist3d = (dist_it != active_pairs.end()) ? dist_it->second : 0.0;

        for (auto const* mo : mit->second) {
            if (mo->confidence < my_min_confidence)
                continue;
            for (auto const* oo : oit->second) {
                if (oo->confidence < other_min_confidence)
                    continue;
                if (mo->camera_id != oo->camera_id)
                    continue;
                by_camera[mo->camera_id].push_back({mo->camera_id, mo, oo, other_marker, dist3d});
            }
        }
    }

    for (auto& [camera_id, candidates] : by_camera) {
        std::sort(candidates.begin(), candidates.end(),
                  [](Candidate const& a, Candidate const& b) { return a.dist3d < b.dist3d; });
        if (max_n > 0 && static_cast<int>(candidates.size()) > max_n) {
            candidates.resize(static_cast<size_t>(max_n));
        }

        auto cam_it = cameras.find(camera_id);
        if (cam_it == cameras.end())
            continue;

        for (auto const& c : candidates) {
            auto const& other_marker_name = other_skeleton.markers()[c.other_marker].name;
            auto pos_it = other_anchor_marker_positions.find(other_marker_name);
            if (pos_it == other_anchor_marker_positions.end())
                continue;
            auto proj_opt =
                cam_it->second.project_undistorted(pos_it->second, /*clip_to_bounds=*/false);
            if (!proj_opt.has_value())
                continue;

            Observation anchor;
            anchor.camera_id = camera_id;
            anchor.marker_id = c.mine->marker_id;
            anchor.frame_idx = frame_idx;
            anchor.timestamp = timestamp;
            anchor.position = c.mine->position - c.other->position;
            anchor.position_distorted = c.mine->position_distorted - c.other->position_distorted;
            anchor.confidence = std::min(c.mine->confidence, c.other->confidence);
            anchor.mode = MeasurementMode::PAIR_DIFF;
            anchor.crop_scale = c.mine->crop_scale;
            anchor.anchor_position = *proj_opt;
            anchor.force_inlier =
                false;  // always subject to the outlier gate (identity-switch guard)

            double const sigma_pose_mine = my_pose_noise_std * c.mine->crop_scale;
            double const sigma_pose_other = other_pose_noise_std * c.other->crop_scale;

            // sigma_anchor: Stage 3's Jacobian-based per-marker uncertainty when
            // available for this (camera, marker), mildly inflated as a guard
            // against decentralized-fusion "data incest" (see the plan's
            // "measurement model" section); always floored at
            // anchor_noise_std_floor regardless, including when no Stage 3 value
            // was computed for this (camera, marker) at all.
            constexpr double kAnchorNoiseInflationFactor = 1.2;
            double sigma_anchor = anchor_noise_std_floor;
            auto cam_lookup = anchor_noise_std_by_camera_marker.find(camera_id);
            if (cam_lookup != anchor_noise_std_by_camera_marker.end()) {
                auto marker_lookup = cam_lookup->second.find(c.other_marker);
                if (marker_lookup != cam_lookup->second.end()) {
                    sigma_anchor = std::max(marker_lookup->second * kAnchorNoiseInflationFactor,
                                            anchor_noise_std_floor);
                }
            }

            anchor.noise_std_override =
                std::sqrt(sigma_pose_mine * sigma_pose_mine + sigma_pose_other * sigma_pose_other +
                          sigma_anchor * sigma_anchor);

            result.push_back(anchor);
        }
    }

    return result;
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

    marker_name_to_id_.resize(persons_.size());
    for (size_t i = 0; i < persons_.size(); ++i) {
        auto const& markers = persons_[i]->skeleton.markers();
        for (size_t m = 0; m < markers.size(); ++m) {
            marker_name_to_id_[i][markers[m].name] = static_cast<int>(m);
        }
    }
}

void MultiPersonTracker::update_contact_gate() {
    std::vector<PersonGatingInput> inputs(persons_.size());
    for (size_t i = 0; i < persons_.size(); ++i) {
        auto& ctx = *persons_[i];
        inputs[i].cross_person_max_world_mm = ctx.tracker_config.cross_person_max_world_mm;
        inputs[i].marker_name_to_id = marker_name_to_id_[i];
        if (ctx.tracker_config.cross_person_max_world_mm > 0.0) {
            auto positions = ctx.fk->compute(ctx.tracker->state());
            inputs[i].marker_world_positions.insert(positions.begin(), positions.end());
        }
    }
    update_contact_pairs(inputs, active_contact_pairs_);
}

std::vector<Observation>
MultiPersonTracker::build_anchor_observations(int idx, int step,
                                              std::vector<char> const& processed_this_frame) {
    std::vector<Observation> result;
    auto& ctx = *persons_[idx];
    if (ctx.tracker_config.cross_person_max_world_mm <= 0.0)
        return result;

    double t_start = ctx.start_time + step * ctx.dt - ctx.dt / 2.0;
    double t_end = t_start + ctx.dt;
    double t_effective = t_start + ctx.dt / 2.0;
    auto my_frame_obs = ctx.observations.get_all_in_range(t_start, t_end);

    // Which other persons does the active set reference for idx? Group so each
    // other person's frame_obs/anchor-state is fetched/computed at most once.
    std::vector<int> other_idxs;
    for (auto const& [pair, dist] : active_contact_pairs_) {
        (void)dist;
        if (pair.person_a == idx &&
            std::find(other_idxs.begin(), other_idxs.end(), pair.person_b) == other_idxs.end()) {
            other_idxs.push_back(pair.person_b);
        } else if (pair.person_b == idx && std::find(other_idxs.begin(), other_idxs.end(),
                                                     pair.person_a) == other_idxs.end()) {
            other_idxs.push_back(pair.person_a);
        }
    }

    // Stage 2 placeholder for sigma_anchor: Stage 3 replaces this constant with a
    // real per-marker Jacobian-based projected-uncertainty value (see
    // phase5-cross-person-plan.md, "Per-marker anchor uncertainty").
    constexpr double kAnchorNoiseStdFloorPx = 5.0;

    for (int other_idx : other_idxs) {
        auto& other_ctx = *persons_[other_idx];
        if (step >= other_ctx.num_steps)
            continue;

        // Anchor freshness: current-frame posterior if already stepped this
        // frame, else a one-frame constant-velocity extrapolation of their
        // frame-(t-1) posterior (see plan's "Anchor freshness").
        State anchor_state = other_ctx.tracker->state();
        if (static_cast<size_t>(other_idx) >= processed_this_frame.size() ||
            !processed_this_frame[static_cast<size_t>(other_idx)]) {
            ConstantVelocityModel extrapolation_model(other_ctx.layout);
            anchor_state = extrapolation_model.propagate(anchor_state, other_ctx.dt);
        }
        auto anchor_positions_map = other_ctx.fk->compute(anchor_state);
        std::map<std::string, Eigen::Vector3d> anchor_positions(anchor_positions_map.begin(),
                                                                anchor_positions_map.end());

        auto other_frame_obs = other_ctx.observations.get_all_in_range(t_start, t_end);

        // Stage 3: per-(camera, marker) anchor uncertainty, lazily computed only
        // for the markers this active-pair set actually references for
        // other_idx, and only for cameras other_idx has. Each marker_projection_std()
        // call amortizes the FK/Pinocchio-Jacobian setup across every marker
        // requested for that camera, so this is one such call per camera, not
        // per marker.
        std::vector<int> other_markers_needed;
        for (auto const& [pair, dist] : active_contact_pairs_) {
            (void)dist;
            if (pair.person_a == idx && pair.person_b == other_idx) {
                other_markers_needed.push_back(pair.marker_b);
            } else if (pair.person_b == idx && pair.person_a == other_idx) {
                other_markers_needed.push_back(pair.marker_a);
            }
        }
        std::unordered_map<int, std::unordered_map<int, double>> anchor_noise_std_by_camera_marker;
        if (!other_markers_needed.empty()) {
            for (auto const& [camera_id, camera] : other_ctx.cameras_by_id) {
                (void)camera;
                anchor_noise_std_by_camera_marker[camera_id] =
                    other_ctx.tracker->marker_projection_std(camera_id, other_markers_needed);
            }
        }

        int max_n = std::min(ctx.tracker_config.cross_person_max_n,
                             other_ctx.tracker_config.cross_person_max_n);
        auto anchors = build_cross_person_anchors(
            idx, other_idx, active_contact_pairs_, my_frame_obs, other_frame_obs, anchor_positions,
            other_ctx.skeleton, ctx.cameras_by_id, ctx.tracker_config.cross_person_min_confidence,
            other_ctx.tracker_config.cross_person_min_confidence, max_n,
            ctx.tracker_config.pose_noise_std, other_ctx.tracker_config.pose_noise_std,
            kAnchorNoiseStdFloorPx, step, t_effective, anchor_noise_std_by_camera_marker);
        result.insert(result.end(), anchors.begin(), anchors.end());
    }

    return result;
}

void MultiPersonTracker::run() {
    for (auto& ctx : persons_) {
        step_person_context_frame0(*ctx);
    }
    update_contact_gate();

    int max_steps = 0;
    for (auto const& ctx : persons_) {
        max_steps = std::max(max_steps, ctx->num_steps);
    }

    std::vector<int> order(persons_.size());
    std::iota(order.begin(), order.end(), 0);

    // "for frame: for person", per Stage 1's shape -- Stage 2 fills in the
    // per-frame contact-gate refresh and per-person anchor injection.
    for (int step = 1; step < max_steps; ++step) {
        // Rotate the processing order every frame so the "who goes first only
        // sees stale anchors" asymmetry doesn't consistently favor one person.
        if (!order.empty())
            std::rotate(order.begin(), order.begin() + 1, order.end());

        std::vector<char> processed_this_frame(persons_.size(), 0);
        for (int idx : order) {
            auto& ctx = *persons_[static_cast<size_t>(idx)];
            if (step >= ctx.num_steps) {
                processed_this_frame[static_cast<size_t>(idx)] = 1;
                continue;
            }
            auto anchors = build_anchor_observations(idx, step, processed_this_frame);
            step_person_context(ctx, step, verbose_, opts_.quiet, anchors);
            processed_this_frame[static_cast<size_t>(idx)] = 1;
        }

        update_contact_gate();
    }

    for (auto& ctx : persons_) {
        finalize_person_context(*ctx, opts_.smooth_output, opts_.quiet, verbose_);
    }
}

}  // namespace posetrak
