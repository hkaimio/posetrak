#include <CLI/CLI.hpp>
#include <fmt/core.h>

#include "posetrak/core/config.hpp"
#include "posetrak/io/camera_loader.hpp"
#include "posetrak/io/observation_loader.hpp"
#include "posetrak/io/skeleton_loader.hpp"
#include "posetrak/io/statistics_tracker.hpp"
#include "posetrak/io/sync_loader.hpp"
#include "posetrak/io/tracking_export.hpp"
#include "posetrak/kinematics/forward_kinematics.hpp"
#include "posetrak/kinematics/pinocchio_model_builder.hpp"
#include "posetrak/tracking/tracker.hpp"  // Must be before config.hpp for inline function
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace posetrak;

// Helper: Convert camera map with string keys to int keys
std::unordered_map<int, Camera>
convert_camera_map(std::unordered_map<std::string, Camera> const& cameras_by_name,
                   std::unordered_map<std::string, int>& name_to_id) {
    std::unordered_map<int, Camera> cameras_by_id;
    int next_id = 0;
    for (auto const& [name, cam] : cameras_by_name) {
        name_to_id[name] = next_id;
        cameras_by_id.emplace(next_id, cam);
        next_id++;
    }
    return cameras_by_id;
}

// Helper: Update observation camera IDs from names
void update_observation_camera_ids(ObservationSet& obs_set,
                                   std::unordered_map<std::string, int> const& name_to_id) {
    for (auto& [cam_name, sequence] :
         const_cast<std::map<std::string, ObservationSequence>&>(obs_set.sequences())) {
        auto it = name_to_id.find(cam_name);
        if (it != name_to_id.end()) {
            sequence.camera_id = it->second;
            for (auto& obs : sequence.observations) {
                obs.camera_id = it->second;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    CLI::App app{"Posetrak - Motion Capture Tracker"};

    std::string config_path;
    app.add_option("config", config_path, "Configuration file (TOML)")
        ->required()
        ->check(CLI::ExistingFile);

    bool verbose = false;
    app.add_flag("-v,--verbose", verbose, "Verbose output (show per-frame statistics)");

    bool quiet = false;
    app.add_flag("-q,--quiet", quiet, "Quiet mode (only show errors)");

    CLI11_PARSE(app, argc, argv);

    try {
        // Load configuration
        if (!quiet) {
            fmt::print("Loading configuration: {}\n", config_path);
        }
        auto config = TrackerAppConfig::load(config_path);
        config.validate();

        // Load skeleton
        if (!quiet) {
            fmt::print("Loading skeleton: {}\n", config.skeleton_path.string());
        }
        auto skeleton = load_skeleton_from_yaml(config.skeleton_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} joints\n", skeleton.joints().size());
        }

        // Load cameras
        if (!quiet) {
            fmt::print("Loading cameras: {}\n", config.cameras_path.string());
        }
        auto cameras_by_name = load_cameras_from_toml(config.cameras_path.string());
        if (!quiet) {
            fmt::print("  Loaded {} cameras\n", cameras_by_name.size());
        }

        // Convert camera map to use integer IDs
        std::unordered_map<std::string, int> camera_name_to_id;
        auto cameras = convert_camera_map(cameras_by_name, camera_name_to_id);

        // Load sync (optional)
        if (config.sync_path) {
            if (!quiet) {
                fmt::print("Loading sync: {}\n", config.sync_path->string());
            }
            auto sync_data = load_sync_metadata(config.sync_path->string());
            // Apply sync to cameras (modifies cameras_by_name in-place)
            apply_sync_metadata(cameras_by_name, sync_data, false);
            // Re-convert to int keys
            cameras = convert_camera_map(cameras_by_name, camera_name_to_id);
        }

        // Load observations
        if (!quiet) {
            fmt::print("Loading observations: {}\n", config.observations_dir.string());
        }

        uint32_t end_frame =
            config.max_frames < 0 ? UINT32_MAX : config.start_frame + config.max_frames;
        auto observations_set =
            load_openpose_sequence(config.observations_dir.string(), cameras_by_name, skeleton,
                                   {config.start_frame, end_frame}, 0.1, config.person_id);

        // Update observation camera IDs to match the integer IDs
        update_observation_camera_ids(observations_set, camera_name_to_id);

        // Determine frame range from observations
        int start_frame = config.start_frame;
        int num_frames_to_process = 0;

        // Get unique timestamps
        auto timestamps = observations_set.get_unique_timestamps();
        if (timestamps.empty()) {
            throw std::runtime_error("No observations found");
        }

        num_frames_to_process = static_cast<int>(timestamps.size());
        if (config.max_frames > 0) {
            num_frames_to_process = std::min(num_frames_to_process, config.max_frames);
        }

        if (!quiet) {
            fmt::print("  Found {} timesteps\n", timestamps.size());
            fmt::print("  Will process {} frames starting at frame {}\n", num_frames_to_process,
                       start_frame);
        }

        // Create tracker
        if (!quiet) {
            fmt::print("\nInitializing tracker...\n");
        }

        auto tracker_config = config.to_tracker_config();
        Tracker tracker(skeleton, cameras, tracker_config);

        // Initialize from rest pose (IK initialization is currently broken)
        if (!quiet) {
            fmt::print("  Initializing from skeleton rest pose (IK disabled)\n");
        }

        tracker.initialize_from_rest_pose(timestamps[0]);

        if (!quiet) {
            fmt::print("  Initialization successful\n\n");
        }

        // Create FK for computing marker positions
        pinocchio::Model model;
        pinocchio::Data data;
        PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
        auto marker_frame_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
        ForwardKinematics fk(model, data, marker_frame_map, skeleton);

        // Create exporters
        std::unique_ptr<TrackingExporter> exporter;
        std::unique_ptr<StatisticsTracker> stats_tracker;

        if (config.export_tracking_results) {
            exporter = std::make_unique<TrackingExporter>(config.output_dir, skeleton, cameras);
            exporter->open();
        }

        if (config.export_statistics) {
            stats_tracker = std::make_unique<StatisticsTracker>();
        }

        // Track sequence
        if (!quiet) {
            fmt::print("Tracking:\n");
        }

        auto start_time = std::chrono::steady_clock::now();
        int frames_tracked = 0;
        int frames_lost = 0;

        for (int i = 1; i < num_frames_to_process; ++i) {
            double frame_timestamp = timestamps[i];

            // Get observations at this time
            auto frame_obs = observations_set.get_all_at_time(frame_timestamp);

            if (frame_obs.empty()) {
                if (verbose) {
                    fmt::print("  t={:.3f}: No observations, skipping\n", frame_timestamp);
                }
                continue;
            }

            // Track
            auto result = tracker.track_frame(frame_obs, frame_timestamp);

            if (result.tracking_lost) {
                frames_lost++;
                if (verbose) {
                    fmt::print("  t={:.3f}: Tracking LOST ({} obs)\n", frame_timestamp,
                               result.update_info.num_observations);
                }
            } else {
                frames_tracked++;
                if (verbose) {
                    fmt::print("  t={:.3f}: {} inliers, {} outliers\n", frame_timestamp,
                               result.update_info.num_inliers, result.update_info.num_outliers);
                }
            }

            // Export
            if (exporter) {
                // Compute marker positions using FK
                auto marker_positions_3d_map = fk.compute(result.state);

                // Convert to std::map (exporter expects std::map, not unordered_map)
                std::map<std::string, Eigen::Vector3d> marker_positions_3d(
                    marker_positions_3d_map.begin(), marker_positions_3d_map.end());

                exporter->write_frame(start_frame + i, frame_timestamp, result.state,
                                      marker_positions_3d, frame_obs, result.update_info);
            }

            if (stats_tracker) {
                stats_tracker->add_frame_stats(start_frame + i, frame_timestamp, result.update_info,
                                               result.covariance, result.tracking_lost);
            }

            // Progress indicator
            if (!quiet && !verbose && i % 10 == 0) {
                double percent = 100.0 * i / num_frames_to_process;
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                double elapsed_sec =
                    std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() / 1000.0;
                double fps = i / elapsed_sec;
                int eta_sec = static_cast<int>((num_frames_to_process - i) / fps);

                fmt::print("  Progress: {}/{} ({:.1f}%) | {:.1f} fps | ETA: {}s\r", i,
                           num_frames_to_process, percent, fps, eta_sec);
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

        // Write statistics
        if (stats_tracker) {
            stats_tracker->write_frame_stats(config.output_dir / "tracking_stats.csv");

            nlohmann::json metadata;
            metadata["sequence_name"] = config.observations_dir.filename().string();
            metadata["skeleton_file"] = config.skeleton_path.filename().string();
            metadata["num_cameras"] = cameras.size();
            metadata["num_markers"] = skeleton.markers().size();
            metadata["start_frame"] = start_frame;
            metadata["num_frames"] = num_frames_to_process;
            metadata["config"] = {
                {"process_noise_std", config.process_noise_std},
                {"measurement_noise_std", config.measurement_noise_std},
                {"outlier_threshold", config.outlier_threshold},
                {"ukf_alpha", config.ukf_alpha},
            };

            stats_tracker->write_summary_stats(config.output_dir / "overall_stats.json", metadata);
        }

        // Final summary
        auto end_time = std::chrono::steady_clock::now();
        auto total_elapsed = end_time - start_time;
        double total_sec =
            std::chrono::duration_cast<std::chrono::milliseconds>(total_elapsed).count() / 1000.0;
        double avg_fps = frames_tracked / total_sec;

        if (!quiet) {
            fmt::print("\nTracking complete!\n");
            fmt::print("  Tracked: {}/{} frames ({:.1f}%)\n", frames_tracked, num_frames_to_process,
                       100.0 * frames_tracked / num_frames_to_process);
            fmt::print("  Lost: {} frames\n", frames_lost);
            fmt::print("  Average FPS: {:.1f}\n", avg_fps);
            fmt::print("  Total time: {:.1f}s\n", total_sec);

            if (stats_tracker) {
                fmt::print("  Mean reprojection error: {:.2f}px\n",
                           stats_tracker->mean_reprojection_error());
                fmt::print("  Outlier rate: {:.1f}%\n", stats_tracker->outlier_rate() * 100.0);
            }

            fmt::print("\nResults exported to: {}\n", config.output_dir.string());
            if (config.export_tracking_results) {
                fmt::print("  - tracking_results.csv\n");
                fmt::print("  - joint_angles.csv\n");
                fmt::print("  - root_pose.csv\n");
                fmt::print("  - marker_projections.csv\n");
                fmt::print("  - observations.csv\n");
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
