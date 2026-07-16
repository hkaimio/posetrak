/**
 * Stage 1 regression test for the cross-person relative observations plan
 * (docs/roadmap/features/error-improvements/phase5-cross-person-plan.md):
 * before any cross-person coupling exists, tracking N people through
 * MultiPersonTracker must produce output that is bitwise-identical to
 * tracking each person through the single-person pipeline
 * (build_person_context/step_person_context/finalize_person_context, the
 * same functions run_track_from_db() itself now calls) separately.
 *
 * Builds a small self-contained session DB (3 cameras, a real trackable
 * skeleton, a handful of frames of synthetic observations duplicated across
 * person_id 0 and 1) and compares final tracker state/covariance between a
 * direct single-person run and MultiPersonTracker's run for the same person.
 */
#include <posetrak/core/skeleton.hpp>
#include <posetrak/io/skeleton_loader.hpp>
#include <posetrak/kinematics/forward_kinematics.hpp>
#include <posetrak/kinematics/pinocchio_model_builder.hpp>
#include <posetrak/tracking/multi_person_tracker.hpp>

#include <catch2/catch_test_macros.hpp>
#include <sqlite3.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace posetrak;
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// SQL helpers (same pattern as test_session_reader.cpp)
// ---------------------------------------------------------------------------

std::vector<uint8_t> encode_float64_blob(std::vector<double> const& vals) {
    std::vector<uint8_t> out(vals.size() * sizeof(double));
    std::memcpy(out.data(), vals.data(), out.size());
    return out;
}

std::vector<uint8_t> encode_float32_blob(std::vector<float> const& vals) {
    std::vector<uint8_t> out(vals.size() * sizeof(float));
    std::memcpy(out.data(), vals.data(), out.size());
    return out;
}

void exec_sql(sqlite3* db, std::string const& sql) {
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string msg = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw std::runtime_error("SQL error: " + msg + "\nSQL: " + sql);
    }
}

void bind_blob_and_step(sqlite3* db, std::string const& sql,
                        std::vector<std::vector<uint8_t>> const& blobs) {
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("SQL prepare failed: " + sql);
    }
    for (size_t i = 0; i < blobs.size(); ++i) {
        sqlite3_bind_blob(stmt, static_cast<int>(i) + 1, blobs[i].data(),
                          static_cast<int>(blobs[i].size()), SQLITE_STATIC);
    }
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::string err = sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
        throw std::runtime_error("SQL step failed: " + err);
    }
    sqlite3_finalize(stmt);
}

// ---------------------------------------------------------------------------
// Synthetic camera rig + trajectory (adapted from test_tracker_integration.cpp's
// TrackerIntegrationFixture -- same semi-circle rig, same sinusoidal ground
// truth, but here the projected 2D observations get written into a session DB
// instead of being fed to a Tracker directly).
// ---------------------------------------------------------------------------

struct SyntheticCamera {
    Camera camera;         // in-memory Camera (for projecting ground truth)
    Eigen::Matrix3d R_wc;  // world-to-camera rotation (DB extrinsics convention)
    Eigen::Vector3d t_wc;  // world-to-camera translation (DB extrinsics convention)
};

std::vector<SyntheticCamera> make_synthetic_cameras(int num_cameras = 3, double radius = 4.0,
                                                    double height = 1.5) {
    std::vector<SyntheticCamera> out;
    for (int i = 0; i < num_cameras; ++i) {
        double angle = M_PI * static_cast<double>(i) / static_cast<double>(num_cameras - 1);
        Eigen::Vector3d pos(radius * std::cos(angle), radius * std::sin(angle), height);

        Eigen::Vector3d target(0, 0, height);
        Eigen::Vector3d look_dir = (target - pos).normalized();
        Eigen::Vector3d up(0, 0, 1);
        Eigen::Vector3d right = look_dir.cross(up).normalized();
        up = right.cross(look_dir).normalized();

        Eigen::Matrix3d R_cam_to_world;
        R_cam_to_world.col(0) = right;
        R_cam_to_world.col(1) = -up;
        R_cam_to_world.col(2) = look_dir;
        Eigen::Matrix3d R_wc = R_cam_to_world.transpose();  // world-to-camera

        Intrinsics intr;
        intr.fx = 600.0;
        intr.fy = 600.0;
        intr.cx = 640.0;
        intr.cy = 360.0;
        intr.width = 1280;
        intr.height = 720;
        intr.model = Intrinsics::DistortionModel::BrownConrady;
        intr.distortion_coeffs = {0, 0, 0, 0, 0};

        Extrinsics extr;
        extr.position = pos;
        extr.orientation = Eigen::Quaterniond(R_wc);

        SyntheticCamera sc{Camera(i, "cam" + std::to_string(i), intr, extr), R_wc, -R_wc * pos};
        out.push_back(std::move(sc));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Fixture DB construction
// ---------------------------------------------------------------------------

/// Builds a self-contained session DB at *path*: 3 cameras, the real
/// tests/data/simple_humanoid.yaml skeleton, one tracker config, one
/// pose_observation_sequence, and *num_frames* frames of synthetic
/// observations for both person_id 0 and 1 (identical data -- two
/// independent trackers should therefore also agree with each other, as an
/// extra determinism sanity check beyond the direct-vs-orchestrated
/// comparison the tests below make).
void create_fixture_db(fs::path const& path, int num_frames, double dt) {
    if (fs::exists(path))
        fs::remove(path);

    sqlite3* db = nullptr;
    if (sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                        nullptr) != SQLITE_OK) {
        throw std::runtime_error("Could not create fixture DB");
    }
    exec_sql(db, "PRAGMA foreign_keys=ON;");

    // ---- Schema (registry-style + session tables + tracking-results tables,
    // matching the exact columns SessionReader/ResultWriter read and write) ----
    exec_sql(db, R"(
        CREATE TABLE camera_models (
            id TEXT PRIMARY KEY, manufacturer TEXT, model_name TEXT, sensor_size TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE camera_modes (
            id TEXT PRIMARY KEY, camera_model_id TEXT NOT NULL REFERENCES camera_models(id),
            width_px INTEGER NOT NULL DEFAULT 0, height_px INTEGER NOT NULL DEFAULT 0,
            nominal_fps REAL NOT NULL DEFAULT 0.0, codec TEXT, notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE camera_instances (
            id TEXT PRIMARY KEY, camera_model_id TEXT NOT NULL REFERENCES camera_models(id),
            serial_number TEXT, label TEXT NOT NULL
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE intrinsics_calibrations (
            id TEXT PRIMARY KEY, camera_mode_id TEXT NOT NULL REFERENCES camera_modes(id),
            calibrated_at TEXT NOT NULL, calibration_tool TEXT,
            distortion_model TEXT NOT NULL DEFAULT 'radtan',
            fx REAL NOT NULL, fy REAL NOT NULL, cx REAL NOT NULL, cy REAL NOT NULL,
            dist_coeffs BLOB, rms_error REAL, notes TEXT, matrix_original BLOB
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE skeletons (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, parent_id TEXT REFERENCES skeletons(id),
            person_label TEXT, source TEXT, yaml_content TEXT NOT NULL,
            created_at TEXT NOT NULL, notes TEXT
        );
    )");
    // Mirrors db/registry_schema.sql's tracker_configs (kept in sync manually --
    // SessionReader::load_tracker_config() selects every column by name).
    exec_sql(db, R"(
        CREATE TABLE tracker_configs (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, parent_id TEXT REFERENCES tracker_configs(id),
            created_at TEXT NOT NULL,
            alpha REAL, beta REAL, kappa REAL,
            process_noise_std REAL, process_noise_vel_std REAL, velocity_half_life_s REAL,
            measurement_noise_std REAL, outlier_threshold REAL, tracker_fps REAL,
            ik_max_iterations INTEGER, ik_tolerance REAL,
            init_position_std REAL, init_orientation_std REAL, init_joint_std REAL,
            init_velocity_std REAL, min_cameras_for_init INTEGER,
            velocity_mode_camera_ids TEXT, velocity_measurement_noise_std REAL, notes TEXT,
            pose_noise_std REAL, use_relative_observations INTEGER, relative_min_confidence REAL,
            cross_pair_max_px REAL, cross_pair_max_n INTEGER,
            process_noise_vel_gain_joint REAL, process_noise_vel_ref_joint REAL,
            process_noise_vel_gain_root REAL, process_noise_vel_ref_root REAL,
            process_noise_vel_joint_names TEXT,
            pose_reg_joint_names TEXT, pose_reg_equal_split_noise_std REAL,
            pose_reg_rest_pose_noise_std REAL,
            nis_feedback_scopes TEXT, nis_feedback_window INTEGER, nis_feedback_threshold REAL,
            nis_feedback_max_multiplier REAL, process_noise_vel_scopes TEXT,
            soft_limit_joint_names TEXT, soft_limit_margin_rad REAL, soft_limit_noise_std REAL,
            near_limit_damping_joint_names TEXT, near_limit_margin_rad REAL,
            near_limit_spread_sigma REAL, near_limit_damping_factor REAL,
            edited_kp_noise_std REAL,
            cross_person_max_world_mm REAL, cross_person_min_confidence REAL,
            cross_person_max_n INTEGER
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE mocap_sessions (
            id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, location TEXT, notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE session_cameras (
            session_id TEXT NOT NULL REFERENCES mocap_sessions(id),
            camera_instance_id TEXT NOT NULL, label TEXT,
            PRIMARY KEY (session_id, camera_instance_id)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE extrinsic_calibrations (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES mocap_sessions(id),
            calibrated_at TEXT NOT NULL, method TEXT, rms_error REAL
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE extrinsic_entries (
            extrinsic_calibration_id TEXT NOT NULL REFERENCES extrinsic_calibrations(id),
            camera_instance_id TEXT NOT NULL, R BLOB NOT NULL, t BLOB NOT NULL,
            PRIMARY KEY (extrinsic_calibration_id, camera_instance_id)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE captures (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES mocap_sessions(id),
            extrinsic_calibration_id TEXT NOT NULL REFERENCES extrinsic_calibrations(id),
            capture_number INTEGER NOT NULL, label TEXT, notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE capture_videos (
            id TEXT PRIMARY KEY, shot_id TEXT NOT NULL REFERENCES captures(id),
            camera_instance_id TEXT NOT NULL, file_path TEXT NOT NULL,
            first_video_frame INTEGER NOT NULL, last_video_frame INTEGER NOT NULL,
            actual_fps REAL NOT NULL, camera_mode_id TEXT, intrinsics_calibration_id TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE sync_configs (
            id TEXT PRIMARY KEY, shot_id TEXT NOT NULL REFERENCES captures(id),
            created_by TEXT, notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE sync_points (
            sync_config_id TEXT NOT NULL REFERENCES sync_configs(id),
            camera_instance_id TEXT NOT NULL, shot_video_id TEXT NOT NULL REFERENCES capture_videos(id),
            video_frame INTEGER NOT NULL, timestamp_s REAL NOT NULL,
            PRIMARY KEY (sync_config_id, camera_instance_id)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE pose_observation_sequences (
            id TEXT PRIMARY KEY, shot_id TEXT NOT NULL REFERENCES captures(id),
            sync_config_id TEXT NOT NULL REFERENCES sync_configs(id),
            time_start_s REAL NOT NULL, time_end_s REAL NOT NULL,
            pose_model TEXT, notes TEXT,
            pixels_are_undistorted INTEGER NOT NULL DEFAULT 1, detection_run_id TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE pose_observations (
            sequence_id TEXT NOT NULL REFERENCES pose_observation_sequences(id),
            camera_instance_id TEXT NOT NULL, video_frame INTEGER NOT NULL,
            timestamp_s REAL NOT NULL, person_id INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'body', detection_run_id TEXT,
            kp_blob BLOB NOT NULL, noise_scale REAL,
            PRIMARY KEY (sequence_id, camera_instance_id, video_frame, person_id, source)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE pose_observation_edits (
            id TEXT PRIMARY KEY, sequence_id TEXT NOT NULL REFERENCES pose_observation_sequences(id),
            camera_instance_id TEXT NOT NULL, video_frame INTEGER NOT NULL,
            kp_blob BLOB NOT NULL, kp_mask BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
    )");
    // Written by ResultWriter -- column set copied from db/session_schema.sql.
    exec_sql(db, R"(
        CREATE TABLE tracking_runs (
            id TEXT PRIMARY KEY, observation_sequence_id TEXT NOT NULL,
            tracker_config_id TEXT NOT NULL, skeleton_id TEXT NOT NULL,
            extrinsic_calibration_id TEXT, sync_config_id TEXT,
            ran_at TEXT NOT NULL, posetrak_version TEXT,
            active_camera_ids TEXT, marker_names TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE tracking_results (
            run_id TEXT NOT NULL REFERENCES tracking_runs(id),
            person_id INTEGER NOT NULL, tracker_step INTEGER NOT NULL,
            is_smoothed INTEGER NOT NULL DEFAULT 0, timestamp_s REAL NOT NULL,
            tracking_lost INTEGER NOT NULL, n_inlier_observations INTEGER,
            cov_condition_number REAL, nis_value REAL, nis_dof INTEGER,
            state BLOB, cov_diag BLOB,
            PRIMARY KEY (run_id, person_id, tracker_step, is_smoothed)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE tracking_obs_results (
            run_id TEXT NOT NULL REFERENCES tracking_runs(id),
            person_id INTEGER NOT NULL,
            tracker_step INTEGER NOT NULL, obs_blob BLOB NOT NULL,
            PRIMARY KEY (run_id, person_id, tracker_step)
        );
    )");

    // ---- Skeleton: the real, trackable simple_humanoid fixture ----
    std::string yaml_content;
    {
        std::ifstream f("tests/data/simple_humanoid.yaml");
        if (!f)
            throw std::runtime_error("Could not open tests/data/simple_humanoid.yaml");
        std::ostringstream ss;
        ss << f.rdbuf();
        yaml_content = ss.str();
    }
    {
        sqlite3_stmt* stmt = nullptr;
        std::string sql =
            "INSERT INTO skeletons (id,name,yaml_content,created_at) "
            "VALUES ('skel1','simple_humanoid',?,'2026-01-01')";
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, yaml_content.c_str(), -1, SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }
    Skeleton skeleton = load_skeleton_from_yaml_string(yaml_content);

    // ---- Tracker config (mirrors test_tracker_integration.cpp's settings) ----
    double tracker_fps = 1.0 / dt;
    exec_sql(db,
             "INSERT INTO tracker_configs (id,name,created_at,"
             "process_noise_std,measurement_noise_std,tracker_fps,"
             "init_position_std,init_orientation_std,init_joint_std,init_velocity_std,"
             "min_cameras_for_init,ik_max_iterations,ik_tolerance,outlier_threshold) "
             "VALUES ('tc1','test','2026-01-01',"
             "0.5,2.0," +
                 std::to_string(tracker_fps) +
                 ","
                 "0.1,0.1,0.1,0.1,2,1000,0.02,4.0);");

    // ---- Session / cameras / extrinsics / sync ----
    exec_sql(db, "INSERT INTO mocap_sessions VALUES ('sess1','2026-01-01',NULL,NULL);");
    exec_sql(db,
             "INSERT INTO extrinsic_calibrations VALUES ('ec1','sess1','2026-01-01',NULL,NULL);");
    exec_sql(db, "INSERT INTO captures VALUES ('shot1','sess1','ec1',1,NULL,NULL);");
    exec_sql(db, "INSERT INTO sync_configs VALUES ('sc1','shot1',NULL,NULL);");

    auto cams = make_synthetic_cameras();
    for (size_t i = 0; i < cams.size(); ++i) {
        std::string idx = std::to_string(i);
        std::string inst = "inst" + idx;
        std::string mode = "mode" + idx;
        std::string model = "model" + idx;
        std::string ic = "ic" + idx;
        std::string sv = "sv" + idx;
        std::string label = "cam" + idx;

        exec_sql(db,
                 "INSERT INTO camera_models VALUES ('" + model + "','Test','Cam','full-frame');");
        exec_sql(db, "INSERT INTO camera_modes VALUES ('" + mode + "','" + model +
                         "',1280,720,30.0,NULL,NULL);");
        exec_sql(db, "INSERT INTO camera_instances VALUES ('" + inst + "','" + model + "',NULL,'" +
                         label + "');");
        auto const& intr = cams[i].camera.intrinsics();
        exec_sql(db,
                 "INSERT INTO intrinsics_calibrations "
                 "(id,camera_mode_id,calibrated_at,distortion_model,fx,fy,cx,cy) VALUES ('" +
                     ic + "','" + mode + "','2026-01-01','radtan'," + std::to_string(intr.fx) +
                     "," + std::to_string(intr.fy) + "," + std::to_string(intr.cx) + "," +
                     std::to_string(intr.cy) + ");");
        exec_sql(db,
                 "INSERT INTO session_cameras (session_id,camera_instance_id) VALUES ('sess1','" +
                     inst + "');");

        Eigen::Matrix3d const& R = cams[i].R_wc;
        std::vector<double> R_vals = {R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1),
                                      R(1, 2), R(2, 0), R(2, 1), R(2, 2)};
        auto R_blob = encode_float64_blob(R_vals);
        auto t_blob = encode_float64_blob({cams[i].t_wc.x(), cams[i].t_wc.y(), cams[i].t_wc.z()});
        bind_blob_and_step(db,
                           "INSERT INTO extrinsic_entries (extrinsic_calibration_id,"
                           "camera_instance_id,R,t) VALUES ('ec1','" +
                               inst + "',?,?)",
                           {R_blob, t_blob});

        exec_sql(db,
                 "INSERT INTO capture_videos (id,shot_id,camera_instance_id,file_path,"
                 "first_video_frame,last_video_frame,actual_fps,camera_mode_id,"
                 "intrinsics_calibration_id) VALUES ('" +
                     sv + "','shot1','" + inst + "','" + label + ".mp4',0,10000,30.0,'" + mode +
                     "','" + ic + "');");
        exec_sql(db,
                 "INSERT INTO sync_points (sync_config_id,camera_instance_id,shot_video_id,"
                 "video_frame,timestamp_s) VALUES ('sc1','" +
                     inst + "','" + sv + "',0,0.0);");
    }

    exec_sql(db,
             "INSERT INTO pose_observation_sequences "
             "(id,shot_id,sync_config_id,time_start_s,time_end_s) VALUES "
             "('seq1','shot1','sc1',0.0," +
                 std::to_string(num_frames * dt) + ");");

    // ---- Ground truth trajectory + synthetic per-camera observations ----
    pinocchio::Model model;
    pinocchio::Data data;
    PinocchioModelBuilder::build_model_and_data(skeleton, model, data);
    auto marker_map = PinocchioModelBuilder::build_marker_frame_map(model, skeleton);
    auto layout = SkeletonLayout::from_full_skeleton(std::make_shared<const Skeleton>(skeleton));
    ForwardKinematics fk(model, data, marker_map, layout);

    int num_dof = 0;
    for (auto const& joint : skeleton.joints()) {
        if (joint.type == JointType::REVOLUTE) {
            num_dof += 1;
        } else if (joint.type == JointType::SPHERICAL) {
            num_dof += 3;
        }
    }

    std::mt19937 rng(42);
    std::normal_distribution<double> noise_dist(0.0, 1.5);

    // kp_blob width: must cover every marker's coco_id (the YAML's
    // openpose_keypoint field) slot index.
    int max_kp_idx = 0;
    std::unordered_map<std::string, int> marker_kp_idx;
    for (auto const& m : skeleton.markers()) {
        if (!m.coco_id.has_value())
            continue;
        max_kp_idx = std::max(max_kp_idx, *m.coco_id);
        marker_kp_idx[m.name] = *m.coco_id;
    }
    int kp_width = max_kp_idx + 1;

    for (int frame = 0; frame < num_frames; ++frame) {
        double t = frame * dt;

        Eigen::Vector3d root_pos(0, 0, 0);
        Eigen::Quaterniond root_quat = Eigen::Quaterniond::Identity();
        Eigen::VectorXd joint_angles = Eigen::VectorXd::Zero(num_dof);
        for (int i = 0; i < num_dof; ++i) {
            double freq = 0.5 + 0.1 * (i % 5);
            double amplitude = 0.2;
            joint_angles(i) = amplitude * std::sin(2.0 * M_PI * freq * t + i * 0.3);
        }
        Eigen::VectorXd zero_dof = Eigen::VectorXd::Zero(num_dof);
        State gt_state(root_pos, root_quat, joint_angles, Eigen::Vector3d::Zero(),
                       Eigen::Vector3d::Zero(), zero_dof);

        auto marker_positions = fk.compute(gt_state);

        for (size_t cam_idx = 0; cam_idx < cams.size(); ++cam_idx) {
            std::vector<float> kps(static_cast<size_t>(kp_width) * 3, 0.f);
            bool any = false;
            for (auto const& [marker_name, pos_3d] : marker_positions) {
                auto kp_it = marker_kp_idx.find(marker_name);
                if (kp_it == marker_kp_idx.end())
                    continue;
                auto pos_2d_opt = cams[cam_idx].camera.project_undistorted(pos_3d);
                if (!pos_2d_opt.has_value())
                    continue;
                Eigen::Vector2d pos_2d = *pos_2d_opt;
                pos_2d.x() += noise_dist(rng);
                pos_2d.y() += noise_dist(rng);
                if (!cams[cam_idx].camera.is_in_bounds(pos_2d))
                    continue;
                size_t base = static_cast<size_t>(kp_it->second) * 3;
                kps[base + 0] = static_cast<float>(pos_2d.x());
                kps[base + 1] = static_cast<float>(pos_2d.y());
                kps[base + 2] = 0.9f;
                any = true;
            }
            if (!any)
                continue;

            auto kp_blob = encode_float32_blob(kps);
            std::string inst = "inst" + std::to_string(cam_idx);
            for (int person_id : {0, 1}) {
                sqlite3_stmt* stmt = nullptr;
                std::string sql =
                    "INSERT INTO pose_observations (sequence_id,camera_instance_id,"
                    "video_frame,timestamp_s,person_id,kp_blob) VALUES "
                    "('seq1','" +
                    inst + "'," + std::to_string(frame) + "," + std::to_string(t) + "," +
                    std::to_string(person_id) + ",?)";
                sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
                sqlite3_bind_blob(stmt, 1, kp_blob.data(), static_cast<int>(kp_blob.size()),
                                  SQLITE_STATIC);
                sqlite3_step(stmt);
                sqlite3_finalize(stmt);
            }
        }
    }

    sqlite3_close(db);
}

/// Compare two states/covariances for exact bitwise equality -- both runs
/// execute the identical code path on identical inputs, so no tolerance is
/// warranted; any difference indicates the orchestrator perturbed a person's
/// tracking (e.g. via order-dependent shared state).
bool states_bitwise_equal(State const& a, State const& b) {
    return a.root_position() == b.root_position() &&
           a.root_orientation().coeffs() == b.root_orientation().coeffs() &&
           a.root_velocity() == b.root_velocity() &&
           a.root_angular_velocity() == b.root_angular_velocity() &&
           a.joint_angles() == b.joint_angles() && a.joint_velocities() == b.joint_velocities();
}

}  // namespace

TEST_CASE("MultiPersonTracker Stage 1: output matches single-person path bitwise",
          "[multi_person_tracker][tracker]") {
    fs::path db_path = fs::temp_directory_path() / "posetrak_test_multi_person_tracker.db";
    int const num_frames = 12;
    double const dt = 1.0 / 30.0;
    create_fixture_db(db_path, num_frames, dt);

    fs::path out_root = fs::temp_directory_path() / "posetrak_test_multi_person_tracker_out";
    fs::remove_all(out_root);

    BuildPersonContextOptions opts;
    opts.db_path = db_path.string();
    opts.quiet = true;

    // ---- Direct single-person runs (today's run_track_from_db() code path) ----
    PersonSpec spec0;
    spec0.sequence_id = "seq1";
    spec0.skeleton_id = "skel1";
    spec0.config_id = "tc1";
    spec0.person_id = 0;
    spec0.output_dir = out_root / "direct_0";

    auto ctx0 = build_person_context(spec0, opts, /*verbose=*/false);
    step_person_context_frame0(*ctx0);
    for (int step = 1; step < ctx0->num_steps; ++step) {
        step_person_context(*ctx0, step, /*verbose=*/false, /*quiet=*/true);
    }
    finalize_person_context(*ctx0, /*smooth_output=*/false, /*quiet=*/true, /*verbose=*/false);

    REQUIRE(ctx0->frames_tracked > 0);

    // ---- MultiPersonTracker run, same person_id=0 alongside a second person ----
    std::vector<PersonSpec> specs(2);
    specs[0] = spec0;
    specs[0].output_dir.clear();  // overwritten by run_multi_person's naming convention
    specs[1] = spec0;
    specs[1].person_id = 1;

    // Mirror run_multi_person_track_from_db()'s output-dir convention directly
    // (that function lives in cli/track.cpp, not the library) so this test
    // exercises MultiPersonTracker exactly as the CLI does.
    specs[0].output_dir = out_root / "multi" / "person_0";
    specs[1].output_dir = out_root / "multi" / "person_1";

    MultiPersonTracker multi(specs, opts, /*verbose=*/false);
    multi.run();

    REQUIRE(multi.persons().size() == 2);
    auto const& mctx0 = multi.persons()[0];
    auto const& mctx1 = multi.persons()[1];

    SECTION("Person 0 alone vs. person 0 inside MultiPersonTracker: bitwise identical") {
        REQUIRE(mctx0->num_steps == ctx0->num_steps);
        REQUIRE(mctx0->frames_tracked == ctx0->frames_tracked);
        REQUIRE(mctx0->frames_lost == ctx0->frames_lost);
        REQUIRE(states_bitwise_equal(mctx0->tracker->state(), ctx0->tracker->state()));
        REQUIRE(mctx0->tracker->covariance() == ctx0->tracker->covariance());
    }

    SECTION("Two persons with identical input converge to identical state (determinism)") {
        // Same observations for person 0 and person 1 (Stage 1: no coupling), so
        // their independent Trackers must land on exactly the same result.
        REQUIRE(mctx1->frames_tracked == mctx0->frames_tracked);
        REQUIRE(states_bitwise_equal(mctx0->tracker->state(), mctx1->tracker->state()));
        REQUIRE(mctx0->tracker->covariance() == mctx1->tracker->covariance());
    }
}

// ---------------------------------------------------------------------------
// Stage 4: --smooth wires each person's independent RTS pass after the full
// coupled forward pass (see finalize_person_context()'s smooth_output branch,
// shared with the single-person run_track_from_db() path). Not new logic --
// this is a regression test confirming MultiPersonTracker::run() reaches that
// shared code per person with each person's own output_dir.
// ---------------------------------------------------------------------------
TEST_CASE("MultiPersonTracker Stage 4: --smooth runs independent per-person RTS smoothing",
          "[multi_person_tracker][tracker]") {
    fs::path db_path = fs::temp_directory_path() / "posetrak_test_multi_person_smooth.db";
    int const num_frames = 12;
    double const dt = 1.0 / 30.0;
    create_fixture_db(db_path, num_frames, dt);

    fs::path out_root = fs::temp_directory_path() / "posetrak_test_multi_person_smooth_out";
    fs::remove_all(out_root);

    BuildPersonContextOptions opts;
    opts.db_path = db_path.string();
    opts.quiet = true;
    opts.smooth_output = true;

    PersonSpec spec0;
    spec0.sequence_id = "seq1";
    spec0.skeleton_id = "skel1";
    spec0.config_id = "tc1";

    std::vector<PersonSpec> specs(2);
    specs[0] = spec0;
    specs[0].person_id = 0;
    specs[0].output_dir = out_root / "person_0";
    specs[1] = spec0;
    specs[1].person_id = 1;
    specs[1].output_dir = out_root / "person_1";

    MultiPersonTracker multi(specs, opts, /*verbose=*/false);
    multi.run();

    REQUIRE(multi.persons().size() == 2);

    for (auto const& dir : {specs[0].output_dir, specs[1].output_dir}) {
        for (auto const& name : {"smoothed_state_vectors.csv", "smoothed_joint_angles.csv",
                                 "smoothed_root_pose.csv"}) {
            auto path = dir / name;
            REQUIRE(fs::exists(path));
            REQUIRE(fs::file_size(path) > 0);
        }
    }

    // Each person's smoothed output is independent (different output_dir, own
    // Tracker/smoother cache) even though both track identical observations.
    auto count_data_rows = [](fs::path const& path) {
        std::ifstream f(path);
        std::string line;
        int count = -1;  // first line is the header
        while (std::getline(f, line))
            ++count;
        return count;
    };
    int const rows0 = count_data_rows(specs[0].output_dir / "smoothed_state_vectors.csv");
    int const rows1 = count_data_rows(specs[1].output_dir / "smoothed_state_vectors.csv");
    REQUIRE(rows0 > 0);
    REQUIRE(rows0 == rows1);
    REQUIRE(rows0 == multi.persons()[0]->frames_tracked);
}
