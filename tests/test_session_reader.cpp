#include <posetrak/core/skeleton.hpp>
#include <posetrak/db/session_reader.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sqlite3.h>

#include <array>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace posetrak;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers for encoding blobs
// ---------------------------------------------------------------------------

static std::vector<uint8_t> encode_float64_blob(std::vector<double> const& vals) {
    std::vector<uint8_t> out(vals.size() * sizeof(double));
    std::memcpy(out.data(), vals.data(), out.size());
    return out;
}

static std::vector<uint8_t> encode_float32_blob(std::vector<float> const& vals) {
    std::vector<uint8_t> out(vals.size() * sizeof(float));
    std::memcpy(out.data(), vals.data(), out.size());
    return out;
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

static fs::path fixture_db_path() {
    return fs::temp_directory_path() / "posetrak_test_session_reader.db";
}

static void exec_sql(sqlite3* db, char const* sql) {
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string msg = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw std::runtime_error(std::string("SQL error: ") + msg + "\nSQL: " + sql);
    }
}

static void exec_sql(sqlite3* db, std::string const& sql) {
    exec_sql(db, sql.c_str());
}

/// Bind blob helper for fixture inserts
static void bind_and_step(sqlite3* db, std::string const& sql, std::vector<uint8_t> const& blob1,
                          std::vector<uint8_t> const& blob2 = {}) {
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_blob(stmt, 1, blob1.data(), static_cast<int>(blob1.size()), SQLITE_STATIC);
    if (!blob2.empty())
        sqlite3_bind_blob(stmt, 2, blob2.data(), static_cast<int>(blob2.size()), SQLITE_STATIC);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

static void create_fixture_db() {
    fs::path path = fixture_db_path();
    // Remove any leftover from a previous run
    if (fs::exists(path))
        fs::remove(path);

    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                             nullptr);
    if (rc != SQLITE_OK)
        throw std::runtime_error("Could not create fixture DB");

    // ---------- Schema ----------
    exec_sql(db, "PRAGMA foreign_keys=ON;");

    // Registry-style tables (embedded into the session DB for self-contained tests)
    exec_sql(db, R"(
        CREATE TABLE camera_models (
            id TEXT PRIMARY KEY,
            manufacturer TEXT,
            model_name TEXT,
            sensor_size TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE camera_modes (
            id TEXT PRIMARY KEY,
            camera_model_id TEXT NOT NULL REFERENCES camera_models(id),
            width_px INTEGER NOT NULL DEFAULT 0,
            height_px INTEGER NOT NULL DEFAULT 0,
            nominal_fps REAL NOT NULL DEFAULT 0.0,
            codec TEXT,
            notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE camera_instances (
            id TEXT PRIMARY KEY,
            camera_model_id TEXT NOT NULL REFERENCES camera_models(id),
            serial_number TEXT,
            label TEXT NOT NULL
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE intrinsics_calibrations (
            id TEXT PRIMARY KEY,
            camera_mode_id TEXT NOT NULL REFERENCES camera_modes(id),
            calibrated_at TEXT NOT NULL,
            calibration_tool TEXT,
            distortion_model TEXT NOT NULL DEFAULT 'radtan',
            fx REAL NOT NULL,
            fy REAL NOT NULL,
            cx REAL NOT NULL,
            cy REAL NOT NULL,
            dist_coeffs BLOB,
            rms_error REAL,
            notes TEXT,
            matrix_original BLOB
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE skeletons (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT REFERENCES skeletons(id),
            person_label TEXT,
            source TEXT,
            yaml_content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            notes TEXT
        );
    )");
    // Mirrors db/registry_schema.sql's tracker_configs (kept in sync manually --
    // SessionReader::load_tracker_config() selects every column by name).
    exec_sql(db, R"(
        CREATE TABLE tracker_configs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            parent_id TEXT REFERENCES tracker_configs(id),
            created_at TEXT NOT NULL,
            alpha REAL,
            beta REAL,
            kappa REAL,
            process_noise_std REAL,
            process_noise_vel_std REAL,
            velocity_half_life_s REAL,
            measurement_noise_std REAL,
            outlier_threshold REAL,
            tracker_fps REAL,
            ik_max_iterations INTEGER,
            ik_tolerance REAL,
            init_position_std REAL,
            init_orientation_std REAL,
            init_joint_std REAL,
            init_velocity_std REAL,
            min_cameras_for_init INTEGER,
            velocity_mode_camera_ids TEXT,
            velocity_measurement_noise_std REAL,
            notes TEXT,
            pose_noise_std REAL,
            use_relative_observations INTEGER,
            relative_min_confidence REAL,
            cross_pair_max_px REAL,
            cross_pair_max_n INTEGER,
            process_noise_vel_gain_joint REAL,
            process_noise_vel_ref_joint REAL,
            process_noise_vel_gain_root REAL,
            process_noise_vel_ref_root REAL,
            process_noise_vel_joint_names TEXT,
            pose_reg_joint_names TEXT,
            pose_reg_equal_split_noise_std REAL,
            pose_reg_rest_pose_noise_std REAL,
            nis_feedback_scopes TEXT,
            nis_feedback_window INTEGER,
            nis_feedback_threshold REAL,
            nis_feedback_max_multiplier REAL,
            process_noise_vel_scopes TEXT,
            soft_limit_joint_names TEXT,
            soft_limit_margin_rad REAL,
            soft_limit_noise_std REAL,
            near_limit_damping_joint_names TEXT,
            near_limit_margin_rad REAL,
            near_limit_spread_sigma REAL,
            near_limit_damping_factor REAL,
            edited_kp_noise_std REAL
        );
    )");

    // Session tables
    exec_sql(db, R"(
        CREATE TABLE mocap_sessions (
            id TEXT PRIMARY KEY,
            recorded_at TEXT NOT NULL,
            location TEXT,
            notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE session_cameras (
            session_id TEXT NOT NULL REFERENCES mocap_sessions(id),
            camera_instance_id TEXT NOT NULL,
            label TEXT,
            PRIMARY KEY (session_id, camera_instance_id)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE extrinsic_calibrations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES mocap_sessions(id),
            calibrated_at TEXT NOT NULL,
            method TEXT,
            rms_error REAL
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE extrinsic_entries (
            extrinsic_calibration_id TEXT NOT NULL REFERENCES extrinsic_calibrations(id),
            camera_instance_id TEXT NOT NULL,
            R BLOB NOT NULL,
            t BLOB NOT NULL,
            PRIMARY KEY (extrinsic_calibration_id, camera_instance_id)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE captures (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES mocap_sessions(id),
            extrinsic_calibration_id TEXT NOT NULL REFERENCES extrinsic_calibrations(id),
            capture_number INTEGER NOT NULL,
            label TEXT,
            notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE capture_videos (
            id TEXT PRIMARY KEY,
            shot_id TEXT NOT NULL REFERENCES captures(id),
            camera_instance_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            first_video_frame INTEGER NOT NULL,
            last_video_frame INTEGER NOT NULL,
            actual_fps REAL NOT NULL,
            camera_mode_id TEXT,
            intrinsics_calibration_id TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE sync_configs (
            id TEXT PRIMARY KEY,
            shot_id TEXT NOT NULL REFERENCES captures(id),
            created_by TEXT,
            notes TEXT
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE sync_points (
            sync_config_id TEXT NOT NULL REFERENCES sync_configs(id),
            camera_instance_id TEXT NOT NULL,
            shot_video_id TEXT NOT NULL REFERENCES capture_videos(id),
            video_frame INTEGER NOT NULL,
            timestamp_s REAL NOT NULL,
            PRIMARY KEY (sync_config_id, camera_instance_id)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE pose_observation_sequences (
            id TEXT PRIMARY KEY,
            shot_id TEXT NOT NULL REFERENCES captures(id),
            sync_config_id TEXT NOT NULL REFERENCES sync_configs(id),
            time_start_s REAL NOT NULL,
            time_end_s REAL NOT NULL,
            pose_model TEXT,
            notes TEXT,
            pixels_are_undistorted INTEGER NOT NULL DEFAULT 1,
            detection_run_id TEXT
        );
    )");
    // source: 'body' | 'hand_l' | 'hand_r' -- Phase 2 of hand-detection refinement
    // lets a (camera, frame, person) share multiple source rows instead of one.
    exec_sql(db, R"(
        CREATE TABLE pose_observations (
            sequence_id TEXT NOT NULL REFERENCES pose_observation_sequences(id),
            camera_instance_id TEXT NOT NULL,
            video_frame INTEGER NOT NULL,
            timestamp_s REAL NOT NULL,
            person_id INTEGER NOT NULL,
            source TEXT NOT NULL DEFAULT 'body',
            detection_run_id TEXT,
            kp_blob BLOB NOT NULL,
            noise_scale REAL,
            PRIMARY KEY (sequence_id, camera_instance_id, video_frame, person_id, source)
        );
    )");
    exec_sql(db, R"(
        CREATE TABLE pose_observation_edits (
            id TEXT PRIMARY KEY,
            sequence_id TEXT NOT NULL REFERENCES pose_observation_sequences(id),
            camera_instance_id TEXT NOT NULL,
            video_frame INTEGER NOT NULL,
            kp_blob BLOB NOT NULL,
            kp_mask BLOB NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
        );
    )");

    // ---------- Test data ----------

    // Camera model + mode
    exec_sql(db, "INSERT INTO camera_models VALUES ('model1','Test','Cam','full-frame');");
    exec_sql(db, "INSERT INTO camera_modes VALUES ('mode1','model1',1920,1080,120.0,NULL,NULL);");
    exec_sql(db, "INSERT INTO camera_instances VALUES ('inst1','model1',NULL,'cam1');");

    // Intrinsics (no distortion)
    exec_sql(db,
             "INSERT INTO intrinsics_calibrations "
             "(id,camera_mode_id,calibrated_at,distortion_model,fx,fy,cx,cy) "
             "VALUES ('ic1','mode1','2024-01-01','radtan',1000.0,1000.0,960.0,540.0);");

    // Session
    exec_sql(db, "INSERT INTO mocap_sessions VALUES ('sess1','2024-01-01',NULL,NULL);");
    exec_sql(db,
             "INSERT INTO session_cameras (session_id,camera_instance_id) "
             "VALUES ('sess1','inst1');");

    // Extrinsics: identity R, zero t
    exec_sql(db,
             "INSERT INTO extrinsic_calibrations VALUES ('ec1','sess1','2024-01-01',NULL,NULL);");
    {
        // R = identity (row-major: 1,0,0,0,1,0,0,0,1), t = (0,0,0)
        auto R_blob = encode_float64_blob({1, 0, 0, 0, 1, 0, 0, 0, 1});
        auto t_blob = encode_float64_blob({0, 0, 0});

        sqlite3_stmt* stmt = nullptr;
        std::string sql =
            "INSERT INTO extrinsic_entries (extrinsic_calibration_id, camera_instance_id, R, t) "
            "VALUES ('ec1', 'inst1', ?, ?)";
        sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
        sqlite3_bind_blob(stmt, 1, R_blob.data(), static_cast<int>(R_blob.size()), SQLITE_STATIC);
        sqlite3_bind_blob(stmt, 2, t_blob.data(), static_cast<int>(t_blob.size()), SQLITE_STATIC);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    // Shot + video + sync
    exec_sql(db, "INSERT INTO captures VALUES ('shot1','sess1','ec1',1,NULL,NULL);");
    exec_sql(
        db,
        "INSERT INTO capture_videos "
        "(id,shot_id,camera_instance_id,file_path,first_video_frame,last_video_frame,actual_fps,"
        "camera_mode_id,intrinsics_calibration_id)"
        " VALUES ('sv1','shot1','inst1','cam1.mp4',0,1000,120.0,'mode1','ic1');");
    exec_sql(db, "INSERT INTO sync_configs VALUES ('sc1','shot1',NULL,NULL);");
    exec_sql(db,
             "INSERT INTO sync_points "
             "(sync_config_id,camera_instance_id,shot_video_id,video_frame,timestamp_s) "
             "VALUES ('sc1','inst1','sv1',0,0.0);");

    // Skeleton
    exec_sql(db,
             "INSERT INTO skeletons "
             "(id,name,yaml_content,created_at) "
             "VALUES ('skel1','test','name: test\njoints: []','2024-01-01');");

    // TrackerConfig
    exec_sql(db,
             "INSERT INTO tracker_configs "
             "(id,name,created_at,alpha,beta,process_noise_std,measurement_noise_std,tracker_fps) "
             "VALUES ('tc1','test','2024-01-01',0.5,2.0,0.15,20.0,120.0);");

    // Pose observation sequence
    exec_sql(db,
             "INSERT INTO pose_observation_sequences "
             "(id,shot_id,sync_config_id,time_start_s,time_end_s) "
             "VALUES ('seq1','shot1','sc1',0.0,1.0);");

    // Pose observations: 2 frames, 4 keypoints each (x,y,conf as float32)
    // Keypoints: coco IDs 0,1,2,3 with confidence 0.9 each
    {
        // float32[4,3]: (100,200,0.9), (110,210,0.9), (120,220,0.9), (130,230,0.9)
        std::vector<float> kps = {100.f, 200.f, 0.9f, 110.f, 210.f, 0.9f,
                                  120.f, 220.f, 0.9f, 130.f, 230.f, 0.9f};
        auto kp_blob = encode_float32_blob(kps);

        for (int frame = 0; frame < 2; ++frame) {
            double ts = frame * (1.0 / 120.0);
            std::string sql =
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, kp_blob) "
                "VALUES ('seq1', 'inst1', " +
                std::to_string(frame) + ", " + std::to_string(ts) + ", 0, ?)";
            sqlite3_stmt* stmt = nullptr;
            sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
            sqlite3_bind_blob(stmt, 1, kp_blob.data(), static_cast<int>(kp_blob.size()),
                              SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        }
    }

    // ---------- Multi-source (Phase 2 hand-detection refinement) fixture ----------
    // A second sequence, isolated from seq1's fixed observation-count assertions,
    // covering 'body' + 'hand_l' + 'hand_r' rows sharing one (camera, frame) group.
    exec_sql(db,
             "INSERT INTO pose_observation_sequences "
             "(id,shot_id,sync_config_id,time_start_s,time_end_s) "
             "VALUES ('seq_hands','shot1','sc1',0.0,1.0);");
    {
        auto make_kp_blob = [](int n,
                               std::vector<std::pair<int, std::array<float, 3>>> const& set) {
            std::vector<float> kps(static_cast<size_t>(n) * 3, 0.f);
            for (auto const& [idx, v] : set) {
                kps[static_cast<size_t>(idx) * 3 + 0] = v[0];
                kps[static_cast<size_t>(idx) * 3 + 1] = v[1];
                kps[static_cast<size_t>(idx) * 3 + 2] = v[2];
            }
            return encode_float32_blob(kps);
        };
        auto insert_row = [&](int frame, std::string const& source,
                              std::vector<uint8_t> const& blob, double noise_scale) {
            std::string sql =
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source,"
                " kp_blob, noise_scale) "
                "VALUES ('seq_hands', 'inst1', " +
                std::to_string(frame) + ", " + std::to_string(frame * (1.0 / 120.0)) + ", 0, '" +
                source + "', ?, " + std::to_string(noise_scale) + ")";
            sqlite3_stmt* stmt = nullptr;
            sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
            sqlite3_bind_blob(stmt, 1, blob.data(), static_cast<int>(blob.size()), SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        };

        // Frame 0: 'body' (coco 0, 5) + 'hand_l' (local 4 -> coco 95) + 'hand_r' (local 4 -> coco
        // 116). This frame also carries edits (below) -- used by the merge+edit test case.
        insert_row(0, "body", make_kp_blob(133, {{0, {10.f, 20.f, 0.9f}}, {5, {50.f, 60.f, 0.9f}}}),
                   2.0);
        insert_row(0, "hand_l", make_kp_blob(21, {{4, {150.f, 160.f, 0.5f}}}), 0.3);
        insert_row(0, "hand_r", make_kp_blob(21, {{4, {170.f, 180.f, 0.6f}}}), 0.4);

        // Frame 1: 'body' only -- confirms grouping doesn't leak frame 0's hand rows in.
        insert_row(1, "body", make_kp_blob(133, {{0, {11.f, 21.f, 0.9f}}, {5, {51.f, 61.f, 0.9f}}}),
                   2.5);

        // Frame 2: same shape as frame 0 but no edits -- used by the pure-merge test
        // to check raw per-source crop_scale/values survive the merge untouched.
        insert_row(2, "body", make_kp_blob(133, {{0, {12.f, 22.f, 0.9f}}, {5, {52.f, 62.f, 0.9f}}}),
                   2.2);
        insert_row(2, "hand_l", make_kp_blob(21, {{4, {152.f, 162.f, 0.5f}}}), 0.32);
        insert_row(2, "hand_r", make_kp_blob(21, {{4, {172.f, 182.f, 0.6f}}}), 0.42);
    }

    // Edits for seq_hands frame 0: one body-range index (5) and one hand-range
    // index (95, inside the hand_l 91-111 range) overridden in the same frame,
    // exercising db::apply_keypoint_edits against the *merged* 133-wide array.
    {
        std::vector<float> edit_kps(133 * 3, 0.f);
        edit_kps[5 * 3 + 0] = 555.f;  // body-range override
        edit_kps[5 * 3 + 1] = 556.f;
        edit_kps[5 * 3 + 2] = 0.f;     // is_outlier=0 -> apply x/y, confidence=1
        edit_kps[95 * 3 + 0] = 995.f;  // hand-range override
        edit_kps[95 * 3 + 1] = 996.f;
        edit_kps[95 * 3 + 2] = 0.f;
        auto edit_blob = encode_float32_blob(edit_kps);

        std::vector<uint8_t> mask(17, 0);  // ceil(133/8) = 17 bytes
        mask[5 / 8] |= static_cast<uint8_t>(1u << (5 % 8));
        mask[95 / 8] |= static_cast<uint8_t>(1u << (95 % 8));

        bind_and_step(db,
                      "INSERT INTO pose_observation_edits "
                      "(id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask) "
                      "VALUES ('edit1', 'seq_hands', 'inst1', 0, ?, ?)",
                      edit_blob, mask);
    }

    sqlite3_close(db);
}

// One-time fixture setup: created lazily by the first test that calls this
static fs::path ensure_fixture() {
    static bool created = false;
    if (!created) {
        create_fixture_db();
        created = true;
    }
    return fixture_db_path();
}

// ---------------------------------------------------------------------------
// Helper: create a minimal Skeleton with 4 markers (coco IDs 0-3)
// ---------------------------------------------------------------------------
static Skeleton make_test_skeleton() {
    Skeleton s;
    s.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    s.add_marker("nose", 0, Eigen::Vector3d::Zero(), 0);
    s.add_marker("left_eye", 0, Eigen::Vector3d(0.05, 0, 0), 1);
    s.add_marker("right_eye", 0, Eigen::Vector3d(-0.05, 0, 0), 2);
    s.add_marker("left_ear", 0, Eigen::Vector3d(0.1, 0, 0), 3);
    return s;
}

// ---------------------------------------------------------------------------
// Helper: skeleton with markers spanning both a 'body' index (5) and the
// 'hand_l' (coco 95 = 91+4) / 'hand_r' (coco 116 = 112+4) index ranges, for
// exercising the multi-source merge in load_observations.
// ---------------------------------------------------------------------------
static Skeleton make_test_skeleton_with_hands() {
    Skeleton s;
    s.add_joint("root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    s.add_marker("nose", 0, Eigen::Vector3d::Zero(), 0);
    s.add_marker("body5", 0, Eigen::Vector3d(0.02, 0, 0), 5);
    s.add_marker("hand_l4", 0, Eigen::Vector3d(0.03, 0, 0), 95);
    s.add_marker("hand_r4", 0, Eigen::Vector3d(0.04, 0, 0), 116);
    return s;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_CASE("SessionReader load_skeleton_yaml", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto yaml = reader.load_skeleton_yaml("skel1");
    REQUIRE(yaml == "name: test\njoints: []");
}

TEST_CASE("SessionReader load_tracker_config", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cfg = reader.load_tracker_config("tc1");

    REQUIRE(cfg.tracker_fps == Catch::Approx(120.0));
    REQUIRE(cfg.tracker.ukf_alpha == Catch::Approx(0.5));
    REQUIRE(cfg.tracker.ukf_beta == Catch::Approx(2.0));
    REQUIRE(cfg.tracker.process_noise_std == Catch::Approx(0.15));
    REQUIRE(cfg.tracker.calib_noise_std == Catch::Approx(20.0));

    // Columns not set should keep TrackerConfig defaults
    REQUIRE(cfg.tracker.ukf_kappa == Catch::Approx(TrackerConfig{}.ukf_kappa));
}

TEST_CASE("SessionReader load_sequence_info", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto info = reader.load_sequence_info("seq1");
    REQUIRE(info.time_start_s == Catch::Approx(0.0));
    REQUIRE(info.time_end_s == Catch::Approx(1.0));
}

TEST_CASE("SessionReader load_cameras", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras("sess1", "ec1", "sc1");

    REQUIRE(cameras.size() == 1);
    REQUIRE(cameras.count("cam1") == 1);

    auto const& cam = cameras.at("cam1");
    REQUIRE(cam.id() == 0);
    REQUIRE(cam.name() == "cam1");
    REQUIRE(cam.intrinsics().fx == Catch::Approx(1000.0));
    REQUIRE(cam.intrinsics().fy == Catch::Approx(1000.0));
    REQUIRE(cam.intrinsics().cx == Catch::Approx(960.0));
    REQUIRE(cam.intrinsics().cy == Catch::Approx(540.0));
    REQUIRE(cam.intrinsics().width == 1920);
    REQUIRE(cam.intrinsics().height == 1080);

    // Identity R, zero t → position = -R^T * t = [0,0,0]
    REQUIRE(cam.position().norm() == Catch::Approx(0.0));

    // FPS set from sync shot_videos.actual_fps
    REQUIRE(cam.fps() == Catch::Approx(120.0));
}

TEST_CASE("SessionReader load_observations", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras("sess1", "ec1", "sc1");
    auto skeleton = make_test_skeleton();

    auto obs_set = reader.load_observations("seq1", cameras, skeleton, 0.1, 0);

    // Should have 1 camera sequence
    REQUIRE(obs_set.camera_count() == 1);

    // 2 frames x 4 keypoints = 8 observations
    REQUIRE(obs_set.total_observations() == 8);

    // All observations should reference camera id 0
    auto const& seq = obs_set.sequences().begin()->second;
    REQUIRE(seq.camera_id == 0);
    REQUIRE(seq.camera_name == "cam1");

    // Check first observation has valid marker_id and confidence
    REQUIRE(!seq.observations.empty());
    auto const& first = seq.observations[0];
    REQUIRE(first.confidence == Catch::Approx(0.9f));
    REQUIRE(first.marker_id >= 0);
    REQUIRE(first.marker_id < 4);
    REQUIRE(first.camera_id == 0);
}

TEST_CASE("SessionReader load_cameras_for_sequence", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    // Should resolve session/extrinsics/sync automatically from the sequence ID
    auto cameras = reader.load_cameras_for_sequence("seq1");

    REQUIRE(cameras.size() == 1);
    REQUIRE(cameras.count("cam1") == 1);

    auto const& cam = cameras.at("cam1");
    REQUIRE(cam.id() == 0);
    REQUIRE(cam.intrinsics().fx == Catch::Approx(1000.0));
    REQUIRE(cam.position().norm() == Catch::Approx(0.0));
    REQUIRE(cam.fps() == Catch::Approx(120.0));

    REQUIRE_THROWS_AS(reader.load_cameras_for_sequence("nonexistent"), std::runtime_error);
}

TEST_CASE("SessionReader error on missing record", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    REQUIRE_THROWS_AS(reader.load_skeleton_yaml("nonexistent"), std::runtime_error);
    REQUIRE_THROWS_AS(reader.load_tracker_config("nonexistent"), std::runtime_error);
    REQUIRE_THROWS_AS(reader.load_sequence_info("nonexistent"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Phase 2 of hand-detection refinement: multi-source pose_observations rows
// ('body' plus 'hand_l'/'hand_r') sharing one (camera, frame) group must be
// merged into one dense array before edits apply and Observations are built.
// ---------------------------------------------------------------------------

TEST_CASE("SessionReader load_observations merges body and hand source rows", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_hands");
    auto skeleton = make_test_skeleton_with_hands();

    auto obs_set = reader.load_observations("seq_hands", cameras, skeleton, 0.1, 0);
    auto const& seq = obs_set.sequences().begin()->second;

    // Frame 0 (edited) has 4 markers, frame 1 (body-only) has 2 (coco 95/116
    // are absent that frame), frame 2 (unedited merge) has 4 -- 10 total.
    REQUIRE(obs_set.total_observations() == 10);

    // Frame 2: no edits applied -- raw per-source values and crop_scale survive
    // the merge. Observations come out in ascending coco-index order (0, 5, 95, 116).
    std::vector<Observation> frame2;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 2)
            frame2.push_back(o);
    REQUIRE(frame2.size() == 4);

    // coco 0 ("nose") and coco 5 ("body5") -- both from the 'body' row, crop_scale 2.2.
    REQUIRE(frame2[0].position_distorted.x() == Catch::Approx(12.0));
    REQUIRE(frame2[0].position_distorted.y() == Catch::Approx(22.0));
    REQUIRE(frame2[0].crop_scale == Catch::Approx(2.2));
    REQUIRE(frame2[1].position_distorted.x() == Catch::Approx(52.0));
    REQUIRE(frame2[1].crop_scale == Catch::Approx(2.2));

    // coco 95 ("hand_l4", local hand21 index 4) -- from the 'hand_l' row, crop_scale 0.32.
    REQUIRE(frame2[2].position_distorted.x() == Catch::Approx(152.0));
    REQUIRE(frame2[2].position_distorted.y() == Catch::Approx(162.0));
    REQUIRE(frame2[2].crop_scale == Catch::Approx(0.32));

    // coco 116 ("hand_r4", local hand21 index 4) -- from the 'hand_r' row, crop_scale 0.42.
    REQUIRE(frame2[3].position_distorted.x() == Catch::Approx(172.0));
    REQUIRE(frame2[3].position_distorted.y() == Catch::Approx(182.0));
    REQUIRE(frame2[3].crop_scale == Catch::Approx(0.42));
}

TEST_CASE("SessionReader load_observations applies edits to the merged array", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_hands");
    auto skeleton = make_test_skeleton_with_hands();

    // Must not throw: with the naive per-row-Observation approach this would hit
    // apply_keypoint_edits' size-mismatch check when applied to a 21-wide hand
    // row instead of the merged 133-wide array.
    ObservationSet obs_set;
    REQUIRE_NOTHROW(obs_set = reader.load_observations("seq_hands", cameras, skeleton, 0.1, 0));

    auto const& seq = obs_set.sequences().begin()->second;
    std::vector<Observation> frame0;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 0)
            frame0.push_back(o);
    REQUIRE(frame0.size() == 4);

    // coco 5 (body-range): edited to (555, 556), confidence forced to 1.
    REQUIRE(frame0[1].position_distorted.x() == Catch::Approx(555.0));
    REQUIRE(frame0[1].position_distorted.y() == Catch::Approx(556.0));
    REQUIRE(frame0[1].confidence == Catch::Approx(1.0f));

    // coco 95 (hand-range, inside 'hand_l'): edited to (995, 996) -- the edit
    // blob's global index overrides the hand_l row's merged-in value correctly.
    REQUIRE(frame0[2].position_distorted.x() == Catch::Approx(995.0));
    REQUIRE(frame0[2].position_distorted.y() == Catch::Approx(996.0));
    REQUIRE(frame0[2].confidence == Catch::Approx(1.0f));

    // coco 116 (hand_r, not edited): still the raw hand_r-sourced value --
    // no cross-contamination from the body/hand_l edits on the same frame.
    REQUIRE(frame0[3].position_distorted.x() == Catch::Approx(170.0));
    REQUIRE(frame0[3].position_distorted.y() == Catch::Approx(180.0));
    REQUIRE(frame0[3].crop_scale == Catch::Approx(0.4));
}
