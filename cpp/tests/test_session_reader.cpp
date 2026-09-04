// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <posetrak/core/skeleton.hpp>
#include <posetrak/db/session_reader.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <sqlite3.h>

#include <array>
#include <cstdint>
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
            edited_kp_noise_std REAL,
            cross_person_max_world_mm REAL,
            cross_person_min_confidence REAL,
            cross_person_max_n INTEGER
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
    // Keypoint-slot manifest (marker-mocap design doc §4.3) -- a sequence
    // with no rows here (every fixture row inserted before this table
    // existed) keeps the legacy COCO-id-implied layout.
    exec_sql(db, R"(
        CREATE TABLE pose_sequence_keypoints (
            sequence_id TEXT NOT NULL REFERENCES pose_observation_sequences(id),
            keypoint_idx INTEGER NOT NULL,
            name TEXT NOT NULL,
            source TEXT NOT NULL,
            PRIMARY KEY (sequence_id, keypoint_idx)
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

        // Frame 3: Idea 3 (automated post-edit redetection) -- 'hand_l.refined'
        // must override 'hand_l' for the same slot. Inserted deliberately out of
        // precedence order ('.refined' row written *before* its plain 'hand_l'
        // counterpart) since the SELECT has no ORDER BY on source -- this must
        // not matter to the merge result.
        insert_row(3, "hand_l.refined", make_kp_blob(21, {{4, {350.f, 360.f, 0.7f}}}), 0.15);
        insert_row(3, "body", make_kp_blob(133, {{0, {13.f, 23.f, 0.9f}}, {5, {53.f, 63.f, 0.9f}}}),
                   2.3);
        insert_row(3, "hand_l", make_kp_blob(21, {{4, {153.f, 163.f, 0.5f}}}), 0.33);
        insert_row(3, "hand_r", make_kp_blob(21, {{4, {173.f, 183.f, 0.6f}}}), 0.43);

        // Frame 4: NO 'body' row at all -- body detection can drop a frame a
        // refined pass still covers (e.g. a hand track surviving on its own
        // persisted crop past where body tracking lost the person). Also
        // carries an edit (below) touching a body-range index, to confirm the
        // merged array is still a full 133-wide placeholder rather than
        // truncated to hand_l's own 21-wide row.
        insert_row(4, "hand_l", make_kp_blob(21, {{4, {450.f, 460.f, 0.55f}}}), 0.34);
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

    // Edit for seq_hands frame 4 (no 'body' row present -- see fixture above):
    // a body-range index (5), sized for the full 133-wide layout. Must not
    // throw a size-mismatch against a 21-wide hand-only row.
    {
        std::vector<float> edit_kps(133 * 3, 0.f);
        edit_kps[5 * 3 + 0] = 445.f;
        edit_kps[5 * 3 + 1] = 446.f;
        edit_kps[5 * 3 + 2] = 0.f;  // is_outlier=0 -> apply x/y, confidence=1
        auto edit_blob = encode_float32_blob(edit_kps);

        std::vector<uint8_t> mask(17, 0);  // ceil(133/8) = 17 bytes
        mask[5 / 8] |= static_cast<uint8_t>(1u << (5 % 8));

        bind_and_step(db,
                      "INSERT INTO pose_observation_edits "
                      "(id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask) "
                      "VALUES ('edit2', 'seq_hands', 'inst1', 4, ?, ?)",
                      edit_blob, mask);
    }

    // ---------- Marker-based-mocap object sequence (design doc §7.1 sub-phase
    // 1f) -- source='markers', no 'body' row ever, resolved via
    // pose_sequence_keypoints instead of a COCO id. Regression fixture for the
    // "primary-source row is whichever row isn't a recognized overlay, not
    // hardcoded to the literal name 'body'" fix (status.md, 2026-08-30). ----------
    exec_sql(db,
             "INSERT INTO pose_observation_sequences "
             "(id,shot_id,sync_config_id,time_start_s,time_end_s) "
             "VALUES ('seq_markers','shot1','sc1',0.0,1.0);");
    exec_sql(db,
             "INSERT INTO pose_sequence_keypoints (sequence_id, keypoint_idx, name, source) VALUES "
             "('seq_markers', 0, 'hilt:c0', 'aruco'),"
             "('seq_markers', 1, 'hilt:c1', 'aruco'),"
             "('seq_markers', 2, 'hilt:c2', 'aruco'),"
             "('seq_markers', 3, 'hilt:c3', 'aruco');");
    {
        auto make_marker_blob = [](std::vector<std::array<float, 3>> const& corners) {
            std::vector<float> kps;
            for (auto const& c : corners) {
                kps.push_back(c[0]);
                kps.push_back(c[1]);
                kps.push_back(c[2]);
            }
            return encode_float32_blob(kps);
        };
        auto insert_markers_row = [&](int frame, std::vector<std::array<float, 3>> const& corners) {
            std::string sql =
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source,"
                " kp_blob, noise_scale) "
                "VALUES ('seq_markers', 'inst1', " +
                std::to_string(frame) + ", " + std::to_string(frame * (1.0 / 120.0)) +
                ", 0, 'markers', ?, 1.0)";
            sqlite3_stmt* stmt = nullptr;
            sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
            auto blob = make_marker_blob(corners);
            sqlite3_bind_blob(stmt, 1, blob.data(), static_cast<int>(blob.size()), SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        };
        // Frame 0: all 4 corners seen.
        insert_markers_row(0, {{{100.f, 200.f, 1.f}},
                               {{110.f, 200.f, 1.f}},
                               {{110.f, 210.f, 1.f}},
                               {{100.f, 210.f, 1.f}}});
        // Frame 1: corner 2 not seen (NaN-equivalent: confidence 0) -- must be
        // filtered by min_confidence, not crash or misalign the other three.
        insert_markers_row(1, {{{101.f, 201.f, 1.f}},
                               {{111.f, 201.f, 1.f}},
                               {{0.f, 0.f, 0.f}},
                               {{101.f, 211.f, 1.f}}});

        // Frame 2: no 'markers' row at all -- the group's only row is a
        // synthetic 'hand_l'-sourced one, forcing the null-body_row ("no
        // base/primary row for this group") branch even though a real
        // object sequence never actually produces a hand_l overlay row.
        // Regression fixture for the fix generalizing that branch's
        // full-width placeholder from the literal kFullBodyNKp=133 (a
        // COCO-133 person assumption) to this sequence's own manifest
        // width (4, from the pose_sequence_keypoints rows above) --
        // without it, applying the frame-2 edit below (sized to the real
        // 4-keypoint manifest width) against a synthesized 133-wide
        // placeholder throws "edit blob has 4 keypoints, expected 133".
        insert_markers_row(2, {{{1.f, 2.f, 1.f}}});  // 1 corner only; source overridden below
        exec_sql(db,
                 "UPDATE pose_observations SET source='hand_l' "
                 "WHERE sequence_id='seq_markers' AND video_frame=2");

        std::vector<float> edit_kps(4 * 3, 0.f);
        edit_kps[0 * 3 + 0] = 300.f;
        edit_kps[0 * 3 + 1] = 301.f;
        edit_kps[0 * 3 + 2] = 0.f;  // is_outlier=0 -> apply x/y, confidence=1
        auto edit_blob = encode_float32_blob(edit_kps);

        std::vector<uint8_t> mask(1, 0);  // ceil(4/8) = 1 byte
        mask[0] |= 1u;                    // slot 0 only

        bind_and_step(db,
                      "INSERT INTO pose_observation_edits "
                      "(id, sequence_id, camera_instance_id, video_frame, kp_blob, kp_mask) "
                      "VALUES ('edit_markers_ghost', 'seq_markers', 'inst1', 2, ?, ?)",
                      edit_blob, mask);

        // ---------- Anonymous reflective-dot candidates on the same sequence
        // (source='dots'): person_id=0 as a placeholder -- dot candidates are
        // scene-wide, not tied to a tracked subject, but pose_observations'
        // primary key still requires one. A different candidate count per
        // frame (3, then 1) exercises the variable-N blob width. ----------
        // Count-prefixed format (2026-09-04): int32 candidate count, then
        // float32[count, 6] (px, py, area, compactness, major_axis,
        // minor_axis) -- see db_cache.py's encode_dot_candidates().
        auto make_dots_blob = [](std::vector<std::array<float, 6>> const& candidates) {
            std::vector<uint8_t> out;
            auto const n = static_cast<int32_t>(candidates.size());
            out.resize(sizeof(int32_t));
            std::memcpy(out.data(), &n, sizeof(int32_t));
            std::vector<float> vals;
            for (auto const& c : candidates) {
                vals.insert(vals.end(), c.begin(), c.end());
            }
            auto payload = encode_float32_blob(vals);
            out.insert(out.end(), payload.begin(), payload.end());
            return out;
        };
        auto insert_dots_row = [&](int frame, std::vector<std::array<float, 6>> const& candidates) {
            std::string sql =
                "INSERT INTO pose_observations "
                "(sequence_id, camera_instance_id, video_frame, timestamp_s, person_id, source,"
                " kp_blob, noise_scale) "
                "VALUES ('seq_markers', 'inst1', " +
                std::to_string(frame) + ", " + std::to_string(frame * (1.0 / 120.0)) +
                ", 0, 'dots', ?, NULL)";
            sqlite3_stmt* stmt = nullptr;
            sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
            auto blob = make_dots_blob(candidates);
            sqlite3_bind_blob(stmt, 1, blob.data(), static_cast<int>(blob.size()), SQLITE_STATIC);
            sqlite3_step(stmt);
            sqlite3_finalize(stmt);
        };
        // px, py, area, compactness, major_axis, minor_axis
        insert_dots_row(0, {{{400.f, 500.f, 12.5f, 0.90f, 4.0f, 4.0f}},
                            {{410.f, 505.f, 10.0f, 0.85f, 3.6f, 3.6f}},
                            {{420.f, 510.f, 15.0f, 0.92f, 4.4f, 4.4f}}});
        insert_dots_row(1, {{{450.f, 460.f, 8.0f, 0.80f, 3.2f, 3.2f}}});
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
// Helper: root-only "prop" skeleton with track/landmark-bound markers (no
// coco_id at all) -- design doc §5.1/§5.3, resolved via pose_sequence_keypoints
// rather than a COCO id.
// ---------------------------------------------------------------------------
static Skeleton make_test_object_skeleton() {
    Skeleton s;
    s.add_joint("prop_root", std::nullopt, JointType::FIXED, Eigen::Vector3d::Zero());
    s.add_input_track("prop_markers", "labeled_points");
    s.add_marker("hilt:c0", 0, Eigen::Vector3d(0.0, 0.0, 0.0), std::nullopt, "prop_markers",
                 "hilt:c0");
    s.add_marker("hilt:c1", 0, Eigen::Vector3d(0.05, 0.0, 0.0), std::nullopt, "prop_markers",
                 "hilt:c1");
    s.add_marker("hilt:c2", 0, Eigen::Vector3d(0.05, 0.05, 0.0), std::nullopt, "prop_markers",
                 "hilt:c2");
    s.add_marker("hilt:c3", 0, Eigen::Vector3d(0.0, 0.05, 0.0), std::nullopt, "prop_markers",
                 "hilt:c3");
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
    REQUIRE(cfg.tracker.cross_person_max_world_mm ==
            Catch::Approx(TrackerConfig{}.cross_person_max_world_mm));
    REQUIRE(cfg.tracker.cross_person_min_confidence ==
            Catch::Approx(TrackerConfig{}.cross_person_min_confidence));
    REQUIRE(cfg.tracker.cross_person_max_n == TrackerConfig{}.cross_person_max_n);
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
    // are absent that frame), frame 2 (unedited merge) has 4, frame 3
    // (hand_l.refined precedence) has 4, frame 4 (no body row) has 2 -- 16 total.
    REQUIRE(obs_set.total_observations() == 16);

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

// Regression test: a (camera, frame) group with no 'body' row at all (body
// detection dropped the frame; a 'hand_l' pass still covered it) must still
// merge into the full 133-wide layout -- not silently truncate to hand_l's
// own 21-wide row -- so that an edit sized for the full layout applies
// correctly instead of throwing apply_keypoint_edits' size-mismatch error.
TEST_CASE("SessionReader load_observations handles a group with no body row", "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_hands");
    auto skeleton = make_test_skeleton_with_hands();

    ObservationSet obs_set;
    REQUIRE_NOTHROW(obs_set = reader.load_observations("seq_hands", cameras, skeleton, 0.1, 0));

    auto const& seq = obs_set.sequences().begin()->second;
    std::vector<Observation> frame4;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 4)
            frame4.push_back(o);

    // coco 0 ("nose") and coco 116 ("hand_r4"): no source row ever supplied
    // them this frame (no 'body', no 'hand_r') and neither was edited, so
    // they stay confidence 0 and get filtered out -- only 2 markers survive.
    REQUIRE(frame4.size() == 2);

    // coco 5 ("body5", body-range): absent from every source row this frame,
    // but the edit still applies correctly against the full-width array.
    REQUIRE(frame4[0].position_distorted.x() == Catch::Approx(445.0));
    REQUIRE(frame4[0].position_distorted.y() == Catch::Approx(446.0));
    REQUIRE(frame4[0].confidence == Catch::Approx(1.0f));

    // coco 95 ("hand_l4"): raw 'hand_l' row value, untouched by the edit.
    REQUIRE(frame4[1].position_distorted.x() == Catch::Approx(450.0));
    REQUIRE(frame4[1].position_distorted.y() == Catch::Approx(460.0));
    REQUIRE(frame4[1].crop_scale == Catch::Approx(0.34));
}

// ---------------------------------------------------------------------------
// Idea 3 (automated post-edit redetection): a '<base>.refined' source row
// must override its plain '<base>' counterpart for the same slots, and this
// must not depend on fetch order -- the SELECT has no ORDER BY on source.
// ---------------------------------------------------------------------------

TEST_CASE("SessionReader load_observations lets hand_l.refined override hand_l",
          "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_hands");
    auto skeleton = make_test_skeleton_with_hands();

    auto obs_set = reader.load_observations("seq_hands", cameras, skeleton, 0.1, 0);
    auto const& seq = obs_set.sequences().begin()->second;

    std::vector<Observation> frame3;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 3)
            frame3.push_back(o);
    REQUIRE(frame3.size() == 4);

    // coco 0 ("nose") and coco 5 ("body5") -- from the 'body' row, unaffected.
    REQUIRE(frame3[0].position_distorted.x() == Catch::Approx(13.0));
    REQUIRE(frame3[1].position_distorted.x() == Catch::Approx(53.0));

    // coco 95 ("hand_l4"): 'hand_l.refined' (350, 360, crop_scale 0.15) wins
    // over the plain 'hand_l' row (153, 163, crop_scale 0.33) for the same
    // slot, even though the fixture inserted '.refined' *before* 'hand_l'.
    REQUIRE(frame3[2].position_distorted.x() == Catch::Approx(350.0));
    REQUIRE(frame3[2].position_distorted.y() == Catch::Approx(360.0));
    REQUIRE(frame3[2].crop_scale == Catch::Approx(0.15));

    // coco 116 ("hand_r4"): no '.refined' variant for this side -- the plain
    // 'hand_r' row still applies untouched.
    REQUIRE(frame3[3].position_distorted.x() == Catch::Approx(173.0));
    REQUIRE(frame3[3].crop_scale == Catch::Approx(0.43));
}

// ---------------------------------------------------------------------------
// Marker-based-mocap object sequences (design doc §7.1 sub-phase 1f):
// source='markers' rows, resolved via pose_sequence_keypoints instead of a
// COCO id. Regression test for the exact bug this fix addresses: the
// primary/base-layer row used to be found by literal name =='body', so an
// object sequence's 'markers' row was never recognised as the base layer and
// every one of its keypoints was silently discarded (mirrors the Python-side
// observation_merge.py bug, status.md 2026-08-30).
// ---------------------------------------------------------------------------

TEST_CASE("SessionReader load_observations resolves a manifest-bound (markers-source) sequence",
          "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_markers");
    auto skeleton = make_test_object_skeleton();

    auto obs_set = reader.load_observations("seq_markers", cameras, skeleton, 0.1, 0);
    auto const& seq = obs_set.sequences().begin()->second;

    // Frame 0: all 4 corners above the confidence threshold.
    std::vector<Observation> frame0;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 0)
            frame0.push_back(o);
    REQUIRE(frame0.size() == 4);
    // Observations come out in ascending manifest-index order (0..3), i.e.
    // marker_id order hilt:c0, c1, c2, c3 -- confirming resolve_marker_idx
    // correctly mapped each manifest slot to its skeleton marker, not just
    // "found something".
    REQUIRE(frame0[0].position_distorted.x() == Catch::Approx(100.0));
    REQUIRE(frame0[0].position_distorted.y() == Catch::Approx(200.0));
    REQUIRE(frame0[1].position_distorted.x() == Catch::Approx(110.0));
    REQUIRE(frame0[2].position_distorted.x() == Catch::Approx(110.0));
    REQUIRE(frame0[2].position_distorted.y() == Catch::Approx(210.0));
    REQUIRE(frame0[3].position_distorted.x() == Catch::Approx(100.0));

    // Frame 1: corner index 2 has confidence 0 -- filtered by min_confidence,
    // the other three still come through unaffected.
    std::vector<Observation> frame1;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 1)
            frame1.push_back(o);
    REQUIRE(frame1.size() == 3);
    REQUIRE(frame1[0].position_distorted.x() == Catch::Approx(101.0));
    REQUIRE(frame1[1].position_distorted.x() == Catch::Approx(111.0));
    REQUIRE(frame1[2].position_distorted.x() == Catch::Approx(101.0));
    REQUIRE(frame1[2].position_distorted.y() == Catch::Approx(211.0));
}

TEST_CASE(
    "SessionReader load_observations uses manifest width, not COCO-133, "
    "for an object sequence's null-body_row placeholder",
    "[session_reader]") {
    // Frame 2 of seq_markers has no real 'markers' row -- its only row is a
    // synthetic 'hand_l'-sourced one, forcing the "no base/primary row for
    // this group" branch. Before this fix, that branch always synthesized a
    // 133-wide (kFullBodyNKp) placeholder, so applying frame 2's 4-wide edit
    // (matching this sequence's real pose_sequence_keypoints manifest width)
    // threw "edit blob has 4 keypoints, expected 133" instead of applying.
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_markers");
    auto skeleton = make_test_object_skeleton();

    auto obs_set = reader.load_observations("seq_markers", cameras, skeleton, 0.1, 0);
    auto const& seq = obs_set.sequences().begin()->second;

    std::vector<Observation> frame2;
    for (auto const& o : seq.observations)
        if (o.frame_idx == 2)
            frame2.push_back(o);
    // Only the edited slot (0) clears min_confidence -- the other 3 slots
    // of the synthesized placeholder stay confidence 0 and get filtered.
    REQUIRE(frame2.size() == 1);
    REQUIRE(frame2[0].position_distorted.x() == Catch::Approx(300.0));
    REQUIRE(frame2[0].position_distorted.y() == Catch::Approx(301.0));
}

TEST_CASE("SessionReader load_unlabeled_candidates decodes a variable-N dot blob per frame",
          "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    auto cameras = reader.load_cameras_for_sequence("seq_markers");
    auto candidates = reader.load_unlabeled_candidates("seq_markers", cameras);

    // pixels_are_undistorted defaults to 1 for this sequence, so position ==
    // position_distorted -- both checked to confirm the field is actually
    // populated, not left default-constructed.
    std::vector<UnlabeledCandidate> frame0;
    std::vector<UnlabeledCandidate> frame1;
    for (auto const& c : candidates) {
        REQUIRE(c.camera_id == cameras.at("cam1").id());
        REQUIRE(c.position.isApprox(c.position_distorted));
        if (c.frame_idx == 0)
            frame0.push_back(c);
        else if (c.frame_idx == 1)
            frame1.push_back(c);
    }

    REQUIRE(frame0.size() == 3);
    REQUIRE(frame0[0].position.x() == Catch::Approx(400.0));
    REQUIRE(frame0[0].position.y() == Catch::Approx(500.0));
    REQUIRE(frame0[0].area == Catch::Approx(12.5));
    REQUIRE(frame0[0].compactness == Catch::Approx(0.90));
    REQUIRE(frame0[0].major_axis == Catch::Approx(4.0));
    REQUIRE(frame0[0].minor_axis == Catch::Approx(4.0));
    REQUIRE(frame0[1].position.x() == Catch::Approx(410.0));
    REQUIRE(frame0[2].position.x() == Catch::Approx(420.0));
    // No per-candidate detector confidence exists in the blob -- always 1.0.
    REQUIRE(frame0[0].confidence == Catch::Approx(1.0));

    REQUIRE(frame1.size() == 1);
    REQUIRE(frame1[0].position.x() == Catch::Approx(450.0));
    REQUIRE(frame1[0].position.y() == Catch::Approx(460.0));
    REQUIRE(frame1[0].area == Catch::Approx(8.0));
    REQUIRE(frame1[0].compactness == Catch::Approx(0.80));

    // seq_markers' own labeled 'markers'/'hand_l' rows must not leak in --
    // load_unlabeled_candidates() is source='dots' only.
    REQUIRE(candidates.size() == 4);
}

TEST_CASE("SessionReader load_unlabeled_candidates returns empty for a sequence with no dots rows",
          "[session_reader]") {
    auto db_path = ensure_fixture();
    SessionReader reader(db_path.string());

    // seq1 (the plain person sequence used by the basic load_observations
    // test below) has no source='dots' rows at all -- every sequence before
    // the dot-detection write path exists looks like this.
    auto cameras = reader.load_cameras_for_sequence("seq1");
    auto candidates = reader.load_unlabeled_candidates("seq1", cameras);
    REQUIRE(candidates.empty());
}
