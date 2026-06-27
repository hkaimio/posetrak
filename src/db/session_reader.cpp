#include <posetrak/db/blob_codec.hpp>
#include <posetrak/db/session_reader.hpp>

#include <Eigen/Geometry>

#include <nlohmann/json.hpp>

#include <sqlite3.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace posetrak {

// ---------------------------------------------------------------------------
// Stmt implementation
// ---------------------------------------------------------------------------

SessionReader::Stmt::Stmt(sqlite3* db, char const* sql) {
    int rc = sqlite3_prepare_v2(db, sql, -1, &ptr, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(std::string("sqlite3_prepare_v2 failed: ") + sqlite3_errmsg(db) +
                                 " SQL: " + sql);
    }
}

SessionReader::Stmt::~Stmt() {
    if (ptr) {
        sqlite3_finalize(ptr);
        ptr = nullptr;
    }
}

SessionReader::Stmt::Stmt(Stmt&& other) noexcept : ptr(other.ptr) {
    other.ptr = nullptr;
}

SessionReader::Stmt& SessionReader::Stmt::operator=(Stmt&& other) noexcept {
    if (this != &other) {
        if (ptr)
            sqlite3_finalize(ptr);
        ptr = other.ptr;
        other.ptr = nullptr;
    }
    return *this;
}

bool SessionReader::Stmt::step() {
    int rc = sqlite3_step(ptr);
    if (rc == SQLITE_ROW)
        return true;
    if (rc == SQLITE_DONE)
        return false;
    throw std::runtime_error(std::string("sqlite3_step failed: ") +
                             sqlite3_errmsg(sqlite3_db_handle(ptr)));
}

void SessionReader::Stmt::reset() {
    sqlite3_reset(ptr);
    sqlite3_clear_bindings(ptr);
}

// ---------------------------------------------------------------------------
// SessionReader implementation
// ---------------------------------------------------------------------------

SessionReader::SessionReader(std::string const& db_path) {
    int rc = sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READONLY | SQLITE_OPEN_FULLMUTEX,
                             nullptr);
    if (rc != SQLITE_OK) {
        std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        if (db_)
            sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("Failed to open session DB '" + db_path + "': " + err);
    }

    // Enable foreign key enforcement (best-effort; read-only so mostly informational)
    sqlite3_exec(db_, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);
}

SessionReader::~SessionReader() {
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

SessionReader::SessionReader(SessionReader&& other) noexcept : db_(other.db_) {
    other.db_ = nullptr;
}

SessionReader& SessionReader::operator=(SessionReader&& other) noexcept {
    if (this != &other) {
        if (db_)
            sqlite3_close(db_);
        db_ = other.db_;
        other.db_ = nullptr;
    }
    return *this;
}

// ---------------------------------------------------------------------------

std::string SessionReader::resolve_id(std::string const& table, std::string const& prefix) {
    // Use a dynamic query — table name cannot be parameterised in SQLite
    std::string sql = "SELECT id FROM " + table + " WHERE id LIKE ? || '%'";
    Stmt stmt(db_, sql.c_str());
    sqlite3_bind_text(stmt.ptr, 1, prefix.c_str(), -1, SQLITE_STATIC);

    std::string first;
    int count = 0;
    while (stmt.step()) {
        if (count == 0)
            first = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 0));
        ++count;
    }

    if (count == 0)
        throw std::runtime_error("No " + table + " record found with id prefix '" + prefix + "'");
    if (count > 1)
        throw std::runtime_error("Ambiguous prefix '" + prefix + "' matches " +
                                 std::to_string(count) + " " + table + " records");
    return first;
}

// ---------------------------------------------------------------------------

std::string SessionReader::load_skeleton_yaml(std::string const& skeleton_id) {
    Stmt stmt(db_, "SELECT yaml_content FROM skeletons WHERE id = ?");
    sqlite3_bind_text(stmt.ptr, 1, skeleton_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("Skeleton not found: " + skeleton_id);
    }

    auto const* text = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 0));
    if (!text) {
        throw std::runtime_error("Skeleton yaml_content is NULL for id: " + skeleton_id);
    }
    return std::string(text);
}

// ---------------------------------------------------------------------------

DbTrackerConfig SessionReader::load_tracker_config(std::string const& config_id) {
    Stmt stmt(db_,
              "SELECT alpha, beta, kappa, process_noise_std, measurement_noise_std,"
              "       outlier_threshold, tracker_fps, ik_max_iterations, ik_tolerance,"
              "       init_position_std, init_orientation_std, init_joint_std, init_velocity_std,"
              "       min_cameras_for_init, process_noise_vel_std, velocity_half_life_s,"
              "       velocity_mode_camera_ids, velocity_measurement_noise_std,"
              "       COALESCE(pose_noise_std, 0.0) AS pose_noise_std,"
              "       COALESCE(use_relative_observations, 0) AS use_relative_observations,"
              "       COALESCE(relative_min_confidence, 0.5) AS relative_min_confidence,"
              "       COALESCE(cross_pair_max_px, 0.0) AS cross_pair_max_px,"
              "       COALESCE(cross_pair_max_n, 10) AS cross_pair_max_n"
              " FROM tracker_configs WHERE id = ?");
    sqlite3_bind_text(stmt.ptr, 1, config_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("TrackerConfig not found: " + config_id);
    }

    DbTrackerConfig out;
    // Columns: 0=alpha, 1=beta, 2=kappa, 3=process_noise_std,
    //          4=measurement_noise_std (legacy → calib_noise_std),
    //          5=outlier_threshold, 6=tracker_fps, 7=ik_max_iterations, 8=ik_tolerance,
    //          9=init_position_std, 10=init_orientation_std, 11=init_joint_std,
    //         12=init_velocity_std, 13=min_cameras_for_init, 14=process_noise_vel_std,
    //         15=velocity_half_life_s, 16=velocity_mode_camera_ids,
    //         17=velocity_measurement_noise_std, 18=pose_noise_std,
    //         19=use_relative_observations, 20=relative_min_confidence

    auto apply_real = [&](int col, double& field) {
        if (sqlite3_column_type(stmt.ptr, col) != SQLITE_NULL)
            field = sqlite3_column_double(stmt.ptr, col);
    };
    auto apply_int = [&](int col, int& field) {
        if (sqlite3_column_type(stmt.ptr, col) != SQLITE_NULL)
            field = sqlite3_column_int(stmt.ptr, col);
    };
    auto apply_opt_real = [&](int col, std::optional<double>& field) {
        if (sqlite3_column_type(stmt.ptr, col) != SQLITE_NULL)
            field = sqlite3_column_double(stmt.ptr, col);
    };

    apply_real(0, out.tracker.ukf_alpha);
    apply_real(1, out.tracker.ukf_beta);
    apply_real(2, out.tracker.ukf_kappa);
    apply_real(3, out.tracker.process_noise_std);
    apply_real(4, out.tracker.calib_noise_std);  // legacy measurement_noise_std → calib_noise_std
    apply_real(5, out.tracker.outlier_threshold);
    apply_real(6, out.tracker_fps);
    apply_int(7, out.tracker.ik_max_iterations);
    apply_real(8, out.tracker.ik_tolerance);
    apply_real(9, out.tracker.init_position_std);
    apply_real(10, out.tracker.init_orientation_std);
    apply_real(11, out.tracker.init_joint_std);
    apply_real(12, out.tracker.init_velocity_std);
    apply_int(13, out.tracker.min_cameras_for_init);
    apply_opt_real(14, out.tracker.process_noise_vel_std);
    apply_opt_real(15, out.tracker.velocity_half_life_s);

    // velocity_mode_camera_ids: stored as JSON integer array, e.g. "[2]"
    if (sqlite3_column_type(stmt.ptr, 16) != SQLITE_NULL) {
        char const* json_str = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 16));
        if (json_str) {
            auto arr = nlohmann::json::parse(json_str, nullptr, /*allow_exceptions=*/false);
            if (arr.is_array()) {
                for (auto const& elem : arr) {
                    if (elem.is_number_integer())
                        out.tracker.velocity_mode_camera_ids.push_back(elem.get<int>());
                }
            }
        }
    }
    apply_opt_real(17, out.tracker.velocity_measurement_noise_std);
    apply_real(18, out.tracker.pose_noise_std);
    // col 19: use_relative_observations (INTEGER 0/1)
    if (sqlite3_column_type(stmt.ptr, 19) != SQLITE_NULL)
        out.tracker.use_relative_observations = (sqlite3_column_int(stmt.ptr, 19) != 0);
    apply_real(20, out.tracker.relative_min_confidence);
    apply_real(21, out.tracker.cross_pair_max_px);
    apply_int(22, out.tracker.cross_pair_max_n);

    return out;
}

// ---------------------------------------------------------------------------

SequenceInfo SessionReader::load_sequence_info(std::string const& sequence_id) {
    Stmt stmt(db_,
              "SELECT time_start_s, time_end_s"
              " FROM pose_observation_sequences WHERE id = ?");
    sqlite3_bind_text(stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("Sequence not found: " + sequence_id);
    }

    SequenceInfo info;
    info.time_start_s = sqlite3_column_double(stmt.ptr, 0);
    info.time_end_s = sqlite3_column_double(stmt.ptr, 1);
    return info;
}

// ---------------------------------------------------------------------------

SequenceMetadata SessionReader::load_sequence_metadata(std::string const& sequence_id) {
    Stmt stmt(db_,
              "SELECT s.session_id, s.extrinsic_calibration_id, pos.sync_config_id"
              " FROM pose_observation_sequences pos"
              " JOIN captures s ON s.id = pos.shot_id"
              " WHERE pos.id = ?");
    sqlite3_bind_text(stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("pose_observation_sequence not found: " + sequence_id);
    }

    auto col_str_or_empty = [&](int col) -> std::string {
        auto const* p = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, col));
        return p ? p : std::string{};
    };

    SequenceMetadata meta;
    meta.session_id = col_str_or_empty(0);
    meta.extrinsic_calibration_id = col_str_or_empty(1);
    meta.sync_config_id = col_str_or_empty(2);
    return meta;
}

// ---------------------------------------------------------------------------

std::map<std::string, Camera>
SessionReader::load_cameras_for_sequence(std::string const& sequence_id) {
    Stmt stmt(db_,
              "SELECT s.session_id, s.extrinsic_calibration_id, pos.sync_config_id"
              " FROM pose_observation_sequences pos"
              " JOIN captures s ON s.id = pos.shot_id"
              " WHERE pos.id = ?");
    sqlite3_bind_text(stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("pose_observation_sequence not found: " + sequence_id);
    }

    auto col_str = [&](int col, char const* name) -> std::string {
        auto const* p = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, col));
        if (!p)
            throw std::runtime_error(std::string(name) +
                                     " is not set for sequence: " + sequence_id);
        return p;
    };

    std::string session_id = col_str(0, "session_id");
    std::string extrinsics_id = col_str(1, "extrinsic_calibration_id");
    std::string sync_id = col_str(2, "sync_config_id");

    return load_cameras(session_id, extrinsics_id, sync_id);
}

// ---------------------------------------------------------------------------

std::map<std::string, Camera>
SessionReader::load_cameras(std::string const& session_id,
                            std::string const& extrinsic_calibration_id,
                            std::string const& sync_config_id) {
    // Step 1: Fetch camera rows from capture_videos for the capture referenced by sync_config.
    // Intrinsics are now per capture_video (v11+); mode/dimensions come from camera_modes.
    // Uses sync_config_id to identify the capture so the right mode/intrinsics are loaded.
    Stmt cam_stmt(db_,
                  "SELECT ci.id, ci.label,"
                  "       ic.fx, ic.fy, ic.cx, ic.cy, ic.dist_coeffs, ic.distortion_model,"
                  "       COALESCE(cm.width_px, 0), COALESCE(cm.height_px, 0), ic.matrix_original"
                  " FROM capture_videos sv"
                  " JOIN captures sh ON sh.id = sv.shot_id"
                  " JOIN sync_configs scfg ON scfg.shot_id = sh.id"
                  " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
                  " LEFT JOIN intrinsics_calibrations ic ON ic.id = sv.intrinsics_calibration_id"
                  " LEFT JOIN camera_modes cm ON cm.id = sv.camera_mode_id"
                  " WHERE scfg.id = ?"
                  " ORDER BY ci.label ASC");
    sqlite3_bind_text(cam_stmt.ptr, 1, sync_config_id.c_str(), -1, SQLITE_STATIC);

    // Collect camera rows first so we can iterate with sequential IDs
    struct CamRow {
        std::string instance_id;
        std::string label;
        double fx, fy, cx, cy;
        std::vector<double> dist_coeffs;
        Intrinsics::DistortionModel dist_model;
        int width, height;
        std::optional<Eigen::Matrix3d> K_original;  // from matrix_original blob; nullopt if absent
    };
    std::vector<CamRow> rows;

    auto require_text = [&](sqlite3_stmt* s, int col, char const* name) -> std::string {
        auto const* p = reinterpret_cast<char const*>(sqlite3_column_text(s, col));
        if (!p)
            throw std::runtime_error(std::string("NULL value for required column: ") + name);
        return p;
    };

    while (cam_stmt.step()) {
        CamRow row;
        row.instance_id = require_text(cam_stmt.ptr, 0, "camera_instances.id");
        row.label = require_text(cam_stmt.ptr, 1, "camera_instances.label");
        row.fx = sqlite3_column_double(cam_stmt.ptr, 2);
        row.fy = sqlite3_column_double(cam_stmt.ptr, 3);
        row.cx = sqlite3_column_double(cam_stmt.ptr, 4);
        row.cy = sqlite3_column_double(cam_stmt.ptr, 5);

        // dist_coeffs blob (nullable)
        if (sqlite3_column_type(cam_stmt.ptr, 6) != SQLITE_NULL) {
            void const* blob = sqlite3_column_blob(cam_stmt.ptr, 6);
            int nbytes = sqlite3_column_bytes(cam_stmt.ptr, 6);
            row.dist_coeffs = db::decode_float64_blob(blob, nbytes);
        }

        std::string dist_model_str;
        if (sqlite3_column_type(cam_stmt.ptr, 7) != SQLITE_NULL) {
            dist_model_str = reinterpret_cast<char const*>(sqlite3_column_text(cam_stmt.ptr, 7));
        }
        row.dist_model = (dist_model_str == "fisheye") ? Intrinsics::DistortionModel::Fisheye
                                                       : Intrinsics::DistortionModel::BrownConrady;

        row.width = sqlite3_column_int(cam_stmt.ptr, 8);
        row.height = sqlite3_column_int(cam_stmt.ptr, 9);

        // matrix_original blob (nullable) — 9 float64 values, row-major
        if (sqlite3_column_type(cam_stmt.ptr, 10) != SQLITE_NULL) {
            void const* blob = sqlite3_column_blob(cam_stmt.ptr, 10);
            int nbytes = sqlite3_column_bytes(cam_stmt.ptr, 10);
            auto vals = db::decode_float64_blob(blob, nbytes);
            if (vals.size() == 9) {
                Eigen::Matrix3d K;
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        K(r, c) = vals[static_cast<size_t>(r * 3 + c)];
                row.K_original = K;
            }
        }

        rows.push_back(std::move(row));
    }

    if (rows.empty()) {
        throw std::runtime_error("No cameras found for session: " + session_id);
    }

    // Prepare extrinsics and sync queries for per-camera lookup
    Stmt ext_stmt(db_,
                  "SELECT R, t FROM extrinsic_entries"
                  " WHERE extrinsic_calibration_id = ? AND camera_instance_id = ?");

    Stmt sync_stmt(db_,
                   "SELECT sp.video_frame, sp.timestamp_s, sv.actual_fps"
                   " FROM sync_points sp"
                   " JOIN capture_videos sv ON sv.id = sp.shot_video_id"
                   " WHERE sp.sync_config_id = ? AND sp.camera_instance_id = ?"
                   " ORDER BY sp.video_frame ASC");

    std::map<std::string, Camera> result;
    int camera_id = 0;

    for (auto const& row : rows) {
        // Step 2: Fetch extrinsics for this camera
        ext_stmt.reset();
        sqlite3_bind_text(ext_stmt.ptr, 1, extrinsic_calibration_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(ext_stmt.ptr, 2, row.instance_id.c_str(), -1, SQLITE_STATIC);

        if (!ext_stmt.step()) {
            throw std::runtime_error("No extrinsic entry for camera '" + row.label +
                                     "' in calibration '" + extrinsic_calibration_id + "'");
        }

        void const* R_blob = sqlite3_column_blob(ext_stmt.ptr, 0);
        int R_bytes = sqlite3_column_bytes(ext_stmt.ptr, 0);
        void const* t_blob = sqlite3_column_blob(ext_stmt.ptr, 1);
        int t_bytes = sqlite3_column_bytes(ext_stmt.ptr, 1);

        Eigen::Matrix3d R_matrix = db::decode_rotation_matrix(R_blob, R_bytes);
        Eigen::Vector3d tvec = db::decode_translation(t_blob, t_bytes);

        // Same convention as camera_loader.cpp:
        //   point_cam = R * point_world + t  →  camera_position = -R^T * t
        Eigen::Vector3d camera_position = -R_matrix.transpose() * tvec;
        Eigen::Quaterniond orientation(R_matrix);

        Intrinsics intrinsics{row.fx,    row.fy,     row.cx,         row.cy,
                              row.width, row.height, row.dist_model, row.dist_coeffs};
        Extrinsics extrinsics{camera_position, orientation};

        Camera cam(camera_id++, row.label, intrinsics, extrinsics);
        if (row.K_original.has_value()) {
            cam.set_K_original(row.K_original.value());
        }

        // Step 3: Fetch all sync points for this camera
        sync_stmt.reset();
        sqlite3_bind_text(sync_stmt.ptr, 1, sync_config_id.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(sync_stmt.ptr, 2, row.instance_id.c_str(), -1, SQLITE_STATIC);

        std::vector<SyncPoint> sync_pts;
        double actual_fps = 0.0;
        while (sync_stmt.step()) {
            int video_frame = sqlite3_column_int(sync_stmt.ptr, 0);
            double timestamp_s = sqlite3_column_double(sync_stmt.ptr, 1);
            actual_fps = sqlite3_column_double(sync_stmt.ptr, 2);
            sync_pts.push_back({static_cast<uint32_t>(video_frame), timestamp_s});
        }
        if (!sync_pts.empty()) {
            cam.set_fps(actual_fps);
            cam.set_sync_points(sync_pts);
        }

        result.emplace(row.label, std::move(cam));
    }

    return result;
}

// ---------------------------------------------------------------------------

ObservationSet SessionReader::load_observations(std::string const& sequence_id,
                                                std::map<std::string, Camera> const& cameras,
                                                Skeleton const& skeleton, double min_confidence,
                                                int person_id, bool use_relative_obs,
                                                double relative_min_conf, double pose_noise_std,
                                                double cross_pair_max_px, int cross_pair_max_n) {
    // Step 0: Read pixels_are_undistorted flag for this sequence
    bool pixels_are_undistorted = true;  // default: assume undistorted (safe for existing data)
    {
        Stmt flag_stmt(db_,
                       "SELECT pixels_are_undistorted"
                       " FROM pose_observation_sequences WHERE id = ?");
        sqlite3_bind_text(flag_stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);
        if (flag_stmt.step()) {
            if (sqlite3_column_type(flag_stmt.ptr, 0) != SQLITE_NULL) {
                pixels_are_undistorted = (sqlite3_column_int(flag_stmt.ptr, 0) != 0);
            }
        }
    }

    // Step 1: Build instance_id → Camera const* map.
    // Enumerate cameras from capture_videos for the capture this sequence belongs to.
    Stmt inst_stmt(db_,
                   "SELECT ci.id, ci.label"
                   " FROM pose_observation_sequences pos"
                   " JOIN captures s ON s.id = pos.shot_id"
                   " JOIN capture_videos sv ON sv.shot_id = s.id"
                   " JOIN camera_instances ci ON ci.id = sv.camera_instance_id"
                   " WHERE pos.id = ?");
    sqlite3_bind_text(inst_stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);

    std::unordered_map<std::string, Camera const*> inst_to_cam;
    while (inst_stmt.step()) {
        std::string inst_id = reinterpret_cast<char const*>(sqlite3_column_text(inst_stmt.ptr, 0));
        std::string label = reinterpret_cast<char const*>(sqlite3_column_text(inst_stmt.ptr, 1));

        auto it = cameras.find(label);
        if (it != cameras.end()) {
            inst_to_cam[inst_id] = &it->second;
        }
    }

    // Step 2: Build COCO keypoint ID → skeleton marker index map
    std::unordered_map<int, int> coco_to_marker_idx;
    auto const& markers = skeleton.markers();
    for (size_t i = 0; i < markers.size(); ++i) {
        if (markers[i].coco_id.has_value()) {
            coco_to_marker_idx[markers[i].coco_id.value()] = static_cast<int>(i);
        }
    }

    // Step 2.6: Build marker parent map (hierarchical RELATIVE pairs, Phase 3) and
    // all-pairs distance matrix (spatial cross-pairs, Phase 4).
    // For each marker, find the nearest ancestor joint that also has a marker attached.
    std::unordered_map<int, int> marker_parent_map;  // marker_id → parent_marker_id (-1 = none)
    // marker_dist_matrix[a][b] = joint-hop distance between marker a and marker b.
    std::vector<std::vector<int>> marker_dist_matrix;
    if (use_relative_obs || cross_pair_max_px > 0.0) {
        auto const& joints = skeleton.joints();
        // joint_index → first marker_idx attached to it
        std::unordered_map<uint32_t, int> joint_to_first_marker;
        for (int i = 0; i < static_cast<int>(markers.size()); ++i) {
            auto key = markers[static_cast<size_t>(i)].joint_index;
            if (joint_to_first_marker.find(key) == joint_to_first_marker.end())
                joint_to_first_marker[key] = i;
        }
        for (int i = 0; i < static_cast<int>(markers.size()); ++i) {
            int parent_marker = -1;
            auto opt_parent = joints[markers[static_cast<size_t>(i)].joint_index].parent_index;
            while (opt_parent.has_value()) {
                auto it = joint_to_first_marker.find(*opt_parent);
                if (it != joint_to_first_marker.end()) {
                    parent_marker = it->second;
                    break;
                }
                opt_parent = joints[*opt_parent].parent_index;
            }
            marker_parent_map[i] = parent_marker;
        }

        // Build all-pairs joint-hop distance matrix for spatial cross-pair selection.
        if (cross_pair_max_px > 0.0) {
            int const n_joints = static_cast<int>(joints.size());
            int const n_markers_i = static_cast<int>(markers.size());
            constexpr int INF = std::numeric_limits<int>::max();

            std::vector<std::vector<int>> adj(n_joints);
            for (int j = 0; j < n_joints; ++j) {
                if (joints[j].parent_index.has_value()) {
                    int p = static_cast<int>(*joints[j].parent_index);
                    adj[j].push_back(p);
                    adj[p].push_back(j);
                }
            }

            std::vector<std::vector<int>> joint_dist(n_joints, std::vector<int>(n_joints, INF));
            for (int src = 0; src < n_joints; ++src) {
                joint_dist[src][src] = 0;
                std::queue<int> q;
                q.push(src);
                while (!q.empty()) {
                    int cur = q.front();
                    q.pop();
                    for (int nb : adj[cur]) {
                        if (joint_dist[src][nb] == INF) {
                            joint_dist[src][nb] = joint_dist[src][cur] + 1;
                            q.push(nb);
                        }
                    }
                }
            }

            marker_dist_matrix.assign(n_markers_i, std::vector<int>(n_markers_i, INF));
            for (int a = 0; a < n_markers_i; ++a) {
                int ja = static_cast<int>(markers[a].joint_index);
                for (int b = 0; b < n_markers_i; ++b) {
                    int jb = static_cast<int>(markers[b].joint_index);
                    marker_dist_matrix[a][b] = joint_dist[ja][jb];
                }
            }
        }
    }

    // Step 2.5: Load all edits for this sequence upfront.
    // If pose_observation_edits does not exist (older DB schema), gracefully skip.
    struct FrameEdit {
        std::vector<uint8_t> kp_blob;
        std::vector<uint8_t> mask;
    };
    std::unordered_map<std::string, std::unordered_map<int, FrameEdit>> edits;
    {
        sqlite3_stmt* edit_raw = nullptr;
        int rc = sqlite3_prepare_v2(db_,
                                    "SELECT camera_instance_id, video_frame, kp_blob, kp_mask"
                                    " FROM pose_observation_edits WHERE sequence_id = ?",
                                    -1, &edit_raw, nullptr);
        if (rc == SQLITE_OK) {
            sqlite3_bind_text(edit_raw, 1, sequence_id.c_str(), -1, SQLITE_STATIC);
            while (sqlite3_step(edit_raw) == SQLITE_ROW) {
                std::string inst = reinterpret_cast<char const*>(sqlite3_column_text(edit_raw, 0));
                int frame = sqlite3_column_int(edit_raw, 1);
                auto const* kb = static_cast<uint8_t const*>(sqlite3_column_blob(edit_raw, 2));
                int kb_n = sqlite3_column_bytes(edit_raw, 2);
                auto const* mb = static_cast<uint8_t const*>(sqlite3_column_blob(edit_raw, 3));
                int mb_n = sqlite3_column_bytes(edit_raw, 3);
                FrameEdit fe;
                fe.kp_blob.assign(kb, kb + kb_n);
                fe.mask.assign(mb, mb + mb_n);
                edits[inst][frame] = std::move(fe);
            }
            sqlite3_finalize(edit_raw);
        }
        // rc != SQLITE_OK means table absent — edits stays empty, silently skipped
    }

    // Step 3: Fetch all observations for this sequence and person
    Stmt obs_stmt(db_,
                  "SELECT po.camera_instance_id, po.video_frame, po.timestamp_s, po.kp_blob,"
                  " COALESCE(po.noise_scale, 1.0) AS crop_scale"
                  " FROM pose_observations po"
                  " WHERE po.sequence_id = ? AND po.person_id = ?"
                  " ORDER BY po.camera_instance_id, po.video_frame");
    sqlite3_bind_text(obs_stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(obs_stmt.ptr, 2, person_id);

    // Accumulate observations per instance_id
    std::unordered_map<std::string, std::vector<Observation>> seq_observations;
    int rows_total = 0;
    int rows_skipped_camera = 0;
    int rows_skipped_confidence = 0;
    int rows_skipped_coco = 0;

    while (obs_stmt.step()) {
        std::string inst_id = reinterpret_cast<char const*>(sqlite3_column_text(obs_stmt.ptr, 0));
        int video_frame = sqlite3_column_int(obs_stmt.ptr, 1);
        double timestamp_s = sqlite3_column_double(obs_stmt.ptr, 2);
        void const* kp_data = sqlite3_column_blob(obs_stmt.ptr, 3);
        int kp_bytes = sqlite3_column_bytes(obs_stmt.ptr, 3);
        double crop_scale = sqlite3_column_double(obs_stmt.ptr, 4);
        ++rows_total;

        // Skip rows whose camera instance is not in the camera map
        auto cam_it = inst_to_cam.find(inst_id);
        if (cam_it == inst_to_cam.end()) {
            ++rows_skipped_camera;
            continue;
        }
        Camera const* camera = cam_it->second;

        // Step 4: Decode keypoints, apply any edits, then create Observation objects
        auto kps = db::decode_keypoints(kp_data, kp_bytes);
        {
            auto edit_inst_it = edits.find(inst_id);
            if (edit_inst_it != edits.end()) {
                auto edit_frame_it = edit_inst_it->second.find(video_frame);
                if (edit_frame_it != edit_inst_it->second.end()) {
                    auto const& fe = edit_frame_it->second;
                    db::apply_keypoint_edits(kps, fe.kp_blob.data(),
                                             static_cast<int>(fe.kp_blob.size()), fe.mask.data(),
                                             static_cast<int>(fe.mask.size()));
                }
            }
        }

        // Collect POSITION observations for this frame/camera first, then generate RELATIVE pairs.
        std::vector<Observation> frame_obs;
        for (int i = 0; i < static_cast<int>(kps.size()); ++i) {
            if (kps[static_cast<size_t>(i)].confidence < static_cast<float>(min_confidence)) {
                ++rows_skipped_confidence;
                continue;
            }
            auto it = coco_to_marker_idx.find(i);
            if (it == coco_to_marker_idx.end()) {
                ++rows_skipped_coco;
                continue;
            }

            Observation obs;
            obs.camera_id = camera->id();
            obs.marker_id = it->second;
            obs.frame_idx = video_frame;
            obs.timestamp = timestamp_s;
            obs.position_distorted =
                Eigen::Vector2d(kps[static_cast<size_t>(i)].x, kps[static_cast<size_t>(i)].y);
            // When pixels_are_undistorted, coordinates are already in K_new space;
            // skip undistortion to avoid applying the distortion model a second time.
            obs.position = pixels_are_undistorted ? obs.position_distorted
                                                  : camera->undistort(obs.position_distorted);
            obs.confidence = kps[static_cast<size_t>(i)].confidence;
            obs.crop_scale = crop_scale;
            frame_obs.push_back(obs);
        }

        auto& dest = seq_observations[inst_id];
        dest.insert(dest.end(), frame_obs.begin(), frame_obs.end());

        // Generate RELATIVE observations: one per (child, parent) pair visible in this frame.
        if (use_relative_obs && pose_noise_std > 0.0 && !frame_obs.empty()) {
            // Build marker_id → observation index for this frame
            std::unordered_map<int, int> marker_to_frame_idx;
            for (int i = 0; i < static_cast<int>(frame_obs.size()); ++i)
                marker_to_frame_idx[frame_obs[i].marker_id] = i;

            for (auto const& child_obs : frame_obs) {
                if (child_obs.confidence < static_cast<float>(relative_min_conf))
                    continue;
                auto parent_it = marker_parent_map.find(child_obs.marker_id);
                if (parent_it == marker_parent_map.end() || parent_it->second < 0)
                    continue;
                int parent_marker_id = parent_it->second;
                auto parent_idx_it = marker_to_frame_idx.find(parent_marker_id);
                if (parent_idx_it == marker_to_frame_idx.end())
                    continue;
                auto const& parent_obs = frame_obs[parent_idx_it->second];
                if (parent_obs.confidence < static_cast<float>(relative_min_conf))
                    continue;

                Observation rel;
                rel.camera_id = child_obs.camera_id;
                rel.marker_id = child_obs.marker_id;
                rel.ref_marker_id = parent_marker_id;
                rel.frame_idx = child_obs.frame_idx;
                rel.timestamp = child_obs.timestamp;
                // Observed measurement = child_pixel - parent_pixel
                rel.position = child_obs.position - parent_obs.position;
                rel.position_distorted =
                    child_obs.position_distorted - parent_obs.position_distorted;
                rel.confidence = std::min(child_obs.confidence, parent_obs.confidence);
                rel.mode = MeasurementMode::PAIR_DIFF;
                rel.crop_scale = child_obs.crop_scale;
                // Noise = pose_noise_std * sqrt(2) * crop_scale (calib error cancels in diff).
                // Baked into noise_std_override so the confidence scaling still applies.
                rel.noise_std_override = pose_noise_std * std::sqrt(2.0) * crop_scale;
                dest.push_back(rel);
            }
        }

        // Generate SPATIAL cross-pair RELATIVE observations (Phase 4):
        // pairs of visible markers close in image space but far in the skeleton tree.
        // Calibration error cancels (same camera, same frame) → noise = ep * sqrt(2) * scale.
        if (cross_pair_max_px > 0.0 && pose_noise_std > 0.0 && frame_obs.size() >= 2) {
            struct Candidate {
                int ai, bi;
                double dist_px;
            };
            std::vector<Candidate> candidates;
            int const n_fo = static_cast<int>(frame_obs.size());
            for (int ai = 0; ai < n_fo; ++ai) {
                for (int bi = ai + 1; bi < n_fo; ++bi) {
                    double d = (frame_obs[ai].position - frame_obs[bi].position).norm();
                    if (d >= cross_pair_max_px)
                        continue;
                    int ma = frame_obs[ai].marker_id;
                    int mb = frame_obs[bi].marker_id;
                    int hdist = (!marker_dist_matrix.empty() &&
                                 ma < static_cast<int>(marker_dist_matrix.size()) &&
                                 mb < static_cast<int>(marker_dist_matrix.size()))
                                    ? marker_dist_matrix[ma][mb]
                                    : std::numeric_limits<int>::max();
                    if (hdist <= 2)
                        continue;
                    candidates.push_back({ai, bi, d});
                }
            }
            std::sort(candidates.begin(), candidates.end(),
                      [](Candidate const& a, Candidate const& b) { return a.dist_px < b.dist_px; });
            if (cross_pair_max_n > 0 && static_cast<int>(candidates.size()) > cross_pair_max_n)
                candidates.resize(static_cast<size_t>(cross_pair_max_n));

            for (auto const& c : candidates) {
                auto const& oa = frame_obs[c.ai];
                auto const& ob = frame_obs[c.bi];
                Observation rel;
                rel.camera_id = oa.camera_id;
                rel.marker_id = oa.marker_id;
                rel.ref_marker_id = ob.marker_id;
                rel.frame_idx = oa.frame_idx;
                rel.timestamp = oa.timestamp;
                rel.position = oa.position - ob.position;
                rel.position_distorted = oa.position_distorted - ob.position_distorted;
                rel.confidence = std::min(oa.confidence, ob.confidence);
                rel.mode = MeasurementMode::PAIR_DIFF;
                rel.crop_scale = oa.crop_scale;
                rel.noise_std_override = pose_noise_std * std::sqrt(2.0) * oa.crop_scale;
                dest.push_back(rel);
            }
        }
    }

    // Step 5: Build ObservationSet — throw a diagnostic error if nothing came through
    if (rows_total == 0) {
        // Query returned no rows — most likely wrong person_id. Show available IDs.
        Stmt pid_stmt(db_,
                      "SELECT DISTINCT person_id FROM pose_observations"
                      " WHERE sequence_id = ? ORDER BY person_id");
        sqlite3_bind_text(pid_stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);
        std::string available;
        while (pid_stmt.step()) {
            if (!available.empty())
                available += ", ";
            available += std::to_string(sqlite3_column_int(pid_stmt.ptr, 0));
        }
        throw std::runtime_error("No pose_observations rows found for sequence '" + sequence_id +
                                 "' with person_id=" + std::to_string(person_id) +
                                 ". Available person_ids: [" +
                                 (available.empty() ? "none" : available) + "]");
    }
    if (seq_observations.empty() && rows_total > 0) {
        throw std::runtime_error(
            "load_observations: " + std::to_string(rows_total) + " DB rows found for sequence '" +
            sequence_id + "' person_id=" + std::to_string(person_id) +
            ", but all were filtered out.\n"
            "  cameras in map: " +
            std::to_string(inst_to_cam.size()) + " (skipped " +
            std::to_string(rows_skipped_camera) +
            " rows with unknown camera)\n"
            "  COCO markers in skeleton: " +
            std::to_string(coco_to_marker_idx.size()) + " (skipped " +
            std::to_string(rows_skipped_coco) +
            " keypoints with unknown COCO id)\n"
            "  skipped " +
            std::to_string(rows_skipped_confidence) + " keypoints below confidence threshold " +
            std::to_string(min_confidence));
    }

    ObservationSet obs_set(person_id);
    for (auto const& [inst_id, obs_list] : seq_observations) {
        auto* cam = inst_to_cam.at(inst_id);
        ObservationSequence seq;
        seq.camera_id = cam->id();
        seq.camera_name = cam->name();
        seq.observations = obs_list;
        obs_set.add_sequence(seq);
    }
    return obs_set;
}

}  // namespace posetrak
