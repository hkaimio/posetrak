#include <posetrak/db/blob_codec.hpp>
#include <posetrak/db/session_reader.hpp>

#include <Eigen/Geometry>

#include <sqlite3.h>

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
              "       min_cameras_for_init, process_noise_vel_std, velocity_half_life_s"
              " FROM tracker_configs WHERE id = ?");
    sqlite3_bind_text(stmt.ptr, 1, config_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("TrackerConfig not found: " + config_id);
    }

    DbTrackerConfig out;
    // Columns: 0=alpha, 1=beta, 2=kappa, 3=process_noise_std, 4=measurement_noise_std,
    //          5=outlier_threshold, 6=tracker_fps, 7=ik_max_iterations, 8=ik_tolerance,
    //          9=init_position_std, 10=init_orientation_std, 11=init_joint_std,
    //         12=init_velocity_std, 13=min_cameras_for_init, 14=process_noise_vel_std,
    //         15=velocity_half_life_s

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
    apply_real(4, out.tracker.measurement_noise_std);
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
              " JOIN shots s ON s.id = pos.shot_id"
              " WHERE pos.id = ?");
    sqlite3_bind_text(stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("pose_observation_sequence not found: " + sequence_id);
    }

    SequenceMetadata meta;
    meta.session_id = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 0));
    meta.extrinsic_calibration_id = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 1));
    meta.sync_config_id = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 2));
    return meta;
}

// ---------------------------------------------------------------------------

std::map<std::string, Camera>
SessionReader::load_cameras_for_sequence(std::string const& sequence_id) {
    Stmt stmt(db_,
              "SELECT s.session_id, s.extrinsic_calibration_id, pos.sync_config_id"
              " FROM pose_observation_sequences pos"
              " JOIN shots s ON s.id = pos.shot_id"
              " WHERE pos.id = ?");
    sqlite3_bind_text(stmt.ptr, 1, sequence_id.c_str(), -1, SQLITE_STATIC);

    if (!stmt.step()) {
        throw std::runtime_error("pose_observation_sequence not found: " + sequence_id);
    }

    std::string session_id = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 0));
    std::string extrinsics_id = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 1));
    std::string sync_id = reinterpret_cast<char const*>(sqlite3_column_text(stmt.ptr, 2));

    return load_cameras(session_id, extrinsics_id, sync_id);
}

// ---------------------------------------------------------------------------

std::map<std::string, Camera>
SessionReader::load_cameras(std::string const& session_id,
                            std::string const& extrinsic_calibration_id,
                            std::string const& sync_config_id) {
    // Step 1: Fetch camera rows ordered by label for deterministic int ID assignment
    Stmt cam_stmt(db_,
                  "SELECT ci.id, ci.label,"
                  "       ic.fx, ic.fy, ic.cx, ic.cy, ic.dist_coeffs, ic.distortion_model,"
                  "       cm.width_px, cm.height_px, ic.matrix_original"
                  " FROM session_cameras sc"
                  " JOIN camera_instances ci ON ci.id = sc.camera_instance_id"
                  " JOIN intrinsics_calibrations ic ON ic.id = sc.intrinsics_calibration_id"
                  " JOIN camera_modes cm ON cm.id = sc.camera_mode_id"
                  " WHERE sc.session_id = ?"
                  " ORDER BY ci.label ASC");
    sqlite3_bind_text(cam_stmt.ptr, 1, session_id.c_str(), -1, SQLITE_STATIC);

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

    while (cam_stmt.step()) {
        CamRow row;
        row.instance_id = reinterpret_cast<char const*>(sqlite3_column_text(cam_stmt.ptr, 0));
        row.label = reinterpret_cast<char const*>(sqlite3_column_text(cam_stmt.ptr, 1));
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
                   " JOIN shot_videos sv ON sv.id = sp.shot_video_id"
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
                                                int person_id) {
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

    // Step 1: Build instance_id → Camera const* map
    Stmt inst_stmt(db_,
                   "SELECT ci.id, ci.label"
                   " FROM pose_observation_sequences pos"
                   " JOIN shots s ON s.id = pos.shot_id"
                   " JOIN session_cameras sc ON sc.session_id = s.session_id"
                   " JOIN camera_instances ci ON ci.id = sc.camera_instance_id"
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

    // Step 3: Fetch all observations for this sequence and person
    Stmt obs_stmt(db_,
                  "SELECT po.camera_instance_id, po.video_frame, po.timestamp_s, po.kp_blob"
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
        ++rows_total;

        // Skip rows whose camera instance is not in the camera map
        auto cam_it = inst_to_cam.find(inst_id);
        if (cam_it == inst_to_cam.end()) {
            ++rows_skipped_camera;
            continue;
        }
        Camera const* camera = cam_it->second;

        // Step 4: Decode keypoints and create Observation objects
        auto kps = db::decode_keypoints(kp_data, kp_bytes);
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
            seq_observations[inst_id].push_back(obs);
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
