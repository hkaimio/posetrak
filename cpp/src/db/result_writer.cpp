// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <posetrak/db/result_writer.hpp>
#include <posetrak/version.hpp>

#include <fmt/chrono.h>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include <sqlite3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace posetrak {

// ---------------------------------------------------------------------------
// UUID v4 generator
// ---------------------------------------------------------------------------

static std::string generate_uuid_v4() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFFu);

    uint32_t data[4];
    for (auto& d : data)
        d = dist(gen);

    // Set version bits (version 4)
    data[1] = (data[1] & 0xFFFF0FFFu) | 0x00004000u;
    // Set variant bits (variant 10xx)
    data[2] = (data[2] & 0x3FFFFFFFu) | 0x80000000u;

    return fmt::format("{:08x}-{:04x}-{:04x}-{:04x}-{:04x}{:08x}", data[0],
                       (data[1] >> 16) & 0xFFFF, data[1] & 0xFFFF, (data[2] >> 16) & 0xFFFF,
                       data[2] & 0xFFFF, data[3]);
}

// ---------------------------------------------------------------------------
// Current UTC timestamp as ISO-8601 string
// ---------------------------------------------------------------------------

static std::string utc_iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &t);
#else
    gmtime_r(&t, &tm_utc);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm_utc, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}

// ---------------------------------------------------------------------------
// ResultWriter constructor
// ---------------------------------------------------------------------------

ResultWriter::ResultWriter(std::string const& db_path, std::string const& sequence_id,
                           std::string const& skeleton_id, std::string const& config_id,
                           std::string const& extrinsic_calibration_id,
                           std::string const& sync_config_id, int person_id,
                           std::map<std::string, Camera> const& cameras, Skeleton const& skeleton)
    : person_id_(person_id) {
    int rc = sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
                             nullptr);
    if (rc != SQLITE_OK) {
        std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        if (db_)
            sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("ResultWriter: failed to open DB '" + db_path + "': " + err);
    }

    sqlite3_exec(db_, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);

    run_id_ = generate_uuid_v4();

    // Build active_camera_ids JSON array: sorted camera labels
    for (auto const& [label, _] : cameras)
        camera_labels_.push_back(label);
    std::sort(camera_labels_.begin(), camera_labels_.end());
    nlohmann::json cam_json = camera_labels_;
    std::string active_camera_ids = cam_json.dump();

    // Build marker_names JSON array from skeleton markers
    for (auto const& m : skeleton.markers())
        marker_names_.push_back(m.name);
    nlohmann::json mk_json = marker_names_;
    std::string marker_names = mk_json.dump();

    std::string ran_at = utc_iso_timestamp();

    // Insert tracking_runs row
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO tracking_runs"
        " (id, observation_sequence_id, tracker_config_id, skeleton_id,"
        "  extrinsic_calibration_id, sync_config_id, ran_at, posetrak_version,"
        "  active_camera_ids, marker_names)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(std::string("ResultWriter: prepare INSERT tracking_runs: ") +
                                 sqlite3_errmsg(db_));
    }

    sqlite3_bind_text(stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, sequence_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, config_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, skeleton_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 5, extrinsic_calibration_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 6, sync_config_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 7, ran_at.c_str(), -1, SQLITE_TRANSIENT);
    std::string const ver = version_string();
    sqlite3_bind_text(stmt, 8, ver.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 9, active_camera_ids.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 10, marker_names.c_str(), -1, SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ResultWriter: INSERT tracking_runs failed: ") +
                                 sqlite3_errmsg(db_));
    }
}

// ---------------------------------------------------------------------------
// Attach-mode constructor -- patch into an existing tracking run
// ---------------------------------------------------------------------------

ResultWriter::ResultWriter(std::string const& db_path, std::string const& run_id, int person_id)
    : run_id_(run_id), person_id_(person_id) {
    int rc = sqlite3_open_v2(db_path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_FULLMUTEX,
                             nullptr);
    if (rc != SQLITE_OK) {
        std::string err = db_ ? sqlite3_errmsg(db_) : "unknown error";
        if (db_)
            sqlite3_close(db_);
        db_ = nullptr;
        throw std::runtime_error("ResultWriter: failed to open DB '" + db_path + "': " + err);
    }

    sqlite3_exec(db_, "PRAGMA foreign_keys=ON;", nullptr, nullptr, nullptr);

    // Load camera_labels_/marker_names_ from the parent's tracking_runs row so
    // patch_obs_results() can compute obs_blob slot indices without the caller
    // having to re-derive and re-pass the same camera/skeleton metadata.
    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT active_camera_ids, marker_names FROM tracking_runs WHERE id=?";
    rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(std::string("ResultWriter: prepare SELECT tracking_runs: ") +
                                 sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
    rc = sqlite3_step(stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("ResultWriter: no tracking_runs row for run_id '" + run_id_ + "'");
    }
    if (sqlite3_column_type(stmt, 0) != SQLITE_NULL) {
        auto const* txt = reinterpret_cast<char const*>(sqlite3_column_text(stmt, 0));
        camera_labels_ = nlohmann::json::parse(txt).get<std::vector<std::string>>();
    }
    if (sqlite3_column_type(stmt, 1) != SQLITE_NULL) {
        auto const* txt = reinterpret_cast<char const*>(sqlite3_column_text(stmt, 1));
        marker_names_ = nlohmann::json::parse(txt).get<std::vector<std::string>>();
    }
    sqlite3_finalize(stmt);
}

// ---------------------------------------------------------------------------

ResultWriter::~ResultWriter() {
    try {
        flush();
    } catch (...) {
        // Swallow exceptions in destructor
    }
    if (db_) {
        sqlite3_close(db_);
        db_ = nullptr;
    }
}

// ---------------------------------------------------------------------------

std::vector<uint8_t> ResultWriter::encode_vector(Eigen::VectorXd const& v) {
    std::vector<uint8_t> buf(static_cast<size_t>(v.size()) * sizeof(double));
    std::memcpy(buf.data(), v.data(), buf.size());
    return buf;
}

// ---------------------------------------------------------------------------

std::vector<double> ResultWriter::decode_doubles(void const* blob, int n_bytes) {
    if (n_bytes <= 0 || blob == nullptr)
        return {};
    std::vector<double> out(static_cast<size_t>(n_bytes) / sizeof(double));
    std::memcpy(out.data(), blob, out.size() * sizeof(double));
    return out;
}

std::vector<uint8_t> ResultWriter::encode_doubles(std::vector<double> const& v) {
    std::vector<uint8_t> buf(v.size() * sizeof(double));
    std::memcpy(buf.data(), v.data(), buf.size());
    return buf;
}

void ResultWriter::apply_patch(std::vector<double>& vec, std::vector<int> const& indices,
                               std::vector<double> const& values, char const* field_name) {
    if (indices.size() != values.size()) {
        throw std::invalid_argument(std::string("ResultWriter::patch_frame: ") + field_name +
                                    "_indices/values size mismatch");
    }
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        if (idx < 0 || static_cast<size_t>(idx) >= vec.size()) {
            throw std::invalid_argument(std::string("ResultWriter::patch_frame: ") + field_name +
                                        " index " + std::to_string(idx) + " out of range (size " +
                                        std::to_string(vec.size()) + ")");
        }
        vec[static_cast<size_t>(idx)] = values[i];
    }
}

// ---------------------------------------------------------------------------

void ResultWriter::write_frame(int step, double timestamp, Eigen::VectorXd const& state,
                               Eigen::MatrixXd const& covariance, bool tracking_lost,
                               int n_inlier_observations, double cov_condition_number,
                               double nis_value, int nis_dof) {
    auto state_blob = encode_vector(state);

    std::vector<uint8_t> cov_diag_blob;
    if (covariance.size() > 0) {
        Eigen::VectorXd diag = covariance.diagonal();
        cov_diag_blob = encode_vector(diag);
    }

    pending_.emplace_back(step, timestamp, std::move(state_blob), std::move(cov_diag_blob),
                          tracking_lost ? 1 : 0, n_inlier_observations, cov_condition_number,
                          nis_value, nis_dof, 0);

    if (static_cast<int>(pending_.size()) >= kBatchSize)
        flush_pending();
}

// ---------------------------------------------------------------------------

void ResultWriter::write_smoothed_frame(int step, double timestamp, Eigen::VectorXd const& state,
                                        Eigen::MatrixXd const& covariance) {
    auto state_blob = encode_vector(state);

    std::vector<uint8_t> cov_diag_blob;
    if (covariance.size() > 0) {
        Eigen::VectorXd diag = covariance.diagonal();
        cov_diag_blob = encode_vector(diag);
    }

    pending_.emplace_back(step, timestamp, std::move(state_blob), std::move(cov_diag_blob), 0, 0,
                          0.0, 0.0, 0, 1);

    if (static_cast<int>(pending_.size()) >= kBatchSize)
        flush_pending();
}

// ---------------------------------------------------------------------------

void ResultWriter::flush_pending() {
    if (pending_.empty())
        return;

    // Begin transaction
    char* errmsg = nullptr;
    int rc = sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string err = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw std::runtime_error("ResultWriter: BEGIN failed: " + err);
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO tracking_results"
        " (run_id, person_id, tracker_step, is_smoothed, timestamp_s,"
        "  tracking_lost, n_inlier_observations, cov_condition_number,"
        "  nis_value, nis_dof, state, cov_diag)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";

    rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw std::runtime_error(std::string("ResultWriter: prepare INSERT tracking_results: ") +
                                 sqlite3_errmsg(db_));
    }

    bool failed = false;
    std::string fail_msg;

    for (auto const& row : pending_) {
        auto const& [step, ts, state_blob, cov_diag_blob, tl, n_inliers, cov_cond, nis_val, nis_d,
                     is_smoothed] = row;

        sqlite3_bind_text(stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_int(stmt, 2, person_id_);
        sqlite3_bind_int(stmt, 3, step);
        sqlite3_bind_int(stmt, 4, is_smoothed);
        sqlite3_bind_double(stmt, 5, ts);
        sqlite3_bind_int(stmt, 6, tl);
        sqlite3_bind_int(stmt, 7, n_inliers);
        sqlite3_bind_double(stmt, 8, cov_cond);
        sqlite3_bind_double(stmt, 9, nis_val);
        sqlite3_bind_int(stmt, 10, nis_d);

        if (!state_blob.empty())
            sqlite3_bind_blob(stmt, 11, state_blob.data(), static_cast<int>(state_blob.size()),
                              SQLITE_STATIC);
        else
            sqlite3_bind_null(stmt, 11);

        if (!cov_diag_blob.empty())
            sqlite3_bind_blob(stmt, 12, cov_diag_blob.data(),
                              static_cast<int>(cov_diag_blob.size()), SQLITE_STATIC);
        else
            sqlite3_bind_null(stmt, 12);

        int step_rc = sqlite3_step(stmt);
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);

        if (step_rc != SQLITE_DONE) {
            failed = true;
            fail_msg = sqlite3_errmsg(db_);
            break;
        }
    }

    sqlite3_finalize(stmt);

    if (failed) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw std::runtime_error("ResultWriter: INSERT tracking_results failed: " + fail_msg);
    }

    rc = sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, &errmsg);
    if (rc != SQLITE_OK) {
        std::string err = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw std::runtime_error("ResultWriter: COMMIT failed: " + err);
    }

    pending_.clear();
}

// ---------------------------------------------------------------------------

void ResultWriter::write_obs_results(int step, std::vector<ObservationResult> const& observations) {
    if (camera_labels_.empty() || marker_names_.empty())
        return;

    int n_cams = static_cast<int>(camera_labels_.size());
    int n_markers = static_cast<int>(marker_names_.size());
    constexpr int kFields = 8;

    static const float kNaN = std::numeric_limits<float>::quiet_NaN();

    // Build index maps
    std::unordered_map<std::string, int> cam_idx, marker_idx;
    for (int i = 0; i < n_cams; ++i)
        cam_idx[camera_labels_[i]] = i;
    for (int i = 0; i < n_markers; ++i)
        marker_idx[marker_names_[i]] = i;

    // Fill blob with NaN (absent slot)
    std::vector<float> blob(static_cast<size_t>(n_cams * n_markers * kFields), kNaN);
    // Set pad field (index 7) to 0.0 for all slots
    for (int c = 0; c < n_cams; ++c)
        for (int m = 0; m < n_markers; ++m)
            blob[static_cast<size_t>((c * n_markers + m) * kFields + 7)] = 0.0f;

    for (auto const& obs : observations) {
        // camera_id in ObservationResult is the integer index, not a label — look up directly
        if (obs.camera_id < 0 || obs.camera_id >= n_cams)
            continue;
        auto mi = marker_idx.find(obs.marker_name);
        if (mi == marker_idx.end())
            continue;

        int c = obs.camera_id;
        int m = mi->second;
        float* slot = blob.data() + (c * n_markers + m) * kFields;
        slot[0] = static_cast<float>(obs.actual[0]);
        slot[1] = static_cast<float>(obs.actual[1]);
        slot[2] = static_cast<float>(obs.predicted[0]);
        slot[3] = static_cast<float>(obs.predicted[1]);
        slot[4] = static_cast<float>(obs.mahalanobis_distance);
        slot[5] = obs.is_outlier ? 0.0f : 1.0f;  // used_in_update
        slot[6] = obs.is_outlier ? 1.0f : 0.0f;  // is_outlier
        slot[7] = 0.0f;
    }

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT OR REPLACE INTO tracking_obs_results"
        " (run_id, person_id, tracker_step, obs_blob)"
        " VALUES (?, ?, ?, ?)";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK)
        throw std::runtime_error(
            std::string("ResultWriter: prepare INSERT tracking_obs_results: ") +
            sqlite3_errmsg(db_));

    sqlite3_bind_text(stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, person_id_);
    sqlite3_bind_int(stmt, 3, step);
    sqlite3_bind_blob(stmt, 4, blob.data(), static_cast<int>(blob.size() * sizeof(float)),
                      SQLITE_TRANSIENT);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE)
        throw std::runtime_error(std::string("ResultWriter: INSERT tracking_obs_results failed: ") +
                                 sqlite3_errmsg(db_));
}

// ---------------------------------------------------------------------------

void ResultWriter::patch_obs_results(int step, std::vector<ObservationResult> const& observations,
                                     std::vector<uint8_t> const& pair_diff_reconstructed,
                                     std::vector<std::string> const& parent_owned_markers) {
    if (pair_diff_reconstructed.size() != observations.size()) {
        throw std::invalid_argument(
            "ResultWriter::patch_obs_results: pair_diff_reconstructed/observations size "
            "mismatch");
    }
    if (camera_labels_.empty() || marker_names_.empty()) {
        throw std::runtime_error(
            "ResultWriter::patch_obs_results: no camera_labels_/marker_names_ metadata "
            "available (tracking_runs row has no active_camera_ids/marker_names)");
    }

    // Make sure a row this same writer may have just batched is visible to the
    // SELECT below.
    flush_pending();

    int n_cams = static_cast<int>(camera_labels_.size());
    int n_markers = static_cast<int>(marker_names_.size());
    constexpr int kFields = 8;

    std::unordered_map<std::string, int> marker_idx;
    for (int i = 0; i < n_markers; ++i)
        marker_idx[marker_names_[i]] = i;
    std::unordered_set<std::string> const parent_owned(parent_owned_markers.begin(),
                                                       parent_owned_markers.end());

    sqlite3_stmt* select_stmt = nullptr;
    const char* select_sql =
        "SELECT obs_blob FROM tracking_obs_results"
        " WHERE run_id=? AND person_id=? AND tracker_step=?";
    int rc = sqlite3_prepare_v2(db_, select_sql, -1, &select_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("ResultWriter: prepare SELECT tracking_obs_results: ") +
            sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(select_stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(select_stmt, 2, person_id_);
    sqlite3_bind_int(select_stmt, 3, step);

    rc = sqlite3_step(select_stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(select_stmt);
        throw std::runtime_error(fmt::format(
            "ResultWriter::patch_obs_results: no tracking_obs_results row for run_id={}, "
            "person_id={}, tracker_step={}",
            run_id_, person_id_, step));
    }

    int n_bytes = sqlite3_column_bytes(select_stmt, 0);
    size_t const expected_bytes =
        static_cast<size_t>(n_cams) * static_cast<size_t>(n_markers) * kFields * sizeof(float);
    if (static_cast<size_t>(n_bytes) != expected_bytes) {
        sqlite3_finalize(select_stmt);
        throw std::runtime_error(fmt::format(
            "ResultWriter::patch_obs_results: obs_blob size {} does not match expected {} "
            "({} cameras x {} markers x {} fields)",
            n_bytes, expected_bytes, n_cams, n_markers, kFields));
    }
    std::vector<float> blob(static_cast<size_t>(n_cams) * static_cast<size_t>(n_markers) * kFields);
    std::memcpy(blob.data(), sqlite3_column_blob(select_stmt, 0), expected_bytes);
    sqlite3_finalize(select_stmt);

    for (size_t i = 0; i < observations.size(); ++i) {
        ObservationResult const& obs = observations[i];
        if (parent_owned.count(obs.marker_name) != 0)
            continue;  // parent-wins: shared-marker slots are never overwritten by a child
        if (obs.camera_id < 0 || obs.camera_id >= n_cams)
            continue;
        auto mi = marker_idx.find(obs.marker_name);
        if (mi == marker_idx.end())
            continue;

        int c = obs.camera_id;
        int m = mi->second;
        float* slot = blob.data() + (c * n_markers + m) * kFields;
        slot[0] = static_cast<float>(obs.actual[0]);
        slot[1] = static_cast<float>(obs.actual[1]);
        slot[2] = static_cast<float>(obs.predicted[0]);
        slot[3] = static_cast<float>(obs.predicted[1]);
        slot[4] = static_cast<float>(obs.mahalanobis_distance);
        slot[5] = obs.is_outlier ? 0.0f : 1.0f;  // used_in_update
        slot[6] = obs.is_outlier ? 1.0f : 0.0f;  // is_outlier
        slot[7] = pair_diff_reconstructed[i] ? 1.0f : 0.0f;
    }

    sqlite3_stmt* update_stmt = nullptr;
    const char* update_sql =
        "UPDATE tracking_obs_results SET obs_blob=?"
        " WHERE run_id=? AND person_id=? AND tracker_step=?";
    rc = sqlite3_prepare_v2(db_, update_sql, -1, &update_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(
            std::string("ResultWriter: prepare UPDATE tracking_obs_results: ") +
            sqlite3_errmsg(db_));
    }
    sqlite3_bind_blob(update_stmt, 1, blob.data(), static_cast<int>(blob.size() * sizeof(float)),
                      SQLITE_TRANSIENT);
    sqlite3_bind_text(update_stmt, 2, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(update_stmt, 3, person_id_);
    sqlite3_bind_int(update_stmt, 4, step);

    rc = sqlite3_step(update_stmt);
    sqlite3_finalize(update_stmt);
    if (rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ResultWriter: UPDATE tracking_obs_results failed: ") +
                                 sqlite3_errmsg(db_));
    }
}

// ---------------------------------------------------------------------------

void ResultWriter::patch_frame(int step, bool is_smoothed, std::vector<int> const& state_indices,
                               std::vector<double> const& state_values,
                               std::vector<int> const& cov_diag_indices,
                               std::vector<double> const& cov_diag_values) {
    // Make sure a row this same writer may have just batched is visible to the
    // SELECT below.
    flush_pending();

    sqlite3_stmt* select_stmt = nullptr;
    const char* select_sql =
        "SELECT state, cov_diag FROM tracking_results"
        " WHERE run_id=? AND person_id=? AND tracker_step=? AND is_smoothed=?";
    int rc = sqlite3_prepare_v2(db_, select_sql, -1, &select_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(std::string("ResultWriter: prepare SELECT tracking_results: ") +
                                 sqlite3_errmsg(db_));
    }
    sqlite3_bind_text(select_stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(select_stmt, 2, person_id_);
    sqlite3_bind_int(select_stmt, 3, step);
    sqlite3_bind_int(select_stmt, 4, is_smoothed ? 1 : 0);

    rc = sqlite3_step(select_stmt);
    if (rc != SQLITE_ROW) {
        sqlite3_finalize(select_stmt);
        throw std::runtime_error(fmt::format(
            "ResultWriter::patch_frame: no tracking_results row for run_id={}, person_id={}, "
            "tracker_step={}, is_smoothed={}",
            run_id_, person_id_, step, is_smoothed ? 1 : 0));
    }

    bool state_is_null = sqlite3_column_type(select_stmt, 0) == SQLITE_NULL;
    bool cov_is_null = sqlite3_column_type(select_stmt, 1) == SQLITE_NULL;
    std::vector<double> state_vec =
        decode_doubles(sqlite3_column_blob(select_stmt, 0), sqlite3_column_bytes(select_stmt, 0));
    std::vector<double> cov_vec =
        decode_doubles(sqlite3_column_blob(select_stmt, 1), sqlite3_column_bytes(select_stmt, 1));
    sqlite3_finalize(select_stmt);

    if (state_is_null && !state_indices.empty()) {
        throw std::invalid_argument(
            "ResultWriter::patch_frame: cannot patch state, row's state is NULL");
    }
    if (cov_is_null && !cov_diag_indices.empty()) {
        throw std::invalid_argument(
            "ResultWriter::patch_frame: cannot patch cov_diag, row's cov_diag is NULL");
    }

    apply_patch(state_vec, state_indices, state_values, "state");
    apply_patch(cov_vec, cov_diag_indices, cov_diag_values, "cov_diag");

    auto state_blob = encode_doubles(state_vec);
    auto cov_blob = encode_doubles(cov_vec);

    sqlite3_stmt* update_stmt = nullptr;
    const char* update_sql =
        "UPDATE tracking_results SET state=?, cov_diag=?"
        " WHERE run_id=? AND person_id=? AND tracker_step=? AND is_smoothed=?";
    rc = sqlite3_prepare_v2(db_, update_sql, -1, &update_stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(std::string("ResultWriter: prepare UPDATE tracking_results: ") +
                                 sqlite3_errmsg(db_));
    }

    if (!state_is_null)
        sqlite3_bind_blob(update_stmt, 1, state_blob.data(), static_cast<int>(state_blob.size()),
                          SQLITE_STATIC);
    else
        sqlite3_bind_null(update_stmt, 1);

    if (!cov_is_null)
        sqlite3_bind_blob(update_stmt, 2, cov_blob.data(), static_cast<int>(cov_blob.size()),
                          SQLITE_STATIC);
    else
        sqlite3_bind_null(update_stmt, 2);

    sqlite3_bind_text(update_stmt, 3, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(update_stmt, 4, person_id_);
    sqlite3_bind_int(update_stmt, 5, step);
    sqlite3_bind_int(update_stmt, 6, is_smoothed ? 1 : 0);

    rc = sqlite3_step(update_stmt);
    sqlite3_finalize(update_stmt);
    if (rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ResultWriter: UPDATE tracking_results failed: ") +
                                 sqlite3_errmsg(db_));
    }
}

// ---------------------------------------------------------------------------

void ResultWriter::set_stage_status(std::string const& group_name, std::string const& status,
                                    bool set_started, bool set_completed) {
    std::string const now = utc_iso_timestamp();

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT INTO tracking_run_stages"
        " (run_id, person_id, group_name, status, started_at, completed_at)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(run_id, person_id, group_name) DO UPDATE SET"
        "   status = excluded.status,"
        "   started_at = COALESCE(excluded.started_at, tracking_run_stages.started_at),"
        "   completed_at = COALESCE(excluded.completed_at, tracking_run_stages.completed_at)";

    int rc = sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        throw std::runtime_error(std::string("ResultWriter: prepare UPSERT tracking_run_stages: ") +
                                 sqlite3_errmsg(db_));
    }

    sqlite3_bind_text(stmt, 1, run_id_.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, person_id_);
    sqlite3_bind_text(stmt, 3, group_name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, status.c_str(), -1, SQLITE_TRANSIENT);
    if (set_started)
        sqlite3_bind_text(stmt, 5, now.c_str(), -1, SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt, 5);
    if (set_completed)
        sqlite3_bind_text(stmt, 6, now.c_str(), -1, SQLITE_TRANSIENT);
    else
        sqlite3_bind_null(stmt, 6);

    rc = sqlite3_step(stmt);
    sqlite3_finalize(stmt);
    if (rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ResultWriter: UPSERT tracking_run_stages failed: ") +
                                 sqlite3_errmsg(db_));
    }
}

// ---------------------------------------------------------------------------

void ResultWriter::flush() {
    flush_pending();
}

}  // namespace posetrak
