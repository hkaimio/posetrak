#include <posetrak/db/result_writer.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <sqlite3.h>

#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace posetrak;
using Catch::Matchers::WithinAbs;

namespace {

namespace fs = std::filesystem;

void exec_sql(sqlite3* db, std::string const& sql) {
    char* errmsg = nullptr;
    if (sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errmsg) != SQLITE_OK) {
        std::string err = errmsg ? errmsg : "unknown";
        sqlite3_free(errmsg);
        throw std::runtime_error("exec_sql failed: " + err + " (" + sql + ")");
    }
}

std::vector<uint8_t> encode_doubles(std::vector<double> const& v) {
    std::vector<uint8_t> buf(v.size() * sizeof(double));
    std::memcpy(buf.data(), v.data(), buf.size());
    return buf;
}

std::vector<double> decode_doubles(void const* blob, int n_bytes) {
    std::vector<double> out(static_cast<size_t>(n_bytes) / sizeof(double));
    std::memcpy(out.data(), blob, out.size() * sizeof(double));
    return out;
}

/// Minimal DB with just the two tables patch_frame() touches, plus one
/// pre-existing tracking_results row to patch. is_smoothed=0 row has both
/// state and cov_diag; is_smoothed=1 row has state but a NULL cov_diag,
/// mirroring write_smoothed_frame()'s "empty MatrixXd -> NULL" convention.
fs::path make_fixture_db(fs::path const& path, std::string const& run_id, int person_id, int step,
                         std::vector<double> const& state, std::vector<double> const& cov_diag) {
    if (fs::exists(path))
        fs::remove(path);

    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                            nullptr) == SQLITE_OK);
    exec_sql(db, "PRAGMA foreign_keys=ON;");

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

    exec_sql(db,
             "INSERT INTO tracking_runs (id,observation_sequence_id,tracker_config_id,"
             "skeleton_id,ran_at) VALUES ('" +
                 run_id + "','seq1','tc1','skel1','2026-01-01');");

    auto state_blob = encode_doubles(state);
    sqlite3_stmt* stmt = nullptr;
    std::string sql =
        "INSERT INTO tracking_results (run_id,person_id,tracker_step,is_smoothed,timestamp_s,"
        "tracking_lost,state,cov_diag) VALUES (?,?,?,0,0.0,0,?,?)";
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, person_id);
    sqlite3_bind_int(stmt, 3, step);
    sqlite3_bind_blob(stmt, 4, state_blob.data(), static_cast<int>(state_blob.size()),
                      SQLITE_TRANSIENT);
    auto cov_blob = encode_doubles(cov_diag);
    if (!cov_diag.empty()) {
        sqlite3_bind_blob(stmt, 5, cov_blob.data(), static_cast<int>(cov_blob.size()),
                          SQLITE_TRANSIENT);
    } else {
        sqlite3_bind_null(stmt, 5);
    }
    REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    // Smoothed-family row: state present, cov_diag NULL (as write_smoothed_frame() would
    // produce when covariance is unavailable).
    sql =
        "INSERT INTO tracking_results (run_id,person_id,tracker_step,is_smoothed,timestamp_s,"
        "tracking_lost,state,cov_diag) VALUES (?,?,?,1,0.0,0,?,NULL)";
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, person_id);
    sqlite3_bind_int(stmt, 3, step);
    sqlite3_bind_blob(stmt, 4, state_blob.data(), static_cast<int>(state_blob.size()),
                      SQLITE_TRANSIENT);
    REQUIRE(sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);

    sqlite3_close(db);
    return path;
}

std::vector<double> read_row(fs::path const& path, std::string const& run_id, int person_id,
                             int step, int is_smoothed, std::vector<double>* cov_out = nullptr,
                             bool* cov_is_null_out = nullptr) {
    sqlite3* db = nullptr;
    REQUIRE(sqlite3_open_v2(path.string().c_str(), &db, SQLITE_OPEN_READONLY, nullptr) ==
            SQLITE_OK);
    sqlite3_stmt* stmt = nullptr;
    std::string sql =
        "SELECT state, cov_diag FROM tracking_results WHERE run_id=? AND person_id=? AND "
        "tracker_step=? AND is_smoothed=?";
    sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
    sqlite3_bind_text(stmt, 1, run_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int(stmt, 2, person_id);
    sqlite3_bind_int(stmt, 3, step);
    sqlite3_bind_int(stmt, 4, is_smoothed);
    REQUIRE(sqlite3_step(stmt) == SQLITE_ROW);
    auto state = decode_doubles(sqlite3_column_blob(stmt, 0), sqlite3_column_bytes(stmt, 0));
    if (cov_is_null_out)
        *cov_is_null_out = sqlite3_column_type(stmt, 1) == SQLITE_NULL;
    if (cov_out)
        *cov_out = decode_doubles(sqlite3_column_blob(stmt, 1), sqlite3_column_bytes(stmt, 1));
    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return state;
}

}  // namespace

TEST_CASE("ResultWriter::patch_frame overwrites only the given state indices",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_state.db";
    std::string run_id = "run1";
    std::vector<double> initial_state = {1.0, 2.0, 3.0, 4.0, 5.0};
    make_fixture_db(path, run_id, /*person_id=*/0, /*step=*/7, initial_state, {0.1, 0.2, 0.3});

    {
        ResultWriter writer(path.string(), run_id, /*person_id=*/0);
        writer.patch_frame(/*step=*/7, /*is_smoothed=*/false, /*state_indices=*/{1, 3},
                           /*state_values=*/{20.0, 40.0});
    }  // ~ResultWriter() closes the DB handle before the fs::remove() below.

    auto patched = read_row(path, run_id, 0, 7, 0);
    REQUIRE(patched.size() == 5);
    CHECK_THAT(patched[0], WithinAbs(1.0, 1e-12));
    CHECK_THAT(patched[1], WithinAbs(20.0, 1e-12));
    CHECK_THAT(patched[2], WithinAbs(3.0, 1e-12));
    CHECK_THAT(patched[3], WithinAbs(40.0, 1e-12));
    CHECK_THAT(patched[4], WithinAbs(5.0, 1e-12));

    fs::remove(path);
}

TEST_CASE("ResultWriter::patch_frame patches state and cov_diag together",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_both.db";
    std::string run_id = "run1";
    make_fixture_db(path, run_id, 0, 3, {1.0, 2.0, 3.0}, {0.1, 0.2, 0.3});

    {
        ResultWriter writer(path.string(), run_id, 0);
        writer.patch_frame(3, false, {0, 2}, {100.0, 300.0}, {1}, {0.99});
    }

    std::vector<double> cov;
    auto state = read_row(path, run_id, 0, 3, 0, &cov);
    CHECK_THAT(state[0], WithinAbs(100.0, 1e-12));
    CHECK_THAT(state[1], WithinAbs(2.0, 1e-12));
    CHECK_THAT(state[2], WithinAbs(300.0, 1e-12));
    REQUIRE(cov.size() == 3);
    CHECK_THAT(cov[0], WithinAbs(0.1, 1e-12));
    CHECK_THAT(cov[1], WithinAbs(0.99, 1e-12));
    CHECK_THAT(cov[2], WithinAbs(0.3, 1e-12));

    fs::remove(path);
}

TEST_CASE("ResultWriter::patch_frame patches the smoothed-family row independently",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_smoothed.db";
    std::string run_id = "run1";
    make_fixture_db(path, run_id, 0, 5, {1.0, 2.0, 3.0}, {0.1, 0.2, 0.3});

    {
        ResultWriter writer(path.string(), run_id, 0);
        writer.patch_frame(5, /*is_smoothed=*/true, {0}, {999.0});
    }

    // Smoothed row patched...
    auto smoothed = read_row(path, run_id, 0, 5, 1);
    CHECK_THAT(smoothed[0], WithinAbs(999.0, 1e-12));
    // ...filtered row (is_smoothed=0) untouched.
    auto filtered = read_row(path, run_id, 0, 5, 0);
    CHECK_THAT(filtered[0], WithinAbs(1.0, 1e-12));

    fs::remove(path);
}

TEST_CASE("ResultWriter::patch_frame throws on out-of-range index",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_oob.db";
    std::string run_id = "run1";
    make_fixture_db(path, run_id, 0, 1, {1.0, 2.0}, {0.1, 0.2});

    {
        ResultWriter writer(path.string(), run_id, 0);
        CHECK_THROWS_AS(writer.patch_frame(1, false, {5}, {1.0}), std::invalid_argument);
    }

    fs::remove(path);
}

TEST_CASE("ResultWriter::patch_frame throws on mismatched indices/values length",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_mismatch.db";
    std::string run_id = "run1";
    make_fixture_db(path, run_id, 0, 1, {1.0, 2.0}, {0.1, 0.2});

    {
        ResultWriter writer(path.string(), run_id, 0);
        CHECK_THROWS_AS(writer.patch_frame(1, false, {0, 1}, {1.0}), std::invalid_argument);
    }

    fs::remove(path);
}

TEST_CASE("ResultWriter::patch_frame throws when patching cov_diag on a NULL cov_diag row",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_null_cov.db";
    std::string run_id = "run1";
    make_fixture_db(path, run_id, 0, 9, {1.0, 2.0}, {0.1, 0.2});

    {
        ResultWriter writer(path.string(), run_id, 0);
        // is_smoothed=1 row was seeded with a NULL cov_diag.
        CHECK_THROWS_AS(writer.patch_frame(9, true, {}, {}, {0}, {1.0}), std::invalid_argument);
    }

    fs::remove(path);
}

TEST_CASE("ResultWriter::patch_frame throws when no matching row exists",
          "[result_writer][patch_frame]") {
    fs::path path = fs::temp_directory_path() / "test_result_writer_patch_missing.db";
    std::string run_id = "run1";
    make_fixture_db(path, run_id, 0, 1, {1.0, 2.0}, {0.1, 0.2});

    {
        ResultWriter writer(path.string(), run_id, 0);
        CHECK_THROWS_AS(writer.patch_frame(/*step=*/999, false, {0}, {1.0}), std::runtime_error);
    }

    fs::remove(path);
}
