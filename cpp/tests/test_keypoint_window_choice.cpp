// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include <Eigen/Geometry>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "posetrak/tracking/keypoint_window_choice.hpp"
#include <cmath>

using namespace posetrak::kpchoice;

namespace {

constexpr double kDt = 1.0 / 100.0;
constexpr double kBox = 100.0;  // box height in pixels

/// One camera, `n_frames` frames, `n_keypoints` keypoints; keypoints 0 and 1 are neighbours.
Input make_input(int n_frames, int n_keypoints = 3, int n_cameras = 1) {
    Input in;
    in.n_keypoints = n_keypoints;
    in.frames.assign(static_cast<size_t>(n_cameras), {});
    for (auto& cam : in.frames)
        for (int f = 0; f < n_frames; ++f)
            cam.push_back({f, f * kDt, kBox});
    in.active.assign(static_cast<size_t>(n_cameras), 1);
    in.projection.assign(static_cast<size_t>(n_cameras), Eigen::Matrix<double, 3, 4>::Identity());
    in.neighbours.assign(static_cast<size_t>(n_keypoints), {});
    if (n_keypoints >= 2) {
        in.neighbours[0] = {{1, 0.1}};
        in.neighbours[1] = {{0, 0.1}};
    }
    in.own_mask_label = 1;
    return in;
}

Candidate cand(int camera, int keypoint, int frame, double x, double y, int tracklet, int rank = 0,
               float peak = 0.5f) {
    Candidate c;
    c.camera = static_cast<std::uint8_t>(camera);
    c.keypoint = static_cast<std::uint8_t>(keypoint);
    c.frame = frame;
    c.x = c.ux = static_cast<float>(x);
    c.y = c.uy = static_cast<float>(y);
    c.tracklet = tracklet;
    c.rank = static_cast<std::int16_t>(rank);
    c.peak = peak;
    return c;
}

Settings plain() {
    Settings s;
    s.delta = 0.0;  // one camera: no cross-camera support
    s.beta = 0.0;
    s.gamma = 0.0;
    s.support_window_s = 2.5 * kDt;
    s.rounds = 3;
    return s;
}

/// Tracklet id of the pick of a keypoint-frame, -1 for none.
int picked_tracklet(WindowChoice const& w, int camera, int keypoint, int frame) {
    int const r = w.pick(camera, keypoint, frame);
    return r < 0 ? -1 : w.candidates()[static_cast<size_t>(r)].tracklet;
}

}  // namespace

TEST_CASE("keypoint choice: a rejected tracklet range is never picked", "[kpchoice]") {
    int const n = 12;
    auto in = make_input(n);
    for (int f = 0; f < n; ++f) {
        // Tracklet 1 is the strongest maximum and the neighbour's pick supports it; it is rejected
        // over frames 3..8 and there only tracklet 2 is left.
        auto strong = cand(0, 1, f, 100, 100, 1, 0, 0.9f);
        strong.removed = f >= 3 && f <= 8;
        in.candidates.push_back(strong);
        in.candidates.push_back(cand(0, 1, f, 300, 300, 2, 1, 0.1f));
        in.candidates.push_back(cand(0, 0, f, 100, 110, 10));
    }
    auto s = plain();
    s.gamma = 0.2;
    WindowChoice w(std::move(in), s);
    w.run();
    for (int f = 0; f < n; ++f) {
        if (f >= 3 && f <= 8)
            CHECK(picked_tracklet(w, 0, 1, f) == 2);
        else
            CHECK(picked_tracklet(w, 0, 1, f) == 1);
    }
}

TEST_CASE("keypoint choice: an assigned tracklet is always picked", "[kpchoice]") {
    int const n = 10;
    auto in = make_input(n);
    for (int f = 0; f < n; ++f) {
        in.candidates.push_back(cand(0, 1, f, 100, 100, 1, 0, 0.9f));   // supported and strong
        in.candidates.push_back(cand(0, 1, f, 400, 400, 2, 1, 0.05f));  // far from everything
        in.candidates.push_back(cand(0, 0, f, 100, 105, 10));
    }
    in.fixed.push_back({0, 1, 4, 2});
    in.fixed.push_back({0, 1, 5, 2});
    auto s = plain();
    s.gamma = 0.2;
    WindowChoice w(std::move(in), s);
    w.run();
    for (int f = 0; f < n; ++f)
        CHECK(picked_tracklet(w, 0, 1, f) == ((f == 4 || f == 5) ? 2 : 1));
}

TEST_CASE("keypoint choice: a neighbour's pick changes a pick only through the support term",
          "[kpchoice]") {
    int const n = 8;
    auto build = [&](double kp0_x) {
        auto in = make_input(n, 4);
        for (int f = 0; f < n; ++f) {
            in.candidates.push_back(
                cand(0, 0, f, kp0_x, 200, 10));  // the neighbour's only candidate
            // Keypoint 1 has two maxima: one near x = 100, one near x = 300.
            in.candidates.push_back(cand(0, 1, f, 100, 200, 1, 0, 0.5f));
            in.candidates.push_back(cand(0, 1, f, 300, 200, 2, 1, 0.5f));
            // Keypoints 2 and 3 are not neighbours of keypoint 0 (3 has none at all).
            in.candidates.push_back(cand(0, 2, f, 100, 200, 3, 0, 0.5f));
            in.candidates.push_back(cand(0, 2, f, 300, 200, 4, 1, 0.5f));
        }
        in.neighbours[2] = {};
        auto s = plain();
        WindowChoice w(std::move(in), s);
        w.run();
        return w;
    };
    auto near_left = build(110.0);
    auto near_right = build(310.0);
    for (int f = 0; f < n; ++f) {
        CHECK(picked_tracklet(near_left, 0, 1, f) == 1);
        CHECK(picked_tracklet(near_right, 0, 1, f) == 2);
        // Keypoint 2 has no neighbours, so the neighbour's position changes nothing there: the
        // tie goes to the first maximum in both.
        CHECK(picked_tracklet(near_left, 0, 2, f) == picked_tracklet(near_right, 0, 2, f));
    }
}

TEST_CASE("keypoint choice: the strict filter can empty a keypoint-frame and then there is no pick",
          "[kpchoice]") {
    int const n = 30;
    auto in = make_input(n);
    for (int f = 0; f < n; ++f) {
        auto on = cand(0, 1, f, 100, 100, 1);
        on.mask_label = 1;  // on the person's mask
        auto off = cand(0, 1, f, 300, 100, 2, 1);
        off.mask_label = 2;  // on another person's
        in.candidates.push_back(on);
        in.candidates.push_back(off);
        auto lonely = cand(0, 2, f, 50, 50, 3);
        lonely.mask_label = 2;  // the only candidate of its keypoint, and off the mask
        in.candidates.push_back(lonely);
    }
    auto s = plain();
    WindowChoice w(std::move(in), s);
    auto stats = w.apply_strict_filter(10, 0.5);
    w.run();
    CHECK(stats.keypoint_frames == 2 * n);
    CHECK(stats.emptied == n);
    CHECK(stats.candidates_removed == 2 * n);
    for (int f = 0; f < n; ++f) {
        CHECK(picked_tracklet(w, 0, 1, f) == 1);
        CHECK(w.pick(0, 2, f) == -1);
    }
}

TEST_CASE("keypoint choice: an unknown mask is not counted by the strict filter", "[kpchoice]") {
    auto in = make_input(5);
    for (int f = 0; f < 5; ++f) {
        auto c = cand(0, 1, f, 100, 100, 1);
        c.mask_label = kMaskUnknown;
        in.candidates.push_back(c);
    }
    WindowChoice w(std::move(in), plain());
    auto stats = w.apply_strict_filter(10, 0.5);
    CHECK(stats.candidates_removed == 0);
}

TEST_CASE("keypoint choice: window averages are truncated, not padded, at the ends of the range",
          "[kpchoice]") {
    // Tracklet 1 lies at the neighbour's pick in frames 0..2 and far from it later; tracklet 2 is
    // 0.6 supported throughout. The window is +-2 frames. At frame 0 the window holds frames 0..2,
    // so tracklet 1 averages 1.0; padded with zeros it would average 0.6 and tie with tracklet 2,
    // which comes first in rank order and would win.
    int const n = 10;
    auto in = make_input(n);
    double const limit = 0.1 * kBox;
    double const decay_to_06 = 0.05 * kBox * -std::log(0.6);
    for (int f = 0; f < n; ++f) {
        in.candidates.push_back(cand(0, 0, f, 100, 100, 10));
        in.candidates.push_back(cand(0, 1, f, 100 + limit + decay_to_06, 100, 2, 0));
        in.candidates.push_back(cand(0, 1, f, f <= 2 ? 100 : 100 + 10 * kBox, 100, 1, 1));
    }
    for (int f = 0; f < n; ++f)
        in.fixed.push_back({0, 0, f, 10});
    WindowChoice w(std::move(in), plain());
    w.run();
    CHECK(picked_tracklet(w, 0, 1, 0) == 1);
    CHECK(picked_tracklet(w, 0, 1, 1) == 1);  // 0.75 against 0.6
    CHECK(picked_tracklet(w, 0, 1, 5) == 2);
    CHECK(picked_tracklet(w, 0, 1, n - 1) == 2);
}

TEST_CASE("keypoint choice: unlinked candidates are tracklets of their own", "[kpchoice]") {
    // Two unlinked candidates must not share support through a common id.
    int const n = 6;
    auto in = make_input(n);
    for (int f = 0; f < n; ++f) {
        in.candidates.push_back(cand(0, 0, f, 100, 100, 10));
        in.candidates.push_back(cand(0, 1, f, 100 + 10 * kBox, 100, -1, 0));
        in.candidates.push_back(cand(0, 1, f, 100, 100, -1, 1));
    }
    for (int f = 0; f < n; ++f)
        in.fixed.push_back({0, 0, f, 10});
    WindowChoice w(std::move(in), plain());
    w.run();
    for (int f = 0; f < n; ++f)
        CHECK(w.candidates()[static_cast<size_t>(w.pick(0, 1, f))].rank == 1);
}

namespace {

/// Camera looking at the origin from `position`, undistorted pixels, focal length 1000.
Eigen::Matrix<double, 3, 4> look_at(Eigen::Vector3d const& position) {
    Eigen::Vector3d z = (-position).normalized();
    Eigen::Vector3d x = z.cross(Eigen::Vector3d::UnitZ()).normalized();
    Eigen::Vector3d y = z.cross(x);
    Eigen::Matrix3d R;
    R.row(0) = x;
    R.row(1) = y;
    R.row(2) = z;
    Eigen::Matrix3d K;
    K << 1000, 0, 500, 0, 1000, 500, 0, 0, 1;
    Eigen::Matrix<double, 3, 4> P;
    P.leftCols<3>() = K * R;
    P.col(3) = K * (-R * position);
    return P;
}

Eigen::Vector2d project(Eigen::Matrix<double, 3, 4> const& P, Eigen::Vector3d const& X) {
    Eigen::Vector3d p = P * X.homogeneous();
    return {p.x() / p.z(), p.y() / p.z()};
}

}  // namespace

TEST_CASE("keypoint choice: consensus flags the candidates that triangulate together",
          "[kpchoice]") {
    int const n = 3;
    auto in = make_input(n, 1, 4);
    in.neighbours.assign(1, {});
    std::vector<Eigen::Vector3d> positions = {{4, 0, 1}, {0, 4, 1}, {-4, 0, 1}, {0, -4, 2}};
    for (size_t c = 0; c < positions.size(); ++c)
        in.projection[c] = look_at(positions[c]);
    Eigen::Vector3d const X(0.1, 0.2, 0.3);
    for (int f = 0; f < n; ++f) {
        for (int c = 0; c < 4; ++c) {
            auto p = project(in.projection[static_cast<size_t>(c)], X);
            in.candidates.push_back(cand(c, 0, f, p.x(), p.y(), 100 * c + 1, 1, 0.3f));
            // A stronger maximum 150 px away from the true projection on every camera.
            in.candidates.push_back(
                cand(c, 0, f, p.x() + 150, p.y() + 80 * (c + 1), 100 * c + 2, 0, 0.9f));
        }
    }
    auto s = plain();
    s.beta = 0.3;
    WindowChoice w(std::move(in), s);
    w.flag_consistent();
    for (auto const& c : w.candidates())
        CHECK(c.consistent == (c.rank == 1));
    // The first-round pick follows the agreement, not the peak.
    w.run();
    for (int c = 0; c < 4; ++c)
        CHECK(w.candidates()[static_cast<size_t>(w.pick(c, 0, 1))].rank == 1);
}
