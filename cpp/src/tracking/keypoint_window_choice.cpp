// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/keypoint_window_choice.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

namespace posetrak::kpchoice {

namespace {

// Consensus (flag_consistent): a keypoint at a time is "agreed" when the candidates of at least
// this many cameras triangulate to a point that projects within a per-camera radius of each.
constexpr int kConsensusMinCameras = 3;
constexpr double kConsensusToleranceS = 0.005;
constexpr double kConsensusMinRadiusPx = 12.0;
constexpr double kConsensusMaxRadiusPx = 60.0;
constexpr int kConsensusCalibrationStride = 9;
constexpr double kConsensusRadiusScale = 2.5;  // times the 30th percentile of the residuals
constexpr double kConsensusRadiusPercentile = 30.0;

Eigen::Matrix3d skew(Eigen::Vector3d const& t) {
    Eigen::Matrix3d m;
    m << 0, -t.z(), t.y(), t.z(), 0, -t.x(), -t.y(), t.x(), 0;
    return m;
}

/// Fundamental matrix with x_o^T F x_c = 0 for undistorted pixels of cameras c and o.
Eigen::Matrix3d fundamental(Eigen::Matrix<double, 3, 4> const& pc,
                            Eigen::Matrix<double, 3, 4> const& po) {
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(pc, Eigen::ComputeFullV);
    Eigen::Vector4d centre = svd.matrixV().col(3);
    Eigen::Vector3d epipole = po * centre;
    Eigen::Matrix<double, 4, 3> pinv = pc.transpose() * (pc * pc.transpose()).inverse();
    return skew(epipole) * po * pinv;
}

/// numpy.percentile with linear interpolation.
double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    double const pos = q / 100.0 * static_cast<double>(v.size() - 1);
    size_t const lo = static_cast<size_t>(std::floor(pos));
    size_t const hi = std::min(lo + 1, v.size() - 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - static_cast<double>(lo));
}

/// Windowed mean of a value along one tracklet, by prefix sums over time.
struct Series {
    std::vector<double> t;
    std::vector<double> cum{0.0};
    void add(double time, double value) {
        t.push_back(time);
        cum.push_back(cum.back() + value);
    }
    double mean(double centre, double half, double fallback) const {
        auto i0 = std::lower_bound(t.begin(), t.end(), centre - half) - t.begin();
        auto i1 = std::upper_bound(t.begin(), t.end(), centre + half) - t.begin();
        if (i1 <= i0)
            return fallback;
        return (cum[static_cast<size_t>(i1)] - cum[static_cast<size_t>(i0)]) /
               static_cast<double>(i1 - i0);
    }
};

struct SeriesSet {
    std::unordered_map<std::int64_t, Series> by_tracklet;
    void add(std::int64_t key, double time, double value) { by_tracklet[key].add(time, value); }
    double mean(std::int64_t key, double centre, double half, double fallback) const {
        auto it = by_tracklet.find(key);
        return it == by_tracklet.end() ? fallback : it->second.mean(centre, half, fallback);
    }
};

/// Unlinked candidates are tracklets of their own.
std::int64_t tracklet_key(Candidate const& c, int index) {
    return c.tracklet >= 0 ? static_cast<std::int64_t>(c.tracklet)
                           : -1 - static_cast<std::int64_t>(index);
}

}  // namespace

WindowChoice::WindowChoice(Input input, Settings const& settings)
    : input_(std::move(input)), settings_(settings) {
    candidates_ = std::move(input_.candidates);
    input_.candidates.clear();
    build_index();
}

void WindowChoice::build_index() {
    size_t const n_cameras = input_.frames.size();
    int const n_keypoints = input_.n_keypoints;
    std::sort(candidates_.begin(), candidates_.end(), [](Candidate const& a, Candidate const& b) {
        if (a.camera != b.camera)
            return a.camera < b.camera;
        if (a.keypoint != b.keypoint)
            return a.keypoint < b.keypoint;
        if (a.frame != b.frame)
            return a.frame < b.frame;
        return a.rank < b.rank;
    });
    cell_base_.assign(n_cameras + 1, 0);
    for (size_t c = 0; c < n_cameras; ++c)
        cell_base_[c + 1] = cell_base_[c] + n_keypoints * static_cast<int>(input_.frames[c].size());
    int const n_cells = cell_base_[n_cameras];
    cell_first_.assign(static_cast<size_t>(n_cells) + 1, 0);
    for (auto const& cand : candidates_)
        ++cell_first_[static_cast<size_t>(cell(cand.camera, cand.keypoint, cand.frame)) + 1];
    for (size_t i = 0; i < static_cast<size_t>(n_cells); ++i)
        cell_first_[i + 1] += cell_first_[i];
    picks_.assign(static_cast<size_t>(n_cells), -1);
    fixed_.assign(static_cast<size_t>(n_cells), -1);
    for (auto const& f : input_.fixed) {
        if (!input_.active[static_cast<size_t>(f.camera)])
            continue;
        int const id = cell(f.camera, f.keypoint, f.frame);
        for (int r = cell_first_[static_cast<size_t>(id)];
             r < cell_first_[static_cast<size_t>(id) + 1]; ++r) {
            if (candidates_[static_cast<size_t>(r)].tracklet == f.tracklet) {
                fixed_[static_cast<size_t>(id)] = r;
                break;
            }
        }
    }
}

int WindowChoice::pick(int camera, int keypoint, int frame) const {
    return picks_[static_cast<size_t>(cell(camera, keypoint, frame))];
}

// ---------------------------------------------------------------------------------------------
// Consensus
// ---------------------------------------------------------------------------------------------

void WindowChoice::flag_consistent() {
    int const n_cameras = static_cast<int>(input_.frames.size());
    int const n_keypoints = input_.n_keypoints;
    for (auto& c : candidates_)
        c.consistent = false;
    if (n_cameras < kConsensusMinCameras)
        return;

    // The reference camera is the fastest; buckets are its frames.
    int ref = 0;
    double best_fps = -1.0;
    for (int c = 0; c < n_cameras; ++c) {
        auto const& fr = input_.frames[static_cast<size_t>(c)];
        if (fr.size() < 2)
            continue;
        std::vector<double> dt;
        for (size_t i = 1; i < fr.size(); ++i)
            dt.push_back(fr[i].time - fr[i - 1].time);
        std::sort(dt.begin(), dt.end());
        size_t const m = dt.size();
        double const median = m % 2 ? dt[m / 2] : 0.5 * (dt[m / 2 - 1] + dt[m / 2]);
        double const fps = median > 0 ? 1.0 / median : 0.0;
        if (fps > best_fps) {
            best_fps = fps;
            ref = c;
        }
    }

    auto nearest_frame = [&](int c, double t) -> int {
        auto const& fr = input_.frames[static_cast<size_t>(c)];
        if (fr.empty())
            return -1;
        auto it = std::lower_bound(fr.begin(), fr.end(), t,
                                   [](Frame const& f, double v) { return f.time < v; });
        int j = static_cast<int>(it - fr.begin());
        if (j >= static_cast<int>(fr.size()))
            j = static_cast<int>(fr.size()) - 1;
        else if (j > 0 && std::abs(fr[static_cast<size_t>(j) - 1].time - t) <=
                              std::abs(fr[static_cast<size_t>(j)].time - t))
            --j;
        return std::abs(fr[static_cast<size_t>(j)].time - t) <= kConsensusToleranceS ? j : -1;
    };

    auto const& P = input_.projection;

    // One pass over the buckets given. When *residuals* is set it collects the reprojection
    // residual of every camera in the best set of each keypoint.
    auto consensus = [&](std::vector<int> const& buckets, std::vector<double> const& radius,
                         std::vector<std::vector<double>>* residuals) {
        std::vector<int> frame_of(static_cast<size_t>(n_cameras));
        for (int rf : buckets) {
            double const t = input_.frames[static_cast<size_t>(ref)][static_cast<size_t>(rf)].time;
            int n_found = 0;
            for (int c = 0; c < n_cameras; ++c) {
                frame_of[static_cast<size_t>(c)] = nearest_frame(c, t);
                n_found += frame_of[static_cast<size_t>(c)] >= 0;
            }
            if (n_found < kConsensusMinCameras)
                continue;
            for (int k = 0; k < n_keypoints; ++k) {
                // Candidates per camera of this keypoint at this time.
                std::vector<std::pair<int, std::vector<int>>> cand;
                for (int c = 0; c < n_cameras; ++c) {
                    int const f = frame_of[static_cast<size_t>(c)];
                    if (f < 0)
                        continue;
                    int const id = cell(c, k, f);
                    std::vector<int> rows;
                    for (int r = cell_first_[static_cast<size_t>(id)];
                         r < cell_first_[static_cast<size_t>(id) + 1]; ++r)
                        if (!candidates_[static_cast<size_t>(r)].removed)
                            rows.push_back(r);
                    if (!rows.empty())
                        cand.emplace_back(c, std::move(rows));
                }
                if (static_cast<int>(cand.size()) < kConsensusMinCameras)
                    continue;

                int best_count = -1;
                double best_err = 1e9;
                std::vector<std::pair<int, int>> best_members;  // (camera, candidate)
                std::vector<double> best_res;                   // per member
                for (size_t ia = 0; ia < cand.size(); ++ia) {
                    for (size_t ib = ia + 1; ib < cand.size(); ++ib) {
                        int const a = cand[ia].first, b = cand[ib].first;
                        double pair_val = -std::numeric_limits<double>::infinity();
                        int pair_count = 0;
                        double pair_err = 0;
                        std::vector<std::pair<int, int>> pair_members;
                        std::vector<double> pair_res;
                        for (int ra : cand[ia].second) {
                            for (int rb : cand[ib].second) {
                                auto const& ca = candidates_[static_cast<size_t>(ra)];
                                auto const& cb = candidates_[static_cast<size_t>(rb)];
                                Eigen::Matrix4d A;
                                A.row(0) =
                                    static_cast<double>(ca.ux) * P[static_cast<size_t>(a)].row(2) -
                                    P[static_cast<size_t>(a)].row(0);
                                A.row(1) =
                                    static_cast<double>(ca.uy) * P[static_cast<size_t>(a)].row(2) -
                                    P[static_cast<size_t>(a)].row(1);
                                A.row(2) =
                                    static_cast<double>(cb.ux) * P[static_cast<size_t>(b)].row(2) -
                                    P[static_cast<size_t>(b)].row(0);
                                A.row(3) =
                                    static_cast<double>(cb.uy) * P[static_cast<size_t>(b)].row(2) -
                                    P[static_cast<size_t>(b)].row(1);
                                Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
                                Eigen::Vector4d X = svd.matrixV().col(3);
                                X /= X(3);

                                bool sel = true;
                                int count = 2;
                                double err = 0;
                                std::vector<std::pair<int, int>> members;
                                std::vector<double> res;
                                for (auto const& [ci, rows] : cand) {
                                    Eigen::Vector3d pr = P[static_cast<size_t>(ci)] * X;
                                    bool const ok_depth = pr.z() > 0;
                                    double const z = pr.z() == 0 ? 1.0 : pr.z();
                                    double const px = pr.x() / z, py = pr.y() / z;
                                    double const rad = radius[static_cast<size_t>(ci)];
                                    if (ci == a || ci == b) {
                                        auto const& own = ci == a ? ca : cb;
                                        double const e = std::hypot(px - own.ux, py - own.uy);
                                        sel = sel && ok_depth && e < rad * 2;
                                        err += e;
                                        members.emplace_back(ci, ci == a ? ra : rb);
                                        res.push_back(e);
                                        continue;
                                    }
                                    double dm = std::numeric_limits<double>::infinity();
                                    int jmin = -1;
                                    for (int rc : rows) {
                                        auto const& cc = candidates_[static_cast<size_t>(rc)];
                                        double const d = std::hypot(px - cc.ux, py - cc.uy);
                                        if (d < dm) {
                                            dm = d;
                                            jmin = rc;
                                        }
                                    }
                                    bool const hit = dm < rad && ok_depth;
                                    if (hit) {
                                        ++count;
                                        err += dm;
                                        members.emplace_back(ci, jmin);
                                        res.push_back(dm);
                                    }
                                }
                                if (!sel)
                                    count = -1;
                                double const val = count - 1e-4 * err;
                                if (val > pair_val) {
                                    pair_val = val;
                                    pair_count = count;
                                    pair_err = err;
                                    pair_members = std::move(members);
                                    pair_res = std::move(res);
                                }
                            }
                        }
                        if (pair_count > best_count ||
                            (pair_count == best_count && pair_err < best_err)) {
                            best_count = pair_count;
                            best_err = pair_err;
                            best_members = std::move(pair_members);
                            best_res = std::move(pair_res);
                        }
                    }
                }
                if (best_count >= kConsensusMinCameras) {
                    for (size_t i = 0; i < best_members.size(); ++i) {
                        candidates_[static_cast<size_t>(best_members[i].second)].consistent = true;
                        if (residuals != nullptr)
                            (*residuals)[static_cast<size_t>(best_members[i].first)].push_back(
                                best_res[i]);
                    }
                }
            }
        }
    };

    // Pass 1: a generous radius on every ninth bucket gives each camera's own error scale
    // (calibration, synchronisation and detection error differ between cameras). Pass 2 flags
    // with a radius of 2.5 times the 30th percentile of the residuals.
    int const n_ref = static_cast<int>(input_.frames[static_cast<size_t>(ref)].size());
    std::vector<int> calibration_buckets, all_buckets(static_cast<size_t>(n_ref));
    std::iota(all_buckets.begin(), all_buckets.end(), 0);
    for (int i = 0; i < n_ref; i += kConsensusCalibrationStride)
        calibration_buckets.push_back(i);
    std::vector<std::vector<double>> residuals(static_cast<size_t>(n_cameras));
    consensus(calibration_buckets,
              std::vector<double>(static_cast<size_t>(n_cameras), kConsensusMaxRadiusPx),
              &residuals);
    radius_.assign(static_cast<size_t>(n_cameras), kConsensusMinRadiusPx);
    for (size_t c = 0; c < radius_.size(); ++c)
        if (!residuals[c].empty())
            radius_[c] = std::clamp(kConsensusRadiusScale *
                                        percentile(residuals[c], kConsensusRadiusPercentile),
                                    kConsensusMinRadiusPx, kConsensusMaxRadiusPx);
    for (auto& c : candidates_)
        c.consistent = false;
    consensus(all_buckets, radius_, nullptr);
}

// ---------------------------------------------------------------------------------------------
// Strict mask filter
// ---------------------------------------------------------------------------------------------

FilterStats WindowChoice::apply_strict_filter(int half_window_frames, double min_fraction) {
    FilterStats stats;
    size_t const n = candidates_.size();
    // Candidates of a tracklet, in video frame order.
    std::map<std::tuple<int, int, std::int64_t>, std::vector<int>> groups;
    for (size_t i = 0; i < n; ++i) {
        auto const& c = candidates_[i];
        if (c.removed)
            continue;
        groups[{c.camera, c.keypoint, tracklet_key(c, static_cast<int>(i))}].push_back(
            static_cast<int>(i));
    }
    // A keypoint-frame counts as emptied when it had a candidate and has none after the filter.
    std::vector<char> had(picks_.size(), 0);
    for (size_t i = 0; i < n; ++i)
        if (!candidates_[i].removed)
            had[static_cast<size_t>(
                cell(candidates_[i].camera, candidates_[i].keypoint, candidates_[i].frame))] = 1;
    for (auto& [key, rows] : groups) {
        auto const& frames = input_.frames[static_cast<size_t>(std::get<0>(key))];
        auto video = [&](int r) {
            return frames[static_cast<size_t>(candidates_[static_cast<size_t>(r)].frame)]
                .video_frame;
        };
        std::sort(rows.begin(), rows.end(), [&](int a, int b) { return video(a) < video(b); });
        size_t const m = rows.size();
        // Prefix counts of detections with a known mask and of those on the person's mask.
        std::vector<int> known(m + 1, 0), inside(m + 1, 0);
        for (size_t j = 0; j < m; ++j) {
            auto const label = candidates_[static_cast<size_t>(rows[j])].mask_label;
            known[j + 1] = known[j] + (label != kMaskUnknown);
            inside[j + 1] = inside[j] + (label == input_.own_mask_label);
        }
        size_t lo = 0, hi = 0;
        for (size_t j = 0; j < m; ++j) {
            int const v = video(rows[j]);
            while (video(rows[lo]) < v - half_window_frames)
                ++lo;
            while (hi < m && video(rows[hi]) <= v + half_window_frames)
                ++hi;
            int const counted = known[hi] - known[lo];
            if (counted == 0)
                continue;
            double const fraction = static_cast<double>(inside[hi] - inside[lo]) / counted;
            if (fraction < min_fraction) {
                candidates_[static_cast<size_t>(rows[j])].removed = true;
                ++stats.candidates_removed;
            }
        }
    }
    std::vector<char> has(picks_.size(), 0);
    for (size_t i = 0; i < n; ++i)
        if (!candidates_[i].removed)
            has[static_cast<size_t>(
                cell(candidates_[i].camera, candidates_[i].keypoint, candidates_[i].frame))] = 1;
    for (size_t i = 0; i < had.size(); ++i) {
        stats.keypoint_frames += had[i];
        stats.emptied += had[i] && !has[i];
    }
    return stats;
}

// ---------------------------------------------------------------------------------------------
// Choice
// ---------------------------------------------------------------------------------------------

std::vector<int> WindowChoice::start_picks() const {
    std::vector<int> picks(picks_.size(), -1);
    int const n_cameras = static_cast<int>(input_.frames.size());
    for (int c = 0; c < n_cameras; ++c) {
        if (!input_.active[static_cast<size_t>(c)])
            continue;
        int const n_frames = static_cast<int>(input_.frames[static_cast<size_t>(c)].size());
        for (int k = 0; k < input_.n_keypoints; ++k) {
            std::int64_t previous = -1;
            for (int f = 0; f < n_frames; ++f) {
                int const id = cell(c, k, f);
                int agreeing = -1, continuing = -1, first = -1;
                for (int r = cell_first_[static_cast<size_t>(id)];
                     r < cell_first_[static_cast<size_t>(id) + 1]; ++r) {
                    auto const& cand = candidates_[static_cast<size_t>(r)];
                    if (cand.removed)
                        continue;
                    if (first < 0)
                        first = r;
                    if (cand.consistent &&
                        (agreeing < 0 ||
                         cand.peak > candidates_[static_cast<size_t>(agreeing)].peak))
                        agreeing = r;
                    if (continuing < 0 && previous >= 0 && cand.tracklet == previous)
                        continuing = r;
                }
                if (first < 0) {
                    previous = -1;
                    continue;
                }
                int const chosen =
                    agreeing >= 0 ? agreeing : (continuing >= 0 ? continuing : first);
                picks[static_cast<size_t>(id)] = chosen;
                previous = candidates_[static_cast<size_t>(chosen)].tracklet;
            }
        }
    }
    return picks;
}

void WindowChoice::run() {
    int const n_cameras = static_cast<int>(input_.frames.size());
    int const n_keypoints = input_.n_keypoints;
    auto const& s = settings_;
    auto is_active = [&](int c) { return input_.active[static_cast<size_t>(c)] != 0; };

    picks_ = start_picks();
    for (size_t i = 0; i < picks_.size(); ++i)
        if (fixed_[i] >= 0 && picks_[i] != fixed_[i])
            picks_[i] = fixed_[i];
    // A fixed pick sits in a cell of an active camera only; for others it has no effect.

    // Epipolar geometry and the nearest frame of every other camera.
    std::vector<std::vector<Eigen::Matrix3d>> F(
        static_cast<size_t>(n_cameras),
        std::vector<Eigen::Matrix3d>(static_cast<size_t>(n_cameras)));
    std::vector<std::vector<std::vector<int>>> near(
        static_cast<size_t>(n_cameras),
        std::vector<std::vector<int>>(static_cast<size_t>(n_cameras)));
    for (int c = 0; c < n_cameras; ++c) {
        if (!is_active(c))
            continue;
        for (int o = 0; o < n_cameras; ++o) {
            if (o == c || !is_active(o))
                continue;
            F[static_cast<size_t>(c)][static_cast<size_t>(o)] =
                fundamental(input_.projection[static_cast<size_t>(c)],
                            input_.projection[static_cast<size_t>(o)]);
            auto const& fc = input_.frames[static_cast<size_t>(c)];
            auto const& fo = input_.frames[static_cast<size_t>(o)];
            auto& nf = near[static_cast<size_t>(c)][static_cast<size_t>(o)];
            nf.assign(fc.size(), -1);
            for (size_t f = 0; f < fc.size(); ++f) {
                if (fo.empty())
                    break;
                auto it = std::lower_bound(fo.begin(), fo.end(), fc[f].time,
                                           [](Frame const& a, double v) { return a.time < v; });
                int j = static_cast<int>(it - fo.begin());
                if (j >= static_cast<int>(fo.size()))
                    j = static_cast<int>(fo.size()) - 1;
                else if (j > 0 && std::abs(fo[static_cast<size_t>(j) - 1].time - fc[f].time) <=
                                      std::abs(fo[static_cast<size_t>(j)].time - fc[f].time))
                    --j;
                if (std::abs(fo[static_cast<size_t>(j)].time - fc[f].time) <= s.sync_tolerance_s)
                    nf[f] = j;
            }
        }
    }

    auto support = [&](double d, double limit) {
        return d <= limit ? 1.0 : std::exp(-(d - limit) / s.decay);
    };
    double const nan = std::numeric_limits<double>::quiet_NaN();
    bool const use_mask = s.mask_weight != 0.0;
    auto mask_value = [&](Candidate const& c) {
        return static_cast<double>(c.mask_label == input_.own_mask_label && c.mask_label > 0) -
               static_cast<double>(c.mask_label > 0 && c.mask_label != input_.own_mask_label);
    };

    for (int round = 0; round < s.rounds; ++round) {
        std::vector<int> next(picks_.size(), -1);
        for (int c = 0; c < n_cameras; ++c) {
            if (!is_active(c))
                continue;
            auto const& frames = input_.frames[static_cast<size_t>(c)];
            int const n_frames = static_cast<int>(frames.size());
            if (n_frames == 0)
                continue;
            // Per-candidate support of this camera (NaN = none), from the previous round's picks.
            std::vector<double> sup, xs;
            int const first_cand = cell_first_[static_cast<size_t>(cell(c, 0, 0))];
            int const last_cand =
                cell_first_[static_cast<size_t>(cell(c, n_keypoints - 1, n_frames - 1)) + 1];
            sup.assign(static_cast<size_t>(last_cand - first_cand), nan);
            xs.assign(static_cast<size_t>(last_cand - first_cand), nan);
            for (int k = 0; k < n_keypoints; ++k) {
                for (int f = 0; f < n_frames; ++f) {
                    int const id = cell(c, k, f);
                    double const hh = frames[static_cast<size_t>(f)].box_height;
                    for (int r = cell_first_[static_cast<size_t>(id)];
                         r < cell_first_[static_cast<size_t>(id) + 1]; ++r) {
                        auto const& cand = candidates_[static_cast<size_t>(r)];
                        if (cand.removed)
                            continue;
                        double sum = 0;
                        int n = 0;
                        for (auto const& [nb, limit] : input_.neighbours[static_cast<size_t>(k)]) {
                            int const rn = picks_[static_cast<size_t>(cell(c, nb, f))];
                            if (rn < 0)
                                continue;
                            auto const& other = candidates_[static_cast<size_t>(rn)];
                            double const d = std::hypot(static_cast<double>(cand.x) - other.x,
                                                        static_cast<double>(cand.y) - other.y) /
                                             hh;
                            sum += support(d, limit);
                            ++n;
                        }
                        if (n)
                            sup[static_cast<size_t>(r - first_cand)] = sum / n;
                        if (s.delta != 0.0) {
                            double xsum = 0;
                            int xn = 0;
                            Eigen::Vector3d const x(cand.ux, cand.uy, 1.0);
                            for (int o = 0; o < n_cameras; ++o) {
                                if (o == c || !is_active(o))
                                    continue;
                                int const fo = near[static_cast<size_t>(c)][static_cast<size_t>(o)]
                                                   [static_cast<size_t>(f)];
                                if (fo < 0)
                                    continue;
                                int const ro = picks_[static_cast<size_t>(cell(o, k, fo))];
                                if (ro < 0)
                                    continue;
                                auto const& other = candidates_[static_cast<size_t>(ro)];
                                Eigen::Vector3d const l =
                                    F[static_cast<size_t>(c)][static_cast<size_t>(o)] * x;
                                double const d =
                                    std::abs(Eigen::Vector3d(other.ux, other.uy, 1.0).dot(l)) /
                                    std::hypot(l.x(), l.y()) / hh;
                                xsum += support(d, s.epipolar_threshold);
                                ++xn;
                            }
                            if (xn)
                                xs[static_cast<size_t>(r - first_cand)] = xsum / xn;
                        }
                    }
                }
            }
            // Windowed means along the tracklets and the choice, per keypoint.
            for (int k = 0; k < n_keypoints; ++k) {
                SeriesSet a_series, x_series, m_series;
                bool const mask_here = use_mask && !input_.mask_keypoint.empty() &&
                                       input_.mask_keypoint[static_cast<size_t>(k)];
                for (int f = 0; f < n_frames; ++f) {
                    int const id = cell(c, k, f);
                    double const t = frames[static_cast<size_t>(f)].time;
                    for (int r = cell_first_[static_cast<size_t>(id)];
                         r < cell_first_[static_cast<size_t>(id) + 1]; ++r) {
                        auto const& cand = candidates_[static_cast<size_t>(r)];
                        if (cand.removed)
                            continue;
                        auto const key = tracklet_key(cand, r);
                        double const a = sup[static_cast<size_t>(r - first_cand)];
                        double const x = xs[static_cast<size_t>(r - first_cand)];
                        if (!std::isnan(a))
                            a_series.add(key, t, a);
                        if (!std::isnan(x))
                            x_series.add(key, t, x);
                        if (mask_here)
                            m_series.add(key, t, mask_value(cand));
                    }
                }
                for (int f = 0; f < n_frames; ++f) {
                    int const id = cell(c, k, f);
                    double const t = frames[static_cast<size_t>(f)].time;
                    double best = -9.0;
                    int chosen = -1;
                    for (int r = cell_first_[static_cast<size_t>(id)];
                         r < cell_first_[static_cast<size_t>(id) + 1]; ++r) {
                        auto const& cand = candidates_[static_cast<size_t>(r)];
                        if (cand.removed)
                            continue;
                        auto const key = tracklet_key(cand, r);
                        double score = a_series.mean(key, t, s.support_window_s, 0.5);
                        if (s.delta != 0.0)
                            score += s.delta * x_series.mean(key, t, s.support_window_s, 0.5);
                        score += s.beta * (cand.consistent ? 1.0 : 0.0) + s.gamma * cand.peak;
                        if (mask_here)
                            score += s.mask_weight * m_series.mean(key, t, s.mask_window_s, 0.0);
                        if (score > best) {
                            best = score;
                            chosen = r;
                        }
                    }
                    next[static_cast<size_t>(id)] = fixed_[static_cast<size_t>(id)] >= 0
                                                        ? fixed_[static_cast<size_t>(id)]
                                                        : chosen;
                }
            }
        }
        picks_ = std::move(next);
    }
}

}  // namespace posetrak::kpchoice
