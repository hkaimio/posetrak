// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file keypoint_window_choice.hpp
 * @brief Choice of one heatmap maximum per person, camera, keypoint and frame over a time window.
 *
 * A pose model's heatmap has several local maxima per keypoint. The choice scores every
 * candidate of a keypoint-frame by the evidence of the other picks around it and takes the best:
 *
 *     score = A + delta * X + beta * consistent + gamma * peak + m * M
 *
 *  - A  skeleton-neighbour support: for each neighbouring keypoint with a pick in that frame, 1 if
 *       the candidate lies within the pair's distance limit (in box heights) of that pick,
 *       decaying exponentially beyond it; the mean over the neighbours.
 *  - X  cross-camera support: for each other camera with a pick of the same keypoint at the nearest
 *       frame, 1 if that pick lies within a threshold of the candidate's epipolar line (in box
 *       heights), decaying beyond it; the mean over the cameras.
 *  - M  mask term: +1 on the person's own mask, -1 on another person's, 0 on the background.
 *  - A, X and M are averaged over the candidate's tracklet within a window around the frame, so a
 *    tracklet that jumps to another keypoint loses support around the jump only.
 *  - `consistent` is set when the candidate belongs to the best multi-camera triangulation of its
 *    keypoint at that time (flag_consistent()).
 *
 * The scores are computed from the picks of the previous round for all cameras at once; the first
 * round starts from the strongest candidate that agrees with the other cameras, else the candidate
 * that continues the previous frame's tracklet, else the first. A keypoint-frame without a
 * candidate (none recorded, rejected, or removed by the strict mask filter) has no pick.
 *
 * The class has no database or tracker dependency: the caller loads the candidates, fills in
 * undistorted positions and mask labels, and turns the picks into observations.
 */
#pragma once

#include <Eigen/Core>

#include <cstdint>
#include <utility>
#include <vector>

namespace posetrak::kpchoice {

/// One camera frame of the person: a row of the sequence's candidates.
struct Frame {
    int video_frame = 0;
    double time = 0.0;        ///< Seconds on the capture timeline.
    double box_height = 1.0;  ///< Pixels; distances of the support terms are divided by it.
};

/// One heatmap maximum.
struct Candidate {
    float x = 0.f, y = 0.f;      ///< Distorted pixels, the frame the neighbour distances are in.
    float ux = 0.f, uy = 0.f;    ///< Undistorted pixels, the frame of the epipolar geometry.
    float peak = 0.f;            ///< Heatmap value, 0..1.
    std::int32_t tracklet = -1;  ///< Unique per camera across keypoints; negative: not linked.
    std::int32_t frame = 0;      ///< Index into Input::frames[camera].
    std::int32_t source = -1;    ///< The caller's index of the candidate; unused by the choice.
    std::int16_t rank = 0;       ///< 0 = strongest maximum of its keypoint in the frame.
    /// Mask label at the candidate's position: 0 background, > 0 a person, -1 outside the image,
    /// kMaskUnknown when the frame has no mask.
    std::int16_t mask_label = 0;
    std::uint8_t camera = 0;
    std::uint8_t keypoint = 0;
    bool consistent = false;  ///< Set by flag_consistent().
    bool removed = false;  ///< Rejected by a decision or the strict filter: takes part in nothing.
};

constexpr std::int16_t kMaskUnknown = -2;

/// A pick decided outside the choice (an `assign` decision): the candidate of this tracklet in the
/// keypoint-frame is the pick whatever its score.
struct FixedPick {
    int camera = 0;
    int keypoint = 0;
    int frame = 0;  ///< Index into Input::frames[camera].
    int tracklet = -1;
};

struct Input {
    int n_keypoints = 0;
    /// Frames of the person per camera, ascending in time.
    std::vector<std::vector<Frame>> frames;
    /// Per camera: takes part in the choice (picks and support). A camera that is not active keeps
    /// the strongest maxima as observations; its candidates still vote in flag_consistent().
    std::vector<char> active;
    /// Per camera: projection matrix of undistorted pixels.
    std::vector<Eigen::Matrix<double, 3, 4>> projection;
    /// Per keypoint: (neighbouring keypoint, distance limit in box heights). A pair without an
    /// entry gives no support.
    std::vector<std::vector<std::pair<int, double>>> neighbours;
    /// Per keypoint: the mask term applies. Empty: none.
    std::vector<char> mask_keypoint;
    /// Per keypoint: the strict mask filter applies. Empty: all keypoints.
    std::vector<char> filter_keypoint;
    std::vector<Candidate> candidates;  ///< Any order; the class sorts them.
    std::vector<FixedPick> fixed;
    int own_mask_label = 0;  ///< The person's label in the masks.
};

struct Settings {
    double support_window_s = 1.0;  ///< Half width of the window A and X are averaged over.
    double mask_window_s = 2.0;     ///< Half width of the window M is averaged over.
    double delta = 1.0;             ///< Weight of the cross-camera support.
    double beta = 0.3;              ///< Weight of the consensus flag.
    double gamma = 0.2;             ///< Weight of the peak.
    double mask_weight = 0.0;       ///< m: weight of the mask term; 0 = off.
    int rounds = 3;
    double epipolar_threshold = 0.04;  ///< Box heights within which a pick fully supports.
    double decay = 0.05;               ///< Box heights over which support decays beyond a limit.
    double sync_tolerance_s =
        0.008;  ///< Largest time difference of "the nearest frame" of another camera.
};

/// What the strict mask filter did.
struct FilterStats {
    long keypoint_frames = 0;  ///< Keypoint-frames with at least one candidate before the filter.
    long emptied = 0;          ///< Of those, the ones left without a candidate.
    long candidates_removed = 0;
};

class WindowChoice {
   public:
    WindowChoice(Input input, Settings const& settings);

    /// Sets Candidate::consistent from the multi-camera consensus (see keypoint_consensus in the
    /// .cpp). Uses the candidates not removed so far, of all cameras, active or not.
    void flag_consistent();

    /// Removes a candidate unless at least *min_fraction* of the detections of its tracklet within
    /// +-*half_window_frames* video frames lie on the person's own mask. Only keypoints flagged in
    /// Input::filter_keypoint are filtered. Candidates without a mask
    /// label are not counted; a candidate with no counted detection stays.
    FilterStats apply_strict_filter(int half_window_frames, double min_fraction);

    /// Counts the keypoint-frames with a candidate, per camera, without filtering (the figures
    /// apply_strict_filter() would report with nothing emptied). Call before run() when the filter
    /// is off.
    void count_keypoint_frames();

    /// Per camera: the figures of the last apply_strict_filter() or count_keypoint_frames().
    std::vector<FilterStats> const& camera_stats() const { return camera_stats_; }

    /// Computes the picks.
    void run();

    /// Index into candidates() of the pick of a keypoint-frame, -1 for none.
    int pick(int camera, int keypoint, int frame) const;
    std::vector<Candidate> const& candidates() const { return candidates_; }
    Input const& input() const { return input_; }

    /// Per-camera inlier radius (pixels) found by flag_consistent(), for diagnostics.
    std::vector<double> const& consensus_radius() const { return radius_; }

   private:
    int cell(int camera, int keypoint, int frame) const {
        return cell_base_[static_cast<size_t>(camera)] +
               keypoint * static_cast<int>(input_.frames[static_cast<size_t>(camera)].size()) +
               frame;
    }
    void build_index();
    std::vector<int> start_picks() const;

    Input input_;
    Settings settings_;
    std::vector<Candidate> candidates_;  // sorted by (camera, keypoint, frame, rank)
    std::vector<int> cell_base_;         // per camera: first cell id
    std::vector<int> cell_first_;        // per cell: first candidate; one more entry at the end
    std::vector<int> picks_;             // per cell
    std::vector<int> fixed_;             // per cell: fixed candidate or -1
    std::vector<double> radius_;
    std::vector<FilterStats> camera_stats_;
};

}  // namespace posetrak::kpchoice
