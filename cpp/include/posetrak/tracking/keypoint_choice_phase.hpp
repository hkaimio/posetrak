// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

/**
 * @file keypoint_choice_phase.hpp
 * @brief The tracking-run phase that chooses keypoint candidates over a time window.
 *
 * Runs between loading and the first filter step (see
 * docs/roadmap/features/keypoint-candidates/keypoint-candidates-design.md, section 10):
 *
 *  1. load the candidates of the tracked window plus the support margin on both sides;
 *  2. apply the tracklet decisions (reject / other / forbid remove candidates, assign fixes a
 * pick);
 *  3. read the mask labels at the candidates when a mask option needs them;
 *  4. flag the multi-camera consensus, apply the strict mask filter, compute the picks
 *     (kpchoice::WindowChoice);
 *  5. replace the fixed POSITION observations of the markers bound to the keypoints by the picks,
 *     and store the picks with the run.
 *
 * Nothing in the step loop changes: the filter sees the picks as ordinary observations.
 */
#pragma once

#include "posetrak/tracking/multi_person_tracker.hpp"
#include "posetrak/tracking/tracklet_assignment.hpp"
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief What a pose model contributes to the choice: which keypoints are skeleton neighbours
/// (with the largest plausible distance between them, in box heights) and which class each
/// keypoint is of, for the mask term. Keyed by the qualified landmark ("soma77:Hips").
struct KeypointChoiceModel {
    struct Pair {
        std::string a;
        std::string b;
        double limit = 0.0;
    };
    std::vector<Pair> pairs;
    std::unordered_map<std::string, std::string> keypoint_class;
};

/// @brief Read a choice model CSV: lines `pair,<landmark>,<landmark>,<limit>` and
/// `class,<landmark>,<class>`; blank lines and lines starting with `#` are skipped.
/// @throws std::runtime_error if the file cannot be opened or a line is malformed.
KeypointChoiceModel load_keypoint_choice_model(std::string const& path);

struct KeypointChoiceInputs {
    std::string db_path;
    std::string sequence_id;  ///< Full sequence id.
    int person_id = 0;
    /// The pose model's confidence scale: the stored peak is the heatmap peak times this (5.0 for
    /// SOMA-77). The peak weight gamma applies to the heatmap peak.
    double confidence_scale = 5.0;
    KeypointChoiceModel model;
    /// Masks, for the strict filter and the mask term. seg_run_id empty: no masks.
    std::string seg_run_id;
    int seg_mask_label = 0;
    int seg_revision = -1;
    std::unordered_map<int, std::string> camera_shot_video_ids;  ///< camera id -> shot video id
    TrackletAssignments const* rules = nullptr;
    /// Write every candidate with its consensus flag, whether it was removed and whether it was
    /// picked to this CSV file (diagnostics); empty: none.
    std::string dump_path;
    bool quiet = false;
};

struct KeypointChoiceReport {
    long candidates = 0;
    long removed_by_decisions = 0;
    long keypoint_frames = 0;  ///< With a candidate, before the strict filter.
    long emptied_by_filter = 0;
    long picked = 0;
    long observations =
        0;  ///< Picks of bound markers above the confidence floor: the observations they give.
    double load_s = 0.0;
    double masks_s = 0.0;
    double consensus_s = 0.0;
    double choice_s = 0.0;
    double total_s = 0.0;
    double peak_memory_before_mb = 0.0;  ///< Peak working set of the process when the phase starts.
    double peak_memory_mb = 0.0;  ///< Peak working set of the process at the end of the phase.
};

/// @brief Run the phase for one person: rewrite ctx.observations and store the picks.
/// @throws std::runtime_error if the sequence holds no candidates, or a mask option is on and no
///         masks were given.
KeypointChoiceReport run_keypoint_choice_phase(PersonContext& ctx, KeypointChoiceInputs const& in);

/// @brief Peak working set of the process in megabytes (0 if unavailable).
double peak_memory_mb();

}  // namespace posetrak
