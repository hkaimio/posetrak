// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#include "posetrak/tracking/dot_assignment.hpp"

#include "posetrak/tracking/assignment.hpp"

namespace posetrak {

std::unordered_map<int, SubjectDotAssignment> resolve_dot_assignment(
    std::vector<SubjectDotPredictions> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    double gate_mahalanobis, int frame_idx, double timestamp) {
    std::unordered_map<int, SubjectDotAssignment> result;

    // One column per (subject, marker) slot with a prediction for this camera.
    struct Column {
        int subject_id;
        int marker_id;
        MarkerPrediction const* prediction;
    };

    for (auto const& [camera_id, candidates] : candidates_by_camera) {
        if (candidates.empty())
            continue;

        std::vector<Column> columns;
        for (auto const& subject : subjects) {
            auto cam_it = subject.predictions_by_camera.find(camera_id);
            if (cam_it == subject.predictions_by_camera.end())
                continue;
            for (auto const& [marker_id, prediction] : cam_it->second) {
                columns.push_back(Column{subject.subject_id, marker_id, &prediction});
            }
        }
        if (columns.empty())
            continue;

        int const n_rows = static_cast<int>(candidates.size());
        int const n_cols = static_cast<int>(columns.size());
        std::vector<double> cost(static_cast<size_t>(n_rows) * static_cast<size_t>(n_cols));
        for (int r = 0; r < n_rows; ++r) {
            Eigen::Vector2d const cand_pos = candidates[static_cast<size_t>(r)].position;
            for (int c = 0; c < n_cols; ++c) {
                MarkerPrediction const& pred = *columns[static_cast<size_t>(c)].prediction;
                Eigen::Vector2d const diff = cand_pos - pred.position;
                double const mahal_sq = diff.transpose() * pred.covariance.inverse() * diff;
                cost[static_cast<size_t>(r) * static_cast<size_t>(n_cols) +
                     static_cast<size_t>(c)] = mahal_sq;
            }
        }

        auto pairs = solve_assignment(cost, n_rows, n_cols, gate_mahalanobis);
        for (auto const& pair : pairs) {
            Column const& col = columns[static_cast<size_t>(pair.col)];
            UnlabeledCandidate const& cand = candidates[static_cast<size_t>(pair.row)];

            Observation obs;
            obs.camera_id = camera_id;
            obs.marker_id = col.marker_id;
            obs.frame_idx = frame_idx;
            obs.timestamp = timestamp;
            obs.position = cand.position;
            obs.position_distorted = cand.position_distorted;
            obs.confidence = cand.confidence;
            // Same reasoning as the ArUco corner and dot-detector write paths'
            // own noise_scale=0.0 convention: a dot candidate's centroid comes
            // from thresholding the full-resolution frame directly, not a
            // fixed-input-resolution network, so there is no crop-scaled
            // detection-algorithm error to describe -- calibration error (ec)
            // alone should dominate Observation::measurement_noise_std().
            obs.crop_scale = 0.0;

            result[col.subject_id].resolved.push_back(obs);
        }
    }

    return result;
}

std::unordered_map<int, SubjectDotAssignment> resolve_shared_dot_assignment(
    std::vector<DotAssignmentSubject> const& subjects,
    std::unordered_map<int, std::vector<UnlabeledCandidate>> const& candidates_by_camera,
    TrackerConfig const& config, int frame_idx, double timestamp) {
    std::vector<SubjectDotPredictions> predictions;
    predictions.reserve(subjects.size());
    for (auto const& subject : subjects) {
        SubjectDotPredictions sp;
        sp.subject_id = subject.subject_id;
        for (auto const& [camera_id, candidates] : candidates_by_camera) {
            if (candidates.empty())
                continue;
            sp.predictions_by_camera[camera_id] =
                subject.tracker->predict_dot_slot_predictions(camera_id);
        }
        predictions.push_back(std::move(sp));
    }

    return resolve_dot_assignment(predictions, candidates_by_camera,
                                  config.dot_assignment_gate_mahalanobis, frame_idx, timestamp);
}

}  // namespace posetrak
