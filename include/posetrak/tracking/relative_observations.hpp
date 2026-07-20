/**
 * @file relative_observations.hpp
 * @brief Build PAIR_DIFF observations against a fixed reference marker.
 *
 * For the hierarchical body/hand solver's child filter (see
 * docs/roadmap/features/hierarchical-solver/hierarchical-solver-design.md,
 * "Measurement model"): once the reference marker (MRK-wrist) is inside the
 * child's own skeleton, every other hand marker's observation is exactly the
 * existing, older within-person relative-observation mechanism --
 * MeasurementMode::PAIR_DIFF with ref_marker_id pointing at the reference
 * marker's index in the SAME skeleton, reprojected fresh per sigma point
 * like any other in-state reference marker. No anchor_position, no external
 * constant -- the existing PAIR_DIFF branch in ukf.cpp handles this
 * unmodified. This mirrors the within-person relative-pair construction in
 * session_reader.cpp (search "Generate RELATIVE observations"), just against
 * a single fixed reference marker instead of a per-marker skeleton-hierarchy
 * parent, and independent of the DB-loading path so it can be exercised
 * directly against any per-frame observation source.
 */
#pragma once

#include "posetrak/core/observation.hpp"
#include <vector>

namespace posetrak {

/// @brief Build one PAIR_DIFF observation per (marker, camera) where both a
/// non-reference marker and the reference marker were detected in that
/// camera this frame.
///
/// @param frame_obs      Raw per-camera detections for this frame, for
///                       markers in the child's own observation group
///                       (including the reference marker itself -- its own
///                       entries are read as the reference, never emitted
///                       as a result observation).
/// @param ref_marker_id  Index of the reference marker (e.g. MRK-wrist) in
///                       the child's own skeleton -- becomes each result
///                       observation's ref_marker_id.
/// @param pose_noise_std Pose estimation error (pixels, model input image).
///                       Noise = pose_noise_std * sqrt(2) * crop_scale --
///                       calibration error cancels in the pixel difference,
///                       matching every other PAIR_DIFF use in this codebase
///                       (session_reader.cpp's within-person relative pairs).
/// @param min_confidence Both the marker and the reference marker must meet
///                       this confidence for a camera to contribute a pair.
/// @return PAIR_DIFF observations, one per qualifying (marker, camera).
std::vector<Observation>
build_ref_marker_pair_observations(std::vector<Observation> const& frame_obs, int ref_marker_id,
                                   double pose_noise_std, double min_confidence = 0.0);

}  // namespace posetrak
