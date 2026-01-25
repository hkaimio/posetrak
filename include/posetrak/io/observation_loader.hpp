#pragma once

#include <posetrak/core/camera.hpp>
#include <posetrak/core/observation.hpp>

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace posetrak {

/// @brief Load observations from a single OpenPose JSON file
/// @param filepath Path to OpenPose JSON file (contains "people" array)
/// @param camera Camera to undistort observations with
/// @param camera_name Name to use for this camera in the observation sequence
/// @param frame_idx Frame index for these observations
/// @param min_confidence Minimum confidence threshold for keypoints (default: 0.1)
/// @param person_id Which person to extract (0-based index, -1 for first person)
/// @return ObservationSequence with all keypoint observations
/// @throws std::runtime_error if file cannot be read or parsed
ObservationSequence load_openpose_frame(std::string const& filepath, Camera const& camera,
                                        std::string const& camera_name, uint32_t frame_idx,
                                        double min_confidence = 0.1, int person_id = 0);

/// @brief Load observations from a directory structure for a single person
/// @details Expected structure: base_dir/camera_name/camera_name_NNNNNN.json
/// @param base_dir Base directory containing camera subdirectories
/// @param cameras Map of camera name to Camera object
/// @param frame_range Pair of (start_frame, end_frame) inclusive. Use {0, UINT32_MAX} for all
/// frames
/// @param min_confidence Minimum confidence threshold for keypoints
/// @param person_id Which person to extract (0-based index, default 0 for first person)
/// @return ObservationSet containing sequences from all cameras for the specified person
/// @throws std::runtime_error if directories don't exist or files malformed
ObservationSet load_openpose_sequence(std::string const& base_dir,
                                      std::unordered_map<std::string, Camera> const& cameras,
                                      std::pair<uint32_t, uint32_t> frame_range = {0, UINT32_MAX},
                                      double min_confidence = 0.1, int person_id = 0);

}  // namespace posetrak
