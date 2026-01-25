#pragma once

#include <posetrak/core/skeleton.hpp>

#include <string>

namespace posetrak {

/// @brief Load skeleton from YAML file
/// @param filepath Path to YAML skeleton file
/// @return Parsed skeleton structure
/// @throws std::runtime_error if file not found or parsing fails
Skeleton load_skeleton_from_yaml(std::string const& filepath);

}  // namespace posetrak
