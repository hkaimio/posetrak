#pragma once
#include <string>

namespace posetrak {

/// Returns a string encoding the git commit and build timestamp, e.g.
/// "65893c35 built 2026-06-24T06:31:00Z" or "65893c35+ built ..." for dirty trees.
std::string version_string();

}  // namespace posetrak
