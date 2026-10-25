// SPDX-FileCopyrightText: 2026 Harri Kaimio
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace posetrak {

/// @brief How long a session database connection waits for another process's write lock
/// before failing with "database is locked", in milliseconds. Matches BUSY_TIMEOUT_S in
/// the Python db layer.
constexpr int kSessionDbBusyTimeoutMs = 60'000;

}  // namespace posetrak
