# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

# CLI executable
cli_sources = files(
  'track.cpp',
)

executable('posetrak-tracker',
  cli_sources,
  include_directories: posetrak_inc,
  dependencies: [posetrak_dep, cli11_dep, pinocchio_dep, sqlite3_dep],
  install: true
)
