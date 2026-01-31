# CLI executable
cli_sources = files(
  'track.cpp',
)

executable('posetrak',
  cli_sources,
  include_directories: posetrak_inc,
  dependencies: [posetrak_dep, cli11_dep, pinocchio_dep],
  install: true
)
