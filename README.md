# Posetrak — Multi-Camera Motion Capture Tracker

Video-based skeletal motion capture using an Unscented Kalman Filter in joint space.
Supports arbitrary number of synchronized cameras, configurable skeleton definitions,
and fisheye/wide-angle lenses.

See [Posetrak web page](https://hkaimio.github.io/posetrak/)

## Applications

| Command | Description |
|---|---|
| `posetrak-ui` | Main GUI: sessions, capture setup, pose extraction, tracking, keypoint editing |
| `posetrak track config.toml` | Run the UKF tracker on a capture session |
| `posetrak scale config.toml` | Post-process a bone-length calibration run |
| `posetrak-mcp` | Read-only MCP diagnostic server for tracking runs |

`posetrak` also covers session/capture/detection management without the GUI
(`posetrak session`, `posetrak detect`, `posetrak track`, etc.) — see
[Architecture: Python apps](docs/architecture/python-apps.md) for the full CLI.

## Getting started

There's no packaged release yet — using Posetrak today means setting up a
development environment and building/running from source. See
**[docs/setup.md](docs/setup.md)** for the full setup guide (Linux and
Windows, native build or cross-compiled exe).

## Documentation

- [Setup guide](docs/setup.md) — prerequisites, build instructions, both platforms
- [Architecture overview](docs/architecture/overview.md) — system design, data model, the UKF solver
- [Skeleton format](docs/skeleton-format.md) — YAML skeleton file format
- [Python guidelines](docs/python-guidelines.md) — Python code conventions

## License

Apache License 2.0 — see [LICENSE](LICENSE). This project is [REUSE](https://reuse.software/) compliant; per-file licensing is in [REUSE.toml](REUSE.toml) and file headers.
