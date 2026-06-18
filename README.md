# Posetrak — Multi-Camera Motion Capture Tracker

Video-based skeletal motion capture using an Unscented Kalman Filter in joint space.
Supports arbitrary number of synchronized cameras, configurable skeleton definitions,
and fisheye/wide-angle lenses.

## Applications

| Command | Description |
|---|---|
| `posetrak track config.toml` | Run the UKF tracker on a capture session |
| `posetrak scale config.toml` | Post-process a bone-length calibration run |
| `posetrak-ui` | Main viewer: sessions, tracking runs, keypoint editing |
| `posetrak-pose` | Detection pipeline: YOLO + RTMPose on video, track assignment |
| `posetrak-db` | Database CLI: import, export, query sessions |
| `posetrak-mcp` | Read-only MCP diagnostic server for tracking runs |

## Getting started

See **[docs/setup.md](docs/setup.md)** for full platform-specific setup instructions.

**Linux (development)** — build the C++ tracker from source, install the Python tools:

```bash
meson setup builddir       # configure (downloads deps via wraps)
meson compile -C builddir  # build
uv sync --group dev        # Python tools + test deps
uv run pytest              # run Python tests
./run_tests.sh             # run C++ tests
```

**Windows** — use the cross-compiled exe (see docs/setup.md), install Python tools:

```bash
uv sync                    # base Python tools
```

## Documentation

- [Setup guide](docs/setup.md) — prerequisites, build instructions, both platforms
- [Architecture overview](docs/cpp-architecture-overview.md) — C++ tracker design
- [Skeleton format](docs/skeleton-format.md) — YAML skeleton file format
- [Python guidelines](docs/python-guidelines.md) — Python code conventions

## License

MIT
