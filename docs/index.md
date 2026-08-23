# Posetrak

![Posetrak](assets/banner.png){ align=right width=240 }

Video-based markerless motion capture using an Unscented Kalman Filter in joint
space. Supports an arbitrary number of synchronized cameras and configurable
skeleton definitions, tracking results exported as BVH files.

Main motivation for Posetrak development were chalelnges in tracking
multi-person, close-contact scenes. read here more about project's
[background](background.md).

## Getting started

See the [setup guide](setup.md) for full platform-specific instructions, then
[tutorial](user-guide/tutorial1.md) for a walkthrough of processing a simple
motion capture sequence.

## Applications

| Command | Description |
|---|---|
| `posetrak-ui` | Main GUI: sessions, capture setup, pose extraction, tracking, keypoint editing |
| `posetrak` CLI | Command line client |
| `posetrak-mcp` | Read-only MCP diagnostic server for tracking runs |


## Documentation

- **[Setup](setup.md)** — prerequisites, build instructions, both platforms
- **[User Guide](user-guide/first-capture.md)** — capturing, calibrating, tracking, troubleshooting
- **[Architecture](architecture/overview.md)** — system design, data model, the UKF solver
- **[Reference](skeleton-format.md)** — file formats (skeleton YAML, state vector, sync metadata)
- **[Roadmap](roadmap.md)** — where the project is headed
- **[Background](background.md)** — why Posetrak exists

## License

Apache License 2.0. This project is [REUSE](https://reuse.software/) compliant.
