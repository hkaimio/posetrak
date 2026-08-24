# Posetrak

![Posetrak](assets/banner.png){ align=right width=240 }

Video-based markerless motion capture using an Unscented Kalman Filter in joint
space. Supports an arbitrary number of heterogenous cameras and tracking
multiple performers simultaneously. Tracking results exported as BVH files
usable in Blender and other 3D animation applications.

Main motivation for Posetrak development were challenges in tracking
multi-person, close-contact scenes like martial arts. Read here more about
project's [background](background.md).


<div class="video-embed">
<iframe src="https://player.vimeo.com/video/1220935922?muted=1&loop=1"
        allow="autoplay; fullscreen; picture-in-picture"
        allowfullscreen></iframe>
</div>


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
- **[Tutorial](user-guide/tutorial1.md)** — quick walkthrough of main Posetrak features using a concrete example
- **[User Guide](user-guide/first-capture.md)** — capturing, calibrating, tracking, troubleshooting
- **[Architecture](architecture/overview.md)** — system design, data model, the UKF solver
- **[Reference](skeleton-format.md)** — file formats (skeleton YAML, state vector, sync metadata)
- **[Roadmap](roadmap.md)** — where the project is headed
- **[Background](background.md)** — why Posetrak exists

## License

Apache License 2.0. This project is [REUSE](https://reuse.software/) compliant.
