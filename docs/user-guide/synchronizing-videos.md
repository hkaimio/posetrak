# Synchronizing videos

*Draft — outline only.*

Cameras record independently, so their video frames aren't captured at
the same real-world moments (each has its own start delay, and frame
rate can drift slightly over a long take). Posetrak needs to know, for
every camera, which of its frames corresponds to which moment on a
shared timeline before it can combine multiple cameras' observations of
the same instant.

## Setting up sync points

- Each capture has a **Set up sync…** button. For every video, you set
  at least 2 (frame, shared timestamp) points — Posetrak interpolates
  between them, and can flag drift if you set more than 2.
- Picking frames that are easy to identify precisely in every camera is
  the whole game here. Two techniques that work well:
  - **Clapper / hand-clap** — a sharp, visually distinct action performed
    at a single instant. Doing it at both the start and end of a take
    syncs the beginning and lets you catch drift over a long take.
  - **Blinking light** (LED, headlamp) placed in view of every camera —
    a light turning *on* is an even sharper, easier-to-pick single frame
    than a clap, and doesn't need a performer.
- *(TBD: what the sync UI itself looks like — the frame scrubber, how a
  sync point actually gets marked, how per-camera offset/drift is
  reported once 2+ points are set.)*

## Troubleshooting

- **Frame rate ratio warning when marking a sync pair** — some phones'
  slow-motion recording modes capture at a much higher rate (e.g. 120
  fps) than the frame rate the video file itself declares (e.g. 30 fps,
  intended for slow-motion playback). Posetrak computes each camera's
  actual frame rate from your sync points and warns when it doesn't
  match what the file claims. Click the suggested corrected fps in the
  warning dialog — the timeline recalculates using the real capture
  rate, which is what lets that camera line up with the others.
- *(TBD: what it looks like when sync is off in other ways — warnings at
  trial creation? bad triangulation later on? how to tell sync is the
  actual cause rather than, say, extrinsics.)*

## See also

- [Your first capture](first-capture.md)
- [Extrinsics calibration](extrinsics-calibration.md) — assumes sync is
  already set up for the capture.
