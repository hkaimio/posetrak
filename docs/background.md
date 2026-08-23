# Posetrak History

## Why another motion tracking project?

Aikido might sound like a surprising starting point for a motion capture
project, but it turned out to be much better — and harder — match than I
could have imagined.

Professionally I've worked as a software engineer with images, graphics, and
cameras for more than 30 years; aikido has been my primary hobby for most of
that time. In summer 2025 I started looking for ways to capture aikido
techniques, for both visualization and analysis.

I tried the obvious options first — open source projects like FreeMoCap and
Pose2Sim, and some of the AI-based commercial services that were emerging at the
time (I left the high-end commercial systems like OptiTrack and Rokoko out of
consideration; this was, and is, a hobby project with a hobby budget).

None of them held up - aikido seemed to break their assumption at every level.

Aikido is a martial art practiced with one or multiple partners: tori (the
person applying a technique) guides uke (the person receiving it) into a
circular movement around herself, often at very close distance. There's usually
no camera angle from which the two don't occlude each other — and usually both
occlude different parts of each other at the same time, and the occlusions
change multiple times per second.

Many of the techniques are wrist locks, where one person grabs the other's hand,
so hand tracking obviously in crucial, but pose detectors have a hard time
telling which hand belongs to whom once they're locked together. And a lock
rarely stays at the wrist: it forces the whole joint chain from wrist through
elbow and shoulder up to the spine into an extreme pose, which means that
without careful parameterization, an IK solver spends most of its time hovering
right next to its singularities.

Quite often the existing tools failed to detect some of the practitioners at
all, or mixed their body parts into weird constellations. Most tools did
something as I expected but failed in other areas.

So I started writing small tools to fill the gaps & glue tools together: a
Blender plugin to [edit the raw pose detection
data](https://github.com/hkaimio/PoseEdit) by hand, and scripts to run OpenSim's
IK solver against Blender armatures and [export the
results](https://github.com/hkaimio/mocap-helpers/tree/main/opensim-to-bvh) as
BVH files for other animation tools.

These got the motion data where I wanted it, but the outliers and noise were
still a problem. Movements turned very jerky whenever a camera lost visibility
of a key body part. That sent me looking for ways to improve temporal stability
with filtering, first by smoothing individual keypoint trajectories with a
Kalman filter while still re-running inverse kinematics every frame to get the
skeleton pose.

The turning point was realizing those didn't need to be two separate steps. If
the filter operated directly on the skeleton's joint angles instead of on
individual 3D points, outlier rejection and temporal smoothing could be solved
against the skeleton itself — triangulation and IK were actually just getting in
the way.

That's the key idea behind Posetrak's solver: every frame it directly updates
the skeleton pose from the 2D keypoint detections using a massive unscented
Kalman filter, with no separate triangulation/IK step except to initialize the
very first frame (and even that isn't strictly necessary, the filter usually
converges quickly without it).

This has real benefits: the probabilistic
framework makes outlier detection robust, temporal smoothness vs. reaction speed
to quick movements becomes something you can tune, and the filter can use a
detection even when a keypoint is visible in only a single camera at a given
frame. It can even mix material from cameras running at different frame rates.

The downside is that the filter is computationally heavy. I wrote the first
version in Python, like the rest of my tooling, but switched to C++ soon after.
It's still far from real-time but usually fast enough for batch processing.

Eventually the project diverged too far from its origins to keep bolting onto
Pose2Sim-compatible plain-text files, so I migrated everything into a relational
database and built proper GUI and command-line tools around it.

## Posetrak now

Posetrak now largely achieves the goals I started with: I can capture pair and
multi-person aikido practice with equipment that's affordable for a serious
hobbyist, and I've used it successfully for plenty of activities beyond aikido
too. It's still very much a tool built for my own needs, though. Polishing it
into something more than that is a separate hurdle.

The Posetrak project has 3 parts:

- the actual unscented Kalman filter based solver written in C++

- the backplane that stores data needed for the pipeline in an SQLite database

- and tools for executing other parts of the capture pipeline (some of which are
  written by me; others utilize existing open source projects).


These are largely separate. The backplane, for example, could be useful with
other solvers too. This is one of the reasons I decided to release Posetrak as
an open source project. Currently the parts are not as nicely decoupled as I
would like them to be, so this is one potentially interesting area of work.

Looking ahead, I want to keep improving the solver and usability. An immediate
need is to add marker-based capture. My main use case for that is better
tracking of props like weapons.

Single-camera and face mocap are both low on my list: for aikido specifically,
I don't think single-camera capture is feasible until AI models have enough
training material on close-contact multi-person scenes like this — which is,
after all, the whole reason Posetrak exists. For now the focus stays on
multi-camera.

One more thing I'd like to try, eventually: my wife works as a professional
animal trainer in the film industry, and I'd love to do some mocap work with her
animals.
