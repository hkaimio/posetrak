# Posetrak History

## Why another motion tracking project?

Aikido is a strange test case for motion capture. It's a martial art practiced with a partner: tori (the person applying a technique) guides uke (the person receiving it) into a circular movement around herself, often at very close distance. There's usually no camera angle from which the two don't occlude each other — usually both occlude different parts of each other at the same time, and the occlusions change multiple times per second. Many of the techniques are wrist locks, where one person grabs the other's hand — hand tracking is obviously what matters here, but pose detectors have a hard time telling which hand belongs to whom once they're locked together. And a lock rarely stays at the wrist: it forces the whole joint chain from wrist through elbow and shoulder up to the spine into an extreme pose, which means that without careful parameterization, an IK solver spends most of its time hovering right next to its singularities.

This is the problem that got me here. Professionally I've worked as a software engineer with images, graphics, and cameras for more than 30 years; aikido has been my primary hobby for most of that time. In summer 2025 I started looking for ways to capture aikido techniques, for both visualization and analysis. I tried the obvious options first — open source projects like FreeMoCap and Pose2Sim, and some of the AI-based commercial services that were emerging at the time (I intentionally left the high-end commercial systems like OptiTrack and Rokoko out of consideration; this was, and is, a hobby project with a hobby budget).

None of them held up — for exactly the reasons above. Most affordable motion tracking solutions are built around capturing a single person, and aikido breaks that assumption at every level.

So I started writing small tools to fill the gaps: a Blender plugin to edit the raw pose detection data by hand, and scripts to run OpenSim's IK solver against Blender armatures and export the results as BVH files for other animation tools. These got the motion data where I wanted it, but the outliers and noise were still a problem — movements turned very jerky whenever a camera lost visibility of a key body part. That sent me looking for ways to improve temporal stability with filtering, first by smoothing individual keypoint trajectories with a Kalman filter while still re-running inverse kinematics every frame to get the skeleton pose.

The turning point was realizing those didn't need to be two separate steps. If the filter operated directly on the skeleton's joint angles instead of on individual 3D points, outlier rejection and temporal smoothing could work against the skeleton itself — triangulation and IK, run every frame, were actually just getting in the way.

That's the key idea behind Posetrak's solver: every frame it directly updates the skeleton pose from the 2D keypoint detections using a massive unscented Kalman filter, with no separate triangulation/IK step except optionally to initialize the very first frame (and even that isn't strictly necessary — the filter usually converges quickly without it). This has real benefits: the probabilistic framework makes outlier detection robust, temporal smoothness vs. reaction speed to quick movements becomes something you can tune, and the filter can use a detection even when a keypoint is visible in only a single camera at a given frame — it can even mix cameras running at different frame rates.

The downside is that the filter is computationally heavy. I wrote the first version in Python, like the rest of my tooling, but switched to C++ soon after — it's still not as fast as I'd like.

Eventually the project diverged too far from its origins to keep bolting onto Pose2Sim-compatible plain-text files, so I migrated everything into a relational database and built proper GUI and command-line tools around it — Posetrak, in its current form.

## Posetrak now

Posetrak now largely achieves the goals I started with: I can capture pair and multi-person aikido practice with equipment that's affordable for a serious hobbyist, and I've used it successfully for plenty of activities beyond aikido too. It's still very much a tool built for my own needs, though — polishing it into something more than that is a separate hurdle.

Looking ahead, I want to keep improving the solver and usability, and to add marker-based capture — my immediate use case there is better tracking of props like weapons. Single-camera and face mocap are both low on that list: for aikido specifically, I don't think single-camera capture is feasible until AI models have enough training material on close-contact multi-person scenes like this — which is, after all, the whole reason Posetrak exists. For now the focus stays on multi-camera.

One more thing I'd like to try, eventually: my wife works as a professional animal trainer in the film industry, and I'd love to do some mocap work with her animals.
