# Posetrak command line interface

I want all (or as much as possible) of the Posetrak UI functionality to be available also via command line.

## Key features

* View existing database content:
  posetrak-cli capture|trial|detection|tracking list
  posetrak-cli camera list
  posetrak-cli capture show <capture-id>  # details of the capture in human readable form
  posetrak-cli trial list --capture-id <capture-id> --json # All trials of a capture as JSONL

* Import & export data. Exporting e.g. a trial will create a new database file with that trial, all of its children (detections, tracking runs, etc.) and parents + other data like cameras needed for that trial.

* Create new captures (import video files & extrinsics, assign cameras), trials, detections (run the pose detection algorithms) and tracking runs (run the actual posetrak tracker).

* Import, export & scale skeletons.

* Export tracking results as BVH/glTF/USD/etc. formats.
