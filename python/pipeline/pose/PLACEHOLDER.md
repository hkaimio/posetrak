# Pipeline pose extraction scripts

Copy the following files from `rtmlib/harritests/` into this directory:

- `poseanalysis.py`   (core YOLO/RTMpose library wrapper)
- `pose_extraction.py` (Marimo app)
- `video_sync.py`     (Marimo app)
- `export_to_openpose.py` (utility)

Note: `poseanalysis.py` currently imports `ipycanvas`, `ipywidgets`, and `IPython` at module
level — these are unused Jupyter dependencies to be removed in Phase 1g.
