# Pose models and their licences

Posetrak finds the body keypoints of each person with a pose model. Two families are available:

| Model | Keypoints | Notes |
|---|---|---|
| **SOMA-77** (`soma77-vitpose-h`) | 77 joint centres of NVIDIA's SOMA skeleton, including the fingers | The recommended model and the preselected one. Slower per person than RTMPose, with a flip test and a person-mask gate that make it steadier. |
| **RTMPose-133** (`rtmpose-l-133kp`), ViTPose-133 | 133 COCO whole-body surface landmarks | Used when SOMA-77 is not available; also has a separate hand refinement pass. |

## Getting the SOMA-77 weights

The SOMA-77 weights are NVIDIA's GEM-X 2D pose model. They are not part of Posetrak or its installer.
The first time you choose SOMA-77 (in the detection dialogs, in the segmentation panel's pose row, or with
`posetrak detect run`), Posetrak shows you the texts of two licences (both ship with Posetrak, so reading and accepting them
needs no internet connection; only the download does):

* the NVIDIA Open Model License Agreement, and
* the DINOv3 License Agreement (the model's image backbone is DINOv3; until NVIDIA confirms otherwise Posetrak
  assumes its terms apply).

If you accept both, the weights (about 3.4 GB) are downloaded from the `nvidia/GEM-X` repository on Hugging
Face into `~/.posetrak/models` (set `POSETRAK_MODEL_DIR` to put them elsewhere) and your acceptance is stored
beside them together with a hash of the licence texts you accepted. You are asked once, and again after an
update of Posetrak that ships a changed licence text.

If you decline, or the download is not possible, the run uses RTMPose-133 instead and the program tells you why.
Both models stay selectable at any time.

On the command line:

```
posetrak -s session.db detect run --capture <id> --sync <id> --start 12 --end 105
posetrak -s session.db detect run ... --accept-licences      # no terminal: accept the licences named above
posetrak -s session.db detect run ... --pose-model rtmpose-l-133kp
```

If you obtained the model file yourself under its licences, point `POSETRAK_SOMA77_MODEL` at its
`onnx/vitpose.onnx`; the download and the licence prompt are skipped.

## Notices

Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

Built with DINOv3.

The same notices are in **Help → About PoseTrak**. The licences forbid, among other things, processing video of
people without the legal authority and consent where it is required; Posetrak is meant for motion capture of
consenting performers.

## Running SOMA-77 on boxes you already have

`--segmentation RUN` takes the person boxes from a segmentation, and `--from-run RUN` reuses the boxes and track
ids of an earlier detection run, so a new pose model can be tried without detecting people again:

```
posetrak -s session.db detect run --capture <id> --sync <id> --start 12 --end 105 \
    --segmentation <seg run>              # boxes from the masks; heatmaps are gated by the person's mask
posetrak -s session.db detect run --capture <id> --sync <id> --start 12 --end 105 \
    --from-run <detection run>            # boxes and track assignments of that run
```
