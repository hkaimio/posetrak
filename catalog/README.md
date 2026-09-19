# Catalog

Nominal, reusable marker definitions belong here. `catalog/modules/` holds the
per-module marker layouts, one `<module>.marker-module.yaml` each, read by
`posetrak.markers.catalog` (`catalog_module("leg")`). The file format is described
in that module's docstring, and the design in
`docs/roadmap/features/marker-based-mocap/productization-architecture-and-plan.md` §3.2.

Capture-specific results do not belong in the repository. Calibrated
attachment sets and dot-augmented skeletons are products of one capture and
are kept beside that capture's session database; the session database holds
the skeletons a tracking run actually used.

Three capture-specific files remain only because scripts that are still the
working path read them by this relative path:

| File | Read by |
|---|---|
| `ball.single-reflective-dot.*.yaml` | `python/tools/finalize_ball_cutie_detection.py`; import it as the ball's marker body with `posetrak marker-body import` |
| `pen.calibrated.*.yaml`, `pad.calibrated.*.yaml` | nothing; the script that read them is replaced by `posetrak marker-body import`, `capture object add` and `sequence finalise-object --object` |

Remove them in the change that deletes those scripts (`posetrak sequence
compose`, plan WS1 item 1).
