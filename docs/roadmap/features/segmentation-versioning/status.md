```toml
name = "Segmentation Mask Versioning"
status = "planned"
progress_pct = 0
description = """
Append-only mask revisions per segmentation run, the revision recorded with every detection and \
tracking run that used masks, a latest-revision view for current readers, staleness check and \
incremental refresh of mask-sourced detection runs.
"""
categories = ["data-model", "segmentation"]
target_release = "TBD"
last_updated = 2026-10-17
```

# Segmentation Mask Versioning — Implementation Status

See [segmentation-versioning-design.md](segmentation-versioning-design.md). Planned as part of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md). The open
issue "Segmentation masks are mutable but tracking runs do not record which version they used" in
`docs/open-issues.md` is closed by this feature.

| Item | Status |
|---|---|
| `seg_mask_revisions`, `seg_masks.revision`, `seg_masks_latest` view | Planned |
| Revision handling in the segmentation panel | Planned |
| Concurrent use: pinned reads, one open revision per run | Planned |
| Revision recorded by detection and tracking runs | Planned |
| Readers moved to the view or to a pinned revision | Planned |
| `detect stale` and incremental refresh | Planned |
| `seg prune` | Planned |
