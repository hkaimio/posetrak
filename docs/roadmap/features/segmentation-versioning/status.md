```toml
name = "Segmentation Mask Versioning"
status = "in_progress"
progress_pct = 95
description = """
Append-only mask revisions per segmentation run, the revision recorded with every detection and \
tracking run that used masks, a latest-revision view for current readers, staleness check and \
incremental refresh of mask-sourced detection runs.
"""
categories = ["data-model", "segmentation"]
target_release = "TBD"
last_updated = 2026-10-20
```

# Segmentation Mask Versioning — Implementation Status

See [segmentation-versioning-design.md](segmentation-versioning-design.md). Built as stage 8 of
[the SOMA/GEM-X productization plan](../pose-model-spike/soma-gemx-productization-plan.md). The open
issue "Segmentation masks are mutable but tracking runs do not record which version they used" in
`docs/open-issues.md` is closed by this feature.

| Item | Status |
|---|---|
| `seg_mask_revisions`, `seg_masks.revision`, `seg_masks_latest` view, revision-0 trigger (session schema 68; provisional number) | Implemented |
| Migration: `seg_masks` rebuilt with revision 0, progress logged | Implemented; measured on copies of two real session databases (design §4) |
| Revision handling in the segmentation panel (open on first write, close on job end / queue end / close, read-only second editor, crash recovery) | Implemented |
| Concurrent use: pinned reads, one open revision per run | Implemented; two-process test on a real tracking run (design §4) |
| Revision recorded by detection runs (mask box source, SOMA mask gate) and tracking runs (`--seg-revision`, `tracking_run_persons`) | Implemented |
| Readers moved to the view or to a pinned revision | Implemented: person view overlay, session tree and segmentation counts, mask box source, object finalisation mask filter, tracklet mask test, tracker, visualisation tool |
| `detect stale` and `--refresh` | Implemented |
| `seg prune` | Implemented |

Open points:

- A detection run assembled from several pose jobs (the segmentation window's "Queue Pose") records the
  revision of its first job; a job started after more edits reads a newer revision without updating the
  record.
- The crash-recovery offer works for editors on the same host only; a revision left open by another host
  stays read-only until it is saved or discarded from there.
- The migration leaves the old mask table's pages free inside the database file; `VACUUM` returns them.
- The schema number 68 is provisional until the branch merges.
