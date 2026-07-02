# Documentation restructuring plan

Analysis and proposed structure for the public documentation rewrite.

---

## Current state assessment

### Severely outdated docs

The `README.md` and `implementation-status.md` describe the project in Phase 0–1 of C++
development (circa February 2026), when the tracker still failed on real data.  They make no
mention of the three GUI applications, the SQLite data model, or any Python tooling.  A
first-time reader would have no idea what the project actually is today.

`CONTRIBUTING.md` references GTest (the project uses Catch2), links to
`cpp-implementation-plan.md` as the active roadmap, and describes a single-language C++
workflow with no mention of the Python apps.

`cpp-architecture-overview.md` (980 lines) and `cpp-detailed-architecture.md` (1252 lines)
are pre-implementation design documents.  The C++ API shapes shown in pseudocode do not
match the actual code.  They describe planned features (Python bindings, libarchive,
OpenMP parallelization, GTest, separate TRC/BVH exporters) that are not implemented.  The
accurate C++ architecture description is in `CLAUDE.md`, not in any public doc.

### Dead planning docs (6085 lines total; none should be published)

| File | Reason |
|---|---|
| `cpp-implementation-plan.md` (893 lines) | Phased plan — executed |
| `cpp-requirements.md` (477 lines) | Pre-implementation requirements |
| `cpp-detailed-architecture.md` (1252 lines) | Pre-implementation design, wrong now |
| `rebuild-plan.md` (263 lines) | Repo restructuring — done |
| `error-state-refactoring-plan.md` (230 lines) | Completed refactoring |
| `phase2-implementation-plan.md` (202 lines) | Completed phase |
| `phase-7-8-detailed-plan.md` (1034 lines) | Completed phases |
| `camera-id-refactoring-design.md` (556 lines) | Completed refactoring |
| `refactoring-full-dof-storage.md` (346 lines) | Completed refactoring |
| `plans/` subdirectory | Phase planning history |
| `pinocchio-header-only-analysis.md` (115 lines) | Decision made; summary already in CLAUDE.md |

### Debug / investigation notes (archive, do not publish)

- `bisect-regression-findings.md` (113 lines)
- `fk-validation-debug.md` (185 lines)
- `filter-inconsistency-diagnosis-20260322-teacup.md` (98 lines)
- `implementation-status.md` — Feb 2026 snapshot; tracker "fails on real data"; everything changed

### Blog content (not documentation)

- `blog-post.md`, `blog-posetrak-chapter-draft.md` — move to repo root or `blog/`

### Good docs that lack a proper home

| File | Status |
|---|---|
| `architecture-notes.md` | Accurate, recent; written as "source material for a future doc" — use it |
| `data-model-and-storage.md` | Excellent design doc; mostly current; C++ path sections need status note |
| `workflow-session-to-bvh.md` | Real, working workflow; needs minor update for current DB/app flows |
| `ui-status-and-roadmap.md` | Most current roadmap (May 2026); use as source for roadmap section |
| `python-guidelines.md` | Accurate; keep under architecture or contributing |
| `skeleton-format.md`, `state-vector-format.md`, `sync-metadata-format.md` | Accurate references; need a home |

### Future design docs (keep, but organise under `design/`)

These describe planned features; none is published as user-facing documentation until implemented.

- `hierarchical-ukf-design.md`, `hierarchical-tracker-redesign.md`
- `per-frame-measurement-noise-design.md`, `analytical-init-seeding-design.md`
- `segmentation-keypoint-weighting-design.md`
- `extrinsics-calibration-design.md`, `triangulated-distance-calibration-design.md`
- `skeleton-scaling-calibration-design.md`, `cc-skeleton-export-design.md`
- `aruco-prop-tracking-design.md`
- `new-stitching-ui-concept.md`, `new-stitching-ui-design.md`
- `skeleton-visualization-design.md`
- `rerun-visualization-design.md`, `rerun-implementation-phase1.md`
- `cutie-init-widget-plan.md`
- `camera-management-design.md` (check if implemented)
- `pose-extraction-app-design.md`, `pipeline-ui-requirements.md` (check currency)
- `keypoint-editing/keypoint-editing-brief.md`, `keypoint-editing/keypoint-editing-design.md`

### Research notes (fold insights into architecture docs; archive originals)

- `ukf-parameter-sweep.md`
- `seg-pose-experiment-notes.md`

---

## Topics missing from current documentation

1. **Installation guide** — nothing currently tells a new user how to install or run the apps
2. **Accurate top-level README** — does not describe what the project is today
3. **System overview** — no diagram showing how the three apps + CLI + C++ tracker relate
4. **Camera calibration workflow** — how to set up intrinsics, extrinsics, and sync
5. **Tracker configuration reference** — TOML params are not documented for users
6. **Output formats reference** — the eight CSV files the tracker writes, BVH format
7. **UKF algorithm explanation** — user-facing description of what the noise and outlier
   parameters do (needed to tune the tracker intelligently)
8. **Tracking result interpretation** — NIS, inlier count, covariance condition number
9. **Troubleshooting guide** — common failure patterns, parameter tuning strategies
10. **C++ tracker architecture** — accurate post-implementation description

---

## Proposed structure

```
README.md                         rewrite: what is it, quick feature list,
                                  pointer to docs/user-guide/installation.md

CONTRIBUTING.md                   update: current tools (Catch2 not GTest),
                                  Python test setup, commit conventions

docs/
│
├── doc-restructuring-plan.md     (this file)
│
├── user-guide/
│   ├── installation.md           NEW: build C++ tracker, install Python apps,
│   │                             uv setup, platform notes
│   ├── workflow-overview.md      UPDATE from workflow-session-to-bvh.md
│   │                             end-to-end step sequence; pointers to detail pages
│   ├── camera-setup.md           NEW: intrinsics, extrinsics, sync; setup wizard
│   ├── pose-extraction.md        UPDATE from pose-extraction-app-design.md:
│   │                             how to run detection, stitch, finalise
│   ├── running-the-tracker.md    NEW: UI dialog + CLI; skeleton + config; output
│   ├── keypoint-editing/         MOVE user guide from
│   │   └── keypoint-editing-user-guide.md   docs/roadmap/features/keypoint-editing/
│   │                             (design docs + status.md stay in roadmap/features)
│   ├── configuration-reference.md  NEW: all TOML config params, UKF param meaning
│   └── troubleshooting.md        NEW: interpreting stats, common failures, tuning
│
├── architecture/                 ← being written now
│   ├── overview.md               NEW: system diagram + data flow
│   ├── data-model.md             UPDATE from data-model-and-storage.md (trimmed)
│   ├── cpp-tracker.md            NEW: accurate post-implementation C++ description
│   ├── python-apps.md            NEW: three apps, shared DB, code layout
│   └── algorithms/
│       └── ukf.md                NEW: user-facing UKF explanation
│
├── reference/
│   ├── skeleton-format.md        MOVE from docs/
│   ├── state-vector-format.md    MOVE from docs/
│   ├── sync-metadata-format.md   MOVE from docs/
│   └── output-formats.md         NEW: 8 CSV files, BVH, obs_blob layout
│
├── status-and-roadmap.md         NEW: current status + near/long-term plans
│                                 (replaces implementation-status.md + ui roadmap §5
│                                 + open-issues.md)
│
└── design/                       future feature designs; not user-facing
    ├── ui/
    │   ├── new-stitching-ui/
    │   ├── skeleton-visualization-design.md
    │   └── rerun-visualization/
    ├── tracker/
    │   ├── hierarchical-ukf.md
    │   ├── per-frame-noise.md
    │   ├── analytical-init.md
    │   └── segmentation-weighting.md
    ├── calibration/
    │   ├── extrinsics-calibration.md
    │   └── triangulated-distance-calibration.md
    ├── skeleton/
    │   ├── skeleton-scaling.md
    │   └── cc-export.md
    ├── export/
    │   └── aruco-prop-tracking.md
    └── keypoint-editing/         technical design docs (brief + design)
```

**Archive** (move to `docs/archive/`, excluded from published output):

All "dead planning docs" and "debug notes" listed in the assessment above, plus
`implementation-status.md`, `cpp-architecture-overview.md`, and the `plans/` subdirectory.

---

## Work order

| Priority | File | Source material | Status |
|---|---|---|---|
| 1 | `architecture/overview.md` | `architecture-notes.md`, CLAUDE.md | **Draft written** |
| 2 | `architecture/cpp-tracker.md` | CLAUDE.md, `architecture-notes.md` §C++ | **Draft written** |
| 3 | `architecture/python-apps.md` | `architecture-notes.md` §Python, `ui-status-and-roadmap.md` §1-2 | **Draft written** |
| 4 | `architecture/data-model.md` | `data-model-and-storage.md` (trim) | **Draft written** |
| 5 | `architecture/algorithms/ukf.md` | CLAUDE.md, `ukf-parameter-sweep.md` | **Draft written** |
| 6 | `README.md` | rewrite from scratch | — |
| 7 | `user-guide/installation.md` | CLAUDE.md build section | — |
| 8 | `user-guide/workflow-overview.md` | `workflow-session-to-bvh.md` | — |
| 9 | `reference/output-formats.md` | `state-vector-format.md`, tracker source | — |
| 10 | `user-guide/configuration-reference.md` | `workflow-session-to-bvh.md` §8, tracker source | — |
| 11 | `status-and-roadmap.md` | `ui-status-and-roadmap.md`, open-issues.md | — |
