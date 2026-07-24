# Tracker configuration & person-model improvements — design proposal

## Status

Implementation started 2026-07-24, phases 0-3 done (plus three rounds of
live-review fixes on top, see below), phases 4-5 done. Written in response to
Harri's brief (`confg-improvement-brief.md`, this directory) plus a
codebase investigation done before drafting this doc — several of the
brief's proposed mechanisms turned out to already exist in the registry
backend (`python/posetrak/db/manage_config.py`) but are unused by the GUI,
and one investigation finding was a real, silent data-loss bug (see
"Prerequisite fix" below), now fixed.

**Phase 0 — done.** `edit_config()`/`create_config_from_toml()`
(`python/posetrak/db/manage_config.py`) rewritten to derive their column
list from `PRAGMA table_info(tracker_configs)` at call time instead of a
hardcoded parameter list, so a future migration needs no change here to
stay correct. `edit_config()` also now copies every `tracker_config_stages`
row forward to the new config ID (previously silently dropped). Both
functions' call sites (`python/posetrak/cli/config.py`,
`python/posetrak/db/cli.py`, `python/tools/param_sweep.py`,
`python/tools/run_project.py`) needed no changes — all already call with
keyword arguments, which a `**overrides`-based signature accepts
identically. New regression tests in `python/tests/db/test_manage_config.py`
cover: post-v21 scalar columns surviving an edit, a list/dict override
value being JSON-encoded for an arbitrary column (not just the one column
the old code special-cased), `tracker_config_stages` rows surviving an
edit, and `create_config_from_toml()` reading a post-v21 TOML field. Full
`python/tests/db` suite: 248/250 (2 pre-existing, unrelated failures,
confirmed via `git stash` to fail identically without this change —
`test_observation_edits.py::test_edit_marks_outlier_zeroes_confidence` and
`test_posetrak_db.py::test_resolve_path_absolute`).

**Phase 1 — done.** Schema: `tracker_configs.is_named` (added to
`db/registry_schema.sql`, inherited by both fresh registries and fresh
sessions) plus `captures.default_tracker_config_id`/
`trials.default_tracker_config_id` (`db/session_schema.sql`). Registry
schema version bumped 6→7, session 37→38, via new
`_migrate_registry_v6_to_v7()`/`_migrate_session_v37_to_v38()`
(`python/posetrak/db/db.py`).

Along the way, found and fixed a second, adjacent instance of the same
"stale hardcoded migration" bug class Phase 0 fixed: the registry's own
migration chain (`_migrate_registry_v1_to_v2` … `v5_to_v6`) had stopped
tracking `tracker_configs` columns after v6 — every column the *session*
chain added from v22 onward (pose_noise_std and ~34 more) was never
mirrored into a registry migration, so a registry DB created or last
opened before that point would be silently missing most of the table's
current columns. `_migrate_registry_v6_to_v7()` fixes this generically
(builds a reference `tracker_configs` from the current
`registry_schema.sql` in an in-memory DB, diffs its columns — including
type, `NOT NULL`, and `DEFAULT` — against the real connection's, and
`ALTER TABLE ADD COLUMN`s whatever's missing), so it also serves as the
catch-up mechanism for every prior gap, not just `is_named`.

`is_named` needed bespoke handling, not the generic tuning-column
carry-forward machinery Phase 0 built: `manage_config.edit_config()`'s copy
now defaults every new row to `is_named=0` regardless of the source row's
own value (an edit never silently inherits "named" status — matches the
"editing a named config produces an unnamed working copy" rule above),
overridable via an explicit `is_named=True` kwarg for a future "Save"/"Save
as…" action; `create_config_from_toml()` defaults to `is_named=True` (every
current caller already supplies a real, deliberate name), overridable via
`is_named=False`.

New `seed_baseline_tracker_config()` (`manage_config.py`, mirroring
`manage_skeleton.seed_default_skeletons()`) idempotently inserts the fixed
`"factory-defaults"` row (`is_named=1`) the default-config resolution
chain terminates in; wired into `create_registry()` for fresh registries
and into `_migrate_registry_v6_to_v7()` for pre-existing ones being
upgraded. Not seeded into session DBs directly — mirrors the same
copy-on-demand self-containment pattern already used for skeletons
(`copy_config_to_session()`), left for the phase that actually builds the
resolution-chain lookup to wire up.

One incidental finding while writing the migration-downgrade test
fixtures: SQLite's `ALTER TABLE ... DROP COLUMN` does a text-level rewrite
of the table's *stored* CREATE TABLE SQL, and a bare comma inside one of
this schema's descriptive `--` comments near the dropped column can
corrupt that rewrite (`sqlite3.OperationalError: ... incomplete input`) —
confirmed independent of anything in this change, and irrelevant to any
real migration here (every migration in this codebase only ever uses `ADD
COLUMN`). Only mattered for a test helper simulating an old schema by
dropping columns from a live table; worked around there via `CREATE TABLE
... AS SELECT` (which strips comments, sidestepping the bug) rather than
`ALTER TABLE ... DROP COLUMN` directly.

New tests: `python/tests/db/test_manage_config.py` (is_named default/
override semantics for both functions, `seed_baseline_tracker_config()`
idempotency and shape) and `python/tests/db/test_posetrak_db.py`
(`test_migrate_registry_v6_to_v7_catches_up_stale_columns_and_adds_is_named`,
`test_migrate_session_v37_to_v38_adds_config_default_columns`, plus fresh-
create coverage for both). Full `python/tests/db` suite: 256/258 (the
same 2 pre-existing, unrelated failures as Phase 0). Full
`python/tests` collection (1265 tests) succeeds with no import errors;
the full run doesn't complete end-to-end in this environment for reasons
confirmed unrelated to this change (a pre-existing crash partway through
the GUI/app test directories, reproduced identically on a clean `git
stash` of this work).

**Phase 2 — done.** `RunTrackerWidget` (`python/app/pose/run_tracker.py`)
restructured per B1/B2/B3.

- **B1**: the single `QScrollArea`/`QFormLayout` replaced by a
  `QTabWidget` with `setTabPosition(QTabWidget.West)` — Summary (a
  read-only rollup, refreshed via `currentChanged` whenever it becomes
  active, plus once at construction), UKF & process model, Observations &
  outliers, Adaptive process noise, Pose reg. & joint limits, NIS
  feedback, Cross-person coupling, and Hierarchical solver stages
  (the pre-existing `_stage_table`, promoted from a form row into its own
  tab). No "Initialization" tab and no `alpha`/`beta`/`kappa`/
  `near_limit_*` fields anywhere, per Harri's review decision recorded
  above — those stay config-file-only.
- **B2**: new `NumericLineEdit` (`QLineEdit` + `QDoubleValidator`,
  right-aligned) with a `value()`/`setValue()`/`valueChanged` surface
  matching `QDoubleSpinBox`'s, so it's a drop-in replacement everywhere
  `_float_spin()` was used (now removed) — every existing
  `.valueChanged.connect(...)`/`.setEnabled(...)` wire-up needed no other
  change. `cross_pair_max_n`/`cross_person_max_n` stay `QSpinBox` (small,
  genuinely-bounded integer counts — the design doc's stated exception).
- **B3**: a "Configuration:" bar above the tabs with a status label,
  "Load…" (lists `is_named=1` rows via `list_configs()`, newest first,
  applies the chosen row's values to every tab via `_apply_config_row()`),
  and "Save as…" (prompts a name, writes via `edit_config(...,
  is_named=True, name=...)`). `_start_tracking()`'s old bespoke
  `_create_config()` — a hand-rolled `INSERT` covering only the columns
  this dialog had widgets for — is gone, replaced by
  `_collect_config_overrides()` (the same override-dict shape, values only,
  no manual JSON encoding — `edit_config()`'s own `_encode()` handles that
  now) feeding `edit_config(conn, loaded_config_id or BASELINE_CONFIG_ID,
  is_named=False, **overrides)`. This is a genuine behavior improvement,
  not just a refactor: any tuning column this dialog has no widget for
  (`alpha`/`beta`/`kappa`, init-std, `min_cameras_for_init`) now correctly
  carries forward from whatever was loaded (e.g. a config tuned via the
  CLI) instead of always resetting to `NULL` as the old raw `INSERT` did.
- `edit_config()` gained one more small extension beyond Phase 1: an
  optional `name=...` override (previously the new row always kept the
  source's own name), needed for "Save as…" to actually rename — with its
  own test (`test_edit_config_name_override_produces_save_as_semantics`)
  distinguishing "Save" semantics (no `name=`, keeps the source's name,
  extends its version series) from "Save as…" (renames, starts/joins a
  different series).
- `_sync_stage_overrides()` (replacing the stage-table-to-DB half of the
  old `_create_config()`) always `DELETE`s a config's own
  `tracker_config_stages` rows before re-inserting from the stage table --
  necessary because `edit_config()` already copies the *source* config's
  stage rows forward, which would otherwise collide (same
  `(tracker_config_id, group_name)` primary key) with this run's own
  stage selection. Verified directly: loading a config that already has a
  `HandL` stage row, then starting a run with only `HandR` enabled,
  produces a new config whose stages are exactly `{HandR}` — the copied-
  forward `HandL` is gone, and the *original* loaded config's own `HandL`
  row is untouched.
- Incidental fix: `_export_bvh()` used `QProcess` without it ever being
  imported anywhere in the file — a pre-existing `NameError` waiting to
  happen the first time BVH export was actually used. Added to the
  `PySide6.QtCore` import while already touching that line.

Verified: widget constructs headlessly (`QT_QPA_PLATFORM=offscreen`) with
all 8 tabs present and the Summary tab populated; `NumericLineEdit`'s
value/setValue/valueChanged round-trips correctly; a full collect-overrides
→ `edit_config` → save → `_apply_config_row` round trip against a real
in-memory registry DB reproduces every tweaked value exactly, including a
JSON-encoded `process_noise_vel_scopes` list; the stage-override
DELETE-before-INSERT behavior above. Existing tests
(`test_run_tracker.py`, `test_run_tracker_hierarchical.py`, both pure-
helper-function coverage that doesn't touch the widget's internal layout)
still pass unchanged. `python/tests/db` + both `run_tracker` test files:
268/270 (same 2 pre-existing, unrelated failures). The full
`python/tests/app`/`python/tests` run still doesn't complete end-to-end in
this environment; reconfirmed via a second `git stash` comparison that the
crash point is byte-for-byte identical with and without this phase's
changes.

**Phase 3 — done.** Two new manage_config.py functions implement the
resolution chain from section A3: `resolve_default_tracker_config(session,
*, trial_id=..., capture_id=...)` (trial's own default → its capture's →
the baseline, seeding the baseline into *session* on demand -- session DBs
aren't auto-seeded, unlike registries, so this is what actually closes that
gap in practice for phase 3's purposes) and `set_default_tracker_config(...)`
(repoints exactly one of trial/capture, never both, matching "a
capture-level edit never silently changes what a trial-level override
resolves to"). Both have direct tests
(`test_manage_config.py`), including the fall-through and
scope-isolation cases.

On the GUI side, editing a session/capture/trial's *default* config has
nothing to do with people/detection runs/starting a run, so embedding all
of `RunTrackerWidget` just to reach its config tabs would drag in a lot of
unrelated machinery. Extracted the config-tabs-and-load/save portion into
a standalone `TrackerConfigWidget(QWidget)`
(`python/app/pose/run_tracker.py`) with a small public surface
(`set_connection`, `set_skeleton_ids`, `edit_velocity_cameras_context`,
`collect_overrides`, `sync_stage_overrides`, `validate_stage_overrides`,
`load_config_row`, plus public `loaded_config_id`/`loaded_config_name`
attributes).
`RunTrackerWidget` now embeds one `TrackerConfigWidget` instance instead of
building its tabs inline, delegating to it wherever it previously called
its own (now-removed) config methods directly — pure extraction, no
behavior change, reverified with the same collect/apply round-trip test
plus the existing `run_tracker` test files, all still green.

New `DefaultConfigDialog(QDialog)` wraps a standalone `TrackerConfigWidget`:
resolves the effective default for a trial or capture, loads it, and its
one dialog-level action — "Set as default" — always creates a new
`tracker_configs` row via `edit_config()` (copy-on-write, never mutates the
resolved row in place) and repoints only that scope's own
`default_tracker_config_id`. The embedded widget's own "Load…"/"Save as…"
still work unchanged inside this dialog (e.g. to load a totally different
named template, or to additionally save the tweaked default under a name
for reuse elsewhere) but neither of those repoints the scope's default by
themselves — only "Set as default" does. New
`build_default_config_row(conn, *, trial_id=..., capture_id=...)` builds
the actual "Default tracker config: ‹name› [Edit] [Change…]" row: "Edit"
opens `DefaultConfigDialog`; "Change…" is a lighter-weight action (no
`DefaultConfigDialog`, no new row) that lists named configs and repoints
directly — deliberately not bound by the copy-on-write rule since it isn't
an edit. The status label distinguishes a scope's own default ("‹name›")
from one inherited from its capture ("‹name› (inherited)"), verified
directly against a real session DB for all three cases (own, inherited,
falls through to baseline).

Wired into both `TrialPanel` and `CapturePanel`
(`python/app/ui/content_panels.py`) — additive rows, no other layout
changes; both panels construct cleanly with real (if minimal) DB fixtures
in a headless Qt session. New
`python/tests/app/test_run_tracker_default_config.py` (8 cases) covers the
extracted `TrackerConfigWidget`, `DefaultConfigDialog` (including the
scope-isolation case end-to-end against a real session DB, and re-opening
after "Set as default" resolving to the newly-set config rather than the
baseline again), and `build_default_config_row`'s three status-label
cases. Full run: `python/tests/app/test_run_tracker*.py` +
`test_content_panels_hierarchical.py` + `python/tests/db`: 292/294 (same 2
pre-existing, unrelated failures as every earlier phase).

**Live-review fixes, 2026-07-24 (Harri).** Two issues found trying phases
0-3 in the actual UI, both fixed:

1. **No way to save the config an existing tracking run actually used.**
   New `manage_config.name_existing_config(conn, config_id, name)` —
   deliberately **not** `edit_config()`'s copy-on-write: this names the
   *exact* row a run's `tracker_config_id` already points to, in place, so
   the run's own provenance link is unchanged rather than superseded by a
   new row, and no tuning value is touched (only `name`/`is_named`). Wired
   into both `_RunInfoPane` (the live sidebar shown while scrubbing a
   tracking run, e.g. from `PersonPanel`) and `TrackingRunPanel` (the
   dedicated full-page run view) via a shared `_save_run_config_dialog()`
   helper — both showed the same `_cfg_text()`/ID-group pattern already, so
   fixing it in one place without the other would have left them
   inconsistent. `_build_run_ids_group()`/`_populate_run_ids()` (already
   shared by both panels) gained a "Config:" row alongside the existing
   Run/Skeleton/Detection/Trial/Capture ones. New tests in
   `test_manage_config.py` (in-place naming, rename-overwrites-previous,
   invalid-id error).
2. **Vertical tabs rotated their text vertically too.** Plain
   `QTabWidget.setTabPosition(West)` gives Qt's default behavior of
   rotating each tab's *label* 90° to fit the vertical strip — matching
   neither the request nor Blender's/Visual Studio's settings dialogs,
   where the tab *shape* is vertical but the *label* stays upright. New
   `_HorizontalTabBar(QTabBar)` (a standard, well-known Qt/PySide recipe,
   not project-specific): overrides `tabSizeHint()` to report a
   transposed size (so the tab has room for upright text) and
   `paintEvent()` to draw each tab's shape normally but rotate the
   *painter* -90° only while drawing that tab's label. Wired into
   `TrackerConfigWidget`'s `_config_tabs` via `setTabBar()`. Verified
   visually, not just by absence of a crash: grabbed a headless
   (`QT_QPA_PLATFORM=offscreen`) render of both the fixed widget and a
   plain unfixed `QTabWidget` built the same way. The unfixed one stacks
   glyphs vertically per tab (very tall per-tab, needs scroll arrows,
   most tabs not visible at once) — reproducing exactly the reported bug.
   The fixed one lays glyphs out horizontally per tab, with all 8 tabs
   visible without scrolling. (Rendered glyphs are tofu boxes in this
   headless environment's font-less offscreen platform, not real letters —
   irrelevant to the layout/orientation comparison being checked.)

**Live-review fixes round 2, 2026-07-24 (Harri).** A second UI pass, this
time actually running the app rather than a headless screenshot diff, found
the first round's tab-text fix was itself wrong, plus three deeper bugs in
the phase 0-3 work:

1. **Tab text was upside-down and mirrored (read right-to-left).** The
   round-1 fix rotated the painter -90° before drawing each tab's label,
   reasoning "West tabs read top-to-bottom, so rotate -90°" — but Qt's own
   `CE_TabBarTabLabel` drawing (`qcommonstyle.cpp`) *already* rotates the
   painter -90° internally for a West-shaped tab before drawing the label,
   regardless of what our own `paintEvent()` does first. Our extra -90°
   compounded with the style's own -90° to -180° total — upside-down text
   that also reads backwards, exactly the reported symptom. The fix is
   `+90°`, not `-90°`: it cancels the style's own -90° back to identity,
   leaving the label genuinely horizontal. This is the standard, widely-used
   recipe for this exact problem (matches the commonly-cited
   Qt/PySide `+90°` solution for West-shaped `QTabBar`s), not a guess.
   The round-1 headless screenshot comparison didn't catch this because the
   offscreen platform in this environment renders only tofu boxes, not real
   glyphs (confirmed by grabbing a plain unstyled `QLabel` — same tofu) —
   which can distinguish "horizontal layout" from "vertical stack" but not
   "right-side up" from "upside-down and mirrored", since a 180° rotation of
   a horizontal run of boxes still looks like a horizontal run of boxes.
   Genuine visual confirmation of *this* class of bug needs a real windowed
   session or a font that actually rasterizes in this sandbox; the fix here
   instead rests on tracing the exact Qt source behavior that explains the
   reported symptom precisely (double rotation → 180°), which is a stronger
   argument than another screenshot would have been anyway.

2. **Hierarchical solver settings vanished when editing a trial's default
   config.** Two compounding bugs, both in code from phase 3:
   - `DefaultConfigDialog` never called `set_skeleton_ids()` on its embedded
     `TrackerConfigWidget` (deliberately, per its own docstring — a default
     isn't tied to one person's skeleton choice). But that left
     `discover_stage_groups()` with an empty skeleton list, so the
     Hierarchical solver tab could never discover *any* eligible group —
     the checkbox could be ticked but the table stayed permanently empty.
     Fixed by passing every skeleton in the session instead of none (a
     default isn't skeleton-specific, but "no skeletons" was never the
     right substitute for "not skeleton-specific" — "all skeletons in the
     session" is).
   - `TrackerConfigWidget.load_config_row()` never read back a config's
     existing `tracker_config_stages` rows — it always left the enable
     checkbox unchecked and the table empty, regardless of what was
     actually saved. Combined with `_sync_stage_overrides()` always
     rebuilding a config's stage rows from the table's *current* state
     (deleting whatever was there first), re-saving an already-hierarchical
     default without ever seeing its real stage selection silently deleted
     it. New `TrackerConfigWidget._load_stage_overrides()`, called from
     `load_config_row()`: checks the enable box iff any stage rows exist,
     populates the table via the existing `discover_stage_groups()` path,
     then overlays the saved per-group enabled state and override values.
3. **Run Tracker dialog always started from factory defaults, never the
   trial's own default.** `RunTrackerWidget` never loaded anything into its
   `TrackerConfigWidget` on trial selection — starting a run always based
   the snapshot on `BASELINE_CONFIG_ID` unless the user manually used
   "Load…", which only lists *named* configs (most trial defaults are
   unnamed auto-generated snapshots, so usually nothing to pick). New
   `_load_trial_default_config()`, called from `_on_trial_changed()`,
   resolves and loads the trial's effective default (same
   `resolve_default_tracker_config()` chain the trial/capture panels'
   "Default tracker config" row already uses) so the config tabs start
   already reflecting it — at which point they're the same always-editable
   tabs they've been since phase 3, so this alone also addressed Harri's
   related "can't adjust the configuration, only load a named one" report:
   the tabs were editable the whole time, there was just nothing useful
   loaded into them to start from.
4. **The "Based on X" status label didn't distinguish "loaded, untouched"
   from "loaded, then edited".** It always read `Based on "X"` regardless
   of whether the tab values still matched X. New dirty-tracking: a
   snapshot of `collect_overrides()` plus a serialized stage-table snapshot
   is captured whenever `loaded_config_id` is set (load or save), and every
   tuning field's change signal (wired generically via `findChildren()`
   rather than one-by-one, so new fields can't silently fall outside the
   check) triggers a label recompute. Unmodified: bare config name (or
   `(unnamed snapshot)`). Modified: name + ` (modified)`. Never loaded:
   unchanged `Based on factory defaults (unsaved)`.
5. **No way to promote a run's config to be its trial's default.** New
   `manage_config`-backed action, `_set_run_config_as_trial_default()`
   (shared, like `_save_run_config_dialog()`, between `_RunInfoPane` and
   `TrackingRunPanel`): a "Set as trial default" button next to "Save
   config…" repoints the trial's `default_tracker_config_id` to the exact
   config the viewed run used. Mirrored in `RunTrackerWidget` itself as a
   "Set as trial default" checkbox next to the Run button — checking it
   repoints the trial's default to the fresh snapshot `_start_tracking()`
   creates for that run (factored into `_maybe_set_trial_default()` so the
   checkbox's effect is unit-testable without spinning up the tracker
   thread).

Full run: `python/tests/app/test_run_tracker*.py` +
`test_content_panels_hierarchical.py` + `python/tests/db`: 301/303 (same 2
pre-existing, unrelated failures as every earlier phase). 8 new cases added
to `test_run_tracker_default_config.py` covering the status-label dirty
transition, stage-override load/save round-trip, `DefaultConfigDialog`
discovering stages from all session skeletons, and both the auto-loaded and
explicitly-set trial-default paths.

**Deferred: per-skeleton hierarchical eligibility.** Harri raised that a
future multi-person (or multi-object) run may mix skeletons with different
hierarchical-solver support, and one shared "enable hierarchical tracking"
checkbox on the config doesn't obviously generalize to that — likely wants
a per-skeleton eligibility concept alongside the current per-config one.
Noted as a real open question, not addressed here; needs its own design
discussion rather than a quick fix.

**Live-review fixes round 3, 2026-07-24 (Harri).** Live-running the actual
Run Tracker dialog (not just the trial/capture default-config dialog round
2 tested) surfaced one more real bug plus two layout complaints:

1. **Stage table stayed empty after selecting a hierarchical skeleton and
   pressing "Refresh stages" in the Run Tracker dialog.** Root cause: a
   person row's Skeleton `QComboBox` (`RunTrackerWidget._insert_person_row()`)
   had no `currentIndexChanged` connection at all — unlike the adjacent
   Person combo, which does refresh its detection-run combo on change.
   Changing a row's skeleton never reached
   `TrackerConfigWidget.set_skeleton_ids()`, so "Refresh stages" kept
   discovering groups for whichever skeleton the combo happened to default
   to at row-creation time (alphabetically first), not whatever the user
   actually selected. Round 2's `DefaultConfigDialog` fix (passing every
   session skeleton) didn't cover this path since the Run Tracker dialog's
   `TrackerConfigWidget` *is* meant to track a specific selection here.
   Fixed by wiring the skeleton combo's `currentIndexChanged` to the same
   `_update_run_btn()` already responsible for pushing
   `_current_skeleton_ids()` into the config widget. New regression test
   (`test_person_row_skeleton_change_updates_config_widget_skeleton_ids`)
   — verified it actually fails without the fix (reverted the one-line
   connect, watched it fail with the expected `['skel-hier'] !=
   ['skel-plain']`, restored it) before trusting it as a guard.
2. **Run Tracker dialog's default size was too small, hiding tab names.**
   `_HorizontalTabBar` (round 1) reports an honest, wider size hint for
   horizontal-label West tabs, but the dialog's old default
   (`640×420`, resized to `700×650`) was sized for the pre-round-1 narrow
   vertical-text tabs. Too little width made Qt fall back to scroll
   arrows, hiding most tab names — a second, layout-level way to reproduce
   "can't see all the tabs" distinct from round 1's rotation bug. Fixed by
   both raising the dialog's default/minimum size (`900×700` minimum,
   opens at `980×820`) and calling
   `self._config_tabs.setUsesScrollButtons(False)` so an undersized window
   forces the layout to honestly request more room instead of silently
   collapsing the tab strip.
3. **Pane order.** Reordered `RunTrackerWidget`'s root layout: People →
   configuration tabs (still the only stretching widget) → Run (output
   directory / binary path) → Set-as-trial-default checkbox → Run button →
   progress/results, matching the request to read top-to-bottom as "who,
   how, then go" rather than the previous People/Run-mechanics/config
   ordering.

Full run: `test_run_tracker*.py` + `test_content_panels_hierarchical.py` +
`python/tests/db`: 302/304 (same 2 pre-existing, unrelated failures).

**Phase 4 — done.** Schema (D1, D2): new `capture_persons` table
(`db/session_schema.sql`) plus a nullable `capture_person_id` column on
`sequence_persons` and `detection_track_assignments`, additive alongside
their existing free-text `person_name` (old rows keep working unchanged
with `capture_person_id IS NULL`). Session schema version bumped 38→39 via
new `_migrate_session_v38_to_v39()` (`python/posetrak/db/db.py`), following
the same generic-diff-and-backfill shape as the v37→v38 migration. New
`python/posetrak/db/manage_person.py`: `create_person()`,
`find_or_create_person()` (exact case-sensitive name match within a
capture, else creates — the quick-create path phase 5's "+ New person…"
picker will use), `list_persons()`, `get_person()`, `rename_person()`,
`set_default_skeleton()`, `delete_person()` (refuses with a clear
`ValueError` if any `sequence_persons`/`detection_track_assignments` row
still references the person, mirroring the project's existing
detection-run-immutability principle and the same "refuse instead of
crash on an FK violation" precedent as `finalise_to_db`). New tests:
`test_manage_person.py` (17 cases covering create/list/find-or-create/
rename/set-default-skeleton/delete, including both delete-refusal paths)
and a migration test in `test_posetrak_db.py` mirroring the v37→v38 one
(downgrade a fresh session to the pre-v39 shape, re-open, confirm the new
table/column exist and existing rows survive untouched). Full
`python/tests/db` run: 284/286 (same 2 pre-existing, unrelated failures).

**Phase 5 — done.** GUI (D3), built on phase 4's schema:

- **`CapturePanel` "Persons" section**: new `_CapturePersonsSection`
  (`content_panels.py`) -- a `QListWidget` of this capture's
  `capture_persons` (name + default skeleton, or "(no default skeleton)"),
  with Add.../Rename.../Default skeleton…/Remove acting on whichever row is
  selected. "Remove" surfaces `delete_person()`'s refusal (person still
  referenced by detection/tracking data) as a `QMessageBox.critical` rather
  than letting the `ValueError` propagate. Added to `CapturePanel` right
  after its existing default-tracker-config row.
- **`main.py`'s person picker**: rather than rewriting the existing
  free-text/editable combo + stitcher/conflict-resolution UI (complex,
  working, and unrelated to this feature), the combo is now *pre-populated*
  from the current capture's `capture_persons` on run selection
  (`_populate_known_persons()`), so reusing an existing person is picking
  their name from the dropdown instead of retyping it -- additive, same
  accumulate-don't-clear behaviour the combo already had. The actual
  identity link is resolved once, at the existing single choke point both
  `sequence_persons` and `detection_track_assignments` are written from:
  `finalise.py`'s `finalise_to_db()` now calls
  `manage_person.find_or_create_person()` per assigned name and writes the
  resulting id as `capture_person_id` on both tables (auto-creating a row
  for a name typed fresh rather than picked from the list). This delivers
  the design's actual goal -- identity defined once, reused across
  detection runs in a capture -- without touching the stitcher/frame-view
  machinery at all.
- **`RunTrackerWidget`'s people table data source switch**: `_on_trial_changed()`
  now resolves the selected trial's capture and calls
  `manage_person.list_persons()`; if any exist, `_insert_capture_person_rows()`
  builds one row per person *with observations in this trial*
  (`_detection_runs_for_capture_person()` matches by `capture_person_id`,
  falling back to an exact `person_name` match for pre-migration rows) --
  a checkbox (person name as its label) for whether to include them, a
  detection-run picker (only enabled when more than one exists for that
  person in this trial), and a skeleton combo pre-filled from the person's
  own `default_skeleton_id`. If the capture has *no* `capture_persons`
  defined yet, the table falls back to the original free-text-name
  discovery unchanged (row 0 mandatory, "Add person…" visible) -- existing
  captures that haven't adopted the person model keep working exactly as
  before. `_row_included()`/`_row_person_name()` abstract over which mode
  built a given row so `_current_skeleton_ids()`, `_update_run_btn()`, and
  `_start_tracking()`'s person-collection loop work unmodified in both.
- New test files: `test_run_tracker_person_model.py` (5 cases: capture-persons
  mode used when defined, legacy fallback when not, multi-detection-run
  picker enabling, a person with no observations in this trial being
  skipped rather than producing a dead row, unchecking a row excluding it
  from both skeleton discovery and the Run button's enablement),
  `test_capture_persons_section.py` (8 cases covering the list/add/rename/
  set-default-skeleton/remove actions, including the remove-refused path),
  plus two new cases in `test_finalise.py` (a fresh name creates and links
  a `capture_persons` row on both tables; the same name from a second
  detection run in the same capture reuses the existing row rather than
  creating a second one).

Full run: `test_run_tracker*.py` + `test_content_panels_hierarchical.py` +
`test_capture_persons_section.py` + `test_finalise.py` + `python/tests/db`
+ `python/tests/tracker`: 341/343 (same 2 pre-existing, unrelated
failures).

## The brief, in short

1. The tracker-configuration dialog (`RunTrackerDialog` /
   `RunTrackerWidget`, `python/app/pose/run_tracker.py`) is one long flat
   `QFormLayout` of ~25 spin boxes. It should be organized into grouped,
   documented pages.
2. The DB schema and CLI already require saving a config and referencing it
   by ID across runs, but the GUI hides this — every run silently creates a
   fresh, unnamed row. Users should be able to save/name/reuse configs, and
   to set a default config per session/capture/trial that a new run starts
   from (editing it forks a run-specific copy, optionally saved with a
   name later).
3. Related, not strictly config: "person" identity is currently scoped to a
   single detection run. Harri wants it promoted to capture (and/or trial)
   level, with the CLI supporting the same model.

## Investigation findings

Grounding facts, each independently verified against the code (not just
inferred from names), because several turned out to contradict what the
schema's own comments claim.

### `tracker_configs.parent_id` is provenance only, not read-time inheritance

`db/registry_schema.sql:82` has `tracker_configs.parent_id REFERENCES
tracker_configs(id)`, and the `tracker_config_stages` comment
(`db/registry_schema.sql:161-162`) says a NULL stage column "mirrors
tracker_configs' own parent_id inheritance." **This is misleading as
written** — `tracker_configs.parent_id` has no read-time fallback behavior
at all. `SessionReader::load_tracker_config()` (`src/db/session_reader.cpp`)
selects one row by `id` and applies `COALESCE`/hardcoded-constant fallbacks
for NULL columns; it never joins to or walks `parent_id`. The only place
`parent_id` does anything is `manage_config.edit_config()`
(`python/posetrak/db/manage_config.py:106-246`), which **flattens** at write
time: copies every value it knows about from the source row into a brand
new row, then sets the new row's `parent_id` to the source's `id`, purely
for audit/lineage. `tracker_config_stages`' NULL-inherits-from-base-row
behavior (used by the hierarchical solver) is real and does work at read
time (`src/db/session_reader.cpp:419-457`), but that's inheriting from a
stage row's *own* base `tracker_configs` row's same-named column — a
different, narrower mechanism than the comment's wording suggests. This
doc's new "default config per session/capture/trial" mechanism (below) is
therefore a genuinely new mechanism, not a rename or extension of
`parent_id` — nothing about `parent_id` can be reused for it.

### Prerequisite fix: `manage_config.edit_config()` silently drops 30+ columns

`tracker_configs` has grown to ~55 tuning columns across migrations v22-v37
(pose noise, relative/cross-pair/cross-person observations, adaptive
process noise, pose regularization, soft joint limits, near-limit damping,
NIS feedback, edited-keypoint noise — `db/registry_schema.sql:79-158`).
`manage_config.edit_config()` (`python/posetrak/db/manage_config.py:106-246`)
was written against the *original* ~20-column schema and never updated: its
`INSERT` column list stops at `velocity_measurement_noise_std`/`notes`. It
does `SELECT * FROM tracker_configs WHERE id = ?` (line 184), so it *has*
every column's value in `row`, but the hardcoded `INSERT` simply never
mentions the other ~35 columns — they silently become `NULL` in the new
row. `create_config_from_toml()` has the same staleness (only reads
`[tracking]`/`[tracking.ukf]`/`[tracking.initialization]`/`[processing]`
fields that existed at its original writing). Neither function touches
`tracker_config_stages` at all, so "editing" a hierarchical config would
also silently drop its per-stage overrides.

This bug is **currently dormant** — the GUI's own `_create_config()`
(`python/app/pose/run_tracker.py:1178-1301`) does its own from-scratch
`INSERT` covering every field it has a widget for (which is most of the
post-v22 columns, though it also omits `alpha`/`beta`/`kappa`/IK/init-std/
`min_cameras_for_init`/`near_limit_*` — those simply have no widget today
and rely on `SessionReader`'s hardcoded fallback constants; per Harri's
review this stays as-is, see "Decisions from Harri's review" below).
The GUI has never called `edit_config()` — it always creates a new row with
a hardcoded name `"ui-run"` and `parent_id = NULL` (line 1249). But this
doc's whole design leans on `edit_config()`-style copy-on-write being the
mechanism behind "load a default/named config, tweak it, get a run-specific
copy." Building that on the current `edit_config()` would silently strip
hard-won adaptive-noise/pose-reg/hierarchical tuning from any config that
has it, the moment a user edits and re-saves it through the new UI.

**Fix, once, generically, so it can't go stale again**: rewrite
`edit_config()` (and `create_config_from_toml()`) to copy the full row via
its own `sqlite3.Row.keys()` / cursor column names rather than a hardcoded
parameter list — `INSERT INTO tracker_configs SELECT <full column list
built from PRAGMA table_info or the row's own keys> FROM ...` with the
override values substituted in a dict before the insert, not named kwargs
per column. Add the equivalent for `tracker_config_stages`: copy every
existing stage row for the source config, unchanged, into rows for the new
config ID (per-stage editing can layer on top of that copy). This is the
same "centralize it once, so a schema change can't silently reintroduce the
bug" principle already applied to `SkeletonLayout` in the hierarchical-
solver work — a new migration adding column v38 should need zero changes to
`edit_config()` to stay correct.

This should land **before** anything else in this doc, as its own small,
independently-testable fix (new pytest cases: edit a config with, say,
`pose_reg_joint_names` set and a `tracker_config_stages` row, assert both
survive the edit).

### No vertical-tab precedent, but tooltips are already a strong convention

Confirmed no vertical-`QTabWidget` (or any multi-tab dialog) precedent
exists in `python/app/ui/` — the one `QTabWidget` in the codebase
(`python/app/setup/camera_registry.py:184`) is a plain top-tabbed one for
an unrelated dialog, not a pattern to match against. `QTabWidget.setTabPosition(QTabWidget.West)`
gets vertical tabs directly from Qt, no custom widget needed.

Conversely, **tooltips are already the norm, not a gap**: nearly every
widget added in `run_tracker.py` (lines 265-430) already carries a
multi-line explanatory `.setToolTip(...)`. The redesign should carry this
forward unchanged, not treat it as new work.

**Spin boxes are the opposite of a gap** — the brief's "no spin buttons
without good reason" is a genuinely new stylistic rule, not a codification
of existing practice. Current practice actively adds more of them (e.g.
`docs/roadmap/features/error-improvements/implementation-plan.md:52,99`
instructs adding `QDoubleSpinBox`es for new params). Adopting the brief's
rule means a real, if small, new reusable widget (see below), and touching
every existing field in the dialog, not just new ones.

### Person identity has no home above a single detection run

Confirmed: `sequence_persons` (`db/session_schema.sql:172-177`) is scoped to
one `pose_observation_sequences` row, which itself is generally tied to one
`detection_run_id` (`:157-168`). `detection_track_assignments` is scoped by
`detection_run_id` + `shot_video_id` + `track_id` (see its own `PRIMARY KEY`
in `db/session_schema.sql`). `tracking_run_persons`
is scoped per tracking run. **Nothing in the schema defines a person once
per capture or trial** — `run_tracker.py` itself documents this as a known
constraint at its own `_people_table` construction site. This fully
confirms the brief; see the person-model section below for the proposed
fix.

`CapturePanel` (`python/app/ui/content_panels.py:378`) and `TrialPanel`
(`:677`) both already exist as real, populated panels — good integration
points for this doc's additions. No `SessionPanel` exists (sessions have no
dedicated content panel today); given a session DB is already effectively
one physical recording session (the project's own "one DB file per mocap
session" convention), a session-level default config is lower-value than
capture/trial defaults and is treated as optional/deferred below.

`RunTrackerDialog` is already a real `QDialog` (invoked from `PersonPanel`
at `content_panels.py:6334-6343`, pre-seeded with one `sequence_id`) whose
internal widget (`RunTrackerWidget`) already has a trial combo and a
multi-row people table — so it already handles more than one person per
invocation once open, despite launching from a single person's panel today.
Restructuring its internals into tabs, and changing its launch point and
the people table's data source, are both compatible with this existing
architecture; neither requires inventing a new dialog shell.

## Proposed design

### A. Config data model

**A1. Prerequisite fix** (see above): make `edit_config()`/
`create_config_from_toml()` column-set-complete and `tracker_config_stages`-
aware, generically, so future migrations can't silently reintroduce this.

**A2. Named, reusable configs.** `tracker_configs.name` already exists and
`list_configs(name=...)` already supports exact-name lookup, but one small
schema addition is needed: **`is_named INTEGER NOT NULL DEFAULT 0`**.

**Resolved (Harri's review)**: the first draft proposed distinguishing
"browsable template" rows from "one-off run snapshot" rows by an
empty/auto-generated `name` string alone — Harri correctly flagged that
this isn't reliable: a string can't tell you *why* it's there, and nothing
stops an auto-generated name from coincidentally matching, or a user from
deliberately naming a run-snapshot something short and timestamp-shaped.
An explicit flag is unambiguous and cheap (`name` stays `NOT NULL`
throughout — no need to relax that constraint or invent a NULL-means-
unnamed convention). `is_named` is set `1` only by an explicit "Save as…"
action (A2/B3); every auto-generated per-run row (today's `"ui-run"`,
replaced by e.g. a `created_at`-derived label for on-screen display only)
keeps `is_named = 0`. The named-config picker (B3's "Load…") filters to
`is_named = 1` rows; the copy-on-write "editing a named config" flow above
also uses this flag to decide whether "Save" (same series) is offered at
all — it only makes sense when the row being edited from has `is_named = 1`.

**A3. Default config per session / capture / trial.** New nullable columns,
same cross-DB-reference-by-TEXT-id convention `tracking_runs.tracker_config_id`
already uses (no enforced FK across the registry/session DB boundary,
consistent with the project's existing precedent — not a new gap this doc
introduces):

```sql
ALTER TABLE captures ADD COLUMN default_tracker_config_id TEXT;
ALTER TABLE trials   ADD COLUMN default_tracker_config_id TEXT;
-- mocap_sessions: deferred, see above (no session panel exists to edit it from yet)
```

Resolution order when starting a new run for trial *T* in capture *C*:
`T.default_tracker_config_id` → `C.default_tracker_config_id` → a built-in
baseline config (a fixed, checked-in named config, e.g. `"factory
defaults"`, created once by the schema migration itself so the chain always
terminates in something real rather than an empty dialog).

**Editing a default is always copy-on-write, never in-place mutation** —
"Edit default config" at a given scope loads the resolved config (per the
chain above), and on save always produces a *new* `tracker_configs` row
(via the fixed `edit_config()`) and repoints only that scope's
`default_tracker_config_id` to it. This mirrors the project's existing
detection-run-immutability principle (never retroactively mutate a row
other, already-run tracking runs may reference) and means a capture-level
edit never silently changes what a trial-level override resolves to, and
vice versa.

**Resolved (Harri's review): editing a named config, and name collisions.**
Two related questions: what happens when you edit a config that already
has a user-given name, and what happens if "Save as…" is given a name that
already exists?

`tracker_configs.name` is **not** unique today — `list_configs()`'s own
docstring documents returning possibly-multiple rows for one name, ordered
by `created_at`. Rather than fight that with a new uniqueness constraint,
this doc treats a name as a **versioned series**, and leans on the
already-real `parent_id` lineage to make that concrete:

- Editing a named config produces an unnamed working copy (same as editing
  an unnamed one) — the copy does not silently inherit the source's name.
  The dialog's Summary tab shows "editing a copy of ‹name›" (per B1) so
  it's never ambiguous that the name hasn't been claimed yet by the new
  row.
- Two distinct actions once editing: **"Save"** re-uses the *same* name as
  the row being edited from, appending a new version to that name's series
  (`parent_id` → the row it was edited from). **"Save as…"** prompts for a
  name and always starts (or joins) a *different* series — including the
  case where the typed name happens to already exist elsewhere, which is
  allowed, not an error: it simply appends to that other name's series
  instead of forking a new one. (Whether that's the right call for a name
  typed by coincidence vs. deliberately reusing an existing one is a UX
  nuance — a simple mitigation is for "Save as…" to show existing matches
  as autocomplete/typeahead so accidental collisions are visible before
  confirming, not to block on them.)
- The named-config picker (A2's "Load…") resolves a chosen name to its
  **most recent** row (`ORDER BY created_at DESC LIMIT 1`) by default, with
  an expandable "older versions of this name" list driven by walking
  `parent_id` — turning `parent_id` into something the UI actually surfaces
  for named configs, not just inert audit trail.

**Starting a new tracking run**: pre-loads the resolved default for that
trial (per the chain above) into the dialog. If the user changes anything
before starting, the run uses a fresh copy-on-write row (parent pointing at
whichever row it started from) — matching the brief's "editing... creates
tracking run specific configuration" exactly. That per-run row can
optionally be named and saved afterward (A2) but doesn't have to be.

**Self-containment gap, noted but out of scope for this doc**: session DBs
are supposed to be portable without the registry DB present (see
`session_schema.sql`'s own "SELF-CONTAINMENT REQUIREMENT" header), and
`manage_config.copy_config_to_session()` already exists to mirror a
`tracker_configs` row into the session DB — but nothing calls it
automatically today, for either the CLI or GUI run path. This doc's new
default-config columns make the gap slightly more visible (a session's own
default config, not just a past run's, could go stale if the registry DB
is unavailable) but doesn't create it. Worth its own small fix (call
`copy_config_to_session()` wherever a `tracker_config_id` is first attached
to anything in the session DB) — flagged here, not designed further, since
it's orthogonal to what the brief asked for.

### B. Configuration dialog restructure

Keep `RunTrackerDialog` as the dialog shell (it already is one); restructure
`RunTrackerWidget`'s internals.

**B1. Vertical tabs** (`QTabWidget.setTabPosition(QTabWidget.West)`),
replacing the single `QScrollArea`/`QFormLayout`:

| Tab | Fields (from today's flat list / registry columns) |
|---|---|
| **Summary** | Read-only rollup: config name + source ("editing a copy of ‹name›" / "unsaved run-specific config" / "‹name› (unmodified)"), one line per other tab's non-default values — the same idea as `content_panels.py`'s existing `_cfg_text()` sidebar summary from the hierarchical-solver work, extended to the full config, not just the hierarchical part. |
| **UKF & process model** | `process_noise_std`, `process_noise_vel_std`, `velocity_half_life_s`. |
| **Observations & outliers** | `measurement_noise_std`, `pose_noise_std`, `outlier_threshold`, `use_relative_observations`, `relative_min_confidence`, `cross_pair_max_px`, `cross_pair_max_n`, `edited_kp_noise_std`. |
| **Adaptive process noise** | `process_noise_vel_gain_joint/root`, `..._ref_joint/root`, `process_noise_vel_joint_names`, `process_noise_vel_scopes` — today's proximal/distal scope widgets, unchanged in behavior. |
| **Pose regularization & joint limits** | `pose_reg_*`, `soft_limit_*`. |
| **NIS feedback** | `nis_feedback_scopes/window/threshold/max_multiplier`. |
| **Cross-person coupling** | `cross_person_max_world_mm/min_confidence/max_n`. |
| **Hierarchical solver stages** | Today's `_stage_table`, promoted to its own tab instead of appended below the flat form — otherwise unchanged. |

**Resolved (Harri's review): `alpha`/`beta`/`kappa`, IK/init-std,
`min_cameras_for_init`, and `near_limit_*` are left out of the GUI
entirely** — not just `alpha`/`beta`/`kappa`. They stay config-file/TOML-
only, with no widget and no tab; `SessionReader`'s existing hardcoded
fallbacks keep applying whenever a row leaves them NULL, unchanged from
today. This is why there's no "Initialization" tab above — every field it
would have held falls in this excluded bundle — and why "Pose
regularization & joint limits" above only has `pose_reg_*`/`soft_limit_*`,
not `near_limit_*`.

`tracker_fps` and `velocity_mode_camera_ids` stay where they conceptually
belong (`velocity_mode_camera_ids` under UKF/process model, `tracker_fps`
probably Summary-adjacent as a run-level setting) — exact placement is a
detail to settle during implementation, not a design fork.

**B2. Numeric field widget.** A small reusable `NumericLineEdit`
(`QLineEdit` + `QDoubleValidator`/`QIntValidator`, right-aligned) replacing
`_float_spin()`/raw `QSpinBox` uses, applied consistently across every tab
above — not just new fields. Small, genuinely-bounded integer counts (e.g.
`ik_max_iterations`, `cross_pair_max_n`) may reasonably stay spin boxes if
a tight, meaningful range exists; the brief says "usually," not "never," so
exceptions are fine when justified per-field, not by default.

**B3. Config picker + save/load**, visible above the tabs regardless of
active tab: current name/source label, "Load…" (opens `list_configs()`
results, named rows only), "Save as…" (prompts for a name, writes via the
fixed `edit_config()`), and a dirty-indicator once any field changes after
load. "Start Tracking" silently produces a copy-on-write row first if the
in-dialog state differs from the last loaded/saved row (A3).

### C. Trial / Capture panel integration

Add a "Default tracker config: ‹name› [Edit] [Change…]" row to both
`TrialPanel` and `CapturePanel` (both already exist and are populated
panels — this is additive, not a new panel type). "Edit" opens the same
restructured dialog (B), scoped so saving repoints *only* that panel's own
`default_tracker_config_id`, never the other level's. "Change…" opens the
named-config picker (A2) to repoint without editing.

Session-level default: deferred (no `SessionPanel` exists yet to host it;
add only if/when one is built for other reasons).

### D. Person model: promote identity to capture level

**D1. New table**, capture-scoped (not trial-scoped — trials within one
capture are near-certain to share the same physical performers, matching
Harri's own "usually are same" framing in the brief; trial-level override
of an *existing* capture person's skeleton already has a home,
`tracking_run_persons`, so a separate `trial_persons` table would just
duplicate that without adding capability):

```sql
CREATE TABLE IF NOT EXISTS capture_persons (
    id                 TEXT PRIMARY KEY,
    capture_id         TEXT NOT NULL REFERENCES captures(id),
    name               TEXT NOT NULL,
    default_skeleton_id TEXT,  -- references registry: skeletons(id); nullable until assigned
    notes              TEXT,
    created_at         TEXT NOT NULL
);
```

**D2. Migration path for existing per-detection-run identity.** Add a
nullable `capture_person_id TEXT REFERENCES capture_persons(id)` column to
`detection_track_assignments` and to `sequence_persons`, alongside (not
replacing) the existing free-text `person_name` — old rows keep working
unchanged (`capture_person_id` NULL), new assignments made through the
redesigned UI set both (the free-text name mirrors the linked
`capture_persons.name` for display/CSV-export compatibility, avoiding a
join everywhere `person_name` is read today).

**D3. UI changes.**
- `CapturePanel` gains a "Persons" section: list of `capture_persons` (name
  + default skeleton), add/edit/remove — the natural place given the panel
  already exists and already shows this capture's videos/sync/detection
  launcher.
- `python/app/pose/main.py`'s per-track "assign to…" action changes from
  free-text naming to picking an existing `capture_persons` row (plus
  "+ New person…", which creates one inline) for the capture the current
  detection run belongs to.
- `RunTrackerDialog`'s people table changes its data source from scanning
  detection-run-scoped assignments to listing the trial's capture's
  `capture_persons`, each row showing (checkbox to include, default
  skeleton pre-filled with an override field, and — only when more than
  one detection run/sequence exists for this trial with observations for
  that person — a picker for which one to use). This directly delivers the
  brief's "define persons once, then at tracking-run time select which to
  track and optionally override skeleton."

### E. CLI: same model

`cli/track.cpp`'s `--person <sequence> <skeleton> <tracker_config>
<person_id>` 4-tuple and `python/posetrak/cli/track.py`'s wrapper both stay
as the low-level, fully-explicit mechanism (nothing forces a rewrite of the
C++ CLI's own argument contract). The **Python** CLI gains a higher-level
mode once D lands: resolve a capture's `capture_persons` by name to the
4-tuple automatically (auto-selecting the sequence when only one detection
run has observations for that person in the given trial, erroring
descriptively when ambiguous), so a scripted run looks like `posetrak-track
--trial <id> --persons Alice,Bob` instead of requiring the caller to already
know each person's `sequence_id`/`tracker_config_id`. This is additive to
the existing 4-tuple flag, not a replacement — existing scripts/tests
(`tests/regress.toml`, CI) keep working unchanged.

## Phase plan

| Phase | Work | Depends on |
|---|---|---|
| 0 — **done** | Fix `manage_config.edit_config()`/`create_config_from_toml()` to be column-set-complete and `tracker_config_stages`-aware, generically (via row-keys, not a hardcoded parameter list). New pytest coverage for "edit preserves every column, including stage overrides." | — |
| 1 — **done** | Schema: `captures.default_tracker_config_id`, `trials.default_tracker_config_id`, `tracker_configs.is_named`, a checked-in baseline `tracker_configs` row (`is_named=1`) the chain terminates in. | 0 |
| 2 — **done** | GUI: restructure `RunTrackerWidget` into vertical tabs (B1), add the numeric-field widget (B2) across all tabs, add the config picker/save-as (B3). No schema change beyond phase 1. | 0, 1 |
| 3 — **done** | GUI: "Default tracker config" row + Edit/Change on `TrialPanel`/`CapturePanel` (C). | 1, 2 |
| 4 — **done** | Schema: `capture_persons` + nullable `capture_person_id` on `detection_track_assignments`/`sequence_persons` (D1, D2). | — (independent of 0-3) |
| 5 — **done** | GUI: `CapturePanel` persons section, `main.py` assignment picker, `RunTrackerDialog` people-table data-source switch (D3). | 4, and ideally 2 (so the redesigned dialog isn't touched twice) |
| 6 | Python CLI: person-by-name resolution (E). | 4 |

Phases 0-3 (config-only) are independently shippable and deliver the bulk of
the brief's first, more urgent complaint (dialog complexity + hidden
save/reuse) without waiting on the person-model work, which the brief
itself frames as more speculative ("eventually"). Phase 4 (schema) has no
dependency on 0-3 and could start in parallel if useful.

## Decisions from Harri's review

- **`alpha`/`beta`/`kappa`, IK/init-std, `min_cameras_for_init`,
  `near_limit_*`: leave out of the GUI for now.** Not just `alpha`/`beta`/
  `kappa` — the whole bundle. These stay config-file/TOML-only, with no
  widget and no dedicated tab; `SessionReader`'s existing hardcoded
  fallbacks keep applying whenever a `tracker_configs` row leaves them
  NULL, unchanged from today. Already reflected in B1 above: there is no
  "Initialization" tab (every field it would have held falls in this
  bundle), and "Pose regularization & joint limits" only has
  `pose_reg_*`/`soft_limit_*` (which already have widgets today), not
  `near_limit_*`.
- **Session-level default config: defer.** No `SessionPanel` work in this
  proposal; revisit only if one gets built for other reasons.
- **Auto-name vs. explicit flag: explicit flag, `is_named`.** Resolved
  above (A2) — a plain string can't reliably signal "the user chose this
  name on purpose" vs. "this is an auto-generated label," and nothing
  prevents an auto-generated value from coincidentally colliding with a
  real name. `is_named INTEGER NOT NULL DEFAULT 0` avoids both problems
  without touching `name`'s existing `NOT NULL` constraint.
- **`copy_config_to_session()` auto-invocation: agreed real gap, worth
  fixing** — but still a separate, small follow-up outside this proposal's
  phase plan (no phase here depends on it, and it's independent of
  everything in phases 0-6). Tracked here as a confirmed TODO, not
  designed further.

## Explicitly out of scope for this proposal

- Rewriting the C++ CLI's own `--person` 4-tuple contract — the Python CLI
  gains a higher-level mode on top of it (E), the C++ contract is
  unchanged.
- The session-level default config tier (deferred, see above).
- `copy_config_to_session()` auto-invocation / the broader session-DB
  self-containment gap for `tracker_config_id`/`skeleton_id` references
  (agreed real gap, tracked separately, see above).
- Any change to how `tracker_config_stages`' own NULL-inherits-from-base-row
  mechanism works — that part of the existing design is correct and
  untouched by this doc.
- GUI exposure of `alpha`/`beta`/`kappa`, IK/init-std, `min_cameras_for_init`,
  `near_limit_*` (deferred, see above).
