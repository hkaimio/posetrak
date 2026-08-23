```toml
name = "Release Packaging (Windows/Linux Installer)"
status = "in_progress"
progress_pct = 75
description = """
Produce an installable release artifact (Windows installer, Linux AppImage/tarball) that doesn't \
require a compiler or manual `uv sync` -- a thin bootstrapper (uv binary + pre-built C++ tracker + \
pinned lockfile) rather than a fully offline fat bundle, keeping one dependency-resolution path \
for both a release install and a dev checkout.
"""
categories = ["release", "packaging", "build"]
target_release = "TBD"
last_updated = 2026-08-23
```

# Release Packaging — Implementation Status

See [packaging-design.md](packaging-design.md) for the full motivating
problem, current-state trace, target vision, and recommended approach.
See [installer-prototype-plan.md](installer-prototype-plan.md) for the
near-term, Windows/CPU-only prototype-then-small-group-test plan this
is actually starting with. See
[code-signing-plan.md](code-signing-plan.md) for the Windows
code-signing sub-plan — deliberately deferred until there's real
evidence of interest in Posetrak beyond today's use.

## Current state

**2026-08-23: the full Phase 1+2 core loop is validated end to end --
install → first launch → session setup → calibration → detection →
tracking → BVH export, all via the unsigned installer on a clean
Windows Sandbox. Six real bugs found and fixed along the way (missing
skeleton seed data; a Windows TLS cert-chasing gap; false-positive CUDA
detection; a permanently-stuck corrupt checkpoint cache; plus a
still-unconfirmed skeleton-data-reversion possibly related to WAL mode
over Sandbox's shared folder). Also added: a CPU-fallback warning in
the detection UI, and an installer checkbox for optional GPU
segmentation support.**

- Built the optimized C++ tracker (`meson setup --reconfigure optbuild
  --buildtype=release`, `meson compile`) and assembled a bootstrap
  folder (`uv.exe`, `tracker/` with the binary + its 2 runtime DLLs, an
  `app/` snapshot of `pyproject.toml`/`uv.lock`/`README.md`/
  `.python-version`/`python/` extracted via `git archive HEAD` — tracked
  files only, no dev-machine build cruft) plus a `launch.ps1` and a
  `posetrak-bootstrap-test.wsb` Windows Sandbox config, at
  `D:\mocap\posetrak-bootstrap-proto\` (outside the repo, disposable per
  the plan).
- `uv sync --project app` succeeds standalone: resolves and installs 38
  base packages (PySide6, OpenCV, numpy, onnxruntime-gpu, etc.) plus
  `posetrak` itself in editable mode from the snapshot, with no
  dependency on the main repo's own `.venv`. `uv run --project app
  posetrak --help` and an `app.ui.main`/`PySide6.QtWidgets` import both
  work from the synced environment.
- **Concrete size data**: shipped bootstrap payload (`uv.exe` + tracker
  + source snapshot, everything that would go in an installer) is
  **~60MB**. What `uv sync` downloads at first run (the resulting
  `.venv`) is **~1.5GB** — the real, now-measured shape of the "first
  launch needs internet and a few minutes" tradeoff
  packaging-design.md's recommendation accepts.
- Validated the tracker-binary "installed location" mechanism needs no
  new code at all: `launch.ps1` copies `tracker/*` to
  `%USERPROFILE%\.posetrak\`, which
  `posetrak.tracker.runner.default_binary_path()` already checks first.
  Confirmed live: copied the binary there and `default_binary_path()`
  resolved to it.
- **Found and fixed a real bug in the process**: `runner.py`'s
  dev-build fallback path was still `optbuild/cli/...`, stale since the
  C++ source tree moved under `cpp/` (commit `dd6afae`); every existing
  test mocks `default_binary_path()` out entirely so nothing caught it.
  Fixed and regression-tested (commit `7ee7a65`).
- **First real Sandbox finding, before the small-group phase even
  started**: Harri opened the Sandbox and it "does not recognize how to
  run .ps1" — a stock Windows install associates `.ps1` with Notepad on
  double-click rather than running it, and even "Run with PowerShell"
  hits the default execution policy (`Restricted`), which refuses to
  run *any* unsigned script with no override. Added `launch.bat`
  (double-click-safe, calls `powershell.exe -ExecutionPolicy Bypass
  -File launch.ps1` — bypasses only for that one invocation, not a
  system-wide change) as the actual entry point; `launch.ps1` is now an
  implementation detail. `README.txt` updated to match. This is exactly
  the class of thing the plan's Phase 3 (small-group test) exists to
  catch — caught even earlier, during the very first run.
- **Second real finding, same run**: with `launch.bat` in place, `uv
  sync` then failed inside the Sandbox with "did not find executable at
  ...WindowsApps\...python.exe" — `HarriKaimio`'s own path, on a
  Sandbox user account (`WDAGUtilityAccount`) that doesn't have that
  interpreter at all. Cause: Windows Sandbox mounts a mapped folder
  *live* (not a copy), and the dev-machine sanity check above had left
  its own `app\.venv` sitting in the shared bootstrap folder --
  `uv`-created venvs aren't portable between machines (they point back
  at the interpreter that created them via `pyvenv.cfg`), so the
  Sandbox inherited a broken one instead of creating its own fresh
  venv. Removed `app\.venv` (plus `__pycache__`/`posetrak.egg-info`
  cruft from the same sanity check) from the shared folder.
  **Methodology fix for future iterations**: never re-run `uv
  sync`/`uv run` against the live bootstrap folder itself once it's
  meant for a Sandbox test -- test in a disposable copy instead, so the
  shared folder stays pristine.
- **Third real finding, next run**: with the stale venv cleaned up, the
  Sandbox got past `uv sync` and into `posetrak-ui` itself, then failed
  with `FileNotFoundError` for
  `...\posetrak-bootstrap-proto\app\db\registry_schema.sql`. Root
  cause, independent of the prototype: `db.py`'s `_DB_DIR =
  Path(__file__).parents[3] / "db"` was a hardcoded, fixed-depth walk
  from `db.py`'s own file location, assuming a full git-checkout layout
  (a `db/` directory at the repo root, sibling to `python/`). That
  layout doesn't survive being packaged as just the `python/` tree —
  which is exactly what the bootstrap folder's `git archive HEAD --
  ... python` snapshot produces, and what a real wheel/pip install
  would also produce. Fixed by moving `db/` into
  `python/posetrak/db/sql/` (inside the package), adding an
  `__init__.py`, and resolving it at runtime via
  `importlib.resources.files("posetrak.db.sql")` — the same mechanism
  already used for `posetrak.data.skeletons`'s bundled YAML files.
  Declared the new package's data files in pyproject.toml's
  `[tool.setuptools.package-data]`. Committed (`2fbfc4f`); verified
  both via the full `python/tests/db`/`python/tests/cli` suite (no
  regressions) and by re-running `uv sync` + a real `create_registry()`
  / `create_session()` call against a disposable copy of the refreshed
  bootstrap `app/` snapshot.
- **Fourth real finding, next run**: with the db-packaging bug fixed,
  the same still-running Sandbox session got past `uv sync` into
  `posetrak-ui` itself, then failed with `ValueError: registry database
  schema version mismatch: expected 8, got 0`. Root cause: the *first*
  (pre-fix) launch attempt in that session had already called
  `create_registry()` against `~/.posetrak/registry.db`; `sqlite3.
  connect()` creates the file on disk immediately, so when schema
  application then failed (the bug above), it left an empty,
  schema-version-0 file behind at that path. `open_or_create_registry()`
  only checks `path.exists()`, so the retry (with the fix in place)
  found that leftover file and tried to *open* it as an existing
  registry instead of creating a fresh one -- surfacing a confusing
  version-mismatch error unrelated to the actual, already-fixed bug.
  This is a real defect independent of the prototype: any interrupted
  first-run (disk full, permission error, power loss) partway through
  registry/session creation would leave the same wreckage behind in a
  real install. Fixed by making `create_registry()`/`create_session()`
  roll back (delete the partial file + WAL/SHM sidecars) on any failure
  during schema application or seeding, so a retry creates cleanly
  instead of tripping over a broken leftover (commit `990cc19`).
  Verified via regression tests plus an end-to-end repro against a
  disposable copy of the refreshed bootstrap snapshot: a simulated
  schema-application failure now leaves no file behind, and the retry
  succeeds with the correct schema version.
- **Note for the current Sandbox session specifically**: this fix
  doesn't retroactively un-corrupt a `registry.db` already wrecked by
  an earlier attempt *within the same still-running Sandbox instance*
  -- delete `%USERPROFILE%\.posetrak\registry.db` (and any
  `-wal`/`-shm` siblings) inside the Sandbox, or just restart the
  Sandbox VM, before the next retest.
- **First fully successful run**: after the registry cleanup, the app
  opened. Harri confirmed the basic UI works from the bootstrap folder:
  created a new session DB, opened Manage Cameras (correctly empty) and
  Manage Skeletons (correctly shows the bundled default male/female
  skeletons), and opened the new-capture wizard. This validates Phase
  1's core assumption end to end -- the thin-bootstrap shape genuinely
  works on a clean machine with no Python/compiler/uv preinstalled.
  Not tested yet: the detection/tracking pipeline itself (no sample
  video was available inside the Sandbox to drive the capture wizard
  further).
- **Known open risk, not yet hit**: `pyproject.toml` pins
  `onnxruntime-gpu` unconditionally -- there's no CPU-only dependency
  variant yet, even though this prototype's own scope (above) says
  "CPU-only first". Untested whether `onnxruntime-gpu` degrades cleanly
  to its CPU execution provider on a GPU-less Sandbox, or errors/warns
  in a way that looks broken to a tester, once video is actually fed
  through detection. Worth watching for specifically once a sample clip
  is available to test the pipeline end to end.
- **Phase 1 called done; moved to Phase 2** (Harri's call, 2026-08-23):
  UI-level validation is sufficient for now rather than chasing the
  detection/tracking pipeline through the Sandbox next.
- **Phase 2 started**: wrote `packaging/windows/posetrak.iss` (Inno
  Setup script, checked into the repo -- unlike the disposable bootstrap
  folder, this is real release infrastructure). Per-user install
  (`PrivilegesRequired=lowest`, `%LOCALAPPDATA%\Programs\Posetrak`, no
  admin/UAC prompt), Start Menu shortcut + optional desktop icon +
  uninstaller, and an `InfoBeforeFile` page
  (`packaging/windows/smartscreen-notice.txt`) covering Phase 2's
  "documenting the warning" requirement. `SourceDir` is passed via
  `/DSourceDir=...` at build time so a future CI workflow can point it
  at whatever it assembles.
  - Installed Inno Setup 6 via `winget` (wasn't on the dev machine).
  - **Found while compiling**: the live-mapped bootstrap-proto folder
    had picked up `app/.venv`, `__pycache__`, and a `posetrak-ui.log`
    written back by the Sandbox's own successful run -- the same
    live-mount problem as the earlier host-side contamination, just
    from the other direction. Fixed by having the `.iss`'s `[Files]`
    entry `Excludes` those defensively, so the installer can't bundle
    them regardless of what's sitting in the staging folder at build
    time.
  - First compile also failed outright ("process cannot access the
    file") because the Windows Sandbox VM was still running and holding
    a lock on the shared folder; resolved once Harri closed it.
  - Compiled cleanly: `posetrak-setup-0.1.0-proto1.exe`, ~19MB.
  - Set up a second, separate disposable Sandbox config
    (`D:\mocap\posetrak-installer-test\`, read-only mapped folder --
    the installer writes to the Sandbox's own local disk, not back to
    the host, so this one avoids the live-mount contamination class of
    problem entirely) for testing the compiled installer itself.
- **Installer Sandbox test, round 1**: installation itself worked
  cleanly (SmartScreen notice, license page, Start Menu shortcut, first
  launch). Harri then ran the actual tutorial materials
  (`docs/user-guide/tutorial1.md`'s walkthrough, using a disposable copy
  of his local `D:\mocap\tutorial1-template` test fixture mapped in
  read-write alongside the read-only installer folder) through as far
  as the detection step, surfacing two more real findings:
  - **Default skeletons missing in the new-capture wizard's persons
    page.** Root cause: data, not code -- `tutorial1-template.db` had 0
    rows in `skeletons` (confirmed directly: `PRAGMA user_version` = 43,
    correct, but the table was simply never seeded). The wizard's
    "optional default skeleton, `(none)` is a valid choice" design
    (`page_persons.py`) is working as intended; this template file
    predates -- or was created via a path that skipped -- the
    established `create_session()` + `seed_bundled_defaults()`
    convention for sessions a person will actually use. Fixed directly:
    seeded both the original template and its disposable Sandbox-test
    copy with `seed_default_skeletons()` (idempotent, safe to re-run).
  - **rtmlib's model-checkpoint download failed with
    `CERTIFICATE_VERIFY_FAILED: unable to get local issuer
    certificate`.** Root cause: Python's `ssl` module doesn't get the
    automatic AIA (Authority Information Access) chasing that
    browsers/WinINet-based Windows apps get for free from the native
    certificate store, so a machine that's never made an HTTPS
    connection needing a given CA's intermediate cert before -- a fresh
    install or a Sandbox run, not Sandbox-specific -- can fail exactly
    this way even though the same URL opens fine in a browser on the
    same machine. Fixed by pointing Python's default SSL context at
    certifi's root CA bundle via `SSL_CERT_FILE`, set at import time in
    both rtmlib-backed detection backend modules (`backends_rtmdet.py`,
    `backends_rtmpose.py`); added `certifi` as an explicit direct
    dependency (commit `efdf95e`). Installer rebuilt with the fix;
    bootstrap-proto's `app/` snapshot refreshed too.
- **Installer Sandbox test, round 2** (resumed from the same detection
  step, both round-1 fixes in place): the YOLOX detector's checkpoint
  downloaded and loaded fine, but printed a scary-looking onnxruntime
  stderr block ("FAIL ... cublasLt64_12.dll ... missing", "Failed to
  create CUDAExecutionProvider") before falling back to CPU
  internally -- non-fatal, but alarming. Shortly after, the ViTPose
  checkpoint download stopped mid-transfer (20% of 1.15GB) and
  onnxruntime then failed loading it with `InvalidProtobuf`. Two more
  real, general bugs (neither Sandbox-specific), both fixed in commit
  `00f8036`:
  - **False-positive CUDA detection.** `_auto_device()`'s torch-absent
    fallback trusted `onnxruntime.get_available_providers()`, which only
    reflects what `onnxruntime-gpu` (a core dependency) was *compiled*
    with, not whether CUDA is actually installed -- any CPU-only
    machine without `torch` (the installer prototype's base install has
    none; `torch` is only in the optional `segmentation` group) reports
    "cuda" as available regardless and then fails loudly trying to use
    it. Fixed by defaulting to CPU when torch is absent, since there's
    no other reliable signal; `device="cuda"` is still available as an
    explicit override.
  - **Corrupt/truncated rtmlib checkpoint cache, permanently stuck.**
    rtmlib's `download_checkpoint()` treats "a file already exists at
    the cache path" as "already downloaded", and its download never
    verifies the byte count against `Content-Length` before atomically
    renaming into place -- a dropped connection (this specific
    Sandbox's virtualized network, but not Sandbox-specific in general)
    silently produces a truncated-but-"complete" cached file, and every
    subsequent attempt then fails the exact same way forever with no
    recovery path visible to the user. This is structurally the same
    "leftover wreckage from a failed attempt masks the retry" pattern
    as the registry-database bug from earlier in this same prototype
    round. Fixed by adding a self-heal retry
    (`construct_with_corrupt_checkpoint_retry()` in `backends.py`): on
    any failure constructing a detector/estimator, delete the specific
    cached file(s) for that checkpoint's URL and retry once.
  Installer rebuilt with both fixes; bootstrap-proto's `app/` snapshot
  refreshed too.
- **Installer Sandbox test, round 3**: both models now load cleanly
  with all four detection-step fixes in place -- no crash, no scary
  onnxruntime stderr. Detection itself is slow, as expected: the
  Sandbox has no GPU passthrough, so this is CPU inference on shared
  virtualized cores, not a bug (real CPU-only users on real hardware
  will be faster than this, but still slower than GPU -- the CUDA
  prerequisites TODO above covers what's needed for GPU acceleration).
- **One more finding, same session**: at the "run tracker" step, the
  default skeletons were "missing" again -- the `skeletons` table in
  `posetrak-tutorial-test\tutorial1-template.db` had reverted to 0 rows
  despite being seeded earlier in this same testing round. Root cause
  not confirmed; no code path in the app deletes skeleton rows in bulk
  (checked), so the leading theory is that Windows Sandbox's shared
  folder is effectively a network filesystem, and SQLite's WAL mode
  (used unconditionally by this app's `_connect()`) is
  [explicitly documented as unreliable over network filesystems](https://www.sqlite.org/wal.html)
  -- if so this is specific to testing via a live-mapped Sandbox
  folder, not something a real user (session DB on local disk) would
  hit. Reseeded the test file directly (verified via a fresh reopen)
  to unblock; flagged to Harri to watch for recurrence as a signal
  either way. Not yet actually reproduced/confirmed.
- **Installer Sandbox test, round 4 -- full tutorial walkthrough
  succeeds end to end.** With skeletons reseeded, Harri ran the tracker
  and exported BVH successfully. This is the full Phase 1+2 core loop
  validated: install (unsigned, SmartScreen-warned) → first launch (`uv
  sync` provisions Python + deps from scratch) → session/capture setup
  → extrinsics calibration → detection (YOLOX + RTMPose, CPU fallback,
  slow but correct) → UKF tracking → BVH export. Also confirmed the new
  "Install GPU segmentation support" installer checkbox is visible and
  selectable.
- **Not yet done**: measuring first-launch `uv sync` timing; validating
  `uv`'s from-scratch Python provisioning is truly cache-free;
  confirming (or ruling out) the WAL/Sandbox-shared-folder theory
  above; verifying the CUDA-prerequisites TODO on real GPU hardware;
  the in-app "install segmentation extras later" action; the small-group
  test (Phase 3); CI automation; Linux.

## Known issues / open questions

See packaging-design.md's "Open questions" section: code signing
(deferred), auto-update, GPU vendor scope, Linux distro coverage,
whether a fully-offline variant is ever needed, and where the pinned
lockfile snapshot for a given release should live. Plus, newly surfaced
by the prototype: `uv.exe` itself is ~44MB, larger than
packaging-design.md's "a few MB" estimate — still small next to the
~1.5GB dependency download, but worth correcting in that doc.

**TODO: document end-user CUDA prerequisites.** Harri asked (2026-08-23)
what a user needs installed for CUDA acceleration to work, specifically
so this doesn't get forgotten before the small-group test (Phase 3) --
testers will hit this. Best understanding so far, from reading the code
and `uv.lock`'s platform markers (not yet verified on real GPU
hardware):

1. An NVIDIA GPU with a reasonably current driver (needs to support
   CUDA 12.x).
2. The optional `segmentation` dependency group installed (`uv sync
   --group segmentation`) -- **not** part of the default install. This
   pulls `torch` from the pinned `pytorch-cu126` index, and on Windows
   torch's wheel bundles its own CUDA 12.6 + cuDNN runtime DLLs inside
   `torch/lib/`.
3. Nothing else, as far as the code implies: `backends_rtmdet.py`/
   `backends_rtmpose.py` register torch's `lib/` directory onto the DLL
   search path before onnxruntime-gpu is imported, specifically so
   onnxruntime-gpu's CUDA execution provider finds `cublasLt64_12.dll`
   etc. from torch's bundled copies -- no separate system-wide NVIDIA
   CUDA Toolkit/cuDNN install appears to be required. (This is also why
   `onnxruntime-gpu` is pinned `<1.26`: needs CUDA 12.x, and 1.26+ wants
   CUDA 13, which wouldn't match torch's cu126 build.)
4. A current Microsoft Visual C++ Redistributable (onnxruntime's own
   stated requirement -- visible in the "Please install all
   dependencies... latest MSVC runtime" text of the CUDA-unavailable
   warning fixed in the entry above).

Confirmed via `uv.lock`: the `nvidia-*-cu12` PyPI packages (cudnn,
cublas, etc.) are manylinux-only wheels with no Windows build at all --
on Windows, torch really is the sole source of these DLLs, there's no
alternate system-package path to check. **Not yet confirmed**: whether
`uv sync --group segmentation` alone is actually sufficient with no
other install step, on a machine with a real GPU. Verify before writing
this up in real user-facing docs (docs/setup.md and/or a dedicated GPU
prerequisites section).

**Installing `segmentation` extras (torch/Cutie) after the
thin-bootstrap install.** Harri asked (2026-08-23): the installed app
has no `torch`, so segmentation-based detection just doesn't work.
Wants both of the two ideas below eventually; (1) is done as the quick
starting point, (2) is still open:
1. **Done** (commit `7397e05`): an installer-time "Install GPU
   segmentation support" checkbox (unchecked by default), which runs
   `uv.exe sync --group segmentation --project app` as a post-install
   step before the first-launch entry.
2. **Still open**: a more discoverable in-app "install additional
   components later" action (Settings/Tools menu) for anyone who didn't
   check the box at install time, or wants it added after the fact.
   Would shell out to the same `uv sync --group segmentation` command
   in the background, reusing the existing `job_runner.py` pattern
   already used for other long-running operations. Needs real design
   before implementing: where the menu item lives, how failure/retry is
   surfaced, resolving `uv.exe`'s path from an installed layout vs. a
   dev checkout.
