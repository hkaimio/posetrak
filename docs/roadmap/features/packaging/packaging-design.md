# Packaging a real release artifact — design sketch

> **Status (2026-08-23)**: Proposal only, nothing implemented. Written up
> after Harri asked how to produce an installable Windows/Linux release,
> preferably an installer or portable zip, easier than today's
> clone-and-build developer setup.

## Motivating problem

Today, "install Posetrak" means `docs/setup.md`'s full developer path:
clone the repo, install a C++ toolchain (native MSVC or a conda env for
Pinocchio/Boost headers) to build the tracker, install `uv`, and
`uv sync` the Python side. That's a reasonable bar for a contributor: it's
a bad bar for the first release aimed at actual users, who shouldn't need
a compiler at all.

There's no release workflow of any kind today — `.github/workflows/`
only has `docs.yml` (the site build). This is genuinely greenfield.

## Current state, traced concretely

**What's already right and shouldn't change:**

- Model weights (YOLOX/RTMPose via rtmlib, Cutie's segmentation model)
  already download lazily on first use, not bundled. This is the
  correct pattern for a release artifact too — no reason to make the
  installer bigger to pre-fetch something most sessions won't need on
  day one.
- The project already standardizes dependency resolution through `uv`
  and a committed lockfile (`uv.lock`). Whatever install path a release
  package uses, reusing this instead of a parallel packaging-specific
  dependency list keeps exactly one thing to update when a dependency
  changes.
- `[dependency-groups]` in `pyproject.toml` already separates optional,
  heavy dependencies (`segmentation`: `torch`/`torchvision`/etc.) from
  the base install — the right shape for a CPU/GPU packaging split, see
  below.

**What actually drives install size, concretely:**

- `torch`/`torchvision` (segmentation group) are the single biggest
  dependency when installed, and are already correctly gated as
  optional.
- `onnxruntime-gpu` is in `pyproject.toml`'s **base** `dependencies`,
  not gated at all — every install pulls a wheel bundling CUDA/cuDNN
  runtime libraries (order-of-magnitude ~200-300MB), even a CPU-only
  machine that will fall back to `CPUExecutionProvider` anyway per
  `docs/setup.md`'s own GPU-acceleration section. This should split
  into a plain `onnxruntime` in the base dependencies plus
  `onnxruntime-gpu` only in a GPU-enabled variant, independent of the
  rest of this design — worth doing regardless of how packaging turns
  out.
- PySide6 + OpenCV + numpy/scipy/pandas/matplotlib make up the rest of
  the base install — normal desktop-app weight, nothing to trim there.
- The C++ tracker itself, once built, is small — no heavy runtime
  dependencies once statically linked (`default_library=static` for
  the native MSVC path; the existing MinGW cross-compile path already
  produces a runnable `.exe` plus 5 small runtime DLLs, see
  `docs/setup.md`'s Windows section).

## Target vision

A user downloads one file (installer on Windows, AppImage or tarball on
Linux), runs it, and has a working `posetrak-ui` shortcut — no compiler,
no manually-run `uv sync`, no path-finding for Pinocchio/Boost headers.
First launch may need internet access and a few minutes to finish
setting up (see tradeoff below); after that it behaves like a normal
installed desktop app.

## Recommended approach: thin bootstrapper over a fat bundle

Two fundamentally different shapes were worth weighing:

1. **Fat bundle** — a PyInstaller/Nuitka-built single executable, or an
   installer that pre-downloads every dependency (CPU or GPU variant)
   at build time and ships them inside the installer/zip. Fully
   self-contained; works offline immediately after install. Cost: a
   multi-GB artifact, and a packaging pipeline that has to be
   separately maintained and re-validated every time a dependency
   changes (PyInstaller in particular has a long history of needing
   per-package hooks for anything with native extensions — PySide6,
   OpenCV, onnxruntime, torch all qualify).
2. **Thin bootstrapper** — the installer/zip contains only: the `uv`
   binary (a few MB, no install step of its own), the pre-built C++
   tracker binary, and a pinned `pyproject.toml`/`uv.lock` snapshot for
   that release version. A launcher shortcut runs `uv sync` against
   that lockfile on first launch, then `uv run posetrak-ui`.

**Recommendation: (2), the thin bootstrapper.** It means "install the
release" and "set up a dev checkout" resolve dependencies through the
exact same mechanism — one lockfile drives both, so there's no separate
packaging-specific dependency logic to keep in sync as `pyproject.toml`
changes. The real cost is honest and worth stating plainly: first launch
needs internet access and takes a few minutes to become usable, rather
than working immediately offline. Given the project already accepts
this tradeoff for model weights, extending it to the base Python
dependencies is consistent, not a new compromise — and it's a
reasonable, common expectation for a technical desktop tool's first
launch. If real user demand for a fully-offline installer shows up
later, a "fat" variant can be built from the same lockfile as a second
release artifact without redesigning anything here.

### Per-OS artifact

- **Windows**: an [Inno Setup](https://jrsoftware.org/isinfo.php)
  installer (free, scriptable, standard for this kind of thing) wrapping
  the bootstrap contents above. Either ask CPU-vs-GPU as an install-time
  choice, or detect an NVIDIA GPU and pick the lockfile variant
  automatically.
- **Linux**: an [AppImage](https://appimage.org/) — self-contained, runs
  on most distributions without a package-manager install or root
  access, the closest Linux equivalent of "a portable zip." A plain
  tarball with the same bootstrap layout is a reasonable fallback if
  AppImage's runtime requirements turn out to be a problem for target
  distros.

### CI/build pipeline

A GitHub Actions release workflow, triggered on a version tag
(`v0.1.0`-style), matrixed over `windows-latest`/`ubuntu-latest`:

1. Build the optimized C++ tracker (`meson setup --buildtype=release` +
   `meson compile`) for that OS.
2. Assemble the bootstrap contents (tracker binary, `uv` binary, pinned
   lockfile snapshot, launcher script/shortcut definition).
3. Compile the Windows Inno Setup script (`iscc`) / build the Linux
   AppImage.
4. Upload both artifacts to a GitHub Release.

This is new infrastructure — nothing in `.github/workflows/` does this
today (`docs.yml` only builds the docs site).

## Sketch of the changes this implies

Not designed in detail — sizing the work for whoever picks it up:

- **`onnxruntime-gpu` → base `onnxruntime` + GPU-variant
  `onnxruntime-gpu`** in `pyproject.toml`'s dependency groups — small,
  independent, worth doing first regardless of the rest of this doc.
- **A minimal launcher entry point** (or a tiny wrapper script) that
  runs `uv sync` then `uv run posetrak-ui`, for the installer's shortcut
  to target — doesn't exist today; `docs/setup.md`'s "Running the
  applications" section assumes an interactive shell.
- **The Inno Setup script** (Windows) and **AppImage build recipe**
  (Linux) — new files, likely under a new `packaging/` directory at the
  repo root.
- **The GitHub Actions release workflow** — new, `.github/workflows/`.
- **A versioning scheme** — `pyproject.toml`'s `version = "0.1.0"` is
  currently static; releases need this bumped and tagged consistently
  (whether that's manual or tooled isn't decided here).

## Open questions (not resolved here)

1. **Code signing** for the Windows installer — an unsigned installer
   triggers SmartScreen warnings; whether that's acceptable for a first
   release or needs a certificate (cost + process) isn't decided.
2. **Auto-update** — out of scope for a first release, but worth naming
   now so the launcher/bootstrap shape doesn't accidentally foreclose it
   later (e.g. the launcher could someday check the lockfile's version
   against the latest release before running `uv sync`).
3. **GPU vendor scope** — this doc only discusses NVIDIA/CUDA, matching
   `docs/setup.md`'s existing GPU-acceleration section. AMD/Intel GPU
   support isn't part of the current dependency story at all and isn't
   addressed here.
4. **Linux distro coverage for AppImage** — needs validating against
   whatever distros are actually likely targets (Ubuntu LTS versions at
   minimum); some AppImage runtime requirements (FUSE) trip up certain
   minimal/server distros and sandboxed environments.
5. **Fully-offline "fat" variant** — deliberately not building this for
   v1 per the recommendation above, but worth revisiting if early users
   report install-time internet access is a real blocker.
6. **Where the pinned lockfile snapshot for a given release actually
   lives** — a tagged copy of `uv.lock` at release time, vs. something
   more elaborate (a private package index). Simplest option (tag the
   lockfile as part of the release) is probably sufficient and should
   be the default assumption unless something concrete argues otherwise.
