# Posetrak — Setup Guide

## Windows

Use the pre-packaged installer for Windows: **[posetrak-setup-0.1.0-proto1.exe](https://1drv.ms/u/c/5e84dad12af05ffc/IQAn0785a1wFTqraaclSGSMbAcNu8ZJUe636owFxRLTEYhM?e=jsrQwi)**.

This is an early, unsigned prototype build, so Windows SmartScreen will very
likely show a "Windows protected your PC" warning. If you see this message,
click "More info", then "Run anyway". That's expected right now, not a sign
something is broken.

By default the installer does not set up libraries needed for Posetrak's
segmentation. This requires additional large AI models taking several GB disk
space, and currently work only with NVIDIA GPU. If you want to use Posetrak for
multi-person motion capture you'll likely want to enable this option on last
page of installer wizard; for capturing single person segmentation is usually
not necessary.

First launch needs internet access, as Posetrak downloads libraries it needs.
This can take a couple of minutes. On next launch startup should be quicker.

Once it's running, continue to the hands-on
[tutorial](user-guide/tutorial1.md).

## Linux, Mac

Currently there is no installer for other platorms than Windows so you need to
build Posetrak yourself. Linux is fully supported; other platforms have not yet
been tested.

## Development environment setup

Posetrak consist of 2 parts: UI and command line clients written in Python and
the actual tracker implemented in C++. Python component setup is same in all
platforms; C++ environment differs depending on your OS



### Linux setup

#### System prerequisites

**Build tools**

```bash
sudo apt install meson ninja-build gcc g++ libboost-dev
```

**Pinocchio** — installed via the robotpkg APT repository:

```bash
# Add the robotpkg APT source (Ubuntu 24.04 "noble")
echo "deb [arch=amd64] http://robotpkg.openrobots.org/packages/debian/pub noble robotpkg" \
    | sudo tee /etc/apt/sources.list.d/robotpkg.list
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.key \
    | sudo apt-key add -
sudo apt update
sudo apt install robotpkg-pinocchio
```

This installs headers to `/opt/openrobots/include/` — the path expected by `meson.build`.

**UV** — install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Building the C++ tracker

```bash
# Debug build (for unit tests and development)
meson setup builddir
meson compile -C builddir

# Optimized build (use for actual tracking — significantly faster)
meson setup optbuild --buildtype=release
meson compile -C optbuild
```

The CLI is at `builddir/cpp/cli/posetrak-tracker` (debug) or
`optbuild/cpp/cli/posetrak-tracker` (optimized).

### Running C++ tests


```bash
meson test -C builddir
meson test -C builddir -v
```

If conda is active, use the wrapper script to avoid `GLIBCXX` version conflicts:

```bash
./run_tests.sh              # all tests
./run_tests.sh -v           # verbose
./run_tests.sh --test-args="[skeleton]"   # filter by tag
```

### Cross-compiling for Windows (on Linux)

Produces a `posetrak-tracker.exe` that runs on Windows x86-64 without installing any
build tools on the Windows machine.

```bash
sudo apt install mingw-w64
meson setup winbuild --cross-file cpp/cross/mingw-w64-x86_64.ini
meson compile -C winbuild
```

The exe is at `winbuild/cpp/cli/posetrak-tracker.exe`. Copy it to Windows together with the
four MinGW runtime DLLs (see [Windows: C++ tracker](#c-tracker)).


---

### Windows

#### C++ tracker

The C++ components can be built in Windows with Visual Studio. Alternatively,
you can cross compile these in Linux (see above).

**One-time setup:**

1. Visual Studio 2022 (or later) with the "Desktop development with C++"
   workload.
2. [Meson](https://mesonbuild.com/) and Ninja (`pip install meson ninja`, or
   the standalone Meson MSI installer).
3. Pinocchio 3.9.0 headers + a compiled Boost Serialization. Easiest way to
   install it is via a dedicated conda environment (this project only needs the
   headers/libs, not a Python installation of Pinocchio itself):

   ```powershell
   conda create -y -n posetrak-pinocchio -c conda-forge pinocchio=3.9.0
   ```

   **Match the version to whatever `/opt/openrobots` actually has**: check
   the [openrobots package list](https://robotpkg.openrobots.org/) or ask
   whoever maintains the Linux/WSL dev environment. A header-version
   mismatch shows up as compile errors (e.g. `Frame::parentJoint` not
   existing — that field was `Frame::parent` before Pinocchio 3.0), not
   something that just happens to link with slightly wrong behaviour.

4. Run `setup-windows.ps1` (repo root). It configures and builds two build
   directories — `builddir/` (debug, for day-to-day unit testing/debugging) and
   `optbuild/` (release, for actual tracking runs) — and copies the two runtime
   DLLs described below next to each built `.exe`.
   Re-run it after a `meson.build`/`meson_options.txt` changes.


Build with

```
meson compile -C builddir (debug build)
meson compile -C builddir (optimized build)
```

RUn tests:

```
meson test    -C builddir
```

**Before running any built `.exe`**, copy the two runtime DLLs
`boost_serialization.dll`, `yaml-cpp.dll` into the same directory as the `.exe`:

```powershell
Copy-Item "$pinocchioEnv\bin\boost_serialization.dll", "$env:USERPROFILE\miniconda3\Library\bin\yaml-cpp.dll" -Destination builddir\cli
Copy-Item "$pinocchioEnv\bin\boost_serialization.dll", "$env:USERPROFILE\miniconda3\Library\bin\yaml-cpp.dll" -Destination builddir\tests

```

(and the same into `optbuild\cli` / `optbuild\tests`).

Windows always searches an executable's own directory for its DLL dependencies
first, this way the binary is runnable standalone. Posetrak's Puthon UI launches
`posetrak-tracker.exe` as a subprocess, which inherits whatever environment the
UI happens to be running in. Forgetting this step doesn't fail loudly: you'll
likely see some weird warning like  `libshiboken: Overflow` after the exist code
has been marshalled through multiple SW layers in PySide.


##### Cross-compiling in Linux

After cross-compiling copy the following files into a single directory on the
Windows machine:

| File | Source (on the Linux build machine) |
|---|---|
| `posetrak-tracker.exe` | `winbuild/cpp/cli/posetrak-tracker.exe` |
| `libgcc_s_seh-1.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/13-win32/` |
| `libstdc++-6.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/13-win32/` |
| `libgomp-1.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/13-win32/` |
| `libwinpthread-1.dll` | `/usr/x86_64-w64-mingw32/lib/` |

All five files must be in the same directory. `libwinpthread-1.dll` is a
transitive dependency of `libgomp-1.dll` (OpenMP thread pool) and must be
included even though it is not listed in `posetrak-tracker.exe`'s direct imports.

Verify with:

```
posetrak-tracker.exe --help
```

#### Python tools

Install [uv for Windows](https://docs.astral.sh/uv/getting-started/installation/#windows):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then follow the [Python environment setup](#python-environment) section below.

---

### Python environment

All Python commands run from the **repository root** (where `pyproject.toml` lives)
using `uv run`. Never use `python` directly or activate a venv manually.

#### Base install

```bash
uv sync
```

This installs the core posetrak package and the base dependencies (PySide6, OpenCV,
NumPy, pandas, etc.) needed for `posetrak-ui` and the `posetrak` CLI.

#### pre-commit hooks

`pre-commit` is deliberately **not** in any `uv` dependency group (it's a repo tool,
not a package dependency), so install and register it separately once per clone:

```bash
uv pip install pre-commit
pre-commit install
```

This registers the hooks at `.git/hooks/pre-commit` (end-of-file/whitespace fixers,
clang-format for C++). Do this before making your first commit in a new workarea —
see `CLAUDE.md`'s Git conventions.

#### Optional dependency groups

Install only what you need:

```bash
uv sync --group dev           # pytest, coverage — needed to run Python tests
uv sync --group segmentation  # torch/torchvision — GPU inference, Cutie segmentation
uv sync --group analysis      # Marimo + Plotly — for analysis notebooks
uv sync --group mcp-server    # MCP server — for posetrak-mcp diagnostic tool
uv sync --group tools         # standalone utility scripts in python/tools/
uv sync --group docs          # MkDocs + Material — to build this site locally
```

Multiple groups can be combined:

```bash
uv sync --group dev --group segmentation
```

#### GPU acceleration for pose extraction

`onnxruntime-gpu` is in the base dependencies, so GPU inference is enabled
automatically on NVIDIA systems once the CUDA-enabled PyTorch wheel is installed.

**NVIDIA (CUDA 12.x) — Windows or Linux:**

```bash
uv sync --group segmentation
```

This installs `torch 2.9.1+cu126` from the PyTorch CUDA 12.6 index (see
`[tool.uv.sources]` in `pyproject.toml`). The `onnxruntime-gpu` CUDA provider
then works automatically — on Windows the code registers PyTorch's bundled
`lib/` directory as a DLL search path so `cublasLt64_12.dll` is found.

**Non-NVIDIA / CPU-only:**

Skip `--group segmentation`. The base `uv sync` installs CPU `onnxruntime`
alongside `onnxruntime-gpu`; detection falls back to `CPUExecutionProvider`
automatically.

**Version constraints:**

| Package | Constraint | Reason |
|---|---|---|
| `onnxruntime-gpu` | `>=1.19,<1.26` | 1.26+ require CUDA 13 (`cublasLt64_13.dll`) |
| `torch` (segmentation group) | `>=2.7,<2.10` | 2.10 from PyPI is CPU-only; cu126 index tops at 2.9.1 |

#### Cutie segmentation

Cutie (video object segmentation, used in the segmentation init UI) is not on
PyPI. It must be cloned separately. Clone it to the standard location and the
app will find it automatically:

**Linux / macOS:**

```bash
git clone https://github.com/hkchengrex/Cutie ~/.local/share/posetrak/Cutie
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/hkchengrex/Cutie "$env:LOCALAPPDATA\posetrak\Cutie"
```

Alternatively, clone anywhere and point to it with `CUTIE_DIR`:

```bash
CUTIE_DIR=/path/to/Cutie uv run posetrak-ui
```

The `--group segmentation` dependencies (`torch`, `torchvision`, `omegaconf`,
`hydra-core`, `timm`) must still be installed via `uv sync --group segmentation`.
Cutie's own model weights are downloaded automatically on first use via
`get_default_model()`.

#### Running the applications

```bash
uv run posetrak-ui            # Main GUI: setup, pose extraction, tracking, editing
uv run posetrak --help        # CLI: session, capture, detection, tracker, export/import
uv run posetrak-mcp --db-path /path/to/session.db   # MCP diagnostic server

# C++ tracker (Linux — from debug build)
./builddir/cpp/cli/posetrak-tracker track config.toml
# C++ tracker (Linux — from optimized build)
./optbuild/cpp/cli/posetrak-tracker track config.toml
```

#### Running Python tests

```bash
uv run pytest                  # all tests
uv run pytest python/tests/db/ # subset
uv run pytest -v               # verbose
```
