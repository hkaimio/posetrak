# Posetrak — Setup Guide

## Platform overview

| Component | Linux | Windows |
|---|---|---|
| C++ tracker (`posetrak` exe) | Build from source | Build from source (native MSVC) or use a cross-compiled exe |
| Python tools (UI, pose extraction, DB) | `uv sync` | `uv sync` |
| Development (edit & rebuild) | Full support | Full support¹ |

¹ Native MSVC development is fully supported — see
  [CONTRIBUTING.md's "Windows (native, MSVC)"](../CONTRIBUTING.md#windows-native-msvc)
  section and `setup-windows.ps1`. If you'd rather not maintain a native Windows
  toolchain at all, cross-compiling from Linux/WSL (below) still works and produces
  a runnable exe without installing MSVC/Pinocchio/Boost on the Windows machine —
  just without the ability to edit and rebuild there.

---

## Linux

### System prerequisites

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

**Python** — install [uv](https://docs.astral.sh/uv/getting-started/installation/):

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

The CLI is at `builddir/cli/posetrak-tracker` (debug) or `optbuild/cli/posetrak-tracker` (optimized).

### Running C++ tests

If conda is active, use the wrapper script to avoid `GLIBCXX` version conflicts:

```bash
./run_tests.sh              # all tests
./run_tests.sh -v           # verbose
./run_tests.sh --test-args="[skeleton]"   # filter by tag
```

Without conda active:

```bash
meson test -C builddir
meson test -C builddir -v
```

### Cross-compiling for Windows (on Linux)

Produces a `posetrak-tracker.exe` that runs on Windows x86-64 without installing any
build tools on the Windows machine.

```bash
sudo apt install mingw-w64
meson setup winbuild --cross-file cross/mingw-w64-x86_64.ini
meson compile -C winbuild
```

The exe is at `winbuild/cli/posetrak-tracker.exe`. Copy it to Windows together with the
four MinGW runtime DLLs (see [Windows: C++ tracker](#windows-c-tracker)).

---

## Windows

### C++ tracker

**For development (edit & rebuild), build natively with MSVC** — see
[CONTRIBUTING.md's "Windows (native, MSVC)"](../CONTRIBUTING.md#windows-native-msvc)
section, or just run `setup-windows.ps1` from the repo root. This needs Visual
Studio 2022+ (C++ desktop workload) and a small conda environment for Pinocchio/Boost
headers, but produces both a debug (`builddir/`) and release (`optbuild/`) build you
can iterate on locally.

**If you only need to *run* the tracker** (no local edit/rebuild), an exe
cross-compiled on Linux avoids installing any of that. Copy the following files into
a single directory on the Windows machine:

| File | Source (on the Linux build machine) |
|---|---|
| `posetrak-tracker.exe` | `winbuild/cli/posetrak-tracker.exe` |
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

### Python tools

Install [uv for Windows](https://docs.astral.sh/uv/getting-started/installation/#windows):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then follow the [Python environment setup](#python-environment) section below.

---

## Python environment

All Python commands run from the **repository root** (where `pyproject.toml` lives)
using `uv run`. Never use `python` directly or activate a venv manually.

### Base install

```bash
uv sync
```

This installs the core posetrak package and the base dependencies (PySide6, OpenCV,
NumPy, pandas, etc.) needed for `posetrak-ui`, `posetrak-db`, and `posetrak-pose`.

### pre-commit hooks

`pre-commit` is deliberately **not** in any `uv` dependency group (it's a repo tool,
not a package dependency), so install and register it separately once per clone:

```bash
uv pip install pre-commit
pre-commit install
```

This registers the hooks at `.git/hooks/pre-commit` (end-of-file/whitespace fixers,
clang-format for C++). Do this before making your first commit in a new workarea —
see `CLAUDE.md`'s Git conventions.

### Optional dependency groups

Install only what you need:

```bash
uv sync --group dev           # pytest, coverage — needed to run Python tests
uv sync --group pose-app      # YOLO + RTMPose + PyAV — needed for pose extraction
uv sync --group analysis      # Marimo + Plotly — for analysis notebooks
uv sync --group mcp-server    # MCP server — for posetrak-mcp diagnostic tool
uv sync --group pipeline      # Ultralytics — standalone pipeline tools
```

Multiple groups can be combined:

```bash
uv sync --group dev --group pose-app
```

### GPU acceleration for pose extraction

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

### Cutie segmentation

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

### Running the applications

```bash
uv run posetrak-ui            # Main viewer / editor
uv run posetrak-pose          # Pose detection pipeline
uv run posetrak-db --help     # Database CLI
uv run posetrak-mcp --db-path /path/to/session.db   # MCP diagnostic server

# C++ tracker (Linux — from debug build)
./builddir/cli/posetrak-tracker track config.toml
# C++ tracker (Linux — from optimized build)
./optbuild/cli/posetrak-tracker track config.toml
```

### Running Python tests

```bash
uv run pytest                  # all tests
uv run pytest python/tests/db/ # subset
uv run pytest -v               # verbose
```
