# Posetrak — Setup Guide

## Platform overview

| Component | Linux | Windows |
|---|---|---|
| C++ tracker (`posetrak` exe) | Build from source | Use cross-compiled exe |
| Python tools (UI, pose extraction, DB) | `uv sync` | `uv sync` |
| Development (edit & rebuild) | Full support | Not recommended¹ |

¹ Native MSVC builds fail due to Pinocchio/Eigen template depth limits.
  Cross-compile on Linux instead (see [Windows: C++ tracker](#windows-c-tracker)).

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

The CLI is at `builddir/cli/posetrak` (debug) or `optbuild/cli/posetrak` (optimized).

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

Produces a `posetrak.exe` that runs on Windows x86-64 without installing any
build tools on the Windows machine.

```bash
sudo apt install mingw-w64
meson setup winbuild --cross-file cross/mingw-w64-x86_64.ini
meson compile -C winbuild
```

The exe is at `winbuild/cli/posetrak.exe`. Copy it to Windows together with the
four MinGW runtime DLLs (see [Windows: C++ tracker](#windows-c-tracker)).

---

## Windows

### C++ tracker

The recommended approach is to use an exe cross-compiled on Linux (see above).
Copy the following files into a single directory on the Windows machine:

| File | Source (on the Linux build machine) |
|---|---|
| `posetrak.exe` | `winbuild/cli/posetrak.exe` |
| `libgcc_s_seh-1.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/13-win32/` |
| `libstdc++-6.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/13-win32/` |
| `libgomp-1.dll` | `/usr/lib/gcc/x86_64-w64-mingw32/13-win32/` |
| `libwinpthread-1.dll` | `/usr/x86_64-w64-mingw32/lib/` |

All five files must be in the same directory. `libwinpthread-1.dll` is a
transitive dependency of `libgomp-1.dll` (OpenMP thread pool) and must be
included even though it is not listed in `posetrak.exe`'s direct imports.

Verify with:

```
posetrak.exe --help
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

The `pose-app` group installs the CPU `onnxruntime` by default. For GPU inference,
reinstall with the GPU variant after syncing:

```bash
uv sync --group pose-app
uv pip install --reinstall onnxruntime-gpu
```

This must be repeated after any `uv sync` that touches the `pose-app` group,
because uv may reinstall the CPU version as a dependency resolution side effect.

### Running the applications

```bash
uv run posetrak-ui            # Main viewer / editor
uv run posetrak-pose          # Pose detection pipeline
uv run posetrak-db --help     # Database CLI
uv run posetrak-mcp --db-path /path/to/session.db   # MCP diagnostic server

# C++ tracker (Linux — from debug build)
./builddir/cli/posetrak track config.toml
# C++ tracker (Linux — from optimized build)
./optbuild/cli/posetrak track config.toml
```

### Running Python tests

```bash
uv run pytest                  # all tests
uv run pytest python/tests/db/ # subset
uv run pytest -v               # verbose
```
