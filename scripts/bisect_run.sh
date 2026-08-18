#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

# bisect_run.sh — build and run posetrak at a specific git commit, save results.
#
# Usage: bisect_run.sh <commit|HEAD> <label>
#   e.g.: ./scripts/bisect_run.sh HEAD head
#         ./scripts/bisect_run.sh ed01835 pre-pinocchio
#         ./scripts/bisect_run.sh a50b4af baseline
#
# Results saved to tracking_tests/harri-no-palms-<label>/
# HEAD/optbuild is presumed already built; use "HEAD" to skip the build step.

set -euo pipefail

COMMIT="${1:?usage: bisect_run.sh <commit|HEAD> <label>}"
LABEL="${2:?usage: bisect_run.sh <commit|HEAD> <label>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOML_TEMPLATE="$ROOT/tests/harri-no-palms.toml"
OUTPUT_DIR="$ROOT/tracking_tests/harri-no-palms-$LABEL"

echo "=== Bisect run: commit=$COMMIT  label=$LABEL ==="
echo "  Output: $OUTPUT_DIR"

if [[ "$COMMIT" == "HEAD" ]]; then
    BINARY="$ROOT/optbuild/cli/posetrak"
    LD_PATH="$ROOT/optbuild/subprojects/fmt-12.0.0:$ROOT/optbuild/src:$ROOT/optbuild/subprojects/yaml-cpp-0.8.0:$ROOT/optbuild/subprojects/tomlplusplus-3.4.0/src:/opt/openrobots/lib"
    echo "  Binary: optbuild (HEAD already built)"
else
    WORKTREE="$ROOT/../posetrak-bisect-$LABEL"
    BUILDDIR="$WORKTREE/build"
    BINARY="$BUILDDIR/cli/posetrak"
    LD_PATH="$BUILDDIR/subprojects/fmt-12.0.0:$BUILDDIR/src:$BUILDDIR/subprojects/yaml-cpp-0.8.0:$BUILDDIR/subprojects/tomlplusplus-3.4.0/src:/opt/openrobots/lib"

    cd "$ROOT"
    if [[ ! -d "$WORKTREE" ]]; then
        echo "  Creating worktree at $COMMIT..."
        git worktree add "$WORKTREE" "$COMMIT"
    else
        echo "  Worktree already exists: $WORKTREE"
    fi

    # Share package cache to avoid re-downloading subprojects
    mkdir -p "$WORKTREE/subprojects/packagecache"
    for f in "$ROOT/subprojects/packagecache/"*; do
        fname="$(basename "$f")"
        dest="$WORKTREE/subprojects/packagecache/$fname"
        [[ -e "$dest" ]] || ln -s "$f" "$dest"
    done

    cd "$WORKTREE"
    if [[ ! -f "$BUILDDIR/build.ninja" ]]; then
        echo "  Configuring meson..."
        meson setup "$BUILDDIR" \
            --buildtype=release \
            --optimization=3 \
            -Denable_tests=false \
            2>&1 | tail -3
    fi

    echo "  Compiling..."
    ninja -C "$BUILDDIR" cli/posetrak -j"$(nproc)" 2>&1 | tail -3
    echo "  Build done."
    cd "$ROOT"
fi

# Create per-run TOML with overridden output directory
mkdir -p "$OUTPUT_DIR"
TMP_TOML="$(mktemp /tmp/harri-no-palms-XXXXXX.toml)"
sed "s|directory = .*|directory = \"$OUTPUT_DIR\"|" "$TOML_TEMPLATE" > "$TMP_TOML"

echo "  Running tracker..."
LD_LIBRARY_PATH="$LD_PATH" "$BINARY" "$TMP_TOML" 2>&1 | tail -5
rm -f "$TMP_TOML"

echo ""
echo "=== Done: $OUTPUT_DIR ==="
