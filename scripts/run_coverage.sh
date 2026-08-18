#!/bin/bash

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

# Script to run tests with coverage and generate HTML report

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/builddir"

cd "$PROJECT_DIR"

echo "==> Building with coverage instrumentation..."
meson configure "$BUILD_DIR" -Db_coverage=true
meson compile -C "$BUILD_DIR"

echo ""
echo "==> Running tests..."
LD_LIBRARY_PATH="$BUILD_DIR/subprojects/tomlplusplus-3.4.0/src:$BUILD_DIR/subprojects/fmt-12.0.0:$BUILD_DIR/src:/usr/lib/x86_64-linux-gnu" \
  "$BUILD_DIR/tests/test_posetrak" --reporter compact

echo ""
echo "==> Generating coverage data..."
lcov --capture \
  --directory "$BUILD_DIR/src" \
  --directory "$BUILD_DIR/tests" \
  --output-file "$BUILD_DIR/coverage.info" \
  --ignore-errors mismatch,inconsistent \
  --quiet

echo "==> Filtering coverage data..."
lcov --remove "$BUILD_DIR/coverage.info" \
  '/usr/*' '*/miniconda3/*' '*/subprojects/*' '*/tests/*' \
  --output-file "$BUILD_DIR/coverage.filtered.info" \
  --quiet

echo "==> Generating HTML report..."
genhtml "$BUILD_DIR/coverage.filtered.info" \
  --output-directory "$BUILD_DIR/coverage_html" \
  --quiet

echo ""
echo "==> Coverage Summary:"
lcov --summary "$BUILD_DIR/coverage.filtered.info" 2>&1 | grep -E '(lines|functions)'

echo ""
echo "✓ Coverage report generated at: $BUILD_DIR/coverage_html/index.html"
echo "  Open with: xdg-open $BUILD_DIR/coverage_html/index.html"
