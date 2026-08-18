#!/bin/bash

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

rm -rf tracking_tests/cpp-python-comparison/cpp_results/
optbuild/cli/posetrak tests/cpp-python/cpp_test_config.toml
