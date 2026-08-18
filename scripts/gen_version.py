#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Harri Kaimio
#
# SPDX-License-Identifier: Apache-2.0

"""Generate build/version header with git commit hash and build timestamp."""
import subprocess
import datetime
import sys


def git(cmd):
    return subprocess.check_output(['git'] + cmd, stderr=subprocess.DEVNULL).decode().strip()


try:
    commit = git(['rev-parse', '--short', 'HEAD'])
    dirty = bool(git(['status', '--porcelain']))
except Exception:
    commit = 'unknown'
    dirty = False

ts = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
commit_str = commit + ('+' if dirty else '')

with open(sys.argv[1], 'w') as f:
    f.write(f'''\
// AUTO-GENERATED — do not edit. Regenerated on every build by scripts/gen_version.py.
#pragma once
namespace posetrak::build {{
  inline constexpr char const* GIT_COMMIT      = "{commit_str}";
  inline constexpr char const* BUILD_TIMESTAMP = "{ts}";
}}
''')
