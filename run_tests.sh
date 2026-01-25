#!/bin/bash
# Run tests with system libraries (avoids conda library conflicts)
cd "$(dirname "$0")"
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH meson test -C builddir "$@"
