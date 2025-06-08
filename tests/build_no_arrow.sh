#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
build_dir=build-no-arrow
rm -rf "$build_dir"
cmake -S . -B "$build_dir" -DCMAKE_DISABLE_FIND_PACKAGE_Arrow=ON >/dev/null
cmake --build "$build_dir" --target warpdb >/dev/null
