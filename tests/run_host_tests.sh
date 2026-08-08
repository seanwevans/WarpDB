#!/usr/bin/env bash
#
# Compile and run the WarpDB tests that depend only on the host C++ sources
# (the SQL tokenizer/parser in src/expression.cpp). These need neither a CUDA
# toolkit nor a GPU, so they give CI a fast lane that catches parser and
# code-generation regressions even on machines without nvcc.
#
# The main CMake/ctest build still exercises the full GPU-backed suite; this is
# an additive safety net, not a replacement.
set -euo pipefail
cd "$(dirname "$0")/.."

CXX=${CXX:-g++}
STD=${STD:-c++17}

# Tests whose only translation-unit dependency is src/expression.cpp. Keep this
# list in sync when adding new parser-only tests.
host_tests=(
  tests/test_expression.cpp
  tests/tokenizer_tests.cpp
  tests/parsing_error_tests.cpp
  tests/precedence_tests.cpp
  tests/query_parser_test.cpp
  tests/equals_operator_test.cpp
  tests/having_aggregate_test.cpp
  tests/not_equal_operator_test.cpp
  tests/select_star_test.cpp
  tests/eval_logical_ops_test.cpp
)

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

fail=0
for src in "${host_tests[@]}"; do
  name="$(basename "$src" .cpp)"
  # tests/stubs provides lightweight CUDA headers for host-only tests that
  # transitively include <cuda_runtime.h> (e.g. via eval_helpers.hpp); parser
  # tests that don't include it are unaffected.
  if ! "$CXX" "-std=$STD" -Iinclude -Itests/stubs "$src" src/expression.cpp -o "$tmp/$name" 2>"$tmp/$name.log"; then
    echo "FAIL  $name (compile)"
    cat "$tmp/$name.log"
    fail=1
    continue
  fi
  if out="$("$tmp/$name" 2>&1)"; then
    echo "PASS  $name -- $out"
  else
    echo "FAIL  $name (run)"
    echo "$out"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "Host-only test suite FAILED"
  exit 1
fi
echo "Host-only test suite passed (${#host_tests[@]} tests)"
