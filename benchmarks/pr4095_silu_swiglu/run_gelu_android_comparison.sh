#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export BEFORE_REF=${BEFORE_REF:-933679c5^}
export AFTER_REF=${AFTER_REF:-933679c5}
export LAYERS=${LAYERS:-GELU}
export RESULT_DIR=${RESULT_DIR:-"$(git rev-parse --show-toplevel)/benchmark-results/pr4094-gelu-android"}

exec "${SCRIPT_DIR}/run_android_comparison.sh" "$@"
