#!/usr/bin/env bash
set -euo pipefail
: "${CANN_ROOT:?Set CANN_ROOT to the installed CANN root}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source "$CANN_ROOT/set_env.sh"
set -u
"$script_dir/hardware_probe" "${1:-0}"
