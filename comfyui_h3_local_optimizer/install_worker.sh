#!/usr/bin/env bash
set -euo pipefail
node_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 "$node_dir/install_runtime.py"
