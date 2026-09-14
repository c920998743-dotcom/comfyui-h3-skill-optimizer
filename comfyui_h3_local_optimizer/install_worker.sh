#!/usr/bin/env bash
set -euo pipefail
# Run only where installing a separate runtime is supported.
worker_env="${1:?Usage: bash install_worker.sh /writable/path/h3-omni-venv}"
node_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python3 -m venv "$worker_env"
"$worker_env/bin/python" -m pip install --upgrade pip
"$worker_env/bin/python" -m pip install -r "$node_dir/worker-requirements.txt"
"$worker_env/bin/python" -m pip check
echo "Worker Python: $worker_env/bin/python"
echo "Next: download the complete official AWQ model folder, then run check_environment.py."
