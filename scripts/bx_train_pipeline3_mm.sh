#!/usr/bin/env bash
# P3 counterpart of bx_train_ltx_online_mm.sh; no conda/torchrun interpreter ambiguity.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYBIN="${PYBIN:-/private/liqi_workspace/project_code/Omni-LTX-stage2/.venv/bin/python}"
CLUSTER_CONFIG="${CLUSTER_CONFIG:-${ROOT}/configs/p3_three_nodes.json}"
: "${RUN_LOG_DIR:?Set RUN_LOG_DIR to a new directory for this launch}"
args=(--cluster "${CLUSTER_CONFIG}" --log-dir "${RUN_LOG_DIR}")
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then args+=(--preflight-only); fi
exec "${PYBIN}" "${SCRIPT_DIR}/pipeline3_cluster.py" "${args[@]}" -- "$@"
