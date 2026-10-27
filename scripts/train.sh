#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "${NNODES:-1}" == 1 ]]; then
  exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
    -m navpi.train "$@"
else
  exec torchrun --nnodes="$NNODES" --nproc_per_node="${NPROC_PER_NODE:-8}" \
    --node_rank="${NODE_RANK:?}" --master_addr="${MASTER_ADDR:?}" \
    --master_port="${MASTER_PORT:-29500}" -m navpi.train "$@"
fi
