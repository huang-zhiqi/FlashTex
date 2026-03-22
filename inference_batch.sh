#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

#（可选）只编译 3090/4090 的算力，缩短时间 #但其实没用
export TORCH_CUDA_ARCH_LIST="8.6;8.9"

export CUDA_HOME=/home/pubNAS3/zhiqi/.conda/envs/flashtex
export CPATH="${CUDA_HOME}/include:${CPATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${CUDA_HOME}/lib64/stubs:${LIBRARY_PATH}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export CPATH="${CONDA_PREFIX}/include:${CPATH}"
fi

# Optional: activate your FlashTex environment here.
# CONDA_BASE="$(conda info --base)"
# source "${CONDA_BASE}/etc/profile.d/conda.sh"
# conda activate flashtex

# ================= Configuration =================

EXP_NAME="flashtex"
BATCH_TSV="../experiments/common_splits/test.tsv"
OUTPUT_ROOT="../experiments/${EXP_NAME}"
CAPTION_FIELD="caption_short"
MAX_SAMPLES=50

# Multi-GPU configuration
# GPU_IDS: physical GPU ids, comma-separated
# NUM_GPUS: actual number of GPUs to use from GPU_IDS
# WORKERS_PER_GPU: parallel workers per GPU, or "auto"
# GPU_IDS="0,1"
# NUM_GPUS=2
# WORKERS_PER_GPU=1

# 4卡配置 (解开注释使用)
GPU_IDS="0,1,2,3"
# GPU_IDS="3,4,5,6"
NUM_GPUS=4
WORKERS_PER_GPU=1

# Match the single-sample command you used before.
ROTATION_Y=0
GUIDANCE_SDS="SDS_LightControlNet"
CONTROLNET_NAME="kangled/lightcontrolnet"

# =================================================

echo "Starting FlashTex batch inference..."
echo "Batch TSV: ${BATCH_TSV}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Caption field: ${CAPTION_FIELD}"
echo "Max samples: ${MAX_SAMPLES}"
echo "GPU IDs: ${GPU_IDS}, Num GPUs: ${NUM_GPUS}, Workers/GPU: ${WORKERS_PER_GPU}"
if [[ "${WORKERS_PER_GPU}" =~ ^[0-9]+$ ]]; then
  echo "Total parallel workers: $((NUM_GPUS * WORKERS_PER_GPU))"
else
  echo "Total parallel workers: auto"
fi
echo "Guidance: ${GUIDANCE_SDS}"
echo "ControlNet: ${CONTROLNET_NAME}"
echo "Textures will be stored under: ${OUTPUT_ROOT}/textures"

python generate_texture.py \
  --tsv_path "${BATCH_TSV}" \
  --output "${OUTPUT_ROOT}" \
  --caption_field "${CAPTION_FIELD}" \
  --max_samples "${MAX_SAMPLES}" \
  --gpu_ids "${GPU_IDS}" \
  --num_gpus "${NUM_GPUS}" \
  --workers_per_gpu "${WORKERS_PER_GPU}" \
  --skip_existing \
  --rotation_y "${ROTATION_Y}" \
  --guidance_sds "${GUIDANCE_SDS}" \
  --pbr_material \
  --controlnet_name "${CONTROLNET_NAME}" \
  --production
