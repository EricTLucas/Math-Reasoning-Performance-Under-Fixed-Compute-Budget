#!/bin/bash
#$ -S /bin/bash
#$ -N aimo-baseline
#$ -q all.q
#$ -pe smp 4
#$ -l gpu_a100=true
#$ -l ngpus=1
#$ -l h_rt=04:00:00
#$ -j y
set -euo pipefail
cd "$HOME/aimo3"
: "${CUDA_VISIBLE_DEVICES:?GPU allocation variable missing}"
TASK_ID=${SGE_TASK_ID:-1}
if [[ "$TASK_ID" == "undefined" ]]; then TASK_ID=1; fi
exec apptainer exec --nv --cleanenv --pwd / \
  --bind "$HOME/aimo3:/work" \
  --bind "$HOME/aimo3/models:/models:ro" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env JOB_ID="${JOB_ID:-manual}" \
  --env OMP_NUM_THREADS=4 \
  --env VLLM_ATTENTION_BACKEND=TRITON_ATTN \
  --env VLLM_MXFP4_USE_MARLIN=1 \
  --env TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_encodings \
  --env HF_HUB_OFFLINE=1 \
  "$HOME/aimo3/containers/aimo3-env-v2.sif" \
  /opt/aimo-venv/bin/python -u /work/aimo_baseline_bundle/aimo_baseline.py run \
  --experiment /work/baselines/a100-4k-3tries-v1 \
  --task "$TASK_ID" --max-job-seconds 13800
