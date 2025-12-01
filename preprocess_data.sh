#!/bin/bash
# set -euo pipefail

# Usage:
#   ./data_preprocess.sh [CUDA_DEVICES] [INPUT_DATA]
# Examples:
#   ./data_preprocess.sh 0 /data/raw/train
#   ./data_preprocess.sh "0,1" /data/raw/train

source /mnt/project_dft/yunhong/miniconda3/etc/profile.d/conda.sh
conda activate sphnet

export PYTHONPATH=$PYTHONPATH:.
# CUDA_DEVICES="0"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
INPUT_DATA="train_100k"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/src/dataset/mp_preproc_nabla.py"  # adjust to your actual script

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Input data: $INPUT_DATA"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: python script not found at $PY_SCRIPT" >&2
    exit 2
fi

python "$PY_SCRIPT" --data_name "$INPUT_DATA"

# Print the completion message in the requested format
echo "print(\"input_data ${INPUT_DATA} preprocessed finished\")"