#!/bin/bash
# set -euo pipefail

# Usage:
#   ./data_preprocess.sh [CUDA_DEVICES] [INPUT_DATA]
# Examples:
#   ./data_preprocess.sh 0 /data/raw/train
#   ./data_preprocess.sh "0,1" /data/raw/train

source /mnt/project_dft/yunhong/miniconda3/etc/profile.d/conda.sh
conda activate sphnet

# export CUDA_VISIBLE_DEVICES="0,1"

# set python path
export PYTHONPATH=$PYTHONPATH:.
export PROJECT_ROOT=./
export HYDRA_FULL_ERROR=1

rm -rf outputs/debug

python pipelines/train.py \
data_name="custom_nabla"