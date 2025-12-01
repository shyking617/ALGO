#!/bin/bash
# set -euo pipefail

# Usage:
#   ./data_preprocess.sh [CUDA_DEVICES] [INPUT_DATA]
# Examples:
#   ./data_preprocess.sh 0 /data/raw/train
#   ./data_preprocess.sh "0,1" /data/raw/train

source /mnt/project_dft/yunhong/miniconda3/etc/profile.d/conda.sh
conda activate sphnet

export CUDA_VISIBLE_DEVICES="0"

# set python path
export PYTHONPATH=$PYTHONPATH:.
export PROJECT_ROOT=./
export HYDRA_FULL_ERROR=1

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    ngpus=$(nvidia-smi --list-gpus | wc -l)
else
    ngpus=$(echo $CUDA_VISIBLE_DEVICES | tr -d ' ' | awk -F',' '{print NF}')
fi

rm -rf outputs/debug

exp_dir=outputs/nabla100k_200k_lr3e-4_bs16*8_sparse0.7_ema

python pipelines/test.py \
--config-path=../${exp_dir} \
ngpus=${ngpus} \
limit_test_batches=10