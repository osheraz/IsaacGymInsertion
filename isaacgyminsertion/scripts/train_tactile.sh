#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-gt_test}
NUM_ENVS=${4:-1}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth
data=/home/${USER}/tactile_insertion
data_folder=/home/${USER}/tactile_insertion/datastore_${SEED}_${CACHE}
output_dir=outputs/${CACHE}
path_norm=${data_folder}/normalization.pkl

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
train_tactile=True \
task.data_logger.base_folder="${data}" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.data_folder="${data_folder}" \
offline_train.output_dir="${output_dir}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}

