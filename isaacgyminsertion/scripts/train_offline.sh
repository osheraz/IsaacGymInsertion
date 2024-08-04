#!/bin/bash
SEED=${2:-42}
CACHE=${3:-teacher}
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
student_ckpt_path=/home/${USER}/tactile_insertion/datastore_${SEED}_${CACHE}/tac+eef/checkpoints/model_2.pt
tact_path=/home/${USER}/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/${CACHE}/tact/checkpoints/model_last.pt

python train_supervised.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=True \
offline_train.only_bc=True \
offline_train.multi_gpu=False \
offline_train.gpu_ids=[0,1] \
offline_train.train.student_ckpt_path="${student_ckpt_path}" \
offline_train.model.transformer.load_tact=False \
offline_train.model.transformer.tact_path="${tact_path}" \
task.env.tactile=False \
task.external_cam.external_cam=False \
task.data_logger.base_folder="${data}" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.data_folder="${data_folder}" \
offline_train.output_dir="${output_dir}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}

