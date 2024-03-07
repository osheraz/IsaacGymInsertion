#!/bin/bash
GPUS=${1:-0}
SEED=${2:-0}
CACHE=${3:-contact2}
NUM_ENVS=${4:-4}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth
data=/home/roblab20/tactile_insertion
data_folder=/home/roblab20/tactile_insertion/datastore_42_contact2
output_dir=outputs/${CACHE}
path_norm=${data_folder}/normalization.pkl
student_ckpt_path=/home/roblab20/tactile_insertion/datastore_0_contact2/tac+eef/checkpoints/model_2.pt

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=True \
test=False \
offline_training_w_env=False \
offline_train.train.action_regularization=False \
offline_train.model.transformer.full_sequence=False \
offline_train.model.transformer.sequence_length=1 \
offline_train.train.load_checkpoint=False \
offline_train.train.student_ckpt_path="${student_ckpt_path}" \
task.env.tactile=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=3 \
task.tactile.half_image=True \
task.env.smooth_force=True \
task.env.tactile_history_len=1 \
task.env.tactile_wrt_force=False \
task.env.compute_contact_gt=False \
task.env.numObsHist=1 \
task.env.numObservations=18 \
task.data_logger.base_folder="${data}" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.data_folder="${data_folder}" \
offline_train.output_dir="${output_dir}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}

