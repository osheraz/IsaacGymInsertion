#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-dex}
NUM_ENVS=${4:-1}
HEADLESS=${5:-False}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth
data=/home/roblab20/tactile_dex
data_folder=/home/roblab20/tactile_dex/datastore_42_dex_dex/
path_norm=/${data_folder}/normalization.pkl
output_dir=outputs/${CACHE}
student_ckpt_path=/home/roblab20/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/dex/student/checkpoints/model_last.pt

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=False \
test=True \
offline_training_w_env=True \
offline_train.train.action_regularization=False \
offline_train.model.transformer.full_sequence=False \
offline_train.model.transformer.sequence_length=1 \
offline_train.train.load_checkpoint=True \
offline_train.train.student_ckpt_path="${student_ckpt_path}" \
offline_train.train.only_test=True \
offline_train.train.only_validate=False \
task.env.tactile=True \
task.env.numStates=21 \
task.env.numObservations=10 \
task.tactile.tacto.width=64 \
task.tactile.tacto.height=64 \
task.tactile.encoder.width=64 \
task.tactile.encoder.height=64 \
task.tactile.encoder.num_channels=3 \
task.env.tactile_history_len=1 \
task.env.compute_contact_gt=False \
task.env.tactile_wrt_force=False \
task.external_cam.external_cam=True \
task.external_cam.cam_res.w=320 \
task.external_cam.cam_res.h=180 \
task.tactile.half_image=True \
task.env.smooth_force=True \
train.ppo.only_contact=False \
train.ppo.priv_info=True \
task.data_logger.base_folder="${data}" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.data_folder="${data_folder}" \
offline_train.output_dir="${output_dir}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}