#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-gt_test}
NUM_ENVS=${4:-9}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
test=True \
task.data_logger.collect_data=True \
task.env.numEnvs=${NUM_ENVS} \
task.env.tactile=True \
task.env.numStates=7 \
task.env.numObservations=18 \
task.tactile.tacto.width=64 \
task.tactile.tacto.height=64 \
task.tactile.encoder.width=64 \
task.tactile.encoder.height=64 \
task.external_cam.external_cam=True \
task.external_cam.cam_res.w=320 \
task.external_cam.cam_res.h=180 \
task.env.tactile_wrt_force=False \
task.env.tactile_history_len=1 \
task.tactile.encoder.num_channels=3 \
task.env.compute_contact_gt=False \
train.ppo.only_contact=False \
task.env.numObsHist=1 \
task.tactile.half_image=True \
task.env.smooth_force=True \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}

