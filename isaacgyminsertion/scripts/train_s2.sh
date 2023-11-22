#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-"11-21-23/18-16-32"}
NUM_ENVS=${4:-2}
HEADLESS=${5:-True}


array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

C=outputs/${CACHE}/stage1_nn/last.pth

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=False \
offline_training_w_env=False \
task.env.tactile=True \
task.env.numObsHist=5 \
task.env.numObservations=24 \
task.env.numStates=16 \
task.env.tactile_history_len=5 \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.env.tactile_wrt_force=True \
task.tactile.decoder.num_channels=1 \
task.tactile.sim2real=False \
task.tactile.half_image=True \
task.env.compute_contact_gt=False \
task.env.smooth_force=True \
train.ppo.tactile_info=True \
train.ppo.obs_info=False \
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}