#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-test}
NUM_ENVS=${4:-2500}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}

EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
task.env.tactile=False \
task.env.tactile_display_viz=False \
task.env.smooth_force=True \
task.env.numObsHist=1 \
task.env.numObservations=13 \
task.env.compute_contact_gt=False \
task.env.num_points=2 \
train.ppo.only_contact=False \
task.env.numStates=15 \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
task.tactile.tacto.width=8 \
task.tactile.tacto.height=8 \
task.tactile.encoder.width=8 \
task.tactile.encoder.height=8 \
task.tactile.encoder.num_channels=1 \
task.env.tactile_history_len=1 \
train.ppo.output_name="${CACHE}" \
${EXTRA_ARGS}