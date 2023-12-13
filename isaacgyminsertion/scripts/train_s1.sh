#!/bin/bash
GPUS=${1:-0}
SEED=${2:-0}
CACHE=${3:-test}
NUM_ENVS=${4:-3}
HEADLESS=${5:-False}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}

EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
task.env.tactile=False \
task.env.tactile_display_viz=False \
task.env.tactile_wrt_force=False \
task.env.smooth_force=False \
task.env.numObsHist=3 \
task.env.numObservations=24 \
task.env.compute_contact_gt=True \
task.env.numStates=7 \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.env.tactile_history_len=5 \
train.ppo.output_name="${CACHE}" \
${EXTRA_ARGS}