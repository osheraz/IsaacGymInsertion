#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-test}
NUM_ENVS=${4:-4096}
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
task.env.numObsHist=5 \
task.env.compute_contact_gt=True \
task.env.numStates=1 \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
train.ppo.output_name="${CACHE}" \
${EXTRA_ARGS}