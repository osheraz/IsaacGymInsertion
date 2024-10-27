#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-new_re_teacher}
NUM_ENVS=${4:-6}
HEADLESS=${5:-False}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

C=outputs/${CACHE}/stage1_nn/last.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
test=True \
task.grasp_at_init=False \
task.reset_at_fails=True \
task.reset_at_success=False \
task.env.numEnvs=${NUM_ENVS} \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}