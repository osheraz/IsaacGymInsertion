#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}
C=outputs/10-18-23/11-31-58/stage1_nn/ep_1500_step_0036M_reward_123.19.pth

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=True seed=${SEED} \
task.env.numEnvs=32 \
task.env.tactile=True \
train.ppo.tactile_info=True \
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}
