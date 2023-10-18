#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=True seed=${SEED} \
task.env.numEnvs=32 \
env.env.tactile=True \
train.ppo.tactile_info=True
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage1_nn/best.pth \
${EXTRA_ARGS}
