#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

C=outputs/10-18-23/16-49-49/stage1_nn/last.pth

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=True seed=${SEED} \
task.env.numEnvs=32 \
task.env.tactile=True \
task.env.tactile_history_len=3 \
task.tactile.decoder.num_channels=3 \
train.ppo.tactile_info=True \
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}