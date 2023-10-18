#!/bin/bash
GPUS=$1
CACHE=$2
C=outputs/"${CACHE}"/stage1_nn/best.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=True \
task.env.numEnvs=1000 test=True task.on_evaluation=True \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}"