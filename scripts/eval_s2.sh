#!/bin/bash
GPUS=$1
CACHE=$2
C=outputs/"${CACHE}"/stage2_nn/last.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=True \
task.env.numEnvs=32 test=True task.on_evaluation=True \
train.algo=ExtrinsicAdapt \
env.env.tactile=True \
train.ppo.tactile_info=True
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}"
