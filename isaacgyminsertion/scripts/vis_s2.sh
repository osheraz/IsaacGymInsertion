#!/bin/bash
CACHE=$1
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
train.algo=ExtrinsicAdapt \
task.env.tactile=True \
train.ppo.tactile_info=True
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage2_nn/last.pth
