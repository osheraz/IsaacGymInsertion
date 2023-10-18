#!/bin/bash
CACHE=$1


python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
train.algo=PPO \
train.ppo.priv_info=True \
task.env.tactile=True \
task.env.tactile_display_viz=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage1_nn/best.pth
