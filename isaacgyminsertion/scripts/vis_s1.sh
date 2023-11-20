#!/bin/bash
CACHE=$1

C=outputs/test/last.pth
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=2 test=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.env.tactile=True \
task.env.tactile_display_viz=True \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}"