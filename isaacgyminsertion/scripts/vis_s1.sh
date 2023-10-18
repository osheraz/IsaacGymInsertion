#!/bin/bash
CACHE=$1

C=outputs/10-18-23/20-29-00/stage1_nn/ep_1500_step_0036M_reward_123.19.pth
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
tactile.tacto.width=224 \
tactile.tacto.height=224 \
task.env.tactile=True \
task.env.tactile_display_viz=True \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}"