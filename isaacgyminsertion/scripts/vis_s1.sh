#!/bin/bash
CACHE=$1

C=outputs/12-05-23/22-16-14/stage1_nn/last.pth
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu test=True \
task.env.numEnvs=2 \
task.env.tactile=True \
task.env.tactile_display_viz=True \
task.env.tactile_wrt_force=False \
task.env.smooth_force=False \
task.env.numObsHist=25 \
task.env.numObservations=31 \
task.env.compute_contact_gt=False \
task.env.numStates=7 \
train.algo=PPO \
train.ppo.priv_info=False \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=3 \
task.env.tactile_history_len=1 \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}"