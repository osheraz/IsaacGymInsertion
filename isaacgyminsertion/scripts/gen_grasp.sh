#!/bin/bash

python trainV2.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=32 \
task.sim.substeps=5 \
task.env.tactile=True \
task.env.tactile_display_viz=False \
task.env.tactile_wrt_force=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
train.ppo.tactile_info=True \
task.env.smooth_force=False \
task.tactile.half_image=False \
train.ppo.priv_info=True