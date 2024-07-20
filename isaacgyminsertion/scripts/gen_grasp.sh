#!/bin/bash

python train.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=2 \
task.sim.substeps=5 \
task.env.numObsHist=1 \
task.env.numObservations=24 \
task.env.tactile=True \
task.env.tactile_display_viz=False \
task.env.tactile_wrt_force=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
train.ppo.tactile_info=True \
task.env.smooth_force=False \
task.tactile.half_image=False \
train.ppo.priv_info=True