#!/bin/bash
GPUS=${1:-0}

CUDA_VISIBLE_DEVICES=0 \
python trainV2.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=32 \
task.sim.substeps=2 \
task.grasp_logger.total_grasp=1000 \
task.env.numObsHist=10 \
task.env.numObservations=24 \
task.env.numStates=7 \
task.env.tactile=True \
task.env.tactile_display_viz=False \
task.env.tactile_wrt_force=False \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
train.ppo.tactile_info=True \
task.env.smooth_force=False \
task.tactile.half_image=False \
train.ppo.priv_info=True