#!/bin/bash

python trainV2.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=16 \
task.env.tactile=True \
train.ppo.tactile_info=True \
task.tactile.half_image=False \
train.ppo.priv_info=True