#!/bin/bash

python trainV2.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=16 \
task.env.tactile=False \
train.ppo.tactile_info=True \
train.ppo.priv_info=True