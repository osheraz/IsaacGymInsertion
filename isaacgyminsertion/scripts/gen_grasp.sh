#!/bin/bash

python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=2 \
task.env.tactile=False \
train.ppo.tactile_info=True \
train.ppo.priv_info=True