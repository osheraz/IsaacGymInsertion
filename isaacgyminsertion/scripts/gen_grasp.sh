#!/bin/bash

python train.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=512 \
train.ppo.priv_info=True