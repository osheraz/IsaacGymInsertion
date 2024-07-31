#!/bin/bash

python train.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu \
task.env.numEnvs=2000 \
train.ppo.priv_info=True