#!/bin/bash

python train.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=5 \
train.ppo.priv_info=True