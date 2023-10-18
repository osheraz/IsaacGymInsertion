#!/bin/bash

GPUS=$1
SCALE=$2
CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=FactoryTaskGraspTactile headless=True pipeline=cpu \
task.env.numEnvs=32 test=True \
env.env.tactile=True \
train.ppo.tactile_info=True
train.ppo.priv_info=True
