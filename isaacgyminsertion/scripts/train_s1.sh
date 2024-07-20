#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-test}
NUM_ENVS=${4:-3096}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}

EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

#CUDA_VISIBLE_DEVICES=${GPUS} \
torchrun --standalone --nnodes=1 --nproc_per_node=3 \
train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
multi_gpu=True \
task.env.numEnvs=${NUM_ENVS} \
task.env.numObsHist=1 \
task.env.numObservations=18 \
task.env.numStates=14 \
task.env.compute_contact_gt=False \
train.ppo.only_contact=False \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
task.external_cam.external_cam=False \
train.ppo.output_name="${CACHE}" \
${EXTRA_ARGS}