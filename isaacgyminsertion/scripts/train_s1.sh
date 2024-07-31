#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-test}
NUM_ENVS=${4:-1500}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}

EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

#torchrun --standalone --nnodes=1 --nproc_per_node=3 \
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
multi_gpu=False \
restore_train=False \
task.grasp_at_init=False \
task.env.numEnvs=${NUM_ENVS} \
task.env.compute_contact_gt=False \
train.ppo.only_contact=False \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.output_name="${CACHE}" \
${EXTRA_ARGS}