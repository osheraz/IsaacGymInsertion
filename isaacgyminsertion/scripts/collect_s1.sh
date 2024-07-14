#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-no_phys_params}
NUM_ENVS=${4:-40}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
test=True \
task.data_logger.collect_data=True \
task.env.numEnvs=${NUM_ENVS} \
task.env.tactile=False \
task.external_cam.external_cam=True \
train.ppo.priv_info=True \
task.env.numStates=14 \
task.env.numObservations=18 \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.algo=PPO \
train.ppo.extrin_adapt=False \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}

