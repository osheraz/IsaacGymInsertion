#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-gt_test}
NUM_ENVS=${4:-12}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
test=False \
task.collect_rotate=True \
task.data_logger.collect_data=True \
task.rl.max_episode_length=350 \
task.env.numEnvs=${NUM_ENVS} \
task.env.tactile=True \
task.env.numStates=7 \
task.env.numObservations=18 \
task.external_cam.external_cam=False \
task.env.numObsHist=1 \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}

