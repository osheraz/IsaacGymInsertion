#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-contact}
NUM_ENVS=${4:-100}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

C=outputs/${CACHE}/stage1_nn/last.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
test=True \
task.env.numEnvs=${NUM_ENVS} \
task.env.numStates=7 \
task.env.tactile_history_len=1 \
task.env.compute_contact_gt=True \
train.ppo.only_contact=True \
task.env.numObsHist=1 \
task.env.numObservations=18 \
train.algo=PPO \
task.env.tactile=False \
task.env.smooth_force=True \
task.tactile.tacto.width=64 \
task.tactile.tacto.height=64 \
task.tactile.decoder.width=64 \
task.tactile.decoder.height=64 \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}

# task.data_logger.base_folder="/common/users/oa348/inhand_manipulation_data_store" \
#task.data_logger.base_folder="/home/osher/Desktop/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/test/data" \