#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-w_noise}
NUM_ENVS=${4:-16}
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
task.env.tactile=True \
task.env.numStates=16 \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.env.tactile_wrt_force=True \
task.env.tactile_history_len=1 \
task.tactile.decoder.num_channels=1 \
task.env.compute_contact_gt=False \
task.env.numObsHist=10 \
task.env.numObservations=240 \
task.env.numStates=16 \
task.tactile.half_image=False \
task.env.smooth_force=False \
task.data_logger.collect_data=True \
task.data_logger.base_folder="/common/users/dm1487/inhand_manipulation_data_store" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=False \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}

# task.data_logger.base_folder="/common/users/dm1487/inhand_manipulation_data_store" \
#task.data_logger.base_folder="/home/osher/Desktop/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/test/data" \