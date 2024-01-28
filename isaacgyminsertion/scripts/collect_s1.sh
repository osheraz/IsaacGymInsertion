#!/bin/bash
GPUS=${1:-0}
SEED=${2:-0}
# CACHE=${3:-"01-25-24/16-49-34"}
CACHE=${3:-"01-27-24/01-18-12"}
NUM_ENVS=${4:-2}   
HEADLESS=${5:-False}

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
task.env.tactile=False \
task.env.tactile_display_viz=False task.env.tactile_wrt_force=False \
task.env.smooth_force=False \
task.env.numObsHist=1 \
task.env.numObservations=18 \
task.env.compute_contact_gt=True \
task.env.numStates=7 \
train.algo=PPO \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=False \
train.ppo.tactile_info=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.env.tactile_history_len=1 \
task.data_logger.collect_data=True \
task.data_logger.total_trajectories=500 \
task.data_logger.base_folder="/common/users/dm1487/inhand_manipulation_data_store" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.algo=PPO \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}

# task.data_logger.base_folder="/common/users/dm1487/inhand_manipulation_data_store" \
#task.data_logger.base_folder="/home/osher/Desktop/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/test/data" \