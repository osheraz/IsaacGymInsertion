#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-s2_w_noise}
NUM_ENVS=${4:-12}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
test=True \
offline_training=False \
offline_training_w_env=False \
train.algo=ExtrinsicAdapt \
task.env.tactile=True \
task.env.tactile_history_len=5 \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.env.tactile_wrt_force=True \
task.tactile.decoder.num_channels=1 \
task.tactile.half_image=True \
task.env.smooth_force=True \
train.ppo.tactile_info=True \
train.ppo.obs_info=True \
task.data_logger.collect_data=False \
task.data_logger.base_folder="/common/users/dm1487/inhand_manipulation_data_store" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage2_nn/last.pth
