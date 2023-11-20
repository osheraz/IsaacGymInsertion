#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-delta_pos}
NUM_ENVS=${4:-1}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

C=outputs/${CACHE}/stage1_nn/last.pth
# D=outputs/${CACHE}/data
D=/common/users/dm1487/inhand_manipulation_data_store/datastore_${SEED}_${CACHE}
DD=outputs/${CACHE}
N=/common/home/dm1487/Downloads/isaacgym/python/IsaacGymInsertion/algo/models/transformer/normalization.pkl

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=False \
test=True \
offline_training_w_env=True \
offline_train.train.action_regularization=False \
offline_train.model.transformer.full_sequence=False \
offline_train.model.transformer.sequence_length=16 \
offline_train.train.load_checkpoint=False \
offline_train.train.ckpt_path="${DD}/16/model_9.pt" \
offline_train.train.only_test=True \
offline_train.train.only_validate=False \
task.env.tactile=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.env.tactile_history_len=1 \
task.env.tactile_wrt_force=True \
task.data_logger.base_folder="/common/users/dm1487/inhand_manipulation_data_store" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
offline_train.data_folder="${D}" \
offline_train.output_dir="${DD}" \
offline_train.train.normalize_file="${N}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}