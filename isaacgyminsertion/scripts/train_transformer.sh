#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-delta_pos}
NUM_ENVS=${4:-4096}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

C=outputs/${CACHE}/stage1_nn/last.pth
D=outputs/${CACHE}/data
D=/common/users/oa348/inhand_manipulation_data_store/datastore_${SEED}_${CACHE}
DD=outputs/${CACHE}
N=/common/home/oa348/Downloads/isaacgym/python/IsaacGymInsertion/algo/models/transformer/normalization.pkl

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=True \
test=True \
offline_training_w_env=True \
offline_train.train.action_regularization=False \
offline_train.model.transformer.full_sequence=False \
offline_train.model.transformer.sequence_length=32 \
task.env.tactile=True \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.env.tactile_history_len=1 \
task.env.tactile_wrt_force=True \
offline_train.data_folder="${D}" \
offline_train.output_dir="${DD}" \
offline_train.train.normalize_file="${N}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}