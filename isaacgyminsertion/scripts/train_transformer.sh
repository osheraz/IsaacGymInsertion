#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-"12-10-23/15-01-17"}
NUM_ENVS=${4:-4}
HEADLESS=${5:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

C=outputs/${CACHE}/stage1_nn/last.pth
data=/common/users/dm1487/inhand_manipulation_data_store
data_folder=/common/users/dm1487/inhand_manipulation_data_store/datastore_${SEED}_${CACHE}
output_dir=outputs/${CACHE}
path_norm=/common/home/dm1487/Downloads/isaacgym/python/IsaacGymInsertion/algo/models/transformer/normalization.pkl

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_training=False \
test=True \
offline_training_w_env=True \
offline_train.train.action_regularization=False \
offline_train.model.transformer.full_sequence=True \
offline_train.model.transformer.sequence_length=16 \
offline_train.train.load_checkpoint=False \
offline_train.train.ckpt_path="${DD}/16/model_9.pt" \
task.env.tactile=True \
task.env.tactile_display_viz=False \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.tactile.half_image=True \
task.env.smooth_force=False \
task.env.tactile_history_len=1 \
task.env.tactile_wrt_force=False \
task.env.compute_contact_gt=True \
train.ppo.priv_info=True \
task.env.numObsHist=3 \
task.env.numObservations=24 \
task.env.numStates=7 \
task.data_logger.base_folder="${data}" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}_4" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.data_folder="${data_folder}" \
offline_train.output_dir="${output_dir}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${C}" \
${EXTRA_ARGS}

