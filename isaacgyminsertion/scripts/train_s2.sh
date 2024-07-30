#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-no_phys_params}
NUM_ENVS=${4:-100}
HEADLESS=${5:-True}


array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

model_to_load=outputs/${CACHE}/stage1_nn/last.pth
student_ckpt_path=/home/${USER}/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/${CACHE}/student/checkpoints/model_last.pt
data_folder=/home/${USER}/tactile_insertion/datastore_${SEED}_${CACHE}
path_norm=/${data_folder}/normalization.pkl

echo extra "${EXTRA_ARGS}"

#CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
multi_gpu=False \
offline_train.from_offline=True \
offline_train.only_bc=True \
task.reset_at_success=True \
task.env.numEnvs=${NUM_ENVS} \
train.ppo.tactile_info=False \
train.ppo.obs_info=True \
train.ppo.img_info=True \
train.ppo.seg_info=True \
task.env.tactile=False \
task.external_cam.external_cam=True \
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True \
offline_train.gpu_ids=["${GPUS}"] \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.train.student_ckpt_path="${student_ckpt_path}" \
${EXTRA_ARGS}