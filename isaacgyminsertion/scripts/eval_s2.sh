#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-new_re_teacher}
NUM_ENVS=${4:-10}
HEADLESS=${5:-True}


array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

echo extra "${EXTRA_ARGS}"

model_to_load=outputs/${CACHE}/stage1_nn/last.pth
student_ckpt_path=/home/${USER}/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/${CACHE}/student/checkpoints/model_last.pt
data_folder=/home/${USER}/tactile_insertion/datastore_${SEED}_${CACHE}
path_norm=/${data_folder}/normalization.pkl

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
offline_train.from_offline=False \
offline_train.only_bc=True \
task.reset_at_success=True \
task.grasp_at_init=False \
task.reset_at_fails=True \
test=True \
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True \
train.ppo.tactile_info=True \
train.ppo.obs_info=True \
train.ppo.img_info=False \
train.ppo.seg_info=False \
train.ppo.pcl_info=True \
task.env.tactile=True \
task.external_cam.external_cam=True \
task.external_cam.depth_cam=False \
task.external_cam.seg_cam=True \
task.external_cam.pcl_cam=True \
task.data_logger.collect_data=False \
offline_train.gpu_ids=["${GPUS}"] \
offline_train.train.normalize_file="${path_norm}" \
offline_train.train.student_ckpt_path="${student_ckpt_path}" \
task.data_logger.sub_folder="datastore_${SEED}_${CACHE}" \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
${EXTRA_ARGS}