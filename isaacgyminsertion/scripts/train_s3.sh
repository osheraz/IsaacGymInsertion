#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-new_teacher}
NUM_ENVS=${4:-10}
HEADLESS=${5:-False}


array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

model_to_load=outputs/${CACHE}/stage1_nn/last.pth
student_ckpt_path=/home/${USER}/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/${CACHE}/student/checkpoints/model_last.pt
data_folder=/home/${USER}/tactile_insertion/datastore_${SEED}_${CACHE}
path_norm=/${data_folder}/normalization.pkl

echo extra "${EXTRA_ARGS}"

#CUDA_VISIBLE_DEVICES=${GPUS} \
#torchrun --standalone --nnodes=1 --nproc_per_node=4 \
python train.py task=FactoryTaskInsertionTactile headless=${HEADLESS} seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
multi_gpu=False \
restore_train=True \
restore_student=True \
phase=3 \
task.rand_inits=True \
offline_train.from_offline=False \
task.grasp_at_init=False \
offline_train.only_bc=True \
task.reset_at_success=True \
task.reset_at_fails=True \
train.ppo.obs_info=True \
train.ppo.img_info=False \
train.ppo.seg_info=False \
train.ppo.pcl_info=True \
train.ppo.tactile_info=True \
task.env.tactile=True \
task.external_cam.external_cam=True \
task.external_cam.depth_cam=False \
task.external_cam.seg_cam=True \
task.external_cam.pcl_cam=True \
train.algo=ExtrinsicAdapt \
train.ppo.priv_info=True \
offline_train.gpu_ids=[0] \
train.ppo.output_name="${CACHE}" \
checkpoint="${model_to_load}" \
offline_train.train.normalize_file="${path_norm}" \
offline_train.train.student_ckpt_path="${student_ckpt_path}" \
${EXTRA_ARGS}