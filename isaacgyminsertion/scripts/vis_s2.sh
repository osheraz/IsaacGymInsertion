#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-11-21-23}
NUM_ENVS=${4:-1}

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} test=True \
train.algo=ExtrinsicAdapt \
task.env.tactile=True \
task.env.tactile_history_len=5 \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.env.compute_contact_gt=True \
task.env.tactile_wrt_force=True \
task.env.smooth_force=True \
task.tactile.sim2real=False \
task.tactile.half_image=True \
train.ppo.tactile_info=True \
train.ppo.obs_info=True \
task.env.tactile_display_viz=True \
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/22-32-19/stage2_nn/110.00k.pth
