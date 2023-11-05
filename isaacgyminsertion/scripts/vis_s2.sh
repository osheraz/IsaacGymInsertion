#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-215}
NUM_ENVS=${4:-2}

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} \
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
task.env.tactile_display_viz=True \
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage2_nn/last.pth
