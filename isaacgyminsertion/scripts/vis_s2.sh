#!/bin/bash
GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-11-22-23/12-21-35}
NUM_ENVS=${4:-2}

CUDA_VISIBLE_DEVICES=${GPUS} \
python trainV2.py task=FactoryTaskInsertionTactile headless=True pipeline=gpu seed=${SEED} \
task.env.numEnvs=${NUM_ENVS} test=True \
train.algo=ExtrinsicAdapt \
task.env.numObsHist=5 \
task.env.numObservations=24 \
task.env.numStates=16 \
task.env.tactile=True \
task.env.tactile_history_len=5 \
task.tactile.tacto.width=224 \
task.tactile.tacto.height=224 \
task.tactile.decoder.width=224 \
task.tactile.decoder.height=224 \
task.tactile.decoder.num_channels=1 \
task.env.compute_contact_gt=False \
task.env.tactile_wrt_force=True \
task.env.smooth_force=True \
task.tactile.sim2real=False \
task.tactile.half_image=True \
train.ppo.tactile_info=True \
train.ppo.obs_info=False \
task.env.tactile_display_viz=False \
train.ppo.priv_info=True \
train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage2_nn/last.pth
