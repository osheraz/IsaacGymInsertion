#!/bin/bash
CACHE=$1
python trainV2.py task=FactoryTaskInsertionTactile headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
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
train.ppo.priv_info=True train.ppo.extrin_adapt=True \
train.ppo.output_name="${CACHE}" \
checkpoint=outputs/"${CACHE}"/stage2_nn/last.pth
