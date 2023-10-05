#!/bin/bash
CONDA_DIR="$(conda info --base)"
source "${CONDA_DIR}/etc/profile.d/conda.sh"
# conda init bash
conda activate base
conda activate "/common/users/dm1487/envs/inhand_py38"
export LD_LIBRARY_PATH=/common/users/dm1487/envs/inhand_py38/lib
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json