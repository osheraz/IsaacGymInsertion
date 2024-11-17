<div align="center">
<h2>Visuotactile-Based Learning for Insertion with Compliant Hands</h2>


<img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="https://opensource.org/license/apache-2-0" />
<img src="https://img.shields.io/github/last-commit/osheraz/IsaacGymInsertion?style&color=5D6D7E" alt="git-last-commit" />

</div>
<br><br>

<div align="center">
  <img src="visoutactile.gif"
  width="80%">
</div>
<br>

```
@misc{azulay2024visuotactilebasedlearninginsertioncompliant,
      title={Visuotactile-Based Learning for Insertion with Compliant Hands}, 
      author={Osher Azulay and Dhruv Metha Ramesh and Nimrod Curtis and Avishai Sintov},
      year={2024},
      eprint={2411.06408},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2411.06408}, 
}
```

<span style="color:blue;">**Note:** I’ll be updating the code and restructuring the repository soon to provide a cleaner, more organized code base.</span>


---

- [Overview](#overview)
- [Getting Started](#getting-started)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
- [Usage](#usage)
    - [Step 1: Teacher Policy training](#step-1--teacher-policy-training)
    - [Step 2: Evaluate the teacher policy](#step-2--evaluate-the-teacher-policy)
    - [Step 3: Visuotactile Student distillation](#step-3--visuotactile-student-distillation)
    - [Step 4: Step 4: Evaluate the student](#step-4--evaluate-the-student)
    - [Deploy](#deploy)
- [License](#license)


---

## Overview

This repo is a code implementation of the following paper: [Visuotactile-Based Learning for Insertion with Compliant Hands
](https://arxiv.org/abs/2411.06408). 

---

## Getting Started

#### Dependencies

Project was tested on:
- ubuntu >=18.04
- python >= 3.8
- cuda >=11.7
- built on ROS melodic\noetic
  

#### Installation

1. Install IsaacGym:

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. 

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```

2. Clone and install this repo:
```sh
cd isaacgym/python
git clone https://github.com/osheraz/IsaacGymInsertion
cd IsaacGymInsertion && pip install -e .
```

---
## Usage

### Brief
- Train a teacher policy using privliged information with RL
- Train a student policy using visual and\or tactile information
- Deploy on real-robot

**Note:** All configs, logs, and model weights are saved in a folder within ```outputs/```.

As noted before, this repository includes implementations for tactile sensing (built on top of [TACTO](https://github.com/facebookresearch/tacto/tree/main)) and point cloud observation.

To visualize each input, refer to the `cfg/task/FactoryTaskInsertionTactile.yaml` file and set `True` for each input you would like to enable.

To visualize tactile inputs, set the following parameters:
```
tactile=True  
tactile_display_viz=True
```
You can adjust the tactile parameters in the file `allsight\conf\sensor\config_allsight_white.yml`.

To visualize depth, point cloud, or masks, set the following parameters:
```
external_cam=True  
pcl_cam\depth_cam\seg_cam=True 
display=True
```

### Training

#### Step 1: Teacher Policy training
```bash
cd isaacgyminsertion
scripts/train_s1.sh
```
#### Step 2: Evaluate the teacher policy
```bash
scripts/eval_s1.sh
```

#### Step 3: Visuotactile Student distillation
Modify ```scripts/train_s2.sh``` given which student policy you want to use:
```
# for segmented-pcl:
train.ppo.obs_info=True \ train.ppo.img_info=False \ train.ppo.seg_info=False \ train.ppo.pcl_info=True \ train.ppo.tactile_info=False \ task.env.tactile=False \ task.external_cam.external_cam=True \ task.external_cam.depth_cam=False \ task.external_cam.seg_cam=True \ task.external_cam.pcl_cam=True \```
```
```
# for segmented-depth:
train.ppo.obs_info=True \ train.ppo.img_info=True \ train.ppo.seg_info=True \ train.ppo.pcl_info=False \ train.ppo.tactile_info=False \ task.env.tactile=False \ task.external_cam.external_cam=True \ task.external_cam.depth_cam=True \ task.external_cam.seg_cam=True \ task.external_cam.pcl_cam=False \```
```
```
# for visualtactile (pcl+tactile):
train.ppo.obs_info=True \ train.ppo.img_info=False \ train.ppo.seg_info=False \ train.ppo.pcl_info=True \ train.ppo.tactile_info=True \ task.env.tactile=True \ task.external_cam.external_cam=True \ task.external_cam.depth_cam=False \ task.external_cam.seg_cam=True \ task.external_cam.pcl_cam=True \```
```

Next, train the policy:
```bash
scripts/train_s2.sh
```
#### Step 4: Evaluate the student 
Evaluate with same arguments in eval_s2.sh:
```bash
scripts/eval_s2.sh
```

---

## Deploy

See - [deploy_instructions](https://github.com/osheraz/IsaacGymInsertion/blob/main/algo/deploy/README.md)


---

## Acknowledgements

- [isaacgym](https://developer.nvidia.com/isaac-gym)
- [IsaacGymEnvs-Factory](https://github.com/isaac-sim/IsaacGymEnvs)
- [dexenv](https://github.com/Improbable-AI/dexenv) 
- [hora](https://github.com/haozhiqi/hora)

---

## License

This repository is licensed under the Apache [License](/LICENSE). Feel free to use, modify, and distribute the code as per the terms of this license.


---

[↑ Return](#Top)

