# AllSight_Sim : TACTO implementation for the AllSight tactile Sensor

## Overview
This package offers a simulation environment for a small 3D structured vision-based finger sensor called AllSight. It includes the integration of the AllSight model with the [TACTO](https://github.com/facebookresearch/tacto) API simulation, allowing for data collection and simulation of robotic in-hand manipulation scenarios. For more information refer to the corresponding [paper](https://arxiv.org/abs/2307.02928)

---
## Installation

The code has been tested on:
- Ubuntu 18 / 20 
- python >= 3.6

Clone the repository:

```bash
git clone git@github.com:osheraz/allsight_sim.git
cd allsight_sim
```

Install the dependencies:

```bash
pip install -r requirements/requirements.txt
```

---

## Usage 

- [experiments/00_demo_pybullet_allsight.py](experiments/00_demo_pybullet_allsight.py): rendering RGB and Depth readings with Allsight sensor.

![](website/gif/allsight_demo.gif)

![](website/gif/allsight_demo_rect.gif)


- [experiments/01_collect_data_sim.py](experiments/01_collect_data_sim.py): rendering RGB and Depth readings with Allsight sensor.

![](website/gif/allsight_collect_data.gif)


NOTE: Adjust simulation parameters on [experiments/conf](experiments/conf) folder. 

---

## License

This project is licensed under MIT license, as found in the [LICENSE](LICENSE) file.






