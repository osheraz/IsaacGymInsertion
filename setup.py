"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym==0.23.1",
    "torch",
    "omegaconf",
    "termcolor",
    "jinja2",
    "hydra-core>=1.1",
    "rl-games>=1.6.0",
    "pyvirtualdisplay",
    "matplotlib",
    "numpy==1.23.5",
    "scipy",
    "opencv-python",
    "trimesh",
    "imageio",
    "deepdish",
    "efficientnet-pytorch",
    "hyperopt",
    "wandb",
    "tensorboard",
    "pytorch3d",
    "pyrender",
    "open3d",
    "warmup_scheduler"
    ]



# Installation operation
setup(
    name="isaacgyminsertion",
    author="Osher Azulay",
    version="1.4.0",
    description="Tactile insertion with isaac gym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

# EOF
