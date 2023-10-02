# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

import os
import torch
import shlex
import random
import subprocess
import numpy as np


def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")


def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def git_diff_config(name):
    cmd = f'git diff --unified=0 {name}'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret


def set_np_formatting():
    """ formats numpy print """
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed

def multi_gpu_aggregate_stats(values):
    # lazy imports to avoid isaacgym errors
    import torch
    import torch.distributed as dist

    if type(values) is not list:
        single_item = True
        values = [values]
    else:
        single_item = False
    rst = []
    for v in values:
        if type(v) is list:
            v = torch.stack(v)
        if get_world_size() > 1:
            dist.all_reduce(v, op=dist.ReduceOp.SUM)
            v = v / get_world_size()
        if v.numel() == 1:
            v = v.item()
        rst.append(v)
    if single_item:
        rst = rst[0]
    return rst


def get_world_size():
    # return number of gpus
    if "LOCAL_WORLD_SIZE" in os.environ.keys():
        world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    else:
        world_size = 1
    return world_size

def add_to_fifo(tensor, x):
    """Pushes a new value to a tensor, removing the oldest one"""
    # return torch.cat((tensor[:,1:], x), dim=1)
    return torch.cat((x, tensor[:, 0:-1]), dim=1)


class AverageScalarMeter(object):
    def __init__(self, window_size):
        self.window_size = window_size
        self.current_size = 0
        self.mean = 0

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0).cpu().numpy().item()
        size = np.clip(size, 0, self.window_size)
        old_size = min(self.window_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean = 0

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean
