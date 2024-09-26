import json
import os
import pprint
import random
import string
import tempfile
import time
from copy import copy
from datetime import datetime

import absl.flags
import numpy as np
import quaternion
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from mpl_toolkits.mplot3d import Axes3D


def save_metrics(save_path, mse, norm_mse):
    metrics_path = os.path.join(save_path, "metrics.json")
    new_metrics = {
        "mse": mse,
        "norm_mse": norm_mse,
    }

    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            try:
                metrics_list = json.load(f)
            except json.JSONDecodeError:
                metrics_list = []
        if not isinstance(metrics_list, list):
            metrics_list = [metrics_list]
        metrics_list.append(new_metrics)
    else:
        metrics_list = [new_metrics]

    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=4)


def convert_trajectory(eef_pos):
    assert eef_pos.shape[1] == 7, f"Invalid shape for eef_pos: {eef_pos.shape}"

    xyz = eef_pos[:, :3]  # (N, 3)
    quats = eef_pos[:, 3:]  # (N, 4)

    rotation_matrices = quat2R(quats)  # (N, 3, 3)

    rotation_matrices_flattened = rotation_matrices.reshape(rotation_matrices.shape[0], -1)  # (N, 9)

    eef_pos_converted = np.concatenate([xyz, rotation_matrices_flattened], axis=1)  # (N, 12)

    return eef_pos_converted


def unify(quat, eps: float = 1e-9):
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)  # Clamping to avoid division by zero
    return quat / norm


def quat2R(quat):
    assert quat.ndim in {1, 2} and quat.shape[-1] == 4, f"invalid quaternion shape: {quat.shape}"
    quat = unify(quat)
    ndim = quat.ndim
    if ndim == 1: quat = quat.reshape((1, -1))  # (1, 4)

    q0, q1, q2, q3 = quat[..., -1], quat[..., 0], quat[..., 1], quat[..., 2]  # (N, )

    R = np.stack([
        np.stack([1 - 2 * q2 ** 2 - 2 * q3 ** 2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2], axis=-1),
        np.stack([2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 ** 2 - 2 * q3 ** 2, 2 * q2 * q3 - 2 * q0 * q1], axis=-1),
        np.stack([2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 ** 2 - 2 * q2 ** 2], axis=-1)
    ], axis=1)  # (N, 3, 3)

    if ndim == 1: R = R.squeeze()

    return R


def init_plot(dpi=100):
    fig = plt.figure(figsize=(18, 6), dpi=dpi)
    ax_3d = fig.add_subplot(131, projection='3d')
    ax_2d = fig.add_subplot(132)
    ax_action_diff = fig.add_subplot(133)
    canvas = FigureCanvas(fig)
    return fig, ax_3d, ax_2d, ax_action_diff, canvas


def update_plot(ax_3d, ax_2d, ax_action_diff, eef_pos, tactile_imgs, action_gt, action_pred):
    ax_3d.cla()
    if eef_pos.ndim == 1:
        ax_3d.plot([eef_pos[0]], [eef_pos[1]], [eef_pos[2]], marker='o', label='End-Effector Position')
    else:
        ax_3d.plot(eef_pos[:, 0], eef_pos[:, 1], eef_pos[:, 2], color='b', label='GT End-Effector Path')
        pred_path = eef_pos
        pred_path[-1, 0] += action_pred[0] * 0.004
        pred_path[-1, 1] += action_pred[1] * 0.004
        pred_path[-1, 2] += action_pred[2] * 0.005

        ax_3d.plot(pred_path[:, 0], pred_path[:, 1],  pred_path[:, 2], color='r', label='Pred End-Effector Path')

    ax_3d.legend()
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Z')
    ax_3d.set_title('End-Effector Position in 3D Space')

    ax_2d.cla()
    num_camera, channel, width, height = tactile_imgs.shape
    combined_img_list = []
    for cam in range(num_camera):
        img = tactile_imgs[cam].transpose(1, 2, 0)  # Assuming channel is in the second dimension
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
        img = (img * 255).astype(np.uint8)  # Convert to 8-bit unsigned integer
        combined_img_list.append(img)
    combined_img = np.vstack(combined_img_list)
    ax_2d.imshow(combined_img)
    ax_2d.axis('off')
    ax_2d.set_title('Tactile Images')

    ax_action_diff.cla()
    timesteps = range(len(action_gt))
    width = 0.3  # Width of the bars

    # Create bar plots
    ax_action_diff.bar(timesteps, action_gt, width=width, label='Ground Truth Action', align='center')
    ax_action_diff.bar([t + width for t in timesteps], action_pred, width=width, label='Predicted Action',
                       align='center')


def render_frame(fig, canvas, ax_3d, ax_2d, ax_action_diff, eef_pos, tactile_imgs, action_gt, action_pred):
    update_plot(ax_3d, ax_2d, ax_action_diff, eef_pos, tactile_imgs, action_gt, action_pred)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    # Add action text
    font = cv2.FONT_HERSHEY_SIMPLEX
    return img


def save_args(args, output_dir):
    args_dict = vars(args)
    args_str = "\n".join(f"{key}: {value}" for key, value in args_dict.items())
    with open(os.path.join(output_dir, "args_log.txt"), "w") as file:
        file.write(args_str)
    # with open(os.path.join(output_dir, "args_log.json"), "w") as json_file:
    #     json.dump(args_dict, json_file, indent=4)


def generate_random_string(length=4, characters=string.ascii_letters + string.digits):
    """
    Generate a random string of the specified length using the given characters.

    :param length: The length of the random string (default is 12).
    :param characters: The characters to choose from when generating the string
                      (default is uppercase letters, lowercase letters, and digits).
    :return: A random string of the specified length.
    """
    return "".join(random.choice(characters) for _ in range(length))


def get_eef_delta(eef_pose, eef_pose_target):
    pos_delta = eef_pose_target[:3] - eef_pose[:3]
    # axis angle to quaternion
    ee_rot = quaternion.from_rotation_vector(eef_pose[3:])
    ee_rot_target = quaternion.from_rotation_vector(eef_pose_target[3:])
    # calculate the quaternion difference
    rot_delta = quaternion.as_rotation_vector(ee_rot_target * ee_rot.inverse())
    return np.concatenate((pos_delta, rot_delta))


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.mode = "online"
        config.project = "hato"
        config.entity = "user"
        config.output_dir = "."
        config.exp_name = str(datetime.now())[:19].replace(" ", "_")
        config.random_delay = 0.5
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)
        config.time = str(datetime.now())[:19].replace(" ", "_")

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant, prefix=None):
        self.config = self.get_default_config(config)

        for key, val in sorted(self.config.items()):
            if type(val) != str:
                continue
            new_val = _parse(val, variant)
            if val != new_val:
                logging.info(
                    "processing configs: {}: {} => {}".format(key, val, new_val)
                )
                setattr(self.config, key, new_val)

                output = flatten_config_dict(self.config, prefix=prefix)
                variant.update(output)

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()

        output = flatten_config_dict(self.config, prefix=prefix)
        variant.update(output)

        self._variant = copy(variant)

        logging.info(
            "wandb logging with hyperparameters: \n{}".format(
                pprint.pformat(
                    ["{}: {}".format(key, val) for key, val in self.variant.items()]
                )
            )
        )

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0.1, 0.1 + self.config.random_delay))

        self.run = wandb.init(
            entity=self.config.entity,
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            name=self.config.exp_name,
            anonymous=self.config.anonymous,
            monitor_gym=False,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode=self.config.mode,
        )

        self.logging_step = 0

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs, step=self.logging_step)

    def step(self):
        self.logging_step += 1

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self._variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            raise ValueError("Incorrect value type")
    return kwargs


def _parse(s, variant):
    orig_s = copy(s)
    final_s = []

    while len(s) > 0:
        indx = s.find("{")
        if indx == -1:
            final_s.append(s)
            break
        final_s.append(s[:indx])
        s = s[indx + 1:]
        indx = s.find("}")
        assert indx != -1, "can't find the matching right bracket for {}".format(orig_s)
        final_s.append(str(variant[s[:indx]]))
        s = s[indx + 1:]

    return "".join(final_s)


def get_user_flags(flags, flags_def):
    output = {}
    for key in sorted(flags_def):
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            flatten_config_dict(val, prefix=key, output=output)
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None, output=None):
    if output is None:
        output = {}
    for key, val in sorted(config.items()):
        if prefix is not None:
            next_prefix = "{}.{}".format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            flatten_config_dict(val, prefix=next_prefix, output=output)
        else:
            output[next_prefix] = val
    return output


def to_config_dict(flattened):
    config = config_dict.ConfigDict()
    for key, val in flattened.items():
        c_config = config
        ks = key.split(".")
        for k in ks[:-1]:
            if k not in c_config:
                c_config[k] = config_dict.ConfigDict()
            c_config = c_config[k]
        c_config[ks[-1]] = val
    return config.to_dict()


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}
