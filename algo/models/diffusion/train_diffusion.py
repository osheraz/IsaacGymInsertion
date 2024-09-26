import argparse
import collections
import concurrent
import os
import pickle
import sys
import time
import re

from hydra.utils import to_absolute_path

mypath = os.path.dirname(os.path.realpath(__file__))

print("adding", mypath, "to the sys path")

sys.path.append(mypath)

import data_processing
import numpy as np
import torch
from algo.models.diffusion.dataset import Dataset, DataNormalizer
from algo.models.diffusion.diffusion_policy import DiffusionPolicy
from algo.models.diffusion.models import GaussianNoise, ImageEncoder, StateEncoder
from algo.models.transformer.point_mae import MaskedPointNetEncoder
from torch import nn
from torch.nn import ModuleList
from torchvision import transforms
from algo.models.diffusion.utils import WandBLogger, save_args, init_plot, render_frame, save_metrics
from omegaconf import DictConfig, OmegaConf
from algo.models.diffusion.utils import convert_trajectory

import imageio
import json

RT_DIM = {
    "eef_pos": 9,  # 3 + 6
    "arm_joints": 7,
    "hand_joints": 6,
    "action": 6,
    "img": (180, 320),
    "tactile": (64, 64),
    "pcl": (400, 3),
}

TEST_INPUT = {
    "eef_pos": torch.zeros(9),
    "arm_joints": torch.zeros(7),
    "hand_joints": torch.zeros(12),
    "img": torch.zeros(3, 180, 320),
    "tactile": torch.zeros(3, 64, 64),
    "pcl": torch.zeros(400, 3),
    "control": torch.zeros(6),
}


def define_transforms(channel, color_jitter, width, height, crop_width,
                      crop_height, img_patch_size, img_gaussian_noise=0.0, img_masking_prob=0.0):
    # Use color jitter to augment the image
    if color_jitter:
        if channel == 3:
            # no depth
            downsample = nn.Sequential(
                transforms.Resize(
                    (width, height),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ColorJitter(brightness=0.1),
            )

    # Not using color jitter, only downsample the image
    else:
        downsample = nn.Sequential(
            transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        )

    # Crop randomization, normalization
    transform = nn.Sequential(
        transforms.RandomCrop((crop_width, crop_height)),
    )

    # Add gaussian noise to the image
    if img_gaussian_noise > 0.0:
        transform = nn.Sequential(
            transform,
            GaussianNoise(img_gaussian_noise),
        )

    def mask_img(x):
        # Divide the image into patches and randomly mask some of them
        img_patch = x.unfold(2, img_patch_size, img_patch_size).unfold(
            3, img_patch_size, img_patch_size
        )
        mask = (
                torch.rand(
                    (
                        x.shape[0],
                        x.shape[-2] // img_patch_size,
                        x.shape[-1] // img_patch_size,
                    )
                )
                < img_masking_prob
        )
        mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
        x = x.clone()
        x.unfold(2, img_patch_size, img_patch_size).unfold(
            3, img_patch_size, img_patch_size
        )[mask] = 0
        return x

    if img_masking_prob > 0.0:
        transform = lambda x: mask_img(
            nn.Sequential(
                transforms.RandomCrop((crop_width, crop_height)),
            )(x)
        )
    # For evaluation, only center crop and normalize
    eval_transform = nn.Sequential(
        transforms.CenterCrop((crop_width, crop_height)),
    )

    return transform, downsample, eval_transform


class Agent:
    def __init__(
            self,
            output_sizes={
                "eef_pos": 64,
                "hand_joints": 64,
                "arm_joints": 64,
                "img": 128,
                "pcl": 128,
                "tactile": 128,
            },
            dropout={
                "pcl": 0.0,
                "eef_pos": 0.0,
                "hand_joints": 0.0,
                "arm_joints": 0.0,
                "img": 0.0,
                "tactile": 0.0,
            },
            action_dim=6,
            representation_type=["eef_pos", "hand_joints", "arm_joints", "img", "tactile", "pcl"],
            pred_horizon=4,
            obs_horizon=1,
            action_horizon=2,
            identity_encoder=False,
            without_sampling=False,
            predict_eef_delta=False,
            predict_pos_delta=False,
            clip_far=False,
            num_diffusion_iters=100,
            weight_decay=1e-6,
            num_workers=64,
            use_ddim=False,
            policy_dropout_rate=0.0,
            state_noise=0.0,
            img_color_jitter=False,
            img_gaussian_noise=0.0,
            img_masking_prob=0.0,
            img_patch_size=16,
            tactile_color_jitter=False,
            tactile_gaussian_noise=0.0,
            tactile_masking_prob=0.0,
            tactile_patch_size=16,
            compile_train=False,
            num_fingers=3,
            img_type='depth',
            tactile_type='rgb',
            img_height=180,
            img_width=320,
            tactile_height=64,
            tactile_width=64,
            cond_on_grasp=False,
            use_same_encoder=True
    ):

        obs_horizon = obs_horizon + 1 if cond_on_grasp else obs_horizon

        self.to_torch = lambda x: torch.from_numpy(x).float()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_horizon = pred_horizon
        self.cond_on_grasp = cond_on_grasp
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_workers = num_workers

        self.representation_type = representation_type
        self.predict_pos_delta = predict_pos_delta
        self.cpu = torch.device("cpu")

        # img
        self.img_channel = 1 if img_type == "depth" else 3
        self.clip_far = clip_far
        self.img_color_jitter = img_color_jitter
        self.img_width = img_width
        self.img_height = img_height
        self.crop_img_width, self.crop_img_height = self.img_width - 20, self.img_height - 30
        self.img_transform, self.img_downsample, self.img_eval_transform = define_transforms(self.img_channel,
                                                                                             self.img_color_jitter,
                                                                                             self.img_width,
                                                                                             self.img_height,
                                                                                             self.crop_img_width,
                                                                                             self.crop_img_height,
                                                                                             img_patch_size,
                                                                                             img_gaussian_noise,
                                                                                             img_masking_prob)
        # tactile
        self.num_fingers = num_fingers
        self.tactile_channel = 1 if tactile_type == "gray" else 3
        self.half_image = True
        self.tactile_color_jitter = tactile_color_jitter
        self.tactile_width = tactile_width // 2 if self.half_image else tactile_width
        self.tactile_height = tactile_height
        self.crop_tactile_width, self.crop_tactile_height = self.tactile_width - 5, self.tactile_height - 5
        self.tactile_transform, self.tactile_downsample, self.tactile_eval_transform = define_transforms(
            self.tactile_channel,
            self.tactile_color_jitter,
            self.tactile_width,
            self.tactile_height,
            self.crop_tactile_width,
            self.crop_tactile_height,
            tactile_patch_size,
            tactile_gaussian_noise,
            tactile_masking_prob)

        self.stats = None
        self.epi_dir = []
        self.compile_train = compile_train
        obs_dim = 0

        encoders = {}
        if "eef_pos" in self.representation_type:
            eef_dim = RT_DIM["eef_pos"]
            if identity_encoder:
                eef_encoder = nn.Identity(eef_dim)
            else:
                eef_encoder = StateEncoder(
                    input_size=eef_dim,
                    output_size=output_sizes["eef_pos"],
                    hidden_size=128,
                    dropout=dropout["eef_pos"],
                )
                eef_dim = output_sizes["eef_pos"]
            encoders["eef_pos"] = eef_encoder
            obs_dim += eef_dim

        if "hand_joints" in self.representation_type:
            hand_joints_dim = RT_DIM["hand_joints"]
            if identity_encoder:
                hand_joints_encoder = nn.Identity(hand_joints_dim)
            else:
                hand_joints_encoder = StateEncoder(
                    input_size=hand_joints_dim,
                    output_size=output_sizes["hand_joints"],
                    hidden_size=128,
                    dropout=dropout["hand_joints"],
                )
                hand_joints_dim = output_sizes["hand_joints"]
            encoders["hand_joints"] = hand_joints_encoder
            obs_dim += hand_joints_dim

        if "arm_joints" in self.representation_type:
            arm_joints_dim = RT_DIM["arm_joints"]
            if identity_encoder:
                arm_joints_encoder = nn.Identity(arm_joints_dim)
            else:
                arm_joints_encoder = StateEncoder(
                    input_size=arm_joints_dim,
                    output_size=output_sizes["arm_joints"],
                    hidden_size=128,
                    dropout=dropout["arm_joints"],
                )
                arm_joints_dim = output_sizes["arm_joints"]
            encoders["arm_joints"] = arm_joints_encoder
            obs_dim += arm_joints_dim

        if "img" in self.representation_type:
            image_encoder = ImageEncoder(output_sizes["img"], self.img_channel, dropout["img"])
            image_dim = output_sizes["img"]
            encoders["img"] = image_encoder
            obs_dim += image_dim

        if "pcl" in self.representation_type:
            pcl_encoder = MaskedPointNetEncoder(output_sizes["pcl"])
            pcl_dim = output_sizes["pcl"]
            encoders["pcl"] = pcl_encoder
            obs_dim += pcl_dim

        if "tactile" in self.representation_type:
            if use_same_encoder:
                # Use the same tactile encoder for all fingers
                tactile_encoder = ImageEncoder(output_sizes["tactile"], self.tactile_channel, dropout["tactile"])
            else:
                tactile_encoder = ModuleList(
                    [
                        # Use different tactile encoders for each finger
                        ImageEncoder(
                            output_sizes["tactile"], self.tactile_channel, dropout["tactile"]
                        )
                        for _ in range(num_fingers)
                    ]
                )

            tactile_dim = output_sizes["tactile"] * num_fingers

            encoders["tactile"] = tactile_encoder
            obs_dim += tactile_dim

        self.policy = DiffusionPolicy(
            obs_horizon=obs_horizon,
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            action_dim=action_dim,
            representation_type=representation_type,
            encoders=encoders,
            num_diffusion_iters=num_diffusion_iters,
            without_sampling=without_sampling,
            weight_decay=weight_decay,
            use_ddim=use_ddim,
            policy_dropout_rate=policy_dropout_rate,
        )

        # Compile the forward function to accelerate deployment inference
        if self.compile_train:
            self.policy.nets["noise_pred_net"].forward = torch.compile(
                self.policy.nets["noise_pred_net"].forward
            )

        self.policy.to(self.device)
        self.iter = 0
        self.obs_deque = None
        self.state_noise = state_noise

        self.predict_eef_delta = predict_eef_delta

    def _get_image_observation(self, data, done_idx):
        # allocate memory for the image
        img = torch.zeros(
            (len(data), self.img_channel, self.img_width, self.img_height),
            dtype=torch.float32,
        )

        image_size = data[0]["img"].shape
        H, W = image_size[1], image_size[2]

        # Only use rgb
        def process_rgb(d):
            rgb = d["img"].astype(np.float32)
            rgb = np.moveaxis(rgb, -1, 1)
            if H == 320 and W == 180 and self.img_color_jitter == False:
                rgb = self.img_downsample(
                    torch.tensor(rgb)
                )  # [camera_num, 3, self.img_width, self.img_height]
            else:
                rgb = torch.tensor(rgb)
            return rgb

        fn = process_rgb

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
        ) as executor:
            future_to_data = {
                executor.submit(fn, d): (d, i) for i, d in enumerate(data)
            }
            for future in concurrent.futures.as_completed(future_to_data):
                d, i = future_to_data[future]
                try:
                    img[i] = future.result()
                except Exception as exc:
                    print(f"loading image failed: {exc}")

        return img

    def _get_tactile_observation(self, data_path, done_idx):
        # allocate memory for the image
        data_path = data_path[0]
        data_len = done_idx
        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        img_folder = data_path[:-7].replace('obs', 'tactile')
        tactile_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if f.endswith('.npz')], key=natural_sort_key)
        tactile_files = tactile_files[:done_idx]

        image_size = np.load(tactile_files[0])["tactile"].shape

        img = torch.zeros(
            (data_len, self.num_fingers, self.tactile_channel, self.tactile_width, self.tactile_height),
            dtype=torch.float32,
        )

        H, W = image_size[-2], image_size[-1]

        # Only use rgb
        def process_rgb(d):
            rgb = np.load(d)['tactile']
            if H == 112 and W == 224 and self.tactile_color_jitter == False:
                rgb = self.tactile_downsample(
                    torch.tensor(rgb)
                )  # [finger_num, 3, self.img_width, self.img_height]
            else:
                rgb = torch.tensor(rgb)

            return rgb

        fn = process_rgb

        with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
        ) as executor:
            future_to_data = {
                executor.submit(fn, d): (d, i) for i, d in enumerate(tactile_files)
            }
            for future in concurrent.futures.as_completed(future_to_data):
                d, i = future_to_data[future]
                try:
                    img[i] = future.result()
                except Exception as exc:
                    print(f"loading image failed: {exc}")

        return img

    def get_observation(self, data, data_path, done_idx):

        input_data = {}
        for rt in self.representation_type:
            if rt == "img":
                input_data[rt] = self._get_image_observation(data_path, done_idx)
            if rt == "tactile":
                input_data[rt] = self._get_tactile_observation(data_path, done_idx)
            else:
                input_data[rt] = np.array([d[rt] for d in data]).squeeze()
                if rt == 'eef_pos':
                    input_data[rt] = convert_trajectory(input_data[rt])

        return input_data

    def predict(self, obs_deque: collections.deque, num_diffusion_iters=15):
        """
        data: dict
            data['image']: torch.tensor (1,5,224,224)
            data['tactile']: torch.tensor (1,6)
            data['pos']: torch.tensor (1,24)
        """
        pred = self.policy.forward(
            self.stats, obs_deque, num_diffusion_iters=num_diffusion_iters
        )
        return pred

    def get_train_loader(self, batch_size, train_test_split=0.99, eval=False):

        num_train_envs = int(len(self.epi_dir) * train_test_split)
        train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        val_idxs = np.arange(num_train_envs, len(self.epi_dir)).astype(int).tolist()
        training_files = [self.epi_dir[i] for i in train_idxs]
        eval_files = [self.epi_dir[i] for i in val_idxs]

        train_dataset = Dataset(
            traj_list=training_files if not eval else eval_files,
            representation_type=self.representation_type,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            stats=self.stats,
            img_transform=self.img_transform if not eval else self.img_eval_transform,
            tactile_transform=self.tactile_transform if not eval else self.tactile_eval_transform,
            get_img=self._get_image_observation,
            state_noise=self.state_noise if not eval else 0.0,
            img_dim=(self.crop_img_width, self.crop_img_height),
            tactile_dim=(self.crop_tactile_width, self.crop_tactile_height),
            cond_on_grasp=self.cond_on_grasp,

        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=not eval,
            pin_memory=True,
            persistent_workers=True,
        )
        self.policy.data_stat = train_dataset.stats
        return dataloader

    def train(
            self,
            traj_list,
            batch_size=8,
            epochs=100,
            save_path=None,
            save_freq=10,
            eval_freq=10,
            wandb_logger=None,
            train_test_split=0.9
    ):
        torch.cuda.empty_cache()

        self.save_path = save_path
        self.epi_dir = traj_list
        eval_traj = self.epi_dir[-1]
        self.epi_dir.remove(eval_traj)
        print("eval traj:", eval_traj)

        eval_data = self.get_eval_data(eval_traj)

        train_loader = self.get_train_loader(batch_size, train_test_split)

        self.policy.set_lr_scheduler(len(train_loader) * epochs)

        if self.stats is None:
            self.stats = train_loader.dataset.stats
            self.save_stats(save_path)

        if self.compile_train:
            train_loader.dataset.__getitem__ = torch.compile(
                train_loader.dataset.__getitem__
            )

        self.policy.train(
            epochs,
            train_loader,
            save_path=save_path,
            eval_data=eval_data,
            eval_freq=eval_freq,
            save_freq=save_freq,
            wandb_logger=wandb_logger,
        )

        self.policy.to_ema()
        self.eval(eval_traj, save_path=self.save_path)

    def get_train_action(self, data):
        act = self.get_eval_action(data)
        return np.stack(act)

    def get_eval_action(self, data):
        if self.predict_eef_delta:
            act = [d["eef_pos"] for d in data]
            act = np.diff(act, axis=0, append=act[-1:])
        elif self.predict_pos_delta:
            act = [d["arm_joints"] for d in data]
            act = np.diff(act, axis=0, append=act[-1:])
        else:
            act = [d["action"].tolist() for d in data]
        if len(data) == 1:
            act = act[0]
        return act

    def get_eval_data(self, data_path):

        print("GETTING EVAL DATA", end="\r")
        data_path = data_path if isinstance(data_path, list) else [data_path]
        data = data_processing.iterate_npz(data_path)
        B = len(data[0]['action'])

        action = self.get_eval_action(data)

        print("GETTING EVAL TRAJECTORY", end="\r")
        obs = self.get_observation(data, data_path, B)
        obs_list = []

        if "img" in self.representation_type:
            # transfer image type to float32
            obs["img"] = obs["img"].float()
            obs["img"] = self.img_eval_transform(obs["img"])

        if "tactile" in self.representation_type:
            # transfer image type to float32
            tactile = obs["tactile"].float()
            T, F, C, W, H = tactile.shape
            tactile = tactile.view(-1, C, W, H)
            tactile = self.tactile_eval_transform(tactile)
            obs["tactile"] = tactile.view(T, F, C, *tactile.shape[2:])
        for i in range(B):
            obs_list.append({rt: obs[rt][i] for rt in self.representation_type})

        return obs_list, action

    def eval(self, data_path, save_path=None, num_eval_diff_iter=15):
        print("GETTING EVAL DATA")
        eval_data = self.get_eval_data(data_path)
        obs, action_gt = eval_data
        print("EVALUATING")
        action_pred, mse, norm_mse = self.policy.eval(obs, action_gt, self.cond_on_grasp, num_eval_diff_iter)
        print("ACTION_MSE: {}, NORM_MSE: {}".format(mse, norm_mse))

        # Convert tensors to floats for JSON serialization
        mse = mse.item() if hasattr(mse, 'item') else float(mse)
        norm_mse = norm_mse.item() if hasattr(norm_mse, 'item') else float(norm_mse)

        # Create a directory to save results if not provided
        if save_path is None:
            save_path = "./outputs/{}".format(
                data_path.split("/")[-1]
                + "_"
                + time.strftime("%m%d_%H%M%S", time.localtime())
            )
            os.makedirs(save_path, exist_ok=True)

        save_metrics(save_path, mse, norm_mse)
        # Initialize plot with high DPI for better quality
        fig, ax_3d, ax_2d, ax_action_diff, canvas = init_plot(dpi=200)

        # Create frames for video
        frames = []
        for i in range(len(action_pred)):
            # Convert tactile images to frames for video
            tactile_imgs = obs[i]["tactile"].cpu().detach().numpy()
            eef_pos = obs[0]["eef_pos"] if i == 0 else np.stack([obs[j]["eef_pos"] for j in range(i)])
            current_action = action_pred[i]
            current_action_gt = action_gt[i]

            # Render the frame
            frame = render_frame(fig, canvas, ax_3d, ax_2d, ax_action_diff, eef_pos.reshape(-1, 9)[:, :3], tactile_imgs,
                                 current_action_gt, current_action)
            frames.append(frame)

        # Create a video from the frames using imageio
        video_path = os.path.join(save_path, f"{data_path}_last_eval.mp4")
        writer = imageio.get_writer(video_path, fps=15, codec='libx264', bitrate='5000k')

        for frame in frames:
            writer.append_data(frame)

        writer.close()
        print(f"Results saved in {save_path}, including a video of the tactile data.")

    def get_eval_loader(
            self, dir_path, traj_type="plain", prefix="0", batch_size=32, num_workers=16
    ):
        self.num_workers = num_workers
        print(f"GETTING EVAL DATA FROM {dir_path}")
        self.epi_dir = data_processing.get_epi_dir(dir_path, traj_type, prefix)
        eval_loader = self.get_train_loader(batch_size, "", eval=True)
        return eval_loader

    def eval_dir(self, eval_loader, num_diffusion_iters=15):
        self.policy.num_diffusion_iters = num_diffusion_iters
        with torch.no_grad():
            mse, action_mse = self.policy.eval_loader(eval_loader)
        print(f"MSE: {mse}", f"ACTION_MSE: {action_mse}")
        return mse, action_mse

    def load(self, path):
        model_path = os.path.join(path, 'last.ckpt')
        dir_path = os.path.dirname(path)
        stat_path = os.path.join(dir_path, "normalization.pkl")
        self.stats = pickle.load(open(stat_path, "rb"))
        self.policy.data_stat = self.stats
        self.policy.load(model_path)
        print("model loaded: ", model_path)
        print("stats loaded: ", stat_path)

    def save_stats(self, path):
        os.makedirs(path, exist_ok=True)
        stat_path = os.path.join(path, "normalization.pkl")
        if not os.path.exists(stat_path):
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("stats saved")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.ckpt")
        stat_path = os.path.join(path, "normalization.pkl")
        self.policy.save(model_path)

        # if stat not exist, create one
        if not os.path.exists(stat_path):
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("model saved")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


class Runner:

    def __init__(self, cfg=None):

        self.task_cfg = cfg
        cfg = cfg.diffusion_train

        from matplotlib import pyplot as plt
        self.fig = plt.figure(figsize=(20, 15))

        # wandb config
        if cfg.gpu is not None:
            torch.cuda.set_device("cuda:{}".format(cfg.gpu))
            print("Using gpu: {}".format(cfg.gpu))

        curr_time = time.strftime("%m%d_%H%M%S", time.localtime())

        if cfg.use_wandb:
            wandb_config = WandBLogger.get_default_config()
            wandb_config.entity = cfg.wandb_entity_name
            wandb_config.project = cfg.wandb_project_name
            if cfg.wandb_exp_name is not None:
                wandb_config.exp_name = cfg.wandb_exp_name + "_" + curr_time
            else:
                wandb_config.exp_name = curr_time
            wandb_logger = WandBLogger(
                config=wandb_config, variant=vars(cfg), prefix="logging"
            )
        else:
            wandb_logger = None

        agent = Agent(
            dropout={
                "eef_pos": cfg.dropout_rate,
                "hand_joints": cfg.dropout_rate,
                "arm_joints": cfg.dropout_rate,
                "img": cfg.img_dropout_rate,
                "tactile": cfg.tactile_dropout_rate,
                "pcl": cfg.pcl_dropout_rate
            },
            output_sizes={
                "eef_pos": cfg.eef_pos_output_size,
                "hand_joints": cfg.hand_joints_output_size,
                "arm_joints": cfg.arm_joints_output_size,
                "img": cfg.image_output_size,
                "tactile": cfg.tactile_output_size,
                "pcl": cfg.pcl_output_size
            },
            representation_type=cfg.representation_type,
            identity_encoder=cfg.identity_encoder,
            obs_horizon=cfg.obs_horizon,
            pred_horizon=cfg.pred_horizon,
            action_horizon=cfg.action_horizon,
            without_sampling=cfg.without_sampling,
            predict_eef_delta=cfg.predict_eef_delta,
            predict_pos_delta=cfg.predict_pos_delta,
            clip_far=cfg.clip_far,
            img_color_jitter=cfg.img_color_jitter,
            tactile_color_jitter=cfg.tactile_color_jitter,
            num_diffusion_iters=cfg.num_diffusion_iters,
            num_workers=cfg.num_workers,
            weight_decay=cfg.weight_decay,
            use_ddim=cfg.use_ddim,
            policy_dropout_rate=cfg.policy_dropout_rate,
            state_noise=cfg.state_noise,
            img_gaussian_noise=cfg.img_gaussian_noise,
            img_masking_prob=cfg.img_masking_prob,
            img_patch_size=cfg.img_patch_size,
            tactile_gaussian_noise=cfg.tactile_gaussian_noise,
            tactile_masking_prob=cfg.tactile_masking_prob,
            tactile_patch_size=cfg.tactile_patch_size,
            compile_train=cfg.compile_train,
            cond_on_grasp=cfg.cond_on_grasp,
            tactile_width=cfg.tactile_width,
            tactile_height=cfg.tactile_height
        )

        if cfg.load_path:
            agent.load(cfg.load_path)

        output_dir = f'{to_absolute_path("./outputs")}'
        model_path_suffix = f"{curr_time}"
        model_path = os.path.join(output_dir, model_path_suffix)

        if not cfg.eval:

            if cfg.model_save_path:
                model_path = cfg.model_save_path
            else:
                cfg.model_save_path = model_path

            self.save_path = model_path
            print(f"Logging everything to {model_path}")

            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            save_args(cfg, model_path)
            with open(os.path.join(model_path, f"task_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(self.task_cfg))
            with open(os.path.join(model_path, f"train_config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

            if cfg.data_folder is not None:
                data_path = cfg.data_folder
            else:
                assert 'issue with data path'

            if agent.stats is not None:
                agent.save_stats(model_path)
            else:
                from glob import glob
                print('Loading trajectories from', data_path)
                traj_list = glob(os.path.join(data_path, '*/*/obs/*.npz'))
                normalizer = DataNormalizer(cfg, traj_list)
                agent.stats = normalizer.stats
                agent.save_stats(model_path)

            print(f"using data path {data_path}")

            agent.train(
                traj_list,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                save_path=model_path,
                save_freq=cfg.save_freq,
                eval_freq=cfg.eval_freq,
                wandb_logger=wandb_logger,
            )

        else:
            agent.eval(cfg.eval_path, save_path=cfg.load_path)
