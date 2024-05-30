import argparse
import collections
import concurrent
import os
import pickle
import sys
import time

mypath = os.path.dirname(os.path.realpath(__file__))

print("adding", mypath, "to the sys path")

sys.path.append(mypath)

import data_processing
import numpy as np
import torch
from dataset import Dataset, DataNormalizer
from diffusion_policy import DiffusionPolicy
from models import GaussianNoise, ImageEncoder, StateEncoder
from torch import nn
from torch.nn import ModuleList
from torchvision import transforms
from utils import WandBLogger, generate_random_string, get_eef_delta, save_args

LEFT_UR_IDX = list(range(0, 6))
RIGHT_UR_IDX = list(range(12, 18))
LEFT_HAND_IDX = list(range(6, 12))
RIGHT_HAND_IDX = list(range(18, 24))
HAND_IDX = LEFT_HAND_IDX + RIGHT_HAND_IDX

RT_DIM = {
    "eef_pos": 12,  # 3 + 9
    "arm_joints": 7,
    "hand_joints": 6,
    "action": 6,
    "img": (180, 320),
    "tactile": (224, 224),
}

TEST_INPUT = {
    "eef_pos": torch.zeros(12),
    "arm_joints": torch.zeros(7),
    "hand_joints": torch.zeros(12),
    "img": torch.zeros(3, 180, 320),
    "tactile": torch.zeros(3, 224, 224),
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
        else:
            # with depth, only jitter the rgb part
            downsample = lambda x: transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )(
                torch.concat(
                    [transforms.ColorJitter(brightness=0.1)(x[:, :3]), x[:, 3:]],
                    axis=1,
                )
            )

    # Not using color jitter, only downsample the image
    else:
        downsample = nn.Sequential(
            transforms.Resize(
                (width, height),
                interpolation=transforms.InterpolationMode.BILINEAR,
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
                "tactile": 128,
            },
            dropout={
                "eef_pos": 0.0,
                "hand_joints": 0.0,
                "arm_joints": 0.0,
                "img": 0.0,
                "tactile": 0.0,
            },
            action_dim=6,
            representation_type=["eef_pos", "hand_joints", "arm_joints", "img", "tactile"],
            pred_horizon=4,
            obs_horizon=1,
            action_horizon=2,
            identity_encoder=False,
            without_sampling=False,
            predict_eef_delta=False,
            predict_pos_delta=False,
            clip_far=False,
            num_diffusion_iters=100,
            load_img=False,
            weight_decay=1e-6,
            num_workers=64,
            use_ddim=False,
            binarize_tactile=False,
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
            tactile_height=224,
            tactile_width=224,
    ):
        self.to_torch = lambda x: torch.from_numpy(x).float()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_workers = num_workers
        self.binarize_tactile = binarize_tactile

        self.representation_type = representation_type
        self.predict_pos_delta = predict_pos_delta
        self.cpu = torch.device("cpu")

        # img
        self.img_channel = 1 if img_type == "depth" else 3
        self.clip_far = clip_far
        self.load_img = load_img
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
        self.crop_tactile_width, self.crop_tactile_height = self.tactile_width - 20, self.tactile_height - 30
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

        if "tactile" in self.representation_type:
            tactile_encoder = ModuleList(
                [
                    # Use different image encoders for each finger
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
            binarize_tactile=self.binarize_tactile,
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
        self.threshold = 8000
        self.state_noise = state_noise

        self.predict_eef_delta = predict_eef_delta

    def _get_image_observation(self, data):
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

    def _get_tactile_observation(self, data):
        # allocate memory for the image
        img = torch.zeros(
            (len(data), 3, self.tactile_channel, self.tactile_width, self.tactile_height),
            dtype=torch.float32,
        )

        image_size = data[0]["tactile"].shape
        H, W = image_size[1], image_size[2]

        # Only use rgb
        def process_rgb(d):
            rgb = d["tactile"].astype(np.float32)
            rgb = np.moveaxis(rgb, -1, 1)
            if H == 224 and W == 224 and self.tactile_color_jitter == False:
                rgb = self.tactile_downsample(
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

    def get_observation(self, data, load_img=False):
        input_data = {}
        for rt in self.representation_type:
            if rt == "img":
                if load_img:
                    # Load all the image to the memory
                    input_data[rt] = self._get_image_observation(data)
                else:
                    # Only keep the file path, and load the image while training
                    input_data[rt] = np.stack([d["file_path"] for d in data])
            if rt == "tactile":
                if load_img:
                    # Load all the image to the memory
                    input_data[rt] = self._get_tactile_observation(data)
                else:
                    # Only keep the file path, and load the image while training
                    input_data[rt] = np.stack([d["file_path"] for d in data])
            elif rt == "eef_pos":
                input_data[rt] = np.stack([d["eef_pos"] for d in data])
            elif rt == "hand_joints":
                input_data[rt] = np.stack(
                    [d["hand_joints"] for d in data]
                )
            elif rt == "arm_joints":
                input_data[rt] = np.stack([d["arm_joints"] for d in data])
            else:
                input_data[rt] = np.stack([d[rt] for d in data])
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

    def _get_init_train_data(self, total_data_points, img_memmap_loader_path="", tactile_memmap_loader_path=""):

        init_data = {}
        for rt in self.representation_type + ["action"]:
            if rt == "img":
                if img_memmap_loader_path == "":
                    # Not using memmap
                    if self.load_img:
                        # Declear the memory to store the image
                        init_data[rt] = np.empty(
                            (
                                total_data_points,
                                self.img_channel,
                                self.img_width,
                                self.img_height,
                            ),
                            dtype=np.uint16,
                        )
                    else:
                        # Only store the file path
                        assert '?'
                        init_data[rt] = np.empty((total_data_points,), dtype=object)

            if rt == "tactile":
                if tactile_memmap_loader_path == "":
                    # Not using memmap
                    if self.load_img:
                        # Declear the memory to store the image
                        init_data[rt] = np.empty(
                            (
                                total_data_points,
                                self.tactile_channel,
                                self.tactile_width,
                                self.tactile_height,
                            ),
                            dtype=np.uint16,
                        )
                    else:
                        # Only store the file path
                        assert '?'
                        init_data[rt] = np.empty((total_data_points,), dtype=object)
            else:
                # Other representation types
                init_data[rt] = np.zeros((total_data_points, RT_DIM[rt]), dtype=np.float32)
        return init_data

    def get_train_loader_old(self, batch_size, img_memmap_loader_path="", tactile_memmap_loader_path="", eval=False):

        current_epi_dir = self.epi_dir

        total_data_points = sum([len(os.listdir(epi)) for epi in current_epi_dir])
        train_data = {"data": {}, "meta": {}}

        # allocate data structs
        train_data["data"] = self._get_init_train_data(
            total_data_points, img_memmap_loader_path=img_memmap_loader_path,
            tactile_memmap_loader_path=tactile_memmap_loader_path
        )

        cache_memmap = False
        img_shape = (total_data_points, self.img_channel, self.img_width, self.img_height)
        tactile_shape = (total_data_points, self.num_fingers, self.tactile_channel, self.tactile_width, self.tactile_height)
        # todo understand this memmap thing
        if img_memmap_loader_path != "":
            # Use memmap to load imgs
            if os.path.exists(img_memmap_loader_path):
                # Load the existing memmap file
                fp_image = np.memmap(
                    img_memmap_loader_path, dtype=np.uint16, mode="r", shape=img_shape
                )
                train_data["data"]["img"] = fp_image
            else:
                # Create a new memmap file
                cache_memmap = True
                fp_image = np.memmap(
                    img_memmap_loader_path, dtype=np.uint16, mode="w+", shape=img_shape
                )
        else:
            pass

        if tactile_memmap_loader_path != "":
            # Use memmap to load tactiles
            if os.path.exists(tactile_memmap_loader_path):
                # Load the existing memmap file
                fp_tactile = np.memmap(
                    tactile_memmap_loader_path, dtype=np.uint16, mode="r", shape=tactile_shape
                )
                train_data["data"]["tactile"] = fp_tactile
            else:
                # Create a new memmap file
                cache_memmap = True
                fp_tactile = np.memmap(
                    tactile_memmap_loader_path, dtype=np.uint16, mode="w+", shape=tactile_shape
                )
        else:
            pass

        data_index = 0

        print("Loading training data")
        for i, epi in enumerate(current_epi_dir):
            print("loading {}-th data from {}\r".format(i, epi), end="")
            # data = data_processing.iterate(epi, load_img=self.load_img)
            data = np.load(epi)
            done = data["done"]
            done_idx = done.nonzero()[0][-1]

            if len(data) == 0:
                continue

            data_length = done_idx

            # images - (N, self.img_channel, self.img_width, self.img_height)
            # tactile - (N, num_fingers, self.tactile_channel, self.tactile_width, self.tactile_height)

            obs = self.get_observation(data, self.load_img or cache_memmap)

            # obs space
            for rt in self.representation_type:
                if rt == "img":
                    if cache_memmap:
                        fp_image[data_index: data_index + data_length] = obs[rt]
                        fp_image.flush()
                    elif img_memmap_loader_path == "":
                        train_data["data"][rt][
                        data_index: data_index + data_length
                        ] = obs[rt]
                if rt == "tactile":
                    if cache_memmap:
                        fp_tactile[data_index: data_index + data_length] = obs[rt]
                        fp_tactile.flush()
                    elif tactile_memmap_loader_path == "":
                        train_data["data"][rt][
                        data_index: data_index + data_length
                        ] = obs[rt]
                else:
                    train_data["data"][rt][data_index: data_index + data_length] = obs[
                        rt
                    ]

            # action space
            train_data["data"]["action"][data_index: data_index + data_length] = (
                self.get_train_action(data)
            )

            if len(train_data["meta"]["episode_ends"]) == 0:
                train_data["meta"]["episode_ends"].append(data_length)
            else:
                train_data["meta"]["episode_ends"].append(
                    data_length + train_data["meta"]["episode_ends"][-1]
                )
            data_index += data_length

        if cache_memmap:
            fp_image = np.memmap(
                img_memmap_loader_path, dtype=np.uint16, mode="r", shape=img_shape
            )

            train_data["data"]["img"] = fp_image

            fp_tactile = np.memmap(
                tactile_memmap_loader_path, dtype=np.uint16, mode="r", shape=tactile_shape
            )

            train_data["data"]["tactile"] = fp_tactile

        print("Train data loaded")
        for k, v in train_data["data"].items():
            print(k, v.shape)

        train_dataset = Dataset(
            data=train_data,
            representation_type=self.representation_type,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            stats=self.stats,
            load_img=self.load_img or img_memmap_loader_path != "",
            img_transform=self.img_transform if not eval else self.img_eval_transform,
            tactile_transform=self.tactile_transform if not eval else self.tactile_eval_transform,
            get_img=self._get_image_observation,
            binarize_tactile=self.binarize_tactile,
            state_noise=self.state_noise if not eval else 0.0,
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

    def get_train_loader(self, batch_size, train_test_split=0.9, eval=False):

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
            load_img=self.load_img,
            img_transform=self.img_transform if not eval else self.img_eval_transform,
            tactile_transform=self.tactile_transform if not eval else self.tactile_eval_transform,
            get_img=self._get_image_observation,
            binarize_tactile=self.binarize_tactile,
            state_noise=self.state_noise if not eval else 0.0,
            img_dim=(self.crop_img_width, self.crop_img_height),
            tactile_dim=(self.crop_tactile_width, self.crop_tactile_height),

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
            img_memmap_loader_path="",
            tactile_memmap_loader_path="",
            train_path=None,
            test_path=None,
    ):
        torch.cuda.empty_cache()

        if train_path is not None and test_path is not None:
            print(train_path)
            print(test_path)
            self.epi_dir = data_processing.get_epi_dir(train_path)
            print(self.epi_dir)
            eval_trajs = data_processing.get_epi_dir(test_path)
            assert len(eval_trajs) == 1
            eval_traj = eval_trajs[0]
            print("eval traj:", eval_traj)
        else:
            self.epi_dir = traj_list
            eval_traj = self.epi_dir[-1]
            self.epi_dir.remove(eval_traj)
            print("eval traj:", eval_traj)
        eval_data = self.get_eval_data(eval_traj)

        train_loader = self.get_train_loader(batch_size, img_memmap_loader_path, tactile_memmap_loader_path)
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
        self.eval(eval_traj)

    def get_train_action(self, data):
        act = self.get_eval_action(data)
        return np.stack(act)

    def get_eval_action(self, data):
        if self.predict_eef_delta:
            # TODO: make sure this is only used when "control" is eef pose
            act = []
            for d in data:
                left_arm_act = get_eef_delta(
                    d["eef_pos"][:6], d["control"][LEFT_UR_IDX]
                )
                left_hand_act = d["control"][LEFT_HAND_IDX]
                right_arm_act = get_eef_delta(
                    d["eef_pos"][6:], d["control"][RIGHT_UR_IDX]
                )
                right_hand_act = d["control"][RIGHT_HAND_IDX]
                act.append(
                    np.concatenate(
                        [left_arm_act, left_hand_act, right_arm_act, right_hand_act],
                        axis=-1,
                    )
                )
            return act
        elif self.predict_pos_delta:
            # TODO: make sure this is only used when "control" is joint pos
            act = [d["control"] for d in data]
            act = np.diff(act, axis=0, append=act[-1:])
            return act
        else:
            return [d["control"] for d in data]

    def get_eval_data(self, data_path):
        print("GETTING EVAL DATA", end="\r")
        data = data_processing.iterate(data_path)

        action = self.get_eval_action(data)
        B = len(data)

        print("GETTING EVAL OBSERVATION", end="\r")
        obs = self.get_observation(data, load_img=True)
        obs_list = []

        if "img" in self.representation_type:
            # transfer image type to float32
            obs["img"] = obs["img"].float()
            obs["img"] = self.img_eval_transform(obs["img"])
        for i in range(B):
            obs_list.append({rt: obs[rt][i] for rt in self.representation_type})
        return obs_list, action

    def eval(self, data_path, save_path=None):
        print("GETTING EVAL DATA")
        eval_data = self.get_eval_data(data_path)
        obs, action = eval_data
        print("EVALUATING")
        action, mse, norm_mse = self.policy.eval(obs, action)
        print("ACTION_MSE: {}, NORM_MSE: {}".format(mse, norm_mse))
        if save_path is None:
            save_path = "./eval/{}".format(
                data_path.split("/")[-1]
                + "_"
                + time.strftime("%m%d_%H%M%S", time.localtime())
            )
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(action)):
            if save_path is not None:
                with open(os.path.join(save_path, str(i) + ".pkl"), "wb") as f:
                    pickle.dump(
                        {
                            "control": action[i],
                            "joint_positions": eval_data[0][i]["pos"],
                        },
                        f,
                    )

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
        model_path = os.path.join(path)
        dir_path = os.path.dirname(path)
        stat_path = os.path.join(dir_path, "normalization.pkl")
        self.stats = pickle.load(open(stat_path, "rb"))
        self.policy.data_stat = self.stats
        self.policy.load(model_path)
        print("model loaded")

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

        self.cfg = cfg

        from matplotlib import pyplot as plt
        self.fig = plt.figure(figsize=(20, 15))

        # wandb config
        if cfg.gpu is not None:
            torch.cuda.set_device("cuda:{}".format(cfg.gpu))
            print("Using gpu: {}".format(cfg.gpu))

        # automatic naming
        curr_time = time.strftime("%m%d_%H%M%S", time.localtime())
        random_tag = generate_random_string()
        model_id = f"{curr_time}_{random_tag}"

        if cfg.use_wandb:
            wandb_config = WandBLogger.get_default_config()
            wandb_config.entity = cfg.wandb_entity_name
            wandb_config.project = cfg.wandb_project_name
            if cfg.wandb_exp_name is not None:
                wandb_config.exp_name = cfg.wandb_exp_name + "_" + model_id
            else:
                wandb_config.exp_name = model_id
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
                "tactile": cfg.dropout_rate,
            },
            output_sizes={
                "eef_pos": 64,
                "hand_joints": 64,
                "arm_joints": 128,
                "img": cfg.image_output_size,
                "tactile": cfg.tactile_output_size,
            },
            representation_type=cfg.representation_type.split("-"),
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
            load_img=cfg.load_img,
            num_workers=cfg.num_workers,
            weight_decay=cfg.weight_decay,
            use_ddim=cfg.use_ddim,
            binarize_tactile=cfg.binarize_tactile,
            policy_dropout_rate=cfg.policy_dropout_rate,
            state_noise=cfg.state_noise,
            img_gaussian_noise=cfg.img_gaussian_noise,
            img_masking_prob=cfg.img_masking_prob,
            img_patch_size=cfg.img_patch_size,
            tactile_gaussian_noise=cfg.tactile_gaussian_noise,
            tactile_masking_prob=cfg.tactile_masking_prob,
            tactile_patch_size=cfg.tactile_patch_size,
            compile_train=cfg.compile_train,
        )

        if cfg.load_path is not None:
            agent.load(cfg.load_path)

        if not cfg.eval:
            if cfg.model_save_path is None:
                model_path = os.path.join(cfg.output_dir, "ckpts")
            else:
                model_path = cfg.model_save_path

            model_path_suffix = f"{model_id}"

            if cfg.add_model_save_path_suffix:
                cfg_ = [
                    ("identity", cfg.identity_encoder),
                    (
                        "repr",
                        "".join(
                            [(x[0]).upper() for x in cfg.representation_type.split("-")]
                        ),
                    ),
                    ("oh", cfg.obs_horizon),
                    ("ah", cfg.action_horizon),
                    ("ph", cfg.pred_horizon),
                    ("do", cfg.dropout_rate),
                    ("imgos", cfg.image_output_size),
                    ("tctos", cfg.tactile_output_size),
                    ("wd", cfg.weight_decay),
                    ("use_ddim", cfg.use_ddim),
                    ("binarize_tactile", cfg.binarize_tactile),
                ]
                cfg_str = "-".join([f"{k}={v}" for k, v in cfg_])
                if cfg.without_sampling:
                    cfg_str += "-ws"
                if cfg.predict_pos_delta:
                    cfg_str += "-posdelta"
                if cfg.predict_eef_delta:
                    cfg_str += "-eefdelta"
                model_path_suffix += "-" + cfg_str
            model_path = os.path.join(model_path, model_path_suffix)

            print(f"Saving to model path {model_path}")

            if not os.path.exists(model_path):
                os.makedirs(model_path, exist_ok=True)

            save_args(cfg, model_path)

            if cfg.data_folder is not None:
                data_path = cfg.data_folder
            else:
                assert 'issue with data path'

            if agent.stats is not None:
                agent.save_stats(model_path)
            else:
                from glob import glob
                print('Loading trajectories from', data_path)
                traj_list = glob(os.path.join(data_path, '*/*.npz'))
                normalizer = DataNormalizer(cfg, traj_list)
                agent.stats = normalizer.normalize_dict
                agent.save_stats(model_path)

            if cfg.use_train_test_split:
                train_path = data_path + "_train"
                test_path = data_path + "_test"
            else:
                train_path = test_path = None

            print(f"using data path {data_path}")
            if cfg.use_memmap_cache:
                # image
                if cfg.img_memmap_loader_path is not None:
                    img_memmap_loader_path = cfg.img_memmap_loader_path
                else:
                    memmap_base_path = train_path if train_path is not None else data_path
                    img_memmap_loader_path = os.path.join(memmap_base_path, "image-mem.dat")
                # tactile
                if cfg.tactile_memmap_loader_path is not None:
                    tactile_memmap_loader_path = cfg.tactile_memmap_loader_path
                else:
                    memmap_base_path = train_path if train_path is not None else data_path
                    tactile_memmap_loader_path = os.path.join(memmap_base_path, "tactile-mem.dat")
            else:
                img_memmap_loader_path = ""
                tactile_memmap_loader_path = ""

            print("using image memmap loader path:", img_memmap_loader_path)
            print("using tactile memmap loader path:", tactile_memmap_loader_path)

            agent.train(
                traj_list,
                batch_size=cfg.batch_size,
                epochs=cfg.epochs,
                save_path=model_path,
                save_freq=cfg.save_freq,
                eval_freq=cfg.eval_freq,
                wandb_logger=wandb_logger,
                train_path=train_path,
                test_path=test_path,
                img_memmap_loader_path=img_memmap_loader_path,
                tactile_memmap_loader_path=tactile_memmap_loader_path,
            )

        else:
            agent.eval(cfg.eval_path, save_path=cfg.save_path)