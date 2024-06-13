from algo.models.transformer.data import TactileDataset, DataNormalizer, GaussianNoise
from torch.utils.data import DataLoader
from torch import optim
from algo.models.transformer.model import TactileTransformer
from algo.models.transformer.tact import TacT

from tqdm import tqdm
import torch
import pickle
import random
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb
import time
from algo.models.transformer.utils import set_seed
from hydra.utils import to_absolute_path
from warmup_scheduler import GradualWarmupScheduler
from torch import nn
from torch.nn import ModuleList
from torchvision import transforms
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


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


class Runner:
    def __init__(self,
                 cfg=None,
                 agent=None,
                 action_regularization=False,
                 num_fingers=3,

                 ):

        self.task_cfg = cfg
        self.cfg = cfg.offline_train
        self.agent = agent

        self.ppo_step = agent.play_latent_step if ((agent is not None) and (action_regularization)) else None
        self.optimizer = None
        self.scheduler = None
        self.full_sequence = self.cfg.model.transformer.full_sequence
        self.sequence_length = 500 if self.full_sequence else self.cfg.model.transformer.sequence_length
        self.device = 'cuda:0'

        # img
        self.img_channel = 1 if self.cfg.img_type == "depth" else 3
        self.img_color_jitter = self.cfg.img_color_jitter
        self.img_width = self.cfg.img_width
        self.img_height = self.cfg.img_height
        self.crop_img_width, self.crop_img_height = self.img_width - 20, self.img_height - 30
        self.img_transform, self.img_downsample, self.img_eval_transform = define_transforms(self.img_channel,
                                                                                             self.img_color_jitter,
                                                                                             self.img_width,
                                                                                             self.img_height,
                                                                                             self.crop_img_width,
                                                                                             self.crop_img_height,
                                                                                             self.cfg.img_patch_size,
                                                                                             self.cfg.img_gaussian_noise,
                                                                                             self.cfg.img_masking_prob)
        # tactile
        self.num_fingers = num_fingers
        self.tactile_channel = 1 if self.cfg.tactile_type == "gray" else 3
        self.half_image = True
        self.tactile_color_jitter = self.cfg.tactile_color_jitter
        self.tactile_width = self.cfg.tactile_width // 2 if self.half_image else self.cfg.tactile_width
        self.tactile_height = self.cfg.tactile_height
        self.crop_tactile_width, self.crop_tactile_height = self.tactile_width, self.tactile_height
        self.tactile_transform, self.tactile_downsample, self.tactile_eval_transform = define_transforms(
            self.tactile_channel,
            self.tactile_color_jitter,
            self.tactile_width,
            self.tactile_height,
            self.crop_tactile_width,
            self.crop_tactile_height,
            self.cfg.tactile_patch_size,
            self.cfg.tactile_gaussian_noise,
            self.cfg.tactile_masking_prob
        )

        self.model = TacT(context_size=self.sequence_length,
                          num_channels=self.cfg.model.cnn.in_channels,
                          num_lin_features=self.cfg.model.linear.input_size,
                          num_outputs=self.cfg.model.transformer.output_size,
                          tactile_encoder="efficientnet-b0",
                          img_encoder="efficientnet-b0",
                          tactile_encoding_size=self.cfg.model.transformer.tactile_encoding_size,
                          img_encoding_size=self.cfg.model.transformer.img_encoding_size,
                          mha_num_attention_heads=self.cfg.model.transformer.num_heads,
                          mha_num_attention_layers=self.cfg.model.transformer.num_layers,
                          mha_ff_dim_factor=self.cfg.model.transformer.dim_factor, )

        self.src_mask = torch.triu(torch.ones(self.sequence_length, self.sequence_length), diagonal=1).bool().to(
            self.device)

        self.loss_fn_mean = torch.nn.MSELoss(reduction='mean')
        self.loss_fn = torch.nn.MSELoss(reduction='none')

        self.fig = plt.figure(figsize=(20, 15))
        self.train_loss, self.val_loss = [], []


    def train(self, dl, val_dl, ckpt_path, print_every=50, eval_every=250, test_every=500):
        """
        Train the model using the provided data loader, with periodic validation and testing.

        Args:
            dl (DataLoader): Training data loader.
            val_dl (DataLoader): Validation data loader.
            ckpt_path (str): Path to save checkpoints.
            print_every (int): Frequency of printing training loss.
            eval_every (int): Frequency of evaluating the model on validation data.
            test_every (int): Frequency of testing the model.
        """
        self.model.train()
        train_loss, val_loss = [], []
        latent_loss_list, action_loss_list = [], []

        progress_bar = tqdm(enumerate(dl), total=len(dl), desc="Training Progress", unit="batch")

        for i, (tac_input, img_input, lin_input, contacts, obs_hist, latent, action, mask) in progress_bar:
            self.model.train()

            tac_input = tac_input.to(self.device)  # [B T F W H C]
            img_input = img_input.to(self.device)
            lin_input = lin_input.to(self.device)
            latent = latent.to(self.device)
            action = action.to(self.device)
            mask = mask.to(self.device).unsqueeze(-1)

            out = self.model(tac_input, img_input, lin_input)
            loss_action = torch.zeros(1, device=self.device)

            if self.full_sequence:
                loss_latent = torch.sum(self.loss_fn(out, latent), dim=-1).unsqueeze(-1)
                loss_latent = torch.sum(loss_latent * mask) / torch.sum(mask)

                if self.ppo_step is not None:
                    obs_hist = obs_hist.to(self.device).view(obs_hist.shape[0] * self.sequence_length,
                                                             obs_hist.shape[-1])
                    pred_action, _ = self.ppo_step(
                        {'obs': obs_hist, 'latent': out.view(out.shape[0] * out.shape[1], out.shape[-1])})
                    loss_action = torch.sum(self.loss_fn(pred_action.view(*action.shape), action), dim=-1).unsqueeze(-1)
                    loss_action = torch.sum(loss_action * mask) / torch.sum(mask)
                    action_loss_list.append(loss_action.item())

            else:
                loss_latent = self.loss_fn_mean(out, latent[:, -1, :])

                if self.ppo_step is not None:
                    obs_hist = obs_hist[:, -1, :].to(self.device).view(obs_hist.shape[0], obs_hist.shape[-1])
                    pred_action, _ = self.ppo_step({'obs': obs_hist, 'latent': out[:, -1, :]})
                    pred_action = torch.clamp(pred_action, -1, 1)
                    loss_action = self.loss_fn_mean(pred_action, action[:, -1, :].squeeze(1))

            loss = (self.cfg.train.latent_scale * loss_latent) + (self.cfg.train.action_scale * loss_action[0])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.append(loss.item())
            latent_loss_list.append(loss_latent.item())
            if self.ppo_step is not None:
                action_loss_list.append(loss_action.item())

            # Update tqdm description
            progress_bar.set_postfix({
                'Batch': i + 1,
                'Loss': np.mean(train_loss)
            })

            if (i + 1) % print_every == 0:
                print(f'step {i + 1}:', np.mean(train_loss))
                self._wandb_log({'train/loss': np.mean(train_loss),
                                 'train/latent_loss': np.mean(latent_loss_list)})
                if self.ppo_step is not None:
                    self._wandb_log({'train/action_loss': np.mean(action_loss_list)})

                self.fig.clf()
                self.train_loss.append(np.mean(train_loss))
                plt.plot(self.train_loss, '-ro', linewidth=3, label='train loss')
                train_loss = []
                latent_loss_list = []
                action_loss_list = []

            if (i + 1) % eval_every == 0:
                self.log_output(tac_input.clone(), img_input.clone(), lin_input.clone(), out.clone(), latent.clone(),
                                'train')

                val_loss = self.validate(val_dl)
                print(f'validation loss: {val_loss}')
                self.val_loss.append(val_loss)
                plt.plot(self.val_loss, '-ko', linewidth=3, label='val loss')
                plt.legend()
                self.fig.savefig(f'{self.save_folder}/train_val_comp.png', dpi=200, bbox_inches='tight')
                self.model.train()

            if (i + 1) % test_every == 0:
                try:
                    self.test()
                except Exception as e:
                    print(f'Error during test: {e}')
                self.model.train()

        return val_loss

    def validate(self, dl):
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            latent_loss_list, action_loss_list = [], []
            for i, (tac_input, img_input, lin_input, contacts, obs_hist, latent, action, mask) in tqdm(enumerate(dl)):

                tac_input = tac_input.to(self.device)
                img_input = img_input.to(self.device)

                lin_input = lin_input.to(self.device)
                latent = latent.to(self.device)
                action = action.to(self.device)
                mask = mask.to(self.device).unsqueeze(-1)
                # contacts = contacts.to(self.device)

                out = self.model(tac_input, img_input, lin_input)

                loss_action = torch.zeros(1, device=self.device)

                if self.full_sequence:
                    loss_latent = torch.sum(self.loss_fn(out, latent), dim=-1).unsqueeze(-1)
                    loss_latent = torch.sum(loss_latent * mask) / torch.sum(mask)
                    # loss = loss_latent
                    if self.ppo_step is not None:
                        obs_hist = obs_hist.to(self.device).view(obs_hist.shape[0] * self.sequence_length,
                                                                 obs_hist.shape[-1])
                        pred_action, _ = self.ppo_step(
                            {'obs': obs_hist, 'latent': out.view(out.shape[0] * out.shape[1], out.shape[-1])})
                        loss_action = torch.sum(self.loss_fn(pred_action.view(*action.shape), action),
                                                dim=-1).unsqueeze(-1)
                        loss_action = torch.sum(loss_action * mask) / torch.sum(mask)
                else:
                    loss_latent = self.loss_fn_mean(out[:, :], latent[:, -1, :])

                    # loss = loss_latent
                    if self.ppo_step is not None:
                        obs_hist = obs_hist[:, -1, :].to(self.device).view(obs_hist.shape[0], obs_hist.shape[-1])
                        pred_action, _ = self.ppo_step({'obs': obs_hist, 'latent': out[:, -1, :]})
                        pred_action = torch.clamp(pred_action, -1, 1)
                        loss_action = self.loss_fn_mean(pred_action, action[:, -1, :].squeeze(1))

                # TODO: add scaling loss coefficients
                loss = (self.cfg.train.latent_scale * loss_latent) + (self.cfg.train.action_scale * loss_action)

                val_loss.append(loss.item())
                latent_loss_list.append(loss_latent.item())
                if self.ppo_step is not None:
                    action_loss_list.append(loss_action.item())

            # self._wandb_log({
            #     'val/loss': np.mean(val_loss),
            #     'val/latent_loss': np.mean(latent_loss_list),
            # 'val/action_loss': np.mean(action_loss_list)
            # })
            # if self.ppo_step is not None:
            #     self._wandb_log({
            #         'val/action_loss': np.mean(action_loss_list)
            #     })

            self.log_output(tac_input.clone(),
                            img_input.clone(),
                            lin_input.clone(),
                            out.clone(),
                            latent.clone(),
                            'valid')

        return np.mean(val_loss)

    def log_output(self, tac_input, img_input, lin_input, out, latent, session='train'):
        # Selecting the first example from the batch for demonstration
        # tac_input [B T F W H C]

        image_sequence = tac_input[0].cpu().detach().numpy()
        img_input = img_input[0].cpu().detach().numpy()
        # linear_features = lin_input[0].cpu().detach().numpy()
        predicted_output = out[0].cpu().detach().numpy()
        true_label = latent[0, -1, :].cpu().detach().numpy()
        # Plotting
        fig = plt.figure(figsize=(20, 10))

        # Adding subplot for image sequence (adjust as needed)
        ax1 = fig.add_subplot(2, 2, 1)
        concat_images = []
        # image_sequence [T F W H C]
        for finger_idx in range(image_sequence.shape[1]):
            finger_sequence = [np.transpose(img, (1, 2, 0)) for img in image_sequence[:, finger_idx, ...]]
            finger_sequence = np.hstack(finger_sequence)
            concat_images.append(finger_sequence)

        ax1.imshow(np.vstack(concat_images))  # Adjust based on image normalization
        ax1.set_title('Input Tactile Sequence')

        # Adding subplot for linear features (adjust as needed)
        # ax2 = fig.add_subplot(2, 2, 2)
        # ax2.plot(linear_features[:, :], 'ok', label='hand_joints')  # Assuming the rest are actions
        # ax2.set_title('Linear input')
        # ax2.legend()

        # Check if img_input has more than one timestep
        if img_input.ndim == 4 and img_input.shape[0] > 1:
            concat_img_input = []
            for t in range(img_input.shape[0]):
                img = img_input[t]
                img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
                img = img + 0.5  # Adjust normalization if needed
                concat_img_input.append(img)

            # Horizontally stack the images for each timestep
            concat_img_input = np.hstack(concat_img_input)
        else:
            # Handle the case where there is only one timestep
            img = img_input[0] if img_input.ndim == 4 else img_input
            img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
            img = img + 0.5  # Adjust normalization if needed
            concat_img_input = img

        # Plot the concatenated image sequence
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(concat_img_input)
        ax3.set_title('Input Image Sequence')

        # Adding subplot for Output vs. True Label comparison
        ax4 = fig.add_subplot(2, 2, 4)
        width = 0.35
        indices = np.arange(len(predicted_output))
        ax4.bar(indices - width / 2, predicted_output, width, label='Predicted')
        ax4.bar(indices + width / 2, true_label, width, label='True Label')
        ax4.set_title('Model Output vs. True Label')
        ax4.legend()

        # Adjust layout
        plt.tight_layout()
        # Saving the figure
        plt.savefig(f'{self.save_folder}/{session}_example.png')
        # Clean up plt to free memory
        plt.close(fig)

    def test(self):
        with torch.inference_mode():
            normalize_dict = self.normalize_dict.copy()
            num_success, total_trials = self.agent.test(self.get_latent, normalize_dict)
            if total_trials > 0:
                print(f'{num_success}/{total_trials}, success rate on :', num_success / total_trials)
                self._wandb_log({
                    'test/success_rate': num_success / total_trials
                })
            else:
                print('something went wrong, there are no test trials')

    def load_model(self, model_path, device='cuda:0'):
        self.model.load_state_dict(torch.load(model_path))
        # self.model.eval()
        self.device = device

    def get_latent(self, tac_input, img_input, lin_input):
        self.model.eval()
        with torch.no_grad():
            # [envs, seq_len, ... ] => [envs*seq_len, C, W, H]

            tac_input = tac_input.to(self.device)
            img_input = img_input.to(self.device)
            lin_input = lin_input.to(self.device)

            out = self.model(tac_input, img_input, lin_input)
        return out

    def _run(self, file_list, save_folder, epochs=100, train_test_split=0.9, train_batch_size=32, val_batch_size=32,
             learning_rate=1e-4, device='cuda:0', print_every=50, eval_every=250, test_every=500):

        random.shuffle(file_list)
        print('# trajectories:', len(file_list))

        ckpt_path = f'{save_folder}/checkpoints'
        if not os.path.exists(ckpt_path):
            os.makedirs(f'{ckpt_path}')

        self.device = device
        self.model = self.model.to(self.device)
        self.src_mask = self.src_mask.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

        if self.cfg.train.scheduler == 'reduce':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                  mode='min',
                                                                  factor=0.5,
                                                                  patience=3,
                                                                  verbose=True)

        if self.cfg.train.scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        if self.cfg.train.warmup:
            print("Using warmup scheduler")
            self.scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1,
                total_epoch=self.cfg.train.warmup_epochs,
                after_scheduler=self.scheduler,
            )

        num_train_envs = int(len(file_list) * train_test_split)
        train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        val_idxs = np.arange(num_train_envs, len(file_list)).astype(int).tolist()
        training_files = [file_list[i] for i in train_idxs]
        val_files = [file_list[i] for i in val_idxs]

        # Passing trajectories
        train_ds = TactileDataset(files=training_files,
                                  full_sequence=self.full_sequence,
                                  sequence_length=self.sequence_length,
                                  normalize_dict=self.normalize_dict,
                                  img_transform=self.img_transform,
                                  tactile_transform=self.tactile_transform
                                  )
        train_dl = DataLoader(train_ds,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True,
                              )

        val_ds = TactileDataset(files=val_files,
                                full_sequence=self.full_sequence,
                                sequence_length=self.sequence_length,
                                normalize_dict=self.normalize_dict,
                                img_transform=self.img_eval_transform,
                                tactile_transform=self.tactile_eval_transform,
                                )

        val_dl = DataLoader(val_ds,
                            batch_size=val_batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            persistent_workers=True,
                            )

        # training
        for epoch in range(epochs):
            self.validate(val_dl)
            if self.cfg.train.only_test:
                self.test()
            elif self.cfg.train.only_validate:
                self.validate(val_dl)
            else:
                val_loss = self.train(train_dl, val_dl, ckpt_path,
                                      print_every=print_every,
                                      eval_every=eval_every,
                                      test_every=test_every)

                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(np.mean(val_loss))
                    else:
                        self.scheduler.step()
                print('Saving the model')
                torch.save(self.model.state_dict(), f'{ckpt_path}/model_last.pt')  # {epoch}.pt')

    def _wandb_log(self, data):
        if self.cfg.wandb.wandb_enabled:
            wandb.log(data)

    def run(self):
        from datetime import datetime
        from glob import glob

        print('Loading trajectories from', self.cfg.data_folder)

        file_list = glob(os.path.join(self.cfg.data_folder, '*/*/obs/*.npz'))
        save_folder = f'{to_absolute_path(self.cfg.output_dir)}/tact_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(save_folder, exist_ok=True)

        device = 'cuda:0'

        # Load student checkpoint.
        if self.cfg.train.load_checkpoint:
            model_path = self.cfg.train.student_ckpt_path
            self.load_model(model_path, device=device)

        train_config = {
            "epochs": self.cfg.train.epochs,
            "train_test_split": self.cfg.train.train_test_split,
            "train_batch_size": self.cfg.train.train_batch_size,
            "val_batch_size": self.cfg.train.val_batch_size,
            "learning_rate": self.cfg.train.learning_rate,
            "print_every": self.cfg.train.print_every,
            "eval_every": self.cfg.train.eval_every,
            "test_every": self.cfg.train.test_every
        }

        if self.cfg.wandb.wandb_enabled:
            wandb.init(
                # Set the project where this run will be logged
                project=self.cfg.wandb.wandb_project_name,
                # Track hyperparameters and run metadata
                config=train_config,
                dir=save_folder,
            )

        normalizer = DataNormalizer(self.cfg, file_list)
        normalizer.run()
        self.normalize_dict = normalizer.normalize_dict

        self.save_folder = save_folder

        with open(os.path.join(save_folder, f"task_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.task_cfg))
        with open(os.path.join(save_folder, f"train_config.yaml"), "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))

        self._run(file_list, save_folder, device=device, **train_config)