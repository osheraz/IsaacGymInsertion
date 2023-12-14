from algo.models.transformer.data import TactileDataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch import optim
from algo.models.transformer.model import TactileTransformer
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
from matplotlib import pyplot as plt

class Runner:
    def __init__(self, cfg=None, agent=None, action_regularization=False):

        self.cfg = cfg
        self.agent = agent

        self.ppo_step = agent.play_latent_step if ((agent is not None) and (action_regularization)) else None
        self.optimizer = None
        self.scheduler = None
        self.full_sequence = self.cfg.model.transformer.full_sequence
        self.sequence_length = 30 if self.full_sequence else self.cfg.model.transformer.sequence_length
        self.device = 'cuda:0'

        self.model = TactileTransformer(lin_input_size=self.cfg.model.linear.input_size,
                                        in_channels=self.cfg.model.cnn.in_channels,
                                        out_channels=self.cfg.model.cnn.out_channels,
                                        kernel_size=self.cfg.model.cnn.kernel_size,
                                        embed_size=self.cfg.model.transformer.embed_size,
                                        hidden_size=self.cfg.model.transformer.hidden_size,
                                        num_heads=self.cfg.model.transformer.num_heads,
                                        num_layers=self.cfg.model.transformer.num_layers,
                                        max_sequence_length=self.sequence_length,
                                        output_size=self.cfg.model.transformer.output_size)

        self.src_mask = torch.triu(torch.ones(self.sequence_length, self.sequence_length), diagonal=1).bool().to(
            self.device)

        self.loss_fn_mean = torch.nn.MSELoss(reduction='mean')
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.bce_fn = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

    def train(self, dl, val_dl, ckpt_path, print_every=50, eval_every=250, test_every=500):
        from matplotlib import pyplot as plt
        self.model.train()
        train_loss, val_loss = [], 0
        latent_loss_list, action_loss_list = [], []
        for i, (cnn_input, lin_input, obs_hist, latent, action, mask) in tqdm(enumerate(dl)):
            self.model.train()
            cnn_inputs = []
            for finger in cnn_input:
                # temp_finger = finger.clone().detach().cpu().numpy()
                # for ii in range(100):
                #     plt.imshow(temp_finger[0, ii, :, :], cmap='gray')
                #     plt.pause(0.0001)
                #     plt.cla()
                # plt.close()
                finger = finger.to(self.device)
                finger = finger.view(finger.shape[0] * self.sequence_length, *finger.size()[-3:])
                finger = finger.permute(0, 3, 1, 2)
                cnn_inputs.append(finger)

            # temp_latent = latent.clone().detach().cpu().numpy()/10.0
            # for ii in range(100):
            #     print(temp_latent.shape)
            #     for mm in range(8):
            #         plt.plot(temp_latent[0, :ii, mm])
            #     plt.pause(0.0001)
            #     plt.cla()

            lin_input = lin_input.to(self.device)
            latent = latent.to(self.device)
            action = action.to(self.device)
            mask = mask.to(self.device).unsqueeze(-1)

            out = self.model(cnn_inputs, lin_input,
                             batch_size=lin_input.shape[0],
                             embed_size=self.cfg.model.transformer.embed_size // 2,
                             src_mask=self.src_mask)
            
            loss_action = 0 # torch.nn.Parameter(torch.zeros(1), requires_grad=True).to(self.device)
            if self.full_sequence:
                loss_latent = torch.mean(self.bce_fn(out, latent), dim=-1).unsqueeze(-1)
                loss_latent = torch.sum(loss_latent * mask) / torch.sum(mask)
                if self.ppo_step is not None:  # action regularization
                    obs_hist = obs_hist.to(self.device).view(obs_hist.shape[0] * self.sequence_length,
                                                             obs_hist.shape[-1])
                    pred_action, _ = self.ppo_step(
                        {'obs': obs_hist, 'latent': out.view(out.shape[0] * out.shape[1], out.shape[-1])})

                    loss_action = torch.sum(self.loss_fn(pred_action.view(*action.shape), action), dim=-1).unsqueeze(-1)
                    loss_action = torch.sum(loss_action * mask) / torch.sum(mask)
                    with torch.no_grad():
                        action_loss_list.append(loss_action.item())
            else:
                loss_latent = self.loss_fn_mean(out[:, -1, :], latent[:, -1, :])
                if self.ppo_step is not None:  # action regularization
                    obs_hist = obs_hist[:, -1, :].to(self.device).view(obs_hist.shape[0], obs_hist.shape[-1])
                    pred_action, _ = self.ppo_step({'obs': obs_hist, 'latent': out[:, -1, :]})
                    pred_action = torch.clamp(pred_action, -1, 1)
                    loss_action = self.loss_fn_mean(pred_action, action[:, -1, :].squeeze(1))

            # TODO: add scaling loss coefficients
            loss = (1. * loss_latent) + (0.25 * loss_action)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.item())
            latent_loss_list.append(loss_latent.item())
            if self.ppo_step is not None:
                action_loss_list.append(loss_action.item())

            # out1 = out.clone().detach().cpu().numpy()
            # latent1 = latent.clone().detach().cpu().numpy()
            # for ii in tqdm(range(100)):
            #     plt.scatter(list(range(latent.shape[-1])), latent1[0, ii, :], c='b')
            #     plt.scatter(list(range(latent.shape[-1])), out1[0, ii, :], c='r')
            #     plt.pause(0.0001)
            #     plt.cla()

            if (i + 1) % print_every == 0:
                print(f'step {i + 1}:', np.mean(train_loss))
                self._wandb_log({
                    'train/loss': np.mean(train_loss),
                    'train/latent_loss': np.mean(latent_loss_list),
                })
                if self.ppo_step is not None:
                    self._wandb_log({'train/action_loss': np.mean(action_loss_list)})

                train_loss = []
                latent_loss_list = []
                action_loss_list = []

            if (i + 1) % eval_every == 0:
                val_loss = self.validate(val_dl)
                print(f'validation loss: {val_loss}')
                self.model.train()
                self.scheduler.step(val_loss)
                val_loss = 0.

            if (i + 1) % test_every == 0:
                self.test()
                # try:
                #     self.test()
                # except:
                #     pass
                self.model.train()

        return val_loss

    def validate(self, dl):
        self.model.eval()
        with torch.inference_mode():
            val_loss = []
            latent_loss_list, action_loss_list = [], []
            for i, (cnn_input, lin_input, obs_hist, latent, action, mask) in tqdm(enumerate(dl)):
                # [envs, seq_len, W, H, C] => [envs*seq_len, C, W, H]
                
                cnn_inputs = []
                for finger in cnn_input:
                    finger = finger.to(self.device)
                    finger = finger.view(finger.shape[0] * self.sequence_length, *finger.size()[-3:])
                    finger = finger.permute(0, 3, 1, 2)
                    cnn_inputs.append(finger)

                lin_input = lin_input.to(self.device)
                latent = latent.to(self.device)
                action = action.to(self.device)
                mask = mask.to(self.device).unsqueeze(-1)
                out = self.model(cnn_inputs, lin_input,
                                 batch_size=lin_input.shape[0],
                                 embed_size=self.cfg.model.transformer.embed_size // 2,
                                 src_mask=self.src_mask)

                loss_action = 0

                if self.full_sequence:
                    loss_latent = torch.sum(self.bce_fn(out, latent), dim=-1).unsqueeze(-1)
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
                        # loss += loss_action
                else:
                    loss_latent = self.loss_fn_mean(out[:, -1, :], latent[:, -1, :])

                    # loss = loss_latent
                    if self.ppo_step is not None:
                        obs_hist = obs_hist[:, -1, :].to(self.device).view(obs_hist.shape[0], obs_hist.shape[-1])
                        pred_action, _ = self.ppo_step({'obs': obs_hist, 'latent': out[:, -1, :]})
                        pred_action = torch.clamp(pred_action, -1, 1)
                        loss_action = self.loss_fn_mean(pred_action, action[:, -1, :].squeeze(1))
                        # loss += loss_action

                # TODO: add scaling loss coefficients
                loss = (1. * loss_latent) + (0.25 * loss_action)

                val_loss.append(loss.item())
                latent_loss_list.append(loss_latent.item())
                if self.ppo_step is not None:
                    action_loss_list.append(loss_action.item())

            self._wandb_log({
                'val/loss': np.mean(val_loss),
                'val/latent_loss': np.mean(latent_loss_list),
                # 'val/action_loss': np.mean(action_loss_list)
            })
            if self.ppo_step is not None:
                self._wandb_log({
                    'val/action_loss': np.mean(action_loss_list)
                })
        return np.mean(val_loss)

    def test(self):
        with torch.inference_mode():
            normalize_dict = self.normalize_dict.copy()
            num_success, total_trials = self.agent.test(self.predict, normalize_dict)
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

    def predict(self, cnn_input, lin_input):
        self.model.eval()
        with torch.inference_mode():
            # [envs, seq_len, W, H, C] => [envs*seq_len, C, W, H]
            cnn_inputs = []
            for finger in cnn_input:
                # temp_finger = finger.clone().detach().cpu().numpy()
                # for ii in range(100):
                #     self.ax.imshow(temp_finger[0, ii, :, :], cmap='gray')
                #     plt.pause(0.0001)
                #     self.ax.cla()
                # plt.close()

                finger = finger.to(self.device)
                finger = finger.view(finger.shape[0] * self.sequence_length, *finger.size()[-3:])
                finger = finger.permute(0, 3, 1, 2)
                cnn_inputs.append(finger)

            lin_input = lin_input.to(self.device)

            out = self.model(cnn_inputs, lin_input,
                             src_mask=self.src_mask,
                             batch_size=lin_input.shape[0],
                             embed_size=self.cfg.model.transformer.embed_size // 2)
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

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5,
                                                              verbose=True)

        num_train_envs = int(len(file_list) * train_test_split)
        train_idxs = np.arange(0, num_train_envs).astype(int).tolist()
        val_idxs = np.arange(num_train_envs, len(file_list)).astype(int).tolist()
        training_files = [file_list[i] for i in train_idxs]
        val_files = [file_list[i] for i in val_idxs]

        train_ds = TactileDataset(files=training_files, full_sequence=self.full_sequence,
                                  sequence_length=self.sequence_length, normalize_dict=self.normalize_dict)
        train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)

        val_ds = TactileDataset(files=val_files, full_sequence=self.full_sequence, sequence_length=self.sequence_length,
                                normalize_dict=self.normalize_dict)
        val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=True)

        # training
        for epoch in range(epochs):
            if self.cfg.train.only_test:
                self.test()
            elif self.cfg.train.only_validate:
                self.validate(val_dl)
            else:
                val_loss = self.train(train_dl, val_dl, ckpt_path, print_every=print_every, eval_every=eval_every,
                                      test_every=test_every)
                # self.scheduler.step(val_loss)

                torch.save(self.model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')
                # torch.jit.save(torch.jit.script(self.model), f'{ckpt_path}/model_{epoch}.pt')

    def _wandb_log(self, data):
        if self.cfg.wandb.wandb_enabled:
            wandb.log(data)

    def run(self):
        from datetime import datetime
        from glob import glob
        from tqdm import tqdm
        from pathlib import Path

        # Load trajectories
        if 'oa348' in os.getcwd():
            self.cfg.data_folder.replace("dm1487", "oa348")
            self.cfg.output_dir.replace("dm1487", "oa348")
        file_list = glob(os.path.join(self.cfg.data_folder, '*/*.npz'))
        save_folder = f'{to_absolute_path(self.cfg.output_dir)}/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(save_folder, exist_ok=True)

        device = 'cuda:0'

        # Load student checkpoint.
        if self.cfg.train.load_checkpoint:
            # TODO We should also save the optimizer, and load him here.
            # TODO Need to make sure we are loading the same normalization?
            model_path = self.cfg.train.ckpt_path
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

        # Removing failed trajectories
        for file in tqdm(file_list):
            try:
                d = np.load(file)
                # t = d['tactile']
                done_idx = d['done'].nonzero()[0]
                if done_idx == 0:
                    file_list.remove(file)
            except KeyboardInterrupt:
                exit()
            except:
                print(file)
                os.remove(file)

        normalize_keys = self.cfg.train.normalize_keys

        # TODO Move to function, make clean run
        normalization_path = self.cfg.train.normalize_file
        if Path(normalization_path).parent.absolute().exists() is False:
            Path(normalization_path).parent.absolute().mkdir(parents=True, exist_ok=True)
        if os.path.exists(normalization_path):
            # if already exisits, just load
            with open(normalization_path, 'rb') as f:
                normalize_dict = pickle.load(f)
        else:
            # means and standard deviations for normalization
            normalize_dict = {
                "mean": {},
                "std": {}
            }
            for norm_keys in normalize_keys:
                data = []
                print('Creating new normalization file:')
                for file in tqdm(random.sample(file_list, min(1000, len(file_list)))):
                    try:
                        d = np.load(file)
                        done_idx = d['done'].nonzero()[0][-1]
                        data.append(d[norm_keys][:done_idx, :])
                    except:
                        print(file, 'bad file')
                data = np.concatenate(data, axis=0)
                normalize_dict['mean'][norm_keys] = np.mean(data, axis=0)
                normalize_dict['std'][norm_keys] = np.std(data, axis=0)
            print('Saved new normalization file at: ', normalization_path)
            with open(f'{normalization_path}', 'wb') as f:
                pickle.dump(normalize_dict, f)

        self.normalize_dict = normalize_dict
        self._run(file_list, save_folder, device=device, **train_config)
