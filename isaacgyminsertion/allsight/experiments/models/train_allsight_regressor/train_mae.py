import time
import os
import json
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import copy
from misc import normalize, unnormalize, normalize_max_min, unnormalize_max_min, save_df_as_json
from vis_utils import Arrow3D
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR  # Learning rate schedulers
from models import PreTrainedModel, PreTrainedModelWithRef, load_tactile_resnet
from vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z, set_axes_equal
from datasets import TactileDataset, output_map, get_buffer_paths, CircleMaskTransform, print_sensor_ids
from surface import create_finger_geometry
from geometry import convert_quat_wxyz_to_xyzw, convert_quat_xyzw_to_wxyz
from transformations import quaternion_matrix
from scipy import spatial
from tqdm import tqdm
import random
from isaacgyminsertion.allsight.experiments.models.train_allsight_regressor.vit_pytorch import SimpleViT, SmallViT, MAE, \
    ViT
from vit_pytorch.mpp import MPP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
np.set_printoptions(suppress=True, linewidth=np.inf)  # to widen the printed array

random.seed(42)
torch.manual_seed(42)


class Trainer(object):

    def __init__(self, params):

        # Get params, create logger
        self.params = params

        leds = params['leds']
        gel = params['gel']
        indenter = ['sphere3', 'sphere4', 'sphere5', 'square', 'hexagon', 'ellipse']

        # print_sensor_ids(leds, gel, indenter)

        buffer_paths_to_train, buffer_test_paths, sensors_1, sensors_2 = get_buffer_paths(leds, gel, indenter,
                                                                                          train_sensor_id=[3
                                                                                              , 10, 9, 18,
                                                                                                           4, 17, 15,
                                                                                                           16, 12, 14,
                                                                                                           13, 11, 2, 1,
                                                                                                           0, 7, 8, 6,
                                                                                                           5],

                                                                                          # train_sensor_id=[3, 10, 9, 18, 4, 17, 15, 16, 12, 14, 13, 11, 2, 1, 0, 7, 8, 6, 5],
                                                                                          test_sensor_id=[19])

        #####################
        ## SET AGENT PARAMS
        #####################

        self.model_params = {
            'learning_rate': params['learning_rate'],
            'device': params['device'].type,
            'batch_size': params['batch_size'],
            'image_size': params['image_size'],
            'half_image': params['half_image'],
            'remove_ref': params['remove_ref'],
            'epoch': params['epoch'],
            'optimizer': "Adam",
            'portion': params['portion'],
            'logdir': params['logdir'],
            'scheduler': params['scheduler'],
            'model_name': params['model_name'],
            'buffer_paths': buffer_paths_to_train,
            'output': params['output'],
            'norm_method': params['norm_method'],
            'aug': params['aug'],
            'input_type': params['input_type'],
            'leds': leds,
            'num_channels': params['num_channels'],
            'indenter': indenter,
        }

        self.num_channels = params['num_channels']
        self.half_image = params['half_image']
        self.remove_ref = params['remove_ref']

        self.finger_geometry = create_finger_geometry()
        self.tree = spatial.KDTree(self.finger_geometry[0])
        self.prepare_data(buffer_paths_to_train, buffer_test_paths, params['output'])

        self.model = SimpleViT(image_size=params['image_size'],
                               patch_size=14,
                               num_classes=output_map[params['output']],
                               channels=params['num_channels'],
                               dim=512,
                               depth=6,
                               heads=8,
                               mlp_dim=512,
                               # dropout=0.1,
                               # emb_dropout=0.1
                               ).to(device)

        self.mae = MAE(encoder=self.model,
                       masking_ratio=0.5,  # the paper recommended 75% masked patches
                       decoder_dim=512,  # paper showed good results with just 512
                       decoder_depth=6  # anywhere from 1 to 8
                       ).to(device)

        with open(self.params['logdir'] + '/model_params.json', 'w') as fp:
            dic_items = self.model_params.items()
            new_dict = {key: value for key, value in dic_items}
            json.dump(new_dict, fp, indent=3)

        with open(self.params['logdir'] + '/data_statistic.json', 'w') as fp:
            dic_items = self.originalset.data_statistics.items()
            new_dict = {key: value.tolist() for key, value in dic_items}
            json.dump(new_dict, fp, indent=3)

        self.optimizer = getattr(torch.optim, self.model_params['optimizer'])(self.mae.parameters(),
                                                                              lr=params['learning_rate'])

        if params['scheduler'] == 'reduce':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        elif params['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=4, eta_min=1e-8)
        elif params['scheduler'] == 'step':
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.8)
        else:
            self.scheduler = None

        self.fig = plt.figure(figsize=(20, 15))

        self.best_model = copy.deepcopy(self.model)

    def prepare_data(self, paths_train, path_test, output_type):

        for idx, p in enumerate(paths_train):
            if idx == 0:
                df_data = pd.read_json(p).transpose()
                # s = min(df_data.shape[0], 4000)
                # df_data = df_data.sample(n=s)
            else:
                new_df = pd.read_json(p).transpose()
                # s = min(new_df.shape[0], 4000)
                # new_df = new_df.sample(n=s)
                df_data = pd.concat([df_data, new_df], axis=0)

        for idx, p in enumerate(path_test):
            if idx == 0:
                df_data_test = pd.read_json(p).transpose()
                # s = min(df_data_test.shape[0], 4000)
                # df_data_test = df_data_test.sample(n=s)
            else:
                new_df = pd.read_json(p).transpose()
                # s = min(new_df.shape[0], 4000)
                # new_df = new_df.sample(n=s)
                df_data_test = pd.concat([df_data_test, new_df], axis=0)

        if self.half_image:
            condition1 = (df_data['theta_transformed'] > np.pi / 4)
            condition2 = (df_data['theta_transformed'] < 3 * np.pi / 2)
            df_data = df_data[condition1 & condition2]
            df_data = df_data.reset_index(drop=True)

            condition1 = (df_data_test['theta_transformed'] > np.pi / 4)
            condition2 = (df_data_test['theta_transformed'] < 3 * np.pi / 2)
            df_data_test = df_data_test[condition1 & condition2]
            df_data_test = df_data_test.reset_index(drop=True)

        df_data = df_data[df_data.time > 1]  # train only over touching samples!
        df_data = df_data.reset_index(drop=True)

        df_data_test = df_data_test[df_data_test.time > 1]  # train only over touching samples!
        df_data_test = df_data_test.reset_index(drop=True)

        train_df, valid_df = train_test_split(df_data, test_size=0.22, shuffle=False)
        test_df = df_data_test

        if self.num_channels == 3:
            norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        else:
            norm = transforms.Normalize([0.5], [0.5])

        if self.half_image:
            w, h = self.params['image_size'] // 2, self.params['image_size']
        else:
            w, h = self.params['image_size'], self.params['image_size']

        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((480 // 2, 480)),
            transforms.RandomResizedCrop((w, h), scale=(0.05, 1.0)),
            transforms.ToTensor(),
            norm,
        ])

        self.aug_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((w, h)),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(3),  # rotate +/- 10 degrees
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomChoice([
            #     transforms.RandomAdjustSharpness(2),
            #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            #
            # ]),
            transforms.ToTensor(),
            norm
        ])

        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((w, h)),
            transforms.ToTensor(),
            norm
        ])

        self.originalset = TactileDataset(self.model_params, train_df, output_type,
                                          self.train_transform, half_image=self.half_image, num_channels=self.num_channels
                                          , remove_ref=self.remove_ref)

        if self.params['aug']:
            self.augset = TactileDataset(self.model_params, train_df, output_type,
                                         self.aug_transform, half_image=self.half_image, num_channels=self.num_channels, remove_ref=self.remove_ref)
            self.trainset = torch.utils.data.ConcatDataset([self.originalset, self.augset])
        else:
            self.trainset = self.originalset

        self.validset = TactileDataset(self.model_params, valid_df, output_type, self.test_transform,
                                       half_image=self.half_image, num_channels=self.num_channels, remove_ref=self.remove_ref)
        self.testset = TactileDataset(self.model_params, test_df, output_type, self.test_transform,
                                      half_image=self.half_image, num_channels=self.num_channels, remove_ref=self.remove_ref)

        self.trainloader = DataLoader(self.trainset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        self.validloader = DataLoader(self.validset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)
        self.testloader = DataLoader(self.testset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True)

        print(f'Train set length is {len(self.trainset)}')
        print(f'Valid set length is {len(self.validset)}')
        print(f'Test set length is {len(self.testset)}')

    def run_training_loop(self):

        epochs = self.model_params['epoch']
        # init vars at beginning of training
        self.start_time = time.time()
        self.min_valid_loss = np.inf
        mean_train_loss = np.inf

        COSTS, EVAL_COSTS, BATCH_TRAIN_RMSE_LOSS, epoch_cost, eval_cost = [], [], [], [], []

        for epoch in range(epochs):

            self.model.train()
            self.mae.train()

            with tqdm(self.trainloader, unit="batch") as tepoch:
                for (batch_x, batch_x_ref, batch_y) in tepoch:
                    tepoch.set_description(f"Epoch [{epoch}/{epochs}]")

                    # pred_px = self.model(batch_x).to(device)
                    loss, image_recon = self.mae(batch_x)
                    # true_px = batch_y.to(device)
                    # loss = nn.functional.mse_loss(pred_px, true_px)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    cost = loss.item()
                    COSTS.append(cost)

                    # pred, true = self.unormalize(pred_px, true_px)
                    # rmse = self.rmse_loss(pred, true)

                    BATCH_TRAIN_RMSE_LOSS.append(cost)

                    torch.cuda.empty_cache()
                    tepoch.set_postfix(loss=cost, last_train_loss=mean_train_loss)

            rmse_train_loss = np.mean(BATCH_TRAIN_RMSE_LOSS, axis=0)
            mean_train_loss = np.mean(COSTS[-len(self.trainloader):])

            self.log_model_predictions(batch_x, image_recon, rmse_train_loss, 'train')

            EVAL_COSTS = self.run_validation_loop(EVAL_COSTS)
            mean_val_loss = np.mean(EVAL_COSTS[-len(self.validloader):])

            epoch_cost.append(mean_train_loss)
            eval_cost.append(mean_val_loss)

            # apply LR scheduler after each epoch
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(mean_val_loss)

            elif isinstance(self.scheduler, CosineAnnealingLR) or isinstance(self.scheduler, StepLR):
                self.scheduler.step()

            # Lets log a bit
            self.fig.clf()
            plt.plot(epoch_cost, '-ro', linewidth=3, label='train loss')
            plt.plot(eval_cost, '-ko', linewidth=3, label='val loss')
            plt.legend()
            self.fig.savefig(self.params['logdir'] + '/train_val_comp.png', dpi=200, bbox_inches='tight')

        # print("Optimization Finished!")

        np.save(self.params['logdir'] + '/train_val_comp.npy', [epoch_cost, eval_cost])
        self.fig.clf()
        plt.plot(epoch_cost, '-ro', linewidth=3, label='train loss')
        plt.plot(eval_cost, '-ko', linewidth=3, label='val loss')
        plt.legend()
        self.fig.savefig(self.params['logdir'] + '/train_val_comp.png', dpi=200, bbox_inches='tight')

    def run_validation_loop(self, EVAL_COSTS):

        self.model.eval()
        self.mae.eval()

        BATCH_VAL_RMSE_LOSS = []
        with tqdm(self.validloader, unit="batch") as tepoch:
            for (batch_x, batch_x_ref, batch_y) in tepoch:
                tepoch.set_description("Validate")

                with torch.no_grad():
                    # pred_px = self.model(batch_x).to(device)
                    # true_px = batch_y.to(device)
                    # cost = nn.functional.mse_loss(pred_px, true_px)

                    # pred, true = self.unormalize(pred_px, true_px)
                    # rmse = self.rmse_loss(pred, true)

                    loss, image_recon = self.mae(batch_x)
                    rmse = loss.item()

                EVAL_COSTS.append(rmse)
                BATCH_VAL_RMSE_LOSS.append(rmse)
                tepoch.set_postfix(loss=rmse, min_valid_loss=self.min_valid_loss)

        rmse_curr_valid_loss = np.mean(BATCH_VAL_RMSE_LOSS, axis=0)
        mean_curr_valid_loss = np.mean(EVAL_COSTS[-len(self.validloader):])

        if self.min_valid_loss > mean_curr_valid_loss:
            print(f'Validation Loss Decreased {self.min_valid_loss} ---> {mean_curr_valid_loss} \t Saving The Model')
            self.min_valid_loss = mean_curr_valid_loss
            torch.save(self.model.state_dict(), '%s/%s.pth' % (self.params['logdir'] + '/', 'model'))
            self.best_model = copy.deepcopy(self.model)
        self.run_test_loop()

        self.log_model_predictions(batch_x, image_recon, rmse_curr_valid_loss, 'valid')

        return EVAL_COSTS

    def run_test_loop(self):

        TEST_COSTS = []
        BATCH_TEST_RMSE_LOSS = []
        self.model.eval()
        self.mae.eval()
        # self.best_model.eval()

        for b, (batch_x, batch_x_ref, batch_y) in enumerate(self.testloader):
            with torch.no_grad():
                # pred_px = self.model(batch_x).to(device)
                # true_px = batch_y.to(device)
                # cost = nn.functional.mse_loss(pred_px, true_px)
                #
                # pred, true = self.unormalize(pred_px, true_px)
                # rmse = self.rmse_loss(pred, true)
                loss, image_recon = self.mae(batch_x)
                rmse = loss.item()

            TEST_COSTS.append(rmse)
            BATCH_TEST_RMSE_LOSS.append(rmse)

        rmse_curr_test_loss = np.mean(BATCH_TEST_RMSE_LOSS, axis=0)
        mean_curr_test_loss = np.mean(TEST_COSTS)
        print('\nTest loss : ' + str(mean_curr_test_loss))

        self.log_model_predictions(batch_x, image_recon, rmse_curr_test_loss, 'test')

    def unormalize(self, pred_px, true_px):
        # convert
        pred = pred_px.cpu().detach().numpy()
        true = true_px.cpu().detach().numpy()

        # unormalize
        if self.model_params['norm_method'] == 'meanstd':
            pred = unnormalize(pred, self.originalset.data_statistics['mean'], self.originalset.data_statistics['std'])
            true = unnormalize(true, self.originalset.data_statistics['mean'], self.originalset.data_statistics['std'])
        elif self.model_params['norm_method'] == 'maxmin':
            pred = unnormalize_max_min(pred, self.originalset.data_statistics['max'],
                                       self.originalset.data_statistics['min'])
            true = unnormalize_max_min(true, self.originalset.data_statistics['max'],
                                       self.originalset.data_statistics['min'])

        return pred, true

    def rmse_loss(self, pred, true):
        # calc rmse
        rmse = np.sqrt(np.mean((true - pred) ** 2, axis=0))

        return rmse

    def log_model_predictions(self, batch_x, pred, rmse, status):

        log_path = self.params['logdir'] + '/' + f'{status}_eval.txt'
        mode = 'a' if os.path.exists(log_path) else 'w'
        with open(log_path, mode) as f:
            f.write(f'rmse: {rmse}\n')

        # display visual model inputs
        if self.num_channels == 3:
            inv_normalize = transforms.Normalize(
                mean=[-1, -1, -1],
                std=[2.0, 2.0, 2.0]
            )
        else:
            inv_normalize = transforms.Normalize(
                mean=[-1],
                std=[2.0]
            )

        im_inv = inv_normalize(batch_x)

        self.fig.clf()
        im = make_grid(im_inv, nrow=4)
        plt.imshow(im.permute(1, 2, 0).cpu().detach().numpy())
        self.fig.savefig(self.params['logdir'] + '/' + 'visual_input_{}.png'.format(status), bbox_inches='tight')
        self.fig.clf()

        im_inv = inv_normalize(pred)
        self.fig.clf()
        im = make_grid(im_inv, nrow=4)
        plt.imshow(im.permute(1, 2, 0).cpu().detach().numpy())
        self.fig.savefig(self.params['logdir'] + '/' + 'visual_pred_{}.png'.format(status), bbox_inches='tight')
        self.fig.clf()


def main():
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', '-ep', type=int, default=20)

    parser.add_argument('--train_type', '-dt', type=str, default='real')  # real, sim, gan
    parser.add_argument('--data_kind', type=str, default='transformed', help='transformed, aligned')
    parser.add_argument('--sim_data_num', type=int, default=8, help='sim JSON path')
    parser.add_argument('--real_data_num', type=int, default=8, help='real JSON path')
    parser.add_argument('--num_channels', type=int, default=1, help='channel')
    parser.add_argument('--half_image', action='store_true', default=True)
    parser.add_argument('--remove_ref', action='store_true', default=True)

    parser.add_argument('--gan_num', default=1, type=str)
    parser.add_argument('--gan_epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')

    parser.add_argument('--frame_type', type=str, default='frame', help='frame or diff_frame')

    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--portion', '-pr', type=float, default=1.0)
    parser.add_argument('--model_name', '-mn', type=str, default='resnet18')
    parser.add_argument('--input_type', '-it', type=str, default='single')  # with_ref_6c, single
    parser.add_argument('--leds', '-ld', type=str, default='combined')  # rrrgggbbb
    parser.add_argument('--gel', '-gl', type=str, default='combined')  # rrrgggbbb

    parser.add_argument('--norm_method', '-im', type=str, default='meanstd')
    parser.add_argument('--aug', '-aug', default=False)

    parser.add_argument('--output', '-op', type=str, default='pose_force')
    parser.add_argument('--scheduler', '-sch', type=str, default='none')

    parser.add_argument('--image_size', '-iz', type=int, default=224)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_gpu', '-gpu', default=True)
    parser.add_argument('--which_gpu', type=int, default=0)

    args = parser.parse_args()

    # convert to dictionary
    params = vars(args)

    if torch.cuda.is_available() and params["use_gpu"]:
        which_gpu = "cuda:" + str(params["which_gpu"])
        params["device"] = torch.device(which_gpu)
        print("Pytorch is running on GPU", params["which_gpu"])
    else:
        params["device"] = torch.device("cpu")
        print("Pytorch is running on the CPU")

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    path_to_log = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'train_history/{}'.format(params['train_type']))

    if not (os.path.exists(path_to_log)):
        os.makedirs(path_to_log)

    logdir = 'train' + '_' + params['output']
    logdir += '_' + params['model_name']
    logdir += '_' + params['input_type']
    logdir += '_aug' if params['aug'] else ''
    logdir += '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(path_to_log, logdir)
    params['logdir'] = logdir

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\nLOGGING TO: ", logdir, "\n")

    ###################
    ### RUN TRAINING & TESTING
    ###################

    trainer = Trainer(params)
    trainer.run_training_loop()
    trainer.run_test_loop()


if __name__ == "__main__":
    main()
