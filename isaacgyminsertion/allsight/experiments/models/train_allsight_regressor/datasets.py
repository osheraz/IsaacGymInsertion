import os
import torch
from torch.utils.data import DataLoader
from misc import normalize, unnormalize, normalize_max_min, unnormalize_max_min  #
import numpy as np
import cv2
from img_utils import circle_mask
import pandas as pd
import random
from glob import glob
import json
from PIL import Image

np.set_printoptions(precision=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
pc_name = os.getcwd()

output_map = {'pixel': 3,  # px, py, r
              'force': 3,  # fx, fy, f
              'torque': 3,  # mx, my, mz
              'force_torque_full': 6,  # fx, fy, fz, mx, my, mz
              'force_torque': 4,  # fx, fy, fz, mx, my, mz
              'pose': 3,  # x, y ,z
              'pose_force': 6,  # fx, fy, fz
              'touch_detect': 1,  # bool
              'depth': 1,  # max depth
              # order by name
              'pose_force_torque': 7,
              'pose_force_pixel': 9,
              'pose_force_pixel_depth': 10,
              'pose_force_pixel_torque_depth': 11,
              'pose_force_pixel_torque': 10
              }

sensor_dict = {'markers': {'rrrgggbbb': [2, 1, 0, 11],
                           'rgbrgbrgb': [7, 8],
                           'white': [5, 6]},
               'clear': {'rrrgggbbb': [3, 10],
                         'rgbrgbrgb': [9],
                         'white': [4]}
               }


def get_buffer_paths(leds, gel, indenter, sensor_id=None):
    trained_sensor_id = []
    test_sensor_id = []

    buffer_paths_to_train = []
    buffer_paths_to_test = []

    if leds == 'combined':
        leds_list = ['rrrgggbbb', 'rgbrgbrgb', 'white']
    else:
        leds_list = [leds]

    for l in leds_list:
        paths = [f"/home/{pc_name}/catkin_ws/src/allsight/dataset/{gel}/{l}/data/{ind}" for ind in indenter]
        buffer_paths = []
        summ_paths = []

        for p in paths:
            buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*_transformed_annotated.json'))]
            summ_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], 'summary.json'))]

        for bp, s in zip(buffer_paths, summ_paths):
            with open(s, 'rb') as handle:
                summ = json.load(handle)
                summ['sensor_id'] = summ['sensor_id'] if isinstance(summ['sensor_id'], list) else [summ['sensor_id']]

            for sm in summ['sensor_id']:
                if sensor_id is not None:
                    train_sensor = sensor_id
                else:
                    train_sensor = sensor_dict[gel][l][0]

                if sm == train_sensor:
                    buffer_paths_to_train.append(bp)
                    trained_sensor_id.append(sm)

                else:
                    buffer_paths_to_test.append(bp)
                    test_sensor_id.append(sm)

    return buffer_paths_to_train, buffer_paths_to_test, list(set(trained_sensor_id)), list(set(test_sensor_id))


def get_buffer_paths_clear(leds, indenter, params=None):
    if leds == 'combined':
        leds_list = ['rrrgggbbb', 'rgbrgbrgb', 'white']
    else:
        leds_list = [leds]

    buffer_paths = []
    for l in leds_list:
        paths = [(f"/common/home/oa348/Downloads/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/allsight/"
                  f"experiments/allsight_dataset/clear/{l}/data/{ind}") for ind in indenter]
        for p in paths:
            buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*transformed_annotated.json'))]

    return buffer_paths


def get_inputs_and_targets(group, output_type):
    inputs = [group.iloc[idx].frame for idx in range(group.shape[0])]
    inputs_ref = [group.iloc[idx].ref_frame for idx in range(group.shape[0])]

    if output_type == 'pixel':
        target = np.array(group.iloc[idx].contact_px for idx in range(group.shape[0]))
    elif output_type == 'force':
        target = np.array([group.iloc[idx].ft_ee_transformed[:3] for idx in range(group.shape[0])])
    elif output_type == 'force_torque':
        target = np.array(
            [group.iloc[idx].ft_ee_transformed[0, 1, 2, 5] for idx in range(group.shape[0])])
    elif output_type == 'pose':
        target = np.array([group.iloc[idx].pose_transformed[0][:3] for idx in range(group.shape[0])])
    elif output_type == 'pose_force':
        target = np.array([np.hstack((group.iloc[idx].pose_transformed[0][:3],
                                      group.iloc[idx].ft_ee_transformed[:3])) for idx in
                           range(group.shape[0])])
    elif output_type == 'depth':
        target = np.array([group.iloc[idx].depth for idx in range(group.shape[0])])
    elif output_type == 'pose_force_torque':
        target = np.array([np.hstack((group.iloc[idx].pose_transformed[0][:3],
                                      group.iloc[idx].ft_ee_transformed[:3],
                                      group.iloc[idx].ft_ee_transformed[5])) for idx in range(group.shape[0])])
    elif output_type == 'pose_force_pixel':
        target = np.array([np.hstack((group.iloc[idx].pose_transformed[0][:3],
                                      group.iloc[idx].ft_ee_transformed[:3],
                                      group.iloc[idx].contact_px)) for idx in range(group.shape[0])])
    elif output_type == 'pose_force_pixel_depth':
        target = np.array([np.hstack((group.iloc[idx].pose_transformed[0][:3],
                                      group.iloc[idx].ft_ee_transformed[:3],
                                      group.iloc[idx].contact_px,
                                      group.iloc[idx].depth)) for idx in range(group.shape[0])])
    elif output_type == 'pose_force_pixel_torque':
        target = np.array([np.hstack((group.iloc[idx].pose_transformed[0][:3],
                                      group.iloc[idx].ft_ee_transformed[:3],
                                      group.iloc[idx].contact_px,
                                      group.iloc[idx].ft_ee_transformed[5])) for idx in
                           range(group.shape[0])])
    elif output_type == 'pose_force_pixel_torque_depth':
        target = np.array([np.hstack((group.iloc[idx].pose_transformed[0][:3],
                                      group.iloc[idx].ft_ee_transformed[:3],
                                      group.iloc[idx].contact_px,
                                      group.iloc[idx].ft_ee_transformed[5],
                                      group.iloc[idx].depth)) for idx in range(group.shape[0])])

    else:
        target = None
        print('please enter a valid output type')

    return inputs, inputs_ref, target


class TactileDataset(torch.utils.data.Dataset):
    def __init__(self, params, df, output_type='pose_force_pixel',
                 transform=None, apply_mask=False, remove_ref=True, statistics=None):

        self.df = df
        self.transform = transform
        self.output_type = output_type
        self.norm_method = params['norm_method']
        self.normalize_output = True if params['norm_method'] != 'none' else False
        self.apply_mask = apply_mask
        self.remove_ref = remove_ref
        self.w, self.h = 480, 480
        # define the labels:
        self.X, self.X_ref, self.Y = get_inputs_and_targets(df, self.output_type)

        if self.apply_mask:
            self.mask = circle_mask((self.w, self.h))

        if pc_name != 'osher':
            old_path = "/home/osher/catkin_ws/src/allsight/dataset/"
            new_path = os.path.dirname(os.path.abspath(__file__)) + '/../../allsight_dataset/'

            self.X = [f.replace(old_path, new_path) for f in self.X]
            self.X_ref = [f.replace(old_path, new_path) for f in self.X_ref]

        self.Y_ref = [[0] if df.iloc[idx].ref_frame == df.iloc[idx].frame else [1] for idx in range(self.df.shape[0])]

        if statistics is None:
            self.y_mean = self.Y.mean(axis=0)
            self.y_std = self.Y.std(axis=0)
            self.y_max = self.Y.max(axis=0)
            self.y_min = self.Y.min(axis=0)
        else:
            self.y_mean = np.array(statistics['mean'])
            self.y_std = np.array(statistics['std'])
            self.y_max = np.array(statistics['max'])
            self.y_min = np.array(statistics['min'])

        self.data_statistics = {'mean': self.y_mean, 'std': self.y_std, 'max': self.y_max, 'min': self.y_min}

    def __len__(self):
        return len(self.df)

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff

    def __getitem__(self, idx):

        img = cv2.imread(self.X[idx])
        ref_img = cv2.imread(self.X_ref[idx])

        if self.remove_ref:
            img = self._subtract_bg(img, ref_img)

        if self.apply_mask:
            img = (255 * img * self.mask).astype(np.uint8)
            ref_img = (ref_img * self.mask).astype(np.uint8)

        img = cv2.cvtColor((255 * img).astype(np.uint8), cv2.COLOR_BGR2RGB)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)

            img = self.transform(img).to(device)

            random.seed(seed)
            torch.manual_seed(seed)
            ref_img = self.transform(ref_img).to(device)

        y = torch.Tensor(self.Y[idx])
        y_ref = torch.Tensor(self.Y_ref[idx])  # contact or no contact

        if self.normalize_output:
            if self.norm_method == 'maxmin':
                y = normalize_max_min(y, self.data_statistics['max'], self.data_statistics['min']).float()
            elif self.norm_method == 'meanstd':
                y = normalize(y, self.data_statistics['mean'], self.data_statistics['std']).float()

        return img, ref_img, y  # , y_ref


class TactileSimDataset(torch.utils.data.Dataset):
    def __init__(self, params, df, output_type='pose',
                 transform=None, apply_mask=True, remove_ref=False, diff=True, statistics=None):

        self.df = df
        self.transform = transform
        self.output_type = output_type
        self.norm_method = params['norm_method']
        self.normalize_output = True if params['norm_method'] != 'none' else False
        self.apply_mask = apply_mask
        self.remove_ref = remove_ref
        self.w, self.h = 480, 480
        self.diff = diff
        self.X = [df.iloc[idx].frame for idx in range(self.df.shape[0])]
        # self.X_ref = ['/'.join(df.iloc[0].frame.split('/')[:-1]) + '/ref_frame.jpg' for idx in range(self.df.shape[0])]
        self.X_ref = self.X

        if self.apply_mask:
            self.mask = circle_mask((self.w, self.h))

        if pc_name != 'osher':
            self.X = [f.replace('osher', 'roblab20') for f in self.X]
            self.X_ref = [f.replace('osher', 'roblab20') for f in self.X_ref]

        # define the labels
        self.Y = np.array([df.iloc[idx].pose_transformed[0][:3] for idx in range(self.df.shape[0])])

        if statistics is None:
            self.y_mean = self.Y.mean(axis=0)
            self.y_std = self.Y.std(axis=0)
            self.y_max = self.Y.max(axis=0)
            self.y_min = self.Y.min(axis=0)
        else:
            self.y_mean = np.array(statistics['mean'])
            self.y_std = np.array(statistics['std'])
            self.y_max = np.array(statistics['max'])
            self.y_min = np.array(statistics['min'])

        self.data_statistics = {'mean': self.y_mean, 'std': self.y_std, 'max': self.y_max, 'min': self.y_min}

    def __len__(self):
        return len(self.df)

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff

    def __getitem__(self, idx):

        img = cv2.imread(self.X[idx])
        ref_img = cv2.imread(self.X_ref[idx])

        if self.remove_ref:
            img = img - ref_img

        if self.apply_mask:
            img = (img * self.mask).astype(np.uint8)
            ref_img = (ref_img * self.mask).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ref_img = img
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            # Make sure both ref and frame has the same transfrom
            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)

            img = self.transform(img).to(device)

            random.seed(seed)
            torch.manual_seed(seed)
            ref_img = self.transform(ref_img).to(device)

        y = torch.Tensor(self.Y[idx])

        if self.normalize_output:
            if self.norm_method == 'maxmin':
                y = normalize_max_min(y, self.data_statistics['max'], self.data_statistics['min']).float()
            elif self.norm_method == 'meanstd':
                y = normalize(y, self.data_statistics['mean'], self.data_statistics['std']).float()

        return img, ref_img, y


class TactileTouchDataset(torch.utils.data.Dataset):

    def __init__(self, params, df, output_type,
                 transform=None, normalize_output=False, apply_mask=True, remove_ref=False):

        self.df = df
        self.transform = transform
        self.output_type = output_type
        self.normalize_output = normalize_output
        self.apply_mask = apply_mask
        self.remove_ref = remove_ref
        self.w, self.h = cv2.imread(df.frame[0])[:2]

        img_name = params['buffer_name'][0].replace('data', 'img')
        data_path = f'/home/{pc_name}/catkin_ws/src/allsight/dataset/'
        ref_path = data_path + f"images/{img_name}/ref_frame.jpg"

        if self.apply_mask:
            self.mask = circle_mask((self.w, self.h))

        self.ref_frame = (cv2.cvtColor(cv2.imread(ref_path), cv2.COLOR_BGR2RGB) * self.mask).astype(np.uint8)
        self.ref_frame = self.transform(np.array(self.ref_frame)).to(device)

        self.X = [df.iloc[idx].frame for idx in range(self.df.shape[0])]
        self.X_ref = [df.iloc[idx].ref_frame for idx in range(self.df.shape[0])]

        if pc_name != 'osher':
            self.X = [f.replace('osher', 'roblab20') for f in self.X]
            self.X_ref = [f.replace('osher', 'roblab20') if not pd.isna(f) else ref_path for f in self.X_ref]

        self.Y = np.array([df.iloc[idx].touch for idx in range(self.df.shape[0])])

        self.data_statistics = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img = cv2.imread(self.X[idx])
        # ref_img = cv2.imread(self.X_ref[idx])

        # img = self.X[idx]
        # ref_img = self.X_ref[idx]

        # if self.remove_ref:
        #     img = img - ref_img

        if self.apply_mask:
            img = (img * self.mask).astype(np.uint8)
            # ref_img = (ref_img * self.mask).astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        # diff = _diff(img, ref_img)

        if self.transform:
            img = self.transform(img).to(device)
            # ref_img = self.transform(ref_img).to(device)
            # diff = self.transform(diff).to(device)

        y = torch.Tensor([self.Y[idx]])

        return img, y


class CircleMaskTransform(object):
    """Apply a circular mask to an input image using OpenCV.

    Args:
        size (tuple): Desired size of the output mask.
        border (int): Border thickness of the circular mask.
    """

    def __init__(self, size=(224, 224), border=10):
        self.size = size
        self.border = border

    def __call__(self, image):
        # Convert PIL image to NumPy array
        image = np.array(image)

        mask = np.zeros((self.size[0], self.size[1]), dtype=np.uint8)
        m_center = (self.size[0] // 2, self.size[1] // 2)
        m_radius = min(self.size[0], self.size[1]) // 2 - self.border
        mask = cv2.circle(mask, m_center, m_radius, 1, thickness=-1)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        # Apply the circular mask to the input image
        masked_image = np.multiply(image, mask)

        # Convert back to PIL image
        masked_image = Image.fromarray(masked_image.astype('uint8'), 'RGB')

        return masked_image