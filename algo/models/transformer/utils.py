import numpy as np
import random
import torch
import os
from torch import nn
from torchvision import transforms
from torchvision.transforms import v2
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F


class SyncCenterReshapeTransform(nn.Module):
    def __init__(self, crop_size, img_transform, mask_transform):
        super(SyncCenterReshapeTransform, self).__init__()
        self.img_transform = img_transform  # ImageTransform(img_transform)
        self.mask_transform = mask_transform  # ImageTransform(mask_transform)
        self.crop_size = crop_size

    def forward(self, img, mask):
        # Reshape img and mask to [B * T, C, H, W]
        B, T, C, H, W = img.shape
        img = img.view(-1, C, H, W)
        mask = mask.view(-1, C, H, W)

        # Apply synchronized crop
        img = F.center_crop(img, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)

        # Apply the rest of the transformations
        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # Reshape back to [B, T, C, H, W]
        img = img.view(B, T, C, *img.shape[2:])
        mask = mask.view(B, T, C, *mask.shape[2:])

        return img, mask


class SyncRandomReshapeTransform(nn.Module):
    def __init__(self, crop_size, img_transform, mask_transform):
        super(SyncRandomReshapeTransform, self).__init__()
        self.crop_size = crop_size
        self.img_transform = ImageTransform(img_transform)
        self.mask_transform = ImageTransform(mask_transform)

    def forward(self, img, mask):
        # Reshape img and mask to [B * T, C, H, W]
        B, T, C, H, W = img.shape
        img = img.view(-1, C, H, W)
        mask = mask.view(-1, C, H, W)

        # Apply synchronized crop
        img, mask = self.sync_crop(img, mask)

        # Apply the rest of the transformations
        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # Reshape back to [B, T, C, H, W]
        img = img.view(B, T, C, *img.shape[2:])
        mask = mask.view(B, T, C, *mask.shape[2:])

        # Reset the crop parameters for the next image-mask pair
        self.sync_crop.reset()

        return img, mask


class ImageTransform:
    def __init__(self, image_transform=None):
        self.image_transform = image_transform

    def __call__(self, img_input):
        # img_input shape: [B, T, C, H, W]
        B, T, C, H, W = img_input.shape
        img_input = img_input.view(-1, C, H, W)  # Shape: [B * T, C, H, W]
        if self.image_transform is not None:
            img_input = self.image_transform(img_input)

        img_input = img_input.view(B, T, C, *img_input.shape[2:])  # Reshape back to [B, T, C, new_H, new_W]
        return img_input


class SyncTransform(nn.Module):
    def __init__(self, crop_size, downsample, img_transform, mask_transform):
        super(SyncTransform, self).__init__()
        self.sync_crop = SyncRandomCrop(crop_size)
        self.downsample = downsample
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def forward(self, img, mask):
        # Apply synchronized crop
        img = self.downsample(img)
        mask = self.downsample(mask)

        img, mask = self.sync_crop(img, mask)
        # Apply the rest of the transformations
        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        # Reset the crop parameters for the next image-mask pair
        self.sync_crop.reset()

        return img, mask


class SyncEvalTransform(nn.Module):
    def __init__(self, crop_size, downsample, img_transform, mask_transform):
        super(SyncEvalTransform, self).__init__()
        self.crop_size = crop_size
        self.downsample = downsample
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def forward(self, img, mask):
        # Apply synchronized crop
        img = self.downsample(img)
        mask = self.downsample(mask)

        img = F.center_crop(img, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)
        # Apply the rest of the transformations
        img = self.img_transform(img)
        mask = self.mask_transform(mask)

        return img, mask


class TactileTransform:
    def __init__(self, tactile_transform=None):
        self.tactile_transform = tactile_transform
        self.to_gray = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()  # Convert PIL Image back to tensor
        ])

    def __call__(self, tac_input):
        # tac_input shape: [B, T, Num_cam, C, H, W]
        B, T, F, C, H, W = tac_input.shape
        tac_input = tac_input.view(-1, C, H, W)  # Shape: [B * T * F, C, H, W]

        transformed_list = []

        for i in range(tac_input.shape[0]):
            transformed_image = self.tactile_transform(tac_input[i])

            transformed_list.append(transformed_image)
        tac_input = torch.stack(transformed_list)

        tac_input = tac_input.view(B, T, F, C,
                                   *tac_input.shape[2:])  # Reshape back to [B, T, F, C, new_H, new_W]

        return tac_input


def set_seed(seed, torch_deterministic=False, rank=0):
    """ set seed across modules """
    if seed == -1 and torch_deterministic:
        seed = 42 + rank
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    else:
        seed = seed + rank

    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed

class Masking(nn.Module):
    def __init__(self, img_patch_size, img_masking_prob):
        super().__init__()
        self.img_patch_size = img_patch_size
        self.img_masking_prob = img_masking_prob

    def forward(self, x):
        img_patch = x.unfold(2, self.img_patch_size, self.img_patch_size).unfold(
            3, self.img_patch_size, self.img_patch_size
        )
        mask = (
                torch.rand(
                    (
                        x.shape[0],
                        x.shape[-2] // self.img_patch_size,
                        x.shape[-1] // self.img_patch_size,
                    )
                )
                < self.img_masking_prob
        )
        mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
        x = x.clone()
        x.unfold(2, self.img_patch_size, self.img_patch_size).unfold(
            3, self.img_patch_size, self.img_patch_size
        )[mask] = 0
        return x


def define_tactile_transforms(width, height, crop_width, crop_height,
                              img_patch_size=16, img_gaussian_noise=0.0, img_masking_prob=0.0):

    downsample = transforms.Resize((width, height), interpolation=transforms.InterpolationMode.BILINEAR)

    transform = nn.Sequential(
        downsample,
        transforms.RandomCrop((crop_width, crop_height)),
        # v2.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=-0.5, inplace=False),
        # v2.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.3),
        transforms.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.01, 0.1))], p=0.5),
        transforms.RandomApply([v2.RandomRotation(degrees=3)], p=0.5)
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
                transform,
            )(x)
        )

    # For evaluation, only center crop and normalize
    eval_transform = nn.Sequential(
        downsample,
        transforms.CenterCrop((crop_width, crop_height)),
    )

    return transform, eval_transform


# Define your transforms
def define_img_transforms(width, height, crop_width, crop_height,
                          img_patch_size=16, img_gaussian_noise=0.0, img_masking_prob=0.0):
    downsample = transforms.Resize((width, height), interpolation=transforms.InterpolationMode.BILINEAR)

    transform = nn.Sequential(
        # v2.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=-0.5, inplace=False),
        # v2.RandomErasing(p=0.4, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False),
        # v2.GaussianBlur(kernel_size=3, sigma=(0.01, 0.1)),
        v2.RandomRotation(degrees=1)
    )

    mask_transform = nn.Sequential(
        # v2.GaussianBlur(kernel_size=3, sigma=(0.01, 0.1)),
        v2.RandomRotation(degrees=1)
    )

    # Add gaussian noise to the image
    if img_gaussian_noise > 0.0:
        transform = nn.Sequential(
            transform,
            GaussianNoise(img_gaussian_noise),
        )
        mask_transform = nn.Sequential(
            mask_transform,
            GaussianNoise(img_gaussian_noise),
        )

    if img_masking_prob > 0.0:
        transform = nn.Sequential(
            transform,
            Masking(img_patch_size, img_masking_prob),
        )

    # For evaluation, only center crop and normalize
    eval_transform = nn.Sequential(
        downsample,
        transforms.CenterCrop((crop_width, crop_height)),
    )

    sync_transform = SyncTransform((crop_width, crop_height), downsample, transform, mask_transform)
    sync_eval_transform = SyncEvalTransform((crop_width, crop_height), downsample, eval_transform, mask_transform)

    return transform, mask_transform, eval_transform, sync_transform, sync_eval_transform


def define_transforms(channel, color_jitter, width, height, crop_width,
                      crop_height, img_patch_size, img_gaussian_noise=0.0, img_masking_prob=0.0, ):
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
        transforms.Resize(
            (width, height),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
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
        transforms.Resize(
            (width, height),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
        transforms.CenterCrop((crop_width, crop_height)),
    )
    # print('transform {}'.format(transform))
    # print('eval_transform {}'.format(eval_transform))
    # print('downsample {}'.format(downsample))

    return transform, downsample, eval_transform


class SyncRandomCrop(nn.Module):
    def __init__(self, crop_size):
        super(SyncRandomCrop, self).__init__()
        self.crop_size = crop_size
        self.crop_params = None

    def forward(self, img, mask):
        if self.crop_params is None:
            # Generate crop parameters
            i, j, h, w = transforms.RandomCrop.get_params(img, self.crop_size)
            self.crop_params = (i, j, h, w)
        else:
            # Use stored crop parameters
            i, j, h, w = self.crop_params

        img = F.crop(img, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        return img, mask

    def reset(self):
        self.crop_params = None


class SyncCenterCrop(nn.Module):
    def __init__(self, crop_size):
        super(SyncCenterCrop, self).__init__()
        self.crop_size = crop_size

    def forward(self, img, mask):
        img = F.center_crop(img, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)
        return img, mask


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class CenterCropTransform(nn.Module):
    def __init__(self, crop_size, center):
        super().__init__()
        self.crop_size = crop_size
        self.center = center

    def forward(self, img):
        _, _, height, width = img.size()
        crop_width, crop_height = self.crop_size

        center_x, center_y = self.center

        # Ensure the center is within the bounds
        center_x = max(crop_width // 2, min(center_x, width - crop_width // 2))
        center_y = max(crop_height // 2, min(center_y, height - crop_height // 2))

        left = center_x - crop_width // 2
        top = center_y - crop_height // 2
        right = center_x + crop_width // 2
        bottom = center_y + crop_height // 2

        img = img[:, :, top:bottom, left:right]
        return img


def mask_img(x, img_patch_size, img_masking_prob):
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


def log_outpu2t(tac_input, img_input, seg_input, lin_input, out, latent, pos_rpy, save_folder, d_pos_rpy=None,
                session='train'):
    # Selecting the first example from the batch for demonstration
    # tac_input [B T F W H C]

    # image_sequence = tac_input[0].cpu().detach().numpy()
    img_input = img_input[0].cpu().detach().numpy()
    seg_input = seg_input[0].cpu().detach().numpy()

    linear_features = lin_input[0].cpu().detach().numpy()
    if d_pos_rpy is not None:
        d_pos_rpy = d_pos_rpy[0, -1, :].cpu().detach().numpy()
    pos_rpy = pos_rpy[0, -1, :].cpu().detach().numpy()

    predicted_output = out[0].cpu().detach().numpy()
    true_label = latent[0, -1, :].cpu().detach().numpy()
    # Plotting
    fig = plt.figure(figsize=(20, 10))

    # Adding subplot for image sequence (adjust as needed)
    # ax1 = fig.add_subplot(2, 2, 1)
    # concat_images = []
    # # image_sequence [T F W H C]
    # for finger_idx in range(image_sequence.shape[1]):
    #     finger_sequence = [np.transpose(img, (1, 2, 0)) for img in image_sequence[:, finger_idx, ...]]
    #     finger_sequence = np.hstack(finger_sequence)
    #     concat_images.append(finger_sequence)
    #
    # ax1.imshow(np.vstack(concat_images) + 0.5)  # Adjust based on image normalization
    # ax1.set_title('Input Tactile Sequence')

    # Adding subplot for linear features (adjust as needed)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax2.plot(d_pos_rpy[:, :], 'ok', label='hand_joints')  # Assuming the rest are actions
    # ax2.set_title('Linear input')
    # ax2.legend()

    # if d_pos_rpy is not None:
    #     ax2 = fig.add_subplot(2, 2, 2)
    #     width = 0.35
    #     indices = np.arange(len(d_pos_rpy))
    #     ax2.bar(indices - width / 2, d_pos_rpy, width, label='d_pos_rpy')
    #     ax2.bar(indices + width / 2, pos_rpy, width, label='True Label')
    #     ax2.set_title('Model Output vs. True Label')
    #     ax2.legend()

    # Check if img_input has more than one timestep
    if seg_input.ndim == 4 and seg_input.shape[0] > 1:
        concat_seg_input = []
        for t in range(seg_input.shape[0]):
            seg = seg_input[t]
            seg = np.transpose(seg, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
            concat_seg_input.append(seg)

        # Horizontally stack the images for each timestep
        concat_seg_input = np.hstack(concat_seg_input)
    else:
        # Handle the case where there is only one timestep
        seg = seg_input[0] if seg_input.ndim == 4 else seg_input
        seg = np.transpose(seg, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
        concat_seg_input = seg

    # Plot the concatenated image sequence
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(concat_seg_input)
    ax2.set_title('Input Seg Sequence')

    # Check if img_input has more than one timestep
    if img_input.ndim == 4 and img_input.shape[0] > 1:
        concat_img_input = []
        for t in range(img_input.shape[0]):
            img = img_input[t]
            img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
            concat_img_input.append(img)

        # Horizontally stack the images for each timestep
        concat_img_input = np.hstack(concat_img_input)
    else:
        # Handle the case where there is only one timestep
        img = img_input[0] if img_input.ndim == 4 else img_input
        img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
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
    plt.savefig(f'{save_folder}/{session}_example.png')
    # Clean up plt to free memory
    plt.close(fig)


def plot_image_sequence(ax, sequence, title):
    if sequence is not None:
        if sequence.ndim == 4 and sequence.shape[0] > 1:
            concat_sequence = []
            for t in range(sequence.shape[0]):
                img = sequence[t]
                img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
                concat_sequence.append(img)
            concat_sequence = np.hstack(concat_sequence)
        else:
            img = sequence[0] if sequence.ndim == 4 else sequence
            img = np.transpose(img, (1, 2, 0))  # Convert from [W, H, C] to [H, W, C]
            concat_sequence = img
        ax.imshow(concat_sequence)
        ax.set_title(title)


def plot_bar_comparison(ax, predicted, true, title):
    width = 0.35
    indices = np.arange(len(predicted))
    ax.bar(indices - width / 2, predicted, width, label='Predicted')
    ax.bar(indices + width / 2, true, width, label='True Label')
    ax.set_title(title)
    ax.legend()


def log_output(tac_input, img_input, seg_input, lin_input, out, latent, pos_rpy, save_folder, d_pos_rpy=None,
               session='train'):
    img_input = img_input[0].cpu().detach().numpy() if img_input.ndim > 2 else None
    seg_input = seg_input[0].unsqueeze(1).cpu().detach().numpy() if seg_input.ndim > 2 else None
    linear_features = lin_input[0].cpu().detach().numpy() if lin_input is not None else None
    d_pos_rpy = d_pos_rpy[0, -1, :].cpu().detach().numpy() if d_pos_rpy is not None else None
    pos_rpy = pos_rpy[0, -1, :].cpu().detach().numpy() if pos_rpy is not None else None
    predicted_output = out[0].cpu().detach().numpy() if out is not None else None
    true_label = latent[0, -1, :].cpu().detach().numpy() if latent is not None else None
    tac_input = tac_input[0].cpu().detach().numpy() if tac_input.ndim > 2 else None

    fig = plt.figure(figsize=(20, 10))

    if tac_input is not None:
        ax1 = fig.add_subplot(2, 2, 1)
        plot_image_sequence(ax1, tac_input, 'Input Tactile Sequence')

    if seg_input is not None:
        ax2 = fig.add_subplot(2, 2, 2)
        plot_image_sequence(ax2, seg_input, 'Input Seg Sequence')

    if img_input is not None:
        ax3 = fig.add_subplot(2, 2, 3)
        plot_image_sequence(ax3, img_input, 'Input Image Sequence')

    if predicted_output is not None and true_label is not None:
        ax4 = fig.add_subplot(2, 2, 4)
        plot_bar_comparison(ax4, predicted_output, true_label, 'Model Output vs. True Label')

    fig.savefig(f'{save_folder}/{session}_example.png')

