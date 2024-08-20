from isaacgyminsertion.utils.torch_jit_utils import *
from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R

from typing import Union
import pytorch3d.transforms as pt
import torch
import numpy as np
import functools


class DepthImageProcessor:
    def __init__(self, cfg, dis_noise, far_clip, near_clip):
        self.cfg = cfg
        self.dis_noise = dis_noise
        self.far_clip = far_clip
        self.near_clip = near_clip
        # self.resize_transform = transforms.Resize(
        #     (self.cfg.depth.resized[1], self.cfg.depth.resized[0]),
        #     interpolation=transforms.InterpolationMode.BICUBIC
        # )

    def add_seg_noise(self, seg_images_to_noise, flip_prob=0.1):

        # flip inside the segmented object
        object_mask = (seg_images_to_noise > 0)

        flip_mask = torch.rand_like(seg_images_to_noise, dtype=torch.float) < flip_prob

        seg_images_to_noise[object_mask & flip_mask] = 0

        # flip inside the background
        # background_mask = (seg_images_to_noise == 0)
        # seg_images_to_noise[background_mask] = torch.randint(1, 4, size=(background_mask.sum().item(),),
        #                                             device=seg_images_to_noise.device)

        return seg_images_to_noise

    def add_pcl_noise(self, pcl_to_noise, flip_prob=0.1):

        # flip inside the segmented object
        object_mask = (pcl_to_noise > 0)

        flip_mask = torch.rand_like(pcl_to_noise, dtype=torch.float) < flip_prob

        pcl_to_noise[object_mask & flip_mask] = 0

        # flip inside the background
        # background_mask = (pcl_to_noise == 0)
        # pcl_to_noise[background_mask] = torch.randint(1, 4, size=(background_mask.sum().item(),),
        #                                             device=pcl_to_noise.device)

        return pcl_to_noise

    def process_depth_image(self, depth_images):
        # Ensure the input is in the correct shape
        # if depth_images.dim() == 3:
        #     depth_images = depth_images.unsqueeze(0)

        # depth_images = self.crop_depth_image(depth_images)
        noise = self.dis_noise * 2 * (torch.rand(depth_images.shape, device=depth_images.device) - 0.5)
        depth_images += noise
        depth_images = torch.clip(depth_images, -self.far_clip, -self.near_clip)
        depth_images = self.normalize_depth_image(depth_images)
        # depth_images = self.resize_depth_images(depth_images)

        return depth_images.squeeze(0) if depth_images.size(0) == 1 else depth_images

    def normalize_depth_image(self, depth_images):
        depth_images = depth_images * -1
        depth_images = (depth_images - self.near_clip) / (self.far_clip - self.near_clip)
        return depth_images

    def crop_depth_image(self, depth_images):
        # Crop 30 pixels from the left and right and 20 pixels from the bottom and return cropped image
        return depth_images
        # return depth_images[:, :, 30:-30, :-20]

    # def resize_depth_images(self, depth_images):
    #     resized_images = torch.cat([self.resize_transform(img).unsqueeze(0) for img in depth_images])
    #     return resized_images



class PointCloudAugmentations:
    def __init__(self,
                 num_points=400, sigma=0.005, noise_clip=0.003, rotate_range=(-10, 10), scale_range=(0.8, 1.2)):
        self.num_points = num_points  # Fixed number of points
        self.sigma = sigma            # Noise standard deviation
        self.noise_clip = noise_clip  # Clipping value for noise
        self.rotate_range = rotate_range  # Rotation range in degrees
        self.scale_range = scale_range    # Scaling range

    def random_noise(self, pointcloud_batch, noise_prob=0.3):
        """
        Add Gaussian noise to a subset of the points in the batch of point clouds.

        :param pointcloud_batch: Tensor of shape [B, N, 3]
        :param noise_prob: Probability of applying noise to each point (default is 50%).
        :return: Point cloud batch with noise applied selectively.
        """
        B, N, _ = pointcloud_batch.shape
        noise = torch.randn(pointcloud_batch.size(), device=pointcloud_batch.device) * self.sigma
        noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)

        noise_mask = torch.rand(B, N, device=pointcloud_batch.device) < noise_prob
        noise_mask = noise_mask.unsqueeze(-1).expand_as(pointcloud_batch)

        pointcloud_batch = pointcloud_batch + noise * noise_mask.float()

        return pointcloud_batch

    def random_rotate(self, pointcloud_batch):
        """Rotate the entire batch of point clouds randomly around the X-axis."""
        B, N, _ = pointcloud_batch.shape

        # Generate random rotation angles for each point cloud
        angles = torch.FloatTensor(B).uniform_(*self.rotate_range).to(pointcloud_batch.device)
        angles_rad = torch.deg2rad(angles)

        cos_vals = torch.cos(angles_rad)
        sin_vals = torch.sin(angles_rad)

        # Initialize identity matrices (B, 3, 3)
        rotation_matrices = torch.eye(3, device=pointcloud_batch.device).unsqueeze(0).repeat(B, 1, 1)

        # X-axis rotation (fixed axis for all point clouds)
        rotation_matrices[:, 1, 1] = cos_vals
        rotation_matrices[:, 1, 2] = -sin_vals
        rotation_matrices[:, 2, 1] = sin_vals
        rotation_matrices[:, 2, 2] = cos_vals

        # Apply the rotation matrices to the batch of point clouds using batch matrix multiplication
        rotated_pointcloud_batch = torch.bmm(pointcloud_batch, rotation_matrices)

        return rotated_pointcloud_batch

    def random_scale_anisotropic(self, pointcloud_batch):
        """Apply anisotropic scaling to the entire batch of point clouds."""
        B, N, _ = pointcloud_batch.shape
        scale_factors = torch.FloatTensor(B, 3).uniform_(*self.scale_range).to(pointcloud_batch.device)
        return pointcloud_batch * scale_factors[:, None, :]  # Apply scale per batch

    def add_outliers(self, pointcloud_batch, outlier_ratio=0.1, contour_prob=0.75, scale_factor=1.5):
        """
        Replace a portion of the points in the point cloud with outliers.

        :param pointcloud_batch: Tensor of shape [B, N, 3] (batch of point clouds)
        :param outlier_ratio: Ratio of points to be replaced with outliers (e.g., 0.1 for 10%).
        :param contour_prob: Probability of placing outliers around the bounding box contours.
        :param scale_factor: Factor to control how far outliers can be from the existing points.
        :return: Point cloud batch with outliers replacing some original points.
        """
        B, N, _ = pointcloud_batch.shape
        num_outliers = int(N * outlier_ratio)  # Number of points to replace with outliers

        # Calculate the bounding box of the point cloud for each batch
        min_vals, _ = torch.min(pointcloud_batch, dim=1, keepdim=True)
        max_vals, _ = torch.max(pointcloud_batch, dim=1, keepdim=True)

        # Create a mask to determine where to place the outliers
        contour_mask = torch.rand(B, num_outliers, device=pointcloud_batch.device) < contour_prob

        # Generate random outliers: Either near the contour or completely random
        random_outliers = torch.randn(B, num_outliers, 3, device=pointcloud_batch.device) * scale_factor

        # Outliers near the bounding box contour: biased towards min/max bounds
        contour_outliers = torch.empty(B, num_outliers, 3, device=pointcloud_batch.device)

        # Top and bottom along Z-axis
        contour_outliers[:, :, 2] = torch.where(torch.rand(B, num_outliers, device=pointcloud_batch.device) < 0.5,
                                                max_vals[:, :, 2] + torch.abs(
                                                    torch.randn(B, num_outliers, device=pointcloud_batch.device)),
                                                min_vals[:, :, 2] - torch.abs(
                                                    torch.randn(B, num_outliers, device=pointcloud_batch.device)))

        # Left and right along X-axis
        contour_outliers[:, :, 0] = torch.where(torch.rand(B, num_outliers, device=pointcloud_batch.device) < 0.5,
                                                max_vals[:, :, 0] + torch.abs(
                                                    torch.randn(B, num_outliers, device=pointcloud_batch.device)),
                                                min_vals[:, :, 0] - torch.abs(
                                                    torch.randn(B, num_outliers, device=pointcloud_batch.device)))

        # Front and back along Y-axis
        contour_outliers[:, :, 1] = torch.where(torch.rand(B, num_outliers, device=pointcloud_batch.device) < 0.5,
                                                max_vals[:, :, 1] + torch.abs(
                                                    torch.randn(B, num_outliers, device=pointcloud_batch.device)),
                                                min_vals[:, :, 1] - torch.abs(
                                                    torch.randn(B, num_outliers, device=pointcloud_batch.device)))

        # Use the contour points where the mask is true, and random points elsewhere
        outliers = torch.where(contour_mask.unsqueeze(-1), contour_outliers, random_outliers)

        # Randomly select indices in the point cloud to be replaced with outliers
        replace_indices = torch.randint(0, N, (B, num_outliers), device=pointcloud_batch.device)

        # Replace the selected points with outliers
        pointcloud_batch.scatter_(1, replace_indices.unsqueeze(-1).expand(-1, -1, 3), outliers)

        return pointcloud_batch

    def augment(self, pointcloud_batch):
        """
        Apply augmentations to the entire batch of point clouds without loops.
        :param pointcloud_batch: Tensor of shape [B, N, 3]
        :param feature_batch: Optional feature tensor of shape [B, N, C]
        :return: Augmented pointcloud batch, and possibly augmented feature batch
        """
        # Apply augmentations to the entire batch
        pointcloud_batch = self.random_noise(pointcloud_batch)
        pointcloud_batch = self.random_rotate(pointcloud_batch)
        # pointcloud_batch = self.add_outliers(pointcloud_batch)

        return pointcloud_batch

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self,
                 from_rep='quaternion',
                 to_rep='rotation_6d',
                 from_convention=None,
                 to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention)
                         for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: Union[np.ndarray, torch.Tensor]
                ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def axis_angle2quat(axis_angle_error):
    angle = torch.norm(-axis_angle_error, p=2, dim=-1)
    axis = -axis_angle_error / angle.unsqueeze(-1)
    quat = quat_from_angle_axis(angle, axis)
    return quat


def matrix_to_euler_xyz(rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices to Euler angles in the XYZ convention.
    Args:
        rotation_matrix: tensor of shape (..., 3, 3).
    Returns:
        Euler angles as tensor of shape (..., 3).
    """
    # Ensure the input matrix is of shape (..., 3, 3)
    assert rotation_matrix.shape[-2:] == (3, 3), "Invalid shape of rotation matrix"

    # Extract matrix components for readability
    r11 = rotation_matrix[..., 0, 0]
    r21 = rotation_matrix[..., 1, 0]
    r31 = rotation_matrix[..., 2, 0]
    r32 = rotation_matrix[..., 2, 1]
    r33 = rotation_matrix[..., 2, 2]

    # Compute Euler angles
    theta_y = torch.asin(-r31)
    theta_z = torch.atan2(r32, r33)
    theta_x = torch.atan2(r21, r11)

    return torch.stack((theta_x, theta_y, theta_z), -1)


def quat2euler(quat):
    euler = get_euler_xyz(quat)
    euler_x, euler_y, euler_z = euler
    euler = torch.stack([euler_x, euler_y, euler_z], dim=-1)
    return euler


def quat2euler2(quat):
    M = quaternion_to_matrix(quat)
    euler = matrix_to_euler_xyz(M)
    return euler


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/modules/transforms.html#pytorch3d.transforms.euler_angles_to_matrix
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[0] @ matrices[1] @ matrices[2]


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def xyzquat_to_tf_numpy(position_quat: np.ndarray) -> np.ndarray:
    """
    convert [x, y, z, qx, qy, qz, qw] to 4 x 4 transformation matrices
    """
    # try:
    position_quat = np.atleast_2d(position_quat)  # (N, 7)
    N = position_quat.shape[0]
    T = np.zeros((N, 4, 4))
    T[:, 0:3, 0:3] = R.from_quat(position_quat[:, 3:]).as_matrix()
    T[:, :3, 3] = position_quat[:, :3]
    T[:, 3, 3] = 1
    # except ValueError:
    #     print("Zero quat error!")
    # return T.squeeze()
    return T


# pose vector: [position(3), quaternion(4)] = [x, y, z, q1, q2, q3, q0]
# pose matrix (SE3): [[R, p], [0^T, 1]]
# (7, ) or (N, 7) -> (4, 4) or (N, 4, 4)
def pose_vec_to_mat(input_vec: torch.Tensor):
    if isinstance(input_vec, np.ndarray):
        vec = torch.Tensor(input_vec)
    else:
        vec = input_vec

    assert vec.ndim in {1, 2} and vec.size(-1) == 7, f"invalid pose vector shape: {vec.size()}"
    ndim = vec.ndim
    if ndim == 1: vec = vec.view((1, -1))  # (1, 7)
    N = vec.size(0)

    p = vec[..., :3].view((-1, 3, 1))
    quat = vec[..., 3:]
    R = quat2R(quat)

    SE3 = torch.cat((
        torch.cat((R, p), dim=-1),
        torch.cat((torch.zeros((N, 1, 3), device=vec.device), torch.ones((N, 1, 1), device=vec.device)), dim=-1)
    ), dim=-2)

    if ndim == 1: SE3 = SE3.squeeze()

    # if isinstance(input_vec, np.ndarray):
    #     SE3 = SE3.cpu().numpy()

    return SE3


# (unit) quaternion to rotation matrix
# [vec, w] = [q1, q2, q3, q0] -> R
# (4, ) or (N, 4)-> (3, 3) or (N, 3, 3)
def quat2R(quat: torch.Tensor):
    assert quat.ndim in {1, 2} and quat.size(-1) == 4, f"invalid quaternion shape: {quat.size()}"
    quat = unify(quat)
    ndim = quat.ndim
    if ndim == 1: quat = quat.view((1, -1))  # (1, 4)

    q0, q1, q2, q3 = quat[..., -1], quat[..., 0], quat[..., 1], quat[..., 2]  # (N, )
    # print([q.size() for q in [q0, q1, q2, q3]])
    R = torch.stack([
        torch.stack([1 - 2 * q2 ** 2 - 2 * q3 ** 2, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2], dim=-1),
        torch.stack([2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 ** 2 - 2 * q3 ** 2, 2 * q2 * q3 - 2 * q0 * q1], dim=-1),
        torch.stack([2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 ** 2 - 2 * q2 ** 2], dim=-1)
    ], dim=1)  # (N, 3, 3)

    if ndim == 1: R = R.squeeze()

    return R


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def unify(x, eps: float = 1e-9):  # (dim, ) or (N, dim)
    return x / torch.clamp(torch.linalg.norm(x, axis=-1, keepdims=True), min=eps, max=None)
