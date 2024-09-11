import torch
from isaacgym import gymapi
from isaacgym import gymtorch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_point_cloud(points):
    """ Visualize 3D point cloud using Matplotlib """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract x, y, z coordinates from the points
    x = points[::10, 0]
    y = points[::10, 1]
    z = points[::10, 2]

    # Plot the points
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


class PointCloudGenerator:
    def __init__(self, proj_matrix, view_matrix, env_to_global, camera_props=None,
                 height=None, width=None, sample_num=None,
                 depth_max=None, device='cpu'):
        self.cam_width = camera_props.width if camera_props is not None else width
        self.cam_height = camera_props.height if camera_props is not None else height
        self.env_to_global = env_to_global

        fu = 2 / proj_matrix[0, 0]
        fv = 2 / proj_matrix[1, 1]

        self.fu = self.cam_width / fu
        self.fv = self.cam_height / fv
        self.cu = self.cam_width / 2.
        self.cv = self.cam_height / 2.

        self.int_mat = torch.Tensor(
            [[-self.fu, 0, self.cu],
             [0, self.fv, self.cv],
             [0, 0, 1]]
        )
        self.ext_mat = torch.inverse(torch.Tensor(view_matrix)).to(device)
        self.int_mat_T_inv = torch.inverse(self.int_mat.T).to(device)
        self.depth_max = depth_max

        x, y = torch.meshgrid(torch.arange(self.cam_height), torch.arange(self.cam_width))
        self._uv_one = torch.stack((y, x, torch.ones_like(x)), dim=-1).float().to(device)

        self._uv_one_in_cam = self._uv_one @ self.int_mat_T_inv

        self._uv_one_in_cam = self._uv_one_in_cam.repeat(1, 1, 1)
        self.sample_num = sample_num
        self.device = device

    @torch.no_grad()
    def convert(self, depth_buffer):
        depth_buffer = depth_buffer
        if self.depth_max is not None:
            valid_ids = depth_buffer > -self.depth_max
        else:
            valid_ids = torch.ones(depth_buffer.shape,
                                   device=depth_buffer.device,
                                   dtype=bool)
        valid_depth = depth_buffer[valid_ids]
        uv_one_in_cam = self._uv_one_in_cam[valid_ids]

        pts_in_cam = torch.mul(uv_one_in_cam, valid_depth.unsqueeze(-1))

        # plot_point_cloud(pts_in_cam.cpu().detach().numpy())

        pts_in_cam = torch.cat((pts_in_cam,
                                torch.ones(*pts_in_cam.shape[:-1], 1,
                                           device=pts_in_cam.device)),
                               dim=-1)

        pts_in_world = pts_in_cam @ self.ext_mat
        env_to_global = torch.inverse(torch.Tensor(self.env_to_global)).to(self.device)
        pts_in_world = torch.matmul(pts_in_world, env_to_global.T)

        pcd_pts = pts_in_world[..., :3]
        # plot_point_cloud(pcd_pts.cpu().detach().numpy())

        return pcd_pts

    @torch.no_grad()
    def sample_n(self, pts):
        num = pts.shape[0]
        ids = torch.randint(0, num, size=(self.sample_num,))
        pts = pts[ids]
        return pts


class CameraPointCloud:
    def __init__(self, isc_sim, isc_gym, envs, camera_handles,
                 camera_props, sample_num=4000,
                 filter_func=None, pt_in_local=False,
                 depth_max=1.0, graphics_device='cpu',
                 compute_device='cpu'):

        self.sim = isc_sim
        self.gym = isc_gym
        self.envs = envs
        self.camera_handles = camera_handles
        assert pt_in_local
        self.filter_func = filter_func
        print(f'Depth max:{depth_max}')

        self.camera_props = camera_props
        self.graphics_device = graphics_device
        self.compute_device = compute_device
        self.sample_num = sample_num
        self.num_envs = len(self.envs)
        print(f'Number of envs in camera:{self.num_envs}')

        self.pt_generators = []
        for idx in range(len(envs)):

            view_matrix = self.gym.get_camera_view_matrix(self.sim,
                                                          envs[idx],
                                                          camera_handles[idx])

            env_position = self.gym.get_env_origin(envs[idx])
            env_to_global = np.identity(4)
            env_to_global[:3, 3] = np.array([env_position.x, env_position.y, env_position.z])

            proj_matrix = self.gym.get_camera_proj_matrix(self.sim,
                                                          envs[idx],
                                                          camera_handles[idx])
            self.pt_generators.append(
                PointCloudGenerator(
                    camera_props=camera_props[idx],
                    proj_matrix=proj_matrix,
                    view_matrix=view_matrix,
                    env_to_global=env_to_global,
                    depth_max=depth_max,
                    device=self.graphics_device
                )
            )

    @torch.no_grad()
    def compute_view_matrix(self, local_transform):
        trans_vec = local_transform.p
        rot_quat = local_transform.r

        transformation_matrix = torch.eye(4)
        transformation_matrix[:3, :3] = torch.tensor([
            [1 - 2 * rot_quat.y ** 2 - 2 * rot_quat.z ** 2, 2 * rot_quat.x * rot_quat.y - 2 * rot_quat.w * rot_quat.z,
             2 * rot_quat.x * rot_quat.z + 2 * rot_quat.w * rot_quat.y],
            [2 * rot_quat.x * rot_quat.y + 2 * rot_quat.w * rot_quat.z, 1 - 2 * rot_quat.x ** 2 - 2 * rot_quat.z ** 2,
             2 * rot_quat.y * rot_quat.z - 2 * rot_quat.w * rot_quat.x],
            [2 * rot_quat.x * rot_quat.z - 2 * rot_quat.w * rot_quat.y,
             2 * rot_quat.y * rot_quat.z + 2 * rot_quat.w * rot_quat.x, 1 - 2 * rot_quat.x ** 2 - 2 * rot_quat.y ** 2]
        ], dtype=torch.float32)

        transformation_matrix[:3, 3] = torch.tensor([trans_vec.x, trans_vec.y, trans_vec.z], dtype=torch.float32)

        view_matrix = torch.inverse(transformation_matrix)

        return view_matrix

    @torch.no_grad()
    def get_point_cloud(self, depths, env_ids=None, filter_func=None, sample_num=None):
        if filter_func is None:
            filter_func = self.filter_func
        dim_per_pt = 3
        sample_num = self.sample_num if sample_num is None else sample_num
        num_envs = self.num_envs if env_ids is None else len(env_ids)
        out = torch.zeros((num_envs, sample_num, dim_per_pt), device=self.compute_device)
        all_pts = self.get_ptd_cuda(depths, env_ids=env_ids, filter_func=filter_func)

        for env_id in range(num_envs):
            if all_pts[env_id].any():
                env_pt = self.sample_n(all_pts[env_id], sample_num=sample_num)
            else:
                env_pt = out[env_id]
            out[env_id, :, :3] = env_pt.to(self.compute_device)
        return out.detach()

    @torch.no_grad()
    def _proc_pts(self, env_id, depth_images, filter_func=None):
        pts = self.pt_generators[env_id].convert(depth_images)
        if filter_func is not None:
            pts = filter_func(pts)
        elif self.filter_func is not None:
            pts = self.filter_func(pts)
        return pts

    @torch.no_grad()
    def sample_n(self, pts, sample_num=None):
        sample_num = self.sample_num if sample_num is None else sample_num
        num = pts.shape[0]
        ids = torch.randint(0, num, size=(sample_num,))
        pts = pts[ids]
        return pts

    @torch.no_grad()
    def get_ptd_cuda(self, depth_imgs, env_ids=None, filter_func=None):
        all_pts = []
        env_iter = range(len(self.envs)) if env_ids is None else env_ids
        for env_id in env_iter:
            pts = self._proc_pts(env_id=env_id,
                                 depth_images=depth_imgs[env_id],
                                 filter_func=filter_func)
            all_pts.append(pts)
        return all_pts

    @torch.no_grad()
    def clone_img_tensor(self, img_tensors, env_ids=None):
        out = []
        env_iter = range(len(self.envs)) if env_ids is None else env_ids
        out = [torch.stack(img_tensors[i]) for i in env_iter]

        return torch.stack(out)