# Code adopted from MidasTouch Paper, todo @ add licence

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
TACTO rendering class
"""
import numpy as np
from isaacgyminsertion.allsight.experiments.utils.object_loader import object_loader
from isaacgyminsertion.allsight.tacto.renderer import euler2matrix
import cv2
from isaacgyminsertion.allsight.experiments.utils.pose import pose_from_vertex_normal, \
    generate_SE3_pose_single_vertex_normal
from scipy.spatial.transform import Rotation as R
import os
import trimesh
import torch
import hydra
from omegaconf import DictConfig
import random
from isaacgyminsertion.allsight.tacto_allsight_wrapper import allsight_wrapper

DEBUG = False
from dataclasses import dataclass


def circle_mask(size=(224, 224), border=0):
    """
        used to filter center circular area of a given image,
        corresponding to the AllSight surface area
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2, size[1] // 2)
    m_radius = min(size[0], size[1]) // 2 - border
    m = cv2.circle(m, m_center, m_radius, 255, -1)

    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)
    return mask


def matrix2trans(matrix):
    r = R.from_matrix(matrix[:3, :3])
    euler = r.as_euler(seq="xyz")
    translation = matrix[:3, 3]
    return translation, euler


class allsight_renderer:
    def __init__(
            self,
            cfg: DictConfig,
            obj_path: str = None,
            obj_pose: np.ndarray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            randomize: bool = False,
            bg_id=None,
            headless=False,
            finger_idx: int = 0
    ):

        self.render_config = cfg
        self.show_depth = True
        self.finger_idx = finger_idx

        if randomize:
            bg_id = random.randint(12, 19)
            obj_scale = 0.01 * random.randint(135, 150) # 0.01 * random.randint(135, 150)
        else:
            bg_id = 15
            obj_scale = 1.5

        # Create renderer
        self.zrange = 0.002

        self.subtract_bg = True
        leds = 'white'
        path_to_refs = os.path.join(os.path.dirname(__file__), "../")
        bg = cv2.imread(os.path.join(path_to_refs, f"experiments/conf/ref/ref_frame_{leds}{bg_id}.jpg"))
        conf_path = os.path.join(path_to_refs, f"experiments/conf/sensor/config_allsight_{leds}.yml")

        self.renderer = allsight_wrapper.Renderer(
            width=cfg.tacto.width, height=cfg.tacto.width,
            **{"config_path": conf_path},
            background=bg if cfg.with_bg else None,
            headless=True
        )

        self.cam_dist = cfg.cam_dist
        self.pixmm = cfg.pixmm
        self.mask = circle_mask(size=(self.render_config.tacto.width, self.render_config.tacto.height))

        if not DEBUG:
            self.bg_img, self.bg_depth = self.renderer.render()
            self.bg_depth = self.bg_depth[0]
            self.bg_img = self.bg_img[0]
            self.bg_depth_pix = self.correct_pyrender_height_map(self.bg_depth)

        if obj_path is not None:
            self.obj_loader = object_loader(obj_path)
            obj_trimesh = trimesh.load(obj_path)
            obj_trimesh.apply_scale(obj_scale)
            self.obj_mesh = obj_trimesh
            obj_euler = R.from_quat([0.0, 0.0, 0.0, 1.0]).as_euler("xyz", degrees=False)
            self.renderer.add_object(obj_trimesh, "object", orientation=obj_euler)

        self.press_depth = 0.001
        self.randomize_light = False # TODO

    def get_background(self, frame="gel"):
        """
        Return cached bg image
        """
        return self.bg_depth_pix if frame == "gel" else self.bg_depth

    def pix2meter(self, pix):
        """
        Convert pixel to meter
        """
        return pix * self.pixmm / 1000.0

    def meter2pix(self, m):
        """
        Convert meter to pixels
        """
        return m * 1000.0 / self.pixmm

    def update_pose_given_point(self, point, press_depth, shear_mag, delta=0.0):
        """

        """

        # Find the nearest vertice
        dist = np.linalg.norm(point - self.obj_loader.obj_vertices, axis=1)
        idx = np.argmin(dist)

        # idx: the idx vertice, get a new pose
        new_position = self.obj_loader.obj_vertices[idx].copy()

        new_orientation = self.obj_loader.obj_normals[idx].copy()

        delta = np.random.uniform(low=0.0, high=2 * np.pi, size=(1,))[0]

        new_pose = pose_from_vertex_normal(
            new_position, new_orientation, shear_mag, delta
        ).squeeze()

        self.update_pose_given_pose(press_depth, new_pose)

    def update_pose_given_pose(self, press_depth, gel_pose):
        """
        Given tf gel_pose and press_depth, update tacto camera
        """
        self.press_depth = press_depth

        cam_pose = self.gel2cam(gel_pose)
        cam_pose = self.add_press(cam_pose)

        self.renderer.update_camera_pose_from_matrix(self.fix_transform(cam_pose))

        # self.renderer.update_camera_pose_from_matrix(cam_pose)

    def _depth_to_color(self, depth):
        gray = (np.clip(depth / self.zrange, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def updateGUI(self, colors, depths):
        """
        Update images for visualization
        """
        # if not self.visualize_gui:
        #     return

        # concatenate colors horizontally (axis=1)
        color = np.concatenate(colors, axis=1)
        if self.subtract_bg:
            color = (255 * color).astype(np.uint8)

        if self.show_depth:
            # concatenate depths horizontally (axis=1)
            depth = np.concatenate(list(map(self._depth_to_color, depths)), axis=1)

            # concatenate the resulting two images vertically (axis=0)
            color_n_depth = np.concatenate([color, depth], axis=0)

            cv2.imshow(
                f"color and depth{self.finger_idx}", cv2.cvtColor(color_n_depth, cv2.COLOR_RGB2BGR)
            )
        else:
            cv2.imshow("color", cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

        cv2.waitKey(1)

    def fix_transform(self, pose):
        """
        Inverse of transformation in config_digit_shadow.yml
        """
        switch_axes = euler2matrix(angles=[-90, 0, 90], xyz="zyx", degrees=True)
        return np.matmul(pose, switch_axes)

    def add_press(self, pose):
        """
        Add sensor penetration
        """
        pen_mat = np.eye(4)
        pen_mat[2, 3] = -self.press_depth
        return np.matmul(pose, pen_mat)

    def gel2cam(self, gel_pose):
        """
        Convert gel_pose to cam_pose
        """
        cam_tf = np.eye(4)
        cam_tf[2, 3] = self.cam_dist
        return np.matmul(gel_pose, cam_tf)

    def cam2gel(self, cam_pose):
        """
        Convert cam_pose to gel_pose
        """
        cam_tf = np.eye(4)
        cam_tf[2, 3] = -self.cam_dist
        return np.matmul(cam_pose, cam_tf)

    def update_pose_given_sim_pose(self, cam_pose, object_pose, is_matrix=True):

        if is_matrix:
            self.renderer.update_object_pose_from_matrix("object", object_pose)
            self.renderer.update_camera_pose_from_matrix(cam_pose)
        else:
            position, orientation = object_pose
            self.renderer.update_object_pose("object", position, orientation)
            position, orientation = cam_pose
            self.renderer.update_camera_pose(position, orientation)

    #     def get_force(self, cam_name):
    #         # Load contact force
    #
    #         obj_id = self.cameras[cam_name].obj_id
    #         link_id = self.cameras[cam_name].link_id
    #
    #         pts = p.getContactPoints(
    #             bodyA=obj_id, linkIndexA=link_id, physicsClientId=self.cid
    #         )
    #
    #         # accumulate forces from 0. using defaultdict of float
    #         self.normal_forces[cam_name] = collections.defaultdict(float)
    #
    #         for pt in pts:
    #             body_id_b = pt[2]
    #             link_id_b = pt[4]
    #
    #             obj_name = "{}_{}".format(body_id_b, link_id_b)
    #
    #             # ignore contacts we don't care (those not in self.objects)
    #             if obj_name not in self.objects:
    #                 continue
    #
    #             # Accumulate normal forces
    #             self.normal_forces[cam_name][obj_name] += pt[9]
    #
    #         return self.normal_forces[cam_name]

    # input depth is in camera frame here
    def render(self, object_poses=None, normal_forces=None):
        """
        render [tactile image + depth + mask] @ current pose
        """

        if object_poses is not None:
            obj_pos_dict = {'object': matrix2trans(object_poses)}
            normal_force_dict = {'object': 20}  # Todo collect force
        else:
            obj_pos_dict, normal_force_dict = None, None

        color, depth = self.renderer.render(obj_pos_dict, normal_force_dict)

        # Remove the depth from curved gel
        for j in range(len(depth)):
            depth[j] = self.renderer.depth0[j] - depth[j]

        color, depth = color[0], depth[0]
        if self.subtract_bg:
            color = self._subtract_bg(color, self.bg_img) * self.mask
        # color[mask == 0] = 0

        diff_depth = (self.bg_depth) - depth
        contact_mask = diff_depth > np.abs(self.press_depth * 0.2)

        gel_depth = depth

        # cam_depth = self.correct_image_height_map(gel_depth) #  pix in gel frame
        # assert np.allclose(cam_depth, depth), "Conversion to pixels is incorrect"

        if self.randomize_light:
            self.renderer.randomize_light()

        return color, gel_depth, contact_mask

    def correct_pyrender_height_map(self, height_map):
        """
        Input: height_map in meters, in camera frame
        Output: height_map in pixels, in gel frame
        """
        # move to the gel center
        height_map = (self.cam_dist - height_map) * (1000 / self.pixmm)
        return height_map

    def correct_image_height_map(self, height_map, output_frame="cam"):
        """
        Input: height_map in pixels, in gel frame
        Output: height_map in meters, in camera/gel frame
        """
        height_map = (
                -height_map * (self.pixmm / 1000)
                + float(output_frame == "cam") * self.cam_dist
        )
        return height_map

    def get_cam_pose_matrix(self):
        """
        return camera pose matrix of renderer
        """
        return self.renderer.camera_nodes[0].matrix

    def get_cam_pose(self):
        """
        return camera pose of renderer
        """
        # print(f"Cam pose: {tf_to_xyzquat(self.get_cam_pose_matrix())}")
        return self.get_cam_pose_matrix()

    def get_gel_pose_matrix(self):
        """
        return gel pose matrix of renderer
        """
        return self.cam2gel(self.get_cam_pose_matrix())

    def get_gel_pose(self):
        """
        return gel pose of renderer
        """
        # print(f"Gel pose: {tf_to_xyzquat(self.get_gel_pose_matrix())}")
        return self.get_gel_pose_matrix()

    def heightmap2Pointcloud(self, depth, contact_mask=None):
        """
        Convert heightmap + contact mask to point cloud
        [Input]  depth: (width, height) in pixels, in gel frame, Contact mask: binary (width, height)
        [Output] pointcloud: [(width, height) - (masked off points), 3] in meters in camera frame
        """
        depth = self.correct_image_height_map(depth, output_frame="cam")

        if contact_mask is not None:
            heightmapValid = depth * contact_mask  # apply contact mask
        else:
            heightmapValid = depth

        f, w, h = self.renderer.f, self.renderer.width / 2.0, self.renderer.height / 2.0

        if not torch.is_tensor(heightmapValid):
            heightmapValid = torch.from_numpy(heightmapValid)
        # (0, 640) and (0, 480)
        xvals = torch.arange(heightmapValid.shape[1], device=heightmapValid.device)
        yvals = torch.arange(heightmapValid.shape[0], device=heightmapValid.device)
        [x, y] = torch.meshgrid(xvals, yvals)
        x, y = torch.transpose(x, 0, 1), torch.transpose(
            y, 0, 1
        )  # future warning: https://github.com/pytorch/pytorch/issues/50276

        # x and y in meters
        x = ((x - w)) / f
        y = ((y - h)) / f

        x *= depth
        y *= -depth

        heightmap_3d = torch.hstack(
            (x.reshape((-1, 1)), y.reshape((-1, 1)), heightmapValid.reshape((-1, 1)))
        )

        heightmap_3d[:, 2] *= -1
        heightmap_3d = heightmap_3d[heightmap_3d[:, 2] != 0]
        return heightmap_3d

    def render_sensor_trajectory(self, p, mNoise=None, pen_ratio=1.0, over_pen=False):
        """
        Render a trajectory of poses p via allsight_render
        """
        p = np.atleast_3d(p)

        N = p.shape[0]
        images, heightmaps, contactMasks = [None] * N, [None] * N, [None] * N
        gelposes, camposes = np.zeros([N, 4, 4]), np.zeros([N, 4, 4])

        min_press, max_press = (
            self.render_config.pen.min * pen_ratio,
            self.render_config.pen.max * pen_ratio,
        )
        # print(f"min_press: {min_press}, max_press: {max_press}")
        # press_depth = np.random.uniform(low=min_press, high=max_press)
        press_depth = 0.0007
        press_range = max_press - min_press

        idx = 0
        for p0 in p:

            # delta = np.random.uniform(-press_range / 50.0, press_range / 50.0)
            # delta = 0.0
            # if press_depth + delta > max_press or press_depth + delta < min_press:
            #     press_depth -= delta
            # else:
            #     press_depth += delta

            self.update_pose_given_pose(press_depth, p0)

            tactile_img, height_map, contact_mask = self.render()

            if over_pen:
                # Check for over-pen and compensate
                diff_pen = height_map - self.get_background()  # pixels in gel frame
                diff_pen_max = self.pix2meter(np.abs(diff_pen.max())) - max_press
                if diff_pen_max > 0:
                    new_depth = press_depth - diff_pen_max
                    self.update_pose_given_pose(new_depth, p0)
                    tactile_img, height_map, contact_mask = self.render()

            heightmaps[idx], contactMasks[idx], images[idx] = (
                height_map,
                contact_mask,
                tactile_img,
            )
            gelposes[idx, :] = self.get_gel_pose()
            camposes[idx, :] = self.get_cam_pose()
            idx += 1

        # measurement with noise
        rotNoise = np.random.normal(loc=0.0, scale=mNoise["sig_r"], size=(N, 3))
        Rn = R.from_euler("zyx", rotNoise, degrees=True).as_matrix()  # (N, 3, 3)
        tn = np.random.normal(loc=0.0, scale=mNoise["sig_t"], size=(N, 3))
        Tn = np.zeros((N, 4, 4))
        Tn[:, :3, :3], Tn[:, :3, 3], Tn[:, 3, 3] = Rn, tn, 1

        gelposes_meas = gelposes @ Tn

        return heightmaps, contactMasks, images, camposes, gelposes, gelposes_meas

    def render_sensor_poses(self, p, num_depths=1, no_contact_prob=0):
        """
        Render an unordered set of poses p via allsight_render
        """
        p = np.atleast_3d(p)

        N = p.shape[0] * num_depths
        images, heightmaps, contactMasks = [None] * N, [None] * N, [None] * N
        gelposes, camposes = np.zeros([N, 4, 4]), np.zeros([N, 4, 4])

        idx = 0
        for p0 in p:

            # loop over # press depths
            for _ in range(num_depths):
                # randomly sample no contact for no_contact_prob% of trials
                p_no_contact = random.randrange(100) < no_contact_prob
                if p_no_contact:
                    press_depth = -self.render_config.pen.max
                else:
                    press_depth = np.random.uniform(
                        low=self.render_config.pen.min, high=self.render_config.pen.max
                    )

                self.update_pose_given_pose(press_depth, p0)

                tactile_img, height_map, contact_mask = self.render()
                # Check for over-pen and compensate
                diff_pen = height_map - self.get_background()  # pixels in gel frame
                diff_pen_max = (
                        self.pix2meter(np.abs(diff_pen.max())) - self.render_config.pen.max
                )
                if diff_pen_max > 0:
                    press_depth -= diff_pen_max
                    self.update_pose_given_pose(press_depth, p0)
                    tactile_img, height_map, contact_mask = self.render()

                heightmaps[idx], contactMasks[idx], images[idx] = (
                    height_map,
                    contact_mask,
                    tactile_img,
                )
                gelposes[idx, :] = self.get_gel_pose()
                camposes[idx, :] = self.get_cam_pose()
                idx += 1

        return heightmaps, contactMasks, images, camposes, gelposes

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff


import matplotlib.pyplot as plt


# plt.rc("pdf", fonttype=42)
# plt.rc("ps", fonttype=42)
# plt.rc("font", family="serif")
# plt.rc("xtick", labelsize="small")
# plt.rc("ytick", labelsize="small")
#
# def view_subplots(image_data, image_mosaic):
#     """
#     Make subplot mosaic from image data
#     """
#     fig, axes = plt.subplot_mosaic(image_mosaic, constrained_layout=True)
#     for j, (label, ax) in enumerate(axes.items()):
#         ax.imshow(image_data[j])
#         ax.axis("off")
#         ax.set_title(label)
#     plt.show()

@hydra.main(config_path="conf", config_name="test")
def main(cfg: DictConfig) -> None:
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    obj_model = 'cube'
    assets_path = '/home/osher/Desktop/isaacgym/python/IsaacGymUtils/assets/tacto_related/objects/'
    obj_path = assets_path + f"{obj_model}.obj"

    obj_path = '/home/osher/Desktop/isaacgym/python/IsaacGymUtils/assets/urdf/others/hanoi/hanoi_token_nontextured_scaled.stl'

    press_depth = 0.000  # in meter

    allsight_render = allsight_renderer(cfg, obj_path, randomize=False)

    from PIL import Image

    images = []
    vertix_idxs = np.random.choice(1000, size=100)  # [159]

    for vertix_idx in vertix_idxs:
        allsight_render.update_pose_given_point(vertix_idx,
                                                press_depth,
                                                shear_mag=0.0)

        tactile_img, height_map, contact_mask = allsight_render.render()

        # view_subplots(
        #     [
        #         cfg.cam_dist - height_map,
        #         tactile_img / 255.0,
        #         contact_mask,
        #     ],
        #     [
        #         [
        #             "v : {} Heightmap AllSight".format(vertix_idx),
        #             "Tactile image AllSight",
        #             "Contact Mask AllSight",
        #         ]
        #     ],
        # )

        images.append(Image.fromarray(tactile_img))
        # images[0].save(
        #     "augmentations.gif", save_all=True, append_images=images, duration=200, loop=0
        # )
        #     plt.imshow(tactile_img)
        allsight_render.updateGUI([tactile_img], [height_map])
        # plt.pause(0.1)


if __name__ == "__main__":
    main()
