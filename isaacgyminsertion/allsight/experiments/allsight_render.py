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
from time import time
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


class AllSightRenderer:
    def __init__(
            self,
            cfg: DictConfig,
            obj_path: str = None,
            obj_pose: np.ndarray = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            randomize: bool = False,
            bg_id=None,
            headless=False,
            finger_idx: int = 0,
            scale=1.08,
    ):

        self.render_config = cfg
        self.show_depth = False
        self.DEBUG = False

        self.finger_idx = finger_idx
        obj_scale = scale

        if randomize:
            bg_id = random.randint(12, 19)
        else:
            bg_id = 15

        # Create renderer
        self.zrange = 0.002
        width = cfg.tacto.width
        height = cfg.tacto.height

        self.subtract_bg = cfg.diff
        leds = 'white'
        path_to_refs = os.path.join(os.path.dirname(__file__), "../")
        bg = cv2.imread(os.path.join(path_to_refs, f"experiments/conf/ref/ref_frame_{leds}{bg_id}.jpg"))
        conf_path = os.path.join(path_to_refs, f"experiments/conf/sensor/config_allsight_{leds}.yml")

        self.renderer = allsight_wrapper.Renderer(
            width=width, height=height,
            **{"config_path": conf_path},
            background=bg if cfg.with_bg else None,
            headless=True,
            DEBUG=self.DEBUG
        )

        self.mask = circle_mask(size=(width, height))

        if not self.DEBUG:
            self.bg_img, self.bg_depth = self.renderer.render()
            self.bg_depth = self.bg_depth[0]
            self.bg_img = self.bg_img[0]

        if obj_path is not None:
            self.obj_loader = object_loader(obj_path)
            obj_trimesh = trimesh.load(obj_path)
            # obj_trimesh.apply_scale(obj_scale)
            obj_trimesh.vertices[:, 0] *= obj_scale  # Scale x
            obj_trimesh.vertices[:, 1] *= obj_scale  # Scale y
            self.obj_mesh = obj_trimesh
            obj_euler = R.from_quat([0.0, 0.0, 0.0, 1.0]).as_euler("xyz", degrees=False)
            self.renderer.add_object(obj_trimesh, "object", orientation=obj_euler)

        self.press_depth = 0.001
        self.randomize_light = False

    def get_background(self, frame="gel"):
        """
        Return cached bg image
        """
        return self.bg_depth

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

    def update_pose_given_sim_pose(self, cam_pose, object_pose, is_matrix=True):

        if is_matrix:
            self.renderer.update_object_pose_from_matrix("object", object_pose)
            self.renderer.update_camera_pose_from_matrix(cam_pose)
        else:
            position, orientation = object_pose
            self.renderer.update_object_pose("object", position, orientation)
            position, orientation = cam_pose
            self.renderer.update_camera_pose(position, orientation)

    def render(self, object_poses=None, normal_forces=None):
        """
        render [tactile image + depth + mask] @ current pose
        """
        normal_forces = 20 if normal_forces is None else normal_forces

        if object_poses is not None:
            obj_pos_dict = {'object': matrix2trans(object_poses)}
            normal_force_dict = {'object': normal_forces}
        else:
            obj_pos_dict, normal_force_dict = None, None

        color, depth = self.renderer.render(obj_pos_dict, normal_force_dict)

        # Remove the depth from curved gel
        for j in range(len(depth)):
            depth[j] = self.renderer.depth0[j] - depth[j]

        color, depth = color[0], depth[0]

        # if self.subtract_bg:
        #     color = self.remove_bg(color, self.bg_img) * self.mask

        # color[mask == 0] = 0

        # diff_depth = (self.bg_depth) - depth
        # contact_mask = diff_depth > np.abs(self.press_depth * 0.2)

        gel_depth = depth

        if self.randomize_light:
            self.renderer.randomize_light()

        return color, gel_depth #, contact_mask

    def remove_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff


@hydra.main(config_path="conf", config_name="test")
def main(cfg: DictConfig) -> None:
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    obj_model = 'cube'
    assets_path = '/home/osher/Desktop/isaacgym/python/IsaacGymUtils/assets/tacto_related/objects/'
    obj_path = assets_path + f"{obj_model}.obj"

    obj_path = '/home/osher/Desktop/isaacgym/python/IsaacGymUtils/assets/urdf/others/hanoi/hanoi_token_nontextured_scaled.stl'

    press_depth = 0.000  # in meter

    allsight_render = AllSightRenderer(cfg, obj_path, randomize=False)

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
