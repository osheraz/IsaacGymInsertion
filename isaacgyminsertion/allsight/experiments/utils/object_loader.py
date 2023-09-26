'''
Object loader to pyrender

Zilin Si (zsi@andrew.cmu.edu)
Last revision: Sept 2021
'''

import os
from os import path as osp
import numpy as np
import trimesh
import pyrender

class object_loader:
    def __init__(self, obj_path):

        obj_trimesh = trimesh.load(obj_path)
        # get object's mesh, vertices, normals

        # scale up for clearer indentation
        # matrix = np.eye(4)
        # matrix[[0, 1, 2], [0, 1, 2]] = 0.001
        # obj_trimesh = obj_trimesh.apply_transform(matrix)

        self.obj_mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
        self.obj_vertices = obj_trimesh.vertices
        self.obj_normals = obj_trimesh.vertex_normals
        # initial the obj pose
        self.obj_pose = np.array([
            [1.0, 0.0,  0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 1.0],
        ])

if __name__ == "__main__":
    obj_name = "005_tomato_soup_can"
    obj_path = osp.join("..", "data", obj_name, "google_512k", "nontextured.stl")
    obj_loader = object_loader(obj_path)