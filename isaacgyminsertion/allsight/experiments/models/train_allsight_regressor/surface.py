import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from scipy import spatial
from geometry import concatenate_matrices, rotation_matrix #
# from .vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z, set_axes_equal

# src.allsight.train.utils

origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
finger_props = [0, 0, 0, 0.016, 0.012]
h = finger_props[3]
r = finger_props[4]
height = h + r


def create_finger_geometry_parametrize(display=True):
    z = np.linspace(0, height, 30)
    q = np.linspace(0, 2 * np.pi, 30)

    f = np.where(z < h, r,
                 np.sqrt(r ** 2 - (z - h) ** 2)
                 )

    Z, Q = np.meshgrid(z, q)

    X = f * np.cos(Q)
    Y = f * np.sin(Q)

    if display:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(X, Y, Z, c='r', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()


def create_finger_geometry(Nc=50, Mc=10, Mr=10, display=False): #
    '''Calculate sensor geomtry

    Parameters
    ----------
    Nc : int, optional
        _description_, by default 30
    Mc : int, optional
        _description_, by default 5
    Mr : int, optional
        _description_, by default 5
    display : bool, optional
        Display geomtry 3d plot, by default False
    '''

    start_h = 0
    # Init data dictionary
    surface_xyz, surface_rot = [], []
    theta = np.linspace(0, 2 * np.pi, Nc)
    H = np.linspace(start_h, finger_props[3] + start_h, Mc)

    for j, h in enumerate(H):

        for i, q in enumerate(theta):
            H = np.asarray([
                finger_props[0] + finger_props[4] * np.cos(q),
                finger_props[1] + finger_props[4] * np.sin(q),
                finger_props[2] + h,
            ])

            Rz = rotation_matrix(q, zaxis)
            Rt = rotation_matrix(np.pi, zaxis)
            Rx = rotation_matrix(np.pi, xaxis)
            Ry = rotation_matrix(- np.pi / 2, yaxis)

            rot = concatenate_matrices(Rz, Rt, Rx, Ry)[:3, :3]

            rot = R.from_matrix(rot[:3, :3]).as_quat()

            surface_xyz.append(H)
            surface_rot.append(rot)
    # Sphere push points
    phi = np.linspace(0, np.pi / 2, Mr)

    for j, p in reversed(list(enumerate(phi))):
        for i, q in enumerate(theta):
            B = np.asarray([
                finger_props[4] * np.sin(p) * np.cos(q),
                finger_props[4] * np.sin(p) * np.sin(q),
                finger_props[3] + finger_props[4] * np.cos(p),
            ])

            Ry = rotation_matrix(p, yaxis)
            Rz = rotation_matrix(q, zaxis)
            Rt = rotation_matrix(np.pi / 2, yaxis)
            Ryy = rotation_matrix(- np.pi / 2, yaxis)

            rot = concatenate_matrices(Rz, Ry, Rt, Ryy)[:3, :3]

            rot = R.from_matrix(rot).as_quat()

            surface_xyz.append(B)
            surface_rot.append(rot)

    # print('Surface is built from {} points'.format(len(surface_xyz)))
    # display
    # if False:
    #     from matplotlib import pyplot as plt
    #     fig = plt.figure(figsize=(15, 15))
    #     ax = fig.add_subplot(111, projection='3d')
    #     for i in range(len(surface_xyz)):
    #         ax.plot(*surface_xyz[i], 'go')
    #
    #     plt.draw()
    #     plt.show()

    if display:
        import matplotlib.pyplot as plt
        from pytransform3d import rotations as pr
        from pytransform3d import transformations as pt
        from pytransform3d.transform_manager import TransformManager
        tm = TransformManager()

        # amnt = 1
        # for i in range(0, len(surface_xyz), amnt):
        #     object2cam = pt.transform_from_pq(np.hstack((surface_xyz[i],
        #                                                  pr.quaternion_wxyz_from_xyzw(
        #                                                      surface_rot[i]))))
        #
        #     tm.add_transform("object" + str(i), "camera", object2cam)
        #
        # ax = tm.plot_frames_in("camera", s=0.0015, show_name=False)
        #
        # Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 0.012, 0.016)
        # ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
        # Xc, Yc, Zc = data_for_sphere_along_z(0., 0., 0.012, 0.016)
        # ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
        # ax.set_xlim((-0.014, 0.014))
        # ax.set_ylim((-0.014, 0.014))
        # ax.set_zlim((0.0, 0.03))
        # set_axes_equal(ax)
        # plt.show()

    return surface_xyz, surface_rot


if __name__ == "__main__":

    surface = create_finger_geometry()

    point = np.random.uniform(low=height, high=height, size=(20, 3))

    tree = spatial.KDTree(surface[0])
    import time
    for p in point:
        start = time.time()
        distance, index = tree.query(p)
        end = time.time()

        print(1 / (end-start))
