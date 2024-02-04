import cv2
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs
#
#     def draw(self, renderer):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         FancyArrowPatch.draw(self, renderer)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        return np.min(zs)

def update_line(hl, new_data):
    hl.set_xdata(np.asarray(new_data[0]))
    hl.set_ydata(np.asarray(new_data[1]))
    hl.set_3d_properties(np.asarray(new_data[2]))


def update_arrow(hl, new_data):
    hl.set_xdata(np.asarray([new_data[0][0], new_data[0][1]]))
    hl.set_ydata(np.asarray([new_data[1][0], new_data[1][1]]))
    hl.set_3d_properties(np.asarray([new_data[2][0], new_data[2][1]]))


def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 2)
    theta = np.linspace(0, 2 * np.pi, 20)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid


def data_for_sphere_along_z(center_x, center_y, radius, height_z):
    q = np.linspace(0, 2 * np.pi, 15)
    p = np.linspace(0, np.pi / 2, 15)
    p_, q_ = np.meshgrid(q, p)
    x_grid = radius * np.cos(p_) * np.sin(q_)
    y_grid = radius * np.sin(p_) * np.sin(q_)
    z_grid = radius * np.cos(q_) + height_z
    return x_grid, y_grid, z_grid


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

class MousePts:
    def __init__(self, windowname, img, rad):

        self.windowname = windowname
        self.img1 = img.copy()
        self.img = self.img1.copy()
        cv2.namedWindow(windowname, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(windowname, img)
        self.curr_pt = []
        self.point = []
        self.r = max(min(rad, 50), 5)

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.point.append([x, y])
            self.img = cv2.circle(self.img, (x, y), self.r, (0, 255, 0), 2)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.curr_pt = [x, y]
            # print(self.point)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.r -= 5
            # cv2.circle(self.img, (x, y), 50, (0, 255, 0), -1)

    def getpt(self, count=1, img=None):
        if img is not None:
            self.img = img
        else:
            self.img = self.img1.copy()

        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowname, self.img)
        cv2.setMouseCallback(self.windowname, self.select_point)
        self.point = []

        while True:
            cv2.imshow(self.windowname, self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27 or len(self.point) >= count:
                break
            # print(self.point)

        cv2.setMouseCallback(self.windowname, lambda *args: None)
        # cv2.destroyAllWindows()
        self.point.append(self.r)
        return self.point, self.img