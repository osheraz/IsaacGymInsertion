import rospy
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
import numpy as np
import matplotlib.pyplot as plt
from algo.deploy.env.env_utils.deploy_utils import image_msg_to_numpy

np.set_printoptions(suppress=True, formatter={'float_kind': '{: .3f}'.format})
import torch
import cv2
from sklearn.neighbors import NearestNeighbors
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np


class PointCloudPublisher:
    def __init__(self, topic='pointcloud'):
        self.pcl_pub = rospy.Publisher(f'/{topic}', PointCloud2, queue_size=10)

    def publish_pointcloud(self, points):
        """
        Publish the point cloud to a ROS topic.

        :param points: numpy array of shape [N, 3] representing the point cloud
        """
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'base_link'  # Set the frame according to your setup

        # Define the PointCloud2 fields (x, y, z)
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        # Convert the numpy array to PointCloud2 format
        cloud_msg = pc2.create_cloud(header, fields, points)

        # Publish the point cloud message
        self.pcl_pub.publish(cloud_msg)


def remove_statistical_outliers(points, k=20, z_thresh=2.0):
    # Find the k-nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)  # k+1 because the point itself is included
    distances, _ = nbrs.kneighbors(points)

    # Remove the point itself from distance calculation (the first column)
    mean_distances = np.mean(distances[:, 1:], axis=1)

    # Calculate mean and standard deviation of distances
    mean = np.mean(mean_distances)
    std = np.std(mean_distances)

    # Keep points that are within the z_thresh standard deviations
    inliers = np.where(np.abs(mean_distances - mean) < z_thresh * std)[0]

    return points[inliers]


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
    def __init__(self, view_matrix=None,
                 camera_info_topic='/zedm/zed_node/depth/camera_info',
                 height=None, width=None, sample_num=None, proj_matrix=None,
                 depth_max=None, input_type='pcl', device='cpu'):

        # Initialize the view_matrix in torch tensor
        # from world space (robot) to camera space
        view_matrix = torch.tensor([[-0.0163, -0.4122, 0.9109, 0.],
                                    [0.9999, -0.0067, 0.0149, 0.],
                                    [0., 0.9111, 0.4123, 0.],
                                    [0.0065, 0.1445, -0.7189, 1.]], dtype=torch.float32).to(device)

        view_matrix = torch.tensor([[-0., -0.3961, 0.9182, 0.],
                                    [1, -0., 0.0, 0.],
                                    [0., 0.9182, 0.3961, 0.],
                                    [0.0, 0.125, -0.7245, 1.]], dtype=torch.float32).to(device)

        # array([[ 0.    , -0.3961,  0.9182,  0.    ],
        #        [ 1.    ,  0.    , -0.    ,  0.    ],
        #        [-0.    ,  0.9182,  0.3961,  0.    ],
        #        [-0.    ,  0.1383, -0.7245,  1.    ]], dtype=float32)

        if camera_info_topic is not None:
            camera_info = rospy.wait_for_message(camera_info_topic, CameraInfo)

            # Extract camera properties from CameraInfo message
            self.cam_width = camera_info.width
            self.cam_height = camera_info.height
            self.fu = camera_info.K[0]  # fx
            self.fv = camera_info.K[4]  # fy
            self.cu = camera_info.K[2]  # cx
            self.cv = camera_info.K[5]  # cy

            # Extract projection matrix (3x4)
            proj_matrix_ros = camera_info.P
            self.proj_matrix = torch.tensor(proj_matrix_ros).reshape(3, 4).to(device)

        else:
            # Fallback if no ROS info is available
            self.cam_width = width
            self.cam_height = height
            fu = 2 / proj_matrix[0, 0]
            fv = 2 / proj_matrix[1, 1]
            self.fu = self.cam_width / fu
            self.fv = self.cam_height / fv
            self.cu = self.cam_width / 2.
            self.cv = self.cam_height / 2.
            self.proj_matrix = torch.tensor(proj_matrix, dtype=torch.float32).to(device)

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

        self.input_type = input_type

    def convert(self, points):
        # Process depth buffer
        points = torch.tensor(points, device=self.device, dtype=torch.float32)

        if self.input_type == 'depth':
            if self.depth_max is not None:
                valid_ids = points > -self.depth_max
            else:
                valid_ids = torch.ones(points.shape, dtype=bool, device=self.device)

            valid_depth = points[valid_ids]
            uv_one_in_cam = self._uv_one_in_cam[valid_ids]

            # Calculate 3D points in camera coordinates
            pts_in_cam = torch.mul(uv_one_in_cam, valid_depth.unsqueeze(-1))
        else:
            # already pcl
            pts_in_cam = points

        # R_flip_y_up = torch.tensor([
        #     [0, -1, 0],
        #     [1, 0, 0],
        #     [0, 0, 1]
        # ], dtype=torch.float32, device=points.device)
        #
        # # Apply the rotation to convert to Y-up coordinate system
        # pts_in_cam = torch.matmul(pts_in_cam[:, :3], R_flip_y_up.T)

        # plot_point_cloud(pts_in_cam.cpu().detach().numpy())

        pts_in_cam = torch.cat((pts_in_cam,
                                torch.ones(*pts_in_cam.shape[:-1], 1,
                                           device=pts_in_cam.device)),
                               dim=-1)

        pts_in_world = torch.matmul(pts_in_cam, self.ext_mat)

        pcd_pts = pts_in_world[:, :3]
        # plot_point_cloud(pcd_pts.cpu().detach().numpy())

        return pcd_pts.cpu().detach().numpy()


class ZedPointCloudSubscriber:

    def __init__(self, topic='/zedm/zed_node/point_cloud/cloud_registered', display=False):

        self.far_clip = 0.4
        self.near_clip = 0.1
        self.dis_noise = 0.00
        self.w = 320  # 320
        self.h = 180  # 180
        self.display = display

        self.with_seg = True
        self.with_socket = True
        self.relative = False
        self.pointcloud_init = False
        self.init_success = False
        self.got_socket = False
        self.use_depth = True
        self.use_pcl = not self.use_depth

        input_type = 'depth' if self.use_depth else 'pcl'
        self.last_cloud = None
        self.pcl_gen = PointCloudGenerator(input_type=input_type)

        self.first = True

        if self.with_seg:
            from algo.deploy.env.seg_camera import SegCameraSubscriber
            self.pointcloud_socket_pub = PointCloudPublisher(topic='socket')

            self.seg = SegCameraSubscriber(with_socket=self.with_socket)
            self._check_seg_ready()

        self.pointcloud_pub = PointCloudPublisher(topic='pointcloud')

        if self.use_depth:
            self._image_subscriber = rospy.Subscriber('/zedm/zed_node/depth/depth_registered', Image,
                                                      self.image_callback, queue_size=2)
            self._check_camera_ready()

        if self.use_pcl:
            self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
            rospy.loginfo("(topic_name) Subscribing to PointCloud2 topic %s", self._topic_name)
            self._pointcloud_subscriber = rospy.Subscriber(self._topic_name, PointCloud2, self.pointcloud_callback,
                                                           queue_size=2)
            self._check_pointcloud_ready()

    def update_plot(self):

        if self.last_cloud is not None:

            if self.first:
                self.fig = plt.figure()
                self.ax = self.fig.add_subplot(111, projection='3d')
                self.first = False

            self.ax.scatter(self.last_cloud[:, 0],
                            self.last_cloud[:, 1],
                            self.last_cloud[:, 2], 'ro')

            self.ax.set_xlim([0.35, 0.6])
            self.ax.set_ylim([-0.1, 0.1])
            self.ax.set_zlim([0., 0.1])
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            plt.pause(0.0001)
            plt.cla()

    def get_com(self, pcl):

        # Example point cloud

        # Fit k-nearest neighbors to estimate local density
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(pcl)
        distances, indices = nbrs.kneighbors(pcl)

        # Local density: inverse of the mean distance to the nearest neighbors
        density = np.mean(distances, axis=1)
        weights = 1.0 / density

        # Compute the weighted centroid (center of mass)
        weighted_center = np.average(pcl, axis=0, weights=weights)

        return weighted_center

    def image_callback(self, msg):

        try:
            frame = image_msg_to_numpy(msg)
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)

        except Exception as e:
            print(e)
        else:
            if self.with_seg:
                seg = self.seg.get_frame()
                # seg = cv2.resize(seg, (640, 360), interpolation=cv2.INTER_NEAREST)
                if seg is not None:

                    if self.with_socket and not self.got_socket:
                        socket_mask = (seg == self.seg.socket_id).astype(float)
                        socket_mask = self.seg.shrink_mask(socket_mask)
                        proc_socket = self.pcl_gen.convert(frame * socket_mask)
                        proc_socket = self.process_pointcloud(proc_socket)
                        # proc_socket = remove_statistical_outliers(proc_socket)
                        proc_socket = self.sample_n(proc_socket, num_sample=400)
                        socket_mean = proc_socket.mean(axis=0)
                        # print('mean',  socket_mean)
                        # print('com', self.get_com(proc_socket))

                        self.proc_socket = proc_socket
                        self.got_socket = False

                    if self.with_socket:
                        self.pointcloud_socket_pub.publish_pointcloud(self.proc_socket)

                    plug_mask = (seg == self.seg.plug_id).astype(float)
                    plug_mask = self.seg.shrink_mask(plug_mask)
                    frame *= plug_mask
                else:
                    print('Cant find the object')
                    return

            try:

                cloud_points = self.pcl_gen.convert(frame)
                proc_cloud = self.process_pointcloud(cloud_points)
                # proc_cloud = remove_statistical_outliers(proc_cloud)
                proc_cloud = self.sample_n(proc_cloud, num_sample=400)
                self.pointcloud_pub.publish_pointcloud(proc_cloud)

                # self.last_cloud = proc_cloud

                if self.with_socket:

                    self.last_cloud = np.concatenate((proc_cloud, self.proc_socket), axis=0)

                    if self.relative:
                        self.last_cloud -= socket_mean

            except Exception as e:
                print(e)

    def _check_camera_ready(self):

        self.last_frame = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format('/zedm/zed_node/depth/depth_registered'))
        while self.last_frame is None and not rospy.is_shutdown():
            try:
                self.last_frame = rospy.wait_for_message(
                    '{}'.format('/zedm/zed_node/depth/depth_registered'), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format('/zedm/zed_node/depth/depth_registered'))
                self.zed_init = True
                self.last_frame = image_msg_to_numpy(self.last_frame)
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(
                        '/zedm/zed_node/depth/depth_registered'))
        return self.last_frame

    def _check_seg_ready(self):
        print('Waiting for SAM to init')
        while not self.seg.init_success and not rospy.is_shutdown():
            self.init_success &= self.seg.init_success
        print('SAM is ready')

    def _check_pointcloud_ready(self):
        self.last_cloud = None
        rospy.logdebug("Waiting for '{}' to be READY...".format(self._topic_name))
        while self.last_cloud is None and not rospy.is_shutdown():
            try:
                self.last_cloud = rospy.wait_for_message(self._topic_name, PointCloud2, timeout=5.0)
                rospy.logdebug("Current '{}' READY=>".format(self._topic_name))
                self.pointcloud_init = True
                self.last_cloud = self.pointcloud_msg_to_numpy(self.last_cloud)
                self.start_time = rospy.get_time()
            except:
                rospy.logerr("Current '{}' not ready yet, retrying for getting point cloud".format(self._topic_name))
        return self.last_cloud

    def pointcloud_callback(self, msg):
        try:
            cloud_points = self.pointcloud_msg_to_numpy(msg)
        except Exception as e:
            print(e)
        else:
            cloud_points = self.pcl_gen.convert(cloud_points)

            proc_cloud = self.process_pointcloud(cloud_points)

            try:
                if self.with_seg:
                    seg = self.seg.get_frame()
                    mask = (seg == self.seg.plug_id).astype(float)
                    cloud_points = proc_cloud * np.expand_dims(mask, axis=0)
                    self.last_cloud = cloud_points
                    self.pointcloud_pub.publish_pointcloud(self.last_cloud)
            except Exception as e:
                print(e)

    def sample_n(self, pts, num_sample):
        num = pts.shape[0]
        ids = np.random.randint(0, num, size=(num_sample,))
        pts = pts[ids]
        return pts

    def pointcloud_msg_to_numpy(self, msg):

        pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_data), dtype=np.float32)
        points = points[~np.isinf(points).any(axis=1)]
        return points

    def process_pointcloud(self, points):

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        valid1 = (z >= 0.001) & (z <= 0.5)
        valid2 = (x >= 0.4) & (x <= 0.65)
        valid3 = (y >= -0.2) & (y <= 0.2)

        valid = valid1 & valid3 & valid2
        points = points[valid]
        # points = ops.sample_farthest_points(points, torch.tensor(valid, dtype=torch.int), K=512)[0]
        points = self.voxel_grid_sampling(points)

        return points

    def voxel_grid_sampling(self, points, voxel_size=0.001):
        # voxel_size is a tuple (voxel_size_x, voxel_size_y, voxel_size_z)
        voxel_size_x = voxel_size
        voxel_size_y = voxel_size
        voxel_size_z = voxel_size

        # Floor the points into voxel bins with different sizes for x, y, and z
        voxel_grid = np.floor(points / np.array([voxel_size_x, voxel_size_y, voxel_size_z])).astype(int)

        # Use np.unique to find unique voxels and get indices
        unique_voxels, indices = np.unique(voxel_grid, axis=0, return_index=True)

        # Select the first point in each voxel
        sampled_points = points[indices]

        return sampled_points

    def get_pcl(self):
        return self.last_cloud


if __name__ == "__main__":
    rospy.init_node('ZedPointCloud')
    pointcloud_sub = ZedPointCloudSubscriber()
    # pointcloud_pub = PointCloudPublisher()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        # if pointcloud_sub.last_cloud is not None:
        #     pointcloud_pub.publish_pointcloud(pointcloud_sub.last_cloud)

        rate.sleep()