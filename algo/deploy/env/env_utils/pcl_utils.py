import rospy
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


def remove_statistical_outliers(points, k=20, z_thresh=2.0):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)  # k+1 because the point itself is included
    distances, _ = nbrs.kneighbors(points)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    mean = np.mean(mean_distances)
    std = np.std(mean_distances)
    inliers = np.where(np.abs(mean_distances - mean) < z_thresh * std)[0]

    return points[inliers]


def plot_point_cloud(points):
    """ Visualize 3D point cloud using Matplotlib """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = points[::10, 0]
    y = points[::10, 1]
    z = points[::10, 2]
    ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

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
