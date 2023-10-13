#!/usr/bin/env python

import rospy
from finger_ros import TactileFingerROSPublisher

if __name__ == "__main__":

    rospy.init_node('tactile_finger_publisher')
    dev_name = rospy.get_param('~dev_name', 0)

    tactile = TactileFingerROSPublisher(dev_name=dev_name, serial='/dev/video')

    tactile.connect()

    tactile.run()