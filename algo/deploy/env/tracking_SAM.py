import os
import rospy
import numpy as np
from PIL import Image
import cv2
import time
import argparse
from sensor_msgs.msg import Image
from algo.deploy.env.env_utils.deploy_utils import msg_to_pil, image_msg_to_numpy

from isaacgyminsertion.outputs.Tracking_SAM import tracking_SAM

weights_path = '/home/roblab20/osher3_workspace/src/isaacgym/python/IsaacGymInsertion/isaacgyminsertion/outputs/Tracking_SAM'


class SegCameraSubscriber:

    def __init__(self, topic='/zedm/zed_node/rgb/image_rect_color', display=False, device='cuda:0'):
        """
        """
        self.last_frame = None
        self.raw_frame = None
        self.w = 320
        self.h = 180
        self.display = display
        self.init_success = True
        self.device = device
        self.plug_id = 2
        self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=1)
        self._check_camera_ready()

        sam_checkpoint = f"{weights_path}/pretrained_weights/sam_vit_h_4b8939.pth"
        aot_checkpoint = f"{weights_path}/pretrained_weights/AOTT_PRE_YTB_DAV.pth"
        grounding_dino_checkpoint = f"{weights_path}/pretrained_weights/groundingdino_swint_ogc.pth"

        self.model = tracking_SAM.main_tracker(sam_checkpoint, aot_checkpoint, grounding_dino_checkpoint)

    def _check_camera_ready(self):

        self.last_frame = None
        rospy.logdebug(
            "Waiting for '{}' to be READY...".format(self._topic_name))
        while self.last_frame is None and not rospy.is_shutdown():
            try:
                self.last_frame = rospy.wait_for_message(
                    '{}'.format(self._topic_name), Image, timeout=5.0)
                rospy.logdebug(
                    "Current '{}' READY=>".format(self._topic_name))
                self.start_time = rospy.get_time()
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(self._topic_name))
        return self.last_frame

    def image_callback(self, msg):
        try:
            self.raw_frame = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
            return

    def process_frame(self, frame, display=True):

        frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
        image_np_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if self.model.is_tracking():

            pred_np_hw = self.model.propagate_one_frame(image_np_rgb)
            pred_np_hw = pred_np_hw.astype(np.uint8)
            pred_np_hw[pred_np_hw > 0] = 1
            self.plug_mask = (pred_np_hw).astype(int) * self.plug_id

            self.last_frame = self.plug_mask

            if display:
                mask = (self.last_frame == self.plug_id).astype(float)
                self.mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                seg_show = (self.mask_3d * frame).astype(np.uint8)
                cv2.imshow("Mask Image", seg_show)

                red_overlay = np.dstack((np.zeros_like(pred_np_hw), np.zeros_like(pred_np_hw), pred_np_hw))
                viz_img = cv2.addWeighted(frame, 0.5, red_overlay, 0.5, 0)

                cv2.imshow("Raw Image", viz_img)
                cv2.waitKey(1)

        else:
            print('annotate first img')
            cv2.imshow('Video', frame)
            cv2.waitKey(0)
            self.model.annotate_init_frame(image_np_rgb)
            cv2.destroyWindow('Video')
    def shrink_mask(self, mask, shrink_percentage=10):
        """
        Shrink the object in the mask by a certain percentage of its area.

        :param mask: Binary mask image (object = 255, background = 0)
        :param shrink_percentage: Percentage to shrink the object
        :return: Shrunk mask
        """
        # Calculate the original area of the object
        original_area = np.sum(mask > 0)

        # Calculate the target area after shrinkage
        target_area = original_area * (1 - shrink_percentage / 100.0)

        # Create a structuring element (kernel) for erosion
        kernel = np.ones((3, 3), np.uint8)  # Small 3x3 kernel

        # Iteratively erode the mask until the target area is reached
        shrunk_mask = mask.copy()
        while np.sum(shrunk_mask > 0) > target_area:
            shrunk_mask = cv2.erode(shrunk_mask, kernel, iterations=1)

        return shrunk_mask

    def get_raw_frame(self):
        # rospy.wait_for_message('/zedm/zed_node/rgb/image_rect_color')
        return self.raw_frame if not isinstance(self.raw_frame, Image) else None

    def get_frame(self):
        # rospy.wait_for_message('/zedm/zed_node/rgb/image_rect_color')
        return self.last_frame if not isinstance(self.last_frame, Image) else None


if __name__ == "__main__":

    rospy.init_node('Seg')
    seg_cam = SegCameraSubscriber()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        seg_cam.process_frame(seg_cam.get_raw_frame())
        rate.sleep()
