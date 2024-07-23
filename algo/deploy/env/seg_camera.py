import rospy

from sensor_msgs.msg import Image
from algo.deploy.env.env_utils.deploy_utils import msg_to_pil, image_msg_to_numpy
import numpy
import cv2
from algo.models.FastSAM.live_sam import FastSAM, FastSAMPrompt
import time

def masks_to_bool(masks):
    if type(masks) == numpy.ndarray:
        return masks.astype(bool)
    return masks.cpu().numpy().astype(bool)

class SegCameraSubscriber:

    def __init__(self, topic='/zedm/zed_node/rgb/image_rect_color', display=False, device='cuda:0'):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """
        self.last_frame = None
        self.w = 320
        self.h = 180
        self.display = display
        self.init_success = False
        self.device = device
        self._topic_name = rospy.get_param('~topic_name', '{}'.format(topic))
        rospy.loginfo("(topic_name) Subscribing to Images to topic  %s", self._topic_name)
        self.model = FastSAM('/home/roblab20/osher3_workspace/src/isaacgym/python/'
                             'IsaacGymInsertion/isaacgyminsertion/outputs/weights/FastSAM-x.pt')
        self._image_subscriber = rospy.Subscriber(self._topic_name, Image, self.image_callback, queue_size=2)
        self._check_camera_ready()

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
                self.init_success = True
                self.start_time = rospy.get_time()
                # init model overrides
                _ = self.model.predict(source=self.last_frame,
                                       device=self.device,
                                       retina_masks=True,
                                       imgsz=640,
                                       conf=0.4,
                                       iou=0.9, )
            except:
                rospy.logerr(
                    "Current '{}' not ready yet, retrying for getting image".format(self._topic_name))
        return self.last_frame

    def image_callback(self, msg):
        try:
            frame = image_msg_to_numpy(msg)
        except Exception as e:
            print(e)
        else:
            start = time.perf_counter()

            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            results = self.model.predict(source=frame,
                                         device=self.device,
                                         retina_masks=True,
                                         imgsz=(384, 480),
                                         conf=0.6,
                                         iou=0.9,)

            specific_rect = {
                "x_min": 100,
                "y_min": 70,
                "x_max": 260,
                "y_max": 160
            }
            # Define the specific dimensions (width and height)
            max_dims = {
                "width": 50,
                "height": 70
            }

            min_dims = {
                "width": 5,
                "height": 30
            }

            points_to_exclude = [(147, 156), (147, 156)]

            def is_box_within_rect(box, rect, exclude_points):
                x_min, y_min, x_max, y_max = box

                includes_exclude_points = any(
                    x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
                    for point in exclude_points
                )

                within_rect_and_dim = (x_min >= rect["x_min"] and y_min >= rect["y_min"] and
                        x_max <= rect["x_max"] and y_max <= rect["y_max"])

                return within_rect_and_dim and not includes_exclude_points

            def is_box_within_rect_and_dim(box, rect, max_dim, min_dims, exclude_points):
                x_min, y_min, x_max, y_max = box
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Check if the box is within the specific rectangle and matches the dimensions
                within_rect_and_dim = (
                        x_min >= rect["x_min"] and y_min >= rect["y_min"] and
                        x_max <= rect["x_max"] and y_max <= rect["y_max"] and
                        box_width < max_dim["width"] and box_height < max_dim["height"] and
                        box_width > min_dims["width"] and box_height > min_dims["height"]
                )

                # Check if the box includes any of the specific points to exclude
                includes_exclude_points = any(
                    x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
                    for point in exclude_points
                )

                return within_rect_and_dim and not includes_exclude_points


            seg, all = [], []
            for box in results[0].boxes:
                box = box.xyxy.cpu().numpy()[0]
                if is_box_within_rect(box, specific_rect, points_to_exclude):
                    all.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
                if is_box_within_rect_and_dim(box, specific_rect, max_dims, min_dims, points_to_exclude):
                    seg.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
            if len(seg) > 1:
                print('found more than 1 object')

            if len(seg) == 1:
                prompt_process = FastSAMPrompt(frame, results, device=self.device)
                ann = prompt_process.box_prompt(bbox=seg[0])[0]
                mask = masks_to_bool(ann)
                # mask_3d = numpy.repeat(mask[:, :, numpy.newaxis], 3, axis=2)
                # frame *= mask_3d
                #
                # end = time.perf_counter()
                # total_time = end - start
                # fps = 1 / total_time
                self.last_frame = mask

            else:
                for b in all:
                    cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
            if self.display:
                cv2.imshow("Depth Image", frame)
                cv2.waitKey(1)

    def get_frame(self):

        return self.last_frame


if __name__ == "__main__":

    rospy.init_node('Seg')
    tactile = SegCameraSubscriber()
    rate = rospy.Rate(60)

    while not rospy.is_shutdown():
        rate.sleep()
