import cv2
import numpy as np

from algo.deploy.env.finger import Finger


class Hand():

    def __init__(self, dev_names=None, fix=((0, 0), (0, 0), (0, 0))):
        """
        Finger Device class for a single Finger
        :param serial: Finger device serial
        :param name: Human friendly identifier name for the device
        """

        if dev_names is None:
            dev_names = [2, 0, 4]

        self.finger_left = Finger(dev_name=dev_names[0], serial='/dev/video', fix=fix[0])
        self.finger_right = Finger(dev_name=dev_names[1], serial='/dev/video', fix=fix[1])
        self.finger_bottom = Finger(dev_name=dev_names[2], serial='/dev/video', fix=fix[2])
        self.init_success = False

        self.left_bg, self.right_bg, self.bottom_bg = self.get_frames()

    def init_hand(self):
        """
        Sets stream resolution based on supported streams in Finger.STREAMS
        :param resolution: QVGA or VGA from Finger.STREAMS
        :return: None
        """
        self.finger_right.connect()
        self.finger_left.connect()
        self.finger_bottom.connect()
        self.init_success = True

    def get_frames(self, diff=True):
        """
        Returns a single image frame for the device
        :param transpose: Show direct output from the image sensor, WxH instead of HxW
        :return: Image frame array
        """

        left = self.finger_left.get_frame()
        right = self.finger_right.get_frame()
        bottom = self.finger_bottom.get_frame()

        if diff:
            left = self._subtract_bg(left, self.left_bg) * self.finger_left.mask_resized
            right = self._subtract_bg(right, self.right_bg) * self.finger_right.mask_resized
            bottom = self._subtract_bg(bottom, self.bottom_bg) * self.finger_bottom.mask_resized

        return left, right, bottom

    def show_fingers_view(self):
        """
        Creates OpenCV named window with live view of Finger device, ESC to close window
        :param ref_frame: Specify reference frame to show image difference
        :return: None
        """
        left_bg, right_bg, bottom_bg = self.get_frames()

        while True:

            left, right, bottom = self.get_frames()

            cv2.imshow("Hand View", np.concatenate((left, right, bottom), axis=1))

            diff_left = self._subtract_bg(left, left_bg) * self.finger_left.mask_resized
            diff_right = self._subtract_bg(right, right_bg) * self.finger_right.mask_resized
            diff_bottom = self._subtract_bg(bottom, bottom_bg) * self.finger_bottom.mask_resized

            cv2.imshow("Hand View\tLeft\tRight\tMiddle", np.concatenate((diff_left, diff_right, diff_bottom), axis=1))

            if cv2.waitKey(1) == 27:
                break

        cv2.destroyAllWindows()

    def _subtract_bg(self, img1, img2, offset=0.5):
        img1 = np.int32(img1)
        img2 = np.int32(img2)
        diff = img1 - img2
        diff = diff / 255.0 + offset
        return diff


if __name__ == "__main__":
    import os

    pc_name = os.getlogin()

    tactile = Hand()

    tactile.init_hand()

    tactile.show_fingers_view()