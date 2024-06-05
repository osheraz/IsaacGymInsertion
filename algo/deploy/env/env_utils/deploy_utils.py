import numpy as np
from PIL import Image as PILImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# bridge = CvBridge()
import cv2


def msg_to_pil(msg: Image) -> PILImage.Image:
    np_img = image_msg_to_numpy(msg)
    pil_image = PILImage.fromarray(np_img)
    return pil_image


# not in use currently
def pil_to_msg(pil_img: PILImage.Image, encoding="mono8") -> Image:
    img = np.asarray(pil_img)
    ros_image = Image(encoding=encoding)
    ros_image.height, ros_image.width, _ = img.shape
    ros_image.data = img.ravel().tobytes()
    ros_image.step = ros_image.width
    return ros_image


def image_msg_to_numpy(msg, empty_value=None, output_resolution=None, max_depth=None, use_bridge=False) -> np.ndarray:
    """
    Converts a ROS image message to a numpy array, applying format conversions, depth normalization, and resolution scaling.

    This function supports direct numpy conversions without using CvBridge as well as conversions via CvBridge if specified.

    Args:
        msg (Image): The ROS image message to convert.
        empty_value (float, optional): A value to replace missing or invalid data points in the image. Defaults to None.
        output_resolution (tuple of int, optional): The desired resolution (width, height) for the output image. Defaults to the input image's resolution.
        max_depth (int): The maximum depth used for normalizing depth images. Only applies to depth images. Defaults to 5000.
        use_bridge (bool): Set to True to use CvBridge for converting image messages. Defaults to False.

    Returns:
        np.ndarray: The converted image as a numpy array.
    """

    # Set default output resolution to the input message's resolution if not specified
    if output_resolution is None:
        output_resolution = (msg.width, msg.height)

    # Determine the type of image encoding
    is_rgb = "8" in msg.encoding.lower()
    is_depth16 = "16" in msg.encoding.lower()
    is_depth32 = "32" in msg.encoding.lower()

    if not use_bridge:
        # Convert the ROS image to a numpy array directly
        if is_rgb:
            data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)[:, :, :3].copy()
        elif is_depth16:
            data = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy() / -1000
            # max_depth_clip = max_depth if max_depth else np.max(data)
            # data = 1 - (np.clip(data, a_min=0, a_max=max_depth_clip) / max_depth_clip)
            data = np.array(data.astype(np.float32)) #* 255
        elif is_depth32:
            data = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width).copy()
    else:
        # Use CvBridge to convert the ROS image
        bridge = CvBridge()
        try:
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if is_rgb:
                data = cv_img[:, :, :3]
            elif is_depth16 or is_depth32:
                data = np.clip(cv_img, 0, max_depth) / max_depth
                data = np.uint8(data * 255)
        except CvBridgeError as e:
            print(f"Error converting image: {e}")
            return None

    # Replace specified 'empty values' or NaNs with a fixed value
    if empty_value:
        mask = np.isclose(abs(data), empty_value)
    else:
        mask = np.isnan(data)

    fill_value = np.percentile(data[~mask], 99)
    data[mask] = fill_value

    # Resize the image if a specific output resolution is set
    if output_resolution != (msg.width, msg.height):
        data = cv2.resize(data, dsize=(output_resolution[0], output_resolution[1]), interpolation=cv2.INTER_AREA)

    return data