import numpy as np
import cv2
import math

# fix = (4, 4)   # RRRGGGBBB
# fix = (7,-3)    # white
# fix = (5 , -3)    # rgbrgbrgb
# fix = (4, 4)   # RRRGGGBBB

def T_inv(T_in):
    R_in = T_in[:3, :3]
    t_in = T_in[:3, [-1]]
    R_out = R_in.T
    t_out = -np.matmul(R_out, t_in)
    return np.vstack((np.hstack((R_out, t_out)), np.array([0, 0, 0, 1])))


def convert_quat_xyzw_to_wxyz(q):
    q[0], q[1], q[2], q[3] = q[3], q[0], q[1], q[2]
    return q


def convert_quat_wxyz_to_xyzw(q):
    q[3], q[0], q[1], q[2] = q[0], q[1], q[2], q[3]
    return q


def _diff(target, base):
    diff = (target * 1.0 - base) / 255.0 + 0.5
    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 1.0 + 0.5
    diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
    return diff_abs


def _diff_abs(target, base):
    diff = (target * 1.0 - base) / 255.0 + 0.5
    diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 1.0 + 0.5
    diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)

    return diff_abs


def _smooth(target, k=64):
    kernel = np.ones((k, k), np.float32)
    kernel /= kernel.sum()
    diff_blur = cv2.filter2D(target, -1, kernel)
    return diff_blur


def raw_image_2_height_map(img, ref_frame):
    lighting_threshold = 2
    img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref_GRAY = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
    diff_raw = ref_GRAY - img_GRAY - lighting_threshold
    diff_mask = (diff_raw < 100).astype(np.uint8)
    diff = diff_raw * diff_mask + lighting_threshold
    # diff[diff>self.max_index] = self.max_index
    diff = cv2.GaussianBlur(diff.astype(np.float32), (7, 7), 0).astype(
        int)  # this filter can decrease the lighting_threshold to 2
    map = diff  # self.GRAY_Height_list[diff] - self.GRAY_Height_list[self.lighting_threshold]
    for kernel in [7, 7]:
        map = cv2.GaussianBlur(map.astype(np.float32), (kernel, kernel), 0)

    contact_gray_base = 20
    depth_k = 160
    contact_show = np.zeros_like(map)
    contact_show[map > 0] = contact_gray_base
    depth_map = map * depth_k + contact_show
    depth_map = depth_map.astype(np.uint8)

    return depth_map


def resizeAndPad(img, size, padColor=255):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect >= aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(padColor,
                                              (list, tuple, np.ndarray)):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT,
                                    value=padColor)

    return scaled_img

def _mask(target, size=None):
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    gray_circle = cv2.adaptiveThreshold(
        gray_target, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -30)

    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 0.2
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.minThreshold = 1
    params.thresholdStep = 1
    params.maxThreshold = 255
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(target)

    # im_with_keypoints = cv2.drawKeypoints(gray_circle, keypoints, np.array([]), (255, 255, 255),
    #                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    img = gray_circle.copy()
    for x in range(1, len(keypoints)):
        img = cv2.circle(img, (int(keypoints[x].pt[0]), int(keypoints[x].pt[1])),
                         radius=int(min(keypoints[x].size, 5)), color=(255,255,255), thickness=-1)

    return img if size is None else cv2.resize(img, size)

def center_mask(size=(640, 480), rad=80, fix=(0,0)):
    """
        used to filter center circular area of a given image,
        corresponding to the AllSight surface area
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2 - fix[0], size[1] // 2 - fix[1])
    m_radius = rad
    m = cv2.circle(m, m_center, m_radius, 255, -1)
    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)

    return mask

def ring_mask(size=(640, 480), rad=50, fix=(0,0)):
    """
        used to filter center circular area of a given image,
        corresponding to the AllSight surface area
    """
    m = np.zeros((size[1], size[0]))
    m_center = (size[0] // 2 - fix[0], size[1] // 2 - fix[1])
    m_radius = rad
    m = cv2.circle(m, m_center, m_radius // 2, 255, -1)
    m = cv2.circle(m, m_center, m_radius // 3, 0, -1)

    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)

    return mask

def circle_mask(size=(640, 480), border=0, fix=(0,0)):
    """
        used to filter center circular area of a given image,
        corresponding to the AllSight surface area
    """
    m = np.zeros((size[1], size[0]))

    m_center = (size[0] // 2 - fix[0], size[1] // 2 - fix[1])
    m_radius = min(size[0], size[1]) // 2 - border - max(abs(fix[0]), abs(fix[1]))
    m = cv2.circle(m, m_center, m_radius, 255, -1)
    m /= 255
    m = m.astype(np.float32)
    mask = np.stack([m, m, m], axis=2)

    return mask


def align_center_mask(img, mask, size=(640, 480)):
    m = mask[:, :, 0]

    col_sum = np.where(np.sum(m, axis=0) > 0)
    row_sum = np.where(np.sum(m, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    m = m.astype(np.float32)[y1:y2, x1:x2]

    zero_axis_fill = (size[1] - m.shape[0])
    one_axis_fill = (size[0] - m.shape[1])

    top = zero_axis_fill // 2
    bottom = zero_axis_fill - top
    left = one_axis_fill // 2
    right = one_axis_fill - left
    padded_img = np.pad(img[y1:y2, x1:x2], ((top, bottom), (left, right), (0, 0)), mode='constant')

    return padded_img

def align_center(img, fix, size=(640, 480), pad=False):

    center_x, center_y = size[0] // 2 - fix[0], size[1] // 2 - fix[1]
    extra = max(abs(fix[0]), abs(fix[1]))

    half_size = min(size[0], size[1]) // 2 - max(abs(fix[0]), abs(fix[1]))

    # half_size = min(size) // 2
    left = max(0, center_x - half_size)
    top = max(0, center_y - half_size)
    right = min(img.shape[1], center_x + half_size)
    bottom = min(img.shape[0], center_y + half_size)

    cropped_image = img[top:bottom, left:right]

    if pad:
        zero_axis_fill = (size[1] - cropped_image.shape[0])
        one_axis_fill = (size[1] - cropped_image.shape[1])
        top = zero_axis_fill // 2
        bottom = zero_axis_fill - top
        left = one_axis_fill // 2
        right = one_axis_fill - left
        cropped_image = np.pad(cropped_image, ((top, bottom), (left, right), (0, 0)), mode='constant')

    return cropped_image

def square_cut(img, size=480):
    col_sum = np.where(np.sum(img, axis=0) > 0)
    row_sum = np.where(np.sum(img, axis=1) > 0)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]

    img = img[y1:y2+1, x1:x2+1]

    zero_axis_fill = max(0, (size - img.shape[0]))
    one_axis_fill = max((size - img.shape[1]), 0)

    top = zero_axis_fill // 2
    bottom = zero_axis_fill - top
    left = one_axis_fill // 2
    right = one_axis_fill - left

    padded_img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant')

    # print(padded_img.shape)
    return padded_img

def get_coords(x, y, angle, imwidth, imheight):
    x1_length = (imwidth - x) / (math.cos(angle) + 1e-6)
    y1_length = (imheight - y) / (math.sin(angle) + 1e-6)
    length = max(abs(x1_length), abs(y1_length))
    endx1 = x + length * math.cos(angle)
    endy1 = y + length * math.sin(angle)

    x2_length = (imwidth - x) / (math.cos(angle + math.pi) + 1e-6)
    y2_length = (imheight - y) / (math.sin(angle + math.pi) + 1e-6)
    length = max(abs(x2_length), abs(y2_length))
    endx2 = x + length * math.cos((angle + math.pi))
    endy2 = y + length * math.sin((angle + math.pi))

    return (int(endx1), int(endy1)), (int(endx2), int(endy2))


class ContactArea:
    def __init__(
            self, base=None, draw_poly=False, contour_threshold=100, real_time=True, *args, **kwargs
    ):
        self.base = base
        self.draw_poly = draw_poly
        self.contour_threshold = contour_threshold
        self.real_time = real_time

    def __call__(self, target, base=None):
        base = self.base if base is None else base
        if base is None:
            raise AssertionError("A base sample must be specified for Pose.")
        diff = self._diff(target, base)
        diff = self._smooth(diff)
        contours = self._contours(diff)
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time == False:
            raise Exception("No contact area detected.")
        if self._compute_contact_area(contours, self.contour_threshold) == None and self.real_time == True:
            return None
        else:
            (
                poly,
                major_axis,
                major_axis_end,
                minor_axis,
                minor_axis_end,
            ) = self._compute_contact_area(contours, self.contour_threshold)
        if self.draw_poly:
            self._draw_major_minor(
                target, poly, major_axis, major_axis_end, minor_axis, minor_axis_end
            )
        return poly, major_axis, major_axis_end, minor_axis, minor_axis_end

    def _diff(self, target, base):
        diff = (target * 1.0 - base) / 255.0 + 0.5
        diff[diff < 0.5] = (diff[diff < 0.5] - 0.5) * 0.7 + 0.5
        diff_abs = np.mean(np.abs(diff - 0.5), axis=-1)
        return diff_abs

    def _smooth(self, target):
        kernel = np.ones((64, 64), np.float32)
        kernel /= kernel.sum()
        diff_blur = cv2.filter2D(target, -1, kernel)
        return diff_blur

    def _contours(self, target):
        mask = ((np.abs(target) > 0.04) * 255).astype(np.uint8)
        kernel = np.ones((16, 16), np.uint8)
        mask = cv2.erode(mask, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _draw_major_minor(
            self,
            target,
            poly,
            major_axis,
            major_axis_end,
            minor_axis,
            minor_axis_end,
            lineThickness=2,
    ):
        cv2.polylines(target, [poly], True, (255, 255, 255), lineThickness)
        cv2.line(
            target,
            (int(major_axis_end[0]), int(major_axis_end[1])),
            (int(major_axis[0]), int(major_axis[1])),
            (0, 0, 255),
            lineThickness,
        )
        cv2.line(
            target,
            (int(minor_axis_end[0]), int(minor_axis_end[1])),
            (int(minor_axis[0]), int(minor_axis[1])),
            (0, 255, 0),
            lineThickness,
        )

    def _compute_contact_area(self, contours, contour_threshold):
        for contour in contours:
            if len(contour) > contour_threshold:
                ellipse = cv2.fitEllipse(contour)
                poly = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                    int(ellipse[2]),
                    0,
                    360,
                    5,
                )
                center = np.array([ellipse[0][0], ellipse[0][1]])
                a, b = (ellipse[1][0] / 2), (ellipse[1][1] / 2)
                theta = (ellipse[2] / 180.0) * np.pi
                major_axis = np.array(
                    [center[0] - b * np.sin(theta), center[1] + b * np.cos(theta)]
                )
                minor_axis = np.array(
                    [center[0] + a * np.cos(theta), center[1] + a * np.sin(theta)]
                )
                major_axis_end = 2 * center - major_axis
                minor_axis_end = 2 * center - minor_axis
                return poly, major_axis, major_axis_end, minor_axis, minor_axis_end


if __name__ == "__main__":

    import cv2 as cv
    import numpy as np

    # The video feed is read in as
    # a VideoCapture object
    cap = cv.VideoCapture(4)
    c_mask = circle_mask((640, 480))
    c_mask = np.stack([c_mask, c_mask, c_mask], axis=2)
    # ret = a boolean return value from
    # getting the frame, first_frame = the
    # first frame in the entire video sequence
    ret, first_frame = cap.read()

    # Converts frame to grayscale because we
    # only need the luminance channel for
    # detecting edges - less computationally
    # expensive
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    # Creates an image filled with zero
    # intensities with the same dimensions
    # as the frame
    mask = np.zeros_like(first_frame)

    # Sets image saturation to maximum
    mask[..., 1] = 255

    while (cap.isOpened()):

        # ret = a boolean return value from getting
        # the frame, frame = the current frame being
        # projected in the video
        ret, frame = cap.read()

        frame = frame * c_mask
        # Opens a new window and displays the input
        # frame
        cv.imshow("input", frame)

        # Converts each frame to grayscale - we previously
        # only converted the first frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Calculates dense optical flow by Farneback method
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray,
                                           None,
                                           0.5, 3, 15, 3, 5, 1.2, 0)

        # Computes the magnitude and angle of the 2D vectors
        magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])

        # Sets image hue according to the optical flow
        # direction
        mask[..., 0] = angle * 180 / np.pi / 2

        # Sets image value according to the optical flow
        # magnitude (normalized)
        mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

        # Converts HSV to RGB (BGR) color representation
        rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)

        # Opens a new window and displays the output frame
        cv.imshow("dense optical flow", rgb)

        # Updates previous frame
        prev_gray = gray

        # Frames are read by intervals of 1 millisecond. The
        # programs breaks out of the while loop when the
        # user presses the 'q' key
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # The following frees up resources and
    # closes all windows
    cap.release()
    cv.destroyAllWindows()
