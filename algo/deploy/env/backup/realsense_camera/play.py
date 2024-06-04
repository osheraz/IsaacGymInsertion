import os
import sys

import argparse
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import multiprocessing as mp
import yaml

external_cam= True
w= 320
h= 180
cam_type= 'd'
save_im= False
far_clip= 1.0
near_clip= 0.0
dis_noise= 0.001
def process_depth_image(depth_image):
    # These operations are replicated on the hardware
    # resize_transform = transforms.Resize((cfg.depth.resized[1], cfg.depth.resized[0]),
    #                                                       interpolation=transforms.InterpolationMode.BICUBIC)

    depth_image = crop_depth_image(depth_image)
    # depth_image += dis_noise * 2 * (np.random.random(1) - 0.5)[0]
    depth_image = np.clip(depth_image, -far_clip, -near_clip)
    # depth_image = resize_transform(depth_image[None, :]).squeeze()
    depth_image = normalize_depth_image(depth_image)
    return depth_image


def normalize_depth_image(depth_image):
    depth_image = depth_image * -1
    depth_image = (depth_image - near_clip) / (far_clip - near_clip) - 0.5
    return depth_image


def crop_depth_image(depth_image):
    # crop 30 pixels from the left and right and and 20 pixels from bottom and return croped image
    return depth_image
    # return depth_image[:-2, 4:-4]

def main():
    align = rs.align(rs.stream.color)

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = aligned_frames.get_depth_frame()

            depth_frame = rs.decimation_filter(1).process(depth_frame)
            depth_frame = rs.disparity_transform(True).process(depth_frame)
            depth_frame = rs.spatial_filter().process(depth_frame)
            depth_frame = rs.temporal_filter().process(depth_frame)
            depth_frame = rs.disparity_transform(False).process(depth_frame)
            depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_image = np.asanyarray(depth_frame.get_data()) / 1000
            depth_image = np.expand_dims(depth_image, axis=0)

            # color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            # cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('depth image', depth_image)
            # depth_image = process_depth_image(-1 * depth_image)

            cv2.imshow("Depth Image", depth_image.transpose(1, 2, 0))
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            cv2.imshow('Depth Stream', depth_color_image)
            key = cv2.waitKey(1)


    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()