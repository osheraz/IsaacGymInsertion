import os
import sys

import argparse
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
import multiprocessing as mp
import torch
import yaml


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
            # depth_frame = np.array(255*depth_frame.astype(np.float32),dtype=np.uint8)
            # depth_frame = rs.hole_filling_filter().process(depth_frame)

            depth_image = np.expand_dims(np.asanyarray(depth_frame.get_data()), axis=0) / 1000
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

            # cv2.namedWindow('depth image', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('depth image', depth_image)
            cv2.imshow("Depth Image", depth_image.transpose(1, 2, 0))

            key = cv2.waitKey(1)


    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()