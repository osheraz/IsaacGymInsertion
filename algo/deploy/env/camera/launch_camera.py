from dataclasses import dataclass
from multiprocessing import Process
from typing import List, Optional, Tuple

import tyro

from camera_node import ZMQServerCamera, ZMQServerCameraFaster


@dataclass
class Args:
    hostname: str = "127.0.0.1"
    robot_ip: str = "111.111.1.1"
    faster: bool = True
    cam_names: Tuple[str, ...] = "0"
    ability_gripper_grip_range: int = 110
    img_size: Optional[Tuple[int, int]] = None  # (320, 240)


def launch_server_cameras(port: int, camera_id: List[str], args: Args):
    from realsense_camera import RealSenseCamera

    camera = RealSenseCamera(camera_id, img_size=args.img_size)

    if args.faster:
        server = ZMQServerCameraFaster(camera, port=port, host=args.hostname)
    else:
        server = ZMQServerCamera(camera, port=port, host=args.hostname)
    print(f"Starting camera server on port {port}")
    server.serve()


CAM_IDS = {
    "0": "026322070269",
}


def create_camera_server(args: Args) -> List[Process]:
    ids = [CAM_IDS[name] for name in args.cam_names]
    camera_port = 5000
    # start a single python process for all cameras
    print(f"Launching cameras {ids} on port {camera_port}")
    server = Process(target=launch_server_cameras, args=(camera_port, ids, args))
    return server


def main(args):
    camera_server = create_camera_server(args)
    print("Starting camera server process")
    camera_server.start()


if __name__ == "__main__":
    main(tyro.cli(Args))