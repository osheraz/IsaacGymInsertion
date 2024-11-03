## Deploy!


## Requirements - Only once
To recompile for python3 (melodic):
### 1. Install some prerequisites to use Python3 with ROS.

```bash
sudo apt update
sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy
```
### 2. Prepare catkin workspace

```bash
mkdir -p ~/osher3_workspace/src; cd ~/osher3_workspace
catkin_make
source devel/setup.bash
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r
```
### 3. Finally compile for Python 3

```bash
catkin_make --cmake-args \
            -DCMAKE_BUILD_TYPE=Release \
            -DPYTHON_EXECUTABLE=/usr/bin/python3 \
            -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
            -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

### source the workspace
```bash
echo $PYTHONPATH
alias osher3='source ~/osher3_workspace/devel/setup.sh'
echo $PYTHONPATH
```

### Disable conda auto activate, you are messing with my paths
```bash
conda deactivate # or comment conda stuff in the ~/.bashrc
```

### Pytorch and cuda
```bash
 pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
 # actually any torch + cu that's works with zed api.
```

### Issues
### 4. yaml issue
```bash
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg
```

### 4. cv_bridge Issue
```bash
sudo apt-get install python-catkin-tools python3-dev python3-numpy
```

# Usage 

### 1. Turn on everything
- Turn on the kuka\kinova by pressing the switch, wait until everything is up.
- Connect the FT sensor (just plug it).
- Turn on the hand (there is a little switch near the RED LED).
- Connect the usb of the tactile finger, wait a few sec and check connection:
  - ```bash
    v4l2-ctl --list-devices
    ```
  - You should see 3 cameras by the following order (*IMPORTANT*)
    ```
    Piwebcam: UVC Camera (usb-0000:00:14.0-10.1):
       /dev/video2
       /dev/video3

    Piwebcam: UVC Camera (usb-0000:00:14.0-10.2):
        /dev/video4
        /dev/video5
    
    Piwebcam: UVC Camera (usb-0000:00:14.0-10.3):
        /dev/video0
        /dev/video1
    
    # Pay attention to the order, i'm having issues with setting udev rules. fix it if you can.
    # You can also turn on display
    
    rosrun tactile_insertion tactile_display.py
    ```
    
### 2. Launch zed
Set the api on 'neural' depth prediction.
```bash
roslaunch zed_wrapper zedm.launch
```
## Kuka Deploy
### 2. Launch the system
```bash
roslaunch tactile_insertion env_bringup.launch 
```

### 3. Set velocity and acc (Do this with caution)
```bash
rosservice call /iiwa/configuration/pathParameters "{joint_relative_velocity: 0.05, joint_relative_acceleration: 0.05, override_joint_acceleration: 1}"
```

### 4. Run the arm controller(jsut a publisher right now)
```bash
ROS_NAMESPACE=iiwa rosrun tactile_insertion moveit_manipulator.py
```

## Kinova Deploy
### 2. Launch the system
```bash
roslaunch tactile_insertion kinova_bringup.launch 
```


### 3. Run the arm controller(just a publisher right now)
```bash
rosrun tactile_insertion moveit_kinova.py
```
