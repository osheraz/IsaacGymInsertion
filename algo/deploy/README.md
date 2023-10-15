## Deploy!

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

### dont source every terminal, use alias to the ~/.bashrc
```bash
echo $PYTHONPATH
alias osher3='source ~/osher3_workspace/devel/setup.sh'
echo $PYTHONPATH
```

### Disable conda auto activate, your messing with my paths
```bash
conda deactivate # or comment conda stuff in the ~/.bashrc
```

### Pytorch and cuda
```bash
 pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

# Issues
### 4. yaml issue
```bash
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg
```

### 4. cv_bridge Issue
```bash
sudo apt-get install python-catkin-tools python3-dev python3-numpy
```


