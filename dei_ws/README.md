# BFMC - Brain ROS Project

The project includes the software already present in the Brain project and it gives a good startup point for starting developing your car with ROS. 

If ROS is running on the car, you can also interact from the car with the official simulator by followint the Simulator guide in order to
publish/subscribe on it's topics(Camera, movements, posiiton, etc.)

## 1. Download the Raspbery Pi OS  (either Desktop or Lite version) from the following link: 
[Raspbian](https://www.raspberrypi.org/software/operating-systems/) 

If you are unfamiliar with Linux and ROS, we suggest starting with the desktop versions of the SWs and later migrate to the lite versions. 

## 2. Mount the image on the SD with the help of balenaetcher software:

[Balenaetcher](https://www.balena.io/etcher/) 

## 3. Setup the environment. Here you can choose the method of development:
a. Peripherals approach (connect keyboard, monitor and mouse attached) 
b. Remote approach (VNC or SSH connection). The following lines are explaining the ssh aproach
i. You can set up the network by creating a wpa_supplicant.conf file under boot (SD card).

ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=RO

network={
        ssid="SSID"
        psk="Passwd"
}

ii.	You can enable the ssh and the i2c connection by creating a file ssh. another file i2c. and a file camera. under boot (SD card).

iii.	Scan the network for your new IP when you power up te RPi

iv.	You can connect to the RPi IP from any terminal with the command ssh pi@192.168.*.* 


## 4. Add the ROS Debian repo to the OS
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu buster main" > /etc/apt/sources.list.d/ros-noetic.list'


## 5. Add official ROS key
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# 6. Pull all meta info from ROS noetic packages
sudo apt-get update && sudo apt-get upgrade

## 7. Install build dependencies
sudo apt-get install -y python-rosdep python-rosinstall-generator python-wstool python-rosinstall build-essential cmake

## 8. Install pip3
sudo apt install python3-pip

## 9. Install opencv:
sudo apt install libopencv-dev python3-opencv

## 10. Setup ROS dependency sources/repos
sudo rosdep init
rosdep update


## 11. Fetch and install ROS dependencies
mkdir -p ~/ros_catkin_ws
cd ~/ros_catkin_ws

Lite version:
rosinstall_generator ros_comm sensor_msgs cv_bridge --rosdistro noetic --deps --wet-only --tar > noetic-ros_comm-wet.rosinstall 
wstool init src noetic-ros_comm-wet.rosinstall


Desktop version
rosinstall_generator desktop --rosdistro noetic --deps --wet-only --tar > noetic-desktop-wet.rosinstall 
wstool init src noetic-desktop-wet.rosinstall



rosdep install -y --from-paths src --ignore-src --rosdistro noetic -r --os=debian:buster


## 12. Compile ROS packages
Since the ROS project is resource consuming, it is also recommended, but not mandatory, to increase the swap memory to 1 GB. You can decrease it afterwards. By following the same steps and setting it back to 100

		sudo dphys-swapfile swapoff

		sudoedit /etc/dphys-swapfile

			CONF_SWAPSIZE=1024

		sudo dphys-swapfile setup

		sudo dphys-swapfile swapon

sudo src/catkin/bin/catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release --install-space /opt/ros/noetic -j1 -DPYTHON_EXECUTABLE=/usr/bin/python3


## 13. Verify installation
source /opt/ros/noetic/setup.bash

roscore

## 14. You can also set the sourcing at the startup of each new terminal.
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

## 15. Fix missing package
sudo apt install libatlas-base-dev

## 16. Copy the project locally and move inside it

## 17. Install python dependencies
pip3 install -r requirements_rpi.txt

pip3 install numpy --upgrade

## 18. Set up the i2c communication for the IMU by following the Setting up the Raspberry Pi side from this tutorial: 
https://github.com/RPi-Distro/RTIMULib/tree/master/Linux 

## 19. For additional topics, check the official ROS documentation for Raspbery Pi: 
http://wiki.ros.org/ROSberryPi/

## 20. Build and run the prepared project
catkin_make

source devel/setup.bash

roslaunch utils run_automobile_remote.launch

It will run the same brain project only in ROS variant. If you want to test it remotely you can run the remotecontroltransmitter and camerareceiver from the startup project (don’t forget to edit the IP’s from CameraTransmitterProcess on rpi and the remotecontroltransmitterProcess from the remote).


## 21. If you wish to install additional ROS packages after the installation, you will have to:
cd ~/ros_catkin_ws

sudo rm -rf build_isolated/ devel_isolated/ src/

sudo apt-get install -y python-rosdep python-rosinstall-generator python-wstool python-rosinstall build-essential cmake

rosinstall_generator name_of_new_pkg --deps --exclude RPP > new_pkg.rosisntall

wstool init src new_pkg.rosisntall

sudo -s

nano /root/.bashrc

	add source /opt/ros/noetic/setup.bash

source /root/.bashrc

catkin_make_isolated --install -DCMAKE_BUILD_TYPE=Release  --install-space /opt/ros/noetic

nano /root/.bashrc

	remove source /opt/ros/noetic/setup.bash

exit
