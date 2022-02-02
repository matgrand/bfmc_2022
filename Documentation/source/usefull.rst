Usage and Usefull link
======================

In this section, you can find some useful links and some documentations for basic
raspberry usage and image processing. 


Usage of Raspberry Pi 
---------------------

Install the image on the SD card
````````````````````````````````
Download the Raspbery Pi OS  (either Desktop or Lite version) from the following link: 
- `Images`_
.. _`Images`: https://www.raspberrypi.org/software/operating-systems/ 

Mount the chosen image on your RPi with the help of balenaetcher:
- `BalenaEtcher`_
.. _`BalenaEtcher`: https://www.balena.io/etcher/

Setting up WLAN
```````````````
Without directly connecting to the RPi, you can set the LAN from the PC by creating a wpa_supplicanf.conf file under boot folder, in your SD card.
You can set the content as follows:

| ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
| update_config=1
| country=RO
| network=
| {
| ssid="SSID"
| psk="Passwd"
| }

You can also edit the file later on. it can be found under /etc/wpa_supplicant/wpa_supplicant.conf.

Configuring your Raspberry Pi
``````````````````````````````
There is a series of configuration parameters that can be edited on the PC, prior to starting 
the RPI. Some of these are stored in the **config.txt** document, found in the /boot/ folder. 
By editing this document one could, for example, adjust the way the display is detected and 
the desktop displayed or overclock the Raspberry Pi (or return to default clock settings). 
A useful setting is the enabling of the Serial Console so that the Raspberry Pi’s terminal can
be accessed using a serial connection. The file can be also visually edited by using the raspi-config 
application while accessing the RPi. 

Editing the **config.txt** file:
    - https://www.makeuseof.com/tag/edit-boot-config-file-raspberry-pi 
    - https://elinux.org/R-Pi_configuration_file

Some more configurations can be done prior to starting the RPi, such as enabling SSH connection, 
Camera connection or I2C connection. In order to do so, you just have to create an empty file with the 
name of the interface, without an extension, under /boot/ folder. The same parameters can also be edited
with the raspi-config application while accessing the RPi.

Raspberry Pi IP address
````````````````````````
For most of the applications, the IP of the RPi is crucial. After you set up the network for your RPi and 
power it up, you can find the address with different means, such as: using a network scan tool on your PC 
(nmap for linux); connecting to your router and finding your connected devices; ping your device by using 
it's username: 
| ping raspberrypi.local
|
| PING raspberrypi.local (192.168.1.131): 56 data bytes
| 64 bytes from 192.168.1.131: icmp_seq=0 ttl=255 time=2.618 ms
|
| More information can be found at the following link:
- https://www.raspberrypi.org/documentation/remote-access/ip-address.md


Development on the Raspberry Pi 
--------------------------------

You have multiple ways of accessing the RPi: 
- Direct development approach, by using it as a standalone computer and connecting all the pheripherials.
- Direct connection approach, by connecting to it with a Serial cable or with an ethernet cable.
- Remote connection approach, by connecting to it via ssh terminal communication and ftp for file transfer to the remote.

Direct development approach	
````````````````````````````
You can connect all the pheripherials and develop on it as you would do on a PC. The suggested OS is the desktop version. Follow this guide for more info:
- `Setting up your Raspberry Pi`_
.. _`Setting up your Raspberry Pi`: https://projects.raspberrypi.org/en/projects/raspberry-pi-setting-up 

Direct connection approach. 	
````````````````````````````
This option for connecting to your Raspberry Pi in phisical format can be done in may ways, but the most 
handful ones are with a TTL cable or with an Ethernet cable. By using the serial cable (e.g. 
https://www.ftdichip.com/Products/Cables/USBTTLSerial.htm) you can connect to your system for performing 
initial setup, such as WLAN SSID and password. You should have the Serial Console enabled for being able to 
connect to the Raspberry Pi in this way. This can be done easily by editing the config.txt file of the 
system. Another way in doing so is by setting up a LAN between the RPi and the PC, connecting them via 
ethernet and accessing it via ssh. Below you can find some useful resources that describe the steps 
required for connecting to the terminal in this way.
a. https://learn.sparkfun.com/tutorials/headless-raspberry-pi-setup/serial-terminal
#. https://www.instructables.com/Set-Up-Raspberry-Pi-4-Through-Laptoppc-Using-Ether/


Remote Connection approach
```````````````````````````
The best combination in programming your RPi remotely is a combination between a ssh connection for 
terminal commands and a SFTP connection for file sharing (this way, the OS installed can be the lite version). 
On linux the SSH connection can be done in any terminal (ssh user@IP) and the SFTP connection in any file 
explorer (other locations->Connect to Server: sftp://ip). On windows, the PUTTY application can be used for 
ssh connection and the WINSCP can be used for file sharing.  
- https://www.behind-the-scenes.co.za/using-winscp-to-connect-to-a-raspberry-pi

VNC is a graphical desktop sharing system that allows you to remotely control the desktop interface of 
one computer. Running VNC Server, in our case Raspberry Pi, you can connect to it from another computer 
or mobile device (running VNC Viewer). The tutorial below describes how work on it from another device by remote control.
- https://www.raspberrypi.org/documentation/remote-access/vnc


Python on Raspberry Pi
-----------------------
Python is an interpreted high-level general-purpose programming language. It is versatile, easy to use and fast 
to develop, which makes it ideal for rapid prototyping. Some of it's downsides is that it has speed limitation, 
it does not perform well with multithreading, making it inferior in performances to other, low-level programming languages.
The links below represent a good starting point for this type of applications.
- https://www.raspberrypi.org/documentation/usage/python
- https://www.digikey.com/en/maker/blogs/2018/how-to-run-python-programs-on-a-raspberry-pi

CPP on Raspberry Pi
-----------------------
It is one of the oldest, most used and most efficient programming languages. It has a wide support, it is powerful, 
fast and has a small amount of standard libraries. It's major downside beying it's complexity. 
- https://www.aranacorp.com/en/program-your-raspberry-pi-with-c/


Robot Operating System 
-----------------------

ROS (Robot Operating System) is a robotic middleware (a collection of software frameworks for writing robot software). Although ROS is not an operating system , 
it provides services designed for a heterogeneous computer cluster such as hardware abstraction, low-level device control, implementation of commonly used 
functionality, message-passing between processes, and package management. Running sets of ROS-based processes(scripts) are represented in a graph architecture 
where processing takes place in nodes that may receive, post and multiplex sensor data, control, state, planning, actuator, and other messages via "topics", 
"services" and "actions". Despite the importance of reactivity and low latency in robot control, ROS itself is not a real-time OS (RTOS).
The main client libraries (C++, Python, and Lisp) are released under the terms of the BSD license as such as the other majority of available packages. 

ROS distributions
`````````````````
A ROS distribution is a versioned set of ROS packages. These are a kin to Linux distributions (e.g. Ubuntu). The purpose of the ROS distributions is to let developers work 
against a relatively stable codebase until they are ready to roll everything forward. The latest stable distribution that we encourege you to use is ROS Noetic, together 
with Ubuntu 20.04.

ROS installation
````````````````
You first have to install a supported operating system, either on your device or on a virtual machine. We suggest to not use a virtual machine since it may not have the same 
specifications as if installed directly on the HDD/SSD. 
For the Melodic installation, you can follow this link: 
- http://wiki.ros.org/melodic/Installation/Ubuntu

In order to get started with the ROS functionalities, you can follow this guides:
    - http://wiki.ros.org/ROS/Tutorials


Image processing 
-----------------
In this part, you can find some useful link for image processing on Raspberry pi.

Basic Python libraries:
    - `Opencv Official Documentation`_
    - `Opencv with python`_
    - `Lane detection link 1`_
    - `Lane detection link 2`_
    - `Traffic sign recognition link 1`_
    - `Traffic sign recognition link 2`_
    - `Automatic exposure control`_
    - `Automatic gain control`_

.. _`Opencv Official Documentation`: https://docs.opencv.org/4.1.2
.. _`Opencv with python`: https://www.youtube.com/watch?v=kdLM6AOd2vc&list=PLS1QulWo1RIa7D1O6skqDQ-JZ1GGHKK-K
.. _`Lane detection link 1`: https://www.youtube.com/watch?v=eLTLtUVuuy4
.. _`Lane detection link 2`: https://www.youtube.com/watch?v=CvJN_jSVm30
.. _`Traffic sign recognition link 1`: https://www.youtube.com/watch?v=QHra6Xf6Mew
.. _`Traffic sign recognition link 2`: https://www.youtube.com/watch?v=LjK0hD3dfrY&ab_channel=gsnikitin
.. _`Automatic exposure control`: https://www.researchgate.net/publication/228405828_Automatic_camera_exposure_control
.. _`Automatic gain control`: https://ieeexplore.ieee.org/document/1315984

Articles for Road Sign Recognition:

    - A. Mogelmose, M. M. Trivedi and T. B. Moeslund, "Vision-Based Traffic Sign Detection and Analysis for Intelligent Driver Assistance Systems: Perspectives and Survey," 
      in IEEE Transactions on Intelligent Transportation Systems, vol. 13, no. 4, pp. 1484-1497, Dec. 2012. [`link2 <https://ieeexplore.ieee.org/document/6335478/>`_]
    - S. Maldonado-Bascon, S. Lafuente-Arroyo, P. Gil-Jimenez, H. Gomez-Moreno and F. Lopez-Ferreras, "Road-Sign Detection and Recognition Based on Support Vector Machines," 
      in IEEE Transactions on Intelligent Transportation Systems, vol. 8, no. 2, pp. 264-278, June 2007. [`link3 <https://ieeexplore.ieee.org/document/4220659>`_]
    - Y. Han and E. Oruklu, "Traffic sign recognition based on the NVIDIA Jetson TX1 embedded system using convolutional neural networks," 
      2017 IEEE 60th International Midwest Symposium on Circuits and Systems (MWSCAS), Boston, MA, 2017, pp. 184-187. [`link4 <https://ieeexplore.ieee.org/document/8052891>`_]

Articles for Lane detection and tracking:
    - R. Danescu, S. Nedevschi, M. M. Meinecke and T. B. To, "Lane Geometry Estimation in Urban Environments Using a Stereovision System," 
      2007 IEEE Intelligent Transportation Systems Conference, Seattle, WA, 2007, pp. 271-276. [`link5 <https://ieeexplore.ieee.org/document/4357686>`_]
    - R. Labayrade, J. Douret and D. Aubert, "A multi-model lane detector that handles road singularities," 
      2006 IEEE Intelligent Transportation Systems Conference, Toronto, Ont., 2006, pp. 1143-1148. [`link6 <https://ieeexplore.ieee.org/document/1707376>`_]
    - Yue Dong, Jintao Xiong, Liangchao Li and Jianyu Yang, "Robust lane detection and tracking for lane departure warning," 
      2012 International Conference on Computational Problem-Solving (ICCP), Leshan, 2012, pp. 461-464. [`link7 <https://ieeexplore.ieee.org/document/6384266>`_]
