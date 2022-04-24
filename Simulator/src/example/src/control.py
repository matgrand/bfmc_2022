#!/usr/bin/env python3

# Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE

import json
from pynput import keyboard
import cv2 as cv

from RcBrainThread import RcBrainThread
from std_msgs.msg import String
from sensor_msgs.msg import Image
from utils.msg import IMU
from cv_bridge import CvBridge
from time import time, sleep
import numpy as np
import os, sys

import rospy

class RemoteControlTransmitterProcess():
    # ===================================== INIT==========================================
    def __init__(self):
        """Run on the PC. It forwards the commans from the user via KeboardListenerThread to the RcBrainThread. 
        The RcBrainThread converts them into actual commands and sends them to the remote via a socket connection.
        
        """
        self.dirKeys   = ['w', 'a', 's', 'd']
        self.paramKeys = ['t','g','y','h','u','j','i','k', 'r', 'p']
        self.pidKeys = ['z','x','v','b','n','m']
        self.allKeys = self.dirKeys + self.paramKeys + self.pidKeys
        self.rcBrain   =  RcBrainThread()   
        # rospy.init_node('EXAMPLEnode', anonymous=False)     
        self.publisher = rospy.Publisher('/automobile/command', String, queue_size=1)

        #kewboard listener thread, non-blocking
        self.keyboardListenerThread = keyboard.Listener(on_press = self.keyPress, on_release = self.keyRelease)
        self.keyboardListenerThread.start()

    # ===================================== RUN ==========================================
    def run(self):
        """Apply initializing methods and start the threads. 
        """
        with keyboard.Listener(on_press = self.keyPress, on_release = self.keyRelease) as listener: 
            listener.join()
            cv.waitKey(1)
	
    # ===================================== KEY PRESS ====================================
    def keyPress(self,key):
        """Processing the key pressing 
        Parameters
        ----------
        key : pynput.keyboard.Key
            The key pressed
        """                                     
        try:
            if key.char == 'q':
                print('Exiting...')
                cv.destroyAllWindows()
                nod.keyboardListenerThread.stop()   
                raise KeyboardInterrupt
            if key.char == 'r':
                os.system('rosservice call /gazebo/reset_simulation')                             
            if key.char in self.allKeys:
                keyMsg = 'p.' + str(key.char)

                self._send_command(keyMsg)
    
        except: pass
        
    # ===================================== KEY RELEASE ==================================
    def keyRelease(self, key):
        """Processing the key realeasing.
        Parameters
        ----------
        key : pynput.keyboard.Key
            The key realeased. 
        """ 
        if key == keyboard.Key.esc:                        #exit key      
            self.publisher.publish('{"action":"3","steerAngle":0.0}')   
            return False
        try:                                               
            if key.char in self.allKeys:
                keyMsg = 'r.'+str(key.char)

                self._send_command(keyMsg)
    
        except: pass                                                              
                 
    # ===================================== SEND COMMAND =================================
    def _send_command(self, key):
        """Transmite the command to the remotecontrol receiver. 
        Parameters
        ----------
        inP : Pipe
            Input pipe. 
        """
        command = self.rcBrain.getMessage(key)
        if command is not None:
            command = json.dumps(command)
            self.publisher.publish(command)  

IMG_SIZE = 1024
FRAME_SIZE = 640

CAM_FPS = 15.0

class CarVisualizer(): 
    def __init__(self, run_visualization=True) -> None:
        self.use_ros_img = run_visualization
        #image
        self.img = np.zeros((IMG_SIZE, 2*IMG_SIZE, 3), np.uint8)
        #map
        # map = cv.imread('2021_VerySmall.png')
        self.map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')
        # self.map = cv.resize(self.map, (IMG_SIZE, IMG_SIZE)) #run from inside Simulator
        self.const_verysmall = 3541/15.0 #* (IMG_SIZE/3541.0) 

        rospy.init_node('manual_controller_visualizer', anonymous=False)
        rospy.sleep(.1)  # wait for publisher to register to roscore
        
        imu_topic =  "/automobile/IMU"
        self.sub_imu = rospy.Subscriber(imu_topic, IMU, self.imu_callback)
        self.x_true = 0.0       
        self.y_true = 0.0 
        self.yaw = 0.0
        self.last_frame_time = time()

        if self.use_ros_img:
            self.sub_cam_top = rospy.Subscriber("/automobile/image_top", Image, self.camera_top_callback)
            self.cv_image_top = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
            self.bridge = CvBridge()


    def imu_callback(self, data):
        """Receive and store rotation from IMU 

        :param data: a geometry_msgs Twist message
        :type data: object
        """        
        curr_time = time()
        self.x_true = float(data.posx)
        self.y_true = float(data.posy)
        self.yaw = float(data.yaw)
        if curr_time - self.last_frame_time > 1/(2*CAM_FPS):
            map_img = self.draw_car()
            map_img = cv.resize(map_img, (IMG_SIZE, IMG_SIZE))
            self.img[:, IMG_SIZE:] = map_img
            self.last_frame_time = curr_time

    def camera_top_callback(self, data):
        """Receive and store camera frame
        :param data: sensor_msg array containing the image from the camera
        :type data: object
        """        
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        img = cv.resize(img, (IMG_SIZE, IMG_SIZE))
        self.img[:, :IMG_SIZE] = img


    def draw_car(self):
        x = self.x_true
        y = self.y_true
        angle = self.yaw
        car_length = 0.4 #m
        car_width = 0.2 #m
        #match angle with map frame of reference
        angle = self.yaw2world(angle)
        #find 4 corners not rotated
        corners = np.array([[-car_width/2, car_length/2],
                            [car_width/2, car_length/2],
                            [car_width/2, -car_length/2],
                            [-car_width/2, -car_length/2]])
        #rotate corners
        rot_matrix = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        corners = np.matmul(rot_matrix, corners.T)
        #add car position
        corners = corners.T + np.array([x,y])
        img = self.map.copy()
        #draw body
        cv.fillPoly(img, [self.m2pix(corners)], (0,255,0))
        cv.polylines(img, [self.m2pix(corners)], True, (0,150,0), thickness=5, lineType=cv.LINE_AA) 
        return img
    
    def yaw2world(self,angle):
        return -(angle + np.pi/2)
    def m2pix(self, m):
        return np.int32(m*self.const_verysmall)
    def pix2m(self,pix):
        return 1.0*pix/self.const_verysmall
    def yaw2world(self,angle):
        return -(angle + np.pi/2)
    def world2yaw(self,angle):
        return -angle -np.pi/2

if __name__ == '__main__':
    #get arguments
    args = sys.argv
    view_only = False
    control_only = False
    if len(args) > 1:
        if args[1] == '-view-only':
            print('View only mode, [esc] will not stop the program,\nu can only stop the program with [ctrl]+[c]')
            view_only = True
            control_only = False
        elif args[1] == '-control-only':
            print('Control only mode.')
            control_only = True
            view_only = False
        else:
            view_only = False
            control_only = False
            print('Invalid argument, the only valid arguments are "-view-only"')
    run_control = not view_only
    run_visualization = not control_only
    
    try:
        cv.namedWindow('Car visualization', cv.WINDOW_NORMAL)
        cv.resizeWindow('Car visualization', FRAME_SIZE*2, FRAME_SIZE)
        cam = CarVisualizer(run_visualization)
        if run_control:
            nod = RemoteControlTransmitterProcess()

        while not rospy.is_shutdown():
            cv.imshow("Car visualization", cam.img) 
            key = cv.waitKey(1)
            if key == 27 and run_control:
                cv.destroyAllWindows()
                nod.keyboardListenerThread.stop()
                break
            sleep(1/CAM_FPS)

    except rospy.ROSInterruptException:
        pass