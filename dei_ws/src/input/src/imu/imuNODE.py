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

import sys
sys.path.append('.')
from adafruit_extended_bus import ExtendedI2C as I2C
import adafruit_bno055
from adafruit_bno055 import COMPASS_MODE
from adafruit_bno055 import NDOF_FMC_OFF_MODE
from adafruit_bno055 import NDOF_MODE

import os.path
import time
import json

import rospy

from utils.msg import IMU

PUB_FREQ = 100.0

class imuNODE():
    def __init__(self): 
        rospy.init_node('imuNODE', anonymous=False)
        # BNO publisher object
        self.BNO_publisher = rospy.Publisher("/automobile/imu", IMU, queue_size=1)
        
    #================================ RUN ========================================
    def run(self):
        rospy.loginfo("starting imuNODE")
        self._initIMU()
        self._getting()
    
    #================================ INIT IMU ========================================
    def _initIMU(self):
        self.i2c = I2C(1)  # Device is /dev/i2c-1        
        self.imu = adafruit_bno055.BNO055_I2C(self.i2c, address=0x29)

        self.imu.offsets_accelerometer = (-36, -56, -16)
        self.imu.offsets_gyroscope = (-1, -2, 1)
        self.imu.offsets_magnetometer = (128, 0, 0)
        self.imu.radius_accelerometer = 1000
        self.imu.radius_magnetometer = 480

        self.imu.mode = NDOF_MODE# COMPASS MODE
        def myhook():
            data = {}
            data['offsets_accelerometer'] = self.imu.offsets_accelerometer
            data['offsets_gyroscope'] = self.imu.offsets_gyroscope
            data['offsets_magnetometer'] = self.imu.offsets_magnetometer
            data['radius_accelerometer'] = self.imu.radius_accelerometer
            data['radius_magnetometer'] = self.imu.radius_magnetometer

            data = json.dump(data, open("/home/pi/dei_ws/src/input/src/imu/imu_calibration_status.json","w"), indent=1)

            print("shutdown time!")
        rospy.on_shutdown(myhook)

        print("IMU Name: BNO055")

        #print(f'********* this is the imu interval: {self.poll_interval} *********************')

    #================================ GETTING ========================================
    def _getting(self):
        while not rospy.is_shutdown():
            # if self.imu.calibration_status[-1] < 3:
            #     print(f'Magnetometer not calibrated with status {self.imu.calibration_status}, keep moving the car ...')
            print(f'IMU not calibrated with status {self.imu.calibration_status}, keep moving the car ...')

            
            euler_angles = self.imu.euler
            gyro = self.imu.gyro
            accel = self.imu.acceleration    
            
            imudata = IMU()
            publish = True

            if euler_angles[2] is not None:
                imudata.roll   =  - float(euler_angles[2]) + 180
            else:
                publish = False
            if euler_angles[1] is not None:
                imudata.pitch  =  - float(euler_angles[1]) + 180
            else:
                publish = False
            if euler_angles[0] is not None:
                imudata.yaw    =  - float(euler_angles[0]) + 180
            else:
                publish = False
            if accel[0] is not None:
                imudata.accelx =  float(accel[0])
            else:
                publish = False
            if accel[1] is not None:
                imudata.accely =  float(accel[1])
            else:
                publish = False
            if accel[2] is not None:
                imudata.accelz =  float(accel[2])
            else:
                publish = False
            if gyro[0] is not None:
                imudata.gyrox  =  float(gyro[0])
            else:
                publish = False
            if gyro[1] is not None:
                imudata.gyroy  =  float(gyro[1])
            else:
                publish = False
            if gyro[2] is not None:
                imudata.gyroz  =  float(gyro[2])
            else:
                publish = False
        
            if publish:
                self.BNO_publisher.publish(imudata)
        
            time.sleep(1/PUB_FREQ)
        
if __name__ == "__main__":
    imuNod = imuNODE()
    imuNod.run()
