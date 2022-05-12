#!/usr/bin/env python3

import RPi.GPIO as gpio
import time
import sys
#import signal
import rospy
from std_msgs.msg import Float32


class sonarNODE():
    def __init__(self):
        rospy.init_node('sonar_lateal_NODE', anonymous=True)
        self.distance_publisher = rospy.Publisher('/automobile/sonar/lateral',Float32, queue_size=1)
        # self.r = rospy.Rate(15)
        self.sampling_time = 1/60.0
        self.max_train_pulse_time = 0.01
        self.max_fly_time = 0.01*2

    def run(self):
        rospy.loginfo("starting sonarNODE")
        self._initSONAR()
        self._getting()
    
    def _initSONAR(self):
        gpio.setmode(gpio.BCM)
        self.trig = 27
        self.echo = 22
        gpio.setup(self.trig, gpio.OUT)
        gpio.setup(self.echo, gpio.IN)
        print("SONAR Name: HC-SR04")

    def _getting(self):
        gpio.output(self.trig, False)
        time.sleep(0.5)

        while not rospy.is_shutdown() :
            # Send 10 us impulse
            gpio.output(self.trig, True)
            time.sleep(0.00001)# [us]
            gpio.output(self.trig, False)
            error = False
            # once the 10 us pulse is over the device sends out 8 cycle burst of ultrasound at 40 kHz 
            # and raises its echo PIN

            # ultrasound signal starting time            
            start_time = time.time()             
            pulse_start = None
            pulse_end = None

            # wait until echo is HIGH  
            while gpio.input(self.echo) == gpio.LOW:    
                if (time.time() - start_time) > self.max_train_pulse_time:
                    print('Timeout: Echo never HIGH')
                    distance = -1.0
                    error = True
                    break
                else:
                    continue
                    
            if error:
                pulse_start = None
            else:
                # register the instant in which the echo signal is set HIGH 
                pulse_start = time.time()

            while gpio.input(self.echo) == gpio.HIGH:
                if (time.time()-pulse_start) > self.max_fly_time:
                    print('Timeout: Echo never LOW')
                    break
                else:
                    continue
            # register the instant in which the echo signal is set LOW     
            pulse_end = time.time()
                
            if (pulse_end is not None) and (pulse_start is not None):
                pulse_duration = pulse_end - pulse_start
                distance = pulse_duration * 343.0 / 2

            self.dist_sendor(distance)
            time.sleep(self.sampling_time)
        
    def dist_sendor(self,dist):
        data = Float32()
        data.data=dist
        self.distance_publisher.publish(data)


if __name__ == "__main__":
    try:
        sonarNod = sonarNODE()
        sonarNod.run()
    except (KeyboardInterrupt, SystemExit):
        print("Exception from KeyboardInterrupt or SystemExit")
        gpio.cleanup()
        sys.exit(0)
    except Exception as e:
        print('Finally *************++')
        print(e)
        gpio.cleanup()
        sys.exit(0)
        
        
