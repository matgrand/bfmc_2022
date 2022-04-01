#!/usr/bin/env python3

import RPi.GPIO as gpio
import time
import sys
#import signal
import rospy
from std_msgs.msg import Float32

# def signal_handler(signal, frame): # ctrl + c -> exit program
#         print('You pressed Ctrl+C!')
#         sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)


class sonarNODE():
    def __init__(self):
        rospy.init_node('sonarNODE', anonymous=True)
        self.distance_publisher = rospy.Publisher('/automobile/sonar',Float32, queue_size=1)
        # self.r = rospy.Rate(15)
        self.sampling_time = 1/20.0
        self.max_train_pulse_time = 0.01
        self.max_fly_time = 0.01*2

    def run(self):
        rospy.loginfo("starting sonarNODE")
        self._initSONAR()
        self._getting()
    
    def _initSONAR(self):
        gpio.setmode(gpio.BCM)
        self.trig = 23
        self.echo = 24
        gpio.setup(self.trig, gpio.OUT)
        gpio.setup(self.echo, gpio.IN)

    def _getting(self):
        gpio.output(self.trig, False)
        time.sleep(0.5)

        while not rospy.is_shutdown() :
            # Send the impulse
            gpio.output(self.trig, True)
            time.sleep(0.00001)# impulse duration to 10us
            gpio.output(self.trig, False)

            distance = -1.0
            pulse_start = None
            pulse_end = None
            
            # wait for the comeback impulse
            start_time = time.time()            
            while gpio.input(self.echo) == 0:                   
                pulse_start = time.time()
                if (time.time()-start_time) > self.max_train_pulse_time:
                    # print('Error train pulse not finished')
                    distance = -1.0
                    break
            while gpio.input(self.echo) == 1:
                pulse_end = time.time()
                if (time.time()-start_time) > self.max_fly_time:
                    break
            
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
        
        
