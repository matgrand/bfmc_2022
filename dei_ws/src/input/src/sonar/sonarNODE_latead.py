#!/usr/bin/env python3

from turtle import left
import RPi.GPIO as gpio
import time
import sys
#import signal
import rospy
from std_msgs.msg import Float32


class sonarNODE():
    def __init__(self):
        rospy.init_node('sonarNODE', anonymous=True)
        self.sonars_n = 4
        self.publishers = [rospy.Publisher('/automobile/sonar/ahead/center',Float32, queue_size=1),
                           rospy.Publisher('/automobile/sonar/ahead/left',Float32, queue_size=1),
                           rospy.Publisher('/automobile/sonar/ahead/right',Float32, queue_size=1),
                           rospy.Publisher('/automobile/sonar/lateral',Float32, queue_size=1)]

        # self.r = rospy.Rate(15)
        self.sampling_time = 0.06#1/20.0
        self.max_train_pulse_time = 0.01
        self.max_fly_time = 0.01*2

    def run(self):
        rospy.loginfo("starting sonarNODE")
        self._initSONAR()
        self._getting()
    
    def _initSONAR(self):
        gpio.setmode(gpio.BCM)
        self.trig = 23
        self.trig_lateral = 27

        self.echos = [24, 25, 8, 22]

        gpio.setup(self.trig, gpio.OUT)
        gpio.setup(self.trig_lateral, gpio.OUT)
        for echo in self.echos:
            gpio.setup(echo, gpio.IN)

        print("SONAR Name: HC-SR04")

    def _getting(self):
        gpio.output(self.trig, False)
        gpio.output(self.trig_lateral, False)
        time.sleep(0.5)

        while not rospy.is_shutdown() :
            # Send the impulse
            gpio.output(self.trig, True)
            gpio.output(self.trig_lateral, True)
            time.sleep(0.00001)# impulse duration to 10us
            gpio.output(self.trig, False)
            gpio.output(self.trig_lateral, False)

            sonars = ["center", "left", "right", "lateral"]
            echo_flags = [False] * self.sonars_n
            done_flags = [False] * self.sonars_n
            distances = [3.0] * self.sonars_n

            # wait for the comeback impulse
            start_time = time.time()       
            curr_time = start_time
            start_sonar_times = [start_time] * self.sonars_n
            while curr_time - start_time < self.max_fly_time and not all(done_flags):
                curr_time = time.time()
                for i in range(self.sonars_n):
                    if gpio.input(self.echos[i]) == 1 and not echo_flags[i]:
                        # print(f'{sonars[i]} is 1')
                        echo_flags[i] = True
                        start_sonar_times[i] = curr_time
                for i in range(self.sonars_n):
                    if gpio.input(self.echos[i]) == 0 and echo_flags[i] and not done_flags[i]:
                        pulse_duration = curr_time - start_sonar_times[i]
                        distances[i] = pulse_duration * 343.0 / 2
                        done_flags[i] = True
            
            for i in range(self.sonars_n):
                if echo_flags[i]:
                    self.publishers[i].publish(distances[i])
                else:
                    self.publishers[i].publish(-2)

            time.sleep(self.sampling_time)

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
        
        
