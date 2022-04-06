#!/usr/bin/python3

import numpy as np
import cv2 as cv
import os
from time import time, sleep

# from helper_functions import *

        # self.routines = {
        #     'follow_lane':              (True, self.follow_lane),         
        #     'slow_down':                (False, self.slow_down),     
        #     'accelerate':               (False, self.accelerate),         
        #     'control_for_signs':        (False, self.control_for_signs),         
        #     'control_for_semaphore':    (False, self.control_for_semaphore), 
        #     'control_for_pedestrians':  (False, self.control_for_pedestrians), 
        #     'control_for_vehicles':     (False, self.control_for_roadblocks),   
        #     'control_for_roadblocks':   (False, self.control_for_roadblocks),
        # }

STARTING_STATE = { 
    # lane following, between intersections or roundabouts
    'lane_following':           (True, ['follow_lane', 'control_for_pedestrians']),
    # intersection navigation, further divided into the possible directions (left, right, straight)
    'approaching_stop_line':    False,  
    'intersection_navigation':  False,
    'turning_right':            False,
    'turning_left':             False,
    'going_straight':           False,
    #roundabout, for roundabouts the car will track the local path
    'roundabout_navigation':    False,
    #waiting states
    'waiting_for_pedestrian':   False,
    'waiting_for_green':        True,
    'waiting_at_stopline':      False,
    'waiting_for_rerouting':    False,
    #overtaking manouver
    'overtaking_static_car':    False,
    'overtaking_moving_car':    False,
    'tailing_car':              False,
    'avoiding_obstacle':        False,
}

CONDITIONS = {
    'can_switch_right':         False,  # if true, the car can switch to the right lane, for obstacle avoidance
    'can_switch_left':          False,  # if true, the car can switch to the left lane, for overtaking or obstacle avoidance
    'highway':                  False,  # if true, the car is in a highway, the speed should be higher on highway, 
    'trust_gps':                True,   # if true the car will trust the gps, for example in expecting a sign or traffic light
                                        # can be set as false if there is a lot of package loss or the car has not received signal in a while
    'car_on_path':              True,   # if true, the car is on the path, if the gps is trusted and the position is too far from the path it will be set to false
}

class Brain:

    def __init__(self, car, starting_state=STARTING_STATE, starting_conditions=CONDITIONS):
        self.car = car
        self.state = starting_state
        self.conditions = starting_conditions
        #routines
        self.routines = {
            'follow_lane':              (True, self.follow_lane),         
            'slow_down':                (False, self.slow_down),     
            'accelerate':               (False, self.accelerate),         
            'control_for_signs':        (False, self.control_for_signs),         
            'control_for_semaphore':    (False, self.control_for_semaphore), 
            'control_for_pedestrians':  (False, self.control_for_pedestrians), 
            'control_for_vehicles':     (False, self.control_for_roadblocks),   
            'control_for_roadblocks':   (False, self.control_for_roadblocks),
        }

        self.last_run_call = time()


    def run(self):

        pass

    def activate_routines(self, routines_to_activate):
        """
        routines_to_activate are a list of strings (routines)
        """
        #deatcivate all routines
        for k in self.routines.keys():
            self.routines[k][0] = False
        #activate required routines
        for k in routines_to_activate:
            self.routines[k][0] = True
    
    # ROUTINES

    def follow_lane(self):
        print('Routine: follow_lane')
        pass

    def slow_down(self):
        print('Routine: slow_down')
        pass

    def accelerate(self):
        print('Routine: accelerate')
        pass

    def control_for_signs(self):
        print('Routine: control_for_signs')
        pass

    def control_for_semaphore(self):
        print('Routine: control_for_semaphore')
        pass

    def control_for_pedestrians(self):
        print('Routine: control_for_pedestrians')
        pass

    def control_for_vehicles(self):
        print('Routine: control_for_vehicles')
        pass

    def control_for_roadblocks(self):
        print('Routine: control_for_roadblocks')
        pass

    def check_logic(self):
        print('check_logic')
        pass
    










#test
if __name__ == '__main__':
    brain = Brain()
    while True:
        brain.run()