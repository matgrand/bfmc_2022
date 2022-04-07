#!/usr/bin/python3

import numpy as np
import cv2 as cv
import os
from time import time, sleep

from automobile_data import Automobile_Data
from helper_functions import *
from PathPlanning4 import PathPlanning
from controller3 import Controller
from detection import Detection

# from helper_functions import *
IN_SIMULATOR = True

DESIRED_SPEED = 0.5

START_NODE = 86
END_NODE = 116


CONDITIONS = {
    'can_switch_right':         False,  # if true, the car can switch to the right lane, for obstacle avoidance
    'can_switch_left':          False,  # if true, the car can switch to the left lane, for overtaking or obstacle avoidance
    'highway':                  False,  # if true, the car is in a highway, the speed should be higher on highway, 
    'trust_gps':                True,   # if true the car will trust the gps, for example in expecting a sign or traffic light
                                        # can be set as false if there is a lot of package loss or the car has not received signal in a while
    'car_on_path':              True,   # if true, the car is on the path, if the gps is trusted and the position is too far from the path it will be set to false
}

class Brain:

    def __init__(self, car, controller, detection, path_planner, desired_speed=DESIRED_SPEED):
        print("Initialize brain")
        self.car = Automobile_Data(simulator=True) #not needed, just to import he methods in visual studio
        self.car = car
        assert isinstance(self.car, Automobile_Data)
        self.controller = Controller() #again, not needed
        self.controller = controller
        assert isinstance(self.controller, Controller)
        self.detect = Detection() #again, not needed
        self.detect = detection
        assert isinstance(self.detect, Detection)
        self.path_planner = PathPlanning(None)
        self.path_planner = path_planner
        assert isinstance(self.path_planner, PathPlanning)
        
        #navigation instruction is a list of tuples:
        # ()
        self.navigation_instructions = []
        self.desired_speed = desired_speed

        self.conditions = CONDITIONS

        # INITIALIZE STATE
        self.state = { 
            'start_state':              [True, self.start_state],
            'end_state':                [False, self.end_state],
            # lane following, between intersections or roundabouts
            'lane_following':           [False, self.lane_following],
            # intersection navigation, further divided into the possible directions [left, right, straight]
            'approaching_stop_line':    [False,self.approaching_stop_line], 
            'intersection_navigation':  [False,self.intersection_navigation],
            'turning_right':            [False,self.turning_right],
            'turning_left':             [False,self.turning_left],
            'going_straight':           [False,self.going_straight],
            #roundabout, for roundabouts the car will track the local path
            'roundabout_navigation':    [False,self.roundabout_navigation],
            #waiting states
            'waiting_for_pedestrian':   [False,self.waiting_for_pedestrian],
            'waiting_for_green':        [False,self.waiting_for_green],
            'waiting_at_stopline':      [False,self.waiting_at_stopline],
            'waiting_for_rerouting':    [False,self.waiting_for_rerouting],
            #overtaking manouver
            'overtaking_static_car':    [False,self.overtaking_static_car],
            'overtaking_moving_car':    [False,self.overtaking_moving_car],
            'tailing_car':              [False,self.tailing_car],
            'avoiding_obstacle':        [False,self.avoiding_obstacle],
        }
        # INITIALIZE ROUTINES
        self.routines = {
            'follow_lane':              [True, self.follow_lane],         
            'slow_down':                [False, self.slow_down],     
            'accelerate':               [False, self.accelerate],         
            'control_for_signs':        [False, self.control_for_signs],         
            'control_for_semaphore':    [False, self.control_for_semaphore], 
            'control_for_pedestrians':  [True, self.control_for_pedestrians], 
            'control_for_vehicles':     [False, self.control_for_roadblocks],   
            'control_for_roadblocks':   [False, self.control_for_roadblocks],
        }

        self.last_run_call = time()
        print('Brain initialized')


    def run(self):
        self.run_current_state()
        self.run_routines()

    def run_current_state(self):
        for k in self.state.keys():
            if self.state[k][0]:
                self.state[k][1]()

    def run_routines(self):
        for k in self.routines.keys():
            if self.routines[k][0]:
                self.routines[k][1]()

    def activate_routines(self, routines_to_activate):
        """
        routines_to_activate are a list of strings (routines)
        ex: ['follow_lane', 'control_for_signs']
        """
        #deatcivate all routines
        for k in self.routines.keys():
            if k in routines_to_activate:
                self.routines[k][0] = True
            else:
                self.routines[k][0] = False
    
    def switch_to_state(self, to_state):
        """
        to_state is the string of the desired state to switch to
        ex: 'lane_following'
        """
        self.state[to_state][0] = True
        for k in self.state.keys():
            if k != to_state:
                self.state[k][0] = False
    
    #=============== STATES ===============#
    def start_state(self):
        print('State: start_state')
        #exception: states should not perform actions
        self.car.stop()
        #calculate path
        print('Calculating path...')
        self.path_planner.compute_shortest_path(START_NODE, END_NODE)
        self.path_planner.augment_path()
        self.path_planner.draw_path()
        print('Starting in 1 second...')
        sleep(1)
        self.switch_to_state('lane_following')
        self.car.drive_speed(self.desired_speed)

    def end_state(self):
        print('State: end_state')
        pass

    def lane_following(self):
        print('State: lane_following')
        self.activate_routines(['follow_lane'])

    def approaching_stop_line(self):
        print('State: approaching_stop_line')
        pass

    def intersection_navigation(self):
        print('State: intersection_navigation')
        pass

    def turning_right(self):
        print('State: turning_right')
        pass

    def turning_left(self):
        print('State: turning_left')
        pass

    def going_straight(self):
        print('State: going_straight')
        pass

    def roundabout_navigation(self):
        print('State: roundabout_navigation')
        pass

    def waiting_for_pedestrian(self):
        print('State: waiting_for_pedestrian')
        pass

    def waiting_for_green(self):
        print('State: waiting_for_green')
        pass

    def waiting_at_stopline(self):
        print('State: waiting_at_stopline')
        pass

    def waiting_for_rerouting(self):
        print('State: waiting_for_rerouting')
        pass

    def overtaking_static_car(self):
        print('Routine: overtaking_static_car')
        pass

    def overtaking_moving_car(self):
        print('Routine: overtaking_moving_car')
        pass

    def tailing_car(self):
        print('Routine: tailing_car')
        pass

    def avoiding_obstacle(self):
        print('Routine: avoiding_obstacle')
        pass



    
    #=============== ROUTINES ===============#

    def follow_lane(self):
        print('Routine: follow_lane')
        e2, e3, point_ahead = self.detect.detect_lane(self.car.cv_image)
        _, angle_ref = self.controller.get_control(e2, e3, 0, self.desired_speed)
        angle_ref = np.rad2deg(angle_ref)
        print(f'angle_ref: {angle_ref}')
        self.car.drive_angle(angle_ref)

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



    # STATE CHECKS
    def check_logic(self):
        print('check_logic')
        pass
    










# #test
# if __name__ == '__main__':
#     brain = Brain(car=None)
#     while True:
#         brain.run()
#         sleep(0.5)