#!/usr/bin/python3

import numpy as np
import cv2 as cv
import os
from time import time, sleep

from automobile_data_interface import Automobile_Data
from helper_functions import *
from PathPlanning4 import PathPlanning
from controller3 import Controller
from detection import Detection

# from helper_functions import *

IN_SIMULATOR = True
SHOW_IMGS = True

DESIRED_SPEED = 0.5

START_NODE = 86
END_NODE = 273#169#116



#STATES
START_STATE = 'start_state'
END_STATE = 'end_state'
LANE_FOLLOWING = 'lane_following'
APPROACHING_STOP_LINE = 'approaching_stop_line'
INTERSECTION_NAVIGATION = 'intersection_navigation'
TURNING_RIGHT = 'turning_right'
TURNING_LEFT = 'turning_left'
GOING_STRAIGHT = 'going_straight'
ROUNDABOUT_NAVIGATION = 'roundabout_navigation'
WAITING_FOR_PEDESTRIAN = 'waiting_for_pedestrian'
WAITING_FOR_GREEN = 'waiting_for_green'
WAITING_AT_STOPLINE = 'waiting_at_stopline'
WAITING_FOR_REROUTING = 'waiting_for_rerouting'
OVERTAKING_STATIC_CAR = 'overtaking_static_car'
OVERTAKING_MOVING_CAR = 'overtaking_moving_car'
TAILING_CAR = 'tailing_car'
AVOIDING_OBSTACLE = 'avoiding_obstacle'

class State():
    def __init__(self, name, method, activated=False):
        self.name = name
        self.method = method
        self.activated = activated
        self.start_time = None
        self.start_position = None
        self.start_distance = None


# ROUTINES
FOLLOW_LANE = 'follow_lane'
DETECT_STOP_LINE = 'detect_stop_line'
SLOW_DOWN = 'slow_down'
ACCELERATE = 'accelerate'
CONTROL_FOR_SIGNS = 'control_for_signs'
CONTROL_FOR_SEMAPHORE = 'control_for_semaphore'
CONTROL_FOR_PEDESTRIANS = 'control_for_pedestrians'
CONTROL_FOR_VEHICLES = 'control_for_vehicles'
CONTROL_FOR_ROADBLOCKS = 'control_for_roadblocks'

# EVENT TYPES
INTERSECTION_STOP_EVENT = 'intersection_stop_event'
INTERSECTION_TRAFFIC_LIGHT_EVENT = 'intersection_traffic_light_event'
INTERSECTION_PRIORITY_EVENT = 'intersection_priority_event'
JUNCTION_EVENT = 'junction_event'
ROUNDABOUT_EVENT = 'roundabout_event'
CROSSWALK_EVENT = 'crosswalk_event'
PARKING_EVENT = 'parking_event'

EVENT_TYPES = [INTERSECTION_STOP_EVENT, INTERSECTION_TRAFFIC_LIGHT_EVENT, INTERSECTION_PRIORITY_EVENT,
                JUNCTION_EVENT, ROUNDABOUT_EVENT, CROSSWALK_EVENT, PARKING_EVENT]

class Event:
    def __init__(self, name, dist, point, path_ahead):
        self.name = name                # name/type of the event
        self.dist = dist                # distance of event from start of path
        self.point = point              # [x,y] position on the map of the event
        self.path_ahead = path_ahead    # sequence of points after the event, only for intersections or roundabouts


CONDITIONS = {
    'can_switch_right':         False,  # if true, the car can switch to the right lane, for obstacle avoidance
    'can_switch_left':          False,  # if true, the car can switch to the left lane, for overtaking or obstacle avoidance
    'highway':                  False,  # if true, the car is in a highway, the speed should be higher on highway, 
    'trust_gps':                True,   # if true the car will trust the gps, for example in expecting a sign or traffic light
                                        # can be set as false if there is a lot of package loss or the car has not received signal in a while
    'car_on_path':              True,   # if true, the car is on the path, if the gps is trusted and the position is too far from the path it will be set to false
}

# PARAMTERS
STOP_LINE_APPROACH_DISTANCE = 0.3
STOP_LINE_STOP_DISTANCE = 0.1
assert STOP_LINE_STOP_DISTANCE < STOP_LINE_APPROACH_DISTANCE
STOP_WAIT_TIME = 3.0

class Brain:

    def __init__(self, car, controller, detection, path_planner, desired_speed=DESIRED_SPEED):
        print("Initialize brain")
        self.car = Automobile_Data() #not needed, just to import he methods in visual studio
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
        # events are an ordered list of tuples: (type , distance from start, x y position)
        self.events = []
        self.desired_speed = desired_speed
        #current and previous states
        self.curr_state = None
        self.prev_state = None
        #previous and next event
        self.prev_event = None
        self.next_event = None
        self.event_idx = 0

        self.last_switch_state_time = time()
        self.last_switch_state_position = self.car.encoder_distance
        self.just_switched_state = False

        self.conditions = CONDITIONS

        # INITIALIZE STATE
        self.state = { 
            START_STATE:              [False, self.start_state],
            END_STATE:                [False, self.end_state],
            # lane following, between intersections or roundabouts
            LANE_FOLLOWING:           [False, self.lane_following],
            # intersection navigation, further divided into the possible directions [left, right, straight]
            APPROACHING_STOP_LINE:    [False,self.approaching_stop_line], 
            INTERSECTION_NAVIGATION:  [False,self.intersection_navigation],
            TURNING_RIGHT:            [False,self.turning_right],
            TURNING_LEFT:             [False,self.turning_left],
            GOING_STRAIGHT:           [False,self.going_straight],
            #roundabout, for roundabouts the car will track the local path
            ROUNDABOUT_NAVIGATION:    [False,self.roundabout_navigation],
            #waiting states
            WAITING_FOR_PEDESTRIAN:   [False,self.waiting_for_pedestrian],
            WAITING_FOR_GREEN:        [False,self.waiting_for_green],
            WAITING_AT_STOPLINE:      [False,self.waiting_at_stopline],
            WAITING_FOR_REROUTING:    [False,self.waiting_for_rerouting],
            #overtaking manouver
            OVERTAKING_STATIC_CAR:    [False,self.overtaking_static_car],
            OVERTAKING_MOVING_CAR:    [False,self.overtaking_moving_car],
            TAILING_CAR:              [False,self.tailing_car],
            AVOIDING_OBSTACLE:        [False,self.avoiding_obstacle],
        }

        # INITIALIZE ROUTINES
        self.routines = {
            FOLLOW_LANE:              [False, self.follow_lane],      
            DETECT_STOP_LINE:         [False, self.detect_stop_line],   
            SLOW_DOWN:                [False, self.slow_down],     
            ACCELERATE:               [False, self.accelerate],         
            CONTROL_FOR_SIGNS:        [False, self.control_for_signs],         
            CONTROL_FOR_SEMAPHORE:    [False, self.control_for_semaphore], 
            CONTROL_FOR_PEDESTRIANS:  [False, self.control_for_pedestrians], 
            CONTROL_FOR_VEHICLES:     [False, self.control_for_roadblocks],   
            CONTROL_FOR_ROADBLOCKS:   [False, self.control_for_roadblocks],
        }

        self.last_run_call = time()
        print('Brain initialized')
        self.switch_to_state(START_STATE)

    #=============== STATES ===============#
    def start_state(self):
        print('State: start_state')
        #exception: states should not perform actions
        self.car.stop()
        #calculate path
        print('Calculating path...')
        self.path_planner.compute_shortest_path(START_NODE, END_NODE)
        
        #initialize the list of events on the path
        events = self.path_planner.augment_path()
        self.events = self.create_sequence_of_events(events)
        # self.events = events
        self.next_event = self.events[0]
        
        self.path_planner.draw_path()
        print('Starting in 1 second...')
        sleep(1)
        self.switch_to_state(LANE_FOLLOWING)
        self.car.drive_speed(self.desired_speed)

    def end_state(self):
        print('State: end_state')
        self.car.stop()
        sleep(1.5)
        exit()

    def lane_following(self): # LANE FOLLOWING ##############################
        print('State: lane_following')
        print('Next event: ', self.next_event)
        self.activate_routines([FOLLOW_LANE, DETECT_STOP_LINE])

        #check if we are approaching a stop_line
        if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE:
            self.switch_to_state(APPROACHING_STOP_LINE)
            return

    def approaching_stop_line(self):
        print('State: approaching_stop_line')
        print('Next event: ', self.next_event)
        self.activate_routines([FOLLOW_LANE, SLOW_DOWN, DETECT_STOP_LINE])
        dist = self.detect.est_dist_to_stop_line
        #check if we are here by mistake
        if dist > STOP_LINE_APPROACH_DISTANCE:
            self.switch_to_state(LANE_FOLLOWING)
            return
        if dist < STOP_LINE_STOP_DISTANCE:
            if self.next_event[0] == INTERSECTION_STOP_EVENT:
                self.switch_to_state(WAITING_AT_STOPLINE)
            elif self.next_event[0] == INTERSECTION_TRAFFIC_LIGHT_EVENT:
                self.switch_to_state(WAITING_FOR_GREEN)
            elif self.next_event[0] == INTERSECTION_PRIORITY_EVENT:
                self.switch_to_state(GOING_STRAIGHT)
            elif self.next_event[0] == JUNCTION_EVENT:
                self.switch_to_state(TURNING_RIGHT)
            elif self.next_event[0] == ROUNDABOUT_EVENT:
                self.switch_to_state(ROUNDABOUT_NAVIGATION)
            elif self.next_event[0] == CROSSWALK_EVENT:
                self.switch_to_state(WAITING_FOR_PEDESTRIAN)
            elif self.next_event[0] == PARKING_EVENT:
                print('WARNING: UNEXPECTED STOP LINE FOUND WITH PARKING AS NEXT EVENT')
                print('Switching to end state')
                self.car.stop()
                sleep(10.0)
                self.switch_to_state(END_STATE)

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
        #temporary fix
        self.switch_to_state(WAITING_AT_STOPLINE)

    def waiting_at_stopline(self):
        print('State: waiting_at_stopline')
        self.activate_routines([]) #no routines
        self.car.stop()
        if (time() - self.last_switch_state_time) > STOP_WAIT_TIME:
            self.switch_to_state(LANE_FOLLOWING)

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
        e2, e3, point_ahead = self.detect.detect_lane(self.car.frame, SHOW_IMGS)
        _, angle_ref = self.controller.get_control(e2, e3, 0, self.desired_speed)
        angle_ref = np.rad2deg(angle_ref)
        print(f'angle_ref: {angle_ref}')
        self.car.drive_angle(angle_ref)

    def detect_stop_line(self):
        print('Routine: detect_stop_line')
        #update the variable self.detect.est_dist_to_stop_line
        self.detect.detect_stop_line(self.car.frame, SHOW_IMGS)

    def slow_down(self):
        print('Routine: slow_down')
        self.car.drive_speed(self.desired_speed*0.2)

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
    

#===================== STATE MACHINE MANAGEMENT =====================#
    def run(self):
        print('==============================================')
        self.run_current_state()
        print('==============================================')
        print()
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
    
    def switch_to_state(self, to_state, interrupt=False):
        """
        to_state is the string of the desired state to switch to
        ex: 'lane_following'
        """
        self.prev_state = self.curr_state
        self.curr_state = to_state
        self.state[to_state][0] = True
        for k in self.state.keys():
            if k != to_state:
                self.state[k][0] = False
        if not interrupt:
            self.last_switch_state_time = time()
            self.last_switch_state_position = self.car.encoder_distance
        self.just_switched_state = True

    def switch_to_prev_state(self):
        self.switch_to_state(self.prev_state)
    
    def go_to_next_event(self):
        """
        Switches to the next event on the path
        """
        if self.event_idx == len(self.events) - 1:
            #no more events, go to end_state
            self.switch_to_state(END_STATE)
        else:
            self.event_idx += 1
            self.prev_event = self.next_event
            self.next_event = self.events[self.event_idx]

    def create_sequence_of_events(events):
        """
        events is a list of strings (events)
        ex: ['lane_following', 'control_for_signs']
        """
        to_ret = []
        for e in events:
            name = e[0]
            dist = e[1]
            point = e[2]
            path_ahead = e[3]
            event = Event(name, dist, point, path_ahead)
            to_ret.append(event)
        return to_ret

