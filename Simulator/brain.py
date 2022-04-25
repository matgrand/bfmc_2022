#!/usr/bin/python3
import numpy as np
import cv2 as cv
import os
from time import time, sleep
from copy import copy, deepcopy
from numpy.linalg import norm
from collections import deque

from automobile_data_interface import Automobile_Data
from helper_functions import *
from PathPlanning4 import PathPlanning
from controller3 import Controller
from detection import Detection

from helper_functions import *

SHOW_IMGS = True

START_NODE = 86
END_NODE = 285#285#236#169#116          #203 T-park #190 S-park 
CHECKPOINTS = [86,203,190,203,190,280,285,116,236,329,162,226]

#========================= STATES ==========================
START_STATE = 'start_state'
END_STATE = 'end_state'
DOING_NOTHING = 'doing_nothing'
LANE_FOLLOWING = 'lane_following'
APPROACHING_STOP_LINE = 'approaching_stop_line'
INTERSECTION_NAVIGATION = 'intersection_navigation'
TURNING_RIGHT = 'turning_right'
TURNING_LEFT = 'turning_left'
GOING_STRAIGHT = 'going_straight'
TRACKING_LOCAL_PATH = 'tracking_local_path'
ROUNDABOUT_NAVIGATION = 'roundabout_navigation'
WAITING_FOR_PEDESTRIAN = 'waiting_for_pedestrian'
WAITING_FOR_GREEN = 'waiting_for_green'
WAITING_AT_STOPLINE = 'waiting_at_stopline'
WAITING_FOR_REROUTING = 'waiting_for_rerouting'
OVERTAKING_STATIC_CAR = 'overtaking_static_car'
OVERTAKING_MOVING_CAR = 'overtaking_moving_car'
TAILING_CAR = 'tailing_car'
AVOIDING_ROADBLOCK = 'avoiding_roadblock'
PARKING = 'parking'

class State():
    def __init__(self, name=None, method=None, activated=False):
        self.name = name
        self.method = method
        self.active = activated
        self.start_time = None
        self.start_position = None
        self.start_distance = None
        self.just_switched = False
        self.interrupted = False
        #variables specific to state, can be freely assigned
        self.var1 = None
        self.var2 = None
        self.var3 = None
    def __str__(self):
        return self.name.upper() if self.name is not None else 'None'
    def run(self):
        self.method()

#======================== ROUTINES ==========================
FOLLOW_LANE = 'follow_lane'
DETECT_STOP_LINE = 'detect_stop_line'
SLOW_DOWN = 'slow_down'
ACCELERATE = 'accelerate'
CONTROL_FOR_SIGNS = 'control_for_signs'
CONTROL_FOR_SEMAPHORE = 'control_for_semaphore'
CONTROL_FOR_PEDESTRIANS = 'control_for_pedestrians'
CONTROL_FOR_VEHICLES = 'control_for_vehicles'
CONTROL_FOR_ROADBLOCKS = 'control_for_roadblocks'

class Routine():
    def __init__(self, name, method, activated=False):
        self.name = name
        self.method = method
        self.active = activated
        self.start_time = None
        self.start_position = None
        self.start_distance = None
    def __str__(self):
        return self.name
    def run(self):
        self.method()

#========================== EVENTS ==========================
INTERSECTION_STOP_EVENT = 'intersection_stop_event'
INTERSECTION_TRAFFIC_LIGHT_EVENT = 'intersection_traffic_light_event'
INTERSECTION_PRIORITY_EVENT = 'intersection_priority_event'
JUNCTION_EVENT = 'junction_event'
ROUNDABOUT_EVENT = 'roundabout_event'
CROSSWALK_EVENT = 'crosswalk_event'
PARKING_EVENT = 'parking_event'
END_EVENT = 'end_event'
HIGHWAY_EXIT_EVENT = 'highway_exit_event'

EVENT_TYPES = [INTERSECTION_STOP_EVENT, INTERSECTION_TRAFFIC_LIGHT_EVENT, INTERSECTION_PRIORITY_EVENT,
                JUNCTION_EVENT, ROUNDABOUT_EVENT, CROSSWALK_EVENT, PARKING_EVENT, HIGHWAY_EXIT_EVENT]

class Event:
    def __init__(self, name=None, dist=None, point=None, yaw_stopline=None, path_ahead=None, length_path_ahead=None, curvature=None):
        self.name = name                # name/type of the event
        self.dist = dist                # distance of event from start of path
        self.point = point              # [x,y] position on the map of the event
        self.yaw_stopline = yaw_stopline# yaw of the stop line at the event
        self.path_ahead = path_ahead    # sequence of points after the event, only for intersections or roundabouts
        self.length_path_ahead = length_path_ahead   # length of the path after the event, only for intersections or roundabouts
        self.curvature = curvature      # curvature of the path ahead of the event
    def __str__(self):
        return self.name.upper() if self.name is not None else 'None'

#======================== CONDITIONS ==========================
IN_RIGHT_LANE = 'in_right_lane'
IN_LEFT_LANE = 'in_left_lane'
IS_DOTTED_LINE = 'is_dotted_line'
HIGHWAY = 'highway'
TRUST_GPS = 'trust_gps'
CAR_ON_PATH = 'car_on_path'
CONDITIONS = {
    IN_RIGHT_LANE:            False,  # if true, the car can switch to the right lane, for obstacle avoidance
    IN_LEFT_LANE:             False,  # if true, the car can switch to the left lane, for overtaking or obstacle avoidance
    IS_DOTTED_LINE:           False,  # if true, the car is on a dotted line, for obstacle avoidance
    HIGHWAY:                  False,  # if true, the car is in a highway, the speed should be higher on highway, 
    TRUST_GPS:                True,   # if true the car will trust the gps, for example in expecting a sign or traffic light
                                        # can be set as false if there is a lot of package loss or the car has not received signal in a while
    CAR_ON_PATH:              True,   # if true, the car is on the path, if the gps is trusted and the position is too far from the path it will be set to false
}

#======================== ACHIEVEMENTS ========================
#consider adding all the tasks, may be too cumbersome
PARK_ACHIEVED = 'park_achieved'

ACHIEVEMENTS = {
    PARK_ACHIEVED: False
}

#==============================================================
#========================= PARAMTERS ==========================
#==============================================================
YAW_GLOBAL_OFFSET = 0.0 #global offset of the yaw angle between the real track and the simulator map
STOP_LINE_APPROACH_DISTANCE = 0.3
STOP_LINE_STOP_DISTANCE = 0.1
assert STOP_LINE_STOP_DISTANCE < STOP_LINE_APPROACH_DISTANCE
STOP_WAIT_TIME = 0.1 #3.0
OPEN_LOOP_PERCENTAGE_OF_PATH_AHEAD = 0.6 #0.6
STOP_LINE_DISTANCE_THRESHOLD = 0.2 #distance from previous stop_line from which is possible to start detecting a stop line again
POINT_AHEAD_DISTANCE_LOCAL_TRACKING = 0.3
USE_LOCAL_TRACKING_FOR_INTERSECTIONS = True

ALWAYS_TRUST_GPS = True  # if true the car will always trust the gps (bypass)
ALWAYS_DISTRUST_GPS = False #if true, the car will always distrust the gps (bypass)
assert not(ALWAYS_TRUST_GPS and ALWAYS_DISTRUST_GPS), 'ALWAYS_TRUST_GPS and ALWAYS_DISTRUST_GPS cannot be both True'

#PARKING
PARKING_DISTANCE_SLOW_DOWN_THRESHOLD = 1.0
PARKING_DISTANCE_STOP_THRESHOLD = 0.1
SUBPATH_LENGTH_FOR_PARKING = 300 # length in samples of the path to consider around the parking position, max
USE_GPS_FOR_PARKING = False
USE_SIGN_FOR_PARKING = True
assert USE_GPS_FOR_PARKING ^ USE_SIGN_FOR_PARKING, 'XOR-> use at least one and only one of the two'
PARK_MAX_SECONDS_W8_GPS = 10.0 #[s] max seconds to wait for gps to be available 
MAX_PARK_SEARCH_DIST = 2.0 #[m] max distance to search for parking
PARK_SEARCH_SPEED = 0.1 #[m/s] speed to search for parking
PARK_MANOUVER_SPEED = 0.25 #[m/s] speed to perform the parking manouver
DIST_SIGN_FIRST_T_SPOT = 0.8 #[m] distance from the sign to the first parking spot
DIST_T_SPOTS = 0.45 #[m] distance from the sign to the second parking spot
DIST_SIGN_FIRST_S_SPOT = 0.9 #[m] distance from the sign to the first parking spot
DIST_S_SPOTS = 0.7 #[m] distance from the sign to the second parking spot
FURTHER_DIST_S = 0.7 #[m] distance to proceed further in order to perform the s manouver
FURTHER_DIST_T = 0.7 #[m] distance to proceed further in order to perform the t manouver
T_ANGLE = 27.0 #[deg] angle to perform the t manouver
S_ANGLE = 27.0 #[deg] angle to perform the s manouver
DIST_2T = 0.8 #[m] distance to perform the 2nd part of t manouver
DIST_3T = 0.1 #[m] distance to perform the 3rd part of t manouver 
DIST_2S = 0.4 #[m] distance to perform the 2nd part of s manouver
DIST_4S = 0.05 #[m] distance to perform the 4th part of s manouver
STEER_ACTUATION_DELAY_PARK = 0.5 #[s] delay to perform the steering manouver

#==============================================================
#=========================== BRAIN ============================
#==============================================================
class Brain:
    def __init__(self, car, controller, detection, path_planner, checkpoints=None, desired_speed=0.3, debug=True):
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
        self.checkpoints = checkpoints if checkpoints is not None else CHECKPOINTS
        self.checkpoint_idx = 0
        self.desired_speed = desired_speed

        #current and previous states (class State)
        self.curr_state = State()
        self.prev_state = State()
        #previous and next event (class Event)
        self.prev_event = Event()
        self.next_event = Event()
        self.event_idx = 0

        #debug
        self.debug = debug
        if self.debug and SHOW_IMGS:
            cv.namedWindow('brain_debug', cv.WINDOW_NORMAL)
            self.debug_frame = None

        self.conditions = CONDITIONS
        self.achievements = ACHIEVEMENTS

        # INITIALIZE STATES
        self.states = { 
            START_STATE:              State(START_STATE, self.start_state),
            END_STATE:                State(END_STATE, self.end_state),
            DOING_NOTHING:            State(DOING_NOTHING, self.doing_nothing),
            # lane following, between intersections or roundabouts
            LANE_FOLLOWING:           State(LANE_FOLLOWING, self.lane_following),
            # intersection navigation, further divided into the possible directions [left, right, straight]
            APPROACHING_STOP_LINE:    State(APPROACHING_STOP_LINE, self.approaching_stop_line), 
            INTERSECTION_NAVIGATION:  State(INTERSECTION_NAVIGATION, self.intersection_navigation),
            TURNING_RIGHT:            State(TURNING_RIGHT, self.turning_right),
            TURNING_LEFT:             State(TURNING_LEFT, self.turning_left),
            GOING_STRAIGHT:           State(GOING_STRAIGHT, self.going_straight),
            TRACKING_LOCAL_PATH:      State(TRACKING_LOCAL_PATH, self.tracking_local_path),
            #roundabout, for roundabouts the car will track the local path
            ROUNDABOUT_NAVIGATION:    State(ROUNDABOUT_NAVIGATION, self.roundabout_navigation),
            #waiting states
            WAITING_FOR_PEDESTRIAN:   State(WAITING_FOR_PEDESTRIAN, self.waiting_for_pedestrian),
            WAITING_FOR_GREEN:        State(WAITING_FOR_GREEN, self.waiting_for_green),
            WAITING_AT_STOPLINE:      State(WAITING_AT_STOPLINE, self.waiting_at_stopline),
            WAITING_FOR_REROUTING:    State(WAITING_FOR_REROUTING, self.waiting_for_rerouting),
            #overtaking manouver
            OVERTAKING_STATIC_CAR:    State(OVERTAKING_STATIC_CAR, self.overtaking_static_car),
            OVERTAKING_MOVING_CAR:    State(OVERTAKING_MOVING_CAR, self.overtaking_moving_car),
            TAILING_CAR:              State(TAILING_CAR, self.tailing_car),
            AVOIDING_ROADBLOCK:       State(AVOIDING_ROADBLOCK, self.avoiding_roadblock),
            #parking
            PARKING:                  State(PARKING, self.parking),
        }

        # INITIALIZE ROUTINES
        self.routines = {
            FOLLOW_LANE:              Routine(FOLLOW_LANE,  self.follow_lane),      
            DETECT_STOP_LINE:         Routine(DETECT_STOP_LINE,  self.detect_stop_line),   
            SLOW_DOWN:                Routine(SLOW_DOWN,  self.slow_down),     
            ACCELERATE:               Routine(ACCELERATE,  self.accelerate),         
            CONTROL_FOR_SIGNS:        Routine(CONTROL_FOR_SIGNS,  self.control_for_signs),         
            CONTROL_FOR_SEMAPHORE:    Routine(CONTROL_FOR_SEMAPHORE,  self.control_for_semaphore), 
            CONTROL_FOR_PEDESTRIANS:  Routine(CONTROL_FOR_PEDESTRIANS,  self.control_for_pedestrians), 
            CONTROL_FOR_VEHICLES:     Routine(CONTROL_FOR_VEHICLES,  self.control_for_roadblocks),   
            CONTROL_FOR_ROADBLOCKS:   Routine(CONTROL_FOR_ROADBLOCKS,  self.control_for_roadblocks),
        }
        self.active_routines_names = []

        self.last_run_call = time()
        print('Brain initialized')
        self.switch_to_state(START_STATE)
        # self.switch_to_state(DOING_NOTHING)

    #=============== STATES ===============#
    def start_state(self):
        #exception: states should not perform actions
        self.car.stop()
        print('Generating route...')

        #localize the car and go to the first checkpoint
        #for now we will assume to be in the correct position
        # TODO detetct first node and route fom that node

        #get start and end nodes from the chekpoint list
        assert len(self.checkpoints) >= 2, 'List of checkpoints needs 2 ore more nodes'
        start_node = self.checkpoints[self.checkpoint_idx]
        end_node = self.checkpoints[self.checkpoint_idx+1] #already checked in end_state
        #calculate path
        self.path_planner.compute_shortest_path(start_node, end_node)
        #initialize the list of events on the path
        print('Augmenting path...')
        events = self.path_planner.augment_path()
        print(f'Path augmented')
        #add the events to the list of events, increasing it
        self.events = self.create_sequence_of_events(events)
        self.event_idx = 0
        self.next_event = self.events[0]
        print(f'EVENTS: idx: {self.event_idx}')
        for e in self.events:
            print(e)
        self.go_to_next_event()
        #draw the path 
        self.path_planner.draw_path()
        print('Starting...')
        # sleep(3)
        # cv.waitKey(0)
        self.switch_to_state(LANE_FOLLOWING)
        self.car.drive_speed(self.desired_speed)

    def end_state(self):
        self.activate_routines([FOLLOW_LANE])
        if self.curr_state.just_switched: # first call
            prev_event_dist = self.prev_event.dist
            end_dist = len(self.path_planner.path)*0.01
            dist_to_end = end_dist - prev_event_dist
            self.curr_state.var1 = dist_to_end
            self.car.drive_speed(self.desired_speed)
            self.curr_state.just_switched = False
        #end condition on the distance, can be done with GPS instead, or both
        dist_to_end = self.curr_state.var1
        stopping_in = np.abs(dist_to_end - self.car.dist_loc)
        print(f'Stoppig in {stopping_in-0.1} [m]')
        if stopping_in < 0.1:
            self.car.stop() #TODO control in position
            self.go_to_next_event()
            #start routing for next checkpoint
            self.next_checkpoint()
            self.switch_to_state(START_STATE)

    def doing_nothing(self):
        self.activate_routines([])

    def lane_following(self): # LANE FOLLOWING ##############################
        #start driving if it's the first time it has been called
        if self.curr_state.just_switched:
            self.car.drive_speed(self.desired_speed)
            self.activate_routines([FOLLOW_LANE, DETECT_STOP_LINE])
            self.curr_state.just_switched = False
        
        if self.next_event.name == PARKING_EVENT:
            self.activate_routines([FOLLOW_LANE]) #we dont need stoplines in parking
            dist_between_events = self.next_event.dist - self.prev_event.dist
            #Relative positioning is reset at every stopline, so we can use that to approximately calculate the distance to the parking spot
            approx_dist_from_parking = dist_between_events - self.car.dist_loc
            print(f'Approx dist from parking: {approx_dist_from_parking}')
            if approx_dist_from_parking < PARKING_DISTANCE_SLOW_DOWN_THRESHOLD: #we are reasonably close to the parking spot
                # self.activate_routines([FOLLOW_LANE, SLOW_DOWN])
                self.car.stop()
                self.switch_to_state(PARKING)

        #NOTE: TEMPORARY solution
        if self.next_event.name == HIGHWAY_EXIT_EVENT:
            self.go_to_next_event()

        # end of current route, go to end state
        if self.next_event.name == END_EVENT:
            self.switch_to_state(END_STATE)
            
        #check if we are approaching a stop_line, but only if we are far enough from the previous stop_line
        far_enough_from_prev_stop_line = (self.prev_event.name is None) or (self.car.dist_loc > STOP_LINE_DISTANCE_THRESHOLD)
        print(f'stop enough: {self.car.dist_loc}') if self.prev_event.name is not None else None
        if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE and far_enough_from_prev_stop_line and self.routines[DETECT_STOP_LINE].active:
            self.switch_to_state(APPROACHING_STOP_LINE)


    def approaching_stop_line(self):
        self.activate_routines([FOLLOW_LANE, SLOW_DOWN, DETECT_STOP_LINE])
        dist = self.detect.est_dist_to_stop_line
        #check if we are here by mistake
        if dist > STOP_LINE_APPROACH_DISTANCE:
            self.switch_to_state(LANE_FOLLOWING)
            self.activate_routines([FOLLOW_LANE, DETECT_STOP_LINE])
            self.car.drive_speed(self.desired_speed)
        elif dist < STOP_LINE_STOP_DISTANCE: #STOP THE CAR
            next_event_name = self.next_event.name
            # Events with stopline
            if next_event_name == INTERSECTION_STOP_EVENT:
                self.switch_to_state(WAITING_AT_STOPLINE) 
            elif next_event_name == INTERSECTION_TRAFFIC_LIGHT_EVENT:
                self.switch_to_state(WAITING_FOR_GREEN)
            elif next_event_name == INTERSECTION_PRIORITY_EVENT:
                self.switch_to_state(INTERSECTION_NAVIGATION)
            elif next_event_name == JUNCTION_EVENT:
                self.switch_to_state(INTERSECTION_NAVIGATION) #TODO: careful with this
            elif next_event_name == ROUNDABOUT_EVENT:
                self.switch_to_state(ROUNDABOUT_NAVIGATION)
            elif next_event_name == CROSSWALK_EVENT:
                self.switch_to_state(WAITING_FOR_PEDESTRIAN)
            # Events without stopline = LOGIC ERROR
            elif next_event_name == PARKING_EVENT:
                print('WARNING: UNEXPECTED STOP LINE FOUND WITH PARKING AS NEXT EVENT')
                exit() #TODO: handle this case
            elif next_event_name == HIGHWAY_EXIT_EVENT:
                print('WARNING: UNEXPECTED STOP LINE FOUND WITH HIGHWAY EXIT AS NEXT EVENT')
                exit() #TODO: handle this case
            #Every time we stop for a stopline, we reset the local frame of reference
            self.car.reset_rel_pose() 

    def intersection_navigation(self):
        self.activate_routines([])
        if USE_LOCAL_TRACKING_FOR_INTERSECTIONS: #use local tracking, more reliable if local yaw estimaiton and localization are good
            self.switch_to_state(TRACKING_LOCAL_PATH)
        else: # go for precoded manouvers, less precise and less robust, but predictable
            if self.next_event.curvature > 0.5:
                self.switch_to_state(TURNING_RIGHT)
            elif self.next_event.curvature < -0.5:
                self.switch_to_state(TURNING_LEFT)
            else:
                self.switch_to_state(GOING_STRAIGHT)
            self.car.drive_speed(self.desired_speed)
        
    def turning_right(self):
        self.activate_routines([])
        if self.curr_state.just_switched:
            #detetc the stop_line to see how far we are from the line
            self.detect.detect_stop_line(self.car.frame, SHOW_IMGS)
            if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE:
                #create a private variable for this state
                self.curr_state.var1 = self.detect.est_dist_to_stop_line
            else: self.curr_state.var1 = 0.0
            self.curr_state.just_switched = False
        #straight section
        end_straight_section = self.curr_state.start_distance + self.curr_state.var1 + self.car.WB - 0.05
        print('End straight section: ', end_straight_section)
        #curved_section
        end_curved_section = end_straight_section + OPEN_LOOP_PERCENTAGE_OF_PATH_AHEAD*self.next_event.length_path_ahead
        print('End curved section: ', end_curved_section)
        curr_position = self.car.encoder_distance
        print('Current position: ', curr_position)
        if curr_position < end_straight_section:
            print('Driving straight')
            self.car.drive(self.desired_speed, angle=0.0)
        elif end_straight_section < curr_position < end_curved_section: 
            print('Turning right')
            steer_angle = np.arctan(self.car.WB*self.next_event.curvature)
            assert steer_angle > 0.0, 'Steer angle is negative in right turn'
            self.car.drive(speed=self.desired_speed, angle=np.rad2deg(steer_angle))
        else: #end of the maneuver
            self.switch_to_state(LANE_FOLLOWING)
            self.go_to_next_event()

    def turning_left(self):
        self.activate_routines([])
        if self.curr_state.just_switched:
            #detetc the stop_line to see how far we are from the line
            self.detect.detect_stop_line(self.car.frame,SHOW_IMGS)
            if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE:
                #create a private variable for this state
                self.curr_state.var1 = self.detect.est_dist_to_stop_line
            else: self.curr_state.var1 = 0.0
            self.curr_state.just_switched = False
        #straight section
        end_straight_section = self.curr_state.start_distance + self.curr_state.var1 + self.car.WB 
        print('End straight section: ', end_straight_section)
        #curved_section
        end_curved_section = end_straight_section + OPEN_LOOP_PERCENTAGE_OF_PATH_AHEAD*self.next_event.length_path_ahead
        print('End curved section: ', end_curved_section)
        curr_position = self.car.encoder_distance
        print('Current position: ', curr_position)
        if curr_position < end_straight_section:
            print('Driving straight')
            self.car.drive(self.desired_speed, angle=0.0)
        elif end_straight_section < curr_position < end_curved_section: 
            print('Turning left')
            steer_angle = np.arctan(self.car.WB*self.next_event.curvature)
            assert steer_angle < 0.0, 'Steer angle is positive in left turn'
            self.car.drive(speed=self.desired_speed, angle=np.rad2deg(steer_angle))
        else: #end of the maneuver
            self.switch_to_state(LANE_FOLLOWING)
            self.go_to_next_event()

    def going_straight(self):
        self.activate_routines([])
        distance_to_stop = self.curr_state.start_distance+OPEN_LOOP_PERCENTAGE_OF_PATH_AHEAD*self.next_event.length_path_ahead
        if self.car.encoder_distance < distance_to_stop: 
            steer_angle = np.arctan(self.car.WB*self.next_event.curvature)
            assert -0.2 < steer_angle < 0.2, 'Steer angle is too big going straight'
            self.car.drive(speed=self.desired_speed, angle=np.rad2deg(steer_angle))
        else: #end of the maneuver
            self.switch_to_state(LANE_FOLLOWING)
            self.go_to_next_event()

    def tracking_local_path(self):
        print('State: tracking_local_path') #var1=initial distance from stop_line, #var2=path to follow
        self.activate_routines([])
        if self.curr_state.just_switched:
            self.car.stop()
            #get distance from the line and lateral error e2
            self.detect.detect_stop_line(self.car.frame, SHOW_IMGS)
            if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE:
                d = self.detect.est_dist_to_stop_line
            else: d = 0.0
            e2, _, _ = self.detect.detect_lane(self.car.frame, SHOW_IMGS)
            # retrieve global position of the car, can be done usiing the stop_line
            # or using the gps, but for now we will use the stop_line
            # NOTE: in the real car we need to have a GLOBAL YAW OFFSET to match the simulator with the real track

            # local_path_wf = self.next_event.path_ahead # wf = world frame
            stop_line_position = self.next_event.point
            stop_line_yaw = self.next_event.yaw_stopline
            local_path_slf_rot = self.next_event.path_ahead #local path in the stop line frame

            # get orientation of the car in the stop line frame
            yaw_car = self.car.yaw
            yaw_mult_90 = get_yaw_closest_axis(yaw_car)
            alpha = diff_angle(yaw_car, yaw_mult_90) #get the difference from the closest multiple of 90deg
            assert abs(alpha) < np.pi/6, f'Car orientation wrt stopline is too big, it needs to be better aligned, alpha = {alpha}'
            rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            
            # ## get position of the car in the stop line frame
            # # car_position_slf = self.car.position - stop_line_position #with good gps estimate
            car_position_slf = -np.array([+d+0.3+0.13, +e2])#np.array([+d+0.2, -e2])
            local_path_cf = local_path_slf_rot - car_position_slf #cf = car frame
            #rotate from slf to cf
            local_path_cf = local_path_cf @ rot_matrix
            self.curr_state.var1 = local_path_cf

            self.curr_state.just_switched = False

            if SHOW_IMGS:
                img = self.car.frame.copy()
                #project the whole path (true)
                img, _ = project_onto_frame(img, self.car, self.path_planner.path, align_to_car=True, color=(0, 100, 0))
                #project local path (estimated), it should match the true path
                img, _ = project_onto_frame(img, self.car, local_path_cf, align_to_car=False)
                cv.imshow('brain_debug', img)
                cv.waitKey(1)
                self.curr_state.var2 = np.array([self.car.x_true, self.car.y_true]) #var2 hold original position
                true_start_pos_wf = self.curr_state.var2

                alpha = alpha + stop_line_yaw
                rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])

                est_car_pos_slf = car_position_slf
                est_car_pos_slf_rot = est_car_pos_slf @ rot_matrix.T
                est_car_pos_wf = est_car_pos_slf_rot + stop_line_position
                cv.circle(self.path_planner.map, mR2pix(est_car_pos_wf), 25, (255, 0, 255), 5)
                cv.circle(self.path_planner.map, mR2pix(true_start_pos_wf), 30, (0, 255, 0), 5)
                cv.imshow('Path', self.path_planner.map)
                cv.waitKey(1)
                #debug
                self.car.stop()
                # sleep(1.0)
                cv.namedWindow('local_path', cv.WINDOW_NORMAL)
                local_map_img = np.zeros_like(self.path_planner.map)
                h = local_map_img.shape[0]
                w = local_map_img.shape[1]
                local_map_img[w//2-2:w//2+2, :] = 255
                local_map_img[:, h//2-2:h//2+2] = 255
                for i in range(len(local_path_cf)):
                    if (i%3 == 0):
                        p = local_path_cf[i]
                        pix = mR2pix(p)
                        pix = (int(pix[0]+w//2), int(pix[1]-h//2))
                        cv.circle(local_map_img, pix, 10, (0, 150, 150), -1)
                cv.imshow('local_path', local_map_img)
                cv.waitKey(1)  
                self.curr_state.var3 = local_map_img

        D = POINT_AHEAD_DISTANCE_LOCAL_TRACKING
        #track the local path using simple pure pursuit
        local_path = self.curr_state.var1
        car_pos_loc = np.array([self.car.x_loc, self.car.y_loc])
        local_path_cf = local_path - car_pos_loc
        dist_path = norm(local_path_cf, axis=1)
        #get idx of car position on the path
        idx_car_on_path = np.argmin(dist_path)
        dist_path = dist_path[idx_car_on_path:]
        dist_path = np.abs(dist_path - D)
        #get idx of point ahead
        idx_point_ahead = np.argmin(dist_path) + idx_car_on_path
        # idx_point_ahead = int(100*self.car.dist_loc) # m -> cm -> idx
        print(f'idx_point_ahead: {idx_point_ahead} / {len(local_path_cf)}')

        rot_matrix = np.array([[np.cos(self.car.yaw_loc), -np.sin(self.car.yaw_loc)], [np.sin(self.car.yaw_loc), np.cos(self.car.yaw_loc)]])
        local_path_cf = local_path_cf @ rot_matrix

        if SHOW_IMGS:
            local_map_img = self.curr_state.var3
            h = local_map_img.shape[0]
            w = local_map_img.shape[1]
            angle = self.car.yaw_loc_o # + self.car.yaw_loc
            rot_matrix_w = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # show car position in the local frame (from the encoder)
            cv.circle(local_map_img, (mR2pix(car_pos_loc)[0]+w//2, mR2pix(car_pos_loc)[1]-h//2), 5, (255, 0, 255), 2)
            # show the true position to check if they match, translated wrt starting position into the local frame
            true_start_pos_wf = self.curr_state.var2
            true_pos_loc = np.array([self.car.x_true, self.car.y_true]) - true_start_pos_wf
            true_pos_loc = true_pos_loc @ rot_matrix_w
            cv.circle(local_map_img, (mR2pix(true_pos_loc)[0]+w//2, mR2pix(true_pos_loc)[1]-h//2), 7, (0, 255, 0), 2)
            cv.imshow('local_path', local_map_img)
            true_start_pos_wf = self.curr_state.var2
            car_pos_loc_rot_wf = car_pos_loc @ rot_matrix_w.T
            car_pos_wf = true_start_pos_wf + car_pos_loc_rot_wf
            # show car position in wf (encoder)
            cv.circle(self.path_planner.map, mR2pix(car_pos_wf), 5, (255, 0, 255), 2)
            # show the true position to check if they match
            true_pos_wf = np.array([self.car.x_true, self.car.y_true])
            cv.circle(self.path_planner.map, mR2pix(true_pos_wf), 7, (0, 255, 0), 2)
            cv.imshow('Path', self.path_planner.map)
            cv.waitKey(1)
        if np.abs(get_curvature(local_path_cf)) < 0.5: #the local path is straight
            max_idx = len(local_path_cf)-30 #dont follow until the end
        else: #curvy path
            max_idx = len(local_path_cf)-1  #follow until the end
        # State exit conditions
        if idx_point_ahead >= max_idx: #we reached the end of the path
            self.switch_to_state(LANE_FOLLOWING)
            self.go_to_next_event()
        else: #we are still on the path
            point_ahead = local_path_cf[idx_point_ahead]
            if SHOW_IMGS:
                img = self.car.frame.copy()
                img, _ = project_onto_frame(img, self.car, local_path_cf, align_to_car=False)
                img, _ = project_onto_frame(img, self.car, point_ahead, align_to_car=False, color=(0,0,255))
                cv.imshow('brain_debug', img)
                cv.waitKey(1)
            gains = [0.0, .0, .99, 0.08] #k1,k2,k3,k3D
            e2 = local_path_cf[idx_car_on_path][1] 
            yaw_error = np.arctan2(point_ahead[1], point_ahead[0]) 
            out_speed, out_angle = self.controller.get_control(e2, yaw_error, 0.0, self.desired_speed, gains=gains)
            self.car.drive(out_speed, np.rad2deg(out_angle))

    def roundabout_navigation(self):
        self.switch_to_state(TRACKING_LOCAL_PATH)

    def waiting_for_pedestrian(self):
        self.switch_to_state(LANE_FOLLOWING) #temporary
        self.go_to_next_event() #temporary
        pass

    def waiting_for_green(self):
        #temporary fix
        self.switch_to_state(WAITING_AT_STOPLINE)

    def waiting_at_stopline(self):
        self.activate_routines([]) #no routines
        self.car.stop()
        if (time() - self.curr_state.start_time) > STOP_WAIT_TIME:
            self.switch_to_state(INTERSECTION_NAVIGATION)

    def waiting_for_rerouting(self):
        pass

    def overtaking_static_car(self):
        pass

    def overtaking_moving_car(self):
        pass

    def tailing_car(self):
        pass

    def avoiding_roadblock(self):
        pass

    def parking(self):
        #Substates
        LOCALIZING_PARKING_SPOT = 1
        CHECKING_FOR_PARKED_CARS = 2
        STEP0 = 69
        T_STEP1 = 3
        T_STEP2 = 4
        T_STEP3 = 5
        T_STEP4 = 6
        T_STEP5 = 7
        S_STEP1 = 8
        S_STEP2 = 9
        S_STEP3 = 10
        S_STEP4 = 11
        S_STEP5 = 12
        S_STEP6 = 13
        S_STEP7 = 14
        S_STEP8 = 15
        PARK_END = 16
        #park types
        T_PARK = 't'
        S_PARK = 's'
        if self.curr_state.just_switched:
            # We just got in the parking state, we came from lane following, we are reasonably close to the parking spot and we are not moving
            # self.curr_state.var1 will hold the parking substate, the parking type, and if it has just changed state
            park_state = LOCALIZING_PARKING_SPOT
            #find park type with position
            park_pos = self.next_event.point
            s_pos = np.array(self.path_planner.get_coord('177'))
            t_pos = np.array(self.path_planner.get_coord('162'))
            print(park_pos)
            print(s_pos)
            print(t_pos)
            d_s = norm(park_pos - s_pos) #distances from the s parking spot
            d_t = norm(park_pos - t_pos) #distances from the t parking spot
            if d_s < d_t and d_s < 0.2: park_type = S_PARK
            elif d_t <= d_s and d_t < 0.2: park_type = T_PARK
            else:
                print('ERROR: PARKING -> parking spot is not close to expected parking spot position!')
                exit()
            self.curr_state.var1 = (park_state, park_type, True)           
            self.curr_state.just_switched = False
        
        park_pos = self.next_event.point
        park_state, park_type, just_changed = self.curr_state.var1

        #############################################################################################
        #first state: localizing with precision the parking spot
        if park_state == LOCALIZING_PARKING_SPOT:
            self.activate_routines([FOLLOW_LANE]) 

            if USE_GPS_FOR_PARKING: #TODO finish implemeting
                if not self.conditions[TRUST_GPS]:
                    curr_time = time()
                    passed_time = curr_time - self.curr_state.start_time
                    if passed_time > PARK_MAX_SECONDS_W8_GPS:
                        print('ERROR: GPS Timout!')
                        raise NotImplementedError #TODO: implement
                    print(f'Parking: GPS not trusted, waiting for GPS to be trusted for {passed_time}/{PARK_MAX_SECONDS_W8_GPS} [s]...')
                    return
                car_est_pos = np.array([self.car.x_est, self.car.y_est])
                park_index_on_path = int(self.next_event.dist*100) #one sample for every cm in the path
                path_to_analyze = self.path_planner.path[max(0, park_index_on_path-SUBPATH_LENGTH_FOR_PARKING):
                                    min(park_index_on_path+SUBPATH_LENGTH_FOR_PARKING, len(self.path_planner.path))]
                car_idx_on_path = np.argmin(norm(path_to_analyze - car_est_pos, axis=1))
                park_index_on_path = SUBPATH_LENGTH_FOR_PARKING
                if car_idx_on_path < park_index_on_path:
                    print('Behind parking spot')
                else: print('In front of parking spot')
            elif USE_SIGN_FOR_PARKING:
                SIGN_DEQUE_LENGTH = 15
                if just_changed:
                    #create a deque of past SIGN_DEQUE_LENGTH sing detection results
                    park_sign_queue = deque(maxlen=SIGN_DEQUE_LENGTH)
                    self.curr_state.var2 = park_sign_queue #assign var2 to the queue
                    self.car.reset_rel_pose() #reset the car pose to the current pose
                    self.curr_state.var1 = (park_state, park_type, False) #set var1, with just_changed to false
                    self.curr_state.var3 = False #parking sign reached or not

                sign, _, _, _ = self.detect.detect_sign(self.car.frame, show_ROI=True)
                if sign is not None:
                    self.curr_state.var2.append(sign)
                else: self.curr_state.var2.append('None')
                park_sign_queue = self.curr_state.var2
                parking_spot_reached = self.curr_state.var3
                if len(park_sign_queue) > SIGN_DEQUE_LENGTH-2:
                    mode = max(set(park_sign_queue), key=park_sign_queue.count) #get the mode of the deque
                    if mode == 'park' or parking_spot_reached:
                        self.curr_state.var3 = True #parking sign reached
                        print('Reached parking spot, keep going until the sign disappears')
                        # if sign is None or sign != 'park':
                        if mode != 'park':
                            print('Sign disappeared, setting up things for searching for parked cars')
                            self.car.stop()
                            #go to next substate
                            self.curr_state.var1 = (CHECKING_FOR_PARKED_CARS, park_type, True)
                    elif mode != 'park' and not parking_spot_reached and self.car.dist_loc < 2.0:
                        print('Searching...')
                    else:
                        print('ERROR: PARKING: Sign not detected OR we are not in front of a parking spot!')
                        self.car.stop()
                        sleep(5)
                        exit()
                else: 
                    print('Waiting for sign...')
                    self.car.drive_speed(PARK_SEARCH_SPEED)
                    dist_from_search_start = self.car.dist_loc
                    if dist_from_search_start > MAX_PARK_SEARCH_DIST:
                        print('ERROR: PARKING: Car didnt find the parking sign after more than {} [m]'.format(MAX_PARK_SEARCH_DIST))
                        self.car.stop()
                        sleep(5)
                        exit()

        #############################################################################################
        #second state: checking if there are parked cars in the parking spots    
        elif park_state == CHECKING_FOR_PARKED_CARS:
            print('Checking for parked cars...')
            if just_changed:
                self.activate_routines([FOLLOW_LANE])
                self.car.reset_rel_pose()
                self.curr_state.var2 = (False, False, False, False) #(car in spot1, car in spot2, looked for car in spot1, looked for car in spot2)
                self.car.drive_speed(PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)

            car_in_spot1, car_in_spot2, checked1, checked2 = self.curr_state.var2
            curr_dist = self.car.dist_loc

            if park_type == T_PARK:
                dist_first_spot = DIST_SIGN_FIRST_T_SPOT
                dist_spots = DIST_T_SPOTS
                further_dist = FURTHER_DIST_T
            elif park_type == S_PARK:
                dist_first_spot = DIST_SIGN_FIRST_S_SPOT
                dist_spots = DIST_S_SPOTS
                further_dist = FURTHER_DIST_S
            else:
                print('ERROR: PARKING: Unknown parking type!')
                self.car.stop()
                sleep(5)
                exit()

            if (dist_first_spot < curr_dist < (dist_first_spot+0.1)) and not checked1:
                self.car.stop()
                sleep(0.2)
                #get lateral sonar distance
                lateral_sonar_dist = self.car.filtered_lateral_sonar_distance
                checked1 = True
                if lateral_sonar_dist < 0.5:
                    print('Car in spot 1')
                    car_in_spot1 = True
                self.car.drive_speed(PARK_MANOUVER_SPEED)
            elif (dist_first_spot+dist_spots < curr_dist < (dist_first_spot+dist_spots+0.1)) and not checked2:
                self.car.stop()
                sleep(0.2)
                #get lateral sonar distance
                lateral_sonar_dist = self.car.filtered_lateral_sonar_distance
                checked2 = True
                if lateral_sonar_dist < 0.5:
                    print('Car in spot 2')
                    car_in_spot2 = True
                self.car.drive_speed(PARK_MANOUVER_SPEED)
            elif dist_first_spot+dist_spots+further_dist < curr_dist:
                self.car.stop()
                print('Ending search for parked cars')
                print('Car in spot 1: {}'.format(car_in_spot1))
                print('Car in spot 2: {}'.format(car_in_spot2))
                sleep(1)
                self.curr_state.var1 = (STEP0, park_type, True)

            self.curr_state.var2 = (car_in_spot1, car_in_spot2, checked1, checked2) #update var2 at the end of every iteration
        
        #############################################################################################
        #STEP 0 -> ALIGN WITH THE PARKING SPOT
        elif park_state == STEP0:
            #we are standing still at the end of the 2 parking spots 
            print('STEP0 -> Aligning with the parking spot...')

            car_in_spot1, car_in_spot2, _, _ = self.curr_state.var2

            if just_changed:
                if car_in_spot1 and car_in_spot2: #TODO change the behaviour: go to next event or something
                    print('ERROR: PARKING: Car in both spots!')
                    self.car.stop()
                    sleep(3)
                    exit()
                elif not car_in_spot2:
                    print('Spot 2 is free. Going to spot 2 now')
                    park_state = T_STEP2 if park_type == T_PARK else S_STEP2
                    self.curr_state.var1 = (park_state, park_type, True)
                else: #car in spot2, spot1 free
                    print('Spot 1 is free. Going to spot 1 now')
                    self.activate_routines([FOLLOW_LANE])
                    self.car.reset_rel_pose()
                    self.car.drive_speed(-PARK_MANOUVER_SPEED)
                    self.curr_state.var1 = (park_state, park_type, False)

            if not car_in_spot1 and car_in_spot2:
                dist = self.car.dist_loc
                print(f'Distance: {dist}')
                dist_spots = DIST_S_SPOTS if park_type == S_PARK else DIST_T_SPOTS
                if dist > dist_spots:
                    self.car.stop()
                    print('Aligned with first parking spot')
                    sleep(1)
                    park_state = T_STEP2 if park_type == T_PARK else S_STEP2
                    self.curr_state.var1 = (park_state, park_type, True)

        elif park_state == T_STEP2:
            print('T-parking manouver step 2, we are aligned with the parking spot, going right and backward')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(+T_ANGLE)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(-PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_2T:
                self.car.stop()
                self.curr_state.var1 = (T_STEP3, park_type, True)
        elif park_state == T_STEP3:
            print('T-parking manouver step 3, going backward')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_3T}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(0.0)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(-PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_3T:
                self.car.stop()
                print('Parked')
                self.achievements[PARK_ACHIEVED] = True #<><><><><><><><><><><><><> PARK ACHIEVED <><><><><><><><><><><><><>
                sleep(3.0)
                self.curr_state.var1 = (T_STEP4, park_type, True)
        elif park_state == T_STEP4:
            print('T-parking manouver step 4, going forward')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_3T}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(0.0)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_3T:
                self.car.stop()
                self.curr_state.var1 = (T_STEP5, park_type, True)
        elif park_state == T_STEP5:
            print('T-parking manouver step 5, going right and forward')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_2T}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(+T_ANGLE)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(+PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_2T:
                self.car.stop()
                print('Back in lane')
                self.curr_state.var1 = (PARK_END, park_type, True)
        
        # # S parking manouver
        # elif park_state == S_STEP1:
        #     print('S-parking manouver step 1')
        elif park_state == S_STEP2:
            print('S-parking manouver step 2')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_2S}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(+S_ANGLE)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(-PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_2S:
                self.car.stop()
                self.curr_state.var1 = (S_STEP3, park_type, True)
        elif park_state == S_STEP3:
            print('S-parking manouver step 3')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_2S}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(-S_ANGLE)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(-PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_2S:
                self.car.stop()
                self.curr_state.var1 = (S_STEP4, park_type, True)
        elif park_state == S_STEP4:
            print('S-parking manouver step 4')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_4S}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(0.0)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(+PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_4S:
                self.car.stop()
                print('Parked')
                self.achievements[PARK_ACHIEVED] = True #<><><><><><><><><><><><><> PARK ACHIEVED <><><><><><><><><><><><><>
                sleep(3.0)
                self.curr_state.var1 = (S_STEP5, park_type, True)
        elif park_state == S_STEP5:
            print('S-parking manouver step 5')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_4S}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(0.0)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(-PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_4S:
                self.car.stop()
                print('Parked')
                self.curr_state.var1 = (S_STEP6, park_type, True)
        elif park_state == S_STEP6:
            print('S-parking manouver step 6')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_2S}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(-S_ANGLE)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(+PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_2S:
                self.car.stop()
                self.curr_state.var1 = (S_STEP7, park_type, True)
        elif park_state == S_STEP7:
            print('S-parking manouver step 7')
            print(f'Distance: {self.car.dist_loc:.2f}/{DIST_2S}')
            self.activate_routines([])
            if just_changed:
                self.car.drive_angle(+S_ANGLE)
                sleep(STEER_ACTUATION_DELAY_PARK)
                self.car.reset_rel_pose()
                self.car.drive_speed(+PARK_MANOUVER_SPEED)
                self.curr_state.var1 = (park_state, park_type, False)
            if self.car.dist_loc > DIST_2S:
                self.car.stop()
                self.curr_state.var1 = (PARK_END, park_type, True)

        #end of manouver, go to next event
        elif park_state == PARK_END:
            self.switch_to_state(LANE_FOLLOWING)
            self.go_to_next_event()


            
        

    
    #=============== ROUTINES ===============#
    def follow_lane(self):
        e2, e3, point_ahead = self.detect.detect_lane(self.car.frame, SHOW_IMGS)
        _, angle_ref = self.controller.get_control(e2, e3, 0, self.desired_speed)
        angle_ref = np.rad2deg(angle_ref)
        if self.car.speed < 0.1: #inverse driving in reverse
            angle_ref = -angle_ref
        print(f'angle_ref: {angle_ref}')
        self.car.drive_angle(angle_ref)

    def detect_stop_line(self):
        #update the variable self.detect.est_dist_to_stop_line
        self.detect.detect_stop_line(self.car.frame, SHOW_IMGS)

    def slow_down(self):
        self.car.drive_speed(self.desired_speed*0.2)

    def accelerate(self):
        pass

    def control_for_signs(self):
        pass

    def control_for_semaphore(self):
        pass

    def control_for_pedestrians(self):
        pass

    def control_for_vehicles(self):
        pass

    def control_for_roadblocks(self):
        pass

    # STATE CHECKS
    def check_logic(self):
        print('check_logic')
        pass
    
    #===================== STATE MACHINE MANAGEMENT =====================#
    def run(self):
        #update trust_gps
        self.conditions[TRUST_GPS] = self.car.trust_gps and not ALWAYS_DISTRUST_GPS or ALWAYS_TRUST_GPS

        print('==========================================================================')
        print(f'STATE: {self.curr_state}')
        print(f'UPCOMING_EVENT: {self.next_event}')
        print(f'ROUTINES: {self.active_routines_names}')
        self.run_current_state()
        print('==========================================================================')
        print()
        self.run_routines()

    def run_current_state(self):
        self.curr_state.run()

    def run_routines(self):
        for k, r in self.routines.items():
            if r.active: r.run()

    def activate_routines(self, routines_to_activate):
        """
        routines_to_activate are a list of strings (routines)
        ex: ['follow_lane', 'control_for_signs']
        """
        self.active_routines_names = []
        for k,r in self.routines.items():
            r.active = k in routines_to_activate
            if r.active: self.active_routines_names.append(k)
    
    def switch_to_state(self, to_state, interrupt=False):
        """
        to_state is the string of the desired state to switch to
        ex: 'lane_following'
        """
        assert to_state in self.states, f'{to_state} is not a valid state'
        self.prev_state = self.curr_state
        self.curr_state = self.states[to_state]
        for k,s in self.states.items():
            s.active = k == to_state
        if not interrupt:
            self.curr_state.start_time = time()
            self.curr_state.start_position = np.array([self.car.x_est, self.car.y_est]) ######## maybe another position
            self.curr_state.start_distance = self.car.encoder_distance
            self.curr_state.interrupted = False
        else:
            self.curr_state.interrupted = True
        self.curr_state.just_switched = True

    def switch_to_prev_state(self):
        self.switch_to_state(self.prev_state)
    
    def go_to_next_event(self):
        """
        Switches to the next event on the path
        """
        self.prev_event = self.next_event
        if self.event_idx == len(self.events):
            #no more events, for now
            pass
        else:
            self.next_event = self.events[self.event_idx]
            self.event_idx += 1
    
    def next_checkpoint(self):
        self.checkpoint_idx += 1
        if self.checkpoint_idx < (len(self.checkpoints)-1): #check if it's last
            #update events
            self.prev_event = deepcopy(self.next_event)
            pass
        else: 
            #it was the last checkpoint
            print('Reached last checkpoint...\nExiting...')
            cv.destroyAllWindows()
            exit()

    def create_sequence_of_events(self, events):
        """
        events is a list of strings (events)
        ex: ['lane_following', 'control_for_signs']
        """
        to_ret = []
        for e in events:
            name = e[0]
            dist = e[1]
            point = e[2]
            path_ahead = e[3] #path in global coordinates
            if path_ahead is not None:
                loc_path = path_ahead - point
                #get yaw of the stopline
                assert path_ahead.shape[0] > 10, f'path_ahead is too short: {path_ahead.shape[0]}'
                path_first_10 = path_ahead[:10]
                diff10 = path_first_10[1:] - path_first_10[:-1]
                yaw_raw = np.median(np.arctan2(diff10[:,1], diff10[:,0]))
                yaw_stopline = get_yaw_closest_axis(yaw_raw)
                rot_matrix = np.array([[np.cos(yaw_stopline), -np.sin(yaw_stopline)],
                                        [np.sin(yaw_stopline), np.cos(yaw_stopline)]])
                loc_path = loc_path @ rot_matrix
                path_to_ret = loc_path
                curv = get_curvature(path_ahead)
                print(f'yaw_stopline: {yaw_stopline}, name: {name}, curv: {curv}')
                len_path_ahead = 0.01*len(path_ahead)
            else:
                path_to_ret = None
                curv = None
                yaw_stopline = None
                len_path_ahead = None
            event = Event(name, dist, point, yaw_stopline, path_to_ret, len_path_ahead, curv)
            to_ret.append(event)
        #add end of path event
        # ee_dist = (len(self.path_planner.path) - 1 )*0.01
        ee_point = self.path_planner.path[-1]
        end_event = Event(END_EVENT, dist=0.0, point=ee_point)
        to_ret.append(end_event)
        return to_ret

    