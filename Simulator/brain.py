#!/usr/bin/python3

from turtle import speed
import numpy as np
import cv2 as cv
import os
from time import time, sleep

from automobile_data_interface import Automobile_Data
from helper_functions import *
from PathPlanning4 import PathPlanning
from controller3 import Controller
from detection import Detection

from helper_functions import *

IN_SIMULATOR = True
SHOW_IMGS = True

START_NODE = 86
END_NODE = 285#236#169#116

#========================= STATES ==========================
START_STATE = 'start_state'
END_STATE = 'end_state'
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
AVOIDING_OBSTACLE = 'avoiding_obstacle'

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
        return self.name
    
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

EVENT_TYPES = [INTERSECTION_STOP_EVENT, INTERSECTION_TRAFFIC_LIGHT_EVENT, INTERSECTION_PRIORITY_EVENT,
                JUNCTION_EVENT, ROUNDABOUT_EVENT, CROSSWALK_EVENT, PARKING_EVENT]

class Event:
    def __init__(self, name=None, dist=None, point=None, path_ahead=None, length_path_ahead=None, curvature=None):
        self.name = name                # name/type of the event
        self.dist = dist                # distance of event from start of path
        self.point = point              # [x,y] position on the map of the event
        self.path_ahead = path_ahead    # sequence of points after the event, only for intersections or roundabouts
        self.length_path_ahead = length_path_ahead   # length of the path after the event, only for intersections or roundabouts
        self.curvature = curvature      # curvature of the path ahead of the event
    
    def __str__(self):
        return self.name

#======================== CONDITIONS ==========================
CONDITIONS = {
    'can_switch_right':         False,  # if true, the car can switch to the right lane, for obstacle avoidance
    'can_switch_left':          False,  # if true, the car can switch to the left lane, for overtaking or obstacle avoidance
    'highway':                  False,  # if true, the car is in a highway, the speed should be higher on highway, 
    'trust_gps':                True,   # if true the car will trust the gps, for example in expecting a sign or traffic light
                                        # can be set as false if there is a lot of package loss or the car has not received signal in a while
    'car_on_path':              True,   # if true, the car is on the path, if the gps is trusted and the position is too far from the path it will be set to false
}

#========================= PARAMTERS ==========================
YAW_GLOBAL_OFFSET = 0.0 #global offset of the yaw angle between the real track and the simulator map
STOP_LINE_APPROACH_DISTANCE = 0.3
STOP_LINE_STOP_DISTANCE = 0.1
assert STOP_LINE_STOP_DISTANCE < STOP_LINE_APPROACH_DISTANCE
STOP_WAIT_TIME = .5 #3.0
OPEN_LOOP_PERCENTAGE_OF_PATH_AHEAD = 0.6 #0.6
STOP_LINE_DISTANCE_THRESHOLD = 0.15 #distance from previous stop_line from which is possible to start detecting a stop line again
POINT_AHEAD_DISTANCE_LOCAL_TRACKING = 0.3
USE_LOCAL_TRACKING_FOR_INTERSECTIONS = True

#==============================================================
#=========================== BRAIN ============================
#==============================================================
class Brain:
    def __init__(self, car, controller, detection, path_planner, desired_speed=0.3, debug=True):
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

        #current and previous states (class State)
        self.curr_state = State
        self.prev_state = State
        #previous and next event (class Event)
        self.prev_event = Event
        self.next_event = Event
        self.event_idx = 0

        #debug
        self.debug = debug
        if self.debug:
            cv.namedWindow('brain_debug', cv.WINDOW_NORMAL)
            self.debug_frame = None

        self.conditions = CONDITIONS

        # INITIALIZE STATES
        self.states = { 
            START_STATE:              State(START_STATE, self.start_state),
            END_STATE:                State(END_STATE, self.end_state),
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
            AVOIDING_OBSTACLE:        State(AVOIDING_OBSTACLE, self.avoiding_obstacle),
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
        print('Starting in 2 second...')
        sleep(2)
        # cv.waitKey(0)
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
        #start driving if it's the first time it has been called
        if self.curr_state.just_switched:
            self.car.drive_speed(self.desired_speed)
            self.curr_state.just_switched = False
        #check if we are approaching a stop_line, but only if we are far enough from the previous stop_line
        far_enough_from_prev_stop_line = (self.car.encoder_distance - self.curr_state.start_distance) > STOP_LINE_DISTANCE_THRESHOLD
        if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE and far_enough_from_prev_stop_line:
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
            next_event_name = self.next_event.name
            if next_event_name == INTERSECTION_STOP_EVENT:
                self.switch_to_state(WAITING_AT_STOPLINE) 
            elif next_event_name == INTERSECTION_TRAFFIC_LIGHT_EVENT:
                self.switch_to_state(WAITING_FOR_GREEN)
            elif next_event_name == INTERSECTION_PRIORITY_EVENT:
                self.switch_to_state(INTERSECTION_NAVIGATION)
            elif next_event_name == JUNCTION_EVENT:
                self.switch_to_state(TURNING_RIGHT)
            elif next_event_name == ROUNDABOUT_EVENT:
                self.switch_to_state(ROUNDABOUT_NAVIGATION)
            elif next_event_name == CROSSWALK_EVENT:
                self.switch_to_state(WAITING_FOR_PEDESTRIAN)
            elif next_event_name == PARKING_EVENT:
                print('WARNING: UNEXPECTED STOP LINE FOUND WITH PARKING AS NEXT EVENT')
                print('Switching to end state')
                self.car.stop()
                sleep(10.0)
                self.switch_to_state(END_STATE)

    def intersection_navigation(self):
        print('State: intersection_navigation')
        self.activate_routines([])
        if USE_LOCAL_TRACKING_FOR_INTERSECTIONS:
            self.switch_to_state(TRACKING_LOCAL_PATH)
        else:
            if self.next_event.curvature > 0.5:
                self.switch_to_state(TURNING_RIGHT)
            elif self.next_event.curvature < -0.5:
                self.switch_to_state(TURNING_LEFT)
            else:
                self.switch_to_state(GOING_STRAIGHT)
            self.car.drive_speed(self.desired_speed)
        
    def turning_right(self):
        print('State: turning_right')
        self.activate_routines([])
        if self.curr_state.just_switched:
            #detetc the stop_line to see how far we are from the line
            self.detect.detect_stop_line(self.car.frame)
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
        print('State: turning_left')
        self.activate_routines([])
        if self.curr_state.just_switched:
            #detetc the stop_line to see how far we are from the line
            self.detect.detect_stop_line(self.car.frame)
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
        print('State: going_straight')
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
            self.detect.detect_stop_line(self.car.frame)
            if self.detect.est_dist_to_stop_line < STOP_LINE_APPROACH_DISTANCE:
                d = self.detect.est_dist_to_stop_line
            else: d = 0.0
            e2, _, _ = self.detect.detect_lane(self.car.frame, SHOW_IMGS)
            # retrieve global position of the car, can be done usiing the stop_line
            # or using the gps, but for now we will use the stop_line
            # NOTE: in the real car we need to have a GLOBAL YAW OFFSET to match the simulator with the real track
            local_path_wf = self.next_event.path_ahead # wf = world frame
            stop_line_position = self.next_event.point
            # stop line frame: centered in the stop line with same orientation as world
            local_path_slf = local_path_wf - stop_line_position #slf = stop line frame
            alpha = self.car.yaw + YAW_GLOBAL_OFFSET
            rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
            #stop line frame, but now oriented in the same direction of the car
            local_path_slf_rot = np.matmul(rot_matrix, local_path_slf.T).T
            # ## get position of the car in the stop line frame
            # # car_position_slf = self.car.position - stop_line_position #with good gps estimate
            car_position_slf = np.array([+d+0.25, -e2])#np.array([+d+0.2, -e2])
            local_path_cf = local_path_slf_rot + car_position_slf #cf = car frame
            self.curr_state.var1 = local_path_cf

            print('local_path_cf_rot: ', local_path_cf[:20])
            #plot
            if SHOW_IMGS:
                img = self.car.frame.copy()
                #project the whole path (true)
                img, _ = project_onto_frame(img, self.car, self.path_planner.path, align_to_car=True, color=(0, 100, 0))
                #project local path (estimated), it should match the true path
                img, _ = project_onto_frame(img, self.car, local_path_cf, align_to_car=False)
                cv.imshow('brain_debug', img)
                cv.waitKey(1)

            self.car.reset_rel_pose()
            self.curr_state.var2 = np.array([self.car.x_true, self.car.y_true]) #var2 hold original position
            true_start_pos_wf = self.curr_state.var2
            if SHOW_IMGS:
                true_start_pos_wf_pix = m2pix(true_start_pos_wf)
                cv.circle(self.path_planner.map, (true_start_pos_wf_pix[0], true_start_pos_wf_pix[1]), 30, (255, 0, 255), 5)
                cv.imshow('Path', self.path_planner.map)
                cv.waitKey(1)
            self.curr_state.just_switched = False
            #debug
            self.car.stop()
            sleep(1.0)
            if SHOW_IMGS:
                cv.namedWindow('local_path', cv.WINDOW_NORMAL)
                local_map_img = np.zeros_like(self.path_planner.map)
                h = local_map_img.shape[0]
                w = local_map_img.shape[1]
                local_map_img[w//2-2:w//2+2, :] = 255
                local_map_img[:, h//2-2:h//2+2] = 255
                for i in range(len(local_path_cf)):
                    if (i%3 == 0):
                        p = local_path_cf[i]
                        cv.circle(local_map_img, (m2pix(p[0])+w//2, m2pix(p[1])+h//2), 10, (0, 150, 150), -1)
            cv.imshow('local_path', local_map_img)
            cv.waitKey(1)  
            self.curr_state.var3 = local_map_img
        
        D = POINT_AHEAD_DISTANCE_LOCAL_TRACKING

        #track the local path using simple pure pursuit
        path_idx = int(100*self.car.dist_loc) # m -> cm -> idx
        car_pos_loc = np.array([self.car.x_loc, self.car.y_loc])
        local_path = self.curr_state.var1 - car_pos_loc

        rot_matrix = np.array([[np.cos(self.car.yaw_loc), -np.sin(self.car.yaw_loc)], [np.sin(self.car.yaw_loc), np.cos(self.car.yaw_loc)]])
        local_path = np.matmul(rot_matrix, local_path.T).T

        if SHOW_IMGS:
            local_map_img = self.curr_state.var3
            h = local_map_img.shape[0]
            w = local_map_img.shape[1]
            angle = self.car.yaw_loc_o # + self.car.yaw_loc
            rot_matrix_w = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # show car position in the local frame (from the encoder)
            cv.circle(local_map_img, (m2pix(car_pos_loc[0])+w//2, m2pix(car_pos_loc[1])+h//2), 5, (255, 0, 255), 2)
            # show the true position to check if they match, translated wrt starting position into the local frame
            true_start_pos_wf = self.curr_state.var2
            true_pos_loc = np.array([self.car.x_true, self.car.y_true]) - true_start_pos_wf
            true_pos_loc = np.matmul(rot_matrix_w, true_pos_loc.T).T
            cv.circle(local_map_img, (m2pix(true_pos_loc[0])+w//2, m2pix(true_pos_loc[1])+h//2), 7, (0, 255, 0), 2)
            cv.imshow('local_path', local_map_img)
            true_start_pos_wf = self.curr_state.var2
            car_pos_loc_rot_wf = np.matmul(rot_matrix_w.T, car_pos_loc.T).T 
            car_pos_wf = true_start_pos_wf + car_pos_loc_rot_wf
            # show car position in wf (encoder)
            cv.circle(self.path_planner.map, (m2pix(car_pos_wf[0]), m2pix(car_pos_wf[1])), 5, (255, 0, 255), 2)
            # show the true position to check if they match
            true_pos_wf = np.array([self.car.x_true, self.car.y_true])
            cv.circle(self.path_planner.map, (m2pix(true_pos_wf[0]), m2pix(true_pos_wf[1])), 7, (0, 255, 0), 2)
            cv.imshow('Path', self.path_planner.map)
            cv.waitKey(1)
        max_idx = len(local_path)-1
        if path_idx > max_idx:
            self.switch_to_state(LANE_FOLLOWING)
            self.go_to_next_event()
        else:
            point_ahead = local_path[path_idx]
            if SHOW_IMGS:
                img = self.car.frame.copy()
                img, _ = project_onto_frame(img, self.car, local_path, align_to_car=False)
                img, _ = project_onto_frame(img, self.car, point_ahead, align_to_car=False, color=(0,0,255))
                cv.imshow('brain_debug', img)
                cv.waitKey(1)
            yaw_error = np.arctan2(point_ahead[1], point_ahead[0]) 
            out_speed, out_angle = self.controller.get_control(0.0, yaw_error, 0.0, self.desired_speed)
            self.car.drive(out_speed, np.rad2deg(out_angle))


    def roundabout_navigation(self):
        print('State: roundabout_navigation')
        self.switch_to_state(TRACKING_LOCAL_PATH)

    def waiting_for_pedestrian(self):
        print('State: waiting_for_pedestrian')
        self.switch_to_state(LANE_FOLLOWING) #temporary
        self.go_to_next_event() #temporary
        pass

    def waiting_for_green(self):
        print('State: waiting_for_green')
        #temporary fix
        self.switch_to_state(WAITING_AT_STOPLINE)

    def waiting_at_stopline(self):
        print('State: waiting_at_stopline')
        self.activate_routines([]) #no routines
        self.car.stop()
        if (time() - self.curr_state.start_time) > STOP_WAIT_TIME:
            self.switch_to_state(INTERSECTION_NAVIGATION)

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
        self.curr_state.run()

    def run_routines(self):
        for k, r in self.routines.items():
            if r.active: r.run()

    def activate_routines(self, routines_to_activate):
        """
        routines_to_activate are a list of strings (routines)
        ex: ['follow_lane', 'control_for_signs']
        """
        for k,r in self.routines.items():
            r.active = k in routines_to_activate
    
    def switch_to_state(self, to_state, interrupt=False):
        """
        to_state is the string of the desired state to switch to
        ex: 'lane_following'
        """
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
        if self.event_idx == (len(self.events)-1):
            #no more events, go to end_state
            self.switch_to_state(END_STATE)
        else:
            self.event_idx += 1
            self.prev_event = self.next_event
            self.next_event = self.events[self.event_idx]

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
            path_ahead = e[3]
            if path_ahead is not None:
                curv = get_curvature(path_ahead)
                print(f'curv: {curv}')
                len_path_ahead = 0.01*len(path_ahead)
            else:
                curv = None
                len_path_ahead = None
            event = Event(name, dist, point, path_ahead, len_path_ahead, curv)
            to_ret.append(event)
        return to_ret

