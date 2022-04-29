#!/usr/bin/python3

SIMULATOR = True

# Functional libraries
import rospy
import numpy as np
import json
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

from std_msgs.msg import Byte
from utils.msg import environmental
from utils.msg import vehicles

if SIMULATOR:
    from helper_functions import *
else:
    from control.helper_functions import *

VEHICLE_CLOSE_RADIUS = 0.5  # [m]


class EnvironmentalData():
    def __init__(self,
                 trig_v2v = False,
                 trig_v2x = False,
                 trig_semaphore = False
                ) -> None:
        """Manage flow of data with the environmental server

        :param trig_control: trigger on commands, defaults to True
        :type trig_control: bool, optional
        """        
        # ==================== VARIABLES ====================
        # VEHICLE-TO-VEHICLE (V2V)
        self.other_vehicles = {}
        # VEHICLE-TO-EVERYTHING (V2X)
        self.obstacle_list = []
        self.obstacle_map = {
                                'STOP':                     1,
                                'PRIORITY':                 2,
                                'PARKING':                  3,
                                'CROSSWALK':                4,
                                'HIGHWAYENTER':             5,
                                'HIGHWAYEXIT':              6,
                                'ROUNDABOUT':               7,
                                'ONEWAY':                   8,
                                'TRAFFICLIGHT':             9,
                                'STATICCAR_ONROAD':         10,
                                'STATICCAR_PARKING':        11,
                                'PEDESTRIAN_ONCROSSWALK':   12,
                                'PEDESTRIAN_ONROAD':        13,
                                'ROADBLOCK':                14,
                                'BUMPYROAD':                15
                            }
        # SEMAPHORE
        self.semaphore_states = {'master':0, 'slave':0, 'antimaster':0, 'start':0}# can be changed to 1,2,3,4
                
        # ==================== SUBSCRIBERS AND PUBLISHERS ====================
        # VEHICLE-TO-VEHICLE (V2V)
        if trig_v2v:
            self.sub_v2v = rospy.Subscriber('/automobile/vehicles', vehicles, self.v2v_callback)
        # VEHICLE-TO-EVERYTHING (V2X)
        if trig_v2x:
            self.pub_v2x = rospy.Publisher('/automobile/environment', environmental, queue_size=1)
        # SEMAPHORE
        if trig_semaphore:
            if SIMULATOR:
                self.sub_semaphoremaster= rospy.Subscriber("/automobile/trafficlight/master", Byte, self.semaphore_master_callback)
                self.sub_semaphoreslave = rospy.Subscriber("/automobile/trafficlight/slave", Byte, self.semaphore_slave_callback)
                self.sub_semaphoreantimaster = rospy.Subscriber("/automobile/trafficlight/antimaster", Byte, self.semaphore_antimaster_callback)
                self.sub_semaphorestart = rospy.Subscriber("/automobile/trafficlight/start", Byte, self.semaphore_start_callback)
            else:
                self.sub_semaphoremaster= rospy.Subscriber("/automobile/semaphore/master", Byte, self.semaphore_master_callback)
                self.sub_semaphoreslave = rospy.Subscriber("/automobile/semaphore/slave", Byte, self.semaphore_slave_callback)
                self.sub_semaphoreantimaster = rospy.Subscriber("/automobile/semaphore/antimaster", Byte, self.semaphore_antimaster_callback)
                self.sub_semaphorestart = rospy.Subscriber("/automobile/semaphore/start", Byte, self.semaphore_start_callback)

        # I/O interface
        # rospy.init_node('environmental_data', anonymous=False)

    # V2V CALLBACKS AND FUNCTIONS
    def v2v_callback(self, data) -> None:
        """Receive and store positions of other moving vehicles
        :acts on: self.other_vehicles
        """
        ID = data.ID
        self.other_vehicles[ID] = np.array([data.posA, data.posB])
    
    def get_closest_moving_vehicle(self, car_x, car_y):
        """Gives the ID and position of the closest vehicle

        :param car_x: [m] x coordinate of the car
        :type car_x: float
        :param car_y: [m] y coordinate of the car
        :type car_y: float
        :return: ID and position of the closest vehicle if there are, else -1 and the car position
        :rtype: (int, nd-array)
        """        
        if self.other_vehicles:
            ID, pos = min(self.other_vehicles.items(), key=lambda x: np.linalg.norm(np.array([car_x, car_y]) - x[1]))
            return ID, pos
        else:
            # dictionary is empty
            rospy.loginfo("No moving vehicles detected")
            return -1, np.array([car_x, car_y])
    
    def is_other_vehicle_close(self, car_x, car_y):
        """checks if there are moving vehicles near the car

        :param car_x: [m] x coordinate of the car
        :type car_x: float
        :param car_y: [m] y coordinate of the car
        :type car_y: float
        :return: True if there are moving vehicles near the car, else False
        :rtype: bool
        """        
        ID, pos = self.get_closest_moving_vehicle(car_x, car_y)
        if ID < 0:
            return False
        else:
            return np.linalg.norm(np.array([car_x, car_y]) - pos) < VEHICLE_CLOSE_RADIUS

    # V2X CALLBACKS AND FUNCTIONS
    def publish_obstacle(self, type, x, y):
        data = environmental()
        data.obstacle_id    = self.obstacle_map[type]
        data.x              = x
        data.y              = y
        self.pub_v2x.publish(data)
        self.obstacle_list.append(f'{type} found at position ({x},{y}) m')
    
    # SEMAPHORE CALLBACKS AND FUNCTIONS
    def semaphore_master_callback(self, data):
        self.semaphore_states['master'] = data.data

    def semaphore_slave_callback(self, data):
        self.semaphore_states['slave'] = data.data
    
    def semaphore_antimaster_callback(self, data):
        self.semaphore_states['antimaster'] = data.data
    
    def semaphore_start_callback(self, data):
        self.semaphore_states['start'] = data.data
    
    def get_semaphore_state(self, semaphore_key):
        """
        :param semaphore_key: key of the semaphore
        :type semaphore_key: string
        :return: state of the semaphore: 0:RED, 1:YELLOW, 2:GREEN
        :rtype: byte
        """             
        return self.semaphore_states[semaphore_key]

    
    # STATIC METHODS
    

    def __str__(self):
        description = '''
{:#^65s} 
other_vehicles:
{:s}
{:#^65s} 
obstacle_list:
{:s}
{:#^65s} 
semaphore_states:
{:s}
'''
        return description.format(
                                    ' V2V ', json.dumps(self.other_vehicles, sort_keys=True, indent=2, cls=NumpyEncoder),\
                                    ' V2X ', str(self.obstacle_list),\
                                    ' SEMAPHORES ', json.dumps(self.semaphore_states, sort_keys=True, indent=1)
                                 )
                                    



    









