#!/usr/bin/python3

from curses.ascii import BS
from os import path
from re import S, X
from time import sleep
import copy
from cv2 import cubeRoot
import networkx as nx
import numpy as np
import cv2 as cv
import random
from pyclothoids import Clothoid
import math
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline

from helper_functions import *

class PathPlanning(): 
    def __init__(self, map_img):
        # start and end nodes
        self.source = str(86)
        self.target = str(300)

        # initialize path
        self.path = []
        self.navigator = []# set of instruction for the navigator
        self.path_data = [] #additional data for the path (e.g. curvature, 
        self.step_length = 0.03
        # state=following/intersection/roundabout, next_action=left/right/straight, path_params_ahead)

        # previous index of the closest point on the path to the vehicle
        self.prev_index = 0

        # read graph
        self.G = nx.read_graphml('random_stuff/Competition_track.graphml')
        # initialize route subgraph and list for interpolation
        self.route_graph = nx.DiGraph()
        self.route_list = []
        self.old_nearest_point_index = None # used in search target point

        # define intersection central nodes
        #self.intersection_cen = ['347', '37', '39', '38', '11', '29', '30', '28', '371', '84', '9', '20', '20', '82', '75', '74', '83', '312', '315', '65', '468', '10', '64', '57', '56', '66', '73', '424', '48', '47', '46']
        self.intersection_cen = ['37', '39', '38', '11', '29', '30', '28', '371', '84', '9','19', '20', '21', '82', '75', '74', '83', '312', '315', '65', '10', '64', '57', '56', '66', '73', '424', '48', '47', '46']

        # define intersecion entrance nodes
        self.intersection_in = [77,45,54,50,41,79,374,52,43,81,36,4,68,2,34,70,6,32,72,59,15,16,27,14,25,18,61,23,63]
        self.intersection_in = [str(i) for i in self.intersection_in]

        # define intersecion exit nodes
        self.intersection_out = [76,78,80,40,42,44,49,51,53,67,69,71,1,3,5,7,8,31,33,35,22,24,26,13,17,58,60,62]
        self.intersection_out = [str(i) for i in self.intersection_out]

        # define left turns tuples
        #self.left_turns = [(45,42),(43,40),(54,51),(36,33),(34,31),(6,3),(2,7),(2,8),(4,1),(70,67),(72,69),(27,24),(25,22),(16,13),(61,58),(63,60)]

        #define crosswalks 
        self.crosswalk = [92,96,276,295,170,266,177,162] #note: parking spots are included too (177,162)
        self.crosswalk = [str(i) for i in self.crosswalk]

        # define roundabout nodes
        self.ra = [267,268,269,270,271,302,303,304,305,306,307]
        self.ra = [str(i) for i in self.ra]

        self.ra_enter = [301,342,230] #[267,302,305]
        self.ra_enter = [str(i) for i in self.ra_enter]

        self.ra_exit = [272,231,343] # [271,304,306]
        self.ra_exit = [str(i) for i in self.ra_exit]

        self.junctions = [467,314]
        self.junctions = [str(i) for i in self.junctions]

        #stop points
        self.stop_points = np.load('models/stop_points.npy')

        # import nodes and edges
        self.list_of_nodes = list(self.G.nodes)
        self.list_of_edges = list(self.G.edges)

        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        # import map to plot trajectory and car
        self.map = map_img

    def roundabout_navigation(self, prev_node, curr_node, next_node):
        while next_node in self.ra:
            prev_node = curr_node
            curr_node = next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                next_node = None
                break
                
            #print("inside roundabout: ", curr_node)
            
            xp,yp = self.get_coord(prev_node)
            xc,yc = self.get_coord(curr_node)
            xn,yn = self.get_coord(next_node)

            if curr_node in self.ra_enter:
                if not(prev_node in self.ra):
                    dx = xn - xp
                    dy = yn - yp
                    self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
                else:
                    if curr_node == '302':
                        continue
                    else:
                        dx = xn - xp
                        dy = yn - yp
                        self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
            elif curr_node in self.ra_exit:
                if next_node in self.ra:
                    # remain inside roundabout
                    if curr_node == '271':
                        continue
                    else:
                        dx = xn - xp
                        dy = yn - yp
                        self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
                else:
                    dx = xn - xp
                    dy = yn - yp
                    self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
            else:
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))

        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None
        
        self.navigator.append("exit roundabout at "+curr_node)
        self.navigator.append("go straight")
        return prev_node, curr_node, next_node
    
    def intersection_navigation(self, prev_node, curr_node, next_node):
        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None
            return prev_node, curr_node, next_node
        
        prev_node = curr_node
        curr_node = next_node
        if curr_node != self.target:
            next_node = list(self.route_graph.successors(curr_node))[0]
        else:
            next_node = None
        
        self.navigator.append("exit intersection at " + curr_node)
        self.navigator.append("go straight")
        return prev_node, curr_node, next_node
    
    def compute_route_list(self):
        ''' Augments the route stored in self.route_graph'''
        #print("source=",source)
        #print("target=",target)
        #print("edges=",self.route_graph.edges.data())
        #print("nodes=",self.route_graph.nodes.data())
        #print(self.route_graph)
        
        curr_node = self.source
        prev_node = curr_node
        next_node = list(self.route_graph.successors(curr_node))[0]

        #reset route list
        self.route_list = []

        #print("curr_node",curr_node)
        #print("prev_node",prev_node)
        #print("next_node",next_node)
        self.navigator.append("go straight")
        while curr_node != self.target:
            xp,yp = self.get_coord(prev_node)
            xc,yc = self.get_coord(curr_node)
            xn,yn = self.get_coord(next_node)

            #print("from",curr_node,"(",xc,yc,")"," to ",next_node,"(",xn,yn,")")
            #print("edge: ",self.route_graph.get_edge_data(curr_node, next_node))
            #print("prev_node is",prev_node,"(",xp,yp,")")
            #print("*************************\n")
            
            curr_is_junction = curr_node in self.intersection_cen#len(adj_nodes) > 1
            next_is_intersection = next_node in self.intersection_cen#len(next_adj_nodes) > 1
            prev_is_intersection = prev_node in self.intersection_cen#len(prev_adj_nodes) > 1
            next_is_roundabout_enter = next_node in self.ra_enter
            curr_in_roundabout = curr_node in self.ra
            #print(f"CURR: {curr_node}, NEXT: {next_node}, PREV: {prev_node}")
            # ****** ROUNDABOUT NAVIGATION ******
            if next_is_roundabout_enter:
                if curr_node == "342":
                    self.route_list.append((xc,yc,math.degrees(np.arctan2(-1,0))))
                    dx = xc - xp
                    dy = yc - yp
                    self.route_list.append((xc,yc+0.3*dy,math.degrees(np.arctan2(-1,0))))
                else:
                    dx = xc-xp
                    dy = yc-yp
                    self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
                    # add a further node
                    dx = xc - xp
                    dy = yc - yp
                    self.route_list.append((xc+0.3*dx,yc+0.3*dy,math.degrees(np.arctan2(dy,dx))))
                # enter the roundabout
                self.navigator.append("enter roundabout at " + curr_node)
                prev_node, curr_node, next_node = self.roundabout_navigation(prev_node, curr_node, next_node)
                continue               
            elif next_is_intersection:
                dx = xc - xp
                dy = yc - yp
                self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
                # add a further node
                dx = xc - xp
                dy = yc - yp
                self.route_list.append((xc+0.3*dx,yc+0.3*dy,math.degrees(np.arctan2(dy,dx))))
                # enter the intersection
                self.navigator.append("enter intersection at " + curr_node)
                prev_node, curr_node, next_node = self.intersection_navigation(prev_node, curr_node, next_node)
                continue
            elif prev_is_intersection:
                # add a further node
                dx = xn - xc
                dy = yn - yc
                self.route_list.append((xc-0.3*dx,yc-0.3*dy,math.degrees(np.arctan2(dy,dx))))
                # and add the exit node
                dx = xn - xc
                dy = yn - yc
                self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
            else:
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xc,yc,math.degrees(np.arctan2(dy,dx))))
            
            prev_node = curr_node
            curr_node = next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                # You arrived at the END
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xn,yn,math.degrees(np.arctan2(dy,dx))))
                next_node = None
        
        self.navigator.append("stop")

    def compute_shortest_path(self, source=None, target=None, step_length=0.01):
        ''' Generates the shortest path between source and target nodes using Clothoid interpolation '''
        src = str(source) if source is not None else self.source
        tgt = str(target) if target is not None else self.target
        self.source = src
        self.target = tgt
        self.step_length = step_length
        route_nx = []

        # generate the shortest route between source and target      
        route_nx = list(nx.shortest_path(self.G, source=src, target=tgt)) 
        # generate a route subgraph       
        self.route_graph = nx.DiGraph() #reset the graph
        self.route_graph.add_nodes_from(route_nx)
        for i in range(len(route_nx)-1):
            self.route_graph.add_edges_from( [ (route_nx[i], route_nx[i+1], self.G.get_edge_data(route_nx[i],route_nx[i+1])) ] )         # augment route with intersections and roundabouts
        # expand route node and update navigation instructions
        self.compute_route_list()
        
        #self.print_navigation_instructions()
        # interpolate the route of nodes
        self.path = PathPlanning.interpolate_route(self.route_list, step_length)

        # self.augment_path()
        return self.path
    
    def generate_path_passing_through(self, list_of_nodes, step_length=0.01):
        """
        Extend the path generation from source-target to a sequence of nodes/locations
        """
        assert len(list_of_nodes) >= 2, "List of nodes must have at least 2 nodes"
        print("Generating path passing through: ", list_of_nodes)
        src = list_of_nodes[0]
        tgt = list_of_nodes[1]
        complete_path = self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
        for i in range(1,len(list_of_nodes)-1):
            src = list_of_nodes[i]
            tgt = list_of_nodes[i+1]
            self.compute_shortest_path(source=src, target=tgt, step_length=step_length)
            #now the local path is computed in self.path
            #remove first element of self.path
            self.path = self.path[1:]
            #we need to add the local path to the complete path
            complete_path = np.concatenate((complete_path, self.path))
        self.path = complete_path
        self.augment_path()


    def search_target_index(self, car):
        cx = self.path[:,0]
        cy = self.path[:,1]

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [car.rear_x - icx for icx in list(cx)]
            dy = [car.rear_y - icy for icy in list(cy)]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = car.calc_distance(cx[ind],cy[ind])
            while True:
                distance_next_index = car.calc_distance(cx[ind],cy[ind])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = 0.5# [m] #0.1 * car.speed  # update look ahead distance

        # search look ahead target point index
        while Lf > car.calc_distance(cx[ind], cy[ind]):
            if (ind + 1) >= len(cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf

    def get_reference(self, car, v_des, limit_search_to=50, look_ahead=4, frame=None, n=3, training=True): 
        '''
        Returns the reference point, i.e., the point on the path nearest to the vehicle, 
        returns
        x desired
        y desired
        yaw desired
        TODO current curvature
        finished=True when the end of the path is reached
        path paramters ahead
        '''
        prev_pos = self.path[self.prev_index]
        x_prev = self.path[self.prev_index, 0]
        y_prev = self.path[self.prev_index, 1]
        finished = False
        curv = 0.0

        #limit the search in a neighborhood of limit_search_to points around the current point
        max_index = min(self.prev_index + limit_search_to, len(self.path)-2)
        if max_index >= len(self.path)-10:
            finished = True
        min_index = max(self.prev_index, 0)
        path_to_analyze = self.path[min_index:max_index+1, :]
        #get current car position
        curr_pos = np.array([car.x_true, car.y_true])
        #calculate the norm difference of each point to the current point
        diff = path_to_analyze - curr_pos
        diff_norms = np.linalg.norm(diff, axis=1)
        #find the index of the point closest to the current point
        closest_index = np.argmin(diff_norms)
        #check if it's last element
        if closest_index == len(path_to_analyze)-1:
            closest_index = closest_index - (2+look_ahead)
        closest_index = closest_index + look_ahead
        closest_point = path_to_analyze[closest_index]
        #get path ahead
        x_des = closest_point[0]
        y_des = closest_point[1]
        #calculate yaw angle
        next_x = path_to_analyze[closest_index+1, 0]
        next_y = path_to_analyze[closest_index+1, 1]
        yaw_des = - np.arctan2(next_y - y_des, next_x - x_des)
        #update prev_index
        self.prev_index = min_index + closest_index - (1+look_ahead)

        path_ahead = self.get_path_ahead(min_index+closest_index, limit_search_to)
        
        avg_curv = get_curvature(path_ahead, v_des)

        info = self.path_data[min_index+closest_index]

        return x_des, y_des, yaw_des, avg_curv, finished, path_ahead, info

    def get_path_ahead(self, index, look_ahead=100):
        assert index < len(self.path) and index >= 0
        return np.array(self.path[index:min(index + look_ahead, len(self.path)-1), :])

    def get_closest_stop_point(self, nx, ny):
        """
        Returns the closest stop point to the given point
        """
        index_closest = np.argmin(np.hypot(nx - self.stop_points[:,0], ny - self.stop_points[:,1]))
        print(f'Closest stop point is {self.stop_points[index_closest, :]}, Point is {nx, ny}')
        #draw a circle around the closest stop point
        cv.circle(self.map, (m2pix(self.stop_points[index_closest, 0]), m2pix(self.stop_points[index_closest, 1])), 8, (0, 255, 0), 4)
        return self.stop_points[index_closest, 0], self.stop_points[index_closest, 1]

    def augment_path(self, default_distance=100, tol=0.01):
        print("Augmenting path...")
        #intersection ins
        intersection_in_indexes = []
        intersection_in_pos = []
        intersection_in_yaw = []
        for i in range(len(self.path)-1):
            curr_x = self.path[i,0]
            curr_y = self.path[i,1]
            for n in self.intersection_in:
                nx, ny = self.get_coord(n)
                if (nx-curr_x)**2 + (ny-curr_y)**2 <= tol**2:
                    nx, ny = self.get_closest_stop_point(nx, ny)
                    intersection_in_indexes.append(i) 
                    intersection_in_pos.append((nx, ny))
                    yaw = - np.arctan2(ny-self.path[i+1,1], nx-self.path[i+1,0])
                    intersection_in_yaw.append(yaw)
                    break 
        # print("intersection_in_indexes: ", intersection_in_indexes) 
        #intersection outs
        intersection_out_indexes = []
        intersection_out_pos = []
        intersection_out_yaw = []
        for i in range(len(self.path)-1):
            curr_x = self.path[i,0]
            curr_y = self.path[i,1]
            for n in self.intersection_out:
                nx, ny = self.get_coord(n)
                if (nx-curr_x)**2 + (ny-curr_y)**2 <= tol**2:
                    intersection_out_indexes.append(i) 
                    intersection_out_pos.append((nx, ny))
                    yaw = - np.arctan2(ny-self.path[i+1,1], nx-self.path[i+1,0])
                    intersection_out_yaw.append(yaw)
                    break   
        # print("intersection_out_indexes: ", intersection_out_indexes) 
        #roundabouts in
        roundabout_in_indexes = []
        roundabout_in_pos = []
        roundabout_in_yaw = []
        for i in range(len(self.path)-1):
            curr_x = self.path[i,0]
            curr_y = self.path[i,1]
            for n in self.ra_enter:
                nx, ny = self.get_coord(n)
                #and = workaround, ra exits and entrance must be outside the ra
                if (nx-curr_x)**2 + (ny-curr_y)**2 <= tol**2: # and len(roundabout_in_indexes) < 1: #and = workaround 
                    nx, ny = self.get_closest_stop_point(nx, ny)
                    roundabout_in_indexes.append(i) 
                    roundabout_in_pos.append((nx, ny))
                    yaw = - np.arctan2(ny-self.path[i+1,1], nx-self.path[i+1,0])
                    roundabout_in_yaw.append(yaw)
                    break
        # print("roundabout_in_indexes: ", roundabout_in_indexes)
        #roundabouts out
        roundabout_out_indexes = []
        roundabout_out_pos = []
        roundabout_out_yaw = []
        for i in range(len(self.path)-1):
        # for i in range(len(self.path)-2, 0, -1): # REVERSE ORDER, PART OF WORKAROUND
            curr_x = self.path[i,0]
            curr_y = self.path[i,1]
            for n in self.ra_exit:
                nx, ny = self.get_coord(n)
                #and = workaround, ra exits and entrance must be outside the ra
                if (nx-curr_x)**2 + (ny-curr_y)**2 <= tol**2: #and len(roundabout_out_indexes) < 1:
                    roundabout_out_indexes.append(i) 
                    roundabout_out_pos.append((nx, ny))
                    yaw = - np.arctan2(curr_y-self.path[i+1,1], curr_x-self.path[i+1,0])
                    roundabout_out_yaw.append(yaw)
                    break
        # print("roundabout_out_indexes: ", roundabout_out_indexes)
        #junctions
        junction_indexes = []
        junction_pos = []
        junction_yaw = []
        for i in range(len(self.path)-1):
            curr_x = self.path[i,0]
            curr_y = self.path[i,1]
            for n in self.junctions:
                nx, ny = self.get_coord(n)
                if (nx-curr_x)**2 + (ny-curr_y)**2 <= tol**2:
                    junction_indexes.append(i) 
                    junction_pos.append((nx, ny))
                    yaw = - np.arctan2(ny-self.path[i+1,1], nx-self.path[i+1,0])
                    junction_yaw.append(yaw)
                    break 

        #crosswalks
        crosswalk_indexes = []
        crosswalk_pos = []
        crosswalk_yaw = []
        for i in range(len(self.path)-1):
            curr_x = self.path[i,0]
            curr_y = self.path[i,1]
            for n in self.crosswalk:
                nx, ny = self.get_coord(n)
                if (nx-curr_x)**2 + (ny-curr_y)**2 <= tol**2:
                    nx, ny = self.get_closest_stop_point(nx, ny)
                    crosswalk_indexes.append(i) 
                    crosswalk_pos.append((nx, ny))
                    yaw = - np.arctan2(ny-self.path[i+1,1], nx-self.path[i+1,0])
                    crosswalk_yaw.append(yaw)
                    break 

        self.path_data = [None for i in range(len(self.path))]
        # self.path_data[0] = ['start', None, None, self.path[0,0], self.path[0,1], 0.0]
        for i, (x,y), yaw in zip(intersection_in_indexes, intersection_in_pos, intersection_in_yaw):
            self.path_data[i] = ('int_in', None, None, x, y, yaw, None, None)
        for i, (x,y), yaw in zip(intersection_out_indexes, intersection_out_pos, intersection_out_yaw):
            self.path_data[i] = ('int_out', None, None, x, y, yaw, None, None)
        for i, (x,y), yaw in zip(roundabout_in_indexes, roundabout_in_pos, roundabout_in_yaw):
            self.path_data[i] = ('rou_in', None, None, x, y, yaw, None, None)
        for i, (x,y), yaw in zip(roundabout_out_indexes, roundabout_out_pos, roundabout_out_yaw):
            self.path_data[i] = ('rou_out', None, None, x, y, yaw, None, None)
        for i, (x,y), yaw in zip(junction_indexes, junction_pos, junction_yaw):
            self.path_data[i] = ('junction', None, None, x, y, yaw, None, None)
        for i, (x,y), yaw in zip(crosswalk_indexes, crosswalk_pos, crosswalk_yaw):
            self.path_data[i] = ('crosswalk', None, None, x, y, yaw, None, None)
        
        # print("path_data: ", self.path_data)
        
        curr_index , next_index = 0, 1
        prev_event, next_event = 'start', None
        while curr_index < len(self.path):
            #search next event ahead
            # print(f'Searching for next event ahead of {curr_index}, prev event: {prev_event}')
            while next_index < len(self.path):
                if self.path_data[next_index] is None:
                    next_index += 1
                    next_event = None
                else:
                    next_event = self.path_data[next_index][0]
                    # print(f'Found next event: {next_event}, curr_index: {curr_index}, next_index: {next_index}')
                    break
            
            #cases:
            # state=following/intersection/roundabout, next_action=left/right/straight, path_params_ahead)
            #1. no event ahead
            if next_event is None:
                self.path_data[curr_index] = ('road', 'road',  'continue', None, None, None, None)
                #continue previous event
                while curr_index < len(self.path):
                    if self.path_data[curr_index] is None:
                        if prev_event == 'int_in':
                            print('ERROR: Path cannot end inside an intersection')
                        elif prev_event == 'int_out':
                            self.path_data[curr_index] = ('road', 'road', 'continue', None, None, None, None)
                        elif prev_event == 'rou_in':
                            print('ERROR: Path cannot end inside a roundabout')
                        elif prev_event == 'rou_out':
                            self.path_data[curr_index] = ('road', 'road',  'continue', None, None, None, None)
                        elif prev_event == 'junction':
                            self.path_data[curr_index] = ('road', 'road',  'continue', None, None, None, None)
                        elif prev_event == 'crosswalk':
                            self.path_data[curr_index] = ('road', 'road',  'continue', None, None, None, None)
                        else:
                            print('ERROR: Path cannot end at start')
                            cv.waitKey(0)
                    curr_index += 1
                print('Done')
                sleep(1)

            #2. inside intersection
            elif prev_event == 'int_in' and next_event == 'int_out':
                # print('INSIDE INTERSECTION')
                #continue intersection
                angle = diff_angle(self.path_data[curr_index][5], self.path_data[next_index][5])
                action = 'right' if angle > np.pi/4 else 'left' if angle < -np.pi/4 else 'straight'
                while curr_index < next_index:
                    self.path_data[curr_index] = ('intersection', 'road', action, None, None, None, None)
                    curr_index += 1
            #3. inside roundabout
            elif prev_event == 'rou_in' and next_event == 'rou_out':
                # print('INSIDE ROUNDABOUT')
                #continue roundabout
                angle = diff_angle(self.path_data[curr_index][5], self.path_data[next_index][5])
                action = 'right' if angle > np.pi/4 else 'left' if angle < -np.pi/4 else 'straight'
                while curr_index < next_index:
                    self.path_data[curr_index] = ('roundabout', 'road', action, None, None, None, None)
                    curr_index += 1
            #4. intersection ahead
            elif next_event == 'int_in':
                # print('INTERSECTION AHEAD')
                xi, yi = self.path_data[next_index][3], self.path_data[next_index][4] #intersection coordinates
                #search ahead for intersection out
                yaw_out = 0.0
                for i in range(next_index+1, len(self.path)):
                    if self.path_data[i] is not None and self.path_data[i][0] == 'int_out':
                        yaw_out = self.path_data[i][5]
                        break
                yaw_in = self.path_data[next_index][5]
                angle = diff_angle(yaw_in, yaw_out)
                action = 'right' if angle > np.pi/4 else 'left' if angle < -np.pi/4 else 'straight'
                # print(f'curr_index: {curr_index}, next_index: {next_index}')
                while curr_index < next_index:
                    if next_index-curr_index < default_distance:
                        euclidean_distance = -0.2 + np.sqrt((xi-self.path[curr_index,0])**2 + (yi-self.path[curr_index,1])**2)
                        self.path_data[curr_index] = ('road', 'intersection', action, euclidean_distance, None, None, None)
                    else:
                        self.path_data[curr_index] = ('road', 'road', 'continue', None, None, None, None)
                    curr_index += 1
                # print(f'curr_index: {curr_index}, next_index: {next_index}')
            #5. roundabout ahead
            elif next_event == 'rou_in':
                # print('ROUNDABOUT AHEAD')
                xi, yi = self.path_data[next_index][3], self.path_data[next_index][4] #roundabout coordinates
                #search ahead for roundabout out
                yaw_out = 0.0
                for i in range(next_index+1, len(self.path)):
                    if self.path_data[i] is not None and self.path_data[i][0] == 'rou_out':
                        yaw_out = self.path_data[i][5]
                        break
                yaw_in = self.path_data[next_index][5]
                angle = diff_angle(yaw_in, yaw_out)
                action = 'right' if angle > np.pi/4 else 'left' if angle < -np.pi/4 else 'straight'
                while curr_index < next_index:
                    if next_index-curr_index < default_distance:
                        euclidean_distance = -0.2 + np.sqrt((xi-self.path[curr_index,0])**2 + (yi-self.path[curr_index,1])**2)
                        self.path_data[curr_index] = ('road', 'roundabout', action, euclidean_distance, None, None, None)
                    else:
                        self.path_data[curr_index] = ('road', 'road', 'continue', None, None, None, None)
                    curr_index += 1
            #6. junction ahead
            elif next_event == 'junction':
                # print('JUNCTION AHEAD')
                xi, yi = self.path_data[next_index][3], self.path_data[next_index][4] #junction coordinates
                # assert np.isclose(xi, self.path[next_index,0]) and np.isclose(yi, self.path[next_index,1])
                yaw_junction = self.path_data[next_index][5]
                #find yaw ahead 
                idx_ahead = min(next_index+int(50*(0.03/self.step_length)), len(self.path)-2) 
                dx = + self.path[idx_ahead,0] - self.path[idx_ahead+1,0]
                dy = + self.path[idx_ahead,1] - self.path[idx_ahead+1,1]
                yaw_ahead = -np.arctan2(dy, dx)
                angle = diff_angle(yaw_junction, yaw_ahead)
                print(f'angle: {angle}')
                # print(f'angle = {angle}')
                action = 'right' if angle > 0.004 else 'left' if angle < -0.60 else 'straight'
                # print(f'curr_index: {curr_index}, next_index: {next_index}')
                while curr_index < next_index:
                    if next_index-curr_index < default_distance:
                        euclidean_distance = -0.2 + np.sqrt((xi-self.path[curr_index,0])**2 + (yi-self.path[curr_index,1])**2)
                        self.path_data[curr_index] = ('road', 'junction', action, euclidean_distance, None, None, None)
                    else:
                        self.path_data[curr_index] = ('road', 'road', 'continue', None, None, None, None)
                    curr_index += 1
                #continue junction for a while
                while curr_index < next_index+60: # with path_step_length = 0.01 [m]
                    self.path_data[curr_index] = ('road', 'junction', action, None, None, None, None)
                    curr_index += 1
                next_index = curr_index
                # print(f'curr_index: {curr_index}, next_index: {next_index}')
            #7. Crosswalk ahead
            elif next_event == 'crosswalk':
                # print('CROSSWALK AHEAD')
                xi, yi = self.path_data[next_index][3], self.path_data[next_index][4] #junction coordinates
                # assert np.isclose(xi, self.path[next_index,0]) and np.isclose(yi, self.path[next_index,1])
                yaw_crosswalk = self.path_data[next_index][5]
                #find yaw ahead 
                idx_ahead = min(next_index+int(50*(0.03/self.step_length)), len(self.path)-2) 
                dx = + self.path[idx_ahead,0] - self.path[idx_ahead+1,0]
                dy = + self.path[idx_ahead,1] - self.path[idx_ahead+1,1]
                yaw_ahead = -np.arctan2(dy, dx)
                angle = diff_angle(yaw_crosswalk, yaw_ahead)
                print(f'angle: {angle}')
                # print(f'angle = {angle}')
                action = 'right' if angle > 0.004 else 'left' if angle < -0.60 else 'straight'
                # print(f'curr_index: {curr_index}, next_index: {next_index}')
                while curr_index < next_index:
                    if next_index-curr_index < default_distance:
                        euclidean_distance = -0.2 + np.sqrt((xi-self.path[curr_index,0])**2 + (yi-self.path[curr_index,1])**2)
                        self.path_data[curr_index] = ('road', 'crosswalk', action, euclidean_distance, None, None, None)
                    else:
                        self.path_data[curr_index] = ('road', 'road', 'continue', None, None, None, None)
                    curr_index += 1
                # #continue junction for a while
                # while curr_index < next_index+60: # with path_step_length = 0.01 [m]
                #     self.path_data[curr_index] = ('road', 'crosswalk', action, None, None, None, None)
                #     curr_index += 1
                # next_index = curr_index
                # print(f'curr_index: {curr_index}, next_index: {next_index}')
            #8. Error case intersection 
            elif prev_event != 'int_in' and next_event == 'int_out':
                print('Error case intersection')
                cv.waitKey(0)
            #9. Error case roundabout
            elif prev_event != 'rou_in' and next_event == 'rou_out':
                print('Error case roundabout')
                cv.waitKey(0)
            #10. OTHER CASES
            else:
                print('OTHER CASES')
                cv.waitKey(0)

            #update prev_event
            next_index += 1
            # print(f'STATUS: curr_index: {curr_index}, next_index: {next_index}\nPrev_event: {prev_event},  Next_event: {next_event}')
            prev_event = next_event


    def print_path_info(self):
        prev_state = None
        prev_next_state = None
        for i in range(len(self.path_data)-1):
            curr_state = self.path_data[i][0]
            next_state = self.path_data[i][1]
            if curr_state != prev_state or next_state != prev_next_state:
                print(f'{i}: {self.path_data[i]}')
            prev_state = curr_state
            prev_next_state = next_state

    def get_length(self, path=None):
        ''' Calculates the length of the trajectory '''
        if path is None:
            length = 0
            for i in range(len(self.path)-1):
                x1,y1 = self.path[i]
                x2,y2 = self.path[i+1]
                length += math.hypot(x2-x1,y2-y1)
            
            #print(f"Length of the trajectory: {length}")  
            return length
        else:
            length = 0
            for i in range(len(path)-1):
                x1,y1 = path[i]
                x2,y2 = path[i+1]
                length += math.hypot(x2-x1,y2-y1) 
            return length

    def get_coord(self, node):
        x = self.nodes_data[node]['x']
        y = self.nodes_data[node]['y']
        return x, y
    
    def get_path(self):
        return self.path
    
    def print_navigation_instructions(self):
        for i,instruction in enumerate(self.navigator):
            print(i+1,") ",instruction)
    
    @staticmethod
    def compute_path(xi, yi, thi, xf, yf, thf, step_length):
        clothoid_path = Clothoid.G1Hermite(xi, yi, thi, xf, yf, thf)
        length = clothoid_path.length
        [X,Y] = clothoid_path.SampleXY(int(length/step_length))
        return [X,Y]
    
    @staticmethod
    def interpolate_route(route, step_length):
        path_X = []
        path_Y = []

        # interpolate the route of nodes
        for i in range(len(route)-1):
            xc,yc,thc = route[i]
            xn,yn,thn = route[i+1]
            thc = math.radians(thc)
            thn = math.radians(thn)

            #print("from (",xc,yc,thc,") to (",xn,yn,thn,")\n")

            # Clothoid interpolation
            [X,Y] = PathPlanning.compute_path(xc, yc, thc, xn, yn, thn, step_length)

            for x,y in zip(X,Y):
                path_X.append(x)
                path_Y.append(y)
        
        # build array for cv.polylines
        path = []
        prev_x = 0.0
        prev_y = 0.0
        for i, (x,y) in enumerate(zip(path_X, path_Y)):
            if not (np.isclose(x,prev_x, rtol=1e-5) and np.isclose(y, prev_y, rtol=1e-5)):
                path.append([x,y])
            # else:
            #     print(f'Duplicate point: {x}, {y}, index {i}')
            prev_x = x
            prev_y = y
        
        path = np.array(path, dtype=np.float32)
        
        return path

    def draw_path(self):
        # print png image
        #map = cv.imread('Track.png')
        #map = cv.imread('src/models_pkg/track/materials/textures/2021_Medium.png')
        # get sizes
        #height, width, channels = self.map.shape
        #print(height, width)

        # draw nodes
        for node in self.list_of_nodes:
            x,y = self.get_coord(node)
            x = m2pix(x)
            y = m2pix(y)
            cv.circle(self.map, (x, y), 5, (0, 0, 255), -1)
            #add node number
            cv.putText(self.map, str(node), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw edges
        for edge in self.list_of_edges:
            x1,y1 = self.get_coord(edge[0])
            x2,y2 = self.get_coord(edge[1])
            x1 = m2pix(x1)
            y1 = m2pix(y1)
            x2 = m2pix(x2)
            y2 = m2pix(y2)
            #cv.line(self.map, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw all points in given path
        for point in self.route_list:
            x,y,_ = point
            x = m2pix(x)
            y = m2pix(y)
            cv.circle(self.map, (x, y), 5, (255, 0, 0), 1)

        # draw trajectory
        cv.polylines(self.map, [m2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)

        # show windows
        cv.namedWindow('Path', cv.WINDOW_NORMAL)
        cv.imshow('Path', self.map)
        # save current image
        cv.imwrite('my_trajectory.png', self.map)
        cv.waitKey(1)
