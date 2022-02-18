#!/usr/bin/python3

from os import path
from re import X
from time import sleep
import networkx as nx
import numpy as np
import cv2 as cv
import random
from pyclothoids import Clothoid
import math

from helper_functions import *

class PathPlanning(): 
    def __init__(self, map_img, source="86", target="300"):
        # start and end nodes
        self.source = str(source)
        self.target = str(target)

        # initialize path
        self.path = []
        self.navigator = []# set of instruction for the navigator

        # previous index of the closest point on the path to the vehicle
        self.prev_index = 0

        # read graph
        self.G = nx.read_graphml('Competition_track.graphml')
        # initialize route subgraph and list for interpolation
        self.route_graph = nx.DiGraph()
        self.route_list = []
        self.old_nearest_point_index = None # used in search target point

        # define intersection central nodes
        #self.intersection_cen = ['347', '37', '39', '38', '11', '29', '30', '28', '371', '84', '9', '20', '20', '82', '75', '74', '83', '312', '315', '65', '468', '10', '64', '57', '56', '66', '73', '424', '48', '47', '46']
        self.intersection_cen = ['37', '39', '38', '11', '29', '30', '28', '371', '84', '9','19', '20', '21', '82', '75', '74', '83', '312', '315', '65', '468', '10', '64', '57', '56', '66', '73', '424', '48', '47', '46']


        # define intersecion entrance nodes
        self.intersection_in = [77,45,54,50,41,79,374,52,43,81,36,4,68,2,34,70,6,32,72,59,15,16,27,14,25,18,61,23,63]
        self.intersection_in = [str(i) for i in self.intersection_in]

        # define intersecion exit nodes
        self.intersection_out = [76,78,80,40,42,44,49,51,53,67,69,71,1,3,5,7,8,31,33,35,22,24,26,13,17,58,60,62]
        self.intersection_out = [str(i) for i in self.intersection_out]

        # define left turns tuples
        #self.left_turns = [(45,42),(43,40),(54,51),(36,33),(34,31),(6,3),(2,7),(2,8),(4,1),(70,67),(72,69),(27,24),(25,22),(16,13),(61,58),(63,60)]

        # define roundabout nodes
        self.ra = [267,268,269,270,271,302,303,304,305,306,307]
        self.ra = [str(i) for i in self.ra]

        self.ra_enter = [267,302,305]
        self.ra_enter = [str(i) for i in self.ra_enter]

        self.ra_exit = [271,304,306]
        self.ra_exit = [str(i) for i in self.ra_exit]


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

    def compute_shortest_path(self, step_length=0.03):
        ''' Generates the shortest path between source and target nodes using Clothoid interpolation '''
        route_nx = []

        # generate the shortest route between source and target      
        route_nx = list(nx.shortest_path(self.G, source=self.source, target=self.target)) 
        # generate a route subgraph       
        self.route_graph.add_nodes_from(route_nx)
        for i in range(len(route_nx)-1):
            self.route_graph.add_edges_from( [ (route_nx[i], route_nx[i+1], self.G.get_edge_data(route_nx[i],route_nx[i+1])) ] )         # augment route with intersections and roundabouts
        # expand route node and update navigation instructions
        self.compute_route_list()
        
        #self.print_navigation_instructions()
        # interpolate the route of nodes
        self.path = PathPlanning.interpolate_route(self.route_list, step_length)
    
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

    def get_reference(self, car, limit_search_to=50, look_ahead=5): 
        '''
        Returns the reference point, i.e., the point on the path nearest to the vehicle, 
        returns
        x desired
        y desired
        yaw desired
        TODO current curvature
        finished=True when the end of the path is reached
        TODO path paramters ahead
        '''
        prev_pos = self.path[self.prev_index]
        x_prev = self.path[self.prev_index, 0]
        y_prev = self.path[self.prev_index, 1]
        finished = False
        curv = 0.0

        #limit the search in a neighborhood of limit_search_to points around the current point
        max_index = min(self.prev_index + limit_search_to, len(self.path)-1)
        if max_index == len(self.path)-1:
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
        x_des = closest_point[0]
        y_des = closest_point[1]
        #calculate yaw angle
        next_x = path_to_analyze[closest_index+1, 0]
        next_y = path_to_analyze[closest_index+1, 1]
        yaw_des = - np.arctan2(next_y - y_des, next_x - x_des)
        #update prev_index
        self.prev_index = min_index + closest_index - (1+look_ahead)
        return x_des, y_des, yaw_des, curv, finished


    def get_length(self):
        ''' Calculates the length of the trajectory '''
        length = 0
        for i in range(len(self.path)-1):
            x1,y1 = self.path[i]
            x2,y2 = self.path[i+1]
            length += math.hypot(x2-x1,y2-y1)
        
        #print(f"Length of the trajectory: {length}")  
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
