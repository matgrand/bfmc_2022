#!/usr/bin/python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import random
from scipy.interpolate import BSpline, CubicSpline, make_interp_spline
from torch import le
from helper_functions import *

class SimpleController():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0
        self.ff = ff

    def get_control(self, x, y, yaw, xd, yd, yawd, vd, curvature_ahead):
        e_yaw = np.arctan2(np.sin(yawd-yaw), np.cos(yawd-yaw)) # yaw error, in radians, not a simple difference (overflow at +-180)
        ex = xd - x #x error
        ey = yd - y #y error
        #similar to UGV trajectory tracking but frame of reference is left-hand
        self.e1 = -ex * np.cos(yaw) + ey * np.sin(yaw) #x error in the body frame
        self.e2 = -ex * np.sin(yaw) - ey * np.cos(yaw) #y error in the body frame
        self.e3 = e_yaw #yaw error
        output_angle = self.ff*curvature_ahead - self.k2 * self.e2 - self.k3 * self.e3
        output_speed = vd - self.k1 * self.e1
        return output_speed, output_angle


class TrainingTrajectory(): 

    def get_coord(self, node):
        x = self.nodes_data[node]['x']
        y = self.nodes_data[node]['y']
        return x, y

    def get_reference(self, t, t_delta_ahead=0.5):
        finished = False
        #interpolate trajectory at time t
        diff = np.absolute(self.time_vector - t)
        index = np.argmin(diff)
        [xd, yd] = self.trajectory[index]

        #get yaw
        [x0,y0] = self.trajectory[index-1]
        yawd = - np.arctan2(yd-y0, xd-x0)

        # get average curvature of the trajectory 0.5 s in the future 
        t_ahead = t + t_delta_ahead
        diff = np.absolute(self.time_vector - t_ahead)
        index_ahead = np.argmin(diff)
        #check if it's the last element
        if index_ahead > len(self.time_vector)-200:
            finished = True
        # calculate curvature 
        local_traj = self.trajectory[index:index_ahead]
        local_time = self.time_vector[index:index_ahead]
        dx_dt = np.gradient(local_traj[:,0], local_time)
        dy_dt = np.gradient(local_traj[:,1], local_time)
        dp_dt = np.gradient(local_traj, local_time, axis=0)
        v = np.linalg.norm(dp_dt, axis=1)
        ddx_dt = np.gradient(dx_dt, local_time)
        ddy_dt = np.gradient(dy_dt, local_time)
        curv = (dx_dt*ddy_dt-dy_dt*ddx_dt) / np.power(v,1.5)
        avg_curv = np.mean(curv)
        return xd, yd, yawd, avg_curv, finished


    def __init__(self, nodes_ahead=100, speed=0.1):
        #read graph
        self.G = nx.read_graphml('Competition_track.graphml')

        #define intersecion entarnce nodes
        self.intersection_in = [77,45,54,50,41,79,374,52,43,81,36,4,68,2,34,70,6,32,72,59,15,16,27,14,25,18,61,23,63]
        self.intersection_in = [str(i) for i in self.intersection_in]
        #define intersecion exit nodes
        self.intersection_out = [76,78,80,40,42,44,49,51,53,67,69,71,1,3,5,7,8,31,33,35,22,24,26,13,17,58,60,62]
        self.intersection_out = [str(i) for i in self.intersection_out]
        self.roundabout_junctions = ['306','271','304', '347']


        self.list_of_nodes = list(self.G.nodes)
        self.list_of_edges = list(self.G.edges)

        #nodes_data = nx.get_node_attributes(G, 'pos')
        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        #print png image
        # map = cv.imread('Track.png')
        # map = cv.imread('src/models_pkg/track/materials/textures/2021_Medium.png')
        self.map = cv.imread('src/models_pkg/track/materials/textures/2021_VerySmall.png')
        #get sizes
        height, width, channels = self.map.shape
        print(height, width)


        for node in self.list_of_nodes:
            x,y = self.get_coord(node)
            x = m2pix(x)
            y = m2pix(y)
            cv.circle(self.map, (x, y), 5, (0, 0, 255), -1)
            #add node number
            cv.putText(self.map, str(node), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for edge in self.list_of_edges:
            x1,y1 = self.get_coord(edge[0])
            x2,y2 = self.get_coord(edge[1])
            x1 = m2pix(x1)
            y1 = m2pix(y1)
            x2 = m2pix(x2)
            y2 = m2pix(y2)
            cv.line(self.map, (x1, y1), (x2, y2), (0, 255, 255), 2)


        #GENERATE PATH USING CUBIC SPLINE
        start_node = 86-1 #86-1
        end_node = None
        curr_node = self.list_of_nodes[start_node]
        prev_node = curr_node
        path = []
        #GENERATE ROUTE
        for i in range(nodes_ahead):
            adj_nodes = list(self.G.adj[curr_node])
            if len(adj_nodes) == 0 or curr_node == end_node: #exit condition
                break
            else:
                #random choice
                next_node = random.choice(adj_nodes)
                while next_node == '76' or next_node=='316':
                    next_node = random.choice(adj_nodes) #avoid exiting
                # next_node = adj_nodes[0] #choose first node
                next_adj_nodes = list(self.G.adj[next_node])
                prev_adj_nodes = list(self.G.adj[prev_node])
                xc,yc = self.get_coord(curr_node)
                xn,yn = self.get_coord(next_node)
                xp,yp = self.get_coord(prev_node)
                curr_is_junction = len(adj_nodes) > 1
                next_is_junction = len(next_adj_nodes) > 1
                prev_is_junction = len(prev_adj_nodes) > 1
                # print(f"CURR: {curr_node}, NEXT: {next_node}, PREV: {prev_node}")
                if curr_is_junction:
                    #curr node is a junction
                    # print('curr node is a junction')
                    # dont add junctions
                    # exceptions
                    if curr_node in self.roundabout_junctions: #in the roundabout
                        path.append((xc,yc))
                    

                else:
                    #curr node is a straight
                    a = .8
                    b = .8
                    if next_is_junction: 
                        #next node is a junction
                        # print('next node is a junction')
                        #add some points from prev_node to curr_node
                        dx = xc - xp
                        dy = yc - yp
                        path.append((xc, yc))
                        if curr_node in self.intersection_in:
                            path.append((xc+a*dx, yc+a*dy))
                    elif prev_is_junction:
                        #prev node is a junction
                        # print('prev node is a junction')
                        #add some points from curr_node to next_node
                        dx = xn - xc
                        dy = yn - yc
                        if curr_node in self.intersection_out:
                            path.append((xc-b*dx, yc-b*dy))
                        path.append((xc,yc))
                    else:
                        #both prev and next node are straights
                        # print('both prev and next node are straights')
                        #simply add the curr node coords
                        path.append((xc,yc))

                prev_node = curr_node
                curr_node = next_node

        #draw all points
        for point in path:
            x,y = point
            x = m2pix(x)
            y = m2pix(y)
            cv.circle(self.map, (x, y), 5, (255, 0, 0), 1)
        points = np.array(path, dtype=np.float32)
        #points_pix = m2pix(points)
        time = np.linspace(0, 1, len(points))
        spline = BSpline(time, points, 3, extrapolate=True)
        t_new = np.linspace(0, 1, 100*nodes_ahead)
        points_new = spline(t_new)
        self.trajectory = np.array(points_new, dtype=np.float32)

        points_new = m2pix(points_new)
        cv.polylines(self.map, [points_new], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)


        #TRAJECTORY

        #calculate legth of the trajectory
        length = 0
        for i in range(len(self.trajectory)-1):
            x1,y1 = self.trajectory[i]
            x2,y2 = self.trajectory[i+1]
            length += np.sqrt((x2-x1)**2 + (y2-y1)**2)
        print(f"Length of the trajectory: {length}")

        self.total_time = length/speed
        print(f"Total time: {self.total_time}")

        self.time_vector = np.linspace(0, self.total_time, len(self.trajectory))



        cv.namedWindow('Trajectory', cv.WINDOW_NORMAL)
        cv.imshow('Trajectory', self.map)
        cv.waitKey(1)



# #initilize class
# traj = TrainingTrajectory()
# contr = SimpleController()

