#!/usr/bin/python3
import numpy as np
import cv2 as cv
from time import time
import networkx as nx
from pyclothoids import Clothoid
from Simulator.src.helper_functions import *

CAR_LENGTH = 0.4

LANE_KEEPER_PATH = "Simulator/models/lane_keeper.onnx"

class Detection:
    #init 
    def __init__(self) -> None:
        #lane following
        self.lane_keeper = cv.dnn.readNetFromONNX(LANE_KEEPER_PATH)
        self.lane_cnt = 0
        self.avg_lane_detection_time = 0
 
    def estimate_he(self, frame, distance_point_ahead=0.5, show_ROI=True):
        """
        Estimates:
        - the lateral error wrt the center of the lane (e2), 
        - the angular error around the yaw axis wrt a fixed point ahead (e3),
        - the ditance from the next stop line (1/dist)
        """
        start_time = time()
        IMG_SIZE = (32,32) #match with trainer
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = frame[int(frame.shape[0]/3):,:] #/3
        #keep the bottom 2/3 of the image
        #blur
        # frame = cv.blur(frame, (15,15), 0) #worse than blur after 11,11 #with 15,15 is 1ms
        frame = cv.resize(frame, (2*IMG_SIZE[0], 2*IMG_SIZE[1]))
        frame = cv.Canny(frame, 100, 200)
        frame = cv.blur(frame, (3,3), 0) #worse than blur after 11,11
        frame = cv.resize(frame, IMG_SIZE)

        images = frame

        frame_flip = cv.flip(frame, 1) 
        #stack the 2 images
        images = np.stack((frame, frame_flip), axis=0) 
        blob = cv.dnn.blobFromImages(images, 1.0, IMG_SIZE, 0, swapRB=True, crop=False) 
        # assert blob.shape == (2, 1, IMG_SIZE[1], IMG_SIZE[0]), f"blob shape: {blob.shape}"
        self.lane_keeper.setInput(blob)
        out = -self.lane_keeper.forward() #### NOTE: MINUS SIGN IF OLD NET
        output = out[0]
        output_flipped = out[1] 

        e3 = - output[0]

        e3_flipped = - output_flipped[0]

        e3 = (e3 - e3_flipped) / 2

        #calculate estimated of thr point ahead to get visual feedback
        d = distance_point_ahead
        est_point_ahead = np.array([np.cos(e3)*d+0.2, np.sin(e3)*d])
        # print(f"est_point_ahead: {est_point_ahead}")
        
        # print(f"lane_detection: {1000*(time()-start_time):.2f} ms")
        lane_detection_time = 1000*(time()-start_time)
        self.avg_lane_detection_time = (self.avg_lane_detection_time*self.lane_cnt + lane_detection_time)/(self.lane_cnt+1)
        self.lane_cnt += 1
        if show_ROI:
            #edge
            # frame = cv.Canny(frame, 150, 180)
            cv.imshow('lane_detection', frame)
            cv.waitKey(1)
        return e3, est_point_ahead #e2, e3, est_point_ahead


####################################################################################
# CONTROLLER

L = 0.4  #length of the car, matched with lane_detection
class Controller():
    def __init__(self, k=1.0, noise_std=np.deg2rad(20)):
        #controller paramters
        self.k = k
        self.cnt = 0
        self.noise = 0.0

    def get_control(self, alpha, dist_point_ahead):
        d = dist_point_ahead#POINT_AHEAD_CM/100.0 #distance point ahead, matched with lane_detection
        delta = np.arctan((2*L*np.sin(alpha))/d)
        return  - self.k * delta


####################################################################################
# PATH PLANNING
SHOW_IMGS = False
STEP_LENGTH = 0.01

class PathPlanning(): 
    def __init__(self):
        # start and end nodes
        self.source = str(86)
        self.target = str(300)

        # initialize path
        self.path = []
        self.navigator = []# set of instruction for the navigator
        self.path_data = [] #additional data for the path (e.g. curvature, 

        # previous index of the closest point on the path to the vehicle
        self.prev_index = 0

        # read graph
        self.G = nx.read_graphml('Simulator/data/Competition_track.graphml')
        # initialize route subgraph and list for interpolation
        self.route_graph = nx.DiGraph()
        self.route_list = []
        self.old_nearest_point_index = None # used in search target point

        self.nodes_data = self.G.nodes.data()
        self.edges_data = self.G.edges.data()

        # define intersection central nodes
        self.intersection_cen = ['468','12','37', '39', '38', '11', '29', '30', '28', '84', '9','19', '20', '21', '82', '75', '74', '83', '312', '315', '65', '10', '64', '57', '56','55', '66', '73', '48', '47', '46']

        # define roundabout nodes
        self.ra = [267,268,269,270,271,302,303,304,305,306,307]
        self.ra = [str(i) for i in self.ra]

        self.ra_enter = [301,342,230] #[267,302,305]
        self.ra_enter = [str(i) for i in self.ra_enter]

        self.ra_exit = [272,231,343] # [271,304,306]
        self.ra_exit = [str(i) for i in self.ra_exit]
        
        # import nodes and edges
        self.list_of_nodes = list(self.G.nodes)
        self.list_of_edges = list(self.G.edges)

        # import map to plot trajectory and car
        # self.map = cv.imread('Simulator/src/models_pkg/track/materials/textures/2021_VerySmall.png')
        self.map = cv.imread('Simulator/src/models_pkg/track/materials/textures/test_VerySmall.png')

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
            pp = self.get_coord(prev_node)
            xp,yp = pp[0],pp[1]
            pc = self.get_coord(curr_node)
            xc,yc = pc[0],pc[1]
            pn = self.get_coord(next_node)
            xn,yn = pn[0],pn[1]

            if curr_node in self.ra_enter:
                if not(prev_node in self.ra):
                    dx = xn - xp
                    dy = yn - yp
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                else:
                    if curr_node == '302':
                        continue
                    else:
                        dx = xn - xp
                        dy = yn - yp
                        self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            elif curr_node in self.ra_exit:
                if next_node in self.ra:
                    # remain inside roundabout
                    if curr_node == '271':
                        continue
                    else:
                        dx = xn - xp
                        dy = yn - yp
                        self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                else:
                    dx = xn - xp
                    dy = yn - yp
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            else:
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))

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

        curr_node = self.source
        prev_node = curr_node
        next_node = list(self.route_graph.successors(curr_node))[0]

        #reset route list
        self.route_list = []

        self.navigator.append("go straight")
        while curr_node != self.target:
            pp = self.get_coord(prev_node)
            xp,yp = pp[0],pp[1]
            pc = self.get_coord(curr_node)
            xc,yc = pc[0],pc[1]
            pn = self.get_coord(next_node)
            xn,yn = pn[0],pn[1]

            next_is_intersection = next_node in self.intersection_cen#len(next_adj_nodes) > 1
            prev_is_intersection = prev_node in self.intersection_cen#len(prev_adj_nodes) > 1
            next_is_roundabout_enter = next_node in self.ra_enter
            #print(f"CURR: {curr_node}, NEXT: {next_node}, PREV: {prev_node}")
            # ****** ROUNDABOUT NAVIGATION ******
            if next_is_roundabout_enter:
                if curr_node == "342":
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(-1,0))))
                    dx = xc - xp
                    dy = yc - yp
                    self.route_list.append((xc,yc+0.3*dy,np.rad2deg(np.arctan2(-1,0))))
                else:
                    dx = xc-xp
                    dy = yc-yp
                    self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                    # add a further node
                    dx = xc - xp
                    dy = yc - yp
                    self.route_list.append((xc+0.3*dx,yc+0.3*dy,np.rad2deg(np.arctan2(dy,dx))))
                # enter the roundabout
                self.navigator.append("enter roundabout at " + curr_node)
                prev_node, curr_node, next_node = self.roundabout_navigation(prev_node, curr_node, next_node)
                continue               
            elif next_is_intersection:
                dx = xc - xp
                dy = yc - yp
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
                # add a further node
                dx = xc - xp
                dy = yc - yp
                self.route_list.append((xc+0.3*dx,yc+0.3*dy,np.rad2deg(np.arctan2(dy,dx))))
                # enter the intersection
                self.navigator.append("enter intersection at " + curr_node)
                prev_node, curr_node, next_node = self.intersection_navigation(prev_node, curr_node, next_node)
                continue
            elif prev_is_intersection:
                # add a further node
                dx = xn - xc
                dy = yn - yc
                self.route_list.append((xc-0.3*dx,yc-0.3*dy,np.rad2deg(np.arctan2(dy,dx))))
                # and add the exit node
                dx = xn - xc
                dy = yn - yc
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            else:
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xc,yc,np.rad2deg(np.arctan2(dy,dx))))
            
            prev_node = curr_node
            curr_node = next_node
            if curr_node != self.target:
                next_node = list(self.route_graph.successors(curr_node))[0]
            else:
                # You arrived at the END
                dx = xn - xp
                dy = yn - yp
                self.route_list.append((xn,yn,np.rad2deg(np.arctan2(dy,dx))))
                next_node = None
        
        self.navigator.append("stop")

    def compute_shortest_path(self, source=None, target=None):
        ''' Generates the shortest path between source and target nodes using Clothoid interpolation '''
        src = str(source) if source is not None else self.source
        tgt = str(target) if target is not None else self.target
        self.source = src
        self.target = tgt
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
        
        #remove duplicates
        prev_x, prev_y, prev_yaw = 0,0,0
        for i, (x,y,yaw) in enumerate(self.route_list):
            if np.linalg.norm(np.array([x,y]) - np.array([prev_x,prev_y])) < 0.001:
                #remove element from list
                self.route_list.pop(i)
            prev_x, prev_y, prev_yaw = x,y,yaw
            
        # interpolate the route of nodes
        self.path = PathPlanning.interpolate_route(self.route_list)

        return self.path

    def generate_path_passing_through(self, list_of_nodes):
        """
        Extend the path generation from source-target to a sequence of nodes/locations
        """
        assert len(list_of_nodes) >= 2, "List of nodes must have at least 2 nodes"
        print("Generating path passing through: ", list_of_nodes)
        src = list_of_nodes[0]
        tgt = list_of_nodes[1]
        complete_path = self.compute_shortest_path(source=src, target=tgt)
        for i in range(1,len(list_of_nodes)-1):
            src = list_of_nodes[i]
            tgt = list_of_nodes[i+1]
            self.compute_shortest_path(source=src, target=tgt)
            #now the local path is computed in self.path
            #remove first element of self.path
            self.path = self.path[1:]
            #we need to add the local path to the complete path
            complete_path = np.concatenate((complete_path, self.path))
        self.path = complete_path

    def get_reference(self, car, dist_point_ahead, path_ahead_distances): 
        '''
        returns:
        - the heading error
        - the point ahead
        - the sequence of yaws of the points ahead
        - boolean indicating if the car has reached the end of the path
        In the future it may also return something related to the state of the road
        '''
        #limit the search in a neighborhood of max_idx_ahead points around the current point
        max_idx_ahead = int(100*(path_ahead_distances[-1] + 0.1))
        max_index = min(self.prev_index + max_idx_ahead, len(self.path)-2)
        min_index = max(self.prev_index, 0)
        path_to_analyze = self.path[min_index:max_index+1, :]
        #get current car position
        curr_pos = np.array([car.x_true, car.y_true])
        #find the index of the point of the path closest to the current car position
        closest_index = np.argmin(np.linalg.norm(path_to_analyze - curr_pos, axis=1))
        #check if we arrived at the end of the path
        finished = True if closest_index + min_index >= len(self.path)-max_idx_ahead-5 else False
        #check if it's last element
        if closest_index == len(path_to_analyze)-1:
            closest_index = closest_index - 2

        #get point ahead
        path_ahead = path_to_analyze[closest_index:, :]
        # idx_point_ahead = np.argmin(np.abs(np.linalg.norm(path_ahead - curr_pos, axis=1) - dist_point_ahead)) #maybe more precise, but less smooth
        idx_point_ahead = min(closest_index + int(dist_point_ahead*100), len(path_to_analyze)-1) #inaccurate, but smoother
        point_ahead = path_to_analyze[idx_point_ahead]
        #translate into car frame
        point_ahead = to_car_frame(point_ahead, car, return_size=2)
        #calculate heading error
        heading_error = np.arctan2(point_ahead[1], point_ahead[0]+0.4/2) 

        #update prev_index
        self.prev_index = min_index + closest_index - 1

        # sequence of points ahead
        path_ahead_distances = np.asarray(path_ahead_distances)
        seq_indeces = (100*path_ahead_distances).astype(int) 
        seq_points = path_to_analyze[seq_indeces]
        seq_points = to_car_frame(seq_points, car, return_size=2)
        seq_heading_errors = np.arctan2(seq_points[:,1], seq_points[:,0]+0.4/2)

        pts = np.concatenate((np.array([[0,0]]), seq_points), axis=0)
        diffs = pts[1:] - pts[:-1]
        seq_relative_angles = np.arctan2(diffs[:,1], diffs[:,0])
        return heading_error, point_ahead, seq_heading_errors, seq_relative_angles, path_ahead, finished

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
            return 0
        length = 0
        for i in range(len(path)-1):
            x1,y1 = path[i]
            x2,y2 = path[i+1]
            length += np.hypot(x2-x1,y2-y1) 
        return length

    def get_coord(self, node):
        x = self.nodes_data[node]['x']
        y = self.nodes_data[node]['y']
        p = np.array([x,y])
        p = mL2mR(p)
        return p
    
    def compute_path(xi, yi, thi, xf, yf, thf, step_length):
        clothoid_path = Clothoid.G1Hermite(xi, yi, thi, xf, yf, thf)
        length = clothoid_path.length
        [X,Y] = clothoid_path.SampleXY(int(length/step_length))
        return [X,Y]
    
    def interpolate_route(route):
        path_X = []
        path_Y = []

        # interpolate the route of nodes
        for i in range(len(route)-1):
            xc,yc,thc = route[i]
            xn,yn,thn = route[i+1]
            thc = np.deg2rad(thc)
            thn = np.deg2rad(thn)

            #print("from (",xc,yc,thc,") to (",xn,yn,thn,")\n")

            # Clothoid interpolation
            [X,Y] = PathPlanning.compute_path(xc, yc, thc, xn, yn, thn, STEP_LENGTH)

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
        # draw nodes
        for node in self.list_of_nodes:
            p = self.get_coord(node) #change to r coordinate
            cv.circle(self.map, mR2pix(p), 5, (0, 0, 255), -1)
            #add node number
            cv.putText(self.map, str(node), mR2pix(p), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw edges
        for edge in self.list_of_edges:
            p1 = self.get_coord(edge[0])
            p2 = self.get_coord(edge[1])
            #cv.line(self.map, mR2pix(p1), mR2pix(p2), (0, 255, 255), 2)

        # draw all points in given path
        for point in self.route_list:
            x,y,_ = point
            p = np.array([x,y])
            cv.circle(self.map, mR2pix(p), 5, (255, 0, 0), 1)

        # draw trajectory
        cv.polylines(self.map, [mR2pix(self.path)], False, (200, 200, 0), thickness=4, lineType=cv.LINE_AA)
        if SHOW_IMGS:
            cv.imshow('2D-MAP', self.map)
            cv.waitKey(1)

