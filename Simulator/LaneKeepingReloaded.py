import cv2
import numpy as np
import math

from itertools import compress

"""
This class implements the lane keeping algorithm by calculating angles from the detected lines slopes.
"""

class LaneKeepingReloaded:

    lane_width_px = 478

    def __init__(self, width, height):
        wT, hT, wB,hB = 70,350,0,443
        self.width = width
        self.height = height
        self.src_points = np.float32([[wT, hT],[width - wT, hT], [wB, hB], [width - wB, hB]])
        self.warp_matrix = None
        self.inv_warp_matrix = None

    def threshold(self, frame):
        
        imgHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lowerWhite = np.array([0,0,0])
        upperWhite = np.array([0,179,120])
        
        maskWhite = cv2.inRange(imgHsv, lowerWhite, upperWhite)
        cv2.bitwise_not(maskWhite, maskWhite)

        # cv2.imshow("White", maskWhite)

        return maskWhite

    def warp_image(self, frame):
        
        #Get height and width of image
        

        #Destination points for warping
        dst_points = np.float32([[0,0],[self.width,0],[0,self.height],[self.width,self.height]])

        self.warp_matrix = cv2.getPerspectiveTransform(self.src_points, dst_points)
        self.inv_warp_matrix = cv2.getPerspectiveTransform(dst_points, self.src_points)

        #Warp frame
        warped_frame = cv2.warpPerspective(frame, self.warp_matrix, (self.width,self.height))

        #cv2.imshow("Warp Frame", warped_frame)

        return warped_frame

    def calculate_lane_in_pixels(self, frame):

        count = 0
        start_count = False

        for j in range(1,self.height):
            
            if frame[0, j] == 0 and frame[0, j-1]  == 255:
                start_count = True 

            if start_count and frame[0, j] == 0:
                count += 1
            
            if start_count and frame[0, j] == 255 and frame[0, j+1] == 255:
                break
        #print(count)

    def polyfit_sliding_window(self,frame):


        #Check if frame is black
        if frame.max() <= 0:
            return np.array([[0,0,0],[0,0,0]])

        #Compute peaks in the two frame halves, to get an idea of lane start positions

        histogram = None
        
        cutoffs = [int(self.height / 2.0), 0]

        for cutoff in cutoffs:
            histogram = np.sum(frame[cutoff:,:], axis=0)

            if histogram.max() > 0:
                break
        
        if histogram.max() == 0:
            #print('Unable to detect lane lines in this frame. Trying another frame!')
            return (None, None)

        #Calculate peaks of histogram

        midpoint = np.int(self.width / 2.0)

        leftx_base = np.argmax(histogram[:midpoint])


        b = histogram[midpoint:]

        b = b[::-1]

        rightx_base = len(b) - np.argmax(b) - 1 + midpoint
        
        #rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        #print("Rightmost: ", rightx_base)
        #Black image to draw on -VIS-
        out = np.dstack((frame, frame, frame)) * 255

        #Number of sliding windows
        windows_number = 12
        
        #Width of the windows +/- margin
        margin = 100

        #Min number of pixels needed to recenter the window
        minpix = 50

        #Window Height
        window_height = int(self.height / float(windows_number))

        #Min number of eligible pixels needed to fit a 2nd order polynomial as a lane line
        min_lane_pts = 2000

        #Find all pixels that are lane lines on the picture
        nonzero = frame.nonzero()
        nonzerox = np.array(nonzero[1])
        nonzeroy = np.array(nonzero[0])

        #Current position, updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        #Lists for indices of each lane
        left_lane_inds = []
        right_lane_inds = []

        for window in range(windows_number):
            #Find window boundaries in x and y axis
            win_y_low = self.height - (1 + window) * window_height
            win_y_high = self.height - window * window_height



            #LEFT 
            
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
           
            # Draw windows for visualisation
            cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),\
                        (0, 0, 255), 2)

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                            & (nonzerox >= win_xleft_low) & (nonzerox <= win_xleft_high)).nonzero()[0]
            left_lane_inds.append(good_left_inds)

            #RIGHT
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin


            cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high),\
                        (0, 255, 0), 2)


            # Identify the nonzero pixels in x and y within the window
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy <= win_y_high)
                            & (nonzerox >= win_xright_low) & (nonzerox <= win_xright_high)).nonzero()[0]

            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) >  minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))

            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract pixel positions for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        left_fit, right_fit = None, None
        
        # Sanity check; Fit a 2nd order polynomial for each lane line pixels
        if len(leftx) >= min_lane_pts and histogram[leftx_base] != 0: 
            left_fit = np.polyfit(lefty, leftx, 2)

        if len(rightx) >= min_lane_pts and histogram[rightx_base] != 0:
            right_fit = np.polyfit(righty, rightx, 2)

        '''
        # Validate detected lane lines
        valid = True#self.check_validity(left_fit, right_fit)
    
        if not valid:
            # If the detected lane lines are NOT valid:
            # 1. Compute the lane lines as an average of the previously detected lines
            # from the cache and flag this detection cycle as a failure by setting ret=False
            # 2. Else, if cache is empty, return 
            
            if len(cache) == 0:
                return np.array([[0,0,0],[0,0,0]])
            
            avg_params = np.mean(cache, axis=0)
            left_fit, right_fit = avg_params[0], avg_params[1]
            #ret = False
        '''
        # Color the detected pixels for each lane line
        out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [255, 10, 255]

        #cv2.imshow("Windows", out)
        return np.array([left_fit, right_fit])

    def get_poly_points(self, left_fit, right_fit):

        #TODO: CHECK EDGE CASES 

        ysize, xsize = self.height, self.width
    
        # Get the points for the entire height of the image
        plot_y = np.linspace(0, ysize-1, ysize)
        #print(len(plot_y))
        plot_xleft = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
        plot_xright = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]
        
        
        # But keep only those points that lie within the image
        #plot_xleft = plot_xleft[(plot_xleft >= 0) & (plot_xleft <= xsize - 1)]
        #plot_xright = plot_xright[(plot_xright >= 0) & (plot_xright <= xsize - 1)]
        #plot_yleft = np.linspace(ysize - len(plot_xleft), ysize - 1, len(plot_xleft))
        #plot_yright = np.linspace(ysize - len(plot_xright), ysize - 1, len(plot_xright))
        
        plot_yright = plot_y
        plot_yleft = plot_y
        return plot_xleft.astype(np.int), plot_yleft.astype(np.int), plot_xright.astype(np.int), plot_yright.astype(np.int)
    
    def compute_curvature(self, poly_param):

        plot_xleft, plot_yleft, plot_xright, plot_yright = self.get_poly_points(poly_param[0], poly_param[1])
    
        y_eval = np.max(plot_yleft)

        left_fit_cr = np.polyfit(plot_yleft, plot_xleft, 2)
        right_fit_cr = np.polyfit(plot_yright, plot_xright, 2)
        
        left_curverad = ((1 + (2*left_fit_cr[0]* y_eval + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        return left_curverad, right_curverad

    def get_error(self, frame):

        num_lines = 20
        line_im = frame * 255
        line_height = int(self.height / float(num_lines))

        for i in range(num_lines):
            
            line_im = cv2.line(line_im, pt1=(0, -line_height * i + self.height), pt2=(self.width, -line_height * i + self.height), color=(255,255,255), thickness=2)

        #cv2.imshow("Lines", line_im)

        cut_im = np.bitwise_and(frame, line_im)

        #cv2.imshow("Cuts", cut_im)

        nonzero = cut_im.nonzero()
        
        nonzerox = np.array(nonzero[1])

        nonzeroy = np.array(nonzero[0])


        mean_list = []

        
        for i in range(num_lines):

            mask = list([nonzero[0] == i * line_height])
            #print(mask)
            x_coors = nonzerox[mask]
            if len(x_coors) > 0:
                #print("XCOORS", x_coors)
                mean_list.append(int(np.average(x_coors)))
            
        mean_list = np.array(mean_list)
        mean_list = mean_list[np.isfinite(mean_list)]
        if len(mean_list) != 0:
            weighted_mean = self.weighted_average(mean_list)
        else:
            weighted_mean = 320

         

        circle_im = cv2.circle(cut_im, (weighted_mean, self.height), 10, color=(255,255,255), thickness=-1)
        error = weighted_mean - int(self.width / 2.0)
        #print("Error: ", weighted_mean - int(self.width / 2.0) )
        #cv2.imshow("Circle", cut_im)
        setpoint = weighted_mean
        return error, setpoint
        
    def weighted_average(self, num_list):

        mean = 0
        count = len(num_list)

        weights = []

        for i in range(count):
            weights.append( 1.0 / count * (i+1))

        
        for i in range(count):
            mean += num_list[i] * weights[i]
        
        mean = int(mean / np.sum(weights))

        return mean

    def get_angle(self, error):
        max_error = 130
        factor = float(90.0/max_error)

        angle = factor * error

        return angle

    def plot_points(self, left_x, left_y, right_x, right_y, frame):

        out = frame * 0

        for i in range(len(left_x)):
            cv2.circle(out, (left_x[i], left_y[i]), 10, color=(255,255,255), thickness=-1)
            cv2.circle(out, (right_x[i], right_y[i]), 10, color=(255,255,255), thickness=-1)
            
            #cv2.circle(out, (int((left_x[i] + right_x[i]) / 2.0), int((left_y[i] + right_y[i]) / 2.0)), 10, color=(255,255,255), thickness=-1)
            
        return out

    def average_points(self, left_x, right_x):

        mean = np.average()

    def lane_keeping_pipeline(self, frame):

        warped = self.warp_image(frame)
        thresh = self.threshold(warped)
        both_lanes = False

        left, right = self.polyfit_sliding_window(thresh)

        poly_image = 0

        if left is not None and right is not None:
            #print("BOTH LANES")

            left_x, left_y, right_x, right_y = self.get_poly_points(left, right)
            cache = [left,right]
            
            poly_image = self.plot_points(left_x, left_y,right_x, right_y, warped)
            #cv2.imshow("Poly", poly_image)

            
            error, setpoint = self.get_error(poly_image)
            
            nose2wheel = 320

            angle = 90 - math.degrees(math.atan2(nose2wheel, error))
            both_lanes = True
            #print("Coords", nose2wheel, error)
            #print("Angle: ", angle)

        elif right is None and left is not None:
            #print("LEFT LANE")

            x1 = left[0] * 480 ** 2 + left[1] * 480 + left[2]
            x2 = left[2]

            dx = math.fabs(x2 - x1)
            dy = 480

            #tan = float(dy) / float(dx) 

            angle =  abs( 90 - math.degrees(math.atan2(dy, dx)) )
            
            #angle = 20
        
        elif left is None and right is not None:
            #print("RIGHT LANE")

            x1 = right[0] * 480 ** 2 + right[1] * 480 + right[2]
            x2 = right[2]

            dx = math.fabs(x2 - x1)
            dy = 480

            #tan = float(dy) / float(dx) 

            angle = - abs( 90 - math.degrees(math.atan2(dy, dx)) )

            #print("SINGLE LINE ANGLE: " , angle)
        else:
            angle = 0
    
        return angle, both_lanes, poly_image