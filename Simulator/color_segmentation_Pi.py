import numpy as np
import os
import cv2
import glob
import imutils

filters_list = ['red', 'green', 'blue', 'yellow']

def identify_blue(org_img, img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# convert the image to HSV format for color segmentation
	img_hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)

	lower_blue = np.array([111, 127, 50])
	upper_blue = np.array([126, 255, 255])

	mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

	blue_mask = cv2.bitwise_and(img_output, img_output, mask=mask)
	
	blue_mask_gray = cv2.cvtColor(blue_mask, cv2.COLOR_BGR2GRAY)
	blue_mask_gray = cv2.medianBlur(blue_mask_gray, 5)
	_,thresh1 = cv2.threshold(blue_mask_gray, 10 , 255,cv2.THRESH_BINARY)
	# cv2.imshow('blue_mask_gray', blue_mask_gray)
	# cv2.imshow('thresh1', thresh1)
	
	kernel_1 = np.ones((3, 3), np.uint8)
	kernel_2 = np.ones((5, 5), np.uint8)	

	# erosion = cv2.erode(blue_mask_gray, kernel_1, iterations=1)
	# cv2.imshow("erosion", erosion)
	dilation = cv2.dilate(thresh1, kernel_2, iterations=1)
	# cv2.imshow("dilation", dilation)
	opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)
	# cv2.imshow("opening", opening)

	mser_blue = cv2.MSER_create(5, 1000, 5000)
	# Do MSER
	# regions, _ = mser_blue.detectRegions(np.uint8(filtered_b))
	regions, _ = mser_blue.detectRegions(np.uint8(opening))

	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	blank = np.zeros_like(blue_mask)
	cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))
	# cv2.imshow("mser_blue", blank)

	# erosion = cv2.erode(blank, kernel_1, iterations=1)
	# cv2.imshow("erosion", erosion)
	# dilation = cv2.dilate(blank, kernel_2, iterations=1)
	# cv2.imshow("dilation2", dilation)
	closing = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel_2)
	# cv2.imshow("opening2", opening)
	_, b_thresh = cv2.threshold(closing[:, :, 0], 60, 255, cv2.THRESH_BINARY)
	# cv2.imshow("b_thresh", b_thresh)
	return b_thresh

def identify_yellow(org_img, img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# convert the image to HSV format for color segmentation
	img_hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)

	# mask to extract yellow
	lower_yellow = np.array([15, 70, 50])
	upper_yellow = np.array([35, 255, 255])

	mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

	yellow_mask = cv2.bitwise_and(img_output, img_output, mask=mask)
	# cv2.imshow('yellow_mask', yellow_mask)

	yellow_mask_gray = cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2GRAY)
	yellow_mask_gray = cv2.medianBlur(yellow_mask_gray, 5)
	_,thresh1 = cv2.threshold(yellow_mask_gray, 30, 255,cv2.THRESH_BINARY)
	# cv2.imshow('yellow_mask_gray', yellow_mask_gray)

	kernel_1 = np.ones((3, 3), np.uint8)
	kernel_2 = np.ones((5, 5), np.uint8)	

	# erosion = cv2.erode(yellow_mask_gray, kernel_1, iterations=1)
	# cv2.imshow("erosion", erosion)
	dilation = cv2.dilate(thresh1, kernel_2, iterations=1)
	# cv2.imshow("dilation", dilation)
	opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)
	# cv2.imshow("opening", opening)

	mser_yellow = cv2.MSER_create(5, 800, 1500)
	# Do MSER
	# regions, _ = mser_yellow.detectRegions(np.uint8(filtered_b))
	regions, _ = mser_yellow.detectRegions(np.uint8(opening))

	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	blank = np.zeros_like(yellow_mask)
	cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))
	# cv2.imshow("mser_yellow", blank)

	# erosion = cv2.erode(blank, kernel_1, iterations=1)
	# cv2.imshow("erosion", erosion)
	# dilation = cv2.dilate(blank, kernel_2, iterations=1)
	# cv2.imshow("dilation", dilation)
	closing = cv2.morphologyEx(blank, cv2.MORPH_CLOSE, kernel_2)
	# cv2.imshow("opening", opening)
	_, b_thresh = cv2.threshold(closing[:, :, 0], 60, 255, cv2.THRESH_BINARY)
	# cv2.imshow("b_thresh", b_thresh)
	return b_thresh

def identify_red(org_img, img):

	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# mask to extract red
	img_hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)
	lower_red_1 = np.array([0, 60, 63])
	upper_red_1 = np.array([10, 255, 255])
	mask_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
	lower_red_2 = np.array([160, 60, 63])
	upper_red_2 = np.array([180, 255, 255])
	mask_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
	mask = cv2.bitwise_or(mask_1, mask_2)
	red_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

	red_mask_gray = cv2.cvtColor(red_mask, cv2.COLOR_BGR2GRAY)
	red_mask_gray = cv2.medianBlur(red_mask_gray, 5)
	_,thresh1 = cv2.threshold(red_mask_gray, 10, 255,cv2.THRESH_BINARY)
	# cv2.imshow('red_mask_gray', red_mask_gray)
	#print('RED')
	kernel_1 = np.ones((3, 3), np.uint8)
	kernel_2 = np.ones((5, 5), np.uint8)	

	# erosion = cv2.erode(thresh1, kernel_1, iterations=1)
	# cv2.imshow("erosion", erosion)
	dilation = cv2.dilate(thresh1 , kernel_2 , iterations=1)
	# cv2.imshow("dilation", dilation)
	closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_2)
	# cv2.imshow("opening", closing)

	mser_red = cv2.MSER_create(5, 1000, 4000)
	# Do MSER
	#regions, _ = mser_red.detectRegions(np.uint8(filtered_b))
	regions, _ = mser_red.detectRegions(np.uint8(closing))

	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	blank = np.zeros_like(red_mask)
	cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))
	# cv2.imshow("mser_red", blank)

	erosion = cv2.erode(blank, kernel_1, iterations=1)
	# cv2.imshow("erosion2", erosion)
	# dilation = cv2.dilate(erosion, kernel_2, iterations=1)
	# cv2.imshow("dilation2", dilation)
	closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel_2)
	# cv2.imshow("opening2", opening)
	_, b_thresh = cv2.threshold(closing[:, :, 0], 60, 255, cv2.THRESH_BINARY)
	# cv2.imshow("b_thresh", b_thresh)
	return b_thresh

def identify_green(org_img, img):
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

	# equalize the histogram of the Y channel
	img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# convert the image to HSV format for color segmentation
	img_hsv = cv2.cvtColor(org_img, cv2.COLOR_BGR2HSV)

	# mask to extract yellow
	lower_green = np.array([70, 50, 20])
	upper_green= np.array([110, 255, 255])

	mask = cv2.inRange(img_hsv, lower_green, upper_green)

	green_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

	green_mask_gray = cv2.cvtColor(green_mask, cv2.COLOR_BGR2GRAY)
	green_mask_gray = cv2.medianBlur(green_mask_gray, 5)
	_,thresh1 = cv2.threshold(green_mask_gray, 30, 255,cv2.THRESH_BINARY)
	# cv2.imshow('green_mask_gray', green_mask_gray)

	kernel_1 = np.ones((3, 3), np.uint8)
	kernel_2 = np.ones((5, 5), np.uint8)	

	#erosion = cv2.erode(thresh1, kernel_1, iterations=1)
	#cv2.imshow("erosion", erosion)
	dilation = cv2.dilate(thresh1 , kernel_2 , iterations=1)
	# cv2.imshow("dilation", dilation)
	closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_2)
	# cv2.imshow("opening", closing)

	mser_green = cv2.MSER_create(5, 1800, 4500)
	# Do MSER
	#regions, _ = mser_green.detectRegions(np.uint8(filtegreen_b))
	regions, _ = mser_green.detectRegions(np.uint8(closing))

	hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
	blank = np.zeros_like(green_mask)
	cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))
	# cv2.imshow("mser_green", blank)

	#erosion = cv2.erode(blank, kernel_1, iterations=1)
	#cv2.imshow("erosion", erosion)
	dilation = cv2.dilate(blank, kernel_2, iterations=1)
	# cv2.imshow("dilation2", dilation)
	closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_2)
	# cv2.imshow("opening2", opening)
	_, b_thresh = cv2.threshold(closing[:, :, 0], 60, 255, cv2.THRESH_BINARY)
	# cv2.imshow("b_thresh", b_thresh)
	return b_thresh

def define_ROI(result, org_img, color, sign_list, x_coord_list):
	cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# if contourns were detected
	if len(cnts)>0:
		cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
		# for c in cnts_sorted:
		# Choise of the biggest area sign
		c = cnts_sorted[0]

		x, y, w, h = cv2.boundingRect(c)
		if color == 'yellow':
			print("yellow area: ", cv2.contourArea(c))
			x = int(x-0.25*w)
			y = int(y-0.25*h)
			w = int(1.5*w)
			h = int(1.5*h)
		elif color == 'red':
			print("red area: ", cv2.contourArea(c))
			x = int(x-0.1*w)
			y = int(y-0.1*h)
			w = int(1.2*w)
			h = int(1.2*h)	
		elif color == 'blue':
			print("blue area: ", cv2.contourArea(c))
			x = int(x-0.05*w)
			y = int(y-0.05*h)
			w = int(1.1*w)
			h = int(1.1*h)
		elif color == 'green':
			print("green area: ", cv2.contourArea(c))
			x = int(x-0.1*w)
			y = int(y-0.1*h)
			w = int(1.2*w)
			h = int(1.2*h)							
		hull = cv2.convexHull(c)
		# cv2.rectangle(org_img, (x, y), (int(x+w), int(y+h)), (0, 255, 0), 2)
		# cv2.drawContours(org_img, [hull], -1, (0, 255, 0), 2)
		# cv2.imshow("contours", org_img)
		mask = np.zeros_like(org_img)
		cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # Draw filled contour in mask
		cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)
		# cv2.imshow("mask", mask)
		out = np.zeros_like(org_img)  # Extract out the object and place into output image
		out[mask == 255] = org_img[mask == 255]
		x_pixel, y_pixel, _ = np.where(mask == 255)
		(topx, topy) = (np.min(x_pixel), np.min(y_pixel))
		(botx, boty) = (np.max(x_pixel), np.max(y_pixel))
		out = org_img[topx:botx + 1, topy:boty + 1]
		sign_list.append(out)
		x_coord_list.append(x)
		#cv2.imshow("out", out)
		#key = cv2.waitKey(0)

# org_img is the whle camera frame
def extract_close_sign(org_img):
	"""
	Gets as inpt the whole frame
	Returns the (square) ROI with the sign for the classifier
	"""
	sign_list = []
	x_coord_list=[]
	height, width = org_img.shape[:2]
	# org_img = cv2.cvtColor(org_img, cv2.COLOR_RGB2BGR)
	org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
	crop_img = org_img[40:190,380:]

	for color in filters_list:
		#cv2.imshow('crop_img', crop_img)
		if color == 'red':
			filtered_img = identify_red(crop_img, crop_img.copy())
			# cv2.imshow('filtered_img', filtered_img)
			define_ROI(filtered_img, crop_img, color, sign_list,x_coord_list)
		elif color == 'blue':
			filtered_img = identify_blue(crop_img, crop_img.copy())
			# cv2.imshow('filtered_img', filtered_img)
			define_ROI(filtered_img, crop_img, color, sign_list, x_coord_list)
		elif color == 'yellow':
			filtered_img = identify_yellow(crop_img, crop_img.copy())
			# cv2.imshow('filtered_img', filtered_img)
			define_ROI(filtered_img, crop_img, color, sign_list, x_coord_list)		
		elif color == 'green':
			filtered_img = identify_green(crop_img, crop_img.copy())
			# cv2.imshow('filtered_img', filtered_img)
			define_ROI(filtered_img, crop_img, color, sign_list, x_coord_list)	

	# To choose the sign most at the right
	if len(x_coord_list)>0:
		x_max = np.argmax(x_coord_list)
		choice = sign_list[x_max]
		return choice
	return None
