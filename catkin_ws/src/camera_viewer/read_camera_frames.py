#!/usr/bin/python3
# Functional libraries
import rospy
import numpy as np
from cv_bridge import CvBridge
import cv2 as cv
import os

#from utils.srv import subscribing
from sensor_msgs.msg import Image


ROI = [30,260,-100,640]
# ROI = [0,480,0,640]


def camera_callback(data):
    global image_data
    global bridge
    image_data = bridge.imgmsg_to_cv2(data, "bgr8")
    # cv.imshow("image", image_data)
    # print(image_data)


#main
if __name__ == '__main__':

    rospy.init_node('image_viewer', anonymous=False)
    rospy.sleep(1)  # wait for publisher to register to roscore

    #define bridge
    bridge = CvBridge()
    image_data = np.zeros((480,640,3), np.uint8)
    prev_img = image_data

    #finds all images in imgs folder
    img_names = [f for f in os.listdir('imgs') if f.endswith('.png')]
    img_index = len(img_names)


    subscriber = rospy.Subscriber("/automobile/image_raw", Image, camera_callback)

    save_images = False

    print('Press s to save images\nPress d for deleting all images\n press esc to exit')

    while not rospy.is_shutdown():
        if image_data is not None:
            # cv.imshow("image", image_data)
            if (prev_img-image_data).any() > 0:
                prev_img = image_data
                roi = image_data[ROI[0]:ROI[1], ROI[2]:ROI[3]]
                cv.imshow("ROI", roi)
                if save_images:
                    #save roi
                    cv.imwrite(f"imgs/img_{img_index+1}.png", roi)
                    img_index += 1
                    print(f"img_{img_index+1}.png saved")

            # print(image_data)
            key = cv.waitKey(1)
            #if key is 's' save
            if key == ord('s'):
                save_images = not save_images
                print("Saving images: ", save_images)
            if key == ord('d'):
                print("Deleting images")
                img_names = [f for f in os.listdir('imgs') if f.endswith('.png')]
                for f in img_names:
                    os.remove(f"imgs/{f}")
                img_index = 0
                print("Images deleted")
            if key == 27:
                break

