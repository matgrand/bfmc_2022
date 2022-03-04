#!/usr/bin/python3

from dis import dis
import numpy as np
import cv2 as cv

IMG_SIZE = (320, 240)


class SimpleController():
    def __init__(self, k1=1.0,k2=1.0,k3=1.0, ff=1.0, lane_keeper_path="models/lane_keeper.onnx"):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.e1 = 0.0
        self.e2 = 0.0
        self.e3 = 0.0
        self.ff = ff

        #load neural network
        self.lane_keeper = cv.dnn.readNetFromONNX(lane_keeper_path)

    def get_nn_control(self, frame, vd):
        frame = cv.resize(frame, IMG_SIZE)
        #convert to gray
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #set the top section of the image to gray
        frame[:int(IMG_SIZE[1]*0.5),:] = 127
        #convert to binary
        # frame = np.where(frame > 220, 255, 0).astype(np.uint8)
        #blur the image
        frame = cv.GaussianBlur(frame, (5,5), 0)

        cv.imshow("frame", frame)
        cv.waitKey(1)
        blob = cv.dnn.blobFromImage(frame, 1.0, IMG_SIZE, 0, swapRB=True, crop=False)
        assert blob.shape == (1, 1, 240, 320), f"blob shape: {blob.shape}"
        self.lane_keeper.setInput(blob)
        output = self.lane_keeper.forward()
        # print(f'output: {output.shape}')
        output = output[0]

        e2,e3,dist,curv = self.unpack_network_output(output)

        net_out = (e2,e3,dist,curv)

        curvature_ahead = curv 
        output_angle = self.ff*curvature_ahead - self.k2 * e2 - self.k3 * e3
        output_speed = vd - self.k1 * self.e1
        return output_speed, output_angle, net_out
    
    def unpack_network_output(self, output):
        e2 = output[0]
        e3 = output[1]
        dist = output[2]
        curv = output[3]
        return e2,e3,dist,curv
    

