

#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px # this is another plotting library for interactive plot

from sklearn.model_selection import train_test_split
from sklearn import metrics, manifold # we will use the metrics and manifold learning modules from scikit-learn
from pathlib import Path # to interact with file paths
from PIL import Image # to interact with images
from tqdm import tqdm # progress bar
from pprint import pprint # pretty print (useful for a more readable print of objects like lists or dictionaries)
from IPython.display import clear_output # to clear the output of the notebook

import torch
import torch.nn as nn
import torchvision
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2 as cv
import os
import shutil

from Simulator.src.helper_functions import *




# DEFINITIONS
IN, OUT, CONV_LAYERS, FC_LAYERS, DROPOUT = 'IN', 'OUT', 'CONV_LAYERS', 'FC_LAYERS', 'DROPOUT'


MODEL_FOLDER = 'Simulator/models'
model_name = MODEL_FOLDER + '/lane_keeper.pt'
onnx_lane_keeper_path = MODEL_FOLDER + '/lane_keeper.onnx'



# NETWORK ARCHITECTURE
DROPOUT_PROB = 0.3
DEFAULT_CONV_LAYERS = nn.Sequential( #in = 32x32
            nn.Conv2d(1, 4, 5, 1), #out = 28
            nn.ReLU(True),
            nn.Dropout(p=DROPOUT_PROB),
            nn.MaxPool2d(2, 2), #out=14
            nn.BatchNorm2d(4),
            nn.Dropout(p=DROPOUT_PROB),
            nn.Conv2d(4, 4, 5, 1), #out = 10
            nn.ReLU(True),
            nn.Dropout(p=DROPOUT_PROB),
            nn.MaxPool2d(2, 2), #out=5
            nn.Dropout(p=DROPOUT_PROB),
            nn.Conv2d(4, 32, 5, 1), #out = 1
            nn.ReLU(True),
        )
DEFAULT_FC_LAYERS = nn.Sequential(
            nn.Linear(1*1*32, 16),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(16, 1),
        )

NET_PARAMS = {IN:32, OUT:1, CONV_LAYERS:DEFAULT_CONV_LAYERS, FC_LAYERS:DEFAULT_FC_LAYERS, DROPOUT:DROPOUT_PROB}


class HEstimator(nn.Module):
    def __init__(self, net_params=NET_PARAMS):
        super().__init__()
        self.conv = net_params[CONV_LAYERS]
        self.flatten = nn.Flatten()
        self.lin = net_params[FC_LAYERS]

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x


# IMAGE PREPROCESSING AND AUGMENTATION
import cv2 as cv
import numpy as np
from numpy.random import randint
from time import time, sleep

def preprocess_image(img, size=32, keep_bottom=0.66666667, canny1=100, canny2=200, blur=3):
    """
    Preprocesses an image to be used as input for the network.
    Note: the function modifies the image in place
    """
    #set associated parameters to None to skip the step
    skip_canny = canny1 == None or canny2 == None
    skip_blur = blur == None
    #check if its a valid image
    assert len(img.shape) == 3 or len(img.shape) == 2, "Invalid image shape"
    #check if the imge is grayscale
    img_is_gray = len(img.shape) == 2
    if not img_is_gray:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cut the top part
    img = img[int(img.shape[0]*(1-keep_bottom)):,:]
    #resize 1
    img = cv.resize(img, (2*size, 2*size))
    #canny
    if not skip_canny:
        img = cv.Canny(img, canny1, canny2)
    #blur
    if not skip_blur:
        img = cv.blur(img, (3,3), 0)
    #resize 2
    img = cv.resize(img, (size, size))
    return img

def augment_img(img, size=32, keep_bottom=0.66666667, canny1=100, canny2=200, blur=3, 
                max_tilt_fraction=0.1, noise_std=80):
    """
    Augments an image by applying random transformations
    Note: the function modifies the image in place
    """

    # preaugmentation
    img = cv.resize(img, (4*size, 4*size)) # 128x128

    #create random ellipses to simulate light from the sun
    light = np.zeros(img.shape, dtype=np.uint8)
    #add ellipses
    for j in range(2):
        cent = (randint(0, img.shape[0]), randint(0, img.shape[1]))
        axes_length = (randint(int(4*size/42.67),int(4*size/10.67)), randint(int(4*size/10.67), int(size*4/1.70))) #(randint(3, 12), randint(12, 75))
        angle = randint(0, 360)
        light = cv.ellipse(light, cent, axes_length, angle, 0, 360, 255, -1)
    #create an image of random white and black pixels
    light = cv.blur(light, (50,50))
    noise = randint(0, 2, size=img.shape, dtype=np.uint8)*255
    light = cv.subtract(light, noise)
    light = np.clip(light, 0, 51)
    light *= 5
    #add light to the image
    img = cv.add(img, light)

    # dilation/erosion
    r = randint(0, 5)
    if r == 0:
        #dilate
        kernel = np.ones((randint(1, 5), randint(1, 5)), np.uint8)
        img = cv.dilate(img, kernel, iterations=1)
    elif r == 1:
        #erode
        kernel = np.ones((randint(1, 5), randint(1, 5)), np.uint8)
        img = cv.erode(img, kernel, iterations=1)

    #preprocessing
    img = preprocess_image(img, size, keep_bottom, canny1, canny2, blur)

    # second augmentation
    #add random tilt
    max_offset = int(size*max_tilt_fraction)
    offset = randint(-max_offset, max_offset)
    img = np.roll(img, offset, axis=0)
    if offset > 0:
        img[:offset, :] = 0 #randint(0,255)
    elif offset < 0:
        img[offset:, :] = 0 # randint(0,255)

    #add noise 
    std = noise_std
    std = randint(1, std)
    noisem = randint(0, std, img.shape, dtype=np.uint8)
    img = cv.subtract(img, noisem)
    noisep = randint(0, std, img.shape, dtype=np.uint8)
    img = cv.add(img, noisep)

    return img


# DATASET
class MyDataset(Dataset):
    def __init__(self, dataset_file_path, he_distances=[0.5], size=32, keep_bottom=0.66666667, canny1=100, canny2=200, blur=3, 
                max_tilt_fraction=0.1, noise_std=80, device='cpu'):

        self.data = []
        print('decompressing...')
        path = path = np.load('sparcs/sparcs_path_precise.npy').T
        log = np.load(f'{dataset_file_path}.npz')
        imgs, locs = log['imgs'],log['locs']
        assert len(imgs) == len(locs), f'Invalid dataset, imgs and locs have different lengths: {len(imgs)} != {len(locs)}'
        assert locs.shape[1] == 3, f'Invalid dataset, locs must have shape (N,2), got {locs.shape}'
        dataset_length = len(imgs)
        print(f'Dataset: {dataset_file_path}\nDataset length: {dataset_length}')
        self.imgs = torch.zeros((2*dataset_length, 1, size, size), dtype=torch.float32)
        # cv.namedWindow('img', cv.WINDOW_NORMAL)
        # cv.resizeWindow('img', 600, 600)
        for i, (img, (x,y,yaw)) in tqdm(enumerate(zip(imgs,locs))):
            hes = []
            for d in he_distances:
                he, _, _ = get_heading_error(x,y,yaw, path,d)
                hes.append(he)
            hes = np.array(hes)
            self.data.append(hes)
            self.data.append(-hes)
            img = augment_img(img, size, keep_bottom, canny1, canny2, blur, max_tilt_fraction, noise_std)
            fimg = cv.flip(img, 1)
            img = img[:, :,np.newaxis]
            fimg = fimg[:, :,np.newaxis]
            #permute the axis (2, 0, 1)
            img = np.transpose(img, (2, 0, 1))
            fimg = np.transpose(fimg, (2, 0, 1))
            #convert to float32
            img = img.astype(np.float32)
            fimg = fimg.astype(np.float32)
            img = torch.from_numpy(img)
            fimg = torch.from_numpy(fimg)

            assert self.imgs[i].shape == img.shape, f'Invalid image shape: {img.shape} != {self.imgs[i].shape}'
            self.imgs[2*i] = img
            self.imgs[2*i+1] = fimg
            # cv.imshow('img', self.imgs[i])
            # if cv.waitKey(100) == 27:
            #     cv.destroyAllWindows()  
            #     break
        self.data = torch.from_numpy(np.array(self.data, dtype=np.float32))
        #put data on gpu
        self.imgs = self.imgs.to(device)
        self.data = self.data.to(device)

        print(f'size in RAM: {self.imgs.size(0)*self.imgs.size(1)*self.imgs.size(2)*self.imgs.size(3)*4/1024/1024/1024} GB')

    def __len__(self):
        # The length of the dataset is simply the length of the self.data list
        return len(self.data)

    def __getitem__(self, idx):
        return self.imgs[idx], self.data[idx]


# TRAINING FUNCTION
def train_epoch(net, dataloader, regr_loss_fn, optimizer, L1_lambda=0.0, L2_lambda=0.0):
    # Set the net to training mode
    net.train() #train
    # Initialize the loss
    he_losses = []
    # Loop over the training batches
    for (input, regr_label) in tqdm(dataloader):
        # Zero the gradients
        optimizer.zero_grad()
        # Compute the output
        output = net(input)
        he = output[:, 0]
        he_label = regr_label[:, 0]
        # Compute the losses
        he_loss = 1.0*regr_loss_fn(he, he_label)
        #L1 regularization
        L1_norm = sum(p.abs().sum() for p in net.conv.parameters())
        L1_loss = L1_lambda * L1_norm 
        #L2 regularization
        L2_norm = sum(p.pow(2).sum() for p in net.conv.parameters())
        L2_loss = L2_lambda * L2_norm
        #total loss
        loss = he_loss + L1_loss + L2_loss
        # Compute the gradients
        loss.backward()
        # Update the weights
        optimizer.step()
        #batch loss
        he_losses.append(he_loss.detach().cpu().numpy())

    # Return the average training loss
    he_loss = np.mean(he_losses)
    return he_loss

    # VALIDATION FUNCTION
def val_epoch(net, val_dataloader, regr_loss_fn):
    net.eval()
    he_losses = []
    for (input, regr_label) in tqdm(val_dataloader):
        output = net(input)
        regr_out = output
        he = regr_out[:, 0]
        he_label = regr_label[:, 0]
        he_loss = 1.0*regr_loss_fn(he, he_label)
        he_losses.append(he_loss.detach().cpu().numpy())
    return np.mean(he_losses)