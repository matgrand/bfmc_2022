

#Imports
from tracemalloc import start
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
import torch.onnx
from copy import deepcopy
import IPython
from time import time, sleep


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

def create_net(architecture, img_size, dropout):
    return  HEstimator() #TODO

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
    if not img.shape == (4*size, 4*size):
        img = cv.resize(img, (4*size, 4*size))


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
                noise_std=80, max_tilt_fraction=0.1):
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


def calculate_hes(locs, he_dist):
    path = np.load('sparcs/sparcs_path_precise.npy', allow_pickle=True).T
    hes = []
    for x,y,yaw in locs:
        he, _, _ = get_heading_error(x,y,yaw, path, he_dist)
        hes.append(he)
    return np.array(hes)


# DATASET
def prepare_ds(ds_params):
    #name = f'ds_sn{steer_noise_level:.0f}_he{100*he_distance:.0f}_canny{canny1}_{canny2}_blur{blur:.0f}_noise{img_noise:.0f}_keep{100*keep_bottom:.0f}_size{img_size:.0f}_length{ds_length:.0f}'
    #params = {'name':name, 'steer_noise_level': steer_noise_level, 'he_distance': he_distance, 'canny1': canny1, 'canny2': canny2, 'blur': blur, 'img_noise': img_noise, 'keep_bottom': keep_bottom, 'img_size': img_size, 'ds_length': ds_length}
    
    
    name, steer_noise_level, he_distance, canny1, canny2, blur, img_noise, keep_bottom, img_size, ds_length = ds_params['name'], ds_params['steer_noise_level'], ds_params['he_distance'], ds_params['canny1'], ds_params['canny2'], ds_params['blur'], ds_params['img_noise'], ds_params['keep_bottom'], ds_params['img_size'], ds_params['ds_length']
    #check if dataset is already in tmp folder
    ds_path = f'tmp/dss/{name}.npz'
    if os.path.exists(ds_path):
        return
    
    #check if steer_noise_level is already in tmp
    sn_path = f'tmp/dss/ds_{steer_noise_level}.npz'
    if not os.path.exists(sn_path):
        #unzip and save the ds unzipped
        tmp_ds = np.load(f'saved_tests/sim_ds_{steer_noise_level}.npz', allow_pickle=True)
        imgs, locs = tmp_ds['imgs'], tmp_ds['locs']
        np.savez(sn_path, imgs=imgs, locs=locs)
    
    #check if he_distance is already in tmp
    hes_path = f'hes/hes_{steer_noise_level}_{100*he_distance:.0f}.npz'
    if not os.path.exists(hes_path):
        #load the dataset
        tmp_ds = np.load(sn_path, allow_pickle=True)
        imgs, locs = tmp_ds['imgs'], tmp_ds['locs']
        #get the he
        hes = calculate_hes(locs, he_distance)
        #save the dataset
        np.savez(hes_path, hes=hes, he_distance=he_distance, steer_noise_level=steer_noise_level)
    
    #load required components
    ds = np.load(sn_path, allow_pickle=True)
    hes = np.load(hes_path, allow_pickle=True)
    imgs, locs = ds['imgs'], ds['locs']
    hes = hes['hes']
    to_load = min(imgs.shape[0], ds_length)
    #choose to_load random indexes
    indexes = np.random.choice(imgs.shape[0], to_load, replace=False)
    imgs, locs, hes = imgs[indexes], locs[indexes], hes[indexes]

    #augment the dataset
    aug_imgs = []
    for img in imgs:
        aug_img = augment_img(img, img_size, keep_bottom, canny1, canny2, blur, img_noise)
        aug_imgs.append(aug_img)
    aug_imgs = np.array(aug_imgs)
    #save the dataset
    np.savez(ds_path, imgs=aug_imgs, locs=locs, hes=hes, name=name, steer_noise_level=steer_noise_level, he_distance=he_distance, canny1=canny1, canny2=canny2, blur=blur, img_noise=img_noise, keep_bottom=keep_bottom, img_size=img_size)

class MyDataset(Dataset):
    def __init__(self, ds_name, device='cpu'):
        #load the dataset
        ds = np.load(f'tmp/{ds_name}.npz', allow_pickle=True)
        self.img_size = ds['img_size']
        self.imgs, self.hes = ds['imgs'], ds['hes']
        assert len(self.imgs) == len(self.hes)
        self.imgs = self.imgs.astype(np.float32)
        self.imgs = self.imgs[:, np.newaxis, :, :]
        self.hes = self.hes.astype(np.float32)
        self.hes = self.hes[:, np.newaxis]
        #convert to tensors
        self.imgs = torch.from_numpy(self.imgs).to(device)
        self.hes = torch.from_numpy(self.hes).to(device)

    def __len__(self):
        # The length of the dataset is simply the length of the self.data list
        return len(self.hes)

    def __getitem__(self, idx):
        return self.imgs[idx], self.hes[idx]


# TRAINING 
def train_epoch(net, dataloader, regr_loss_fn, optimizer, L1_lambda=0.0, L2_lambda=0.0):
    # Set the net to training mode
    net.train() #train
    # Initialize the loss
    he_losses = []
    # Loop over the training batches
    for (input, regr_label) in dataloader:
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

def val_epoch(net, val_dataloader, regr_loss_fn):
    net.eval()
    he_losses = []
    for (input, regr_label) in val_dataloader:
        output = net(input)
        regr_out = output
        he = regr_out[:, 0]
        he_label = regr_label[:, 0]
        he_loss = 1.0*regr_loss_fn(he, he_label)
        he_losses.append(he_loss.detach().cpu().numpy())
    return np.mean(he_losses)

def train(params, device='cpu'):
    name, ds_name, architecture, batch_size, lr, epochs, L1_lambda, L2_lambda, weight_decay, dropout = params['name'], params['ds_name'], params['architecture'], params['batch_size'], params['lr'], params['epochs'], params['L1_lambda'], params['L2_lambda'], params['weight_decay'], params['dropout']
    
    # print(f'Name: {name}')

    #check if the training has already been done
    comb_path = f'tmp/training_combinations/{name}.npz'
    if os.path.exists(comb_path):
        return

    #create dataset
    ds = MyDataset(ds_name, device)
    
    #create model
    net = create_net(architecture, ds.img_size, dropout)
    net.to(device)

    #create dataloader
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    regr_loss_fn1 = nn.MSELoss() #before epochs/2
    regr_loss_fn2 = nn.MSELoss() #after epochs/2 for finetuning

    #train
    best_val = np.inf
    best_epoch = 0
    best_model = None
    losses = np.zeros((epochs, 2))
    for epoch in range(epochs):
        regr_loss_fn = regr_loss_fn1 if epoch < epochs//2 else regr_loss_fn2
        he_loss = train_epoch(net, train_dataloader, regr_loss_fn, optimizer, L1_lambda, L2_lambda)
        val_he_loss = val_epoch(net, val_dataloader, regr_loss_fn)
        losses[epoch, 0] = he_loss
        losses[epoch, 1] = val_he_loss
        if val_he_loss < best_val:
            best_val = val_he_loss
            best_epoch = epoch
            best_model = deepcopy(net)
            # torch.save(net.state_dict(), f'tmp/{name}.pt')
    
    # EVALUATE ON TEST SET (UNSEEN DATA)
    # net.load_state_dict(torch.load(f'tmp/{name}.pt'))
    net = best_model
    he_loss = val_epoch(net, val_dataloader, regr_loss_fn)

    #export to onnx
    dummy_input = torch.randn(1, 1, ds.img_size, ds.img_size, device=device)
    torch.onnx.export(net, dummy_input, f"tmp/models/{name}.onnx", verbose=False)

    #save losses
    # np.save(f'tmp/{name}_losses.npy', losses)
    torch.save(best_model.state_dict(), f'tmp/models/{name}.pt')
    np.savez(comb_path, losses=losses, net=net, name=name, ds_name=ds_name, 
                architecture=architecture, batch_size=batch_size, lr=lr, epochs=epochs, 
                L1_lambda=L1_lambda, L2_lambda=L2_lambda, weight_decay=weight_decay, dropout=dropout,
                best_epoch=best_epoch, best_val=best_val)

# EVALUATION

REAL_EVALUATION_DATASETS = ['acw0', 'acw2', 'acw4', 'acw6', 'acw8', 'acw10', 'acw12', 'acw14', 'cw0', 'cw2', 'cw4', 'cw6', 'cw8', 'cw10', 'cw12', 'cw14']
SIM_EVALUATION_DATASETS = ['acw0_SIM', 'acw2_SIM', 'acw4_SIM', 'acw6_SIM', 'acw8_SIM', 'acw10_SIM', 'acw12_SIM', 'acw14_SIM', 'cw0_SIM', 'cw2_SIM', 'cw4_SIM', 'cw6_SIM', 'cw8_SIM', 'cw10_SIM', 'cw12_SIM', 'cw14_SIM']
ALL_EVALUATION_DATASETS = REAL_EVALUATION_DATASETS + SIM_EVALUATION_DATASETS
DEFAULT_EVALUATION_DATASETS = REAL_EVALUATION_DATASETS

def evaluate(params, eval_datasets=DEFAULT_EVALUATION_DATASETS, device='cpu'):
    name = params['name']

    #check if the name exists
    comb_path = f'tmp/training_combinations/{name}.npz'
    assert os.path.exists(comb_path), f'Name {name} does not exist'
    #load model
    npz = np.load(comb_path, allow_pickle=True)
    losses, net, name, ds_name, architecture, batch_size, lr, epochs, L1_lambda, L2_lambda, weight_decay, dropout, best_epoch, best_val = npz['losses'], npz['net'], npz['name'], npz['ds_name'], npz['architecture'], npz['batch_size'], npz['lr'], npz['epochs'], npz['L1_lambda'], npz['L2_lambda'], npz['weight_decay'], npz['dropout'], npz['best_epoch'], npz['best_val']
    ds = np.load(f'tmp/dss/{ds_name}.npz', allow_pickle=True)
    train_imgs, train_locs, train_hes, train_steer_noise_level, he_distance, canny1, canny2, blur, img_noise, keep_bottom, img_size = ds['imgs'], ds['locs'], ds['hes'], ds['steer_noise_level'], ds['he_distance'], ds['canny1'], ds['canny2'], ds['blur'], ds['img_noise'], ds['keep_bottom'], ds['img_size']
    net = net.item()

    #load datasets
    to_save = []

    MSEs = []
    for ev_ds in eval_datasets:
        ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{name}.npz'
        if not os.path.exists(ds_train_combination_path):
            ds_path = f'tmp/real_dss/{ev_ds}.npz'
            if not os.path.exists(ds_path):
                tmp = np.load(f'saved_tests/{ev_ds}.npz', allow_pickle=True)
                timgs, tlocs = tmp['imgs'], tmp['locs']
                np.savez(ds_path, imgs=timgs, locs=tlocs)
                print(f'Generating {ev_ds}')
            ds_npz = np.load(ds_path, allow_pickle=True)
            imgs, locs = ds_npz['imgs'], ds_npz['locs']

            #create hes
            hes_path = f'tmp/hes/{ev_ds}_{he_distance*100:.0f}.npz'
            if not os.path.exists(hes_path):
                #get the he
                hes = calculate_hes(locs, he_distance)
                np.savez(hes_path, hes=hes, he_distance=he_distance)
                print(f'Generating {hes_path}')
            hes = np.load(hes_path, allow_pickle=True)['hes']

            #preprocess images
            preproc_imgs_path = f'tmp/real_dss/{ev_ds}_preproc_imgs_{img_size}_{canny1}_{canny2}_{blur}_{img_noise}_{100*keep_bottom:.0f}.npz'
            if not os.path.exists(preproc_imgs_path):
                timgs = np.zeros((len(imgs), img_size, img_size), dtype=np.uint8)
                for i, img in enumerate(imgs):
                    timgs[i] = preprocess_image(img=img,size=int(img_size), keep_bottom=float(keep_bottom), canny1=int(canny1), canny2=int(canny2), blur=int(blur))
                np.savez(preproc_imgs_path, imgs=timgs)
                print(f'Generating {preproc_imgs_path}')
            imgs = np.load(preproc_imgs_path, allow_pickle=True)['imgs'].astype(np.float32)
            imgs = torch.from_numpy(imgs[:,np.newaxis,:,:]).to(device)

            #run inference         
            assert isinstance(net, HEstimator), f'Net is not an HEstimator, it is a {type(net)}'
            net.to(device)
            net.eval()
            est_hes = np.zeros_like(hes)
            with torch.no_grad():
                est_hes = net(imgs).cpu().numpy()
            
            #save
            to_save.append({'ev_ds': ev_ds, 'hes': hes, 'est_hes': est_hes, 'he_distance': he_distance, 'combination':params})  

            #calculate MSE
            mse = np.mean(np.square(hes - est_hes))

            #save
            np.savez(ds_train_combination_path, eval_datasets=eval_datasets, saved=to_save, mse=mse, comb_name=name)
        #load
        npz = np.load(ds_train_combination_path, allow_pickle=True)
        MSEs.append(npz['mse'])
        eval_datasets = npz['eval_datasets']
    return np.mean(np.array(MSEs))

def get_best_result(trainings_combinations, eval_datasets=DEFAULT_EVALUATION_DATASETS, device='cpu'):
    best_combination = None
    best_MSE = np.inf
    all_MSE = []
    for comb in trainings_combinations:
        MSEs = evaluate(comb, eval_datasets=eval_datasets, device=device)
        MSE = np.mean(MSEs)
        all_MSE.append(MSE)
        if MSE < best_MSE:
            best_MSE = MSE
            best_combination = comb
    return best_combination, best_MSE, all_MSE





























