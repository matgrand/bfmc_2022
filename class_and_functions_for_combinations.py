

#Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
from time import time, sleep
from Simulator.src.helper_functions import *
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# DEFINITIONS
IN, OUT, CONV_LAYERS, FC_LAYERS, DROPOUT = 'IN', 'OUT', 'CONV_LAYERS', 'FC_LAYERS', 'DROPOUT'

MODEL_FOLDER = 'Simulator/models'
model_name = MODEL_FOLDER + '/lane_keeper.pt'
onnx_lane_keeper_path = MODEL_FOLDER + '/lane_keeper.onnx'

# NETWORK ARCHITECTURE
class HEstimator(nn.Module):
    def __init__(self,dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential( #in = 32x32
            nn.Conv2d(1, 4, 5, 1), #out = 28
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(2, 2), #out=14
            nn.BatchNorm2d(4),
            nn.Dropout(p=dropout),
            nn.Conv2d(4, 4, 5, 1), #out = 10
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.MaxPool2d(2, 2), #out=5
            nn.Dropout(p=dropout),
            nn.Conv2d(4, 32, 5, 1), #out = 1
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten()
        self.lin = nn.Sequential(
            nn.Linear(1*1*32, 16),
            nn.ReLU(True),
            # nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.lin(x)
        return x

def create_net(architecture, img_size, dropout):
    net = HEstimator(dropout=dropout)
    #load the base network
    base_net_name = f'tmp/models/base_{architecture}_{img_size}.pt'
    net.load_state_dict(torch.load(base_net_name))
    return net

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
    skip_canny = canny1 == None or canny2 == None or canny1==0 or canny2==0
    skip_blur = blur == 0
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
        img = cv.blur(img, (blur,blur), 0)
    #resize 2
    img = cv.resize(img, (size, size))
    return img

def get_est_heading_error(img, onnx_model, size=32, keep_bottom=.8, canny1=100, canny2=200, blur=3):
    img = preprocess_image(img, size, keep_bottom, canny1, canny2, blur)
    img_flipped = cv.flip(img, 1) 
    #stack the 2 images
    images = np.stack((img, img_flipped), axis=0) 
    blob = cv.dnn.blobFromImages(images, 1.0, (size, size), 0, swapRB=True, crop=False) 
    onnx_model.setInput(blob)
    out = onnx_model.forward()
    output = out[0]
    output_flipped = out[1] 
    est_he = (output - output_flipped)/2
    return est_he

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
    std = noise_std if noise_std > 1 else 2
    std = randint(1, std)
    noisem = randint(0, std, img.shape, dtype=np.uint8)
    img = cv.subtract(img, noisem)
    noisep = randint(0, std, img.shape, dtype=np.uint8)
    img = cv.add(img, noisep)

    return img


def calculate_hes(locs, he_dist):
    path = my_load('sparcs/sparcs_path_precise.npy', allow_pickle=True).T
    hes = []
    for x,y,yaw in locs:
        he, _, _ = get_heading_error(x,y,yaw, path, he_dist)
        hes.append(he)
    return np.array(hes)

all_names_used = []

def check_name(name):
    global all_names_used
    if name in all_names_used:
        raise ValueError("Name already used")
    else:
        all_names_used.append(name)

mega_dict_in_ram = {}
def my_load(name, allow_pickle=True):
    global mega_dict_in_ram
    if name not in all_names_used:
        all_names_used.append(name)
    if name in mega_dict_in_ram.keys():
        return mega_dict_in_ram[name]
    else:
        mega_dict_in_ram[name] = np.load(name, allow_pickle=allow_pickle)
        return mega_dict_in_ram[name]

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
        tmp_ds = my_load(f'saved_tests/sim_ds_{steer_noise_level}.npz', allow_pickle=True)
        imgs, locs = tmp_ds['imgs'], tmp_ds['locs']
        np.savez(sn_path, imgs=imgs, locs=locs)
        check_name(sn_path)
        # print(f'Unzipped and saved {ds_path}')
    
    #check if he_distance is already in tmp
    hes_path = f'tmp/hes/hes_{steer_noise_level}_{100*he_distance:.0f}.npz'
    if not os.path.exists(hes_path):
        #load the dataset
        tmp_ds = my_load(sn_path, allow_pickle=True)
        imgs, locs = tmp_ds['imgs'], tmp_ds['locs']
        #get the he
        hes = calculate_hes(locs, he_distance)
        #save the dataset
        np.savez(hes_path, hes=hes, he_distance=he_distance, steer_noise_level=steer_noise_level)
        check_name(hes_path)
        # print(f'Calculated and saved {hes_path}')
    
    #load required components
    ds = my_load(sn_path, allow_pickle=True)
    hes = my_load(hes_path, allow_pickle=True)
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
    check_name(ds_path)

def analyze_ds(ds_params):
    name, steer_noise_level, he_distance, canny1, canny2, blur, img_noise, keep_bottom, img_size, ds_length = ds_params['name'], ds_params['steer_noise_level'], ds_params['he_distance'], ds_params['canny1'], ds_params['canny2'], ds_params['blur'], ds_params['img_noise'], ds_params['keep_bottom'], ds_params['img_size'], ds_params['ds_length']
    #check if dataset is already in tmp folder
    ds_path = f'tmp/dss/{name}.npz'
    assert os.path.exists(ds_path), f'{ds_path} does not exist'
    ds = my_load(ds_path, allow_pickle=True)
    imgs, locs, hes, name, steer_noise_level, he_distance, canny1, canny2, blur, img_noise, keep_bottom, img_size = ds['imgs'], ds['locs'], ds['hes'], ds['name'], ds['steer_noise_level'], ds['he_distance'], ds['canny1'], ds['canny2'], ds['blur'], ds['img_noise'], ds['keep_bottom'], ds['img_size']
    path = my_load('sparcs/sparcs_path_precise.npy', allow_pickle=True).T
    path_yaws = np.zeros(path.shape[0])
    for i in range(path.shape[0]-1):
        path_yaws[i] = np.arctan2(path[i+1,1]-path[i,1], path[i+1,0]-path[i,0])
    path_yaws[-1] = path_yaws[-2]

    dists = np.zeros(locs.shape[0])
    yaw_dists = np.zeros(locs.shape[0])
    for i, (x,y,yaw) in enumerate(locs):
        idx_closest_p = np.argmin(np.linalg.norm(path - np.array([x,y]), axis=1))
        dists[i] = np.linalg.norm(path[idx_closest_p] - np.array([x,y]))
        assert path_yaws[idx_closest_p].shape == yaw.shape, f'{path_yaws[idx_closest_p].shape} != {yaw.shape}'
        yaw_dists[i] = np.abs(diff_angle(path_yaws[idx_closest_p], yaw))
    
    
    print(f'Steer noise: {steer_noise_level}deg -- avg dist: {np.mean(dists):.4f}m, std: {np.std(dists):.4f}m -- avg yaw dist: {np.mean(yaw_dists):.4f}rad, std: {np.std(yaw_dists):.4f}rad')

    fig, ax = plt.subplots(1,1, figsize=(10,5))
    ax.plot(locs[:,0], locs[:,1], 'b.')
    ax.plot(path[:,0], path[:,1], 'r.')
    ax.set_title('Path and locations')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    #same scale
    ax.set_aspect('equal', 'box')
    ax.legend(['Locations', 'Path'])

    plt.show()

def visualize_ds(imgs):
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    for img in imgs:
        cv.imshow('img', img)
        if cv.waitKey(50) == 27:
            break
    cv.destroyAllWindows()
    



class MyDataset(Dataset):
    def __init__(self, ds_name, device='cpu'):
        #load the dataset
        ds = my_load(f'tmp/dss/{ds_name}.npz', allow_pickle=True)
        self.img_size = ds['img_size']
        self.imgs, self.hes = ds['imgs'], ds['hes']

        #add flipped images
        flipped_imgs = np.flip(self.imgs, axis=2)
        flipped_hes = - self.hes
        self.imgs = np.concatenate((self.imgs, flipped_imgs), axis=0)
        self.hes = np.concatenate((self.hes, flipped_hes), axis=0)

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

def reset_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.reset_parameters()

def train(params, device='cpu'):
    #reset everything
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    check_name(comb_path)

# EVALUATION

REAL_EVALUATION_DATASETS = ['acw0', 'acw2', 'acw4', 'acw6', 'acw8', 'acw10', 'acw12', 'acw14', 'cw0', 'cw2', 'cw4', 'cw6', 'cw8', 'cw10', 'cw12', 'cw14']
SIM_EVALUATION_DATASETS = ['acw0_SIM', 'acw2_SIM', 'acw4_SIM', 'acw6_SIM', 'acw8_SIM', 'acw10_SIM', 'acw12_SIM', 'acw14_SIM', 'cw0_SIM', 'cw2_SIM', 'cw4_SIM', 'cw6_SIM', 'cw8_SIM', 'cw10_SIM', 'cw12_SIM', 'cw14_SIM']
REAL_NOISY_DATASETS = ['cw6', 'acw6' ,'cw8', 'acw8', 'acw10', 'cw10']#['acw10', 'acw12', 'acw14', 'cw10', 'cw12', 'cw14']
REAL_CLEAN_DATASETS = ['acw0', 'acw2', 'acw4', 'cw0', 'cw2', 'cw4']
SIM_NOISY_DATASETS = ['acw10_SIM', 'acw12_SIM', 'acw14_SIM', 'cw10_SIM', 'cw12_SIM', 'cw14_SIM']
ALL_EVALUATION_DATASETS = REAL_EVALUATION_DATASETS + SIM_EVALUATION_DATASETS
DEFAULT_EVALUATION_DATASETS = ALL_EVALUATION_DATASETS
LIST_REAL_DATASETS = [REAL_CLEAN_DATASETS, REAL_NOISY_DATASETS, REAL_CLEAN_DATASETS + REAL_NOISY_DATASETS]
LIST_REAL_DATASETS_NAMES = ['Clean datasets', 'Noisy datasets', 'All evaluation datasets']

def evaluate(params, eval_datasets=DEFAULT_EVALUATION_DATASETS, device='cpu', show_imgs=False):
    name = params['name']
    check_name(name)

    #check if the name exists
    comb_path = f'tmp/training_combinations/{name}.npz'
    assert os.path.exists(comb_path), f'Name {name} does not exist'
    #load model
    npz = my_load(comb_path, allow_pickle=True)
    losses, net, name, ds_name, architecture, batch_size, lr, epochs, L1_lambda, L2_lambda, weight_decay, dropout, best_epoch, best_val = npz['losses'], npz['net'], npz['name'], npz['ds_name'], npz['architecture'], npz['batch_size'], npz['lr'], npz['epochs'], npz['L1_lambda'], npz['L2_lambda'], npz['weight_decay'], npz['dropout'], npz['best_epoch'], npz['best_val']
    ds = my_load(f'tmp/dss/{ds_name}.npz', allow_pickle=True)
    train_imgs, train_locs, train_hes, train_steer_noise_level, he_distance, canny1, canny2, blur, img_noise, keep_bottom, img_size = ds['imgs'], ds['locs'], ds['hes'], ds['steer_noise_level'], ds['he_distance'], ds['canny1'], ds['canny2'], ds['blur'], ds['img_noise'], ds['keep_bottom'], ds['img_size']
    net = net.item()
    net.to(device)
    net.eval()

    for ev_ds in eval_datasets:
        ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{name}.npz'
        if not os.path.exists(ds_train_combination_path):
            #create dataset
            ds_path = f'tmp/real_dss/{ev_ds}.npz'
            if not os.path.exists(ds_path):
                tmp = my_load(f'saved_tests/{ev_ds}.npz', allow_pickle=True)
                timgs, tlocs = tmp['imgs'], tmp['locs']
                np.savez(ds_path, imgs=timgs, locs=tlocs)
                check_name(ds_path)
                # print(f'Generating {ev_ds}')
            npz = my_load(ds_path, allow_pickle=True)
            imgs, locs = npz['imgs'], npz['locs']
            
            #create hes
            hes_path = f'tmp/hes/{ev_ds}_{he_distance*100:.0f}.npz'
            if not os.path.exists(hes_path):
                #get the he
                hes = calculate_hes(locs, he_distance)
                np.savez(hes_path, hes=hes, he_distance=he_distance)
                check_name(hes_path)
                # print(f'Generating {hes_path}')
            hes = my_load(hes_path, allow_pickle=True)['hes']

            #preprocess images
            preproc_imgs_path = f'tmp/real_dss/{ev_ds}_preproc_imgs_{img_size}_{canny1}_{canny2}_{blur}_{img_noise}_{100*keep_bottom:.0f}.npz'
            if not os.path.exists(preproc_imgs_path):
                timgs = np.zeros((len(imgs), img_size, img_size), dtype=np.uint8)
                for i, img in enumerate(imgs):
                    timgs[i] = preprocess_image(img=img,size=int(img_size), keep_bottom=float(keep_bottom), canny1=int(canny1), canny2=int(canny2), blur=int(blur))
                np.savez(preproc_imgs_path, imgs=timgs)
                check_name(preproc_imgs_path)
                # print(f'Generating {preproc_imgs_path}')
            imgs = my_load(preproc_imgs_path, allow_pickle=True)['imgs'].astype(np.float32)

            if show_imgs: visualize_ds(imgs)

            imgs = torch.from_numpy(imgs[:,np.newaxis,:,:]).to(device)

            #run inference         
            assert isinstance(net, HEstimator), f'Net is not an HEstimator, it is a {type(net)}'
            est_hes = np.zeros_like(hes)
            with torch.no_grad():
                est_hes = net(imgs).cpu().numpy()
            
            #calculate MSE
            #make hes and est_hes the same shape
            hes = hes.reshape(-1)
            est_hes = est_hes.reshape(-1)
            se = np.square(hes - est_hes)
            assert se.shape == hes.shape, f'se.shape {se.shape} != hes.shape {hes.shape}'
            mse = np.mean(se)

            #save
            np.savez(ds_train_combination_path, ev_ds=ev_ds, hes=hes, est_hes=est_hes, he_distance=he_distance, mse=mse, comb_name=name)
            check_name(ds_train_combination_path)
        else:
            if show_imgs:
                preproc_imgs_path = f'tmp/real_dss/{ev_ds}_preproc_imgs_{img_size}_{canny1}_{canny2}_{blur}_{img_noise}_{100*keep_bottom:.0f}.npz'
                imgs = my_load(preproc_imgs_path, allow_pickle=True)['imgs'].astype(np.float32)
                visualize_ds(imgs)


def get_best_result(training_combinations, eval_datasets=DEFAULT_EVALUATION_DATASETS, device='cpu'):
    all_mses = np.zeros((len(training_combinations), len(eval_datasets)))
    for i, comb in enumerate(tqdm(training_combinations)):
        for j,ev_ds in enumerate(eval_datasets):
            comb_name = comb['name']
            ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{comb_name}.npz'
            npz = my_load(ds_train_combination_path, allow_pickle=True)
            mse = npz['mse']
            all_mses[i,j] = mse
    MSEs = np.mean(all_mses, axis=1)
    best_comb = training_combinations[np.argmin(MSEs)]
    best_MSE = np.min(MSEs)
    return best_comb, best_MSE, MSEs

def get_all_paramters_dict(training_comb):
    name = training_comb['name']
    #check if the name exists
    comb_path = f'tmp/training_combinations/{name}.npz'
    assert os.path.exists(comb_path), f'Name {name} does not exist'
    npz = my_load(comb_path, allow_pickle=True)
    ds_name = npz['ds_name']
    ds = my_load(f'tmp/dss/{ds_name}.npz', allow_pickle=True)
    to_ret = {}
    to_ret['imgs']              = ds['imgs']
    to_ret['locs']              = ds['locs']
    to_ret['hes']               = ds['hes']
    to_ret['steer_noise_level'] = ds['steer_noise_level']
    to_ret['he_distance']       = ds['he_distance']
    to_ret['canny1']            = ds['canny1']
    to_ret['canny2']            = ds['canny2']
    to_ret['blur']              = ds['blur']
    to_ret['img_noise']         = ds['img_noise']
    to_ret['keep_bottom']       = ds['keep_bottom']
    to_ret['img_size']          = ds['img_size']
    to_ret['losses']            = npz['losses']
    to_ret['net']               = npz['net']
    to_ret['name']              = npz['name']
    to_ret['ds_name']           = npz['ds_name']
    to_ret['architecture']      = npz['architecture']
    to_ret['batch_size']        = npz['batch_size']
    to_ret['lr']                = npz['lr']
    to_ret['epochs']            = npz['epochs']
    to_ret['L1_lambda']         = npz['L1_lambda']
    to_ret['L2_lambda']         = npz['L2_lambda']
    to_ret['weight_decay']      = npz['weight_decay']
    to_ret['dropout']           = npz['dropout']
    to_ret['best_epoch']        = npz['best_epoch']
    to_ret['best_val']          = npz['best_val']
    return to_ret



def get_MSEs_for(paramter, training_combinations, list_eval_datasets=LIST_REAL_DATASETS, list_names=LIST_REAL_DATASETS_NAMES, plot=True, log=False):
    list_param_values = []
    list_MSEs = []
    for eval_datasets in list_eval_datasets:
        param_values = {}
        for tr in tqdm(training_combinations):
            params = get_all_paramters_dict(tr)
            assert paramter in params.keys(), f'Parameter {paramter} does not exist'
            p_val = float(params[paramter])
            if p_val not in param_values.keys():
                param_values[p_val] = []
            param_values[p_val].append(tr)
        min_num_vals = np.min([len(v) for v in param_values.values()])
        max_num_vals = np.max([len(v) for v in param_values.values()])
        print(f'Found {len(param_values.keys())} different values for {paramter}, min_num_vals={min_num_vals}, max_num_vals={max_num_vals}')

        if len(param_values.keys()) == 1:
            clear_output()
            return None
        param_values_mses = {}
        for p_val in tqdm(param_values.keys()):
            tmpMSEs = [] 
            for tr in param_values[p_val]:
                for ev_ds in eval_datasets:
                    comb_name = tr['name']
                    ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{comb_name}.npz'
                    npz = my_load(ds_train_combination_path, allow_pickle=True)
                    mse = npz['mse']
                    tmpMSEs.append(mse)
            param_values_mses[p_val] = np.mean(np.array(tmpMSEs))
            # print(f'p_val={p_val}, mses={tmpMSEs}, mean={param_values_mses[p_val]}')
        param_values = np.array(list(param_values_mses.keys()))
        mses = np.array(list(param_values_mses.values()))
        list_param_values.append(param_values)
        list_MSEs.append(mses)
    
    if plot:
        clear_output()
        fig,ax = plt.subplots(figsize=(10, 5))
        for param_values, mses, eval_datasets in zip(list_param_values, list_MSEs, list_eval_datasets):
            ax.plot(param_values, mses)
        ax.set_xlabel(paramter)
        ax.set_ylabel('MSE')
        ax.set_title(f'MSE for different {paramter}')
        ax.legend(list_names)
        ax.grid()
        if log:
            ax.set_xscale('log')
        plt.show()

    return param_values_mses

def get2D_MSEs_for(param1, param2, training_combinations, eval_datasets=DEFAULT_EVALUATION_DATASETS, plot=True):
    p1_values = {}
    p2_values = {}
    p12_values = {}
    for tr in tqdm(training_combinations):
        params = get_all_paramters_dict(tr)
        assert param1 in params.keys(), f'Parameter {param1} does not exist'
        assert param2 in params.keys(), f'Parameter {param2} does not exist'
        p1_val = float(params[param1])
        p2_val = float(params[param2])
        p12_val = (p1_val, p2_val)
        if p1_val not in p1_values.keys():
            p1_values[p1_val] = []
        if p2_val not in p2_values.keys():
            p2_values[p2_val] = []
        if p12_val not in p12_values.keys():
            p12_values[p12_val] = []
        p1_values[p1_val].append(tr)
        p2_values[p2_val].append(tr)
        p12_values[p12_val].append(tr)
    print(f'Found {len(p1_values.keys())} different values for {param1}')
    print(f'Found {len(p2_values.keys())} different values for {param2}')
    print(f'Found {len(p12_values.keys())} different values for {param1} and {param2}')
    if len(p1_values.keys()) == 1 or len(p2_values.keys()) == 1:
        return None

    assert len(p1_values.keys())*len(p2_values.keys()) == len(p12_values.keys()), 'Something went wrong'

    p12_values_mses = {}
    for p12_val in tqdm(p12_values.keys()):
        tmpMSEs = [] 
        for tr in p12_values[p12_val]:
            for ev_ds in eval_datasets:
                comb_name = tr['name']
                ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{comb_name}.npz'
                npz = my_load(ds_train_combination_path, allow_pickle=True)
                mse = npz['mse']
                tmpMSEs.append(mse)
        p12_values_mses[p12_val] = np.mean(np.array(tmpMSEs))
    
    if plot:
        p1s = list(p1_values.keys())
        p2s = list(p2_values.keys())
        mses = np.zeros((len(p1s), len(p2s)))
        for i,p1 in enumerate(p1s):
            for j,p2 in enumerate(p2s):
                mses[i, j] = p12_values_mses[(p1, p2)]
        fig,ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(18, 9))
        X, Y = np.meshgrid(p1s, p2s)
        Z = mses.T

        print(f'X.shape: {X.shape}, Y.shape: {Y.shape}, mses.shape: {mses.shape}')

        surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm,linewidth=0, antialiased=False)
        ax.set_xlabel(param1)
        ax.set_ylabel(param2)
        ax.set_zlabel('MSE')
        ax.set_title(f'MSE for different {param1} and {param2}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    return p12_values_mses


def get_STDs_for(paramter, training_combinations, list_eval_datasets=LIST_REAL_DATASETS, list_names=LIST_REAL_DATASETS_NAMES, plot=True, log=False):
    list_param_values = []
    list_STDs = []
    for eval_datasets in list_eval_datasets:
        param_values = {}
        for tr in tqdm(training_combinations):
            params = get_all_paramters_dict(tr)
            assert paramter in params.keys(), f'Parameter {paramter} does not exist'
            p_val = float(params[paramter])
            if p_val not in param_values.keys():
                param_values[p_val] = []
            param_values[p_val].append(tr)
        min_num_vals = np.min([len(v) for v in param_values.values()])
        max_num_vals = np.max([len(v) for v in param_values.values()])
        print(f'Found {len(param_values.keys())} different values for {paramter}, min_num_vals={min_num_vals}, max_num_vals={max_num_vals}')

        if len(param_values.keys()) == 1:
            clear_output()
            return None
        param_values_mses = {}
        for p_val in tqdm(param_values.keys()):
            tmpSTDs = [] 
            for tr in param_values[p_val]:
                for ev_ds in eval_datasets:
                    comb_name = tr['name']
                    ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{comb_name}.npz'
                    npz = my_load(ds_train_combination_path, allow_pickle=True)
                    hes, est_hes = npz['hes'], npz['est_hes']
                    assert hes.shape == est_hes.shape, f'hes.shape={hes.shape}, est_hes.shape={est_hes.shape}'
                    tmpSTDs.append(np.std(hes-est_hes))
                    # mse = np.mean(np.square(hes-est_hes))
                    # mse = npz['mse']
                    # tmpSTDs.append(mse)
            param_values_mses[p_val] = np.mean(np.array(tmpSTDs))
            # print(f'p_val={p_val}, mses={tmpSTDs}, mean={param_values_mses[p_val]}')
        param_values = np.array(list(param_values_mses.keys()))
        mses = np.array(list(param_values_mses.values()))
        mses = np.rad2deg(mses)
        list_param_values.append(param_values)
        list_STDs.append(mses)
    
    if plot:
        clear_output()
        fig,ax = plt.subplots(figsize=(10, 5))
        for param_values, mses, eval_datasets in zip(list_param_values, list_STDs, list_eval_datasets):
            ax.plot(param_values, mses)
        ax.set_xlabel(paramter)
        ax.set_ylabel('STD')
        ax.set_title(f'STD for different {paramter}')
        ax.legend(list_names)
        ax.grid()
        if log:
            ax.set_xscale('log')
        plt.show()

    return param_values_mses


def get_MAXs_for(paramter, training_combinations, list_eval_datasets=LIST_REAL_DATASETS, list_names=LIST_REAL_DATASETS_NAMES, plot=True, log=False):
    list_param_values = []
    list_MAXs = []
    for eval_datasets in list_eval_datasets:
        param_values = {}
        for tr in tqdm(training_combinations):
            params = get_all_paramters_dict(tr)
            assert paramter in params.keys(), f'Parameter {paramter} does not exist'
            p_val = float(params[paramter])
            if p_val not in param_values.keys():
                param_values[p_val] = []
            param_values[p_val].append(tr)
        min_num_vals = np.min([len(v) for v in param_values.values()])
        max_num_vals = np.max([len(v) for v in param_values.values()])
        print(f'Found {len(param_values.keys())} different values for {paramter}, min_num_vals={min_num_vals}, max_num_vals={max_num_vals}')

        if len(param_values.keys()) == 1:
            clear_output()
            return None
        param_values_mses = {}
        for p_val in tqdm(param_values.keys()):
            tmpMAXs = [] 
            for tr in param_values[p_val]:
                for ev_ds in eval_datasets:
                    comb_name = tr['name']
                    ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{comb_name}.npz'
                    npz = my_load(ds_train_combination_path, allow_pickle=True)
                    hes, est_hes = npz['hes'], npz['est_hes']
                    assert hes.shape == est_hes.shape, f'hes.shape={hes.shape}, est_hes.shape={est_hes.shape}'
                    tmpMAXs.append(np.max(np.abs(hes-est_hes)))
                    # mse = np.mean(np.square(hes-est_hes))
                    # mse = npz['mse']
                    # tmpMAXs.append(mse)
            param_values_mses[p_val] = np.max(np.array(tmpMAXs))
            # print(f'p_val={p_val}, mses={tmpMAXs}, mean={param_values_mses[p_val]}')
        param_values = np.array(list(param_values_mses.keys()))
        mses = np.array(list(param_values_mses.values()))
        list_param_values.append(param_values)
        list_MAXs.append(mses)
    
    if plot:
        clear_output()
        fig,ax = plt.subplots(figsize=(10, 5))
        for param_values, mses, eval_datasets in zip(list_param_values, list_MAXs, list_eval_datasets):
            ax.plot(param_values, mses)
        ax.set_xlabel(paramter)
        ax.set_ylabel('MAX')
        ax.set_title(f'MAX for different {paramter}')
        ax.legend(list_names)
        ax.grid()
        if log:
            ax.set_xscale('log')
        plt.show()

    return param_values_mses


def get_PERCs_for(paramter, training_combinations, list_eval_datasets=LIST_REAL_DATASETS, list_names=LIST_REAL_DATASETS_NAMES, plot=True, log=False):
    list_param_values = []
    list_PERCs = []
    for eval_datasets in list_eval_datasets:
        param_values = {}
        for tr in tqdm(training_combinations):
            params = get_all_paramters_dict(tr)
            assert paramter in params.keys(), f'Parameter {paramter} does not exist'
            p_val = float(params[paramter])
            if p_val not in param_values.keys():
                param_values[p_val] = []
            param_values[p_val].append(tr)
        min_num_vals = np.min([len(v) for v in param_values.values()])
        max_num_vals = np.max([len(v) for v in param_values.values()])
        print(f'Found {len(param_values.keys())} different values for {paramter}, min_num_vals={min_num_vals}, max_num_vals={max_num_vals}')

        if len(param_values.keys()) == 1:
            clear_output()
            return None
        param_values_mses = {}
        for p_val in tqdm(param_values.keys()):
            tmpPERCs = [] 
            for tr in param_values[p_val]:
                for ev_ds in eval_datasets:
                    comb_name = tr['name']
                    ds_train_combination_path = f'tmp/evals/eval_{ev_ds}___{comb_name}.npz'
                    npz = my_load(ds_train_combination_path, allow_pickle=True)
                    hes, est_hes = npz['hes'], npz['est_hes']
                    assert hes.shape == est_hes.shape, f'hes.shape={hes.shape}, est_hes.shape={est_hes.shape}'
                    error = np.abs(hes-est_hes)
                    rel_error = np.where(np.abs(hes) > 1e-2, error / np.abs(hes), np.zeros_like(error))

                    assert np.all(np.isfinite(rel_error)), f'np.any(np.isfinite(rel_error))={np.any(np.isfinite(rel_error))}'
                    assert error.shape == rel_error.shape, f'error.shape={error.shape}, rel_error.shape={rel_error.shape}'
                    percentage_error = np.mean(rel_error)
                    tmpPERCs.append(percentage_error)
                    # mse = np.mean(np.square(hes-est_hes))
                    # mse = npz['mse']
                    # tmpPERCs.append(mse)
            param_values_mses[p_val] = np.mean(np.array(tmpPERCs))
            # print(f'p_val={p_val}, mses={tmpPERCs}, mean={param_values_mses[p_val]}')
        param_values = np.array(list(param_values_mses.keys()))
        mses = np.array(list(param_values_mses.values()))
        list_param_values.append(param_values)
        list_PERCs.append(mses)
    
    if plot:
        clear_output()
        fig,ax = plt.subplots(figsize=(10, 5))
        for param_values, mses, eval_datasets in zip(list_param_values, list_PERCs, list_eval_datasets):
            ax.plot(param_values, mses)
        ax.set_xlabel(paramter)
        ax.set_ylabel('PERC')
        # ax.set_ylim(0, 1)
        ax.set_title(f'PERC for different {paramter}')
        ax.legend(list_names)
        ax.grid()
        if log:
            ax.set_xscale('log')
        plt.show()

    return param_values_mses




























