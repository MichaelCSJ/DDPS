import torch
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import re
import os
from pyntcloud import PyntCloud
from os.path import join
import random
import models.utils as utils


class CreateBasisDataset(torch.utils.data.Dataset):
    def __init__(self, opt, list_name):
        self.opt = opt
        self.scene_names = file_load(os.path.join(opt.dataset_root, list_name))
        
    def __getitem__(self, i):
        scene_name = self.scene_names[i]
        basis_dir = os.path.join(self.opt.dataset_root, scene_name, 'main/diffuse')
        basis_file_names = [str(f).zfill(3)+'.png' for f in range(144)]

        imgs = np.zeros((self.opt.light_R*self.opt.light_C, self.opt.cam_R, self.opt.cam_C, 3), dtype=np.float32)
        for idx, basis_file in enumerate(basis_file_names):
            basis_path = os.path.join(basis_dir, basis_file)

            img = Image.open(basis_path)
            img = np.array(img).astype(np.float32)
            img = img/255
            img = cv2.medianBlur(img, 3)
            imgs[idx] = img

        normals = np.load(os.path.join(self.opt.dataset_root, scene_name, 'geo_normal_img.npy'))
        normals = -(2 * normals - 1)
        # backgound is masked, then we should normalize them
        normals = normals/(np.linalg.norm(normals, axis=-1)[...,None] + 1e-8)

        mask = np.load(os.path.join(self.opt.dataset_root, scene_name, 'geo_normal_img_mask.npy'))
        mask[mask!=0] = 1
        mask = mask[:,:,0]

        kernel = np.ones((7,7), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)

        input_dict = {
            'id': i,
            'scene': scene_name,
            'imgs': imgs,
            'normals': normals,
            'mask': mask
        }
        return input_dict
    
    def __len__(self):
        return len(self.scene_names)



def file_load(path):
    # Read data list
    data_path = []
    f = open("{0}.txt".format(path), 'r')
    while True:
        line = f.readline()
        if not line:
            break
        data_path.append(line[:-1])
    f.close()
    return data_path


def read_RGB(path):
    # RGB for compute diffuse
    img = Image.open(path)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data