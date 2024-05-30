# This code is for provide dataset for our renderer and reconstructor
# This code refer to 'sim_pytorch.py'
# Input is path, disparity, segmentation map, rgb image
# functions compute normal and choose valid normals
# there is two dataset 'monkey', 'colorchecker'
# Output is computed normal, diffuse albedo

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
# from models.renderer import compute_ptcloud_from_depth
import random
import models.utils as utils


class CreateBasisDataset(torch.utils.data.Dataset):
    def __init__(self, opt, list_name):
        self.opt = opt
        self.scene_names = file_load(os.path.join(opt.dataset_root, list_name))
        # self.scene_names = file_load(os.path.join(opt.dataset_root, 'scene_list'))
        # self.scene_names = [s for s in os.listdir(self.opt.dataset_root) if os.path.isdir(os.path.join(self.opt.dataset_root, s))]

    def __getitem__(self, i):
        scene_name = self.scene_names[i]
        basis_dir = os.path.join(self.opt.dataset_root, scene_name, 'main/diffuse')
        # basis_file_names = [f for f in os.listdir(basis_dir) if f.endswith('.png')]
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

        # depth = np.load(os.path.join(self.opt.dataset_root, scene_name, 'depth_rearanged.npy'))
        # depth = (1/(depth+1e-8))/4
        # masked_depth = (mask[:,:,0] * depth[:,:,0])
        # background = masked_depth==0
        # masked_depth[background] = 2

        
        depth = np.load(os.path.join(self.opt.dataset_root, scene_name, 'depth_raw.npy'))
        if len(depth.shape) == 3:
            masked_depth = (mask * depth[:,:,0])
        else:
            masked_depth = (mask * depth)
        depth_near = depth.min()
        depth_far = masked_depth.max()
        background = masked_depth==0
        masked_depth = ((masked_depth - depth_near) / (depth_far - depth_near)) * 0.2 + 0.5
        masked_depth[background] = 2

        ptcloud = self.compute_ptcloud_from_depth(masked_depth, 0, 0, self.opt.cam_R, self.opt.cam_C, self.opt.cam_R, self.opt.cam_C, self.opt.cam_pitch, self.opt.cam_focal_length)

        input_dict = {
            'id': i,
            'scene': scene_name,
            'imgs': imgs,
            'normals': normals,
            'mask': mask,
            'ptcloud': ptcloud
        }
        return input_dict
    
    def __len__(self):
        return len(self.scene_names)

    def compute_ptcloud_from_depth(self, depth, roi_rmin, roi_cmin, roi_height, roi_width, full_height, full_width, pitch, focal_length):
        '''
        make a point cloud by unprojecting the pixels with depth
        '''
        if isinstance(depth, float):
            pass

        else:
            depth = depth[roi_rmin:roi_rmin + roi_height, roi_cmin:roi_cmin + roi_width]
        XYZ = np.zeros((roi_height, roi_width, 3), dtype=np.float32)
        r, c = np.meshgrid(np.linspace(roi_rmin, roi_rmin + roi_height - 1, roi_height),
                              np.linspace(roi_cmin, roi_cmin + roi_width - 1, roi_width),
                              indexing='ij')
        center_c, center_r = full_width / 2, full_height / 2
        XYZ[:, :, 0] = (depth / focal_length) * (c - center_c) * pitch
        XYZ[:, :, 1] = (depth / focal_length) * (r - center_r) * pitch
        XYZ[:, :, 2] = depth

        return XYZ


class CreateRealDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        '''
        File structure of the real dataset
        dataset_test_root
            /real
                {a}_{b}.png
                    a is scene index and b is index of illumination
            /illums
                {b}.png
                    b is index of illumination
        '''
        # initialize parameters
        self.opt = opt


    def __getitem__(self, i):

        # load and compute data
        minr = self.opt.roi_min_r
        maxr = self.opt.roi_min_r + self.opt.roi_height
        minc = self.opt.roi_min_c
        maxc = self.opt.roi_min_c + self.opt.roi_width

        imgs = torch.zeros((self.opt.light_N, self.opt.roi_height, self.opt.roi_width, 3), dtype=self.opt.dtype, device=self.opt.device)  # (# of Illum, Row, Column, 3)

        for j in range(self.opt.light_N):
            img_path = join(self.opt.dataset_test_root, '/real/')
            img = cv2.imread(join(img_path, '%d_%d.png'%(i, j)))
            img = img[minr:maxr, minc:maxc, ::-1]

            img = torch.from_numpy(img.copy())
            imgs[j, :, :, :] = img

        # input dictionary
        input_dict = {
            'imgs': imgs,
            'id': i
        }

        return input_dict

    def __len__(self):
        return len(self.file_list)


class CreateSphereDataset(torch.utils.data.Dataset):
    def __init__(self, opt, Train=True):
        self.opt = opt
        # self.file_list = file_load(os.path.join(opt.dataset_root, 'dat/spheres', 'file_list') if Train else os.path.join(opt.dataset_root, 'dataset_test_list'))
        # self.file_list = file_load(os.path.join(opt.dataset_root, 'dat/spheres_perspective', 'file_list') if Train else os.path.join(opt.dataset_root, 'dataset_test_list'))
        self.file_list = file_load(os.path.join(opt.dataset_root, 'dat/spheres_perspective_fixed_num', 'file_list') if Train else os.path.join(opt.dataset_root, 'dataset_test_list'))

    def compute_ptcloud_from_depth(self, depth, roi_rmin, roi_cmin, roi_height, roi_width, full_height, full_width, pitch, focal_length):
        '''
        make a point cloud by unprojecting the pixels with depth
        '''
        if isinstance(depth, float):
            pass
        else:
            depth = depth[roi_rmin:roi_rmin + roi_height, roi_cmin:roi_cmin + roi_width]
        XYZ = np.zeros((roi_height, roi_width, 3), dtype=np.float32)
        r, c = np.meshgrid(np.linspace(roi_rmin, roi_rmin + roi_height - 1, roi_height),
                              np.linspace(roi_cmin, roi_cmin + roi_width - 1, roi_width),
                              indexing='ij')
        center_c, center_r = full_width / 2, full_height / 2
        XYZ[:, :, 0] = (depth / focal_length) * (c - center_c) * pitch
        XYZ[:, :, 1] = (depth / focal_length) * (r - center_r) * pitch
        XYZ[:, :, 2] = depth

        return XYZ

    def __getitem__(self, i):
        # path = os.path.join(self.opt.dataset_root, 'dat/spheres')
        # path = os.path.join(self.opt.dataset_root, 'dat/spheres_perspective')
        path = os.path.join(self.opt.dataset_root, 'dat/spheres_perspective_fixed_num')
        # i=21
        albedo_gt = torch.tensor(np.load(os.path.join(path, 'albedo_d', self.file_list[i]+'.npy')), dtype=self.opt.dtype)
        specular = torch.tensor(np.load(os.path.join(path, 'albedo_s', self.file_list[i]+'.npy')), dtype=self.opt.dtype)
        roughness = torch.tensor(np.load(os.path.join(path, 'roughness', self.file_list[i]+'.npy')), dtype=self.opt.dtype)
        ptcloud = torch.tensor(np.load(os.path.join(path, 'pt', self.file_list[i]+'.npy')), dtype=self.opt.dtype)
        normal_gt = torch.tensor(np.load(os.path.join(path, 'normal', self.file_list[i]+'.npy')), dtype=self.opt.dtype)

        mask = torch.tensor(np.load(os.path.join(path, 'mask', self.file_list[i]+'.npy')), dtype=self.opt.dtype)
        weight_map = torch.zeros(self.opt.num_basis, 96, 120, 1)
        for i in range(self.opt.num_basis):
            ith_mask = mask==i
            weight_map[i][ith_mask] = 1

        # reference_plane = self.compute_ptcloud_from_depth(self.opt.pt_ref_z, 0, 0, self.opt.original_R, self.opt.original_C, self.opt.original_R, self.opt.original_C, self.opt.cam_pitch, self.opt.cam_focal_length)


        input_dict = {
            'id': i,
            'albedo': albedo_gt,
            'specular': specular,
            'roughness': roughness,
            'ptcloud': ptcloud,
            'normal': normal_gt,
            # 'reference_plane': reference_plane,
            'weight_map': weight_map,
            'file_name': self.file_list[i]
        }

        return input_dict

    def __len__(self):
        return len(self.file_list)


class CreateMonkeyDataset(torch.utils.data.Dataset):
    def __init__(self, opt, Train=True):
        self.opt = opt
        self.file_list = file_load(os.path.join(opt.dataset_root, 'dataset_train_list') if Train else os.path.join(opt.dataset_root, 'dataset_test_list'))


    def read_PFM(self, path):
        # disparity map, material segmentation map
        file = open(path, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        # -------------------------resize when menory is restricted
        # temp = Image.fromarray(data)
        # if use_nearest:
        #     temp = temp.resize((C, R), Image.NEAREST)
        # else:
        #     temp = temp.resize((C, R), Image.ANTIALIAS)
        # data = np.array(temp)
        # data = torch.from_numpy(data.copy()).to('cuda')

        # data = cv2.resize(data, (C, R), cv2.INTER_AREA)
        # data = torch.tensor(data.copy(), dtype=self.opt.dtype)
        # -------------------------

        return data, scale


    def compute_depth_from_disparity(self, disparity):
        # scale the disparity of the monkey dataset to an appropriate range
        normalized_disparity = (disparity - disparity.min()) / (disparity.max()+1e-16)
        scaled_disparity = normalized_disparity * 90 + 10
        depth = 30 / (scaled_disparity+1e-16)  # explain constant. baseline:3000m, focal_length: 10mm

        return depth


    def compute_ptcloud_from_depth(self, depth, roi_rmin, roi_cmin, roi_height, roi_width, full_height, full_width, pitch, focal_length):
        '''
        make a point cloud by unprojecting the pixels with depth
        '''
        if isinstance(depth, float):
            pass
        else:
            depth = depth[roi_rmin:roi_rmin + roi_height, roi_cmin:roi_cmin + roi_width]
        XYZ = np.zeros((roi_height, roi_width, 3), dtype=np.float32)
        r, c = np.meshgrid(np.linspace(roi_rmin, roi_rmin + roi_height - 1, roi_height),
                              np.linspace(roi_cmin, roi_cmin + roi_width - 1, roi_width),
                              indexing='ij')
        center_c, center_r = full_width / 2, full_height / 2
        XYZ[:, :, 0] = (depth / focal_length) * (c - center_c) * pitch
        XYZ[:, :, 1] = (depth / focal_length) * (r - center_r) * pitch
        XYZ[:, :, 2] = depth

        return XYZ


    def correct_invalid_normals(self, normals, light_pos, ptcloud):
        # choose valid normals
        R, C = (normals.shape[0], normals.shape[1])
        light_vec = light_pos[np.newaxis,np.newaxis,:,:,:] - ptcloud[:,:,np.newaxis,np.newaxis,:]
        light_vec_norm = light_vec / (np.linalg.norm(light_vec, axis=-1)[:,:,:,:, np.newaxis] + 1e-16)
        invalid_normal_ang_deg = 89
        invalid = np.arccos(np.clip((normals.reshape(R*C,1,3)*light_vec_norm.reshape(R*C,-1,3)).sum(axis=-1), -1, 1)) >= (np.deg2rad(invalid_normal_ang_deg))
        num_of_minimum_valid_light = light_pos.shape[0] * light_pos.shape[1] # 9*16=144
        invalid = np.sum(invalid, axis=-1) > (num_of_minimum_valid_light)

        normals[invalid.reshape(R, C), :] = np.array([0,0,-1], dtype=np.float32)
        return normals


    def get_mean_val_given_seg(self, im, seg):
        # average colors of same material
        R, C = (im.shape[0], im.shape[1])
        seg = seg.flatten()
        # index_map = torch.unique(seg)
        index_map = np.unique(seg)
        im = im.reshape((im.shape[0] * im.shape[1], 3))
        # im_m = torch.zeros_like(im)
        im_m = np.zeros_like(im)
        for i in index_map:
            im_m[seg == i, :] = im[seg == i, :].mean(axis=0)
        im_m = im_m.reshape(R, C, 3)
        return im_m


    def __getitem__(self, i):
        # Set random patch size (R_min, C_min, height, width)
        if self.opt.usePatch:
            R_min = random.randint(0, self.opt.cam_R - self.opt.roi_height)
            C_min = random.randint(0, self.opt.cam_C - self.opt.roi_width)
            # R_min = (self.opt.cam_R - self.opt.roi_height)//2
            # C_min = (self.opt.cam_C - self.opt.roi_width)//2
            R_height = self.opt.cam_R
            C_width = self.opt.cam_C
        else:
            R_min = 0
            C_min = 0
            R_height = self.opt.cam_R
            C_width = self.opt.cam_C

        # Read albedo sources
        RGB_gt = read_RGB(os.path.join(self.opt.dataset_root, 'frames_cleanpass', self.file_list[i]) + '.png') / 255. # <- scaling
        # RGB_gt = torch.tensor(RGB_gt)
        # RGB_gt = torch.clip(RGB_gt + 0.1, 0, 1)
        seg, _ = self.read_PFM(os.path.join(self.opt.dataset_root, 'material_index', self.file_list[i]) + '.pfm')
        # seg = torch.tensor(seg)

        # Diffuse albedo
        albedo_gt = self.get_mean_val_given_seg(RGB_gt, seg)

        # Fake roughness and specular values
        roughness = seg/(seg.max()+1)
        specular = 1 - seg/(seg.max()+1)
        specular = specular[:,:,np.newaxis]
        # specular = torch.tile(specular.unsqueeze(-1), (1, 1, 1))

        # Disparity, depth, and ptcloud
        disparity, _ = self.read_PFM(os.path.join(self.opt.dataset_root, 'disparity', self.file_list[i]) + '.pfm')
        depth = self.compute_depth_from_disparity(disparity)
        # ptcloud = self.compute_ptcloud_from_depth(depth, R_min, C_min, R_height, C_width, self.opt.cam_R, self.opt.cam_C, self.opt.cam_pitch, self.opt.cam_focal_length)
        ptcloud = self.compute_ptcloud_from_depth(depth, 0, 0, self.opt.original_R, self.opt.original_C, self.opt.original_R, self.opt.original_C, self.opt.cam_pitch, self.opt.cam_focal_length)

        # plane_reference = self.compute_ptcloud_from_depth(self.opt.pt_ref_z, R_min, C_min, R_height, C_width, self.opt.cam_R, self.opt.cam_C, self.opt.cam_pitch, self.opt.cam_focal_length)
        # reference_plane = self.compute_ptcloud_from_depth(self.opt.pt_ref_z, 0, 0, self.opt.original_R, self.opt.original_C, self.opt.original_R, self.opt.original_C, self.opt.cam_pitch, self.opt.cam_focal_length)

        # Compute normal
        # normal_np = compute_normal_from_ptcloud_PCA(ptcloud.data.cpu().numpy(), neighbour_n=9)
        normal_gt = compute_normal_from_ptcloud_PCA(ptcloud, neighbour_n=9)
        # normal_gt = self.correct_invalid_normals(normal_gt, self.opt.light_pos_np, ptcloud) # self.opt.pt_ref.detach().cpu().numpy())
        # normal_gt = torch.tensor(normal_np)
        # normal_gt = self.correct_invalid_normals(normal_gt, self.opt.light_pos, self.opt.pt_ref, R_height, C_width)


        albedo_gt = torch.tensor(cv2.resize(albedo_gt, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)
        specular =  torch.tensor(cv2.resize(specular, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)
        roughness = torch.tensor(cv2.resize(roughness, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)
        depth =     torch.tensor(cv2.resize(depth, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)
        ptcloud =   torch.tensor(cv2.resize(ptcloud, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)
        normal_gt = torch.tensor(cv2.resize(normal_gt, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)
        # reference_plane = torch.tensor(cv2.resize(reference_plane, (self.opt.cam_C, self.opt.cam_R), cv2.INTER_AREA), dtype=self.opt.dtype)

        # input dictionary
        input_dict = {
            'id': i,
            'albedo':           utils.cut_edge(albedo_gt[R_min:R_min+R_height, C_min:C_min+C_width]),
            'specular':         utils.cut_edge(specular[R_min:R_min+R_height, C_min:C_min+C_width]),
            'roughness':        utils.cut_edge(roughness[R_min:R_min+R_height, C_min:C_min+C_width]),
            'depth':            utils.cut_edge(depth[R_min:R_min+R_height, C_min:C_min+C_width]),
            'ptcloud':          utils.cut_edge(ptcloud),
            'normal':           utils.cut_edge(normal_gt),
            # 'reference_plane':  utils.cut_edge(reference_plane),
            'file_name': self.file_list[i]
        }

        return input_dict


    def __len__(self):
        return len(self.file_list)


class SyntheticColorCheckerDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        # initialize parameters
        self.opt = opt
        self.file_list = file_load(os.path.join(opt.dataset_test_root, 'colorchecker'))


    def compute_ptcloud_from_depth(self, depth, roi_rmin, roi_cmin, roi_height, roi_width, full_height, full_width, pitch, focal_length):
        '''
        make a point cloud by unprojecting the pixels with depth
        '''
        if isinstance(depth, float):
            pass
        else:
            depth = depth[roi_rmin:roi_rmin + roi_height, roi_cmin:roi_cmin + roi_width]
        XYZ = np.zeros((roi_height, roi_width, 3), dtype=np.float32)
        r, c = np.meshgrid(np.linspace(roi_rmin, roi_rmin + roi_height - 1, roi_height),
                              np.linspace(roi_cmin, roi_cmin + roi_width - 1, roi_width),
                              indexing='ij')
        center_c, center_r = full_width / 2, full_height / 2
        XYZ[:, :, 0] = (depth / focal_length) * (c - center_c) * pitch
        XYZ[:, :, 1] = (depth / focal_length) * (r - center_r) * pitch
        XYZ[:, :, 2] = depth

        return XYZ

    def __getitem__(self, i):
        # load data
        RGB_gt = read_RGB(os.path.join(self.opt.dataset_test_root, self.file_list[i]) + '.png') / 255  # <- scaling\

        RGB_gt = np.clip(RGB_gt + 0.1, 0, 1)
        albedo_gt = RGB_gt

        R, C, _ = RGB_gt.shape

        roughness_scalar = torch.tensor([0.1], dtype=self.opt.dtype)
        # roughness = torch.tile(roughness_scalar.unsqueeze(0).unsqueeze(0), (R, C, 1))
        roughness = torch.tile(roughness_scalar.unsqueeze(0), (R, C))
        roughness = roughness.numpy()

        specular_scalar = torch.tensor([1.], dtype=self.opt.dtype)
        # specular = torch.tile(specular_scalar.unsqueeze(0).unsqueeze(0), (R, C, 3))
        specular = torch.tile(specular_scalar.unsqueeze(0), (R, C))
        specular = specular.numpy()

        one_normal = torch.tensor([0, 0, -1], dtype=self.opt.dtype)
        normal_gt = torch.tile(one_normal.unsqueeze(0).unsqueeze(0), (R, C, 1))
        normal_gt = normal_gt.numpy()


        reference_plane = self.compute_ptcloud_from_depth(self.opt.pt_ref_z, 0, 0, self.opt.original_R, self.opt.original_C, self.opt.original_R, self.opt.original_C, self.opt.cam_pitch, self.opt.cam_focal_length)
        
        input_dict = {
            'id': i,
            'albedo': utils.cut_edge(albedo_gt),
            'normal': utils.cut_edge(normal_gt),
            'specular': utils.cut_edge(specular),
            'roughness': utils.cut_edge(roughness),
            'ptcloud': utils.cut_edge(reference_plane),
            'file_name': self.file_list[i]
        }
        return input_dict

    def __len__(self):
        return len(self.file_list)


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
    # -------------------------
    # img = img.resize((C, R), Image.ANTIALIAS)
    # img = cv2.resize(img, (C, R), cv2.INTER_AREA)
    # -------------------------
    data = np.asarray(img, dtype="int32")
    return data


def compute_normal_from_ptcloud_PCA(XYZ, neighbour_n=11):
    # fast compute normal function
    N, M, _ = XYZ.shape

    points = pd.DataFrame({'x': XYZ[..., 0].flatten(), 'y': XYZ[..., 1].flatten(), 'z': XYZ[..., 2].flatten()})
    cloud = PyntCloud(points)

    k_neighbors = cloud.get_neighbors(k=neighbour_n)
    cloud.add_scalar_field("normals", k_neighbors=k_neighbors)

    nx = cloud.points['nx(%d)' % (neighbour_n + 1)].to_numpy()
    ny = cloud.points['ny(%d)' % (neighbour_n + 1)].to_numpy()
    nz = cloud.points['nz(%d)' % (neighbour_n + 1)].to_numpy()
    nx = nx.reshape((N, M, 1))
    ny = ny.reshape((N, M, 1))
    nz = nz.reshape((N, M, 1))

    invalid = nz > 0
    nx[invalid] *= -1
    ny[invalid] *= -1
    nz[invalid] *= -1

    normals = np.concatenate((nx, ny, nz), axis=2)
    return normals
