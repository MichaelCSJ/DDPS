import argparse
import os
import torch
import numpy as np
import datetime
from PIL import Image
import sys
current_working_directory = os.getcwd()
sys.path.append(current_working_directory + '/models')

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument('--name', type=str, default='debug')

        # Gpu configuration, num_threads is for num_workers so that it should be 4*#GPU
        self.parser.add_argument('--num_threads', type=int, default=0)

        # Optimization
        self.parser.add_argument('--n_epoch', type=int, default=30)
        self.parser.add_argument('--batch_size', type=int, default=2)
        self.parser.add_argument('--batch_size_valid', type=int, default=2)
        self.parser.add_argument('--valid_N', type=int, default=2)

        self.parser.add_argument('--batch_size_test', type=int, default=1)

        # Save
        self.parser.add_argument('--freq_print', type=int, default=1) 
        self.parser.add_argument('--freq_valid', type=int, default=10) 

        self.parser.add_argument('--light_N', type=int, default=4)
        self.parser.add_argument('--initial_pattern', type=str, default='tri_random')
        self.parser.add_argument('--used_loss', type=str, default='cosine')

        # Training parameters
        self.parser.add_argument('--lr_light', type=float, default=3e-1)  
        self.parser.add_argument('--lr_decay_light', type=float, default=0.3)  
        self.parser.add_argument('--step_size_light', type=int, default=5)
        
        self.parser.add_argument('--lr_superpixel_position', type=float, default=1e-2)  
        self.parser.add_argument('--lr_decay_superpixel_position', type=float, default=0.3)  
        self.parser.add_argument('--step_size_superpixel_position', type=int, default=50)

        self.parser.add_argument('--fix_light_pattern', action='store_true', default=False)
        self.parser.add_argument('--fix_light_position', action='store_true', default=True)

        # Other parameters
        self.parser.add_argument('--load_epoch', type=int, default=0) # epoch that previous stoped
        self.parser.add_argument('--load_step_start', type=int, default=0) # previous step
        self.parser.add_argument('--load_latest', action='store_true', default=False)

        # Data
        self.parser.add_argument('--dataset_root', type=str, default='C:/Users/seokjub/Workspace/lightdesk_dataset/selected_dataset/')
        self.parser.add_argument('--save_dir', type=str, default='./results/')
        self.parser.add_argument('--light_geometric_calib_fn', type=str, default='./calibration/monitor_pixels_h9w16.npy')

        # For loading optimizer
        self.parser.add_argument('--load_monitor_light', action='store_true', default=False)
        self.parser.add_argument('--light_fn', type=str)

        # reference geometry
        self.parser.add_argument('--pt_ref_x', type=float, default=0.)
        self.parser.add_argument('--pt_ref_y', type=float, default=0.)
        self.parser.add_argument('--pt_ref_z', type=float, default=0.5)

        # camera
        self.parser.add_argument('--resizing_factor', type=int, default=1) 
        self.parser.add_argument('--cam_R', type=int, default=512) 
        self.parser.add_argument('--cam_C', type=int, default=612)
        self.parser.add_argument('--usePatch', type=bool, default=False)
        self.parser.add_argument('--roi_min_c', type=int, default=511) 
        self.parser.add_argument('--roi_min_r', type=int, default=288) 
        self.parser.add_argument('--roi_width', type=int, default=200)
        self.parser.add_argument('--roi_height', type=int, default=200)

        self.parser.add_argument('--cam_focal_length_pix', type=float, default=1.1485e+3)
        self.parser.add_argument('--cam_pitch', type=float, default=3.45e-6*2) 

        self.parser.add_argument('--cam_gain', type=float, default=0.0)
        self.parser.add_argument('--cam_gain_base', type=float, default=1.0960) 
        self.parser.add_argument('--static_interval', type=float, default=1/24) # 1s/fps

        # rendering
        self.parser.add_argument('--noise_sigma', type=float, default=0.02) 
        self.parser.add_argument('--rendering_scalar', type=float, default= 1.3e2 / 40) 

        # monitor
        self.parser.add_argument('--light_R', type=int, default=9)
        self.parser.add_argument('--light_C', type=int, default=16)
        self.parser.add_argument('--light_pitch', type=float, default=0.315e-3 * 120 * 2) 
        self.parser.add_argument('--monitor_gamma', type=float, default=2.2) 
        self.parser.add_argument('--vertical_gap', type=float, default=226.8 * 1e-3 / 3)
        self.parser.add_argument('--horizontal_gap', type=float, default=80.64 * 1e-3)

        # weight of the loss functions
        self.parser.add_argument('--weight_normal', type=float, default=1)
        self.parser.add_argument('--weight_recon', type=float, default=5)

        self.initialized = True


    def parse(self, save=True, isTrain=True, parse_args=None):
        if not self.initialized:
            self.initialize()
        if parse_args is None:
            self.opt = self.parser.parse_args()
            self.opt = self.parser.parse_args()
        else:
            self.opt = self.parser.parse_args(parse_args)

        self.opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.opt.dtype = torch.float32

        self.opt.cam_focal_length = self.opt.cam_focal_length_pix * self.opt.cam_pitch
        
        self.opt.cam_shutter_time = self.opt.static_interval / self.opt.light_N

        self.opt.original_R = self.opt.cam_R
        self.opt.original_C = self.opt.cam_C

        self.opt.cam_R //= self.opt.resizing_factor
        self.opt.cam_C //= self.opt.resizing_factor
        self.opt.cam_pitch *= self.opt.resizing_factor

        if not self.opt.usePatch:
            self.opt.roi_height = self.opt.cam_R
            self.opt.roi_width = self.opt.cam_C

        # Dataset paths
        self.opt.dataset_train_root = os.path.join(self.opt.dataset_root, 'train')
        self.opt.dataset_test_root = os.path.join(self.opt.dataset_root, 'val')
        self.opt.tb_dir = os.path.join(self.opt.save_dir, self.opt.name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(self.opt.tb_dir):
            os.makedirs(self.opt.tb_dir)
            os.makedirs(os.path.join(self.opt.tb_dir, 'patterns'))
            os.makedirs(os.path.join(self.opt.tb_dir, 'positions'))
            os.makedirs(os.path.join(self.opt.tb_dir, 'validation'))


        print('------------ Options -------------')
        args = vars(self.opt)
        print('-------------- End ----------------')

        if save:
            file_name = os.path.join(self.opt.tb_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        # Light position
        light_pos = np.load(self.opt.light_geometric_calib_fn)
        light_pos = np.reshape(light_pos, (self.opt.light_R, self.opt.light_C, 3))
        self.opt.light_pos = torch.tensor(light_pos, dtype=self.opt.dtype, device=self.opt.device)
        self.opt.light_pos_np = light_pos


        self.opt.illums = torch.zeros((self.opt.light_N, self.opt.light_R, self.opt.light_C, 3), dtype=self.opt.dtype, device=self.opt.device)  # (# of Illum, Illum's Row, Illum's Column, 3)
        vmin =  0.1
        vmax =  0.9
        self.opt.illums[:,:,:,:] = vmin

        torch.manual_seed(0)
        if self.opt.initial_pattern == 'tri_random':
            self.opt.illums[:, :, :, :] = torch.rand(self.opt.light_N, self.opt.light_R, self.opt.light_C, 3)

        elif self.opt.initial_pattern == 'gray_noise':
            self.opt.illums[:, :, :, :] = torch.normal(mean=0.5, std=0.01, size=(self.opt.light_N, self.opt.light_R, self.opt.light_C, 3))

        elif self.opt.initial_pattern == 'mono_random':
            temp = torch.rand(self.opt.light_N, self.opt.light_R, self.opt.light_C, 1)
            self.opt.illums[:, :, :, :] = torch.tile(temp, (1,1,1,3))

        elif self.opt.initial_pattern == 'mono_gradient':
            # Gradient
            row_indices = torch.arange(self.opt.light_R).float().unsqueeze(1)
            col_indices = torch.arange(self.opt.light_C).float().unsqueeze(0)
            ones = torch.ones(self.opt.light_R, self.opt.light_C)
            dist = row_indices / self.opt.light_R
            self.opt.illums[0] = torch.tile((0.1 + 0.8 * (ones - dist)).unsqueeze(-1), (1,1,3))
            self.opt.illums[1] = torch.flip(self.opt.illums[0], dims=[0])
            dist = col_indices / self.opt.light_C
            self.opt.illums[2] = torch.tile((0.1 + 0.8 * (ones - dist)).unsqueeze(-1), (1,1,3))
            self.opt.illums[3] = torch.flip(self.opt.illums[2], dims=[1])

        elif self.opt.initial_pattern == 'tri_gradient':
            # Color Gradient
            row_indices = torch.arange(self.opt.light_R).float().unsqueeze(1)
            col_indices = torch.arange(self.opt.light_C).float().unsqueeze(0)
            ones = torch.ones(self.opt.light_R, self.opt.light_C)
            dist = row_indices / self.opt.light_R
            self.opt.illums[0,:,:,2] = ((ones - dist))
            self.opt.illums[1,:,:,2] = dist
            dist = col_indices / self.opt.light_C
            self.opt.illums[0,:,:,0] = ((ones - dist))
            self.opt.illums[1,:,:,0] = dist
            # Compute distance from center of image
            dist = torch.sqrt((row_indices - self.opt.light_R/2)**2 + (col_indices - self.opt.light_C/2)**2)
            dist_norm = (dist.max() - dist) / dist.max()
            intensity = dist_norm * ones
            self.opt.illums[0,:,:,1] = intensity
            self.opt.illums[1,:,:,1] = ones - intensity        

        elif self.opt.initial_pattern == 'mono_comnplementary':
            # Binary
            self.opt.illums[1, :, :self.opt.light_C // 2, :] = vmax
            self.opt.illums[0, :, self.opt.light_C // 2:, :] = vmax
            self.opt.illums[2, :self.opt.light_R // 2, :, :] = vmax
            self.opt.illums[3, self.opt.light_R // 2:, :, :] = vmax

        elif self.opt.initial_pattern == 'tri_comnplementary':
            # Color binary
            self.opt.illums[0, :, :self.opt.light_C // 2, 0] = vmax
            self.opt.illums[0, :, self.opt.light_C // 2:, 1] = vmax
            self.opt.illums[0, :self.opt.light_R // 2, :, 2] = vmax
            self.opt.illums[1, :, :self.opt.light_C // 2, 1] = vmax
            self.opt.illums[1, :, self.opt.light_C // 2:, 0] = vmax
            self.opt.illums[1, self.opt.light_R // 2:, :, 2] = vmax


        elif self.opt.initial_pattern == 'OLAT':
            # OLAT
            self.opt.illums[0, :1, -1:, :] = vmax
            self.opt.illums[1, :1, :1, :] = vmax
            self.opt.illums[2, -1:, -1:, :] = vmax
            self.opt.illums[3, -1:, :1, :] = vmax

        elif self.opt.initial_pattern == 'grouped_OLAT':
            # Neighbor OLAT
            self.opt.illums[0, :3, -3:, :] = vmax
            self.opt.illums[1, :3, :3, :] = vmax
            self.opt.illums[2, -3:, -3:, :] = vmax
            self.opt.illums[3, -3:, :3, :] = vmax
        
        self.opt.illums[:, :, :, :] = torch.logit(self.opt.illums)

        # Reference point
        self.opt.pt_ref = torch.tensor([self.opt.pt_ref_x, self.opt.pt_ref_y, self.opt.pt_ref_z], dtype=self.opt.dtype, device=self.opt.device)

        # Reference point
        reference_plane = self.compute_ptcloud_from_depth(self.opt.pt_ref_z, 0, 0, self.opt.cam_R, self.opt.cam_C, self.opt.cam_R, self.opt.cam_C, self.opt.cam_pitch, self.opt.cam_focal_length)
        reference_plane = torch.tensor(reference_plane, device=self.opt.device, dtype=self.opt.dtype)
        self.opt.reference_plane = torch.tile(reference_plane.unsqueeze(0), (self.opt.batch_size, 1, 1, 1))

        return self.opt

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