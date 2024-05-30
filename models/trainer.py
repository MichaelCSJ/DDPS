import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
# from scipy import interpolate
import utils
from PIL import Image
import gc
import os
import datetime

def model_results_to_np(model_results, batch_ind=None):
    if batch_ind is None:
        B = model_results['normal_est'].shape[0]
        batch_ind =range(0,B)

    results = dict()
    results['normal_est'] = model_results['normal_est'][batch_ind, ...].detach().cpu().numpy()
    results['albedo_est'] = model_results['albedo_est'][batch_ind, ...].detach().cpu().numpy()
    
    return results

class Trainer:
    def __init__(self, opt, renderer, reconstructor, last_epoch):

        self.opt = opt
        self.renderer = renderer
        self.reconstructor = reconstructor

        self.model_results = None
        self.loss_dict = None

        self.optimize_light = not opt.fix_light_pattern
        self.optimize_position = not opt.fix_light_position

        # 점들의 수와 거리 설정
        num_points = self.opt.light_R * self.opt.light_C
        distance_between_points = self.opt.light_pitch # 점들 사이의 거리

        # x, y, z 좌표 생성
        x_coords = torch.arange(-8, 8) * distance_between_points
        y_coords = torch.arange(-10, -1) * distance_between_points
        z_coords = torch.zeros(num_points)

        # 모든 좌표의 조합 생성
        x_coords = x_coords.repeat(9, 1).view(-1)
        y_coords = y_coords.repeat_interleave(16)

        # # 조건에 맞는 좌표 선택
        # condition = y_coords < 0
        # selected_x = x_coords[condition]
        # selected_y = y_coords[condition]
        # selected_z = z_coords[condition]

        # 선택된 좌표로 텐서 생성
        # points_tensor = torch.stack([selected_x, selected_y, selected_z], dim=1).to(device=self.opt.device).reshape(9,16,3)
        points_tensor = torch.stack([x_coords, y_coords, z_coords], dim=1).to(device=self.opt.device).reshape(9,16,3)


        # Optimization variable parameterization
        self.monitor_light_patterns = torch.nn.Parameter(self.opt.illums)
        # self.monitor_superpixel_positions = torch.nn.Parameter(self.opt.light_pos)
        self.monitor_superpixel_positions = torch.nn.Parameter(points_tensor)
        self.camera_gain = torch.logit(torch.tensor([self.opt.cam_gain], device=self.opt.device, dtype=self.opt.dtype)/48)

        # Load monitor 
        if opt.load_monitor_light:
            self.monitor_light_patterns = torch.load(opt.light_fn)

        # Optimizers
        lr_decay_monitor_light = opt.lr_decay_light
        step_size_epoch_light = opt.step_size_light
        
        lr_decay_superpixel_position = opt.lr_decay_superpixel_position
        step_size_superpixel_position = opt.step_size_superpixel_position

        # Set optimizers and schedulers
        if self.optimize_light:
            self.optimizer_light = torch.optim.Adam([self.monitor_light_patterns], lr=opt.lr_light)
            self.scheduler_light = torch.optim.lr_scheduler.StepLR(self.optimizer_light, step_size=step_size_epoch_light, gamma=lr_decay_monitor_light, last_epoch=last_epoch)
        if self.optimize_position:
            self.optimizer_superpixel_position = torch.optim.Adam([self.monitor_superpixel_positions], lr=opt.lr_superpixel_position)
            self.scheduler_superpixel_position = torch.optim.lr_scheduler.StepLR(self.optimizer_superpixel_position, step_size=step_size_superpixel_position, gamma=lr_decay_superpixel_position, last_epoch=last_epoch)
        # if self.optimize_gain:= torch.optim.lr_scheduler.StepLR(self.optimizer_gain, step_size=step_size_epoch_gain, gamma=lr_decay_camera_gain, last_epoch=last_epoch)

    def run_model(self, data):

        file_names = data['scene']
        batch_size = len(file_names)

        basis_images = data['imgs']
        normal_gt = data['normals']
        mask = data['mask']
        depth = data['ptcloud']
        depth = self.opt.reference_plane[:batch_size,:,:,:]
        # mask = depth[:,:,:,2]<1

        basis_images = basis_images.to(device=self.opt.device)
        normal_gt = normal_gt.to(device=self.opt.device)
        mask = mask.to(device=self.opt.device)
        depth = depth.to(device=self.opt.device)


        # print()
        # print('Model start ', torch.cuda.memory_allocated()/1024/1024, 'MB')

        # Render image
        I_diffuse = self.renderer.render(basis_images, self.monitor_light_patterns, self.camera_gain)
        # utils.visualize3D(self.opt, I_diffuse[0,0].detach().cpu().numpy(), depth[0].detach().cpu().numpy(), self.monitor_superpixel_positions.data.cpu(), [0,0,0], self.opt.reference_plane[0].cpu().numpy(), self.opt.pt_ref.data.cpu())

        # print('Render ', torch.cuda.memory_allocated()/1024/1024, 'MB')

        # utils.visualize_patterns(self.opt, torch.sigmoid(self.monitor_light_patterns).detach().cpu().numpy())
        # utils.visualize_GT_data(self.opt, file_names, torch.sigmoid(self.monitor_light_patterns).detach().cpu().numpy(), I_diffuse.detach().cpu().numpy())

        incident, _ = self.compute_light_direction(depth, self.monitor_superpixel_positions)

        # Compute distances
        # distance = torch.sqrt(torch.sum(torch.pow(self.monitor_superpixel_positions.unsqueeze(2).unsqueeze(2).unsqueeze(2) - depth.unsqueeze(0).unsqueeze(0), 2), axis=-1))

        # falloff_scalar = 1/distance**2
        # incident = incident * falloff_scalar[...,None] / 2.85


        # Reconstruct normal and albedo
        recon_output = self.reconstructor.forward(I_diffuse, self.monitor_light_patterns, incident, self.camera_gain)
        normal_est, albedo_est = (recon_output['normal'], recon_output['albedo'])

        # print('Reconstruct ', torch.cuda.memory_allocated()/1024/1024, 'MB')

        # utils.visualize_EST_normal(self.opt, file_names, normal_gt.detach().cpu().numpy(), normal_est.detach().cpu().numpy(), mask.detach().cpu().numpy())

        # I_diffuse_est = self.renderer.render_vec(normal_est, albedo_est, self.monitor_light_patterns, incident, self.camera_gain)

        # print('Re-render ', torch.cuda.memory_allocated()/1024/1024, 'MB')

        # Compute error
        # normal_error = torch.rad2deg(torch.arccos((normal_gt * normal_est).sum(-1)))/180
        # normal_error = 1 - torch.abs((normal_gt * normal_est).sum(-1))
        if self.opt.used_loss == 'cosine':
            normal_error = (2 - (1 + (normal_gt * normal_est).sum(-1)))/2
        elif self.opt.used_loss == 'angular':
            normal_error = torch.rad2deg(torch.arccos((normal_gt * normal_est).sum(-1)))/180
    
        normal_error = normal_error * mask

        # I_mask = torch.tile(mask.unsqueeze(1), (1, self.opt.light_N, 1, 1))
        # diffuse_error = torch.abs(I_diffuse - I_diffuse_est) * I_mask.unsqueeze(-1)

        # utils.visualize_unsup_error(self.opt, file_names, torch.sigmoid(self.monitor_light_patterns).detach().cpu().numpy(), I_diffuse.detach().cpu().numpy(), I_diffuse_est.detach().cpu().numpy(), diffuse_error.detach().cpu().numpy())

        # 배치 별로 계산하기.
        # num_interesting_pixel = torch.sum(mask.reshape(batch_size, -1), dim=1)
        num_interesting_pixel = torch.sum(mask)

        normal_error.requires_grad_(True)
        # diffuse_error.requires_grad_(True)

        
        # horizontal and vertical vectors
        vertical_vectors = torch.empty((0,3), dtype=torch.float32, device='cuda')
        horizontal_vectors = torch.empty((0,3), dtype=torch.float32, device='cuda')
        # vertical_pts = torch.empty((0,3), dtype=torch.float32, device='cuda')
        # horizontal_pts = torch.empty((0,3), dtype=torch.float32, device='cuda')

        for r in range(self.opt.light_R-1):
            # vertical_vectors = torch.cat((vertical_vectors, (torch.unsqueeze((self.monitor_superpixel_positions[r] - self.monitor_superpixel_positions[r+1]), axis=0))), dim=0)
            vertical_vectors = torch.cat((vertical_vectors, (self.monitor_superpixel_positions[r] - self.monitor_superpixel_positions[r+1])), dim=0)
        for c in range(self.opt.light_C-1):
            # horizontal_vectors = torch.cat((horizontal_vectors, torch.unsqueeze(self.monitor_superpixel_positions[:,c], self.monitor_superpixel_positions[:,c+1], axis=0)), dim=0)
            horizontal_vectors = torch.cat((horizontal_vectors, (self.monitor_superpixel_positions[:,c] - self.monitor_superpixel_positions[:,c+1])), dim=0)
            
        vertical_vectors_length = torch.unsqueeze(torch.linalg.norm(vertical_vectors, axis=-1), axis=-1)
        horizontal_vectors_length = torch.unsqueeze(torch.linalg.norm(horizontal_vectors, axis=-1), axis=-1)
        
        vertical_vectors_normalized = vertical_vectors/vertical_vectors_length
        horizontal_vectors_normalized = horizontal_vectors/horizontal_vectors_length

        horizontal_vectors_reshape = horizontal_vectors_normalized.reshape(-1,9,3).transpose(1,0)
        # horizontal_vectors_reshape = horizontal_vectors_normalized.reshape(9,-1,3)
        
        GT_vertical_length = torch.zeros_like(vertical_vectors_length, dtype=torch.float32, device='cuda')
        GT_vertical_length[:,:] = self.opt.light_pitch # vertical_gap
        GT_horizontal_length = torch.zeros_like(horizontal_vectors_length, dtype=torch.float32)
        GT_horizontal_length[:,:] = self.opt.light_pitch # horizontal_gap

        # print(vertical_vectors.shape)
        # print(horizontal_vectors.shape)
        # print(vertical_vectors_length[0], horizontal_vectors_length[0])
        mse = torch.nn.MSELoss(reduction='mean')

        horizontal_parallel_tensor = torch.zeros(9,9, 15)
        total_temp_loss = None
        # for p in range(1):
        for p in range(self.opt.light_C-1):
            temp = horizontal_vectors_reshape[:,p,:]
            horizontal_parallel_tensor[:,:,p] = torch.abs(1-(temp @ torch.permute(temp, (1, 0))))
            temp_loss = torch.abs(1-(temp @ torch.permute(temp, (1, 0)))).mean()
            if p == 0:
                total_temp_loss = temp_loss
            else:
                total_temp_loss = total_temp_loss + temp_loss
        total_temp_loss = total_temp_loss / (self.opt.light_C-1)

        loss_length_ver_vis = abs(GT_vertical_length - vertical_vectors_length)
        loss_length_hor_vis = abs(GT_horizontal_length - horizontal_vectors_length)
        # utils.visualize_length_err(self.opt, loss_length_ver_vis.detach().cpu().numpy(), loss_length_hor_vis.detach().cpu().numpy())
        
        loss_perpendicular_vis = torch.abs(vertical_vectors_normalized @ torch.permute(horizontal_vectors_normalized, (1, 0)))
        # utils.visualize_perpendicular_err(self.opt, loss_perpendicular_vis.detach().cpu().numpy())

        loss_parallel_vertical_vis = torch.abs(1-(vertical_vectors_normalized @ torch.permute(vertical_vectors_normalized, (1, 0))))
        # utils.visualize_parallel_err(self.opt, loss_parallel_vertical_vis.detach().cpu().numpy(), horizontal_parallel_tensor.detach().cpu().numpy())

        #___________First derivative_________________________________________________________________________________________________
        monitor_mid_cols = self.monitor_superpixel_positions[:,7:9,:]
        virtual_mid_cols = monitor_mid_cols.sum(axis=1)/2
        # virtual_mid_col_vec = (virtual_mid_cols[1:] - virtual_mid_cols[:-1]).mean(axis=0)
        virtual_mid_col_vec = (virtual_mid_cols[1] - virtual_mid_cols[0])
        
        mid_top_row_vec = monitor_mid_cols[0,1] - monitor_mid_cols[0,0]
        # mid_top_row_vec = (monitor_mid_cols[:,1] - monitor_mid_cols[:,0]).mean(axis=0)

        # parabolar_axis = torch.cross(mid_top_row_vec, virtual_mid_cols[0])
        # parabolar_axis = torch.cross(virtual_mid_cols.mean(axis=0), mid_top_row_vec)
        parabolar_axis = torch.cross(mid_top_row_vec, virtual_mid_col_vec)
        parabolar_axis_norm = parabolar_axis / (torch.unsqueeze(torch.linalg.norm(parabolar_axis, axis=-1), axis=-1) + 1e-8)

        monitor_left_wing_vecs = self.monitor_superpixel_positions[:, :7] - self.monitor_superpixel_positions[:, 1:8]
        monitor_right_wing_vecs = self.monitor_superpixel_positions[:, 9:] - self.monitor_superpixel_positions[:, 8:-1]

        left_wing_lengths = (monitor_left_wing_vecs * parabolar_axis_norm.reshape(1,1,3)).sum(axis=-1)
        right_wing_lengths = (monitor_right_wing_vecs * parabolar_axis_norm.reshape(1,1,3)).sum(axis=-1)

        left_loss = torch.abs(torch.sum(torch.clamp(left_wing_lengths[:,:-1] - left_wing_lengths[:,1:], None, 0))) - torch.sum(torch.clamp(left_wing_lengths, None, 0))
        right_loss = torch.abs(torch.sum(torch.clamp(right_wing_lengths[:,1:] - right_wing_lengths[:,:-1], None, 0))) - torch.sum(torch.clamp(right_wing_lengths, None, 0))
        #_________________________________________________________________________________________________________________________

        #___________Second derivative_________________________________________________________________________________________________
        monitor_mid_cols = self.monitor_superpixel_positions[:,7:9,:]
        virtual_mid_cols = monitor_mid_cols.sum(axis=1)/2
        # virtual_mid_col_vec = (virtual_mid_cols[1:] - virtual_mid_cols[:-1]).mean(axis=0)
        virtual_mid_col_vec = (virtual_mid_cols[1] - virtual_mid_cols[0])
        
        mid_top_row_vec = monitor_mid_cols[0,1] - monitor_mid_cols[0,0]
        # mid_top_row_vec = (monitor_mid_cols[:,1] - monitor_mid_cols[:,0]).mean(axis=0)

        # parabolar_axis = torch.cross(mid_top_row_vec, virtual_mid_cols[0])
        # parabolar_axis = torch.cross(virtual_mid_cols.mean(axis=0), mid_top_row_vec)
        parabolar_axis = torch.cross(mid_top_row_vec, virtual_mid_col_vec)
        parabolar_axis_norm = parabolar_axis / (torch.unsqueeze(torch.linalg.norm(parabolar_axis, axis=-1), axis=-1) + 1e-8)

        monitor_left_wing_vecs = self.monitor_superpixel_positions[:, :7] - self.monitor_superpixel_positions[:, 1:8]
        monitor_right_wing_vecs = self.monitor_superpixel_positions[:, 9:] - self.monitor_superpixel_positions[:, 8:-1]

        left_wing_lengths = (monitor_left_wing_vecs * parabolar_axis_norm.reshape(1,1,3)).sum(axis=-1)
        right_wing_lengths = (monitor_right_wing_vecs * parabolar_axis_norm.reshape(1,1,3)).sum(axis=-1)

        left_loss = torch.abs(torch.sum(torch.clamp(left_wing_lengths[:,:-1] - left_wing_lengths[:,1:], None, 0))) - torch.sum(torch.clamp(left_wing_lengths, None, 0))
        right_loss = torch.abs(torch.sum(torch.clamp(right_wing_lengths[:,1:] - right_wing_lengths[:,:-1], None, 0))) - torch.sum(torch.clamp(right_wing_lengths, None, 0))
        #_________________________________________________________________________________________________________________________

        # Compute loss
        losses = {}
        # losses['normal'] = normal_error.mean()
        # losses['diffuse'] = diffuse_error.mean()
        losses['normal'] = torch.sum(normal_error) / num_interesting_pixel * 10
        # losses['diffuse'] = torch.sum(diffuse_error) / (num_interesting_pixel*self.opt.light_N*3)
        # losses['length'] = (mse(GT_vertical_length, vertical_vectors_length) + mse(GT_horizontal_length, horizontal_vectors_length)) * 10
        # losses['perpendicular'] = loss_perpendicular_vis.mean()

        # losses['parallel_vertical'] = loss_parallel_vertical_vis.mean() * 10
        # losses['parallel_horizontal'] = total_temp_loss * 10

        # losses['curvature'] = (left_loss + right_loss)
        
        model_results = {}
        model_results['normal_est'] = normal_est.detach()
        model_results['albedo_est'] = albedo_est.detach()
        model_results['losses'] = losses

        # print(f'Normal error: {normal_error.mean()}') # Albedo error: {diffuse_error.mean()}')

        # print('Before step ', torch.cuda.memory_allocated()/1024/1024, 'MB')

        return model_results

    def compute_light_direction(self, ptcloud, light_pos):
        # Compute for point light sources
        incident = light_pos.unsqueeze(2).unsqueeze(2).unsqueeze(2) - ptcloud.unsqueeze(0).unsqueeze(0)
        incident = incident / (torch.linalg.norm(incident, axis=-1).unsqueeze(-1) + 1e-8)
        exitant = -(ptcloud / (torch.linalg.norm(ptcloud, axis=-1).unsqueeze(-1) + 1e-8))

        return incident, exitant[None, None, ...]

    def run_optimizers_one_step(self, data):
        
        # Backward pass
        if self.optimize_light:
            self.optimizer_light.zero_grad()
        if self.optimize_position:
            self.optimizer_superpixel_position.zero_grad()

        # ----------------------------------------------
        self.model_results = self.run_model(data)
        self.loss_dict = self.model_results['losses']
        loss_sum = sum(self.loss_dict.values())
        loss_sum.requires_grad_(True)
        loss_sum.backward()

        # Step optimizer
        if self.optimize_light:
            self.optimizer_light.step()
        if self.optimize_position:
            self.optimizer_superpixel_position.step()
        # ----------------------------------------------

    def run_schedulers_one_step(self):
        if self.optimize_light:
            self.scheduler_light.step()
        if self.optimize_position:
            self.scheduler_superpixel_position.step()

    def get_losses(self):
        return self.loss_dict

    def get_model_results(self):
        return self.model_results
