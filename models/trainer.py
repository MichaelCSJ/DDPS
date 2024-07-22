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

        # Optimization variable parameterization
        self.monitor_light_patterns = torch.nn.Parameter(self.opt.illums)
        self.monitor_superpixel_positions = torch.nn.Parameter(self.opt.light_pos)
        self.camera_gain = torch.logit(torch.tensor([self.opt.cam_gain], device=self.opt.device, dtype=self.opt.dtype)/48)

        # Load monitor 
        if opt.load_monitor_light:
            self.monitor_light_patterns = torch.load(opt.light_fn)

        # Optimizers
        lr_decay_monitor_light = opt.lr_decay_light
        step_size_epoch_light = opt.step_size_light

        # Set optimizers and schedulers
        if self.optimize_light:
            self.optimizer_light = torch.optim.Adam([self.monitor_light_patterns], lr=opt.lr_light)
            self.scheduler_light = torch.optim.lr_scheduler.StepLR(self.optimizer_light, step_size=step_size_epoch_light, gamma=lr_decay_monitor_light, last_epoch=last_epoch)
        
    def run_model(self, data):

        file_names = data['scene']
        batch_size = len(file_names)

        basis_images = data['imgs']
        normal_gt = data['normals']
        mask = data['mask']
        depth = self.opt.reference_plane[:batch_size,:,:,:]

        basis_images = basis_images.to(device=self.opt.device)
        normal_gt = normal_gt.to(device=self.opt.device)
        mask = mask.to(device=self.opt.device)
        depth = depth.to(device=self.opt.device)


        # Render image
        I_diffuse = self.renderer.render(basis_images, self.monitor_light_patterns, self.camera_gain)
        # utils.visualize3D(self.opt, I_diffuse[0,0].detach().cpu().numpy(), depth[0].detach().cpu().numpy(), self.monitor_superpixel_positions.data.cpu(), [0,0,0], self.opt.reference_plane[0].cpu().numpy(), self.opt.pt_ref.data.cpu())

        incident, _ = self.compute_light_direction(depth, self.monitor_superpixel_positions)

        # Reconstruct normal and albedo
        recon_output = self.reconstructor.forward(I_diffuse, self.monitor_light_patterns, incident, self.camera_gain)
        normal_est, albedo_est = (recon_output['normal'], recon_output['albedo'])


        # Compute error
        if self.opt.used_loss == 'cosine':
            normal_error = (2 - (1 + (normal_gt * normal_est).sum(-1)))/2
    
        normal_error = normal_error * mask

        num_interesting_pixel = torch.sum(mask)

        normal_error.requires_grad_(True)

        # Compute loss
        losses = {}
        losses['normal'] = torch.sum(normal_error) / num_interesting_pixel
        
        model_results = {}
        model_results['normal_est'] = normal_est.detach()
        model_results['albedo_est'] = albedo_est.detach()
        model_results['losses'] = losses

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

        # ----------------------------------------------
        self.model_results = self.run_model(data)
        self.loss_dict = self.model_results['losses']
        loss_sum = sum(self.loss_dict.values())
        loss_sum.requires_grad_(True)
        loss_sum.backward()

        # Step optimizer
        if self.optimize_light:
            self.optimizer_light.step()
        # ----------------------------------------------

    def run_schedulers_one_step(self):
        if self.optimize_light:
            self.scheduler_light.step()

    def get_losses(self):
        return self.loss_dict

    def get_model_results(self):
        return self.model_results
