import matplotlib.pyplot as plt
from models.dataset import *
import torch.nn.functional as nn
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
import utils
import datetime


class LightDeskRenderer:
    def __init__(self, opt):
        self.opt = opt

    def render(self, basis_images, monitor_light_pattern_nonlinear, camera_gain):
        batch_size = basis_images.shape[0]

        monitor_light_radiance = torch.sigmoid(monitor_light_pattern_nonlinear) ** self.opt.monitor_gamma
        monitor_light_radiance = torch.flip(monitor_light_radiance, dims=[2])
        monitor_light_radiance = monitor_light_radiance.reshape(self.opt.light_N, -1, 3).unsqueeze(0).unsqueeze(-1)

        basis_images = basis_images.permute(0,1,4,2,3)
        # weights: [batch, lightN, R*C, 3, H*W]
        basis_images = basis_images.reshape(batch_size, self.opt.light_R*self.opt.light_C, 3, -1).unsqueeze(1)
        result = monitor_light_radiance * basis_images
        result = result.sum(axis=2)
        result = result.reshape(batch_size, self.opt.light_N, 3, self.opt.cam_R, self.opt.cam_C)
        result = result.permute(0,1,3,4,2)


        gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
        
        I_diffuse = gain_scalar * (self.opt.rendering_scalar * self.opt.cam_shutter_time * result)
        I_diffuse = torch.clamp(I_diffuse, 1e-8, 1)

        return I_diffuse

          