import matplotlib.pyplot as plt
import numpy as np
import utils
from PIL import Image
import cv2
import torch
import torch.nn as nn
import datetime
import os

class Reconstructor:
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


    def forward(self, im_diffuse, light_pattern_nonlinear, incident, camera_gain):
        batch_size = im_diffuse.shape[0]

        normal, albedo = self.photometric_stereo(batch_size, im_diffuse, light_pattern_nonlinear, incident, camera_gain)

        recon_dict = {
            'normal': normal,
            'albedo': albedo
        }

        return recon_dict

    def photometric_stereo(self, batch_size, im_rendered, light_pattern_nonlinear, incident, camera_gain):
        """
        Args:
            im_rendered: rendered image while training, real photo while testing
                    : [batch_size, #patterns, camR, camC, rgb] -> [batch_size, #patterns, camR, camC, rgb] -> [batch_size, #patterns, camR*camC]
            light_pattern_nonlinear: illumination
                    : sigmoid() -> [#patterns, lightR, lightC, rgb] -> [#patterns, lightR, lightC, 1] -> [#pattenrs, lightR*lightC]
            light_pos: coordinate of monitor
                    : [lightR, lightC, xyz] -> light_direction
            ptcloud: reference for photometric stereo, PLANE-NO(we assume that all data point is one same point)
            camera_gain: gain parameter for optimizing
        Output:
            surface normal: (batch, R, C, RGB) reconstructed normals
            diffuse albedo: (batch, R, C, RGB) reconstructed diffuse albedo
            valid mask
        """
        light_num = self.opt.light_N
        monitor_light_radiance = torch.sigmoid(light_pattern_nonlinear) ** self.opt.monitor_gamma
        input_R, input_C = im_rendered.shape[2:4]

        incident = incident.reshape(self.opt.light_R*self.opt.light_C, batch_size*input_R*input_C, 3).permute(1,0,2)
        diffuse_albedo = torch.max(im_rendered, dim=1).values

        r = diffuse_albedo[:,:,:,0].reshape(batch_size*input_R*input_C, 1, 1)
        g = diffuse_albedo[:,:,:,1].reshape(batch_size*input_R*input_C, 1, 1)
        b = diffuse_albedo[:,:,:,2].reshape(batch_size*input_R*input_C, 1, 1)

        # im_rendered: [batch, #pattern, R, C, 3] -> [batch*R*C, 3*#pattern]
        im_rendered = im_rendered.permute(0,2,3,4,1)
        im_rendered = im_rendered.reshape(batch_size*input_R*input_C, 3*light_num)
        im_rendered = im_rendered.unsqueeze(-1)

        im_rendered_red = im_rendered[:, :light_num]
        im_rendered_green = im_rendered[:, light_num:2*light_num]
        im_rendered_blue = im_rendered[:, 2*light_num:3*light_num]

        # light_pattern: [#pattern, r, c, 3] -> [3*#pattern, r*c]
        monitor_light_radiance = monitor_light_radiance.permute(3,0,1,2)
        monitor_light_radiance = monitor_light_radiance.reshape(3*light_num, self.opt.light_R*self.opt.light_C)

        # M: [3*#pattern, 3] -> [3*#pattern, xyz, R*C]
        # => M1:r, M2:g, M3:b   [#patterns, xyz, R*C]
        monitor_light_radiance = torch.tile(monitor_light_radiance.unsqueeze(0), (batch_size * input_R * input_C, 1, 1))
        M = monitor_light_radiance @ (incident)
        M1 = M[:, 0:light_num, :]
        M2 = M[:, light_num:2*light_num, :]
        M3 = M[:, 2*light_num:3*light_num, :]


        # iterative update
        for i in range(1):
            M_temp = torch.zeros_like(M, dtype=torch.float32, device=self.opt.device)
            # element-wise multiplication
            M_temp[:, 0:light_num, :] = r * M1
            M_temp[:, light_num:2 * light_num, :] = g * M2
            M_temp[:, 2 * light_num:3 * light_num, :] = b * M3
            # solve normal
            # invM: [3, 3*#patterns]
            # x:    [3, batch*R*C]
            x = torch.linalg.lstsq(M_temp, im_rendered).solution

            x = (x.squeeze(-1))

            # black-pixel handling
            x = x/(torch.linalg.norm(x, axis=-1).unsqueeze(-1) + 1e-8)
            x = x.unsqueeze(-1)

            # solve each channel
            # invR/G/B: [batch_size*camR*camC, 1, 9]

            #------------------------------------------------------------------------------
            foreshortening = torch.clamp(incident@x, 1e-8, None)
            r_expose = monitor_light_radiance[:,0:light_num,:]@foreshortening
            g_expose = monitor_light_radiance[:,light_num:2*light_num,:]@foreshortening
            b_expose = monitor_light_radiance[:,2*light_num:3*light_num,:]@foreshortening

            r_new = (torch.linalg.lstsq(r_expose, im_rendered_red).solution)
            g_new = (torch.linalg.lstsq(g_expose, im_rendered_green).solution)
            b_new = (torch.linalg.lstsq(b_expose, im_rendered_blue).solution)


            r_ = r_new.reshape(batch_size, input_R, input_C)
            g_ = g_new.reshape(batch_size, input_R, input_C)
            b_ = b_new.reshape(batch_size, input_R, input_C)

            diffuse_albedo = torch.stack([r_, g_, b_], axis=-1)


            gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
            
            diffuse_albedo = diffuse_albedo / (gain_scalar * (self.opt.rendering_scalar * self.opt.cam_shutter_time) + 1e-8)

            surface_normal = x.reshape(batch_size, input_R, input_C, 3)
            

        return surface_normal, diffuse_albedo