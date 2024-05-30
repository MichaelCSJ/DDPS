# Reconstructor do 3-channel photometric stereo in iterative method

import matplotlib.pyplot as plt
import numpy as np
import utils
from PIL import Image
import cv2
import torch
import torch.nn as nn
import datetime
import os
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from FCSF import Alpha_estimator
# from BRDF_PointNet import BRDF_PointNet
# from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering
# from guided_filter_pytorch.guided_filter import GuidedFilter


# class Reconstructor(nn.Module):
class Reconstructor:
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


    # def forward(self, im_diffuse, I_specular, ptcloud, light_pattern_nonlinear, incident, camera_gain):
    def forward(self, im_diffuse, light_pattern_nonlinear, incident, camera_gain):
        batch_size = im_diffuse.shape[0]

        normal, albedo = self.photometric_stereo(batch_size, im_diffuse, light_pattern_nonlinear, incident, camera_gain)
        # normal, albedo = self.photometric_stereo2(batch_size, im_diffuse, light_pattern_nonlinear, incident)
        # normal = self.photometric_stereo(batch_size, im_diffuse, light_pattern_nonlinear, incident)

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
        # monitor_light_radiance= torch.sigmoid(monitor_light_pattern_nonlinear) ** self.opt.monitor_gamma
        # monitor_light_radiance = torch.flip(monitor_light_radiance, dims=[2])
        input_R, input_C = im_rendered.shape[2:4]

        # Random initial
        # incident = incident.reshape(self.opt.light_R*self.opt.light_C, batch_size*input_R*input_C, 3).permute(0,2,1)
        incident = incident.reshape(self.opt.light_R*self.opt.light_C, batch_size*input_R*input_C, 3).permute(1,0,2)
        # diffuse_albedo = im_rendered.mean(axis=1)
        diffuse_albedo = torch.max(im_rendered, dim=1).values
        # diffuse_albedo = full_on

        # plt.imshow(diffuse_albedo[0].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(diffuse_albedo[1].detach().cpu().numpy())
        # plt.show()

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
        # M = monitor_light_radiance @ (incident.permute(2, 0, 1))
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
            # x = torch.linalg.lstsq(M_temp.permute(2,0,1), im_rendered).solution
            x = torch.linalg.lstsq(M_temp, im_rendered).solution
            # x = torch.linalg.pinv(M_temp) @ im_rendered

            # TODO: erase squeeze and unsqueeze, change zero mask
            x = (x.squeeze(-1))

            # black-pixel handling
            # zero_mask = x == 0.
            # x[zero_mask] = x[zero_mask] + 1e-4 # torch.tensor([0, 0, 1])

            # zero_mask = x == torch.tensor([0., 0., 0.], device=self.opt.device)
            # zero_mask = zero_mask[:,0]
            # x[zero_mask] = torch.tensor([-0., -0., -1.], device=self.opt.device)

            # x = x/torch.linalg.norm(x, axis=-2).unsqueeze(-1)
            x = x/(torch.linalg.norm(x, axis=-1).unsqueeze(-1) + 1e-8)
            x = x.unsqueeze(-1)

            # solve each channel
            # invR/G/B: [batch_size*camR*camC, 1, 9]

            # r_new = (torch.linalg.lstsq(M1.permute(2,0,1)@x, im_rendered_red).solution).squeeze(-1).permute(1,0)
            # g_new = (torch.linalg.lstsq(M2.permute(2,0,1)@x, im_rendered_green).solution).squeeze(-1).permute(1,0)
            # b_new = (torch.linalg.lstsq(M3.permute(2,0,1)@x, im_rendered_blue).solution).squeeze(-1).permute(1,0)

            # r_new = (torch.linalg.pinv(M1@x) @ im_rendered_red)
            # g_new = (torch.linalg.pinv(M2@x) @ im_rendered_green)
            # b_new = (torch.linalg.pinv(M3@x) @ im_rendered_blue)
            #------------------------------------------------------------------------------
            foreshortening = torch.clamp(incident@x, 1e-8, None)
            r_expose = monitor_light_radiance[:,0:light_num,:]@foreshortening
            g_expose = monitor_light_radiance[:,light_num:2*light_num,:]@foreshortening
            b_expose = monitor_light_radiance[:,2*light_num:3*light_num,:]@foreshortening

            r_new = (torch.linalg.lstsq(r_expose, im_rendered_red).solution)
            g_new = (torch.linalg.lstsq(g_expose, im_rendered_green).solution)
            b_new = (torch.linalg.lstsq(b_expose, im_rendered_blue).solution)


            # #------------------------------------------------------------------------------
            # mask = foreshortening>1e-8
            # M_ = monitor_light_radiance @ (incident*mask)

            # M1 = M_[:, 0:light_num, :]
            # M2 = M_[:, light_num:2*light_num, :]
            # M3 = M_[:, 2*light_num:3*light_num, :]
            # #------------------------------------------------------------------------------
            # # r_new = (torch.linalg.pinv(r_expose) @ im_rendered_red)
            # # g_new = (torch.linalg.pinv(g_expose) @ im_rendered_green)
            # # b_new = (torch.linalg.pinv(b_expose) @ im_rendered_blue)
            # #------------------------------------------------------------------------------
            r_ = r_new.reshape(batch_size, input_R, input_C)
            g_ = g_new.reshape(batch_size, input_R, input_C)
            b_ = b_new.reshape(batch_size, input_R, input_C)

            diffuse_albedo = torch.stack([r_, g_, b_], axis=-1)
            # diffuse_albedo = diffuse_albedo / diffuse_albedo.max()


            gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
            
            diffuse_albedo = diffuse_albedo / (gain_scalar * (self.opt.rendering_scalar * self.opt.cam_shutter_time) + 1e-8)


            # gain_exponent = torch.sigmoid(camera_gain)*48 + 1
            # diffuse_albedo = diffuse_albedo / ((self.opt.rendering_scalar * (self.opt.cam_gain_base ** gain_exponent) * self.opt.cam_shutter_time) + 1e-8)

            # r = diffuse_albedo[:,:,:,0].reshape(batch_size*input_R*input_C, 1, 1)
            # r = torch.tile(r, (1, 4, 3))
            # g = diffuse_albedo[:,:,:,1].reshape(batch_size*input_R*input_C, 1, 1)
            # g = torch.tile(g, (1, 4, 3))
            # b = diffuse_albedo[:,:,:,2].reshape(batch_size*input_R*input_C, 1, 1)
            # b = torch.tile(b, (1, 4, 3))
            surface_normal = x.reshape(batch_size, input_R, input_C, 3)
            
            # plt.figure(figsize=(5,5),constrained_layout=True)
            # plt.imshow(((-surface_normal+1)/2)[0].detach().cpu().numpy())
            # plt.colorbar()
            # plt.savefig(os.path.join(self.opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + 'normal'+str(i)), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
            # plt.close()
            # print('Reconstruct ', torch.cuda.memory_allocated()/1024/1024, 'MB')

        return surface_normal, diffuse_albedo


    def photometric_stereo2(self, batch_size, im_rendered, light_pattern_nonlinear, incident, normals_pt):
        light_num = self.opt.light_N
        monitor_light_radiance = torch.sigmoid(light_pattern_nonlinear) ** self.opt.monitor_gamma
        input_R, input_C = im_rendered.shape[2:4]

        # Random initial
        incident = incident.reshape(self.opt.light_R*self.opt.light_C, batch_size*input_R*input_C, 3).permute(1,0,2)
        # diffuse_albedo = im_rendered.mean(axis=1)
        diffuse_albedo = torch.max(im_rendered, dim=1).values
        # diffuse_albedo = full_on

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
        # M = monitor_light_radiance @ (incident.permute(2, 0, 1))

        normals_pt = normals_pt.reshape(batch_size*input_R*input_C, 3, 1)
        valid_lights = incident @ normals_pt
        valid_lights[valid_lights>0] = 1
        valid_lights[valid_lights<=0] = 1e-8

        M = monitor_light_radiance @ (incident * valid_lights)
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
            diffuse_albedo = diffuse_albedo / diffuse_albedo.max()
            surface_normal = x.reshape(batch_size, input_R, input_C, 3)

        return surface_normal, diffuse_albedo


    def iteratible_photometric_stereo(self, batch_size, im_rendered, light_pattern_nonlinear, incident, camera_gain, iter):
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
        # monitor_light_radiance= torch.sigmoid(monitor_light_pattern_nonlinear) ** self.opt.monitor_gamma
        # monitor_light_radiance = torch.flip(monitor_light_radiance, dims=[2])
        input_R, input_C = im_rendered.shape[2:4]

        # Random initial
        # incident = incident.reshape(self.opt.light_R*self.opt.light_C, batch_size*input_R*input_C, 3).permute(0,2,1)
        incident = incident.reshape(self.opt.light_R*self.opt.light_C, batch_size*input_R*input_C, 3).permute(1,0,2)
        # diffuse_albedo = im_rendered.mean(axis=1)
        diffuse_albedo = torch.max(im_rendered, dim=1).values
        # diffuse_albedo = full_on

        # plt.imshow(diffuse_albedo[0].detach().cpu().numpy())
        # plt.show()
        # plt.imshow(diffuse_albedo[1].detach().cpu().numpy())
        # plt.show()

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
        # M = monitor_light_radiance @ (incident.permute(2, 0, 1))
        M = monitor_light_radiance @ (incident)
        M1 = M[:, 0:light_num, :]
        M2 = M[:, light_num:2*light_num, :]
        M3 = M[:, 2*light_num:3*light_num, :]


        iterated_normals = torch.zeros(iter, batch_size, input_R, input_C, 3).to('cuda')


        # iterative update
        for i in range(iter):
            M_temp = torch.zeros_like(M, dtype=torch.float32, device=self.opt.device)
            # element-wise multiplication
            M_temp[:, 0:light_num, :] = r * M1
            M_temp[:, light_num:2 * light_num, :] = g * M2
            M_temp[:, 2 * light_num:3 * light_num, :] = b * M3
            # solve normal
            # invM: [3, 3*#patterns]
            # x:    [3, batch*R*C]
            # x = torch.linalg.lstsq(M_temp.permute(2,0,1), im_rendered).solution
            x = torch.linalg.lstsq(M_temp, im_rendered).solution
            # x = torch.linalg.pinv(M_temp) @ im_rendered

            # TODO: erase squeeze and unsqueeze, change zero mask
            x = (x.squeeze(-1))

            # black-pixel handling
            # zero_mask = x == 0.
            # x[zero_mask] = x[zero_mask] + 1e-4 # torch.tensor([0, 0, 1])

            # zero_mask = x == torch.tensor([0., 0., 0.], device=self.opt.device)
            # zero_mask = zero_mask[:,0]
            # x[zero_mask] = torch.tensor([-0., -0., -1.], device=self.opt.device)

            # x = x/torch.linalg.norm(x, axis=-2).unsqueeze(-1)
            x = x/(torch.linalg.norm(x, axis=-1).unsqueeze(-1) + 1e-8)
            x = x.unsqueeze(-1)

            # solve each channel
            # invR/G/B: [batch_size*camR*camC, 1, 9]

            # r_new = (torch.linalg.lstsq(M1.permute(2,0,1)@x, im_rendered_red).solution).squeeze(-1).permute(1,0)
            # g_new = (torch.linalg.lstsq(M2.permute(2,0,1)@x, im_rendered_green).solution).squeeze(-1).permute(1,0)
            # b_new = (torch.linalg.lstsq(M3.permute(2,0,1)@x, im_rendered_blue).solution).squeeze(-1).permute(1,0)

            # r_new = (torch.linalg.pinv(M1@x) @ im_rendered_red)
            # g_new = (torch.linalg.pinv(M2@x) @ im_rendered_green)
            # b_new = (torch.linalg.pinv(M3@x) @ im_rendered_blue)
            #------------------------------------------------------------------------------
            foreshortening = torch.clamp(incident@x, 1e-8, None)
            r_expose = monitor_light_radiance[:,0:light_num,:]@foreshortening
            g_expose = monitor_light_radiance[:,light_num:2*light_num,:]@foreshortening
            b_expose = monitor_light_radiance[:,2*light_num:3*light_num,:]@foreshortening

            r_new = (torch.linalg.lstsq(r_expose, im_rendered_red).solution)
            g_new = (torch.linalg.lstsq(g_expose, im_rendered_green).solution)
            b_new = (torch.linalg.lstsq(b_expose, im_rendered_blue).solution)


            # #------------------------------------------------------------------------------
            # mask = foreshortening>1e-8
            # M_ = monitor_light_radiance @ (incident*mask)

            # M1 = M_[:, 0:light_num, :]
            # M2 = M_[:, light_num:2*light_num, :]
            # M3 = M_[:, 2*light_num:3*light_num, :]
            # #------------------------------------------------------------------------------
            # # r_new = (torch.linalg.pinv(r_expose) @ im_rendered_red)
            # # g_new = (torch.linalg.pinv(g_expose) @ im_rendered_green)
            # # b_new = (torch.linalg.pinv(b_expose) @ im_rendered_blue)
            # #------------------------------------------------------------------------------
            r_ = r_new.reshape(batch_size, input_R, input_C)
            g_ = g_new.reshape(batch_size, input_R, input_C)
            b_ = b_new.reshape(batch_size, input_R, input_C)

            diffuse_albedo = torch.stack([r_, g_, b_], axis=-1)
            # diffuse_albedo = diffuse_albedo / diffuse_albedo.max()


            gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
            
            diffuse_albedo = diffuse_albedo / (gain_scalar * (self.opt.rendering_scalar * self.opt.cam_shutter_time) + 1e-8)
            # diffuse_albedo = diffuse_albedo / diffuse_albedo.max()


            # gain_exponent = torch.sigmoid(camera_gain)*48 + 1
            # diffuse_albedo = diffuse_albedo / ((self.opt.rendering_scalar * (self.opt.cam_gain_base ** gain_exponent) * self.opt.cam_shutter_time) + 1e-8)

            # r = diffuse_albedo[:,:,:,0].reshape(batch_size*input_R*input_C, 1, 1)
            # g = diffuse_albedo[:,:,:,1].reshape(batch_size*input_R*input_C, 1, 1)
            # b = diffuse_albedo[:,:,:,2].reshape(batch_size*input_R*input_C, 1, 1)
            r = diffuse_albedo[:,:,:,0].reshape(batch_size*input_R*input_C, 1, 1)
            r = torch.tile(r, (1, 4, 3))
            g = diffuse_albedo[:,:,:,1].reshape(batch_size*input_R*input_C, 1, 1)
            g = torch.tile(g, (1, 4, 3))
            b = diffuse_albedo[:,:,:,2].reshape(batch_size*input_R*input_C, 1, 1)
            b = torch.tile(b, (1, 4, 3))
            surface_normal = x.reshape(batch_size, input_R, input_C, 3)
            
            # plt.figure(figsize=(5,5),constrained_layout=True)
            # plt.imshow(((-surface_normal+1)/2)[0].detach().cpu().numpy())
            # plt.colorbar()
            # plt.savefig(os.path.join(self.opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + 'normal'+str(i)), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
            # plt.close()
            # print('Reconstruct ', torch.cuda.memory_allocated()/1024/1024, 'MB')
            iterated_normals[i] = surface_normal
        return iterated_normals, diffuse_albedo
