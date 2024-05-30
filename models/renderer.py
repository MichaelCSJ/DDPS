# This code is for rendering images using data from 'dataset.py'
# LightDeskRenderer compute Cook-Torrance brdf model

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
        # monitor_light_radiance = torch.sigmoid(monitor_light_pattern_nonlinear) ** (1/2.06) # ** self.opt.monitor_gamma
        # monitor_light_radiance = torch.sigmoid(monitor_light_pattern_nonlinear) #** (1/2.06)# ** self.opt.monitor_gamma
        monitor_light_radiance = torch.flip(monitor_light_radiance, dims=[2])
        monitor_light_radiance = monitor_light_radiance.reshape(self.opt.light_N, -1, 3).unsqueeze(0).unsqueeze(-1)

        basis_images = basis_images.permute(0,1,4,2,3)
        # weights: [batch, lightN, R*C, 3, H*W]
        basis_images = basis_images.reshape(batch_size, self.opt.light_R*self.opt.light_C, 3, -1).unsqueeze(1)
        # print(basis_images.shape, monitor_light_radiance.shape)
        result = monitor_light_radiance * basis_images
        result = result.sum(axis=2)
        result = result.reshape(batch_size, self.opt.light_N, 3, self.opt.cam_R, self.opt.cam_C)
        result = result.permute(0,1,3,4,2)


        gain_scalar = self.opt.cam_gain_base ** (torch.sigmoid(camera_gain)*48)
        
        I_diffuse = gain_scalar * (self.opt.rendering_scalar * self.opt.cam_shutter_time * result)
        I_diffuse = torch.clamp(I_diffuse, 1e-8, 1)

        return I_diffuse

          
    # def render_vec(self, normal, albedo, specular, roughness, monitor_light_pattern_nonlinear, incident, exitant, camera_gain):
    def render_vec(self, normal, albedo, monitor_light_pattern_nonlinear, incident, camera_gain):
        """
        Render images using Cook-Torrance BRDF equation

        Args:
            albedo: diffuse albedo
            normal: normal map of batch
            ptcloud: point cloud (camR, camC, 3) or (1,1,3)
            camera_gain: gain parameter
            monitor_light_pattern_nonlinear: lighting patterns
            light_pos: coordinates of each point light source of patterns
            specular: specular coefficients from dataset.py
            roughness: roughness form dataset.py

        Return
            scaled_image: rendered image

        """
        # illumination
        monitor_light_radiance= torch.sigmoid(monitor_light_pattern_nonlinear) ** self.opt.monitor_gamma
        batch_size = normal.shape[0]

        # normal_flatten:   [batch_size, roi_height * roi_width, rgb]
        # specular:         [batch_size, roi_height * roi_width, rgb]
        # roughness:        [batch_size, roi_height * roi_width, 1]

        # foreshortening: [batch_size, cam_R, cam_C] -> [batch_size, #pattenrs, cam_R, cam_C, rgb]
        foreshortening = (normal.unsqueeze(0).unsqueeze(0) * incident.reshape(self.opt.light_R, self.opt.light_C, batch_size, self.opt.roi_height, self.opt.roi_width, 3)).sum(axis=-1)
        foreshortening = torch.clip(foreshortening, 0, None)

        # specular reflectance
        # specular_reflectance = self.compute_specular(incident, exitant, normal[None,None,...], specular[None,None,...,None], roughness[None,None,...,None])
        # specular_reflectance = specular_reflectance.reshape(self.opt.light_R, self.opt.light_C, batch_size, self.opt.roi_height, self.opt.roi_width, 1)

        # lighting term including the cosine term and the monitor pattern
        lighting_geom_term = foreshortening[None,...,None] * (monitor_light_radiance.unsqueeze(3).unsqueeze(3).unsqueeze(3))  # ( #pattern, (light_r), (light_c), batch_size, camR, camC, rgb]

        rendered_image_diffuse = (lighting_geom_term * albedo[None,None,None,...]).sum(axis=(1,2)).transpose(0,1)    # ( #pattern, batch_size, camR, camC, rgb]
        # rendered_image_specular = (lighting_geom_term * specular_reflectance[None,...]).sum(axis=(1,2)).transpose(0,1)  # ( #pattern, batch_size, camR, camC, rgb]

        # RAW rendering
        gain_exponent = torch.sigmoid(camera_gain)*48
        scalar = self.opt.rendering_scalar * (self.opt.cam_gain_base**gain_exponent)
        noise_diffuse = torch.normal(0, self.opt.noise_sigma, size=rendered_image_diffuse.shape).to('cuda')
        # noise_specular = torch.normal(0, self.opt.noise_sigma, size=rendered_image_specular.shape).to('cuda')

        raw_image_diffuse = torch.clamp(scalar * ((self.opt.cam_shutter_time * rendered_image_diffuse) + noise_diffuse), 1e-8, 1)
        # raw_image_specular = torch.clamp(scalar * ((self.opt.cam_shutter_time * rendered_image_specular) + noise_specular), 1e-8, 1)
        # raw_image_diffuse = scalar * ((self.opt.cam_shutter_time * rendered_image_diffuse) + noise_diffuse)
        # raw_image_specular = scalar * ((self.opt.cam_shutter_time * rendered_image_specular) + noise_specular)
        # print(raw_image_specular.max())
        # print(raw_image_specular.min())

        return raw_image_diffuse#, raw_image_specular

    def render_basisBRDF(self, normal, albedo, basis_parameters, weight_map, monitor_light_pattern_nonlinear, camera_gain, incident, exitant, insert_noise=False):
        """
        Render images using the Cook-Torrance basis BRDF equation

        Args:
            albedo: diffuse albedo
            normal: normal map of batch
            ptcloud: point cloud (camR, camC, 3) or (1,1,3)
            camera_gain: gain parameter
            monitor_light_pattern: lighting patterns
            light_pos: coordinates of each point light source of patterns
            basis_parameters: basis BRDFs' specular coefficiens and roughness
            weight_map: basis BRDF weight map
            incident
            exitant
        Return
            scaled_image: rendered image
        """

        # illumination
        monitor_light_radiance= torch.sigmoid(monitor_light_pattern_nonlinear) ** self.opt.monitor_gamma
        batch_size = normal.shape[0]

        # foreshortening
        foreshortening = (normal.unsqueeze(0).unsqueeze(0) * incident.reshape(self.opt.light_R, self.opt.light_C, batch_size, self.opt.roi_height, self.opt.roi_width, 3)).sum(axis=-1)
        foreshortening = torch.clamp(foreshortening, 0, None)

        # lighting term including the cosine term and the monitor pattern
        lighting_geom_term = foreshortening[None,...,None] * (monitor_light_radiance.unsqueeze(3).unsqueeze(3).unsqueeze(3))  # ( #pattern, (light_r), (light_c), batch_size, camR, camC, rgb]

        # diffuse renering
        rendered_image_diffuse = (lighting_geom_term * albedo[None,None,None,...]).sum(axis=(1,2)).transpose(0,1)    # ( #pattern, batch_size, camR, camC, rgb]

        # specular renering with basis BRDF
        print(basis_parameters)
        specular = basis_parameters[:, :, 0].T
        specular = specular.unsqueeze(1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        roughness = basis_parameters[:, :, 1].T
        # roughness = roughness**2
        roughness = roughness.unsqueeze(1).unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        # roughness = torch.sqrt(1/(roughness+1e-10))

        # first dimension is the basis BRDF
        # ([1, 18, 32, 2, 14400, 1])
        specular_reflectance_basis = self.compute_specular(incident[None], exitant[None], normal[None,None,None,...], specular[..., None], roughness[..., None])
        specular_reflectance_basis = specular_reflectance_basis.reshape(self.opt.num_basis, self.opt.light_R, self.opt.light_C, batch_size, self.opt.roi_height, self.opt.roi_width, 1)
        specular_reflectance = (specular_reflectance_basis * weight_map.transpose(0,1).unsqueeze(1).unsqueeze(1)).sum(axis=0)  # sum of basis

        rendered_image_specular = (lighting_geom_term * specular_reflectance[None,...]).sum(axis=(1,2)).transpose(0,1)  # ( #pattern, batch_size, camR, camC, rgb]

        # RAW rendering
        gain_exponent = torch.sigmoid(camera_gain)*48 + 1
        scalar = self.opt.rendering_scalar * (self.opt.cam_gain_base**gain_exponent)
        if insert_noise:
            noise_specular = torch.normal(0, self.opt.noise_sigma, size=rendered_image_specular.shape).to('cuda')
            noise_diffuse = torch.normal(0, self.opt.noise_sigma, size=rendered_image_diffuse.shape).to('cuda')
        else:
            noise_specular = 0
            noise_diffuse = 0

        raw_image_specular = torch.clamp(scalar * ((self.opt.cam_shutter_time * rendered_image_specular) + noise_specular), 1e-8, 1)
        raw_image_diffuse = torch.clamp(scalar * ((self.opt.cam_shutter_time * rendered_image_diffuse) + noise_diffuse), 1e-8, 1)

        return raw_image_diffuse, raw_image_specular

    def compute_specular(self, i, o, n, rho_s, m, F=0.98):
        """
        Compute the diffuse/specular Cook-Torrance BRDF model for three channels

        Args
            i: incident vector
                dim: [# of px, 3]
            o: exitant vector
                dim: [# of px, 3]
            n: surface normal
                dim: [# of px, 3]
            rho_s: specular albedo
                dim: [# of px, 3]
            m: roughness
                dim: [# of px, 1]
            F: Fresnel term
                dim: [# of px, 3]

        Return
            f: BRDF
                dim: [# of px, 3]
            fd: diffuse BRDF
                dim: [# of px, 3]
            fs: specular BRDF
                dim: [# of px, 3]
        """

        h = self.unit_vector((i + o) / 2)  # [# of px, 3]

        # compute useful angles
        ni, no, nh = self.dot(i, n), self.dot(o, n), self.dot(h, n)
        # ni, no, nh = torch.clamp(ni, -1 + 1e-8, 1 - 1e-8), torch.clamp(no, -1 + 1e-8, 1 - 1e-8), torch.clamp(nh, -1 + 1e-8, 1 - 1e-8)
        ni, no, nh = torch.clamp(ni, 1e-8, 1 - 1e-8), torch.clamp(no, 1e-8, 1 - 1e-8), torch.clamp(nh, 1e-8, 1 - 1e-8)


        # -------------- specular ---------------

        # micro-facet constants
        tan_th_h = torch.sqrt(torch.abs(1 - nh ** 2)) / (nh + 1e-8)

        # d = self.compute_Beckmann_NDF(m, nh, tan_th_h)
        d = self.compute_GGX_NDF(m, nh, tan_th_h)

        # tan_th_i = torch.sqrt(torch.abs(1 - ni ** 2)) / (ni + 1e-8)
        # tan_th_o = torch.sqrt(torch.abs(1 - no ** 2)) / (no + 1e-8)
        tan_th_i = torch.sqrt(1 - ni ** 2 + 1e-8) / (ni + 1e-8)
        tan_th_o = torch.sqrt(1 - no ** 2 + 1e-8) / (no + 1e-8)
        g = self.compute_Smith_SMF(m, tan_th_i, tan_th_o)

        # specular model
        fs = rho_s * F * d * g / (4 * no * ni + 1e-8)

        return fs

    def unit_vector(self, x):
        """
        Normalize a 3D vector

        Args
            x: vector
                dim: [..., 3]
        Return
            x_n: normalized vector
                dim: [..., 3]
        """
        # x_mag = torch.sqrt(torch.abs(torch.sum(x ** 2, -1)))
        x_mag = torch.sqrt(torch.sum(x ** 2, -1)+1e-8)
        x_n = x / (x_mag[..., np.newaxis] + 1e-8)
        return x_n

    def dot(self, x1, x2):
        """
        dot product of two vectors

        Args
            x1: vector 1
                dim: [..., interested axis]
            x2: vector 2
                dim: [..., interested axis]
        Return
            y: dot product of the two vectors
                dim: [..., 1]
        """
        return torch.sum(x1 * x2, -1)[..., np.newaxis]

    def compute_Beckmann_NDF(self, m, cos_th_h, tan_th_h):
        """
        Compute the Beckmann's normal distribution function

        Args
            m: roughness
                dim: [# of px, 1]
            th_h: half-wave angle in rad
                dim: [# of px, 1]
        Return
            D: Beckmann's distribution
                dim: [# of px, 1]
        """

        tan_th_h[tan_th_h > 1e3] = 1e3

        D = torch.exp(-((-tan_th_h / m) ** 2)) / ((m ** 2) * (cos_th_h ** 4) + 1e-8)  #
        return D

    def compute_GGX_NDF(self, m, cos_th_h, tan_th_h):

        tan_th_h[tan_th_h > 1e3] = 1e3

        D = m**2 / (4 * cos_th_h**4 * (m**2 + tan_th_h**2)**2 + 1e-8)

        return D

    def compute_Smith_SMF(self, m, tan_th_i, tan_th_o):
        """
        Compute the Smith's shading/masking function

        Args
            m: roughness
                dim: [# of px, 1]
            tan_th_i: incident angle in rad
                dim: [# of px, 1]
            tan_th_o: exitant angle in rad
                dim: [# of px, 1]
        Return
            S: Smith's distribution
                dim: [# of px, 1]
        """
        G = (2 / (1 + torch.sqrt(1 + (m ** 2) * (tan_th_i ** 2)))) * (
                    2 / (1 + torch.sqrt(1 + (m ** 2) * (tan_th_o ** 2))))
        return G