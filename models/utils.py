import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import trainer as trainer_utils
import datetime


# def print_training_status(writer, epoch, n_epoch, step, debug_step, loss_dict):
def print_training_status(epoch, n_epoch, step, debug_step):
    message = f'(epoch: {epoch}/{n_epoch}, iters: {step}, debug_iters: {debug_step}) '
    # message += 'total: [%.6f]: ' % sum(loss_dict.values())
    # writer.add_scalar('train/total', sum(loss_dict.values()).data, step)
    # for loss_name, loss in loss_dict.items():
    #     message += '%s:%.6f, ' % (loss_name, loss)
    #     writer.add_scalar('train/%s' % loss_name, loss.data, step)

    print(message)


def display(writer, model_results, step, mode, batch_index, dir):

    results = trainer_utils.model_results_to_np(model_results, batch_ind=batch_index)

    writer.add_image(f'normal_est/{mode+str(batch_index)}', ((-results['normal_est'].transpose((2,0,1))+1)/2), step)
    writer.add_image(f'albedo_est/{mode+str(batch_index)}', results['albedo_est'].transpose((2,0,1)), step)


def save_model(monitor_light_patterns, superpixel_position, epoch, step, dir):

    if monitor_light_patterns is not None:
        torch.save(monitor_light_patterns, os.path.join(dir, f'monitor_light_patterns_epoch{str(epoch).zfill(5)}_step{str(step).zfill(5)}.pth'))
        torch.save(monitor_light_patterns, os.path.join(dir, 'monitor_light_patterns_latest.pth'))

        torch.save(superpixel_position, os.path.join(dir, f'superpixel_position_epoch{str(epoch).zfill(5)}_step{str(step).zfill(5)}.pth'))
        torch.save(superpixel_position, os.path.join(dir, 'superpixel_position_latest.pth'))

def save_model_pattern(monitor_light_patterns, epoch, step, dir):
    if monitor_light_patterns is not None:
        torch.save(monitor_light_patterns, os.path.join(dir, f'monitor_light_patterns_epoch{str(epoch).zfill(5)}_step{str(step).zfill(5)}.pth'))
        torch.save(monitor_light_patterns, os.path.join(dir, 'monitor_light_patterns_latest.pth'))
        
def save_model_position(superpixel_position, epoch, step, dir):
    if superpixel_position is not None:
        torch.save(superpixel_position, os.path.join(dir, f'superpixel_position_epoch{str(epoch).zfill(5)}_step{str(step).zfill(5)}.pth'))
        torch.save(superpixel_position, os.path.join(dir, 'superpixel_position_latest.pth'))

def load_monitor_light_patterns(outdir, epoch, latest=False):
    if latest:
        fn = os.path.join(outdir, 'monitor_light_patterns_latest.pth')
    else:
        fn = os.path.join(outdir, 'monitor_light_patterns_epoch%d.pth' % (epoch))
    if not os.path.isfile(fn):
        raise FileNotFoundError('%s not exists yet!' % fn)
    else:
        return torch.load(fn)


def load_camera_gain(outdir, epoch, latest=False):
    if latest:
        fn = os.path.join(outdir, 'camera_gain_latest.pth')
    else:
        fn = os.path.join(outdir, 'camera_gain_epoch%d.pth' % (epoch))
    if not os.path.isfile(fn):
        raise FileNotFoundError('%s not exists yet!' % fn)
    else:
        return torch.load(fn)


def cut_edge_batch(target_img):
    # torch tensor
    h = target_img.size(2)
    w = target_img.size(3)

    h1 = (h % 8)//2
    w1 = (w % 8)//2
    h2 = h - ((h % 8)-h1)
    w2 = w - ((w % 8)-w1)
    target_img = target_img[:, :, h1:h2, w1:w2]
    return target_img


def cut_edge(target_img):
    # numpy ndarray
    h = target_img.shape[0]
    w = target_img.shape[1]

    h1 = (h % 8)//2
    w1 = (w % 8)//2
    h2 = h - ((h % 8)-h1)
    w2 = w - ((w % 8)-w1)
    target_img = target_img[h1:h2, w1:w2]
    return target_img

    
def visualize3D(opt, rgb, ptcloud, light_pos, camera_pos, reference_plane, reference_point):
    # This code visualizes ptcloud, monitor, camera

    fig = plt.figure(figsize=(18, 15))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    # draw 3D scene
    ax.scatter(ptcloud[..., 0], ptcloud[..., 1], ptcloud[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.view_init(-45, -90)
    # draw monitor light source
    ax.scatter(light_pos[..., 0], light_pos[..., 1], light_pos[..., 2], c='red', marker='*', s=10)
    ax.scatter(light_pos[0,0, 0], light_pos[0,0, 1], light_pos[0,0, 2], c='blue', marker='*', s=20)
    # draw camera center
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='blue', marker='o', s=20)
    # ax.scatter(reference_plane[..., 0], reference_plane[..., 1], reference_plane[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    # ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='green', marker='o', s=20)

    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    ax.set_zlim([-0., 2.])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    # draw 3D scene
    ax.scatter(ptcloud[..., 0], ptcloud[..., 1], ptcloud[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.view_init(0, 0)
    # draw monitor light source
    ax.scatter(light_pos[..., 0], light_pos[..., 1], light_pos[..., 2], c='red', marker='*', s=10)
    # draw camera center
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='blue', marker='o', s=20)
    # ax.scatter(reference_plane[..., 0], reference_plane[..., 1], reference_plane[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    # ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='green', marker='o', s=20)

    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    ax.set_zlim([-0., 2.])

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    # draw 3D scene
    ax.scatter(ptcloud[..., 0], ptcloud[..., 1], ptcloud[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.view_init(0, 90)
    # draw monitor light source
    ax.scatter(light_pos[..., 0], light_pos[..., 1], light_pos[..., 2], c='red', marker='*', s=10)
    # draw camera center
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='blue', marker='o', s=20)
    # ax.scatter(reference_plane[..., 0], reference_plane[..., 1], reference_plane[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    # ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='green', marker='o', s=20)

    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    ax.set_zlim([-0., 2.])

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    # draw 3D scene
    ax.scatter(ptcloud[..., 0], ptcloud[..., 1], ptcloud[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.view_init(90, 0)
    # draw monitor light source
    ax.scatter(light_pos[..., 0], light_pos[..., 1], light_pos[..., 2], c='red', marker='*', s=10)
    # draw camera center
    ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], c='blue', marker='o', s=20)
    # ax.scatter(reference_plane[..., 0], reference_plane[..., 1], reference_plane[..., 2], c=rgb.reshape(-1, 3), marker='.', s=1, alpha=0.25)
    # ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='green', marker='o', s=20)

    # ax.set_xlim([-1.5, 1.5])
    # ax.set_ylim([-1.5, 1.5])
    # ax.set_zlim([-0.5, 2.5])
    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    ax.set_zlim([-0., 2.])
    
    # plt.show()
    plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + 'pointcloud'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_patterns(opt, monitor_light_patterns):
    plt.figure(figsize=(10, 2*opt.light_N), constrained_layout=True)
    plt.suptitle('Pattern')
    for illum_idx in range(opt.light_N):
        pattern = (monitor_light_patterns[illum_idx])
        r = np.zeros_like(pattern)
        g = np.zeros_like(pattern)
        b = np.zeros_like(pattern)
        r[:, :, 0] = pattern[:, :, 0]
        g[:, :, 1] = pattern[:, :, 1]
        b[:, :, 2] = pattern[:, :, 2]

        plt.subplot(opt.light_N, 4, illum_idx*4+1)
        plt.imshow(pattern)
        plt.title(f'Light pattern {illum_idx+1} RGB Channel')

        plt.subplot(opt.light_N, 4, illum_idx*4+2)
        plt.imshow(r)
        plt.title(f'Light pattern {illum_idx+1} R Channel')

        plt.subplot(opt.light_N, 4, illum_idx*4+3)
        plt.imshow(g)
        plt.title(f'Light pattern {illum_idx+1} G Channel')

        plt.subplot(opt.light_N, 4, illum_idx*4+4)
        plt.imshow(b)
        plt.title(f'Light pattern {illum_idx+1} B Channel')
    plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_0_Monitor_patterns'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_GT_data(opt, file_names, monitor_light_patterns, I_diffuse):
    batch_size = len(file_names)
    for batch_idx in range(batch_size):
        plt.figure(figsize=(8, opt.light_N*2), constrained_layout=True)
        plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])
        for illum_idx in range(opt.light_N):
            plt.subplot(opt.light_N, 2, illum_idx*2+1)
            plt.imshow((monitor_light_patterns[illum_idx]))
            plt.title(f'Light pattern {illum_idx+1}')
            plt.subplot(opt.light_N, 2, illum_idx*2+2)
            plt.imshow(I_diffuse[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_L{illum_idx+1}_Diffuse')
        plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_1_{batch_idx+1}th_Data_Rendered_images.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()

def visualize_EST_normal(opt, file_names, normal_gt, normal_est, mask):
    batch_size = len(file_names)
    for batch_idx in range(batch_size):
        plt.figure(figsize=(8,8), constrained_layout=True)
        plt.suptitle(f'Batch {batch_idx + 1}: ' + file_names[batch_idx])

        plt.subplot(2, 2, 1)
        plt.imshow(((-normal_gt[batch_idx]+1)/2))
        plt.title(f'Batch{batch_idx+1}_Normal_GT')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(((-normal_est[batch_idx]+1)/2))
        plt.title(f'Batch{batch_idx+1}_Normal_EST')
        plt.colorbar()


        # plt.imshow((1-np.abs(normal_gt[batch_idx] * normal_est[batch_idx]).sum(-1)))
        # plt.colorbar()
        # plt.title(f'Batch{batch_idx+1}_Normal_LOSS')
        # temp = (1-np.abs(normal_gt[batch_idx] * normal_est[batch_idx]).sum(-1))
        # temp[temp==1] = 0

        normal_cos_error = 1 - np.abs((normal_gt[batch_idx] * normal_est[batch_idx]).sum(-1))
        normal_angular_error = np.rad2deg(np.arccos((normal_gt[batch_idx] * normal_est[batch_idx]).sum(-1)))
        # temp = (1-np.abs(normal_gt[batch_idx] * normal_est[batch_idx]).sum(-1))
        # normal_cos_error[normal_cos_error==1] = 0
        

        plt.subplot(2, 2, 2)
        plt.imshow(normal_cos_error * mask[batch_idx])
        plt.colorbar()
        plt.title(f'Batch{batch_idx+1}_Normal_Cosine_LOSS')


        plt.subplot(2, 2, 4)
        plt.imshow(normal_angular_error * mask[batch_idx])
        plt.colorbar()
        plt.title(f'Batch{batch_idx+1}_Normal_Angular_LOSS')

        plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_1_{batch_idx+1}th_Data_EST_Normal.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
        plt.close()
        
def visualize_EST_diffuse(opt, file_names, I_diffuse, diffuse_est, mask):
    batch_size = len(file_names)
    for batch_idx in range(batch_size):
        plt.figure(figsize=(8,8), constrained_layout=True)
        plt.suptitle(f'Batch {batch_idx + 1}: ' + file_names[batch_idx])

        plt.subplot(2, 2, 1)
        plt.imshow(((-normal_gt[batch_idx]+1)/2))
        plt.title(f'Batch{batch_idx+1}_Normal_GT')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        plt.imshow(((-normal_est[batch_idx]+1)/2))
        plt.title(f'Batch{batch_idx+1}_Normal_EST')
        plt.colorbar()

        diffuse_error = torch.abs(I_diffuse - I_diffuse_est)
        
        plt.subplot(2, 2, 2)
        plt.imshow(normal_cos_error * mask[batch_idx])
        plt.colorbar()
        plt.title(f'Batch{batch_idx+1}_Normal_Cosine_LOSS')


        plt.subplot(2, 2, 4)
        plt.imshow(normal_angular_error * mask[batch_idx])
        plt.colorbar()
        plt.title(f'Batch{batch_idx+1}_Normal_Angular_LOSS')

        plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_3_{batch_idx+1}th_Data_EST_Rerender.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
        plt.close()

def visualize_unsup_error(opt, file_names, monitor_light_patterns, I_diffuse, I_diffuse_est, I_diffuse_error):
    batch_size = len(file_names)
    for batch_idx in range(batch_size):
        plt.figure(figsize=(8, opt.light_N*2), constrained_layout=True)
        plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])
        for illum_idx in range(opt.light_N):
            plt.subplot(opt.light_N, 4, illum_idx*4+1)
            plt.imshow((monitor_light_patterns[illum_idx]))
            plt.title(f'Light pattern {illum_idx+1}')

            plt.subplot(opt.light_N, 4, illum_idx*4+2)
            plt.imshow(I_diffuse[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_L{illum_idx+1}_I_GT')
            
            plt.subplot(opt.light_N, 4, illum_idx*4+3)
            plt.imshow(I_diffuse_est[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_L{illum_idx+1}_I_Rerender')

            plt.subplot(opt.light_N, 4, illum_idx*4+4)
            plt.imshow(I_diffuse_error[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_L{illum_idx+1}_Error')

        plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f'_2_{batch_idx+1}th_Data_Rerender.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()

def visualize_length_err(opt, loss_length_ver_vis, loss_length_hor_vis):
    plt.figure(figsize=(4, 2), constrained_layout=True)
    # plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])

    plt.subplot(1, 2, 1)
    plt.imshow((loss_length_ver_vis.reshape(8,16,-1)))
    plt.colorbar()
    plt.title(f'vertical length error')

    plt.subplot(1, 2, 2)
    plt.imshow((loss_length_hor_vis.reshape(9,15,-1)))
    plt.colorbar()
    plt.title(f'horizontal length error')

    plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%H%M%S") + f'_01_monitor_length.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_perpendicular_err(opt, loss_perpendicular_vis):
    plt.figure(figsize=(2, 2), constrained_layout=True)
    # plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])

    plt.subplot(1, 1, 1)
    plt.imshow((loss_perpendicular_vis.reshape(8,15,-1)[:,:,0]))
    plt.colorbar()
    plt.title(f'perpendicular error')

    plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%H%M%S") + f'_02_monitor_perpendicular.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
    plt.close()
    
def visualize_parallel_err(opt, loss_parallel_vertical_vis, horizontal_parallel_tensor):
    # plt.figure(figsize=(4, 2), constrained_layout=True)
    # plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])

    # plt.subplot(1, 2, 1)
    # plt.imshow((loss_parallel_vertical_vis.reshape(8,15,-1)))
    # plt.title(f'vertical length error')

    plt.figure(figsize=(4, 9*2), constrained_layout=True)
    for i in range(9):
        plt.subplot(9, 2, i*2 + 2)
        plt.imshow((horizontal_parallel_tensor[i].reshape(9,15,-1)))
        plt.colorbar()
        plt.title(f'horizontal length error')

    plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%H%M%S") + f'_03_monitor_parallel.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_weight_specular(opt, batch_size, file_names, weight_map, basis_parameters):
    for batch_idx in range(batch_size):
        plt.figure(figsize=(8,8), constrained_layout=True)
        plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])
        for basis_idx in range(opt.num_basis):
            plt.subplot(3, 3, basis_idx+1)
            plt.imshow(weight_map[batch_idx, basis_idx])
            plt.colorbar()
            plt.title(f"Batch{batch_idx+1}'s {basis_idx+1}'th weight map")
        plt.subplot(3,3,8)
        plt.imshow((basis_parameters[batch_idx,:,0].reshape(opt.num_basis,1,1,1)*weight_map[batch_idx]).sum(axis=0))
        plt.colorbar()
        plt.title(f"Batch{batch_idx+1}'s specular map")
        plt.subplot(3,3,9)
        plt.imshow((basis_parameters[batch_idx,:,1].reshape(opt.num_basis,1,1,1)*weight_map[batch_idx]).sum(axis=0))
        plt.colorbar()
        plt.title(f"Batch{batch_idx+1}'s roughness map")
        plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_3_EST_weight_specular.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
        plt.close()

def visualize_comparison(opt, batch_size, file_names, I_diffuse, I_specular, I_diffuse_est, I_specular_est):
    I_observed = np.clip(I_diffuse + I_specular, 0, 1)
    I_recon = np.clip((I_diffuse_est + I_specular_est),0,1)

    for batch_idx in range(batch_size):
        plt.figure(figsize=(8, 2*opt.light_N), constrained_layout=True)
        plt.suptitle(f'Batch {batch_idx+1}: '+file_names[batch_idx])
        for illum_idx in range(opt.light_N):
            plt.subplot(opt.light_N*2, 3, illum_idx*6+1)
            plt.imshow(I_observed[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_I{illum_idx+1}_GT_Rendered')
            plt.subplot(opt.light_N*2, 3, illum_idx*6+2)
            plt.imshow(I_diffuse[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_I{illum_idx+1}_GT_Diffuse')
            plt.subplot(opt.light_N*2, 3, illum_idx*6+3)
            plt.imshow(I_specular[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_I{illum_idx+1}_GT_Specular')

            plt.subplot(opt.light_N*2, 3, illum_idx*6+4)
            plt.imshow(I_recon[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_I{illum_idx+1}_EST_Rendered')
            plt.subplot(opt.light_N*2, 3, illum_idx*6+5)
            plt.imshow(I_diffuse_est[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_I{illum_idx+1}_EST_Diffuse')
            plt.subplot(opt.light_N*2, 3, illum_idx*6+6)
            plt.imshow(I_specular_est[batch_idx, illum_idx])
            plt.title(f'B{batch_idx+1}_I{illum_idx+1}_EST_Specular')
        plt.savefig(os.path.join(opt.tb_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_4_Compare.png'), facecolor='#eeeeee', bbox_inches='tight', dpi=300)
        plt.close()
