import time
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from options.option import Options
from models.dataset import *
from models.trainer import Trainer
from models.renderer import *
from models.reconstructor import *
import models.utils as utils
import models.trainer as trainer_utils

def train():
    # Option
    opt = Options().parse(save=True, isTrain=True)

    # Tensorboard
    writer = SummaryWriter(opt.tb_dir)
    print('=================================================================================')
    print('tensorboard output: %s' % opt.tb_dir)
    print('=================================================================================')

    # Image formation & Reconstruction & Trainer
    optics_model = LightDeskRenderer(opt)
    recon_model = Reconstructor(opt)
    last_epoch = -1 if opt.load_step_start == 0 else opt.load_epoch
    trainer = Trainer(opt, optics_model, recon_model, last_epoch)

    # Step
    total_step = 0 # opt.load_step_start
    debug_step = 0

    train_dataset = CreateBasisDataset(opt, 'selected_train')
    validation_dataset = CreateBasisDataset(opt, 'selected_test')

    dataloader_train = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_threads, pin_memory=True) # drop_last=True
    dataloader_valid = DataLoader(validation_dataset, batch_size=opt.batch_size_valid, num_workers=opt.num_threads, pin_memory=True)
    
    loss_values_for_saving = []
    # Training
    for epoch in range(1, opt.n_epoch+1):

        # Validation
        if debug_step%1==0:        
            valid_loss_sum = 0
            for i, data_ in enumerate(dataloader_valid):
                with torch.no_grad():
                    model_results = trainer.run_model(data_)
                loss = sum(model_results['losses'].values())
                valid_loss_sum += loss

                for j in range(len(data_['scene'])):
                    utils.display(writer, model_results, total_step, f'valid{i}', j, os.path.join(opt.tb_dir, 'validation'))

                if i == opt.valid_N-1:
                    break
            valid_loss_sum /= opt.valid_N
            loss_values_for_saving.append(valid_loss_sum.item())
            writer.add_scalar('valid/loss', valid_loss_sum, total_step)

        train_loss_sum = 0
        train_loss_normal_sum = 0


        for data in tqdm(dataloader_train, leave=False):
            batch_size = len(data['scene'])

            utils.save_model_pattern(trainer.monitor_light_patterns, epoch, total_step, os.path.join(opt.tb_dir, 'patterns'))
            utils.save_model_position(trainer.monitor_superpixel_positions, epoch, total_step, os.path.join(opt.tb_dir, 'positions'))

            # Run a optimization step
            trainer.run_optimizers_one_step(data)

            loss_dict = trainer.get_losses()
            
            loss = sum(loss_dict.values()).data
            train_loss_sum += loss*batch_size
            loss_normal = loss_dict['normal'].data
            train_loss_normal_sum += loss_normal*batch_size


            total_step += batch_size
            debug_step += 1
            writer.add_scalar('train', loss, total_step)

        trainer.run_schedulers_one_step()

        writer.add_scalar('train/batch', train_loss_sum/len(train_dataset), total_step)
        writer.add_scalar('train/batch_normal', train_loss_normal_sum/len(train_dataset), total_step)
        utils.print_training_status(epoch, opt.n_epoch, total_step, debug_step)
    
    loss_np = np.array(loss_values_for_saving)
    np.save(os.path.join(opt.tb_dir, f'{opt.initial_pattern}_{opt.used_loss}_optmization_loss_valid'), loss_np)
    writer.close()

if __name__=='__main__':
    train()