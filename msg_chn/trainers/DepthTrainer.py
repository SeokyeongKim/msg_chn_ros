"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from trainers.trainer import Trainer  # from CVLPyDL repo
import torch
import random
import matplotlib.pyplot as plt
import os.path
from utils.AverageMeter import AverageMeter
from utils.saveTensorToImage import *
from utils.ErrorMetrics import *
import time
from modules.losses import *
import cv2
import wandb

err_metrics = ['MAE(self.device)', 'RMSE(self.device)','iMAE(self.device)', 'iRMSE(self.device)']
err_metrics_name = ['MAE(mm)', 'RMSE(mm)','iMAE(1/km)', 'iRMSE(1/km)']

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class KittiDepthTrainer(Trainer):
    def __init__(self, net, params, optimizer, objective, lr_scheduler, dataloaders, 
                 dataset_sizes, workspace_dir, sets=['train', 'val'], device = 'cpu', 
                 use_load_checkpoint=None, debug = False, weights = [1.0, 1.0], 
                 useWandb = False, runName = " "):

        # Call the constructor of the parent class (trainer)
        super(KittiDepthTrainer, self).__init__(net, optimizer, lr_scheduler, objective, 
                                                use_gpu=params.use_gpu,
                                                workspace_dir=workspace_dir)

        self.lr_scheduler = lr_scheduler
        self.dataloaders = dataloaders
        self.dataset_sizes = dataset_sizes
        self.use_load_checkpoint = use_load_checkpoint
        self.params = params
        self.save_chkpt_each = params.save_chkpt_each
        self.sets = sets
        if (params.save_out_imgs):
            self.img_dir = os.path.join(self.workspace_dir, "predictions/")
            self.save_images = saveTensorToImage(self.img_dir, debug)
        self.load_rgb = params.load_rgb if 'load_rgb' in params else False
        self.exp_name = params.exp_name
        self.print_time_epoch = params.print_time_each_epoch
        self.print_time_iter = params.print_time_each_iter
        self.useWandb = useWandb
        self.debug = debug
        self.device = device
        for s in self.sets: self.stats[s + '_loss'] = []


    ####### Training Function #######
    def train(self):
        self.header("Experiment Parameters")
        for key, value in (self.params.items()):
            print("%s: %r" % (key, value))
        print("*" * 60)

        # Load last save checkpoint
        if self.use_load_checkpoint != None:
            if isinstance(self.use_load_checkpoint, int):
                if self.use_load_checkpoint > 0:
                    print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint))
                    if self.load_checkpoint(self.use_load_checkpoint):
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
                elif self.use_load_checkpoint == -1:
                    print('=> Loading last checkpoint ...')
                    if self.load_checkpoint():
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
            elif isinstance(self.use_load_checkpoint, str):
                print('loading checkpoint from : ' + self.use_load_checkpoint)
                if self.load_checkpoint(self.use_load_checkpoint):
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')

        start_full_time = time.time()
        print('\nEpoch starting time:', time.strftime('%d/%m/%y %H:%M:%S', time.localtime(time.time())))

        for epoch in range(self.params.epochs):
            start_epoch_time = time.time()

            self.epoch = epoch
            self.header(('Training Epoch {}: (lr={}) '.format(epoch, self.optimizer.param_groups[0]['lr'])))
            # print('Training Epoch {}: (lr={}) '.format(epoch, self.optimizer.param_groups[0]['lr'])) 

            # Train the epoch
            loss_meter, train_err = self.train_epoch()
            self.lr_scheduler.step()  # LR decay //chec, if this change was correctly made


            # Add the average loss for this epoch to stats
            for s in self.sets: self.stats[s + '_loss'].append(loss_meter[s].avg)

            # Save checkpoint
            if self.use_save_checkpoint and (self.epoch) % self.save_chkpt_each == 0:
                self.save_checkpoint()
                print('\n => Checkpoint was saved successfully!\n')

            end_epoch_time = time.time()
            print('End the %d th epoch at ' % self.epoch)
            print(time.strftime('%m.%d.%H:%M:%S\n', time.localtime(time.time())))
            epoch_duration = end_epoch_time - start_epoch_time
            self.training_time += epoch_duration
            if self.print_time_epoch:
                hours, rem = divmod(self.training_time, 3600)
                minutes, seconds = divmod(rem, 60)
                hours_epoch, rem_epoch = divmod(epoch_duration, 3600)
                minutes_epoch, seconds_epoch = divmod(rem_epoch, 60)
                print("Trained for {:0>2}:{:0>2}:{:05.2f}, and  {:0>2}:{:0>2}:{:05.2f} per epoch".format(
                                        int(hours),int(minutes),seconds,
                                        int(hours_epoch),int(minutes_epoch),seconds_epoch))

            if self.epoch % 1 == 0:
                self.evaluate()
        # Save the final model
        torch.save(self.net, self.workspace_dir + '/final_model.pth')

        hours, rem = divmod(self.training_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Training [%s] finished in {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        return self.net

    def train_epoch(self):
        data_time = AverageMeter()
        update_ratio_meter = AverageMeter()
        iter_duration_meter = AverageMeter()
        end = time.time()
        loss_meter = {}
        # AverageMeters for error metrics
        train_err = {}
        for m in err_metrics: train_err[m] = AverageMeter()
        for s in self.sets: loss_meter[s] = AverageMeter()

        for s in self.sets:
            if s == 'train':
                # Iterate over data.
                for i, data in enumerate(self.dataloaders[s]):
                    start_iter_time = time.time()
                    data_time.update(time.time() - end)

                    inputs_d, C, labels, inputs_rgb, img_name = data
                    inputs_d = inputs_d.to(self.device)
                    C = C.to(self.device)
                    labels = labels.to(self.device)
                    inputs_rgb = inputs_rgb.to(self.device)
                    
                    outputs = self.net(inputs_d, inputs_rgb, self.params.weights_layers)
                    # Calculate loss for valid pixel in the ground truth
                    loss11 = self.objective(outputs[0], labels)
                    loss12 = self.objective(outputs[1], labels)
                    loss14 = self.objective(outputs[2], labels)

                    if self.epoch < 6:
                        loss = loss14 + loss12 + loss11

                    elif self.epoch < 11:
                        loss = 0.1 * loss14 + 0.1 * loss12 + loss11
                    else:
                        loss = loss11

                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # statistics
                    loss_meter[s].update(loss11.item(), inputs_d.size(0))
                    if len(outputs) > 1:
                        output = outputs[0]
                    else:
                        output = outputs
                        
                    train_err = self.calculate_metrics(output, labels, inputs_d, train_err)

                    end_iter_time = time.time()
                    iter_duration = end_iter_time - start_iter_time
                    iter_duration_meter.update(iter_duration)
                    if self.print_time_iter:
                        print('finish the iteration in %.2f s.\n' % (
                            iter_duration))
                        print('Loss within the curt iter: {:.8f}\n'.format(loss_meter[s].avg))

                    # Code from Bonnetal
                    lr=self.optimizer.param_groups[0]["lr"]
                    update_ratios = []
                    for _, value in self.net.named_parameters():
                      if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(-lr * value.grad.cpu().numpy().reshape((-1)))
                        # End training if update values are too high
                        with np.errstate(all='raise', divide='ignore'):
                          try:
                            up = update/w
                            if up == np.Inf: ## Check this later why its going to infinite
                              up = 100
                            else:
                              pass
                            update_ratios.append(up)
                          except FloatingPointError:
                            print (e)
                            sys.exit(1)
                    update_ratios = np.array(update_ratios)
                    update_mean = update_ratios.mean()
                    update_std = update_ratios.std()
                    update_ratio_meter.update(update_mean)  # over the epoch

                    # Print all important values per iteration
                    if i % 5 == 0:
                        print('{setname}, Lr: {lr:.3e} | '
                              'Update: {umean:.3e} mean, {ustd:.3e} std | '
                              'Epoch: [{0}][{1}/{2}]  \n'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | '
                              'Data {data_time.val:.3f} ({data_time.avg:.3f}) | '
                              'Loss {loss.val:.4f} ({loss.avg:.4f})  '
                              .format(self.epoch, i, len(self.dataloaders[s]),
                               setname = s, lr=lr, 
                               umean=update_mean, ustd=update_std,
                               batch_time=iter_duration_meter, data_time=data_time,
                               loss = loss_meter[s]))
                    if self.useWandb:
                        for i, m in enumerate(err_metrics): wandb.log({
                            ('{}_{}'.format(s, err_metrics_name[i])) : (train_err[m].avg),
                            'epochs': self.epoch})
                        

                        wandb.log({'train_loss': loss_meter['train'].avg,
                                   'epochs' : self.epoch,
                            })

                torch.cuda.empty_cache()

        return loss_meter, train_err

    ####### Evaluation Function #######

    def evaluate(self):
        self.header("Evaluation Mode")
        # Load last save checkpoint

        if self.use_load_checkpoint != None:
            if isinstance(self.use_load_checkpoint, int):
                if self.use_load_checkpoint > 0:
                    print('=> Loading checkpoint {} ...'.format(self.use_load_checkpoint))
                    if self.load_checkpoint(self.use_load_checkpoint):
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
                elif self.use_load_checkpoint == -1:
                    print('=> Loading last checkpoint ...')
                    if self.load_checkpoint():
                        print('Checkpoint was loaded successfully!\n')
                    else:
                        print('Evaluating using initial parameters')
            elif isinstance(self.use_load_checkpoint, str):
                print('Loading checkpoint from : ' + self.use_load_checkpoint)
                if self.load_checkpoint(self.use_load_checkpoint):
                    # print(self.load_checkpoint)
                    print('Checkpoint was loaded successfully!\n')
                else:
                    print('Evaluating using initial parameters')

        self.net.train(False)

        # AverageMeters for Loss
        loss_meter = {}
        for s in self.sets: loss_meter[s] = AverageMeter()

        # AverageMeters for error metrics
        err = {}
        for m in err_metrics: err[m] = AverageMeter()
        # AverageMeters for time
        times = AverageMeter()
        imgs_wandb = []
        with torch.no_grad():
            for s in self.sets:
                if s == 'val' or s == 'test':

                    print('Evaluating on [{}] set, Epoch [{}] ! \n'.format(s, str(self.epoch)))
                    # Iterate over data.
                    count_save = 0

                    Start_time = time.time()
                    for data in self.dataloaders[s]:

                        torch.cuda.synchronize()
                        start_time = time.time()
                        inputs_d, C, labels, inputs_rgb, name_image = data

                        inputs_d = inputs_d.to(self.device)
                        C = C.to(self.device)
                        labels = labels.to(self.device)
                        inputs_rgb = inputs_rgb.to(self.device)

                        outputs = self.net(inputs_d, inputs_rgb, self.params.weights_layers)

                        if len(outputs) > 1:
                            outputs = outputs[0]

                        torch.cuda.synchronize()
                        duration = time.time() - start_time
                        times.update(duration / inputs_d.size(0), inputs_d.size(0))


                        # Calculate loss for valid pixel in the ground truth
                        loss = self.objective(outputs, labels, self.epoch)
                        # statistics
                        loss_meter[s].update(loss.item(), inputs_d.size(0))
                        if self.debug:
                            torch.autograd.detect_anomaly()

                    
                        val_err = self.calculate_metrics(outputs, labels, inputs_d, err)
                        if count_save < 10:
                            random_imgs = self.save_images.do(outputs, 
                                                              inputs_d, 
                                                              inputs_rgb, 
                                                              name_image, 
                                                              labels)
                        if count_save < 1 and self.useWandb:
                            for imgs in random_imgs:
                                imgs_wandb.append(imgs)
                            

                        if self.useWandb:
                            for i, m in enumerate(err_metrics): wandb.log({
                                ('{}_{}'.format(s, err_metrics_name[i])): (val_err[m].avg),
                                'epochs': self.epoch})
                            wandb.log({(s + '_loss'): loss_meter[s].avg,
                                       'epochs' : self.epoch})

                        count_save += 1


      
                    average_time = (time.time() - Start_time) / len(self.dataloaders[s].dataset)

                    print('Evaluation results on [{}]:\n============================='.format(s))
                    print('[{}]: {:.8f}'.format('Loss', loss_meter['val'].avg))
                    for i, m in enumerate(err_metrics): print('[{}]: {:.4f}'.format(err_metrics_name[i], val_err[m].avg))
                    print('[{}]: {:.4f}'.format('Time', times.avg))
                    print('[{}]: {:.4f}'.format('Time_av', average_time))

                    self.log_txt(val_err, loss_meter, times, s)
                    
                    torch.cuda.empty_cache()

        if self.useWandb:
            imgs_wandb = np.array(imgs_wandb, dtype=object)
            wandb.log({'Ground Truth': (imgs_wandb[:,0]).tolist(), 
                        'Input Depth' : imgs_wandb[:,2].tolist(), 
                        'Predictions': imgs_wandb[:,1].tolist(),
                        'Input RGB': imgs_wandb[:,3].tolist(),
                        'epochs':self.epoch})



    def calculate_metrics(self, outputs, labels, inputs_d, err):
    # Convert data to depth in meters before error metrics
        outputs[outputs == 0] = -1
        if not self.load_rgb:
            outputs[outputs == outputs[0, 0, 0, 0]] = -1
        labels[labels == 0] = -1
        if self.params.invert_depth:
            outputs = 1 / outputs
            labels = 1 / labels
        outputs[outputs == -1] = 0
        labels[labels == -1] = 0

        # Calculate error metrics
        for m in err_metrics:
            if m.find('Delta') >= 0:
                fn = globals()['Deltas']()
                error = fn(outputs, labels)

                if error < 100000.0:
                    err['Delta1'].update(error[0], inputs_d.size(0))
                    err['Delta2'].update(error[1], inputs_d.size(0))
                    err['Delta3'].update(error[2], inputs_d.size(0))
                break
            else:

                fn = eval(m)  # globals()[m]()
                error = fn(outputs, labels)
                if error < 100000.0:
                    err[m].update(error.item(), inputs_d.size(0))

        return err

    def header(self, name):
        print("*" * 60)
        string_length = len(name)
        space = round((60-string_length)/2)
        print(" " * space, name)
        print("*" * 60)

    def log_txt(self, val_err, loss_meter, times, s):
        # Save evaluation metric to text file
        with open(os.path.join(self.workspace_dir, 'results.txt'), 'a+') as text_file:
            text_file.write(
                '\nEvaluation results on [{}], Epoch [{}]:\n==========================================\n'.format(
                    s, self.epoch))
            text_file.write('[{}]: {:.8f}\n'.format('RGB Weight', self.params.weights_layers[0]))
            text_file.write('[{}]: {:.8f}\n'.format('Depth Weight', self.params.weights_layers[1]))

            text_file.write('[{}]: {:.8f}\n'.format('Validation_Loss', loss_meter[s].avg))
            for i, m in enumerate(err_metrics): text_file.write('[{}]: {:.4f}\n'.format(err_metrics_name[i], val_err[m].avg))
            text_file.write('[{}]: {:.4f}\n'.format('Time', times.avg))