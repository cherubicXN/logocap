from __future__ import print_function

import logging
import logocap
import os
import io
import PIL
import time
import torch
import torchvision
from logocap.utils.utils import AverageMeter
import wandb
class MultiTaskLossMeter():
    def __init__(self, loss_names):
        assert isinstance(loss_names,tuple) or isinstance(loss_names,list)
        self.loss_names = loss_names
        self.num_loss_items = len(loss_names)
        self.loss_meters = [AverageMeter() for i in range(self.num_loss_items)]

    def update(self, loss_dict, n=1):
        for i, name in enumerate(self.loss_names):
            self.loss_meters[i].update(loss_dict[name],n)

    def __str__(self):
        msg = ''
        for name,meter in zip(self.loss_names, self.loss_meters):
            msg += '{name}: {meter.val:.3e} ({meter.avg:.3e})\t'.format(name=name, meter=meter)
        return msg



def do_train(cfg, model, data_loader, targets_encoder, optimizer, lr_scheduler, epoch, output_dir, writer_dict, logger, use_wandb = False):
    # LOG = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()

    loss_meters = MultiTaskLossMeter(['loss.oks','loss.heatmaps','loss.offsets','loss.hm'])
    model.train()

    end = time.time()

    for i, (images_batch, targets_batch, joints_batch, areas_batch, meta_batch) in enumerate(data_loader):
        data_time.update(time.time()-end)
        images_batch, targets_batch =\
            targets_encoder(images_batch, targets_batch, joints_batch, areas_batch,meta_batch)
        outputs_dict, loss_dict = model(images_batch, targets_batch)
        loss_meters.update(loss_dict,images_batch.size(0))

        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if cfg.RANK==0:
        if logocap.utils.comm.get_rank() == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed: {speed:.1f} samples/s\t' \
                  'Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'LR: {lr}\t' \
                  '{loss_msg}'.format(
                epoch, i, len(data_loader),
                batch_time=batch_time,
                speed=images_batch.size(0) / batch_time.val,
                data_time=data_time,
                lr = optimizer.param_groups[0]['lr'],
                loss_msg = str(loss_meters),
            )

            global_steps = writer_dict['train_global_steps']

            if i % cfg.PRINT_FREQ == 0 :
                logger.info(msg)
            
            if i % cfg.PRINT_FREQ == 0 and use_wandb:
                logdict = {
                        'epoch': epoch,
                        'iter': i,
                        'batch_time': batch_time.avg,
                        'data_time': data_time.avg,
                        'lr': optimizer.param_groups[0]['lr'],   
                    }

                for loss_name, loss_meter in zip(loss_meters.loss_names,loss_meters.loss_meters):
                    logdict[loss_name] = loss_meter.avg
                wandb.log(
                    logdict,
                    step = global_steps
                )

            writer_dict['train_global_steps'] = global_steps + 1


