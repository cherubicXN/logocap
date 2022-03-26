# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import logging
from collections import namedtuple

import torch
import torch.optim as optim
import torch.nn as nn
import sys
import logocap

class LearningRateLambda():
    # This code is from [Openpifpaf](http://openpifpaf.github.io/)
    def __init__(self, decay_schedule, *,
                 decay_factor=0.1,
                 decay_epochs=1.0,
                 warm_up_start_epoch=0,
                 warm_up_epochs=2.0,
                 warm_up_factor=0.01,
                 warm_restart_schedule=[],
                 warm_restart_duration=0.5):
        self.decay_schedule = decay_schedule
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.warm_up_start_epoch = warm_up_start_epoch
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_factor = warm_up_factor
        self.warm_restart_schedule = warm_restart_schedule
        self.warm_restart_duration = warm_restart_duration

    def __call__(self, step_i):
        lambda_ = 1.0
        if step_i <= self.warm_up_start_epoch:
            lambda_ *= self.warm_up_factor
        elif self.warm_up_start_epoch < step_i < self.warm_up_start_epoch + self.warm_up_epochs:
            lambda_ *= self.warm_up_factor**(
                1.0 - (step_i - self.warm_up_start_epoch) / self.warm_up_epochs
            )

        for d in self.decay_schedule:
            if step_i >= d + self.decay_epochs:
                # import pdb; pdb.set_trace()
                lambda_ *= self.decay_factor
            elif step_i > d:
                lambda_ *= self.decay_factor**(
                    (step_i - d) / self.decay_epochs
                )

        for r in self.warm_restart_schedule:
            if r <= step_i < r + self.warm_restart_duration:
                lambda_ = lambda_**(
                    (step_i - r) / self.warm_restart_duration
                )

        return lambda_

def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)

    if logocap.utils.comm.get_world_size() == 1:
        logger.addHandler(ch)
    # logger.info('hahha')

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def create_logger(cfg, cfg_name, name=None, filename=None):
    root_output_dir = cfg.OUTPUT_DIR
    basename = os.path.basename(cfg_name).split('.')[0]
    final_output_dir = os.path.join(root_output_dir,basename)
    # os.makedirs(final_output_dir,exist_ok=True)
    # logger = setup_logger(name,final_output_dir,0,filename)
    
    
    # tensorboard_log_dir = None

    return final_output_dir


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer

def get_lrscheduler(cfg, optimizer, epochsize, last_epoch=-1):
    TRAIN_CFG = cfg.TRAIN
    decay_schedule = [s*epochsize for s in TRAIN_CFG.LR_STEP]
    decay_factor = TRAIN_CFG.LR_FACTOR
    decay_epochs = TRAIN_CFG.LR_DECAY_EPOCHS*epochsize
    warm_up_start_epoch = TRAIN_CFG.WARMUP_START_EPOCH*epochsize
    warm_up_epochs = TRAIN_CFG.WARMUP_EPOCHS*epochsize
    warm_restart_schedule = [s*epochsize for s in TRAIN_CFG.WARMUP_RESTART_SCHEDULE]
    warm_restart_duration = TRAIN_CFG.WARMUP_RESTART_DURATION*epochsize
    lr_lambda = LearningRateLambda(decay_schedule,
                decay_factor=decay_factor,
                decay_epochs=decay_epochs,
                warm_up_start_epoch=warm_up_start_epoch,
                warm_up_epochs=warm_up_epochs,
                warm_restart_schedule=warm_restart_schedule,
                warm_restart_duration=warm_restart_duration,
                )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda,
    last_epoch=last_epoch
    )
    return lr_scheduler

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    #print(states, is_best)
    if is_best and 'state_dict' in states:
        torch.save(
            states['best_state_dict'],
            os.path.join(output_dir, 'model_best.pth.tar')
        )


def get_model_summary(model, *input_tensors, item_length=26, verbose=True):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            try:
                summary.append(
                    ModuleDetails(
                        name=layer_name,
                        input_size=list(input[0].size()),
                        output_size=list(output.size()),
                        num_parameters=params,
                        multiply_adds=flops)
                )
            except:
                if params != 0:
                    raise ValueError(layer_name+' meets some problems and you should check it.')

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,}".format(flops_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
