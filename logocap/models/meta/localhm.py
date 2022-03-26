from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import pdb
from time import time
from . import META


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import BasicBlock, Bottleneck

from logocap.dataset.transforms import FLIP_CONFIG

import matplotlib.pyplot as plt


@META.register()
class LocalCAPose(nn.Module):
    def __init__(self, cfg, backbone):
        super(LocalCAPose, self).__init__()
        self.backbone = backbone
        self.num_joints = 17
        self.flip_test = cfg.TEST.FLIP_TEST
        self.flip_config = FLIP_CONFIG[cfg.MODEL.TEMPLATE.upper()]

        # self.joints_mlp = 
        self.attn_norm_affine_num = cfg.MODEL.EX['MODEL']['EXTRA']['AN_NUM_AFFINE']

        

    def forward(self, images_batch, targets_batch):
        if self.training:
            return self.forward_training(images_batch, targets_batch)
        else:
            return self.inference(images_batch, targets_batch) 

    