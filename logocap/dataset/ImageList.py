# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os
import os.path

import cv2
import json_tricks as json
import numpy as np
import torchvision
from torch.utils.data import Dataset
import torch

from pycocotools.cocoeval import COCOeval
from logocap.utils import zipreader
from logocap.utils.transforms import resize_align_multi_scale, get_multi_scale_size, get_affine_transform

logger = logging.getLogger(__name__)

class ToTensor(object):
    def __call__(self,image):
        tensor = torch.from_numpy(image).float()/255.0
        tensor = tensor.permute((2,0,1)).contiguous()
        return tensor

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image):
        image[0] = (image[0]-self.mean[0])/self.std[0]
        image[1] = (image[1]-self.mean[1])/self.std[1]
        image[2] = (image[2]-self.mean[2])/self.std[2]
        return image

class ImageListDataset(Dataset):

    def __init__(self, root, 
                 input_size,
                 ext = 'jpg'
                 ):
        from pycocotools.coco import COCO
        self.root = root
        self.input_size = input_size
        self.transforms = torchvision.transforms.Compose(
        [
            ToTensor(),
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        )
        filenames = [f for f in os.listdir(self.root) if f.endswith(ext)]
        self.filenames = filenames


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image_for_network, raw_image, meta). target is the object returned by ``coco.loadAnns``.
        """
        filename = self.filenames[index]
        img = cv2.imread(
                os.path.join(self.root,filename),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        meta = {
            'file_name': filename,
            'width': img.shape[1],
            'height': img.shape[0],
            'id': index,
        }

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size_resized, center, scale = get_multi_scale_size(img, self.input_size, 1.0, 1.0)
        meta['center'] = center
        meta['scale'] = scale
        trans = get_affine_transform(center, scale, 0, size_resized)
        image_resized = cv2.warpAffine(img, trans, size_resized)

        image_resized = self.transforms(image_resized)

        return image_resized, torch.from_numpy(img), meta

    def __len__(self):
        return len(self.filenames)

