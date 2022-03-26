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

class CocoTestDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, dataset, data_format,
                 input_size,
                 remove_images_without_annotations = False,
                 start_img_index = 0,
                 end_img_index = 0):
        from pycocotools.coco import COCO
        self.root = root
        self.dataset = dataset
        self.data_format = data_format
        self.coco = COCO(self._get_anno_file_name())
        if remove_images_without_annotations:
            self.ids = self.coco.getImgIds(catIds=[1])
            self.filter_for_keypoint_annotations()
        else:
            self.ids = self.coco.getImgIds()

        if end_img_index-start_img_index>0:
            self.ids = self.ids[start_img_index:end_img_index]
        self.input_size = input_size
        self.transforms = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToTensor(),
            ToTensor(),
            # torchvision.transforms.Normalize(
                # mean=[0.485, 0.456, 0.406],
                # std=[0.229, 0.224, 0.225]
            # )
            Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
        )


    def filter_for_keypoint_annotations(self):
        logger.info('filter for keypoint annotations ...')
        def has_keypoint_annotations(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id,
                                          catIds=[1])

            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids if has_keypoint_annotations(image_id)]
        logger.info('... done.')

    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        # image_info_test-dev2017.json
        if 'test' in self.dataset:
            return os.path.join(
                self.root,
                'annotations',
                'image_info_{}.json'.format(
                    self.dataset
                )
            )
        else:
            return os.path.join(
                self.root,
                'annotations',
                'person_keypoints_{}.json'.format(
                    self.dataset
                )
            )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        dataset = 'test2017' if 'test' in self.dataset else self.dataset
        if self.data_format == 'zip':
            return os.path.join(images_dir, dataset) + '.zip@' + file_name
        else:
            return os.path.join(images_dir, dataset, file_name)
    def get_image_info(self, index):
        img_id = self.ids[index]
        return self.coco.loadImgs(img_id)[0]
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image_for_network, raw_image, meta). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        meta = coco.loadImgs(img_id)[0]
        file_name = meta['file_name']
        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        size_resized, center, scale = get_multi_scale_size(img, self.input_size, 1.0, 1.0)
        meta['center'] = center
        meta['scale'] = scale
        trans = get_affine_transform(center, scale, 0, size_resized)
        image_resized = cv2.warpAffine(img, trans, size_resized)

        trans = np.concatenate((trans,np.array([0,0,1]).reshape(1,3)), axis=0)

        kpts_annotations = []
        area_annotations = []
        for ann in target:
            if ann['num_keypoints'] == 0 or ann['iscrowd'] > 0:
                continue
            keypoints = torch.tensor(ann['keypoints'],dtype=torch.float32).reshape(-1,3)

            kx = keypoints[:,0]*trans[0,0] + keypoints[:,1]*trans[0,1] + trans[0,2]
            ky = keypoints[:,0]*trans[1,0] + keypoints[:,1]*trans[1,1] + trans[1,2]
            # keypoints[:,2] *= kv
            keypoints[:,0] = kx
            keypoints[:,1] = ky
            kpts_annotations.append(keypoints)
            area_annotations.append(ann['area'])
        if len(kpts_annotations)> 0:
            kpts_annotations = torch.stack(kpts_annotations)
            area_annotations = torch.tensor(area_annotations).float()

            meta['annotations'] = {'kpts':kpts_annotations,'areas':area_annotations}
        meta['transform'] = torch.from_numpy(trans).float()
        meta['transform_inv'] = torch.inverse(meta['transform'])

        image_resized = self.transforms(image_resized)

        return image_resized, torch.from_numpy(img), meta

    def __len__(self):
        return len(self.ids)

class CocoMSTestDataset(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where dataset is located to.
        dataset (string): Dataset name(train2017, val2017, test2017).
        data_format(string): Data format for reading('jpg', 'zip')
        transform (callable, optional): A function/transform that  takes in an opencv image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, dataset, data_format,
                 input_size,
                 scales = [0.5, 1.0, 2.0],
                 remove_images_without_annotations = False,
                 start_img_index = 0,
                 end_img_index = 0):
        from pycocotools.coco import COCO
        self.root = root
        self.dataset = dataset
        self.data_format = data_format
        self.coco = COCO(self._get_anno_file_name())
        self.scales = sorted(scales)[::-1]
        if remove_images_without_annotations:
            self.ids = self.coco.getImgIds(catIds=[1])
            self.filter_for_keypoint_annotations()
        else:
            self.ids = self.coco.getImgIds()

        if end_img_index-start_img_index>0:
            self.ids = self.ids[start_img_index:end_img_index]
        self.input_size = input_size
        self.transforms = torchvision.transforms.Compose(
            [
                # torchvision.transforms.ToTensor(),
                ToTensor(),
                # torchvision.transforms.Normalize(
                    # mean=[0.485, 0.456, 0.406],
                    # std=[0.229, 0.224, 0.225]
                # )
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )


    def filter_for_keypoint_annotations(self):
        logger.info('filter for keypoint annotations ...')
        def has_keypoint_annotations(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id,
                                          catIds=[1])

            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids if has_keypoint_annotations(image_id)]
        logger.info('... done.')

    def _get_anno_file_name(self):
        # example: root/annotations/person_keypoints_tran2017.json
        # image_info_test-dev2017.json
        if 'test' in self.dataset:
            return os.path.join(
                self.root,
                'annotations',
                'image_info_{}.json'.format(
                    self.dataset
                )
            )
        else:
            return os.path.join(
                self.root,
                'annotations',
                'person_keypoints_{}.json'.format(
                    self.dataset
                )
            )

    def _get_image_path(self, file_name):
        images_dir = os.path.join(self.root, 'images')
        dataset = 'test2017' if 'test' in self.dataset else self.dataset
        if self.data_format == 'zip':
            return os.path.join(images_dir, dataset) + '.zip@' + file_name
        else:
            return os.path.join(images_dir, dataset, file_name)
    def get_image_info(self, index):
        img_id = self.ids[index]
        return self.coco.loadImgs(img_id)[0]
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image_for_network, raw_image, meta). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        meta = coco.loadImgs(img_id)[0]
        file_name = meta['file_name']
        if self.data_format == 'zip':
            img = zipreader.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
        else:
            img = cv2.imread(
                self._get_image_path(file_name),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        

        centers = []
        scales = []
        msimages = []
        for scale_factor in self.scales:
            size_resized, center, scale = get_multi_scale_size(img, self.input_size, scale_factor, 1.0)
            trans = get_affine_transform(center, scale, 0, size_resized)
            image_resized = cv2.warpAffine(img, trans, size_resized)#flags=cv2.INTER_AREA)
            image_resized = self.transforms(image_resized)
            centers.append(center)
            scales.append(scale)
            msimages.append(image_resized)

        meta['center'] = centers[0]
        meta['scale'] = scales[0]
        meta['scale_factors'] = self.scales
        

        return msimages, torch.from_numpy(img), meta

    def __len__(self):
        return len(self.ids)