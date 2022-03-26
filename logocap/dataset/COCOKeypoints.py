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

import logging

import numpy as np

import pycocotools
from .COCODataset import CocoDataset
from .target_generators import HeatmapGenerator
import copy
import torch
logger = logging.getLogger(__name__)

from .transforms.build import FLIP_CONFIG

class CocoKeypoints(CocoDataset):
    def __init__(self,
                 cfg,
                 dataset_name,
                 remove_images_without_annotations,
                 heatmap_generator,
                 offset_generator=None,
                 transforms=None):
        super(CocoKeypoints,self).__init__(cfg.DATASET.ROOT,
                         dataset_name,
                         cfg.DATASET.DATA_FORMAT,
                         cfg.DATASET.NUM_JOINTS,
                         cfg.DATASET.GET_RESCORE_DATA)

        if cfg.DATASET.WITH_CENTER:
            assert cfg.DATASET.NUM_JOINTS == 18, 'Number of joint with center for COCO is 18'
        else:
            raise NotImplementedError()
            # assert cfg.DATASET.NUM_JOINTS == 17, 'Number of joint for COCO is 17'

        self.num_scales = self._init_check(heatmap_generator)
        self.num_joints = cfg.DATASET.NUM_JOINTS
        self.with_center = cfg.DATASET.WITH_CENTER
        self.num_joints_without_center = self.num_joints - 1 \
            if self.with_center else self.num_joints
        self.base_sigma = cfg.DATASET.BASE_SIGMA
        self.base_size = cfg.DATASET.BASE_SIZE
        self.min_sigma = cfg.DATASET.MIN_SIGMA
        self.center_sigma = cfg.DATASET.CENTER_SIGMA
        self.sigma = cfg.DATASET.SIGMA
        self.bg_weight = cfg.DATASET.BG_WEIGHT

        self.use_mask = cfg.DATASET.USE_MASK
        self.use_bbox_center = cfg.DATASET.USE_BBOX_CENTER

        if remove_images_without_annotations:
            self.ids = [
                img_id
                for img_id in self.ids
                if len(self.coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
            ]
        # self.ids = self.ids[12:24]
        # self.ids = self.ids[:24]

        self.transforms = transforms
        self.heatmap_generator = heatmap_generator
        self.offset_generator = offset_generator

    def __getitem__(self, idx):
        img, anno = super(CocoKeypoints,self).__getitem__(idx)
        img_raw = copy.deepcopy(img)
        im_h, im_w = img.shape[:2]
        mask = self.get_mask(anno, idx)

        anno = [
            obj for obj in anno
            if obj['iscrowd'] == 0 or obj['num_keypoints'] > 0
        ]

        joints, area = self.get_joints(anno)

        mask_list = [mask.copy() for _ in range(self.num_scales)]
        joints_list = [joints.copy() for _ in range(self.num_scales)]

        meta = {}
        if self.transforms:
            img, mask_list, joints_list, area, meta = self.transforms(
                img, mask_list, joints_list, area, meta
            )

        # if meta['HFlip']:
            # img_raw = img_raw[:,::-1]
        
        kpts_annotations = []
        area_annotations = []
        for ann in anno:
            if ann['num_keypoints'] == 0 or ann['iscrowd'] > 0:
                continue
            keypoints = torch.tensor(ann['keypoints'],dtype=torch.float32).reshape(-1,3)

            if meta['HFlip']:
                keypoints = keypoints[FLIP_CONFIG['COCO']]
                # keypoints[:,0] = img_raw[:,:,]
            trans = meta['transform']
            kx = keypoints[:,0]*trans[0,0] + keypoints[:,1]*trans[0,1] + trans[0,2]
            ky = keypoints[:,0]*trans[1,0] + keypoints[:,1]*trans[1,1] + trans[1,2]
            kv = ((kx>=0)*(kx<img.shape[1])*(ky>=0)*(ky<=img.shape[1])).float()
            keypoints[:,2] *= kv
            kpts_annotations.append(keypoints)
            area_annotations.append(ann['area'])
        if kpts_annotations:
            kpts_annotations = torch.stack(kpts_annotations)
            area_annotations = torch.tensor(area_annotations).float()
        else:
            kpts_annotations = torch.zeros((1,17,3),dtype=torch.float)
            area_annotations = torch.zeros((1,),dtype=torch.float)
        meta['annotations'] = {'kpts':kpts_annotations,'areas':area_annotations}
        meta['transform'] = torch.from_numpy(meta['transform']).float()
        meta['transform_inv'] = torch.inverse(meta['transform'])
        
        # area_annotations = torch.stack()
        
        target_t, ignored_t = self.heatmap_generator[0](joints_list[0],self.sigma[0][0],self.center_sigma,self.bg_weight[0][0])
        # import pdb; pdb.set_trace()
        heatmaps_target = target_t.astype(np.float32)
        heatmaps_weight = mask_list[0]*ignored_t
        joints_with_centers = joints_list[0]
        offset_targets, offsets_weight = self.offset_generator[0](joints_with_centers,area)

        target_dict = {
            'heatmaps.t': heatmaps_target,
            'heatmaps.w': heatmaps_weight,
            'offsets.t': offset_targets,
            'offsets.w': offsets_weight,
        }


        return img, target_dict, joints_with_centers, area, meta

    def get_joints(self, anno):
        num_people = len(anno)
        area = np.zeros((num_people, 1))
        joints = np.zeros((num_people, self.num_joints, 3))

        for i, obj in enumerate(anno):
            joints[i, :self.num_joints_without_center, :3] = \
                np.array(obj['keypoints']).reshape([-1, 3])

            if self.use_mask == True:
                area[i, 0] = obj['area']
            else:
                area[i, 0] = obj['bbox'][2]*obj['bbox'][3]

            if self.with_center:
                if obj['area'] < 32**2:
                    joints[i, -1, 2] = 0
                    continue
                bbox = obj['bbox']
                center_x = (2*bbox[0] + bbox[2]) / 2.
                center_y = (2*bbox[1] + bbox[3]) / 2.
                joints_sum = np.sum(joints[i, :-1, :2], axis=0)
                num_vis_joints = len(np.nonzero(joints[i, :-1, 2])[0])
                if self.use_bbox_center or num_vis_joints <= 0:
                    joints[i, -1, 0] = center_x
                    joints[i, -1, 1] = center_y
                else:
                    joints[i, -1, :2] = joints_sum / num_vis_joints
                joints[i, -1, 2] = 1

        return joints, area

    def get_mask(self, anno, idx):
        coco = self.coco
        img_info = coco.loadImgs(self.ids[idx])[0]

        m = np.zeros((img_info['height'], img_info['width']))

        for obj in anno:
            if obj['iscrowd']:
                rle = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                m += pycocotools.mask.decode(rle)
            elif obj['num_keypoints'] == 0:
                rles = pycocotools.mask.frPyObjects(
                    obj['segmentation'], img_info['height'], img_info['width'])
                for rle in rles:
                    m += pycocotools.mask.decode(rle)

        return m < 0.5

    def _init_check(self, heatmap_generator):
        assert isinstance(heatmap_generator, (list, tuple)
                          ), 'heatmap_generator should be a list or tuple'
        return len(heatmap_generator)
