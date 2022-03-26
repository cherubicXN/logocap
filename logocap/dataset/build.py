# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL.Image import Image

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from .COCODataset import CocoDataset as coco
from .COCOTest import CocoTestDataset as coco_test
from .COCOTest import CocoMSTestDataset as coco_mstest

from .COCOKeypoints import CocoKeypoints as coco_kpt
from .COCOKeypointsDebug import CocoKeypoints as coco_kpt_debug
from .transforms import build_transforms, build_debug_transforms
from .target_generators import HeatmapGenerator
from .target_generators import OffsetGenerator
from .ImageList import ImageListDataset
import numpy as np
import random

def collate_fn_train(batch):
    images_batch = default_collate([b[0] for b in batch])
    targets_batch = {
        'heatmaps.t': default_collate([b[1]['heatmaps.t'] for b in batch]),
        'heatmaps.w': default_collate([b[1]['heatmaps.w'] for b in batch]),
        'offsets.t': default_collate([b[1]['offsets.t'] for b in batch]),
        'offsets.w': default_collate([b[1]['offsets.w'] for b in batch]),
    }
    joints_batch = [torch.from_numpy(b[2]).float() for b in batch]
    areas_batch = [torch.from_numpy(b[3]).float() for b in batch]
    meta_batch = [b[4] for b in batch]
    return images_batch, targets_batch, joints_batch, areas_batch, meta_batch


def collate_fn_test(batch):
    images_batch = default_collate([b[0] for b in batch])
    imgraw_batch = default_collate([b[1] for b in batch])
    meta_batch = [b[2] for b in batch]

    return images_batch, imgraw_batch, meta_batch

def collate_fn_mstest(batch):
    assert len(batch) == 1
    # images_batch = default_collate([b[0] for b in batch])
    # images_batch = [default_collate(img) for b in batch for img in b[0] ]
    num_scales = len(batch[0][0])
    images_batch = [default_collate([b]) for b in batch[0][0]]
    
    imgraw_batch = default_collate([b[1] for b in batch])
    meta_batch = [b[2] for b in batch]

    return images_batch, imgraw_batch, meta_batch

def build_dataset(cfg, is_train):
    transforms = build_transforms(cfg, is_train)

    _HeatmapGenerator = HeatmapGenerator

    heatmap_generator = [
        _HeatmapGenerator(
            output_size, cfg.DATASET.NUM_JOINTS
        ) for output_size in cfg.DATASET.OUTPUT_SIZE
    ]

    offset_generator = None
    if cfg.DATASET.OFFSET_REG:
        offset_generator = [
            OffsetGenerator(
                output_size,
                output_size,
                cfg.DATASET.NUM_JOINTS,
                cfg.DATASET.OFFSET_RADIUS
            ) for output_size in cfg.DATASET.OUTPUT_SIZE
        ]

    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST

    dataset = eval(cfg.DATASET.DATASET)(
        cfg,
        dataset_name,
        is_train,
        heatmap_generator,
        offset_generator,
        transforms
    )

    return dataset

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataloader(cfg, is_train=True, distributed=False):
    if is_train:
        images_per_gpu = cfg.TRAIN.IMAGES_PER_GPU
        shuffle = True
    else:
        images_per_gpu = cfg.TEST.IMAGES_PER_GPU
        shuffle = False
    images_per_batch = images_per_gpu * len(cfg.GPUS)

    dataset = build_dataset(cfg, is_train)

    if is_train and distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset
        )
        shuffle = False
    else:
        train_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=images_per_batch,
        shuffle=shuffle,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        sampler=train_sampler,
        collate_fn = collate_fn_train,
        worker_init_fn=seed_worker,
    )

    return data_loader

def make_debugloader(cfg, is_train = False, remove_images_without_annotations=True,
                         start_img_index = 0,
                         end_img_index = -1):
    
    transforms = build_debug_transforms(cfg)

    heatmap_generator = [
        HeatmapGeneratorNS(cfg.DATASET.NUM_JOINTS) for _ in cfg.DATASET.OUTPUT_SIZE
    ]
    offset_generator = [
        OffsetGeneratorNS(cfg.DATASET.NUM_JOINTS,cfg.DATASET.OFFSET_RADIUS) for _ in cfg.DATASET.OUTPUT_SIZE
    ]
    dataset_name = cfg.DATASET.TRAIN if is_train else cfg.DATASET.TEST
    
    dataset = coco_kpt_debug(cfg, dataset_name, remove_images_without_annotations,heatmap_generator,
    start_img_index, end_img_index,
    offset_generator,transforms )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        collate_fn = collate_fn_debug,
    )

    return data_loader, dataset


def make_test_dataloader(cfg, remove_images_without_annotations=False,
                         start_img_index = 0,
                         end_img_index = -1,
                         distributed = False):
    if 'coco' in cfg.DATASET.DATASET_TEST:
        dataset = coco_test(cfg.DATASET.ROOT,
                            cfg.DATASET.TEST,
                            cfg.DATASET.DATA_FORMAT,
                            cfg.DATASET.INPUT_SIZE,
                            remove_images_without_annotations=remove_images_without_annotations,
                            start_img_index=start_img_index,
                            end_img_index=end_img_index
                            )
    
    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        test_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        sampler = test_sampler,
        collate_fn = collate_fn_test,
    )

    return data_loader, dataset

def make_mstest_dataloader(cfg, remove_images_without_annotations=False,
                         start_img_index = 0,
                         end_img_index = -1,
                         distributed = False):
    if 'coco' in cfg.DATASET.DATASET_TEST:
        dataset = coco_mstest(cfg.DATASET.ROOT,
                            cfg.DATASET.TEST,
                            cfg.DATASET.DATA_FORMAT,
                            cfg.DATASET.INPUT_SIZE,
                            scales = cfg.TEST.SCALE_FACTOR,
                            remove_images_without_annotations=remove_images_without_annotations,
                            start_img_index=start_img_index,
                            end_img_index=end_img_index
                            )

    if distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        test_sampler = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
        sampler = test_sampler,
        collate_fn = collate_fn_mstest,
    )

    return data_loader, dataset

def make_imageloader(cfg, root, ext='jpg', res = None):
    if res is None:
        res = cfg.DATASET.INPUT_SIZE
    dataset = ImageListDataset(root,
                               res,
                               ext = ext)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers = cfg.WORKERS,
        pin_memory = True,
        collate_fn = collate_fn_test,
    )                    

    return data_loader    