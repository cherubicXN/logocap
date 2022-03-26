# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn).
# Modified by Zigang Geng (aa397601@mail.ustc.edu.cn).
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.DATA_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0
_C.VERBOSE = True
_C.DIST_BACKEND = 'nccl'
_C.MULTIPROCESSING_DISTRIBUTED = True
_C.AMP_OPT_LEVEL = 'O0'

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.TEMPLATE = 'coco' #coco, or crowdpose
_C.MODEL.FINE_TUNE = False
_C.MODEL.NAME = 'pose_hrnet'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = ''
_C.MODEL.NUM_JOINTS = 17
_C.MODEL.TYPE = 'coco'
_C.MODEL.EXTRA = CN(new_allowed=True)
_C.MODEL.DECODER = CN()
_C.MODEL.DECODER.LOCAL_HM_SIZE = 11
_C.MODEL.DECODER.GLOBAL_HM_SIZE = 96
_C.MODEL.DECODER.SIGMA = 16
_C.MODEL.DECODER.TOPK_CENTER  = 30
_C.MODEL.DECODER.KSIZE = 5

# _C.DECODER = CN(new_allowed=True)
# _C.DECODER.KSIZE = 5


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'coco_kpt'
_C.DATASET.DATASET_TEST = ''
_C.DATASET.NUM_JOINTS = 17
_C.DATASET.MAX_NUM_PEOPLE = 30
_C.DATASET.TRAIN = 'train2017'
_C.DATASET.TEST = 'val2017'
_C.DATASET.GET_RESCORE_DATA = False
_C.DATASET.DATA_FORMAT = 'jpg'
_C.DATASET.USE_MASK = False
_C.DATASET.USE_BBOX_CENTER = False
_C.DATASET.OFFSET_REG = False
_C.DATASET.OFFSET_RADIUS = 4
_C.DATASET.BG_WEIGHT = [1.0]

# training data augmentation
_C.DATASET.MAX_ROTATION = 30
_C.DATASET.MIN_SCALE = 0.75
_C.DATASET.MAX_SCALE = 1.25
_C.DATASET.SCALE_TYPE = 'short'
_C.DATASET.MAX_TRANSLATE = 40
_C.DATASET.INPUT_SIZE = 512
_C.DATASET.OUTPUT_SIZE = [128, 256, 512]
_C.DATASET.FLIP = 0.5

# heatmap generator (default is OUTPUT_SIZE/64)
_C.DATASET.SIGMA = [2.0,]
_C.DATASET.CENTER_SIGMA = 4
_C.DATASET.BASE_SIZE = 256.0
_C.DATASET.BASE_SIGMA = 2.0
_C.DATASET.MIN_SIGMA = 1
_C.DATASET.WITH_CENTER = False

# train
_C.TRAIN = CN()

_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = [90, 110]
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_DECAY_EPOCHS = 0

_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0001
_C.TRAIN.NESTEROV = False
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.WARMUP_START_EPOCH = 0
_C.TRAIN.WARMUP_EPOCHS = 1
_C.TRAIN.WARMUP_FACTOR = 0.001
_C.TRAIN.WARMUP_RESTART_SCHEDULE = []
_C.TRAIN.WARMUP_RESTART_DURATION = 0.5

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140

_C.TRAIN.RESUME = False
_C.TRAIN.CHECKPOINT = ''

_C.TRAIN.IMAGES_PER_GPU = 32
_C.TRAIN.SHUFFLE = True


_C.LOSS_FACTORS = CN(new_allowed=True)
# testing
_C.TEST = CN()

# size of images for each device
_C.TEST.IMAGES_PER_GPU = 1
_C.TEST.INSTANCE_NMS = CN()
_C.TEST.INSTANCE_NMS.ACTIVATED = True
_C.TEST.INSTANCE_NMS.SCORE_THRESHOLD = 0.05
_C.TEST.INSTANCE_NMS.NUM_JOINTS = 8
_C.TEST.INSTANCE_NMS.DECREASE = 0.8
_C.TEST.DECREASE = 0.8
_C.TEST.NMS_THRE = 0.15
_C.TEST.NMS_NUM_THRE = 10
_C.TEST.FLIP_TEST = True
_C.TEST.SCALE_FACTOR = [1]
# group
_C.TEST.MODEL_FILE = ''
_C.TEST.ADAPTATION_LEVEL = 'full'
_C.TEST.LOG_PROGRESS = True


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    if hasattr(args,'opts'):
        cfg.merge_from_list(args.opts)

    if not os.path.exists(cfg.DATASET.ROOT):
        cfg.DATASET.ROOT = os.path.join(
            cfg.DATA_DIR, cfg.DATASET.ROOT
        )

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    if cfg.DATASET.WITH_CENTER:
        cfg.DATASET.NUM_JOINTS += 1
        cfg.MODEL.NUM_JOINTS = cfg.DATASET.NUM_JOINTS

    if not isinstance(cfg.DATASET.OUTPUT_SIZE, (list, tuple)):
        cfg.DATASET.OUTPUT_SIZE = [cfg.DATASET.OUTPUT_SIZE]


    cfg.freeze()


def check_config(cfg):
    return

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
