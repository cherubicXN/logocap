from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import shutil
import warnings
import json
from tqdm import tqdm
import time

import logocap
from logocap.config import cfg
from logocap.config import update_config
import logocap.dataset as D
from logocap.utils.utils import get_optimizer, get_lrscheduler, save_checkpoint, setup_logger
from logocap.utils import comm
from logocap.trainer import do_train
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import wandb
import random
import numpy as np
try:
    # noinspection PyUnresolvedReferences
    import apex
    from apex import amp
except ImportError:
    amp = None


def parser_args():
    parser = argparse.ArgumentParser(description='Train LOGO-CAP Network')
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--amp',
                        default='O0',
                        type=str)
    parser.add_argument('--syncbn',default=False,
                        action='store_true')
    parser.add_argument('--seed',default=0, type=int, help='Random seed')
    
    # distributed training
    parser.add_argument('--gpu',
                        help='gpu id for multiprocessing training',
                        type=str)
    parser.add_argument('--world-size',
                        default=1,
                        type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--dist-url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')


    parser.add_argument('--wandb',
                        type=str,
                        default = None,
                        help = 'using wandb to record the training')
    args = parser.parse_args()

    return args

def run_test(cfg,infshell, model, test_loader, final_output_dir, logger, use_wandb = False):
    DETECTION_RESULTS = []
    model.eval()
    local_rank = logocap.utils.comm.get_rank()

    if local_rank == 0:
        pbar = tqdm(total=len(test_loader))

    for i, (images, rgb, meta) in enumerate(test_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'
        with torch.no_grad():
            outputs, _ = model(images.cuda())
        
        final_poses, scores = infshell(images, rgb, outputs, meta[0])
        if final_poses is None:
            continue
        for pose_id, (pose, score) in enumerate(zip(final_poses, scores)):
            xmin = pose[:,0].min().item()
            ymin = pose[:,1].min().item()
            xmax = pose[:,0].max().item()
            ymax = pose[:,1].max().item()
            bw = xmax - xmin
            bh = ymax - ymin

            ans = {
                'image_id': meta[0]['id'],
                'category_id': 1,
                'keypoints': pose.reshape(-1).tolist(),
                'score': score.item(),
                'bbox': [xmin,ymin,bw,bh]
                }
            DETECTION_RESULTS.append(ans)
        if local_rank == 0:
            pbar.update(1)

    if local_rank == 0:
        pbar.close()

    if logocap.utils.comm.get_world_size()>1:
        dist.barrier()

    DETECTION_RESULTS_ALL = comm.all_gather(DETECTION_RESULTS)
    DETECTION_RESULTS_ALL = sum(DETECTION_RESULTS_ALL,[])

    if not logocap.utils.comm.is_main_process():
        return 0.0

    results_path_gathered = os.path.join(final_output_dir,'eval_results.json')
    with open(results_path_gathered, 'w') as writer:
        json.dump(DETECTION_RESULTS_ALL,writer)
    
    coco = test_loader.dataset.coco
    coco_eval = coco.loadRes(results_path_gathered)
    if 'coco' in cfg.DATASET.DATASET:
        from pycocotools.cocoeval import COCOeval as COCOEvaluator
    elif 'crowd_pose' in cfg.DATASET.DATASET:
        from crowdposetools.cocoeval import COCOeval as COCOEvaluator
    evaluator = COCOEvaluator(coco, coco_eval, 'keypoints')
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    if 'coco' in cfg.DATASET.DATASET:
        validation_msg = 'AP = {:.3f}| AP50 = {:.3f}| AP75 = {:.3f}| APM = {:.3f}| APL = {:.3f}'.format(evaluator.stats[0],evaluator.stats[1],evaluator.stats[2],evaluator.stats[3],evaluator.stats[4])
    else:
        validation_msg = 'AP = {:.3f}| AP50 = {:.3f}| AP75 = {:.3f}| APE = {:.3f}| APM = {:.3f}| APH = {:.3f}'.format(evaluator.stats[0],evaluator.stats[1],evaluator.stats[2],evaluator.stats[8],evaluator.stats[9], evaluator.stats[10])

    logger.info('validation: {}'.format(validation_msg))
    if use_wandb:
        wandb.log({
            "AP": evaluator.stats[0],
            "AP50": evaluator.stats[1],
            "AP75": evaluator.stats[2],
            "APM": evaluator.stats[3],
            "APL": evaluator.stats[4],
        })
    return float(evaluator.stats[0])
    

def main():
    args = parser_args()
    update_config(cfg, args)

    cfg.defrost()
    cfg.RANK = args.rank
    cfg.freeze()

    final_output_dir = os.path.join(cfg.OUTPUT_DIR,os.path.basename(args.cfg).split('.')[0])

    os.makedirs(final_output_dir,exist_ok=True)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or cfg.MULTIPROCESSING_DISTRIBUTED

    ngpus_per_node = torch.cuda.device_count()
    if cfg.MULTIPROCESSING_DISTRIBUTED:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args, final_output_dir)
        )
    else:
        # Simply call main_worker function
        main_worker(
            ','.join([str(i) for i in cfg.GPUS]),
            ngpus_per_node,
            args,
            final_output_dir,
        )

def main_worker(
            gpu, ngpus_per_node, args, final_output_dir
    ):
    # random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # cudnn related setting
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        
        if cfg.MULTIPROCESSING_DISTRIBUTED:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print('Init process group: dist_url: {}, world_size: {}, rank: {}'.
              format(args.dist_url, args.world_size, args.rank))
        dist.init_process_group(
            backend=cfg.DIST_BACKEND,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
    update_config(cfg, args)

    if args.wandb:
        if logocap.utils.comm.get_rank() == 0:
            wandb.init(project='logocap',entity=args.wandb, config=cfg)
            final_output_dir = wandb.run.dir
        else:
            args.wandb = None
        
        # wandb.config = cfg

    logger = setup_logger('logocap',final_output_dir,logocap.utils.comm.get_rank(),filename='train.log')
    
    model, targets_encoder = logocap.models.build_model(cfg, is_train=True)
    optimizer = get_optimizer(cfg, model)

    writer_dict = {
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            if args.amp != 'O0':
                model, optimizer = amp.initialize(model, optimizer,
                                    opt_level=args.amp)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu],
                # find_unused_parameters=True
            )
            if args.syncbn:
                model = apex.parallel.convert_syncbn_model(model)
        else:
            model.cuda()
            if args.amp != 'O0':
                model, optimizer = amp.initialize(model, optimizer,
                                    opt_level=args.amp)
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.syncbn:
                model = apex.parallel.convert_syncbn_model(model)
    else:
        model = model.cuda()
        if args.amp != 'O0':
                model, optimizer = amp.initialize(model, optimizer,
                                    opt_level=args.amp)

    
    train_loader = D.make_dataloader(cfg, is_train=True, distributed=args.distributed)

    test_loader, _ = D.make_test_dataloader(cfg, distributed=args.distributed)
    
    InferenceShell = logocap.LogoCapInference(cfg)

    best_perf = -1.0
    best_model = False
    last_epoch = -1

    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth.tar')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        # if logger:
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file,
                                map_location=lambda storage, loc: storage)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    # lr_scheduler = get_lrscheduler(cfg,optimizer,len(train_loader),last_epoch=last_epoch)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        do_train(cfg,model,train_loader,targets_encoder,optimizer,lr_scheduler, epoch,final_output_dir, writer_dict, logger, 
        use_wandb = args.wandb)
        lr_scheduler.step()

        perf_indicator = run_test(cfg, InferenceShell, model, test_loader, final_output_dir, logger, use_wandb = args.wandb)

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        if not cfg.MULTIPROCESSING_DISTRIBUTED or (
                cfg.MULTIPROCESSING_DISTRIBUTED
                and args.rank == 0
        ):
            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': cfg.MODEL.NAME,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

if __name__ == '__main__':
    main()
    