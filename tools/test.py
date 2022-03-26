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

import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import logocap
from logocap.config import cfg,update_config,check_config
# from logocap.config import 
import logocap.dataset as D
from logocap.utils.utils import get_optimizer, create_logger, save_checkpoint, setup_logger
from logocap.trainer import do_train
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
torch.set_grad_enabled(False)
torch.multiprocessing.set_sharing_strategy('file_system')

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('--ckpt',
                        default = None,
                        type=str,
                        help='checkpoint path for inference'
    )

    args = parser.parse_args()

    return args

def run_test(cfg,infshell, model, test_loader, final_output_dir, logger):
    DETECTION_RESULTS = []
    model.eval()
    local_rank = logocap.utils.comm.get_rank()

    if local_rank == 0:
        pbar = tqdm(total=len(test_loader))
    
    total_time = 0
    import time

    timing = {}
    imgIds = []
    for i, (images, rgb, meta) in enumerate(test_loader):
        assert 1 == images.size(0), 'Test batch size should be 1'
        imgIds.append(meta[0]['id'])
        start = time.time()
        outputs, timing_per_im = model(images.cuda())
        total_time += time.time() - start
        for key, val in timing_per_im.items():
            if key in timing:
                timing[key] += val
            else:
                timing[key] = val

        final_poses, scores = infshell(images, rgb, outputs, meta[0])
        if final_poses is None:
            pbar.update(1)
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
                # 'bbox': [xmin.item(),ymin.item(),bw.item(),bh.item()]
                'bbox': [xmin,ymin,bw,bh]
                # 'area': bw*bh
                }
            DETECTION_RESULTS.append(ans)
        if local_rank == 0:
            pbar.update(1)
    pbar.close()
    
    for key in timing.keys():
        timing[key] /= float(len(test_loader))
        
    average_fps = len(test_loader)/total_time
    logger.info('Speed = {:.1f} FPS'.format(average_fps))
    logger.info(timing)

    results_path_gathered = os.path.join(final_output_dir,'eval_results-{}.json'.format(cfg.DATASET.TEST))
    with open(results_path_gathered, 'w') as writer:
        json.dump(DETECTION_RESULTS,writer)
    
    coco = test_loader.dataset.coco
    coco_eval = coco.loadRes(results_path_gathered)
    default = 'coco'
    
    from pycocotools.cocoeval import COCOeval as COCOEvaluator
    
    try:
        evaluator = COCOEvaluator(coco, coco_eval, 'keypoints')
        evaluator.params.imgIds = imgIds
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    except:
        return 0.0
    
    validation_msg = 'AP = {:.3f}| AP50 = {:.3f}| AP75 = {:.3f}| APM = {:.3f}| APL = {:.3f}'.format(evaluator.stats[0],evaluator.stats[1],evaluator.stats[2],evaluator.stats[3],evaluator.stats[4])
    
    logger.info('validation: {}'.format(validation_msg))
    
    return float(evaluator.stats[0])
    

def main():
    # import pdb; pdb.set_trace()
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)

    final_output_dir = create_logger(cfg,args.cfg)

    os.makedirs(final_output_dir,exist_ok=True)

    
    # cudnn.benchmark = cfg.CUDNN.BENCHMARK
    # torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    # torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED


    logger = setup_logger('logocap',final_output_dir,logocap.utils.comm.get_rank(),filename='test.log')
    model, targets_encoder = logocap.models.build_model(cfg, is_train=False)

    if args.ckpt:
        model_state_file = args.ckpt
    elif cfg.TEST.MODEL_FILE != "":
        model_state_file = cfg.TEST.MODEL_FILE
    else:
        raise RuntimeError()
    
    logger.info('=> loading model from {}'.format(model_state_file))
    model.load_state_dict(torch.load(model_state_file,map_location='cpu'))
    model.eval()
    
    model = model.to('cuda')
    
    test_loader, _ = D.make_test_dataloader(cfg)
    
    InferenceShell = logocap.LogoCapInference(cfg)

    
    perf_indicator = run_test(cfg, InferenceShell, model, test_loader, final_output_dir, logger)


if __name__ == '__main__':
    main()
    
