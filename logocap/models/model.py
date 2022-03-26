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

import numpy as np

from torch.profiler import profile, record_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_module import BasicBlock, Bottleneck, HighResolutionModule
from .loss import LossEvaluator
from .decoder import Decoder
from .attentive_norm import AttnBatchNorm2d
from logocap.dataset.transforms import FLIP_CONFIG

from logocap.dataset.constants import PERSON_SIGMA_DICT, PERSON_SKELETON_DICT
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}


class Model(nn.Module):
    def compute_oks(self, pose_pool, keypoints_gt, areas_gt):
        num_centers, num_poses_per_center_y, num_poses_per_center_x = pose_pool.shape[:
                                                                                      3]
        pose_pool_reshaped = pose_pool.reshape(
            num_centers * num_poses_per_center_y * num_poses_per_center_x, self.num_joints,
            2)
        deltas = torch.sum(
            (pose_pool_reshaped[:, None, :, :2] - keypoints_gt[None, ..., :2])
            **2,
            dim=-1) * (keypoints_gt[None, ..., -1] > 0).float()
        var = (self.coco_sigma_constant.cuda() * 2)**2
        visible = (keypoints_gt[None, ..., -1] > 0).float()
        err = deltas / (2 * var * areas_gt[None, :, None]).clamp_min(1e-6)
        oks = torch.exp(-err) * visible
        oks_person = torch.sum(oks, dim=-1) / torch.sum(visible,dim=-1).clamp_min(1)
        oks_parts = oks.reshape(num_centers, num_poses_per_center_y,
                                num_poses_per_center_x, keypoints_gt.shape[0],
                                -1)
        oks_person = oks_person.reshape(num_centers, num_poses_per_center_y,
                                        num_poses_per_center_x, -1)

        oks_person_best = torch.sum(
            oks_parts.max(dim=1)[0].max(dim=1)[0] * visible,
            dim=-1) / torch.sum(visible, dim=-1).clamp_min(1)
        
        return oks_parts, oks_person, oks_person_best

    def compute_oks_with_gt(self,
                            pose_pool,
                            meta,
                            height,
                            width,
                            debug_image=None):
        device = pose_pool.device
        keypoints, areas = meta['annotations']['kpts'], meta['annotations'][
            'areas']
        keypoints = keypoints.to(device)
        areas = areas.to(device)
        inv_transform = meta['transform_inv']
        pose_pool_image_coordinate = torch.zeros_like(pose_pool)
        pose_pool_image_coordinate[..., -1] = pose_pool[..., -1]
        px = pose_pool[...,
                       0] if not meta['HFlip'] else width - 1 - pose_pool[..., 0]
        py = pose_pool[..., 1]

        # else:
        pose_pool_image_coordinate[..., 0] = px * inv_transform[
            0, 0] + py * inv_transform[0, 1] + inv_transform[0, 2]
        pose_pool_image_coordinate[..., 1] = px * inv_transform[
            1, 0] + py * inv_transform[1, 1] + inv_transform[1, 2]

        oks_parts, oks_person_, oks_person_best = self.compute_oks(
            pose_pool_image_coordinate, keypoints, areas)

        oks_max_, assignment = oks_person_best.max(dim=-1)
        
        pose_gt_image_coordinate = keypoints[assignment].clone()
        areas_ = areas[assignment].clone()[:, None] * (
            (self.coco_sigma_constant.cuda()[0] * 2)**2)
        pose_gt = torch.zeros_like(pose_gt_image_coordinate)

        transform = meta['transform']

        pose_gt[:, :, 0] = transform[
            0, 0] * pose_gt_image_coordinate[:, :, 0] + transform[
                0, 1] * pose_gt_image_coordinate[:, :, 1] + transform[0, 2]
        pose_gt[:, :, 1] = transform[
            1, 0] * pose_gt_image_coordinate[:, :, 0] + transform[
                1, 1] * pose_gt_image_coordinate[:, :, 1] + transform[1, 2]
        if meta['HFlip']:
            pose_gt[:, :, 0] = width - 1 - pose_gt[:, :, 0]

        pose_gt[:, :, 2] = transform[0, 0] * transform[1, 1] * areas_ * (
            pose_gt_image_coordinate[:, :, 2] > 0).float()
        pose_gt[:, :, 2] *= (pose_gt[:, :, 0] >= 0).float() * (
            pose_gt[:, :, 0] < width).float() * (pose_gt[:, :, 1] >= 0).float(
            ) * (pose_gt[:, :, 1] < height).float()
        center_idx = torch.arange(assignment.shape[0], device=device)
        oks_parts = oks_parts[center_idx, :, :, assignment]
        oks_person = oks_person_[center_idx, :, :, assignment]

        return oks_parts, oks_person, pose_gt

    def compute_oks_batch(self, pose_pool, keypoints, areas, debug = None):
        batch_size, num_centers, num_poses_per_center_y, num_poses_per_center_x = pose_pool.shape[:4]
        pose_pool_reshaped = pose_pool.reshape(batch_size, -1, self.num_joints, 2)

        deltas = torch.sum( (pose_pool_reshaped[:,:,None] - keypoints[:,None,...,:2])**2,dim=-1)*(keypoints[:,None,...,-1]>0)
        var = (self.coco_sigma_constant[None].cuda()*2)**2
        visible = keypoints[:,None,...,-1]>0
        err = deltas/(2*var*areas[:,None,:,None]).clamp_min(1e-6)

        oks = torch.exp(-err)*visible

        oks_person = torch.sum(oks, dim=-1) / torch.sum(visible,dim=-1).clamp_min(1)
        oks_person = oks_person.reshape(batch_size, num_centers, num_poses_per_center_y,
                                num_poses_per_center_x, keypoints.shape[1])

        oks_parts = oks.reshape(batch_size,num_centers,num_poses_per_center_y,num_poses_per_center_x,keypoints.shape[1],-1)

        oks_person_best = torch.sum(oks_parts.max(dim=2)[0].max(dim=2)[0] * visible, dim=-1) / torch.sum(visible, dim=-1).clamp_min(1)
        if debug:
            return deltas, var, visible, err
        return oks_parts, oks_person, oks_person_best

    def compute_oks_with_gt_batch(self, pose_pool, meta, height, width):
        device = pose_pool.device
        keypoints = meta['keypoints']
        areas = meta['areas']
        inv_transform = meta['transforms_inv'].reshape(-1,3,3,1,1,1,1)
        pose_pool_image_coordinate = torch.zeros_like(pose_pool)

        flip_stats = meta['HFlip'].reshape(-1,1,1,1,1)
        px = torch.where(flip_stats>0, width-1-pose_pool[...,0],pose_pool[...,0])
        py = pose_pool[...,1]

        pose_pool_image_coordinate[..., 0] = px * inv_transform[:,0,0] + py * inv_transform[:,0,1] + inv_transform[:,0,2]
        pose_pool_image_coordinate[..., 1] = px * inv_transform[:,1,0] + py * inv_transform[:,1,1] + inv_transform[:,1,2]

        oks_parts, oks_person_, oks_person_best = self.compute_oks_batch(pose_pool_image_coordinate, keypoints, areas)
        
        oks_max_, assignment = oks_person_best.max(dim=-1)
        
        ass_arange = torch.arange(pose_pool.shape[0],device='cuda').reshape(-1,1).expand_as(assignment)

        pose_gt_image_coordinate = keypoints[ass_arange,assignment].clone()

        areas_ = areas[ass_arange,assignment][:,:,None]*((self.coco_sigma_constant.cuda()*2)**2)

        pose_gt = torch.zeros_like(pose_gt_image_coordinate)

        transform = meta['transforms']
        pose_gt[:,:,:,0] = transform[:,0,0,None,None] * pose_gt_image_coordinate[:,:,:,0] + transform[:,0,1,None,None] * pose_gt_image_coordinate[:,:,:,1] + transform[:,0,2,None,None]
        pose_gt[:,:,:,1] = transform[:,1,0,None,None] * pose_gt_image_coordinate[:,:,:,0] + transform[:,1,1,None,None] * pose_gt_image_coordinate[:,:,:,1] + transform[:,1,2,None,None]

        pose_gt[:,:,:,0] = torch.where(meta['HFlip'].reshape(-1,1,1)>0, width-1-pose_gt[:,:,:,0], pose_gt[:,:,:,0])

        pose_gt[:,:,:,2] = (transform[:,0,0]*transform[:,1,1]).reshape(-1,1,1).expand_as(areas_)*areas_*(pose_gt_image_coordinate[:,:,:,2]>0)

        ctl_arange = torch.arange(pose_pool_image_coordinate.shape[1],device='cuda').reshape(1,-1).expand_as(assignment)
        
        oks_parts = oks_parts[ass_arange,ctl_arange,:,:,assignment]

        oks_person = oks_person_[ass_arange,ctl_arange,:,:,assignment]

        return oks_parts, oks_person, pose_gt

    def meta_batchify(self, meta_batch):
        batch_size = len(meta_batch)
        keypoints_batch = torch.zeros(batch_size,self.decoder.topk_center,self.num_joints,3,device='cuda')
        areas_batch = torch.zeros(batch_size,self.decoder.topk_center,device='cuda')
        transforms_batch = torch.zeros(batch_size,3,3,device='cuda')
        inv_transforms_batch = torch.zeros(batch_size,3,3,device='cuda')
        flip_stat_batch = torch.zeros(batch_size,device='cuda')

        for batch_id, meta in enumerate(meta_batch):
            keypoints = meta['annotations']['kpts']
            keypoints_batch[batch_id,:keypoints.shape[0]] = keypoints

            areas = meta['annotations']['areas']
            areas_batch[batch_id,:keypoints.shape[0]] = areas
            transforms_batch[batch_id] = meta['transform']
            inv_transforms_batch[batch_id] = meta['transform_inv']
            flip_stat_batch[batch_id] = meta['HFlip']

        return {'keypoints': keypoints_batch, 
                'areas':areas_batch,
                'transforms': transforms_batch,
                'transforms_inv': inv_transforms_batch,
                'HFlip': flip_stat_batch
                }
        
    def __init__(self, cfg, backbone):
        super(Model, self).__init__()
        self.backbone = backbone
        inp_channels = backbone.output_channels

        multi_output_config_heatmap = cfg['MODEL']['EXTRA'][
            'MULTI_LEVEL_OUTPUT_HEATMAP']

        multi_output_config_regression = cfg['MODEL']['EXTRA'][
            'MULTI_LEVEL_OUTPUT_REGRESSION']

        attn_norm_affine_num = cfg['MODEL']['EXTRA']['AN_NUM_AFFINE']
        attn_norm_for_all = cfg['MODEL']['EXTRA']['AN_FOR_ALL']

        self.adaptation_level_test = cfg.TEST.ADAPTATION_LEVEL
        assert self.adaptation_level_test in ['full', 'partial', 'none']

        self.transition_cls = nn.Sequential(
            nn.Conv2d(inp_channels,
                      multi_output_config_heatmap['NUM_CHANNELS'][0],
                      1,
                      1,
                      0,
                      bias=False),
            nn.BatchNorm2d(multi_output_config_heatmap['NUM_CHANNELS'][0]),
            nn.ReLU(True))

        self.transition_reg = nn.Sequential(
            nn.Conv2d(inp_channels,
                      multi_output_config_regression['NUM_CHANNELS'][0],
                      1,
                      1,
                      0,
                      bias=False),
            nn.BatchNorm2d(multi_output_config_regression['NUM_CHANNELS'][0]),
            nn.ReLU(True))

        self.final_layer_cls = nn.Conv2d(
            multi_output_config_heatmap['NUM_CHANNELS'][0], cfg.DATASET.NUM_JOINTS, 1, 1, 0)
        self.final_layer_reg = nn.Conv2d(
            multi_output_config_regression['NUM_CHANNELS'][0], (cfg.DATASET.NUM_JOINTS-1)*2, 1, 1, 0)

        NUM_JOINTS_WITHOUT_CENTER = cfg.DATASET.NUM_JOINTS-1
        
        cmp_config = cfg.MODEL.EXTRA.CONV_MSG_PASSING

        self.transition_embedding = nn.Sequential(
            nn.Conv2d(inp_channels, cmp_config.DIM_EMBEDDING, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cmp_config.DIM_EMBEDDING), nn.ReLU(True)
            )
        
        conv_msg_passing = []
        input_dim = NUM_JOINTS_WITHOUT_CENTER \
                    * cmp_config.DIM_EMBEDDING
        for layer_id, (output_dim, use_an) in enumerate(zip(cmp_config.DIM_CONVOLUTION,
                                      cmp_config.USE_AN)):
            
            if 'KSIZE' in cmp_config:
                ksize = cmp_config['KSIZE'][layer_id]
            else:
                ksize = 3
            padding = ksize //2 
            conv_msg_passing.append(
                nn.Conv2d(input_dim, output_dim, ksize, 1, padding=padding, bias=False)
            )
            normlayer = AttnBatchNorm2d(
                output_dim, 
                attn_norm_affine_num, 
                use_bn=False) if use_an else nn.BatchNorm2d(output_dim)
            
            conv_msg_passing.append(normlayer)
            conv_msg_passing.append(nn.ReLU(inplace=True))
            input_dim = output_dim
        
        conv_msg_passing.append(
            nn.Conv2d(input_dim, NUM_JOINTS_WITHOUT_CENTER, 1, 1, 0)
        )
        self.num_joints = NUM_JOINTS_WITHOUT_CENTER
        self.joints_mlp = nn.Sequential(*conv_msg_passing)
        self.cap_sigma = cfg.MODEL.DECODER.SIGMA
        self.cap_local_hm_size = cfg.MODEL.DECODER.LOCAL_HM_SIZE
        self.cap_global_hm_size = cfg.MODEL.DECODER.GLOBAL_HM_SIZE
        self.cap_topk_center = cfg.MODEL.DECODER.TOPK_CENTER
        self.cap_ksize = cfg.MODEL.DECODER.KSIZE
        
        self.loss_evaluator = LossEvaluator(cfg)
        self.decoder = Decoder(cfg)
        self.flip_test = True
        self.coco_sigma_constant = torch.tensor(PERSON_SIGMA_DICT[cfg.MODEL.TEMPLATE],
                                                dtype=torch.float32)[None,
                                                                     None]
        self.debug_with_gt = False
        self.debug_step = 0

    def feature_bilinear_sampling_per_image(self, features, px, py):
        _, height, width = features.shape
        shape = px.shape
        px = px.reshape(-1)
        py = py.reshape(-1)
        px0 = px.floor().clamp(min=0, max=width - 1)
        py0 = py.floor().clamp(min=0, max=height - 1)
        px1 = (px0 + 1).clamp(min=0, max=width - 1)
        py1 = (py0 + 1).clamp(min=0, max=height - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        delta_x0 = (px1 - px).clamp(min=0, max=1.0)
        delta_y0 = (py1 - py).clamp(min=0, max=1.0)
        delta_x1 = (px - px0).clamp(min=0, max=1.0)
        delta_y1 = (py - py0).clamp(min=0, max=1.0)
        features_00 = features[:, py0l, px0l] * delta_y0[None] * delta_x0[None]
        features_01 = features[:, py0l, px1l] * delta_y0[None] * delta_x1[None]
        features_10 = features[:, py1l, px0l] * delta_y1[None] * delta_x0[None]
        features_11 = features[:, py1l, px1l] * delta_y1[None] * delta_x1[None]

        out = features_00 + features_01 + features_10 + features_11
        out = out.permute((1, 0)).contiguous()
        out = out.reshape(*shape, -1)

        return out

    def feature_bilinear_sampling_batch(self, features, px, py):
        batch_size, _, height, width = features.shape
        shape = px.shape[1:]
        px = px.reshape(batch_size,-1)
        py = py.reshape(batch_size,-1)
        px0 = px.floor().clamp(min=0, max=width - 1)
        py0 = py.floor().clamp(min=0, max=height - 1)
        px1 = (px0 + 1).clamp(min=0, max=width - 1)
        py1 = (py0 + 1).clamp(min=0, max=height - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        batch_arange = torch.arange(batch_size,device=features.device).reshape(-1,1).expand_as(px0l)
        delta_x0 = (px1 - px).clamp(min=0, max=1.0)
        delta_y0 = (py1 - py).clamp(min=0, max=1.0)
        delta_x1 = (px - px0).clamp(min=0, max=1.0)
        delta_y1 = (py - py0).clamp(min=0, max=1.0)
        features_00 = features[batch_arange, :, py0l, px0l] * delta_y0[...,None] * delta_x0[...,None]
        features_01 = features[batch_arange, :, py0l, px1l] * delta_y0[...,None] * delta_x1[...,None]
        features_10 = features[batch_arange, :, py1l, px0l] * delta_y1[...,None] * delta_x0[...,None]
        features_11 = features[batch_arange, :, py1l, px1l] * delta_y1[...,None] * delta_x1[...,None]

        out = features_00 + features_01 + features_10 + features_11
        out = out.reshape(batch_size,*shape,-1)
        return out


    def forward_training(self, images_batch, targets_batch):
        assert self.training
        batch_size = images_batch.shape[0]
        features = self.backbone(images_batch)
        transition_cls = self.transition_cls(features)
        transition_reg = self.transition_reg(features)

        heatmaps = self.final_layer_cls(transition_cls)
        offsets = self.final_layer_reg(transition_reg)
        output_dict = {
            'heatmaps': heatmaps,
            'offsets': offsets
        }

        heatmaps_hr = torch.nn.functional.interpolate(
            heatmaps,
            size=(images_batch.shape[2], images_batch.shape[3]),
            mode='bilinear',
            align_corners=False)
        embedding = self.transition_embedding(features)

        oks_joints_batch = []
        oks_pred_batch = []
        oks_person_batch = []

        out_hm_batch = []
        gt_hm_batch = []

        debug_cond = self.debug_step in []

        decoder_output = self.decoder(images_batch, {'centermaps':heatmaps_hr[:,-1:].detach(),'offsets':offsets.detach()})
        
        yy, xx = torch.meshgrid(
                torch.arange(-self.cap_global_hm_size//2, self.cap_global_hm_size//2 + 1, device='cuda', dtype=torch.float32),
                torch.arange(-self.cap_global_hm_size//2, self.cap_global_hm_size//2 + 1, device='cuda', dtype=torch.float32))
        yy = yy[None, None]
        xx = xx[None, None]

        g = (torch.exp(-(xx**2 + yy**2) / (2 * self.cap_sigma * self.cap_sigma)))
        g *= g >= 1e-4

        latent_code_batch = []
        in_hm_batch = []
        for batch_id, (meta, image) in enumerate(
                zip(targets_batch['meta'], images_batch)):

            pose_pool_per_im = decoder_output['allposes'][batch_id]  # 30,11,11,17,2
            ##  heatmap refinement
            # import pdb; pdb.set_trace()
            oks_joints, oks_person, pose_gt = self.compute_oks_with_gt(
                pose_pool_per_im,
                meta,
                height=image.shape[1],
                width=image.shape[2],
                debug_image=image)  # 30,11,11,17   30,11,11

            mask = oks_person.max(dim=-1)[0].max(dim=-1)[0] >= 0.5
            if mask.sum() == 0:
                mask[0] = 1
            pose_pool_per_im = pose_pool_per_im[mask]
            oks_joints = oks_joints[mask]
            oks_person = oks_person[mask]
            pose_gt = pose_gt[mask]
            pose_gt[..., -1] *= (pose_gt[..., 0] >= 0) * (
                pose_gt[..., 0] < image.shape[2]) * (pose_gt[..., 1] >= 0) * (
                    pose_gt[..., 1] < image.shape[1])

            latent_code = self.feature_bilinear_sampling_per_image(
                embedding[batch_id], pose_pool_per_im[..., 0] / 4.0,
                pose_pool_per_im[..., 1] / 4.0)
            #  30,17,64,11,11
            latent_code = latent_code.permute((0, 3, 4, 1, 2)).contiguous()
            
            latent_code = latent_code.reshape(latent_code.shape[0], -1,
                                            self.decoder.ksize * 2 + 1,
                                            self.decoder.ksize * 2 + 1)
            
            latent_code_batch.append(latent_code)
            # oks_pred = self.joints_mlp(latent_code)

            
            center_poses = pose_pool_per_im[:, self.cap_ksize, self.cap_ksize]
            pose_xx = xx + center_poses[:, :, 0, None, None]
            pose_yy = yy + center_poses[:, :, 1, None, None]
            in_hm = self.decoder.jointness_bilinear_sampling_single_image(
                heatmaps_hr[batch_id, :-1],
                pose_xx.transpose(0, 1).contiguous(),
                pose_yy.transpose(0, 1).contiguous())
            in_hm = in_hm.transpose(0, 1).contiguous()
            
            in_hm *= g
            in_hm_batch.append(in_hm)
            
            # in_hm = in_hm.reshape(1, -1, in_hm.shape[2], in_hm.shape[3])
            
            # kernel = torch.flip((oks_pred / self.cap_local_hm_size**2).reshape(-1, 1, self.cap_local_hm_size, self.cap_local_hm_size),
            #                     [2, 3])
            # out_hm = F.conv2d(in_hm, kernel, padding=self.cap_ksize, groups=kernel.shape[0])
            # out_hm = out_hm.reshape(pose_pool_per_im.shape[0], self.num_joints,
            #                         *out_hm.shape[2:])
            xx_g = pose_xx - pose_gt[:, :, 0, None, None]
            yy_g = pose_yy - pose_gt[:, :, 1, None, None]
            gt_hm = torch.exp(-(xx_g**2 + yy_g**2) / (2 * 4*4)) #TODO: MAYBE A BUG EXISTED

            # oks_pred_batch.append(oks_pred)
            oks_joints_batch.append(oks_joints)
            oks_person_batch.append(oks_person)
            # out_hm_batch.append(out_hm)
            gt_hm_batch.append(gt_hm)
        latent_code_batch = torch.cat(latent_code_batch)
        oks_pred_batch = self.joints_mlp(latent_code_batch)
        in_hm_batch = torch.cat(in_hm_batch)
        kernel = torch.flip((oks_pred_batch / self.cap_local_hm_size**2).reshape(-1, 1, self.cap_local_hm_size, self.cap_local_hm_size),[2, 3])
        in_hm_batch = in_hm_batch.reshape(1, -1, in_hm_batch.shape[2], in_hm_batch.shape[3])
        out_hm_batch = F.conv2d(in_hm_batch,kernel, padding=self.cap_ksize, groups=kernel.shape[0])
        out_hm_batch = out_hm_batch.reshape(-1, self.num_joints, *out_hm_batch.shape[2:])

            # oks_pred_batch.append(code)
        # oks_pred_batch = torch.cat(oks_pred_batch)

    
        oks_joints_batch = torch.cat(oks_joints_batch)

        oks_joints_batch = oks_joints_batch.permute((0, 3, 1, 2)).contiguous()
        oks_person_batch = torch.cat(oks_person_batch)

        # out_hm_batch = torch.cat(out_hm_batch) if len(out_hm_batch) > 0 else out_hm_batch
        gt_hm_batch = torch.cat(gt_hm_batch) if len(out_hm_batch) > 0 else gt_hm_batch

        # err1 = (oks_joints_batch-oks_joints_batch_).abs().max()
        # err2 = (out_hm_batch-out_hm_batch_).abs().max()
        # err3 = (oks_person_batch - oks_person_batch).abs().max()
        # err4 = (gt_hm_batch - gt_hm_batch_).abs().max()
        # err = err1+err2+err3+err4
        # print(err)
        

        output_dict['oks.joints'] = oks_pred_batch
        output_dict['out_hm'] = out_hm_batch

        targets_batch['oks.joints'] = oks_joints_batch
        targets_batch['oks.person'] = oks_person_batch
        targets_batch['out_hm'] = gt_hm_batch

        loss_dict = self.loss_evaluator(output_dict, targets_batch)
        # output_dict['prof'] = prof
        self.debug_step +=1
        return output_dict, loss_dict

    def refine_none(self, allposes, embedding, heatmaps, **kwargs):
        # height, width = embedding.shape[2:]
        height, width = heatmaps.shape[1:]
        center_poses = allposes[:, self.cap_ksize, self.cap_ksize]

        pose_xx = center_poses[:, :, 0, None, None]
        pose_yy = center_poses[:, :, 1, None, None]
        in_hm = self.decoder.jointness_bilinear_sampling_single_image(
            heatmaps,
            pose_xx.transpose(0, 1).contiguous(),
            pose_yy.transpose(0, 1).contiguous())
        in_hm = in_hm.transpose(0, 1).contiguous().squeeze(-1)

        return torch.cat((center_poses, in_hm), dim=-1), None

    def refine_partial(self, allposes, embedding, heatmaps, **kwargs):
        height, width = embedding.shape[2:]
        latent_code = self.feature_bilinear_sampling_per_image(
            embedding[0], allposes[..., 0] / 4.0, allposes[..., 1] / 4.0)

        latent_code = latent_code.permute((0, 3, 4, 1, 2)).contiguous()
        latent_code = latent_code.reshape(allposes.shape[0], -1,
                                          self.decoder.ksize * 2 + 1,
                                          self.decoder.ksize * 2 + 1)

        oks_joints = self.joints_mlp(latent_code).clamp(0, 1.0)
        # oks_joints = self.joints_mlp(latent_code).sigmoid()

        oks_joints = oks_joints.reshape(oks_joints.shape[0],self.num_joints,-1).permute((0,2,1)).contiguous()
        max_val, argmax = oks_joints.max(dim=1)
        argmax_mat = torch.zeros_like(oks_joints)
        argmax_mat.scatter_(1,argmax[:,None],1)

        allposes = allposes.reshape(oks_joints.shape[0],-1,17,2)
        final_poses = torch.sum(allposes*argmax_mat[...,None],dim=1)
        heat_values = max_val.clone()
        final_poses = torch.cat((final_poses,heat_values[...,None]),dim=-1)

        return final_poses, None

    def refine(self, allposes, embedding, heatmaps, **kwargs):
        stride = kwargs.get('stride',4.0)
        torch.cuda.synchronize()
        now = time()
        height, width = embedding.shape[2:]
        latent_code = self.feature_bilinear_sampling_per_image(
            embedding[0], allposes[..., 0] / stride, allposes[..., 1] / stride)
        
        # latent_code_flip = self.feature_bilinear_sampling_per_image(embedding[1], width-1-allposes[...,0]/4.0, allposes[...,1]/4.0)
        # latent_code_flip = latent_code_flip[:,:,:,self.decoder.flip_config]

        # latent_code = 0.5*(latent_code+latent_code_flip)
        meta = kwargs.get('meta',None)

        latent_code = latent_code.permute((0, 3, 4, 1, 2)).contiguous()
        latent_code = latent_code.reshape(allposes.shape[0], -1,
                                          self.decoder.ksize * 2 + 1,
                                          self.decoder.ksize * 2 + 1)

        oks_joints = self.joints_mlp(latent_code).clamp(0, 1.0)
        oks_mask = (allposes[...,0]/stride>=0)*(allposes[...,0]/stride<=width-1)*(allposes[...,1]/stride>=0)*(allposes[...,1]/stride<=height-1)
        
        # oks_joints *= oks_mask.float().permute((0,3,1,2))

        torch.cuda.synchronize()
        kwargs['timing']['local-kam'] += time()-now
        
        torch.cuda.synchronize()
        now = time()
        # oks_joints = code.reshape(code.shape[0],17,11,11)
        # import pdb; pdb.set_trace()

        # kernel = oks_joints/121.0
        # kernel = kernel.transpose(0,1).contiguous()
        x_start = -self.cap_global_hm_size//2
        x_end = self.cap_global_hm_size//2
        y_start = -self.cap_global_hm_size//2
        y_end = self.cap_global_hm_size//2


        yy, xx = torch.meshgrid(
            torch.arange(y_start,
                         y_end + 1,
                         device='cuda',
                         dtype=torch.float32),
            torch.arange(x_start,
                         x_end + 1,
                         device='cuda',
                         dtype=torch.float32))
        yy = yy[None, None]
        xx = xx[None, None]
        # import pdb; pdb.set_trace()
        # xx = xx*sigmas
        # yy = yy*sigmas
        center_poses = allposes[:, self.cap_ksize, self.cap_ksize]
        pose_xx = xx + center_poses[:, :, 0, None, None]
        pose_yy = yy + center_poses[:, :, 1, None, None]
        
        in_hm = self.decoder.jointness_bilinear_sampling_single_image(
            heatmaps,
            pose_xx.transpose(0, 1).contiguous(),
            pose_yy.transpose(0, 1).contiguous())
        # import pdb; pdb.set_trace()

        in_hm = in_hm.transpose(0, 1).contiguous()

        g = (torch.exp(-(xx**2 + yy**2) / (2 * self.cap_sigma * self.cap_sigma)))
        g *= g >= 1e-4 #TODO: hardcode
        in_hm *= g
        in_hm = in_hm.reshape(1, -1, in_hm.shape[2], in_hm.shape[3])
        kernel = torch.flip((oks_joints / self.cap_local_hm_size**2).reshape(-1, 1, self.cap_local_hm_size, self.cap_local_hm_size),
                            [2, 3]) 

        out_hm = F.conv2d(in_hm, kernel, padding=self.cap_ksize, groups=kernel.shape[0])
        out_hm = out_hm.reshape(allposes.shape[0], self.num_joints, -1)

        value, idx = out_hm.topk(2)

        topk_y = torch.div(idx,xx.shape[-1],rounding_mode='trunc')
        topk_x = idx % xx.shape[-1]
        
        topk_v = value
        lambda1 = 0.75
        lambda2 = 1 - lambda1
        topk_y = lambda1 * topk_y[..., :1] + lambda2 * topk_y[..., 1:]
        topk_x = lambda1 * topk_x[..., :1] + lambda2 * topk_x[..., 1:]
        topk_v = lambda1 * topk_v[..., :1] + lambda2 * topk_v[..., 1:]
        # topk_v = 1.0 * topk_v[..., :1] + 0.0 * topk_v[..., 1:]

        offset = torch.cat((topk_x, topk_y), dim=-1)
        offset[..., 0] += xx.min()
        offset[..., 1] += yy.min()


        pose = center_poses + offset
        pose = torch.cat((pose, topk_v), dim=-1)
        torch.cuda.synchronize()

        kwargs['timing']['global-kam'] += time() - now
        return pose, None

    def forward_multiscale(self, images_batch, scale_factors, targets_batch=None):
        timing = {
            'backbone': 0,
            'local-kem': 0,
            'local-kam': 0,
            'global-kam': 0,
            }

        main_index = scale_factors.index(1.0)
        if self.flip_test:
            for i in range(len(images_batch)):
                img_flip = torch.flip(images_batch[i],[3])
                images_batch[i] = torch.cat((images_batch[i],img_flip),dim=0)
            

        features = self.backbone(images_batch[main_index])

        transition_cls = self.transition_cls(features)
        transition_reg = self.transition_reg(features)
        heatmaps = self.final_layer_cls(transition_cls)
        offsets = self.final_layer_reg(transition_reg)
        embedding = self.transition_embedding(features)

        if self.flip_test:
            heatmaps_flip = torch.flip(heatmaps[1:,self.decoder.flip_config_with_center],[3])
            heatmaps = (heatmaps[:1] + heatmaps_flip)*0.5
            offsets_flip = torch.flip(offsets[1:],[3])
            offsets_flip = offsets_flip.reshape(1,-1,2,offsets.shape[2],offsets.shape[3])[:,self.decoder.flip_config]
            offsets_flip[:,:, 0] *= -1.0
            offsets_flip = offsets_flip.reshape(1,-1,offsets.shape[2],offsets.shape[3])
            offsets = (offsets[:1] + offsets_flip)*0.5
        
        heatmaps_hr = torch.nn.functional.interpolate(heatmaps,
            size=(images_batch[main_index].size(2),images_batch[main_index].size(3)), mode='bilinear',align_corners=False)

        initial_poses = []
        initial_dict = self.decoder(images_batch[main_index],{'centermaps':heatmaps_hr[:,-1:],'offsets':offsets})
        initial_pose = initial_dict['allposes'][0]
        mask = initial_dict['centers'][0,...,-1]>=1e-2
        if mask.sum() == 0:
            mask[0] =1
        
        initial_poses = [initial_pose[mask]]
        initial_centers = [initial_dict['centers'][0,...,-1][mask]]

        max_score = initial_centers[0].max()

        for i in range(len(images_batch)):
            if i == main_index:
                continue
            features_ = self.backbone(images_batch[i])
            embedding_ = self.transition_embedding(features_)
            embedding += torch.nn.functional.interpolate(embedding_, size=(embedding.size(2),embedding.size(3)),mode='bilinear',align_corners=False)

            offsets_ = self.final_layer_reg(self.transition_reg(features_))
            offsets_f = torch.flip(offsets_[1:], [3])
            offsets_f = offsets_f.reshape(1,-1,2,offsets_.shape[2],offsets_.shape[3])[:,self.decoder.flip_config]
            offsets_f[:,:, 0] *= -1.0
            offsets_f = offsets_f.reshape(1,-1,offsets_.shape[2],offsets_.shape[3])
            offsets_ = (offsets_[:1] + offsets_f)*0.5
            heatmaps_ = self.final_layer_cls(self.transition_cls(features_))
            heatmaps_f = torch.flip(heatmaps_[1:,self.decoder.flip_config_with_center],[3])
            # heatmaps_f[...,1:] = heatmaps_f[...,:-1]
            heatmaps_ = (heatmaps_[:1] + heatmaps_f)*0.5
            heatmaps_hr_ = torch.nn.functional.interpolate(heatmaps_,
            size=(images_batch[main_index].size(2),images_batch[main_index].size(3)), mode='bilinear',align_corners=False)

            initial_dict = self.decoder(images_batch[i], {'centermaps':heatmaps_hr_[:,-1:],'offsets':offsets_})

            initial_pose = initial_dict['allposes'][0]
            mask = initial_dict['centers'][0,...,-1]>=0.1
            if mask.sum() > 1000:
                initial_pose = initial_pose[mask]/scale_factors[i]
                initial_poses.append(initial_pose)
                scores = initial_dict['centers'][0,...,-1][mask]*0.9
                # max_score_scale = scores.max()
                # scores = scores/max_score_scale*max_score*0.9
                initial_centers.append(initial_dict['centers'][0,...,-1][mask])
            heatmaps_hr += heatmaps_hr_

        # heatmaps_hr = torch.nn.functional.interpolate(heatmaps,
        #     size=(images_batch[main_index].size(2),images_batch[main_index].size(3)), mode='bilinear',align_corners=False)

        heatmaps_hr = heatmaps_hr/len(images_batch)
        embedding = embedding/len(images_batch)
        # embedding[0] = (embedding[0] + torch.flip(embedding[1],[2]))
        # import pdb; pdb.set_trace()
        output_dict = {
            'centermaps': heatmaps_hr[:,-1:],
            'offsets': offsets,
        }

        torch.cuda.synchronize()
        now = time()
        timing['backbone'] += time() - now
        loss_dict = {}
        

        allposes = torch.cat(initial_poses)
        allcenters = torch.cat(initial_centers)
        torch.cuda.synchronize()
        timing['local-kem'] += time() - now

        now = time()
        
        if self.adaptation_level_test == 'none':
            refine_fn = self.refine_none
            embedding = features
        elif self.adaptation_level_test == 'partial':
            refine_fn = self.refine_partial            
        else:
            refine_fn = self.refine

        final_poses, hm_instances = refine_fn(allposes,
                                                embedding,
                                                heatmaps_hr[0],
                                                debug_image=images_batch,timing=timing,
                                                meta = None if targets_batch is None else targets_batch['meta'])
        # final_poses[..., -1] *= output_dict['centers'][0][mask][..., None, -1]
        temp_x = final_poses[...,0]
        temp_y = final_poses[...,1]
        temp_x = temp_x.t()[:,None,None]
        temp_y = temp_y.t()[:,None,None]
        values = self.decoder.jointness_bilinear_sampling_single_image(heatmaps_hr[0], temp_x, temp_y)[:,0,0,]
        # self.decoder.jointness_bilinear_sampling_single_image(heatmaps_hr, final_poses[...,None,0].permute(), final_poses[...,None,1])
        # final_poses[..., -1] = torch.sqrt(final_poses[..., -1]*values.t())
        # final_poses[..., -1] = (0.75*final_poses[..., -1]+0.25*values.t())
        final_poses[..., -1] *= allcenters[:,None]
        final_poses[...,0] = final_poses[...,0].clamp(min=0,max=images_batch[1].shape[-1])
        final_poses[...,1] = final_poses[...,1].clamp(min=0,max=images_batch[1].shape[-2])
        
        scores = final_poses[..., -1].mean(dim=-1)
        # valid_poses = scores >= 1e-3
        valid_poses = scores >0
        num_valid_poses = valid_poses.sum()
        if num_valid_poses == 0:
            output_dict['poses'] = None
            return output_dict, loss_dict

        output_dict['poses'] = final_poses[valid_poses]
        output_dict['scores'] = scores[valid_poses]
        output_dict['allposes'] = allposes[valid_poses]
        output_dict['centers'] = allcenters
        output_dict['heatmaps'] = heatmaps_hr
        
        return output_dict, timing

    def forward(self, images_batch, targets_batch=None):
        if self.training:
            return self.forward_training(images_batch, targets_batch)
        if self.flip_test:
            temp = torch.flip(images_batch, [3])
            images_batch_flip = temp
            images_batch = torch.cat((images_batch, images_batch_flip), dim=0)
        
        timing = {
            'backbone': 0,
            'local-kem': 0,
            'local-kam': 0,
            'global-kam': 0,
            }
        torch.cuda.synchronize()
        now = time()
        features = self.backbone(images_batch)

        transition_cls = self.transition_cls(features)
        transition_reg = self.transition_reg(features)
        heatmaps = self.final_layer_cls(transition_cls)
        offsets = self.final_layer_reg(transition_reg)

        # jointsmaps, centermaps = heatmaps[:,:-1], heatmaps[:,-1:]

        if self.flip_test:
            heatmaps_flip = torch.flip(heatmaps[1:,self.decoder.flip_config_with_center],[3])
            heatmaps = (heatmaps[:1] + heatmaps_flip)*0.5
            offsets_flip = torch.flip(offsets[1:],[3])
            offsets_flip = offsets_flip.reshape(1,-1,2,offsets.shape[2],offsets.shape[3])[:,self.decoder.flip_config]
            offsets_flip[:,:, 0] *= -1.0
            offsets_flip = offsets_flip.reshape(1,-1,offsets.shape[2],offsets.shape[3])
            offsets = (offsets[:1] + offsets_flip)*0.5
        
        heatmaps_hr = torch.nn.functional.interpolate(heatmaps,
            size=(images_batch.size(2),images_batch.size(3)), mode='bilinear',align_corners=False)
        
        output_dict = {
            'centermaps': heatmaps_hr[:,-1:],
            # 'centermaps': heatmaps[:,-1:],
            'offsets': offsets,
        }

        torch.cuda.synchronize()
        timing['backbone'] += time() - now
        
        torch.cuda.synchronize()
        now = time()
        loss_dict = {}
        
        output_dict.update(
            self.decoder(images_batch,
                         output_dict,
                         flip_testing=self.flip_test))

        mask = output_dict['centers'][0,..., -1] >= 1e-2
        # mask = output_dict['centers'][0,..., -1] >= -1
        if mask.sum() == 0:
            mask[0] = 1

        allposes = output_dict['allposes'][0][mask]

        torch.cuda.synchronize()
        timing['local-kem'] += time() - now

        now = time()
        if self.adaptation_level_test == 'none':
            refine_fn = self.refine_none
            embedding = features
        elif self.adaptation_level_test == 'partial':
            refine_fn = self.refine_partial
            embedding = self.transition_embedding(features)
        else:
            refine_fn = self.refine
            embedding = self.transition_embedding(features)
        final_poses, hm_instances = refine_fn(allposes,
                                                embedding,
                                                heatmaps_hr[0],
                                                debug_image=images_batch,timing=timing,)                                                                         
        final_poses[..., -1] *= output_dict['centers'][0][mask][..., None, -1]
        # final_poses[..., -1] = output_dict['centers'][0][mask][..., None, -1]
        # final_poses[...,0] = final_poses[...,0].clamp(min=0,max=images_batch.shape[-1])
        # final_poses[...,1] = final_poses[...,1].clamp(min=0,max=images_batch.shape[-2])
        scores = final_poses[..., -1].mean(dim=-1)
        # valid_poses = scores >= 1e-3
        valid_poses = scores >0
        num_valid_poses = valid_poses.sum()
        if num_valid_poses == 0:
            output_dict['poses'] = None
            return output_dict, loss_dict

        output_dict['poses'] = final_poses[valid_poses]
        output_dict['scores'] = scores[valid_poses]
        output_dict['allposes'] = allposes[valid_poses]
        output_dict['centers'] = output_dict['centers'][0][mask]
        output_dict['heatmaps'] = heatmaps_hr
        return output_dict, timing
