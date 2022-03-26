import torch
import numpy as np
from logocap.dataset.transforms import FLIP_CONFIG
from logocap.dataset.constants import PERSON_SIGMA_DICT, PERSON_SKELETON_DICT


class Decoder(object):
    def __init__(self, cfg):
        self.topk_center = cfg.MODEL.DECODER.TOPK_CENTER
        self.ksize = cfg.MODEL.DECODER.KSIZE
        self.person_skeleton = PERSON_SKELETON_DICT[cfg.MODEL.TEMPLATE]
        self.person_sigma    = PERSON_SIGMA_DICT[cfg.MODEL.TEMPLATE]
        self.num_keypoints = len(self.person_sigma)
        self.flip_config = FLIP_CONFIG[cfg.MODEL.TEMPLATE.upper()]
        self.flip_config_with_center = FLIP_CONFIG[cfg.MODEL.TEMPLATE.upper()+'_WITH_CENTER']

    def center_nms(self, centermaps, kernel_size=3):
        height, width = centermaps.shape[2:]
        batch_size = centermaps.shape[0]

        map_nms = torch.nn.functional.max_pool2d(centermaps.clamp(min=0), kernel_size, 1, kernel_size//2)
        map_after_nms = (map_nms == centermaps).float() * (centermaps)
        map_after_nms_reshaped = map_after_nms.reshape(batch_size,-1)
        topk_conf, topk_idx = map_after_nms_reshaped.topk(k=self.topk_center,sorted=True,dim=-1)

        topk_x = topk_idx % width
        # topk_y = topk_idx // width
        topk_y =  torch.div(topk_idx, width, rounding_mode='trunc')

        centers_xyv = torch.stack((topk_x.float(), topk_y.float(), topk_conf),
                                  dim=-1)
        return centers_xyv

    def jointness_bilinear_sampling_single_image(self, jointness, px, py):
        device = jointness.device
        _, height, width = jointness.shape
        px0 = px.floor().clamp(min=0, max=width - 1)
        py0 = py.floor().clamp(min=0, max=height - 1)
        px1 = (px0 + 1).clamp(min=0, max=width - 1)
        py1 = (py0 + 1).clamp(min=0, max=height - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        jtype_idx = torch.arange(self.num_keypoints, device=device).reshape(-1, 1, 1,1).expand_as(px)
        dy0 = (py1 - py).clamp(min=0, max=1.0)
        dy1 = (py - py0).clamp(min=0, max=1.0)
        dx0 = (px1 - px).clamp(min=0, max=1.0)
        dx1 = (px - px0).clamp(min=0, max=1.0)

        value_00 = jointness[jtype_idx, py0l, px0l] * dy0 * dx0
        value_01 = jointness[jtype_idx, py0l, px1l] * dy0 * dx1
        value_10 = jointness[jtype_idx, py1l, px0l] * dy1 * dx0
        value_11 = jointness[jtype_idx, py1l, px1l] * dy1 * dx1

        # out = value_00 + value_01 + value_10 + value_11
             
        return value_00 + value_01 + value_10 + value_11

    def jointness_bilinear_sampling_batch(self, jointness, px, py):
        device = jointness.device
        batch_size, _, height, width = jointness.shape 

        px0 = px.floor().clamp(min=0, max=width - 1)
        py0 = py.floor().clamp(min=0, max=height - 1)
        px1 = (px0 + 1).clamp(min=0, max=width - 1)
        py1 = (py0 + 1).clamp(min=0, max=height - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()
        jtype_idx = torch.arange(self.num_keypoints, device=device).reshape(1,-1,1,1,1).expand_as(px)
        batch_idx = torch.arange(batch_size, device=device).reshape(-1,1,1,1,1).expand_as(px)

        dy0 = (py1 - py).clamp(min=0, max=1.0)
        dy1 = (py - py0).clamp(min=0, max=1.0)
        dx0 = (px1 - px).clamp(min=0, max=1.0)
        dx1 = (px - px0).clamp(min=0, max=1.0)
        value_00 = jointness[batch_idx, jtype_idx, py0l, px0l] * dy0 * dx0
        value_01 = jointness[batch_idx, jtype_idx, py0l, px1l] * dy0 * dx1
        value_10 = jointness[batch_idx, jtype_idx, py1l, px0l] * dy1 * dx0
        value_11 = jointness[batch_idx, jtype_idx, py1l, px1l] * dy1 * dx1

        return value_00 + value_01 + value_10 + value_11

    def pose_excitation(self, center_poses_xy, stride=1):
        ksize = self.ksize
        batch_size = center_poses_xy.shape[0]
        
        dy, dx = torch.meshgrid(
            torch.linspace(-ksize, ksize, 2 * ksize + 1, device='cuda'),
            torch.linspace(-ksize, ksize, 2 * ksize + 1, device='cuda'))
        dy = dy.reshape(1, 1, 2 * ksize + 1, 2 * ksize + 1)
        dx = dx.reshape(1, 1, 2 * ksize + 1, 2 * ksize + 1)
        dy = dy.repeat((batch_size, self.topk_center, self.num_keypoints, 1, 1)) 
        dx = dx.repeat((batch_size, self.topk_center, self.num_keypoints, 1, 1)) 

        # dy = dy.repeat((self.num_keypoints, self.topk_center, 1, 1))
        # dx = dx.repeat((self.num_keypoints, self.topk_center, 1, 1))
        dxy = torch.stack((dx, dy), dim=-1)  #*2
        sigma = torch.tensor(self.person_sigma, device='cuda').float()
        scale = sigma / sigma.min()
        # scale = scale.reshape(-1, 1, 1, 1, 1)
        scale = scale.reshape(1,1,-1,1,1,1)
        dxy *= scale
        allposes = center_poses_xy[:,:,:,None,None] + dxy
        allposes = allposes.permute(0,1,3,4,2,5)
        return allposes

    def __call__(self, images, outputs, **kwargs):
        centermaps = outputs['centermaps']
        offsets = outputs['offsets']
        

        stride = centermaps.size(2) / offsets.size(2)

        if stride == 1:
            centers_xyv = self.center_nms(centermaps)
            centers_xyv_scaled = centers_xyv.clone()
            centers_xyv_scaled[:,:,:-1] *= 4
            stride = 4
        else:
            pool_kernel = 3
            # size = (centermaps.size(-1)+centermaps.size(-2))*0.5
            # if size > 300*2:
            #     pool_kernel = 7
            # elif size > 200*2:
            #     pool_kernel = 5
            # else:
            #     pool_kernel = 3
            centers_xyv_scaled = self.center_nms(centermaps,kernel_size=pool_kernel)
            centers_xyv = centers_xyv_scaled.clone()
            centers_xyv[:, :, :-1] /= stride

        batch_arange = torch.arange(centermaps.shape[0],device=centermaps.device).reshape(-1,1).repeat(1,self.topk_center)

        offsets_vectors = offsets[batch_arange,:,
                                  centers_xyv[...,1].long().clamp(min=0,max=offsets.shape[2]),
                                  centers_xyv[...,0].long().clamp(min=0,max=offsets.shape[3])]
        offsets_vectors = -offsets_vectors.reshape(centermaps.shape[0],self.topk_center,-1,2)
        center_poses_xy = (centers_xyv[:,:,None,:2] + offsets_vectors)*stride
        allposes = self.pose_excitation(center_poses_xy)

        return {
            'allposes': allposes,
            'centers': centers_xyv_scaled,
            # 'heatmaps': jointsmaps_hr if self.use_hr_joints else jointsmaps,
        }
    