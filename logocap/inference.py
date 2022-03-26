import torch
import numpy as np
from logocap.utils.transforms import get_final_preds

class LogoCapInference(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
    
    @staticmethod
    def cal_area_2_torch(v):
        w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
        h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
        return w * w + h * h

    @staticmethod
    def nms_core(cfg, pose_coord, heat_score):
        num_people, num_joints, _ = pose_coord.shape
        pose_area = LogoCapInference.cal_area_2_torch(pose_coord)[:,None].repeat(1,num_people*num_joints)
        pose_area = pose_area.reshape(num_people,num_people,num_joints)
        pose_diff = pose_coord[:, None, :, :] - pose_coord
        pose_diff.pow_(2)
        pose_dist = pose_diff.sum(3)
        pose_dist.sqrt_()
        pose_thre = cfg.TEST.INSTANCE_NMS.SCORE_THRESHOLD * torch.sqrt(pose_area)
        pose_dist = (pose_dist < pose_thre).sum(2)
        nms_pose = pose_dist > cfg.TEST.INSTANCE_NMS.NUM_JOINTS

        ignored_pose_inds = []
        keep_pose_inds = []

        for i in range(nms_pose.shape[0]):
            if i in ignored_pose_inds:
                continue

            keep_inds = nms_pose[i].nonzero().cpu().numpy()
            keep_inds = [list(kind)[0] for kind in keep_inds]
            keep_scores = heat_score[keep_inds]
            if len(keep_inds) == 0:
                continue
            # print(len(keep_inds))
            # import pdb; pdb.set_trace()
            ind = torch.argmax(keep_scores)
            keep_ind = keep_inds[ind]
            if keep_ind in ignored_pose_inds:
                continue
            keep_pose_inds += [keep_ind]
            ignored_pose_inds += list(set(keep_inds)-set(ignored_pose_inds))
        return keep_pose_inds

    def __call__(self, image_tensor, rgb, outputs, meta_info, **kwargs):
        center, scale = meta_info['center'], meta_info['scale']
        poses = outputs['poses']
        if poses is None:
            return None, None
        
        poses = poses.cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        final_poses = get_final_preds([poses], center, scale, [image_tensor.size(-1),image_tensor.size(-2)])

        inds = self.nms_core(self.cfg, torch.from_numpy(final_poses[...,:-1]),torch.from_numpy(scores))
        inds = np.array(inds)

        final_poses = final_poses[inds]
        scores = scores[inds]


        if final_poses.shape[0] > self.cfg.DATASET.MAX_NUM_PEOPLE:
            _, ind = torch.tensor(scores).topk(k=self.cfg.DATASET.MAX_NUM_PEOPLE)
            ind = ind.numpy()
            final_poses = final_poses[ind]
            scores = scores[ind]
            # import pdb; pdb.set_trace()

        # print(final_poses.shape[0], final_poses.shape[0]>30)

        return final_poses, scores