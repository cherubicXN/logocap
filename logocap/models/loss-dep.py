import torch
import torch.nn as nn

class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss

class JointsMSELoss(torch.nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints

class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

class LossEvaluator(torch.nn.Module):
    def __init__(self, cfg):
        super(LossEvaluator, self).__init__()

        if cfg.LOSS.USE_OHKM:
            self.heatmap_loss = JointsOHKMMSELoss(cfg.LOSS.USE_TARGET_WEIGHT,cfg.LOSS.TOPK)
        else:
            self.heatmap_loss = JointsMSELoss(cfg.LOSS.USE_TARGET_WEIGHT)

        self.heatmap_loss_factor = cfg.LOSS_FACTORS.HEATMAP

        self.use_logocap = cfg.MODEL.USE_LOGOCAP

        if self.use_logocap:
            self.oks_loss_factor = cfg.LOSS_FACTORS.OKS
            self.ca_loss_factor = cfg.LOSS_FACTORS.CA
            self.loss_keys = ['loss.heatmaps','loss.oks','loss.ca']
        else:
            self.loss_keys = ['loss.heatmaps']

    def logo_loss(self, outputs, targets):
        oks_joints_pred = outputs['oks.joints']
        oks_joints_gt = targets['oks.joints']
        oks_person_gt = targets['oks.person']

        center = oks_joints_gt.shape[2]//2
        oks_weight = oks_joints_gt[:,:].max(dim=-1)[0].max(dim=-1)[0] - oks_joints_gt[:,:,center,center]
        oks_weight *= (oks_person_gt[:,None]>=0.5).float()
        
        loss_oks = ((oks_joints_pred-oks_joints_gt)**2)*oks_weight[:,:,None,None]
        
        loss_oks = loss_oks.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        loss_oks = torch.sum(loss_oks)/torch.sum((oks_person_gt>=0.5).float()).clamp_min(1)
        
        loss_outhm = self.heatmap_loss(outputs['out_hm'],targets['out_hm'],oks_weight[:,:,None,None])

        return loss_oks, loss_outhm

    def forward(self, outputs, targets):
        loss_heatmaps = self.heatmap_loss(outputs['heatmaps'],targets['heatmaps.t'],targets['heatmaps.w'])

        loss_dict = {
            'loss.heatmaps': loss_heatmaps*self.heatmap_loss_factor,
        }
        
        if self.use_logocap:
            loss_oks, loss_ca = self.logo_loss(outputs, targets)
            loss_dict['loss.oks'] = loss_oks*self.oks_loss_factor
            loss_dict['loss.ca'] = loss_ca*self.ca_loss_factor

        return loss_dict