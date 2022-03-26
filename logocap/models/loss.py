import torch


class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt, mask):
        assert pred.size() == gt.size()
        loss = ((pred - gt)**2) * mask
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        return loss

class OffsetsLoss(torch.nn.Module):
    def __init__(self):
        super(OffsetsLoss, self).__init__()

    def smooth_l1_loss(self, pred, gt, beta=1. / 9):
        l1_loss = torch.abs(pred - gt)
        cond = l1_loss < beta
        loss = torch.where(cond, 0.5*l1_loss**2/beta, l1_loss-0.5*beta)
        return loss

    def forward(self, pred, gt, weights):
        assert pred.size() == gt.size()
        num_pos = torch.nonzero(weights > 0).size()[0]
        loss = self.smooth_l1_loss(pred, gt) * weights
        if num_pos == 0:
            num_pos = 1.
        loss = loss.sum() / num_pos
        return loss


class LossEvaluator(torch.nn.Module):
    def __init__(self, cfg):
        super(LossEvaluator, self).__init__()
        self.heatmap_loss = HeatmapLoss()
        self.offsets_loss = OffsetsLoss()
        self.heatmap_loss_factor = cfg.LOSS_FACTORS.HEATMAP
        self.regression_loss_factor = cfg.LOSS_FACTORS.REGRESSION
        self.oks_loss_factor = cfg.LOSS_FACTORS.OKS
        self.hm_loss_factor = cfg.LOSS_FACTORS.get('HM',1.0)
        self.loss_keys = ['loss.heatmaps','loss.offsets','loss.oks','loss.hm']


    def forward(self, outputs, targets):
        loss_heatmaps = self.heatmap_loss(outputs['heatmaps'],targets['heatmaps.t'],targets['heatmaps.w'])
        loss_offsets = self.offsets_loss(outputs['offsets'],targets['offsets.t'],targets['offsets.w'])

        oks_joints_pred = outputs['oks.joints']
        oks_joints_gt = targets['oks.joints']
        oks_person_gt = targets['oks.person']
        oks_person_gt = oks_person_gt.max(dim=-1)[0].max(dim=-1)[0]

        center = oks_joints_gt.shape[2]//2
        oks_weight = oks_joints_gt[:,:].max(dim=-1)[0].max(dim=-1)[0] - oks_joints_gt[:,:,center,center]
        oks_weight *= (oks_person_gt[:,None]>=0.5).float()
        
        loss_oks = ((oks_joints_pred-oks_joints_gt)**2)*oks_weight[:,:,None,None]
        
        loss_oks = loss_oks.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        loss_oks = torch.sum(loss_oks)/torch.sum((oks_person_gt>=0.5).float()).clamp_min(1)

        if len(outputs['out_hm'])>0:
            loss_outhm = self.heatmap_loss(outputs['out_hm'],targets['out_hm'],oks_weight[:,:,None,None])
        else:
            loss_outhm = 0.0
        
        loss_dict = {
            'loss.heatmaps': loss_heatmaps*self.heatmap_loss_factor,
            'loss.offsets': loss_offsets*self.regression_loss_factor,
            'loss.oks': loss_oks*self.oks_loss_factor,
            'loss.hm': loss_outhm*self.hm_loss_factor,
        }

        return loss_dict