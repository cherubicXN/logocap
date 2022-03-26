class Encoder(object):
    def __init__(self, cfg):
        pass

    def __call__(self, images_batch, targets_batch, joints_batch, areas_batch, meta_batch):
        images_batch = images_batch.cuda()
        for k in targets_batch.keys():
            targets_batch[k] = targets_batch[k].cuda()
        
        
        targets_batch['meta'] = meta_batch
        return images_batch, targets_batch