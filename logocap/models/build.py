from .backbone import get_backbone
from .model import Model
from .encoder import Encoder

def build_model(cfg, is_train = True, **kwargs):
    backbone = get_backbone(cfg, is_train, **kwargs)
    model = Model(cfg, backbone)
    encoder = Encoder(cfg)

    return model, encoder
