import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig


def count_parameters(model):
    """
    计算PyTorch模型的总参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


pointNeXt_cfg = "./pointNext_m.yaml"
cfg = EasyConfig()
cfg.load(pointNeXt_cfg, recursive=True)
pointNeXt = build_model_from_cfg(cfg.model).cuda()

x = torch.randn(2, 1024, 3).cuda()
c = torch.randn(2, 3, 1024).cuda()

y = pointNeXt(x, c)
print(y.shape)
print(count_parameters(pointNeXt))
