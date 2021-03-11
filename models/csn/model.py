import warnings

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import BasicStem, BasicBlock, Bottleneck

from models.blocks_3D import Conv3DDepthwise, BasicStem_Pool, IPConv3DDepthwise
from models.csn.utils import _generic_resnet


__all__ = ["ir_csn_152", "ip_csn_152"]


class csn_152_(nn.Module):
    def __init__(self, name, pretraining="", use_pool1=True,
                 classes=2, progress=False):
        super(csn_152_, self).__init__()

        avail_pretrainings = [
            "ig65m_32frms",
            "ig_ft_kinetics_32frms",
            "sports1m_32frms",
            "sports1m_ft_kinetics_32frms",
        ]
        if pretraining in avail_pretrainings:
            arch = name + pretraining
            pretrained = True

            if pretraining == "ig65m_32frms":
                pretrained_classes = 359
        else:
            arch = name
            pretrained = False
            pretrained_classes = classes

        model = _generic_resnet(
            arch,
            pretrained,
            progress,
            block=Bottleneck,
            conv_makers=[Conv3DDepthwise
                         if name == "ir_csn_152_" else IPConv3DDepthwise] * 4,
            layers=[3, 8, 36, 3],
            stem=BasicStem_Pool if use_pool1 else BasicStem,
            num_classes=pretrained_classes)
        self.model = model
        self.model.fc = nn.Linear(2048, classes)


    def forward(self, x):
        # change forward here
        x = self.model(x)
        return x
