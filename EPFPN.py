# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig, OptConfigType
from .EPFusion import EPFusion


@MODELS.register_module()
class EPFPN(BaseModule):

    def __init__(self,
                 backbone_channel: int,
                 in_channels: List[int],
                 out_channels: int,
                 num_outs: int,
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 act_cfg: OptConfigType = None,
                 init_cfg: MultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2,
                               self.backbone_channel // 4, 2, 2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel,
                               self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)

        self.fpn_conv = ConvModule(
            out_channels,
            out_channels,
            3,
            padding=1
        )
            
        self.epf = EPFusion(out_channels)

    def forward(self, input: Tensor) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        inputs = []
        inputs.append(self.fpn1(input)) 
        inputs.append(self.fpn2(input))
        inputs.append(self.fpn3(input))
        inputs.append(self.fpn4(input))
        '''
        inputs[0] : [1, 192, 160, 160]
        inputs[2] : [1, 384, 80, 80]
        inputs[3] : [1, 768, 40, 40]
        inputs[4] : [1, 768, 20, 20]
        '''

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        '''
        laterals[0] : [1, 256, 160, 160]  
        laterals[1] : [1, 256, 80, 80]   
        laterals[2] : [1, 256, 40, 40]    
        laterals[3] : [1, 256, 20, 20]    
        '''
        
        # build outputs using EPFusion
        outs = []
        outs.append(self.fpn_conv(laterals[0]))
        outs.append(self.epf([ laterals[0], laterals[1] ]))
        outs.append(self.epf([ laterals[1], laterals[2] ]))
        outs.append(self.epf([ laterals[2], laterals[3] ]))   

        if self.num_outs > len(outs):
            extra_out = outs[-1]
            for i in range(self.num_outs - len(outs)):
                extra_out = self.epf([ laterals[3], F.max_pool2d(extra_out, 1, stride=2) ])
                outs.append(extra_out)
#        for i, out in enumerate(outs):
#            print(f"outs[{i}] shape: {out.shape}")
        '''
        outs[0] : [1, 256, 160, 160]
        outs[1] : [1, 256, 80, 80]
        outs[2] : [1, 256, 40, 40]
        outs[3] : [1, 256, 20, 20]
        outs[4] : [1, 256, 10, 10]
        '''              

        return tuple(outs)
