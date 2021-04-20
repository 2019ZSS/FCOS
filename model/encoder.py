from typing import List

import torch
import torch.nn as nn

from .utils import (get_norm, get_activation, c2_xavier_fill)
try:
    from .DCNv2 import DeformableConv2DLayer as DeformConv2d
except Exception as e:
    print(e)
    from .deform_conv_v2 import DeformConv2d


class DilatedEncoder(nn.Module):
    """
    Dilated Encoder Layer.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """
    def __init__(self, cfg):
        super(DilatedEncoder, self).__init__()
        self.in_channels = cfg.in_channels
        self.encoder_channels = cfg.encoder_channels
        self.block_mid_channels = cfg.block_mid_channels
        self.num_residual_blocks = cfg.num_residual_blocks
        self.block_dilations = cfg.block_dilations
        self.norm_type = cfg.norm_type
        self.act_type = cfg.act_type
        self.conv_type = cfg.conv_type

        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()


    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels, self.encoder_channels, kernel_size=1)
        self.lateral_norm = get_norm(self.norm_type, self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels, 
                                    self.encoder_channels,
                                    kernel_size=3,
                                    padding=1)
        self.fpn_norm = get_norm(self.norm_type, self.encoder_channels)
        encoder_blocks = []
        for i in  range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation,
                    norm_type=self.norm_type,
                    act_type=self.act_type,
                    conv_type=self.conv_type,
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def _init_weight(self):
        c2_xavier_fill(self.lateral_conv)
        c2_xavier_fill(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class Bottleneck(nn.Module):

    def __init__(self,
                in_channels: int = 512,
                mid_channels: int = 128,
                dilation: int = 1,
                norm_type: str = 'BN',
                act_type: str = 'ReLU',
                conv_type: str = 'CNN'):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            get_norm(norm_type, mid_channels),
            get_activation(act_type)
        )
        if conv_type == 'CNN':
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation),
                get_norm(norm_type, mid_channels),
                get_activation(act_type)
            )
        elif conv_type == 'DCN':
            self.conv2 = nn.Sequential(
                DeformConv2d(mid_channels, mid_channels, kernel_size=3, padding=dilation, dilation=dilation),
                get_norm(norm_type, mid_channels),
                get_activation(act_type)
            )
        else:
            return NotImplementedError(conv_type)
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            get_norm(norm_type, in_channels),
            get_activation(act_type)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


if __name__ == "__main__":
    print('test DilatedEncode')

    class DilatedEncoderConfig(object):
        in_channels = 2048
        encoder_channels = 256
        block_mid_channels = 128
        num_residual_blocks = 4
        block_dilations = [1, 2, 5, 1]
        norm_type = 'BN'
        act_type = 'ReLU'

    cfg = DilatedEncoderConfig()
    x = torch.rand((8, 2048, 32, 32))
    model = DilatedEncoder(cfg)
    y = model(x)
    print(y.shape)