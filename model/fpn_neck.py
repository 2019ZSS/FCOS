import torch.nn as nn
import torch.nn.functional as F
import math
from .encoder import DilatedEncoder
try:
    from .DCNv2 import DeformableConv2DLayer as DeformConv2d
except Exception as e:
    print(e)
    from .deform_conv_v2 import DeformConv2d
from .IRNN.irnn import IRCNN


class FPN(nn.Module):
    '''only for resnet50,101,152'''
    def __init__(self,backbone_in_channels=[512, 1024, 2048], features=256,use_p5=True, use_dcn_in=False, use_dcn_out=False):
        super(FPN,self).__init__()
        if not use_dcn_in:
            self.prj_5 = nn.Conv2d(backbone_in_channels[2], features, kernel_size=1) # 降维，方便后面特征融合
            self.prj_4 = nn.Conv2d(backbone_in_channels[1], features, kernel_size=1)
            self.prj_3 = nn.Conv2d(backbone_in_channels[0], features, kernel_size=1)
        else:
            self.prj_5 = DeformConv2d(backbone_in_channels[2], features, kernel_size=1) # 降维，方便后面特征融合
            self.prj_4 = DeformConv2d(backbone_in_channels[1], features, kernel_size=1)
            self.prj_3 = DeformConv2d(backbone_in_channels[0], features, kernel_size=1)
        if not use_dcn_out:
            self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            if use_p5:
                self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
            else:
                self.conv_out6 = nn.Conv2d(backbone_in_channels[-1], features, kernel_size=3, padding=1, stride=2)
            self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_5 = DeformConv2d(features, features, kernel_size=3, padding=1)
            self.conv_4 = DeformConv2d(features, features, kernel_size=3, padding=1)
            self.conv_3 = DeformConv2d(features, features, kernel_size=3, padding=1)
            if use_p5:
                self.conv_out6 = DeformConv2d(features, features, kernel_size=3, padding=1, stride=2)
            else:
                self.conv_out6 = DeformConv2d(backbone_in_channels[-1], features, kernel_size=3, padding=1, stride=2)
            self.conv_out7 = DeformConv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5=use_p5
        self.apply(self.init_conv_kaiming)
        
    def upsamplelike(self,inputs):
        src,target=inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                    mode='nearest')
    
    def init_conv_kaiming(self,module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,x):
        C3,C4,C5=x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)
        
        P4 = P4 + self.upsamplelike([P5,C4]) 
        P3 = P3 + self.upsamplelike([P4,C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)
        
        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3,P4,P5,P6,P7]


class SIMO(nn.Module):

    def __init__(self, encoder_cfg, backbone_level_used=2, features=256, use_dcn_out=False, use_ircnn=False):
        super(SIMO, self).__init__()
        # in_channels_list = [512, 1024, 2048]
        # assert backbone_level_used >= 0 and backbone_level_used <= len(in_channels_list)
        # encoder_cfg.in_channels = in_channels_list[backbone_level_used]
        self.use_dcn = encoder_cfg.use_dcn
        if self.use_dcn:
            self.dcn_conv = DeformConv2d(encoder_cfg.in_channels, encoder_cfg.in_channels, kernel_size=3, padding=1)
        encoder_cfg.encoder_channels = features
        self.backbone_level_used = backbone_level_used
        self.encoder = nn.Sequential(DilatedEncoder(encoder_cfg))
        self.use_dcn_out = use_dcn_out
        self.use_ircnn = use_ircnn

        if self.use_dcn_out:
            self.conv_3 = DeformConv2d(features, features, kernel_size=3, padding=1)
            self.conv_4 = DeformConv2d(features, features, kernel_size=3, padding=1)
            self.conv_out6 = DeformConv2d(features, features, kernel_size=3, padding=1, stride=2)
            self.conv_out7 = DeformConv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
            self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)

        if self.use_ircnn:
            self.ircnn = IRCNN(features, features)

        self.apply(self.init_conv_kaiming)

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        
        C = x[-1]
        if self.use_dcn:
            C = self.dcn_conv(C)

        P5 = self.encoder(C)

        if self.use_ircnn:
            context = self.ircnn(P5)
            P5 = context + P5

        P4 = F.interpolate(P5, scale_factor=2, mode='nearest')
        P3 = F.interpolate(P4, scale_factor=2, mode='nearest')
        
        P4 = self.conv_4(P4)
        P3 = self.conv_3(P3)

        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))

        return [P3, P4, P5, P6, P7]
        