
from torch._C import device
import torch.nn as nn
import torch
import math
import torch.nn.functional as F 
from .asff import ASFF
try:
    from .DCNv2 import DeformableConv2DLayer as DeformConv2d
except Exception as e:
    print(e)
    from .deform_conv_v2 import DeformConv2d


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01,use_asff=False, use_dcn=False, use_3d_maxf=False):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.cnt_on_reg=cnt_on_reg
        self.use_asff = use_asff
        self.use_dcn = use_dcn
        self.use_3d_maxf = use_3d_maxf

        cls_branch=[]
        reg_branch=[]

        for i in range(4):
            # 网络层数增加，这增加了网络的非线性表达能力同时不过多增加网络参数
            if self.use_dcn:
                cls_branch.append(DeformConv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=None))
            else: 
                cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            # BN 需要用到足够大的批大小（例如，每个工作站采用 32 的批量大小）。
            # 一个小批量会导致估算批统计不准确，减小 BN 的批大小会极大地增加模型错误率。
            # 将channel方向分group，然后每个group内做归一化，算(C//G)*H*W的均值， 
            # GN 把通道分为组，并计算每一组之内的均值和方差，以进行归一化。GN 的计算与批量大小无关，其精度也在各种批量大小下保持稳定
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))
            cls_branch.append(nn.ReLU(True))

            if self.use_dcn:
                reg_branch.append(DeformConv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=None))
            else:
                reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU(True))

        if self.use_asff:
            self.asffs = nn.ModuleList([ASFF(2 - i) for i in range(3)])

        if self.use_3d_maxf:
            self.maxf = MaxFiltering(in_channel, kernel_size=3, tau=2)

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits = nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred = nn.Conv2d(in_channel,4,kernel_size=3,padding=1)
        
        self.apply(self.init_conv_RandomNormal)
        
        nn.init.constant_(self.cls_logits.bias,-math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])
    
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self,inputs):
        '''inputs:[P3~P7]'''
        if self.use_3d_maxf:
            inputs = self.maxf(inputs)
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        for index,P in enumerate(inputs):
            if self.use_asff and index < 3:
                asff_P = self.asffs[index](inputs[2], inputs[1], inputs[0])
                cls_conv_out = self.cls_conv(asff_P)
                reg_conv_out = self.reg_conv(asff_P)
            else:
                cls_conv_out = self.cls_conv(P)
                reg_conv_out = self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits,cnt_logits,reg_preds


class MaxFiltering(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        # self.conv = nn.Conv2d(
        #     in_channels,
        #     in_channels,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1
        # )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2

    def forward(self, inputs):
        # features = []
        # for _, x in enumerate(inputs):
        #     features.append(self.conv(x))

        features = inputs
        outputs = []
        for l, x in enumerate(features):
            func = lambda f: F.interpolate(f, size=x.shape[2:], mode="bilinear")
            feature_3d = []
            for k in range(max(0, l - self.margin), min(len(features), l + self.margin + 1)):
                feature_3d.append(func(features[k]) if k != l else features[k])
            feature_3d = torch.stack(feature_3d, dim=2)
            max_pool = self.max_pool(feature_3d)[:, :, min(l, self.margin)]
            output = max_pool + inputs[l]
            outputs.append(self.nonlinear(self.norm(output)))
        return outputs
        
