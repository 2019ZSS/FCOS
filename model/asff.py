
from itertools import compress
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding 


def add_conv(in_ch: int, out_ch: int, ksize: int, stride: int, leaky=True, GN=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch, out_channels=out_ch, 
                                    kernel_size=ksize, stride=stride, 
                                    padding=pad, bias=False))
    if GN:
        stage.add_module('gruop_norm', nn.GroupNorm(16, out_ch))
    else:
        stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):

    def __init__(self, level, rfb=False, vis=False):
        '''
        Args:
        '''
        super(ASFF, self).__init__()
        self.dim = [256, 256, 256]
        assert level >= 0 and level < len(self.dim)
        self.level = level
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(256, self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(256, self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(256, self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, 256, 3, 1)
        
        compress_c = 8 if rfb else 16
        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
    
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis 
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            nn.init.constant_(module.weight, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = F.interpolate(x_level_1, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        
        level__0_weight_v = self.weight_level_0(level_0_resized)
        level__1_weight_v = self.weight_level_1(level_1_resized)
        level__2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level__0_weight_v, level__1_weight_v, level__2_weight_v), dim=1)
        levels_weight = F.softmax(self.weight_levels(levels_weight_v), dim=1)

        fused_out_reduced = (level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :])    

        out = self.expand(fused_out_reduced)
        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


if __name__ == "__main__":
    print('test ASFF module')
    x1 = torch.rand((4, 256, 25, 32))
    x2 = torch.rand((4, 256, 50, 64))
    x3 = torch.rand((4, 256, 100, 128))
    asff_0 = ASFF(0)
    y = asff_0(x1, x2, x3)
    print(y.shape)
    asff_1 = ASFF(1)
    y = asff_1(x1, x2, x3)
    print(y.shape)
    asff_2 = ASFF(2)
    y = asff_2(x1, x2, x3)
    print(y.shape)




