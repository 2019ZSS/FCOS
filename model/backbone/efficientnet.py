import torch
import torch.nn as nn 

from efficientnet_pytorch import EfficientNet as EffNet


class EfficientNet(nn.Module):
    """
    code source: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/efficientdet/model.py
    """

    def __init__(self, compound_coef, load_weights=False):
        super(EfficientNet, self).__init__()
        # model = EffNet.from_pretrained('efficientnet-b%s' % compound_coef, load_weights)
        model = EffNet.from_pretrained(compound_coef, load_weights)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model

    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[2:]


if __name__ == '__main__':
    # 0 - 7
    model_name = [('efficientnet-b%s' % i) for i in range(0, 8)]
    model = EfficientNet(compound_coef=model_name[4], load_weights=True)
    x = torch.rand((4, 3, 128, 128))
    y = model(x)
    for val in y:
        print(val.shape)
    
