
from .resnet import resnet50, resnet101, resnet152
from .efficientnet_lite import EfficientNetLite


def build_backbone(backbone_name, pretrained, out_stages=(2, 4, 6)):
    if backbone_name == 'resnet50':
        return resnet50(pretrained=pretrained, if_include_top=False)
    elif backbone_name == 'resnet101':
        return resnet101(pretrained=pretrained, if_include_top=False)
    elif backbone_name == 'resnet152':
        return resnet152(pretrained=pretrained, if_include_top=False)
    elif backbone_name in ('efficientnet_lite0', 'efficientnet_lite1', 'efficientnet_lite2', 'efficientnet_lite3', 'efficientnet_lite4'):
        return EfficientNetLite(backbone_name, out_stages=out_stages)
    else:
        raise NotImplementedError