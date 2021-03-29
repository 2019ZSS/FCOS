class DilatedEncoderConfig(object):
        in_channels = 448
        encoder_channels = 256
        block_mid_channels = 128
        num_residual_blocks = 4
        block_dilations = [1, 2, 5, 1]
        norm_type = 'BN'
        act_type = 'ReLU'


class DefaultConfig():
    #backbone
    backbone_name='efficientnet_lite4'
    pretrained=True
    out_stages=(6,)
    freeze_stage_1=True
    freeze_bn=True

    #fpn
    fpn_out_channels=256
    use_p5=True
    use_simo=True
    encoder_cfg = DilatedEncoderConfig()
    backbone_level_used = 2

    #head
    class_num=20
    use_GN_head=True
    prior=0.01
    add_centerness=True
    cnt_on_reg=True
    use_asff=False
    use_dcn=False
    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000


