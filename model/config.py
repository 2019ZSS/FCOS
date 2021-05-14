class DilatedEncoderConfig(object):
        use_dcn = False
        in_channels = 2048
        encoder_channels = 256
        block_mid_channels = 128
        num_residual_blocks = 4
        block_dilations = [2, 4, 6, 8]
        norm_type = 'BN'
        act_type = 'ReLU'
        conv_type = 'CNN'


class TransformerConfig(object):
        cls_prediction_min = -2.0
        cls_prediction_max = 2.0
        reg_prediction_min = -2.0
        reg_prediction_max = 2.0
        uncertainty_cls_weight = 0.5
        uncertainty_reg_weight = 0.5
        uncertainty_embedding_dim = 64
        use_iou = True 
        use_cnt = True


class QFLConfig(object):
        name = 'QualityFocalLoss'
        use_sigmoid = True
        beta = 2.0
        loss_weight = 1.0


class DFConfig(object):
        name = 'DistributionFocalLoss'
        loss_weight = 0.25


class GIOUConfig(object):
        loss_weight = 2.0


class GFLConfig(object):
        reg_max = 8
        loss_qfl = QFLConfig()
        loss_dfl = DFConfig()
        loss_bbox = GIOUConfig()


class DefaultConfig():
        #backbone
        backbone_name='resnet50'
        backbone_in_channls = [512, 1024, 2048]
        pretrained=True
        out_stages=(2, 4, 6)
        freeze_stage_1=True
        freeze_bn=True

        #fpn
        fpn_out_channels=256
        use_p5=True
        use_simo=False
        use_dcn_in=True
        use_dcn_out=True
        use_ircnn=False
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
        use_3d_maxf=False
        use_gl=False
        gl_cfg=GFLConfig()

        #training
        strides=[8,16,32,64,128]
        limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]
        is_generate_weight=True
        cnt_loss_mode='bce'
        reg_loss_mode='giou'
        transformer_cfg=TransformerConfig()

        #inference
        score_threshold=0.05
        nms_iou_threshold=0.6
        max_detection_boxes_num=1000


