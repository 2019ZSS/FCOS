from .head import ClsCntRegHead, DetectHead, ClipBoxes
from .gfocal_head import GFLHead
from .fpn_neck import FPN, SIMO
from .backbone import build_backbone
import torch.nn as nn
from .loss import GenTargets, LOSS
from .config import DefaultConfig
from model import config
import torch


class FCOS(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = build_backbone(config.backbone_name, config.pretrained, config.out_stages)
        if config.use_simo:
            self.fpn = SIMO(encoder_cfg=config.encoder_cfg, 
                            backbone_level_used=config.backbone_level_used,
                            features=config.fpn_out_channels,
                            use_dcn_out=config.use_dcn_out)
        else:
            self.fpn = FPN(config.backbone_in_channls, 
                            config.fpn_out_channels,
                            use_p5=config.use_p5)
        if config.use_gl:
            self.head = GFLHead(in_channel=config.fpn_out_channels, class_num=config.class_num,
                                    score_threshold=config.score_threshold, nms_iou_threshold=config.nms_iou_threshold,
                                    max_detection_boxes_num=config.max_detection_boxes_num,
                                    GN=config.use_GN_head, add_centerness=config.add_centerness, 
                                    cnt_on_reg=config.cnt_on_reg, prior=config.prior, 
                                    use_asff=config.use_asff, use_dcn=config.use_dcn, 
                                    use_3d_maxf=config.use_3d_maxf, use_gl=config.use_gl, gl_cfg=config.gl_cfg)
        else:
            self.head = ClsCntRegHead(in_channel=config.fpn_out_channels, class_num=config.class_num,
                                        GN=config.use_GN_head, add_centerness=config.add_centerness, 
                                        cnt_on_reg=config.cnt_on_reg, prior=config.prior, 
                                        use_asff=config.use_asff, use_dcn=config.use_dcn, 
                                        use_3d_maxf=config.use_3d_maxf, use_gl=config.use_gl, gl_cfg=config.gl_cfg)
        self.config = config

    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1 and hasattr(self.backbone, 'freeze_stages'):
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self,x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        # C3,C4,C5=self.backbone(x)
        C = self.backbone(x)
        all_P = self.fpn(C)
        if self.config.add_centerness:
            cls_logits, cnt_logits, reg_preds = self.head(all_P)
            return [cls_logits, cnt_logits, reg_preds]
        else:
            cls_logits, reg_preds = self.head(all_P)
            return [cls_logits, reg_preds]

        
class FCOSDetector(nn.Module):

    def __init__(self, mode="training", config=None):
        super().__init__()
        if config is None:
            config=DefaultConfig
        self.config = config
        self.mode=mode
        self.fcos_body=FCOS(config=config)
        if mode=="training":
            if not self.config.use_gl:
                self.target_layer = GenTargets(strides=config.strides,
                                            limit_range=config.limit_range, 
                                            add_centerness=config.add_centerness,
                                            is_generate_weight=config.is_generate_weight)
                self.loss_layer = LOSS()
        elif mode=="inference":
            if not self.config.use_gl:
                self.detection_head = DetectHead(config.score_threshold,config.nms_iou_threshold,
                                                config.max_detection_boxes_num,config.strides,config)
            self.clip_boxes = ClipBoxes()
        
    
    def forward(self, inputs):
        '''
        inputs 
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''
        if self.mode=="training":

            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.fcos_body(batch_imgs)
            if self.config.use_gl:
                losses = self.fcos_body.head.loss([out,batch_boxes,batch_classes])
            else:
                targets= self.target_layer([out,batch_boxes,batch_classes])
                losses = self.loss_layer([out,targets])
            return losses

        elif self.mode=="inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            if self.config.use_gl:
                scores, classes, boxes = self.fcos_body.head.inference(out)
            else:
                scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs,boxes)
            return scores, classes, boxes



    


