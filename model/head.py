
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
from .config import DefaultConfig
from .loss import coords_fmap2orig


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)


class ClsCntRegHead(nn.Module):
    def __init__(self, in_channel, class_num, GN=True, add_centerness=True, cnt_on_reg=True, prior=0.01, 
                use_asff=False, use_dcn=False, use_3d_maxf=False, use_gl=False, gl_cfg=None):
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
        self.add_centerness=add_centerness
        self.cnt_on_reg=cnt_on_reg
        self.use_asff = use_asff
        self.use_dcn = use_dcn
        self.use_3d_maxf = use_3d_maxf
        self.use_gl = use_gl
        self.gl_cfg = gl_cfg

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
        
        if self.add_centerness:
            self.cnt_logits = nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        
        if self.use_gl:
            self.reg_max = gl_cfg.reg_max
            self.reg_pred = nn.Conv2d(in_channel, 4 * (self.reg_max + 1), kernel_size=3, padding=1)
        else:
            self.reg_pred = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)
        
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
        if self.add_centerness:
            cnt_logits = []
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
            if self.add_centerness:
                if not self.cnt_on_reg:
                    cnt_logits.append(self.cnt_logits(cls_conv_out))
                else:
                    cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
            
        if self.add_centerness:
            return cls_logits, cnt_logits, reg_preds
        else:
            return cls_logits, reg_preds


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
        

class DetectHead(nn.Module):

    def __init__(self,score_threshold,nms_iou_threshold,max_detection_boxes_num,strides,config=None):
        super().__init__()
        self.score_threshold=score_threshold
        self.nms_iou_threshold=nms_iou_threshold
        self.max_detection_boxes_num=max_detection_boxes_num
        self.strides=strides
        if config is None:
            self.config=DefaultConfig
        else:
            self.config=config

    def forward(self,inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [P3~P7,batch_size,class_num,h,w]  
        cnt_logits  list contains five [P3~P7,batch_size,1,h,w]  
        reg_preds   list contains five [P3~P7,batch_size,4,h,w] 
        '''
        cls_logits,coords=self._reshape_cat_out(inputs[0],self.strides)#[batch_size,sum(_h*_w),class_num], [sum(_h*_w), 2]
        if self.config.add_centerness:
            cnt_logits, _ = self._reshape_cat_out(inputs[1],self.strides)#[batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[-1],self.strides)#[batch_size,sum(_h*_w),4]

        cls_preds=cls_logits.sigmoid_()
        if self.config.add_centerness:
            cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords
        
        # tensor -> value,  tensor -> index
        cls_scores,cls_classes=torch.max(cls_preds,dim=-1)#[batch_size,sum(_h*_w)]
        if self.config.add_centerness:
            # FCOS初版采用分类分数以及center-ness之积, 改进使用分类损失函数
            cls_scores = torch.sqrt(cls_scores*(cnt_preds.squeeze(dim=-1)))#[batch_size,sum(_h*_w)]
        # add one -> one-hot
        cls_classes=cls_classes+1#[batch_size,sum(_h*_w)]
        
        # 利用中心点坐标和四个方向的距离复原检测边框
        boxes=self._coords2boxes(coords,reg_preds)#[batch_size,sum(_h*_w),4]

        #select topk
        max_num=min(self.max_detection_boxes_num,cls_scores.shape[-1])
        topk_ind=torch.topk(cls_scores,max_num,dim=-1,largest=True,sorted=True)[1]#[batch_size,max_num]
        _cls_scores=[]
        _cls_classes=[]
        _boxes=[]
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])#[max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])#[max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])#[max_num,4]
        cls_scores_topk=torch.stack(_cls_scores,dim=0)#[batch_size,max_num]
        cls_classes_topk=torch.stack(_cls_classes,dim=0)#[batch_size,max_num]
        boxes_topk=torch.stack(_boxes,dim=0)#[batch_size,max_num,4]
        assert boxes_topk.shape[-1]==4
        return self._post_process([cls_scores_topk,cls_classes_topk,boxes_topk])

    def _post_process(self,preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post=[]
        _cls_classes_post=[]
        _boxes_post=[]
        cls_scores_topk,cls_classes_topk,boxes_topk=preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            # 分数阙值，论文中为0.05
            mask=cls_scores_topk[batch]>=self.score_threshold
            _cls_scores_b=cls_scores_topk[batch][mask]#[?]
            _cls_classes_b=cls_classes_topk[batch][mask]#[?]
            _boxes_b=boxes_topk[batch][mask]#[?,4]
            # nsm处理
            nms_ind=self.batched_nms(_boxes_b,_cls_scores_b,_cls_classes_b,self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores,classes,boxes = torch.stack(_cls_scores_post,dim=0),torch.stack(_cls_classes_post,dim=0),torch.stack(_boxes_post,dim=0)
        
        return scores,classes,boxes
    
    @staticmethod
    def box_nms(boxes,scores,thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0]==0:
            return torch.zeros(0,device=boxes.device).long()
        assert boxes.shape[-1]==4
        x1,y1,x2,y2=boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas=(x2-x1+1)*(y2-y1+1)
        order=scores.sort(0,descending=True)[1]
        keep=[]
        while order.numel()>0:
            if order.numel()==1:
                i=order.item()
                keep.append(i)
                break
            else:
                i=order[0].item()
                keep.append(i)
            
            xmin=x1[order[1:]].clamp(min=float(x1[i]))
            ymin=y1[order[1:]].clamp(min=float(y1[i]))
            xmax=x2[order[1:]].clamp(max=float(x2[i]))
            ymax=y2[order[1:]].clamp(max=float(y2[i]))
            inter=(xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou=inter/(areas[i]+areas[order[1:]]-inter)
            idx=(iou<=thr).nonzero().squeeze()
            if idx.numel()==0:
                break
            order=order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self,boxes, scores, idxs, iou_threshold):
        '''idx: class'''
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self,coords,offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1=coords[None,:,:]-offsets[...,:2]
        x2y2=coords[None,:,:]+offsets[...,2:]#[batch_size,sum(_h*_w),2]
        boxes=torch.cat([x1y1,x2y2],dim=-1)#[batch_size,sum(_h*_w),4]
        return boxes


    def _reshape_cat_out(self,inputs,strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size=inputs[0].shape[0]
        c=inputs[0].shape[1]
        out=[]
        coords=[]
        for pred,stride in zip(inputs,strides):
            pred=pred.permute(0,2,3,1) # [batch_size,h,w,c]
            coord=coords_fmap2orig(pred,stride).to(device=pred.device)
            pred=torch.reshape(pred,[batch_size,-1,c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out,dim=1),torch.cat(coords,dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes