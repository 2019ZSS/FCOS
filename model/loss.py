
from functools import reduce
import torch
import torch.nn as nn
from .config import DefaultConfig
from .utils import weighted_loss


class UncertaintyWithLossFeature(nn.Module):

    def __init__(self, class_num=20, embedding_dim=64):
        super().__init__()
        self.class_num = class_num
        self.embedding_dim = embedding_dim

        self.cls_loss_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim,
                      out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.reg_loss_net = nn.Sequential(
            nn.Linear(in_features=4, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim,
                      out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.prob_net = nn.Sequential(
            nn.Linear(in_features=self.class_num, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim,
                      out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.iou_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim,
                      out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim * 4,
                      out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=2)
        )
        self.init_weight()

    def init_weight(self):
        for m in [self.cls_loss_net, self.reg_loss_net, self.prob_net, self.iou_net, self.predictor]:
            nn.init.normal_(m[0].weight, mean=0.0, std=0.0001)
            nn.init.constant_(m[0].bias, 0)
            nn.init.normal_(m[2].weight, mean=0.0, std=0.0001)
            nn.init.constant_(m[2].bias, 0)

    def forward(self, cls_loss, reg_loss, probs, ious):
        '''
        Args:
        cls_loss:
        reg_loss:
        probs:
        cnt_loss:
        '''
        cls_loss_feature = self.cls_loss_net(cls_loss)
        reg_loss_feature = self.reg_loss_net(reg_loss)
        probs_feature = self.prob_net(probs)
        ious_feature = self.iou_net(ious)
        non_visual_input = torch.cat(
            (cls_loss_feature, reg_loss_feature, probs_feature, ious_feature), dim=-1)
        return self.predictor(non_visual_input)


def coords_fmap2orig(feature, stride):
    '''
    transfor one fmap coords to orig coords
    Args
    feature [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    生成特征图映射到原始图像中的坐标（x,y)
    '''
    h, w = feature.shape[1:3]
    # 创建从0开始 以原图宽度结束 步幅为缩放比例的列表
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


class GenTargets(nn.Module):

    def __init__(self, strides, limit_range, is_generate_weight=False):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        self.is_generate_weight = is_generate_weight
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [P3~P7,batch_size,class_num,h,w]  
        cnt_logits  list contains five [P3~P7,batch_size,1,h,w]  
        reg_preds   list contains five [P3~P7,batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        if self.is_generate_weight:
            label_weight_all_level = []
            bbox_weight_all_leval = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
            if self.is_generate_weight:
                label_weight_all_level.append(level_targets[3])
                bbox_weight_all_leval.append(level_targets[4])

        if self.is_generate_weight:
            return (torch.cat(cls_targets_all_level, dim=1),
                    torch.cat(cnt_targets_all_level, dim=1),
                    torch.cat(reg_targets_all_level, dim=1),
                    torch.cat(label_weight_all_level, dim=1),
                    torch.cat(bbox_weight_all_leval, dim=1))
        else:
            return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(reg_targets_all_level, dim=1)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        正负样本的生成
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]

        cls_logits = cls_logits.permute(
            0, 2, 3, 1)  # [batch_size,h,w,class_num]
        # 生成特征图映射到原始图像中的坐标（x,y)
        coords = coords_fmap2orig(cls_logits, stride).to(
            device=gt_boxes.device)  # [h*w,2]

        cls_logits = cls_logits.reshape(
            (batch_size, -1, class_num))  # [batch_size,h*w,class_num]
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)
        cnt_logits = cnt_logits.reshape(
            (batch_size, -1, 1))  # [batch_size,h * w, 1]
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape(
            (batch_size, -1, 4))  # [batch_size,h * w, 4]

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]
        # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        l_off = x[None, :, None]-gt_boxes[..., 0][:, None, :]
        t_off = y[None, :, None]-gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :]-x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :]-y[None, :, None]
        # [batch_size,h*w,m,4]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)

        # [batch_size,h*w,m] (l+r)*(t+b)
        areas = (ltrb_off[..., 0]+ltrb_off[..., 2]) * \
            (ltrb_off[..., 1]+ltrb_off[..., 3])

        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        # 找到（l,r,t,b）中最小的，如果最小的大于０，那么这个点肯定在对应的gt框里面，则置１，否则为０
        mask_in_gtboxes = off_min > 0
        # 找到（l,r,t,b）中最大的，如果最大的满足范围约束，则置１，否则为０
        mask_in_level = (off_max > limit_range[0]) & (
            off_max <= limit_range[1])

        radiu = stride*sample_radiu_ratio  # 距离因子
        gt_center_x = (gt_boxes[..., 0]+gt_boxes[..., 2])/2
        gt_center_y = (gt_boxes[..., 1]+gt_boxes[..., 3])/2
        # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_l_off = x[None, :, None]-gt_center_x[:, None, :]
        c_t_off = y[None, :, None]-gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :]-x[None, :, None]
        c_b_off = gt_center_y[:, None, :]-y[None, :, None]
        # [batch_size,h*w,m,4]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        mask_center = c_off_max < radiu

        # [batch_size,h*w,m]
        mask_pos = mask_in_gtboxes & mask_in_level & mask_center
        # 找到（l,r,t,b）中最大的，如果最大的满足范围约束，则置１，否则为０
        # 将不满足范围约束的也置为无穷，因为下面的代码要找最小的
        areas[~mask_pos] = 99999999  # 无穷远
        # 找到每个点对应的面积最小的gt框（因为可能有多个，论文取了最小的）
        areas_min_ind = torch.min(areas, dim=-1)[1]  # [batch_size,h*w]
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(
            -1, areas_min_ind.unsqueeze(dim=-1), 1)]  # [batch_size*h*w,4]
        reg_targets = torch.reshape(
            reg_targets, (batch_size, -1, 4))  # [batch_size,h*w,4]

        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[
            0]  # [batch_size,h*w,m]
        cls_targets = classes[torch.zeros_like(
            areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(
            cls_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]

        # center-ness
        left_right_min = torch.min(
            reg_targets[..., 0], reg_targets[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        # origin eps = 1e-10
        eps = 1e-6
        cnt_targets = ((left_right_min*top_bottom_min)/(left_right_max *
                       top_bottom_max+eps)).sqrt().unsqueeze(dim=-1)  # [batch_size,h*w,1]

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        mask_pos_2 = mask_pos.long().sum(dim=-1)  # [batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        if self.is_generate_weight:
            neg_inds = (mask_pos_2 < 1)

        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)

        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1] 负样本
        cnt_targets[~mask_pos_2] = -1
        reg_targets[~mask_pos_2] = -1

        if self.is_generate_weight:
            label_weight = torch.zeros(
                cls_targets.shape, dtype=torch.float, device=cls_targets.device)
            bbox_weight = torch.zeros(
                reg_targets.shape, dtype=torch.float, device=reg_targets.device)

            label_weight[mask_pos_2] = 1.0
            label_weight[neg_inds] = 1.0
            bbox_weight[mask_pos_2] = 1.0

        if self.is_generate_weight:
            return cls_logits, cnt_logits, reg_targets, label_weight, bbox_weight
        else:
            return cls_targets, cnt_targets, reg_targets


def compute_cls_loss(preds, targets, mask, weight=None):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(all_level*_h*_w),1]
    mask: [batch_size,sum(all_level*_h*_w)]
    weight: None or [batch_size, sum(all_level*_h*_w), 1]
    '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(
        min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)
    # [batch_size,sum(_h*_w),class_num]
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]
        target_pos = (torch.arange(1, class_num+1, device=target_pos.device)
                      [None, :] == target_pos).float()  # sparse-->onehot
        if weight is not None:
            loss.append(focal_loss_from_logits(
                pred_pos, target_pos, weight[batch_index]).view(1))
        else:
            loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
    return torch.cat(loss, dim=0)/num_pos  # [batch_size,]


def compute_cnt_loss(preds, targets, mask, mode='bce'):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(
        min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),1]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,]
        assert len(pred_pos.shape) == 1
        if mode == 'bce':
            loss.append(nn.functional.binary_cross_entropy_with_logits(
                input=pred_pos, target=target_pos, reduction='sum').view(1))
        elif mode == 'focal':
            loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError(
                "cnt loss only implemented ['iou','giou']")
    return torch.cat(loss, dim=0)/num_pos  # [batch_size,]


def compute_reg_loss(preds, targets, mask, weight=None, mode='giou'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(all_level*_h*_w), 4]
    mask: [batch_size,sum(all_level*_h*_w)]
    weight: None or [batch_size, sum(all_level*_h*_w), 4]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_pos.shape) == 2
        if weight is not None:
            if mode == 'smooth_l1':
                weight_pos = weight[batch_index][mask[batch_index]]
            else:
                weight_pos = weight[batch_index][mask[batch_index]].view(4, -1)
        else:
            weight_pos = None
        if mode == 'iou':
            loss.append(iou_loss(pred=pred_pos, target=target_pos,
                        weight=weight_pos, reduction='sum').view(1))
        elif mode == 'giou':
            loss.append(giou_loss(pred=pred_pos, target=target_pos,
                        weight=weight_pos, reduction='sum').view(1))
        elif mode == 'smooth_l1':
            loss.append(smooth_l1_loss(pred=pred_pos, target=target_pos,
                        weight=weight_pos, reduction='sum').view(1))
        else:
            raise NotImplementedError(
                "reg loss only implemented ['iou','giou', 'smooth_l1']")
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


@weighted_loss
def iou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt = torch.min(preds[:, :2], targets[:, :2])
    rb = torch.min(preds[:, 2:], targets[:, 2:])
    wh = (rb+lt).clamp(min=0)
    overlap = wh[:, 0]*wh[:, 1]  # [n]
    area1 = (preds[:, 2]+preds[:, 0])*(preds[:, 3]+preds[:, 1])
    area2 = (targets[:, 2]+targets[:, 0])*(targets[:, 3]+targets[:, 1])
    iou = overlap/(area1+area2-overlap)
    loss = -iou.clamp(min=1e-6).log()
    return loss


@weighted_loss
def giou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (rb_min+lt_min).clamp(min=0)
    overlap = wh_min[:, 0]*wh_min[:, 1]  # [n]
    area1 = (preds[:, 2]+preds[:, 0])*(preds[:, 3]+preds[:, 1])
    area2 = (targets[:, 2]+targets[:, 0])*(targets[:, 3]+targets[:, 1])
    union = (area1+area2-overlap)
    iou = overlap/union

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (rb_max+lt_max).clamp(0)
    G_area = wh_max[:, 0]*wh_max[:, 1]  # [n]

    giou = iou-(G_area-union)/G_area.clamp(1e-10)
    loss = 1.-giou
    return loss


@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff *
                       diff / beta, diff - 0.5 * beta)
    return loss


def focal_loss_from_logits(preds, targets, weight=None, gamma=2.0, alpha=0.25):
    '''
    Args:
    preds: [n, class_num] 
    targets: [n, class_num]
    '''
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha*targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    if weight is not None:
        loss = loss.sum(dim=1) * (weight.squeeze())
    return loss.sum()


class LOSS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config
        if hasattr(self.config, 'transformer_cfg'):
            self.transformer_cfg = self.config.transformer_cfg
            self.cls_prediction_min = self.transformer_cfg.cls_prediction_min
            self.cls_prediction_max = self.transformer_cfg.cls_prediction_max
            self.reg_prediction_min = self.transformer_cfg.reg_prediction_min
            self.reg_prediction_max = self.transformer_cfg.reg_prediction_max
            self.uncertainty_cls_weight = self.transformer_cfg.uncertainty_cls_weight
            self.uncertainty_reg_weight = self.transformer_cfg.uncertainty_reg_weight
            self.uncertainty_embedding_dim = self.transformer_cfg.uncertainty_embedding_dim
            self.uncertainty_predictor = UncertaintyWithLossFeature(class_num=self.config.class_num,
                                                                    embedding_dim=self.uncertainty_embedding_dim)

    def forward(self, inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : 
            list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
            or list contains five elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4], [batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        if len(targets) == 3:
            cls_targets, cnt_targets, reg_targets = targets
            label_weight = None
            bbox_weight = None
        else:
            cls_targets, cnt_targets, reg_targets, label_weight, bbox_weight = targets
        if hasattr(self.config, 'transformer_cfg'):
            label_weight, bbox_weight, _ = self.predict_weight(cls_targets, cnt_targets, reg_targets, label_weight, bbox_weight,
                                                            cls_logits, cnt_logits, reg_preds)
        # [batch_size,sum(_h*_w)]
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)
        if hasattr(self.config, 'transformer_cfg'):
            print('')
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos, weight=label_weight).mean()  # []
        cnt_loss = compute_cnt_loss(cnt_logits, cnt_targets, mask_pos, mode=self.config.cnt_loss_mode).mean()
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos, weight=bbox_weight, mode=self.config.reg_loss_mode).mean()
        if self.config.add_centerness:
            total_loss = cls_loss+cnt_loss+reg_loss
            return cls_loss, cnt_loss, reg_loss, total_loss
        else:
            total_loss = cls_loss+reg_loss+cnt_loss*0.0
            return cls_loss, cnt_loss, reg_loss, total_loss

    def predict_weight(self, cls_targets, cnt_targets, reg_targets, label_weight, bbox_weight, cls_logits, cnt_logits, reg_preds):
        
        mask = (cnt_targets > -1).squeeze(dim=-1)

        pos_inds = cls_targets > 0
        postive_score = cls_targets[pos_inds].sigmoid()
        total_scores = torch.zeros(cls_targets.shape, dtype=torch.float, device=cls_targets.device)
        total_scores[pos_inds] = postive_score
        
        def cat(preds, batch_size, c):
            preds_reshape = []
            for pred in preds:
                pred = pred.permute(0, 2, 3, 1)
                pred = torch.reshape(pred, [batch_size, -1, c])
                preds_reshape.append(pred)
            return torch.cat(preds_reshape, dim=1)
        
        batch_size = cls_targets.shape[0]

        cls_logits = cat(cls_logits, batch_size, cls_logits[0].shape[1])
        reg_preds = cat(reg_preds, batch_size, reg_targets.shape[-1])
        
        ious = []
        for batch_index in range(batch_size):
            iou = giou_loss(pred=reg_preds[batch_index], target=reg_targets[batch_index], reduction='none')
            iou[~mask[batch_index]] = 0.0
            ious.append(iou)

        ious = torch.stack(ious, dim=0).unsqueeze(dim=-1)
        uncertainty_prediction = self.uncertainty_predictor(
            cls_logits.sum(dim=-1).unsqueeze(dim=-1).detach().data,
            reg_preds.detach().data,
            total_scores,
            ious,
        )
        
        # losses = dict()

        uncertainty_prediction_cls = uncertainty_prediction[:, :, :1]
        uncertainty_prediction_reg = uncertainty_prediction[:, :, 1:2]
        uncertainty_prediction_cls = torch.clamp(uncertainty_prediction_cls, min=self.cls_prediction_min, max=self.cls_prediction_max)
        uncertainty_prediction_reg = torch.clamp(uncertainty_prediction_reg, min=self.reg_prediction_min, max=self.reg_prediction_max)
        uncertainty_prediction_cls = torch.ones_like(uncertainty_prediction_cls) * uncertainty_prediction_cls.mean()
        
        # losses.update({"loss_uncertainty_cls": uncertainty_prediction_cls.sum() / uncertainty_prediction_cls.numel() * self.uncertainty_cls_weight})
        # losses.update({"loss_uncertainty_reg": uncertainty_prediction_reg[pos_inds].mean() * self.uncertainty_reg_weight})
        
        uncertainty_prediction_reg = torch.exp(-1. * uncertainty_prediction_reg)
        uncertainty_prediction_cls = torch.exp(-1. * uncertainty_prediction_cls)
        
        # losses.update({
        #     "cls_prediction_pos": uncertainty_prediction_cls[pos_inds].mean(),
        #     "cls_prediction_neg": uncertainty_prediction_cls[~pos_inds].mean(),
        #     "cls_prediction_reg": uncertainty_prediction_reg[pos_inds].mean(),
        # })

        label_weight = label_weight.detach().data * uncertainty_prediction_cls
        bbox_weight = bbox_weight.detach().data * uncertainty_prediction_reg

        return label_weight, bbox_weight, None

if __name__ == "__main__":
    loss = compute_cnt_loss([torch.ones([2, 1, 4, 4])]*5, torch.ones(
        [2, 80, 1]), torch.ones([2, 80], dtype=torch.bool), mode='focal')
    print(loss)
