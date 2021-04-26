
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (GIoULoss, distance2bbox, bbox2distance, bbox_overlaps)
from .head import ClsCntRegHead
from .loss import (coords_fmap2orig)
from .gfocal_loss import (QualityFocalLoss, DistributionFocalLoss)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    This layer calculates the target location by :math: `sum{P(y_i) * y_i}`,
    P(y_i) denotes the softmax vector that represents the discrete distribution
    y_i denotes the discrete set, usually {0, 1, 2, ..., reg_max}
    Args:
        reg_max (int): The maximal value of the discrete set. Default: 16. You
            may want to reset it according to your new dataset or related
            settings.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(
            0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        Args:
            x (Tensor): Features of the regression head, shape (N, 4*(n+1)),
                n is self.reg_max.
        Returns:
            x (Tensor): Integral result of box locations, i.e., distance
                offsets from the box center in four directions, shape (N, 4).
        """
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class GFLHead(ClsCntRegHead):

    def __init__(self, in_channel, class_num, score_threshold, nms_iou_threshold, max_detection_boxes_num,
                 GN=True, add_centerness=False, cnt_on_reg=True,
                 prior=0.01, use_asff=False, use_dcn=False, use_3d_maxf=False, use_gl=True, gl_cfg=None,
                 strides=[8, 16, 32, 64, 128], limit_range=[[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]],
                 ):
        super().__init__(in_channel, class_num, GN=GN, add_centerness=add_centerness, cnt_on_reg=cnt_on_reg,
                         prior=prior, use_asff=use_asff, use_dcn=use_dcn, use_3d_maxf=use_3d_maxf, use_gl=use_gl, gl_cfg=gl_cfg)

        self.strides = strides
        self.limit_range = limit_range
        self.distribution_project = Integral(self.gl_cfg.reg_max)
        self.loss_qfl = QualityFocalLoss(use_sigmoid=self.gl_cfg.loss_qfl.use_sigmoid,
                                         beta=self.gl_cfg.loss_qfl.beta,
                                         loss_weight=self.gl_cfg.loss_qfl.loss_weight)
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.gl_cfg.loss_dfl.loss_weight)
        self.loss_bbox = GIoULoss(
            loss_weight=self.gl_cfg.loss_bbox.loss_weight)

        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num

        assert add_centerness == False
        assert use_gl == True
        assert gl_cfg is not None
        assert len(strides) == len(limit_range)

    def loss(self, inputs):
        '''
        inputs  
        [0]list [cls_logits, reg_preds]
        cls_logits  list contains five [batch_size,class_num, h, w] * level
        reg_preds   list contains five [batch_size,4*(reg_max + 1), h, w]  * level
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes  [batch_size,m]  LongTensor
        Returns:
        loss_qfl, 
        loss_bbox, 
        loss_dfl, 
        total_loss
        '''
        cls_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        grid_center_all, cls_targets_all, reg_targets_all, label_weight_all = self.target_assign(
            cls_logits=cls_logits, reg_preds=reg_preds, gt_boxes=gt_boxes, classes=classes)

        batch_size = gt_boxes.shape[0]
        level_size = len(cls_logits)
        batch_loss = []
        bbox_targets = []
        for level in range(level_size):
            bbox_targets.append(self._coords2boxes(grid_center_all[level], reg_targets_all[level]))
        for batch in range(batch_size):
            num_pos = 0
            losses = []
            for level in range(level_size):
                losses.append(
                    self.loss_single(grid_center_all[level], cls_logits[level][batch], reg_preds[level][batch],
                                     cls_targets_all[level][batch], bbox_targets[level][batch],
                                     label_weight_all[level][batch], self.strides[level])
                )
                num_pos += losses[-1][-1]
            loss_qfl = torch.sum(torch.cat([loss[0].view(1) for loss in losses]), dim=0).view(1)
            loss_bbox = torch.sum(torch.cat([loss[1].view(1) for loss in losses]), dim=0).view(1)
            loss_dfl = torch.sum(torch.cat([loss[2].view(1) for loss in losses]), dim=0).view(1)
            if num_pos < 1:
                total_loss = loss_qfl
            else:
                total_loss = (loss_qfl + loss_bbox + loss_dfl) / num_pos
            batch_loss.append([loss_qfl, loss_bbox, loss_dfl, total_loss])

        loss_qfl = torch.cat([loss[0] for loss in batch_loss], dim=0)
        loss_bbox = torch.cat([loss[1] for loss in batch_loss], dim=0)
        loss_dfl = torch.cat([loss[2] for loss in batch_loss], dim=0)
        total_loss = torch.cat([loss[3] for loss in batch_loss], dim=0)

        return loss_qfl, loss_bbox, loss_dfl, total_loss

    def loss_single(self, grid_cell_centers, cls_score, bbox_pred, labels, bbox_targets, label_weights, stride):
        '''
        Args:
        grid_center [h * w, 2]
        cls_score  [class_num, h, w]
        bbox_pred   [4*(reg_max + 1), h, w]
        labels [h * w, 1]
        bbox_targets [h * w, 4]
        label_weight [h * w, 1]
        bbox_weight  [h * w, 4]
        Returns:
        '''
        h, w = cls_score.shape[-2], cls_score.shape[-1]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, cls_score.shape[-3])
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, bbox_pred.shape[-3])
        labels = labels.squeeze(dim=-1)
        pos_inds = torch.nonzero((labels >= 0) & (labels < self.class_num), as_tuple=False).squeeze(1)
        num_pos = torch.count_nonzero(pos_inds)
        score = labels.new_zeros(labels.shape).float()

        if num_pos > 0:

            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]  # (n, 4 * (reg_max + 1))
            pos_grid_cell_centers = grid_cell_centers[pos_inds]

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(
                pos_grid_cell_centers, pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride

            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach().float(),
                                            pos_decode_bbox_targets,
                                            mode='giou',
                                            is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.gl_cfg.reg_max + 1)
            target_corners = bbox2distance(pos_grid_cell_centers,
                                           pos_decode_bbox_targets,
                                           self.gl_cfg.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

        else:

            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).to(cls_score.device)

        # qfl loss
        loss_qfl = self.loss_qfl(cls_score, (labels, score), weight=label_weights)

        return loss_qfl.sum(), loss_bbox.sum(), loss_dfl.sum(), num_pos

    def target_assign(self, cls_logits, reg_preds, gt_boxes, classes):
        assert len(cls_logits) == len(self.strides)
        out = []
        for level in range(len(cls_logits)):
            out.append(self.target_assign_single_level(cls_logits[level], gt_boxes, classes, self.strides[level], self.limit_range[level]))
        grid_center_all = []
        cls_targets_all = []
        reg_targets_all = []
        label_weight_all = []
        for level_out in out:
            grid_center_all.append(level_out[0])
            cls_targets_all.append(level_out[1])
            reg_targets_all.append(level_out[2])
            label_weight_all.append(level_out[3])

        return grid_center_all, cls_targets_all, reg_targets_all, label_weight_all

    def target_assign_single_level(self, cls_logits, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args: \n
        cls_logits  [batch_size,class_num,h,w]
        gt_boxes [batch_size,m,4]  FloatTensor  
        classes  [batch_size,m]  LongTensor
        \n
        Returns:\n
        coords       [h * w, 2]
        cls_targets  [batch_size, h * w, 1]
        reg_targets  [batch_size, h * w, 4]
        label_weight [batch_size, h * w, 1]
        bbox_weight  [batch_size, h * w, 4]
        '''
        batch_size = cls_logits.shape[0]
        h, w = cls_logits.shape[2], cls_logits.shape[3]
        h_mul_w = h * w

        # [batch_size, h, w, class_num]
        cls_logits = cls_logits.permute(0, 2, 3, 1)

        # 生成特征图映射到原始图像中的坐标（x,y)
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)  # [h * w, 2]
        x = coords[:, 0]
        y = coords[:, 1]
        # [h*w,1] - [batch_size,1,m] --> [batch_size,h*w,m]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        # [batch_size,h*w,m,4]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)

        # [batch_size,h*w,m] (l+r) * (t+b)
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * \
            (ltrb_off[..., 1] + ltrb_off[..., 3])

        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        # 找到（l,r,t,b）中最小的，如果最小的大于０，那么这个点肯定在对应的gt框里面，则置１，否则为０
        mask_in_gtboxes = (off_min > 0)
        # 找到（l,r,t,b）中最大的，如果最大的满足范围约束，则置１，否则为０
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio  # 距离因子
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
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
        # [batch_size*h*w,4]
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]

        # to [batch_size, h*w, 4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))

        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]  # [batch_size,h*w,m]
        cls_targets = classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        # [batch_size, h * w]
        maks_pos = mask_pos.long().sum(dim=-1)
        pos_inds = (maks_pos >= 1)
        neg_inds = (maks_pos == 0)

        assert pos_inds.shape == (batch_size, h_mul_w)

        cls_targets[pos_inds] = cls_targets[pos_inds] - 1
        cls_targets[~pos_inds] = self.class_num
        reg_targets[~pos_inds] = -1

        label_weight = torch.zeros(cls_targets.shape, dtype=torch.float, device=cls_targets.device)

        label_weight[pos_inds] = 1.0
        label_weight[neg_inds] = 1.0

        return coords, cls_targets, reg_targets, label_weight

    def _coords2boxes(self, coords, offsets):
        '''
        Args:
        coords  [sum(_h*_w),2]
        offsets [batch_size, sum(_h*_w), 4] ltrb
        Returns:
        boxes [batch_size, sum(_h*_w), 4]
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        # [batch_size, sum(_h*_w), 2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]
        # [batch_size, sum(_h*_w), 4]
        boxes = torch.cat([x1y1, x2y2], dim=-1)
        return boxes

    def inference(self, inputs):
        '''
        Args:
        inputs  list [cls_logits, reg_preds]
        [0] cls_logits  list contains five [batch_size,class_num, h, w] * level
        [1] reg_preds   list contains five [batch_size,4*(reg_max + 1), h, w]  * level
        Returns:
        scores
        classes
        boxes
        '''
        cls_logits, reg_preds = inputs
        cls_preds, boxes = self._reshape_cat_out(cls_logits, reg_preds)
        cls_preds = cls_preds.sigmoid_()

        # tensor -> value,  tensor -> index
        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        # add one -> one-hot
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)             # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _reshape_cat_out(self, cls_logits, reg_preds):
        '''
        Args:
        cls_logits  list contains five [batch_size, class_num, h, w] * level
        reg_preds   list contains five [batch_size, 4*(reg_max + 1), h, w]  * level
        Returns:
        cls_preds   [batch_size, sum(_h * _w), class_num]
        boxes       [batch_size, sum(_h * _w), 4]
        '''
        batch_size = cls_logits[0].shape[0]
        class_num = cls_logits[0].shape[1]
        max_shape = (cls_logits[0].shape[-2], cls_logits[0].shape[-1])
        cls_scores = []
        bboxes = []

        for stride, cls_score, reg_pred in zip(self.strides, cls_logits, reg_preds):
            cls_score = cls_score.permute(0, 2, 3, 1)  # [batch_size, h, w, c]
            coord = coords_fmap2orig(cls_score, stride).to(device=cls_score.device)  # [h * w, 2]
            cls_score = torch.reshape(cls_score, [batch_size, -1, class_num])
            cls_scores.append(cls_score)
            boxes = []
            for batch in range(batch_size):
                bbox_pred = self.distribution_project(reg_pred[batch].permute(1, 2, 0)) * stride
                boxes.append(distance2bbox(coord, bbox_pred, max_shape))
                # boxes[-1].shape [h * w, 4]
            bboxes.append(boxes)

        out = []
        for batch in range(batch_size):
            b_out = []
            for level in range(len(self.strides)):
                b_out.append(bboxes[level][batch])
            out.append(torch.vstack(b_out))

        return torch.cat(cls_scores, dim=1), torch.stack(out, dim=0)

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            # 分数阙值，论文中为0.05
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            # nsm处理
            nms_ind = self.batched_nms(
                _boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(
            _cls_classes_post, dim=0), torch.stack(_boxes_post, dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2-x1+1)*(y2-y1+1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax-xmin).clamp(min=0)*(ymax-ymin).clamp(min=0)
            iou = inter/(areas[i]+areas[order[1:]]-inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx+1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):
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
