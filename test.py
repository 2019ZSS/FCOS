
from functools import partial
import torch
import torch.nn as nn
from torch.nn import init 
import torch.nn.functional as F 
import numpy as np
from torch.nn.modules.activation import ReLU


class MaxFiltering(nn.Module):

    def __init__(self, in_channels: int, kernel_size: int = 3, tau: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.ReLU()
        self.max_pool = nn.MaxPool3d(
            kernel_size=(tau + 1, kernel_size, kernel_size),
            padding=(tau // 2, kernel_size // 2, kernel_size // 2),
            stride=1
        )
        self.margin = tau // 2

    def forward(self, inputs):
        features = []
        for l, x in enumerate(inputs):
            features.append(self.conv(x))

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


def cpu_soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1
 
        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]
 
        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts
 
        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]
 
        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]
 
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua # iou between max box and detection box
 
                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1
 
                    boxes[pos, 4] = weight*boxes[pos, 4]
 
            # if box score falls below threshold, discard the box by swapping with last box
            # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1
 
            pos = pos + 1
 
    keep = [i for i in range(N)]
    return keep


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.001, cuda=0):
    """
    Build a pytorch implement of Soft NMS algorithm.
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """

    # Indexes concatenate boxes with the last column
    N = dets.shape[0]
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
        xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
        yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
        xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())
        
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep


class UncertaintyWithLossFeature(nn.Module):

    def __init__(self, class_num, visual_feature_dim, embedding_dim=64):
        super().__init__()
        self.class_num = class_num
        self.visual_feature_dim = visual_feature_dim
        self.embedding_dim = embedding_dim

        self.cls_loss_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.reg_loss_net = nn.Sequential(
            nn.Linear(in_features=4, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.prob_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.cnt_net = nn.Sequential(
            nn.Linear(in_features=1, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim),
            nn.ReLU()
        )
        self.predictor = nn.Sequential(
            nn.Linear(in_features=self.embedding_dim * 4, out_features=self.embedding_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.embedding_dim, out_features=2)
        )
        self.init_weight()

    def init_weight(self):
        for m in [self.cls_loss_net, self.reg_loss_net, self.prob_net, self.cnt_net, self.predictor]:
            nn.init.normal_(m[0].weight, mean=0.0, std=0.0001)
            nn.init.constant_(m[0].bias, 0)
            nn.init.normal_(m[2].weight, mean=0.0, std=0.0001)
            nn.init.constant_(m[2].bias, 0)
    
    def forward(self, cls_loss, reg_loss, probs, cnt_loss):
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
        cnt_feature = self.cnt_net(cnt_loss)
        non_visual_input = torch.cat((cls_loss_feature, reg_loss_feature, probs_feature, cnt_feature), dim=-1)
        return self.predictor(non_visual_input)


class TransformerConfig(object):
    cls_prediction_min = -2.0
    cls_prediction_max = 2.0
    reg_prediction_min = -2.0
    reg_prediction_max = 2.0
    uncertainty_cls_weight = 0.5
    uncertainty_reg_weight = 0.5
    uncertainty_embedding_dim = 64

if __name__ == "__main__":
    print('test')
    # input = [torch.rand(4, 256, 256 // (2 ** i), 256 // (2 ** i)) for i in range(0, 5)]
    # for x in input:
    #     print(x.shape)
    # max_filter = MaxFiltering(in_channels=256, kernel_size=3, tau=2)
    # output = max_filter(input)
    # for y in output:
    #     print(y.shape)
    # cls_loss = torch.rand((4, 6 * 8, 1))
    # reg_loss = torch.rand((4, 6 * 8, 4))
    # cnt_loss = torch.rand((4, 6 * 8, 1))
    # label_weight = torch.rand((4, 6 * 8, 1))
    # bbox_weight = torch.rand((4, 6 * 8, 4))
    # probs =  cls_loss.softmax(dim=1)
    # model = UncertaintyWithLossFeature(class_num=20, visual_feature_dim=1024)
    # y = model(cls_loss, reg_loss, probs, cnt_loss)
    # print(y.shape)
    # print(y[:, :, :1].shape)
    # print(label_weight.shape, y[0].shape)
    # print((label_weight * y[:, :, :1]).shape)
    # print(bbox_weight.shape, y[:, :, 1:2].shape)
    # print((bbox_weight * y[:, :, 1:2]).shape)
    
    # from model.loss import compute_cls_loss, compute_reg_loss, GenTargets
    # level = 5
    # h, w = (8, 8)
    # preds = (torch.rand((level, 4, 20, h, w)) * 21).long()
    # targets = torch.rand((4, level * h * w, 1)) * 21
    # mask = torch.rand((4, level * h * w)) * 21
    # weight = torch.rand((4, level *h * w, 1))
    # compute_cls_loss(preds, targets, mask, weight)

    # preds = torch.rand((level, 4, 4, h, w)) * 10
    # targets = torch.rand((4, level * h * w, 4)) * 10 
    # mask = torch.rand((4, level * h * w)).bool()
    # weight = torch.rand((4, level *h * w, 4))
    # weight = None
    # compute_reg_loss(preds, targets, mask, weight, mode='giou')

    # strides = [8, 16, 32, 64, 128]
    # limit_range = [[-1,64],[64,128],[128,256],[256,512],[512,999999]]
    # level = len(strides)

    # batch_size = 4
    # class_num = 20
    # h, w = 8, 8
    # cls_logits = torch.rand((level, batch_size, class_num, h, w))
    # cnt_logits = torch.rand((level, batch_size, 1, h, w)) 
    # reg_logits = torch.rand((level, batch_size, 4, h, w)) * 16
    # out = [cls_logits, cnt_logits, reg_logits]
    # m = 5
    # gt_boxes = (torch.rand((batch_size, m, 4)) * 16).float()
    # classes = (torch.rand((batch_size, m)) * 20).long()

    # gen_targent = GenTargets(strides, limit_range)
    # output = gen_targent([out, gt_boxes, classes])
    # for y in output:
    #     print(y.shape) 
    # zero = torch.zeros(output[0].shape, dtype=torch.float, device=output[0].device)
    # print(zero.shape)

    from model.config import DefaultConfig
    from model.gfocal_head import GFLHead

    config = DefaultConfig()
    in_channel = 64
    gfl_head = GFLHead(in_channel=in_channel, class_num=config.class_num,
                        score_threshold=config.score_threshold, nms_iou_threshold=config.nms_iou_threshold,
                        max_detection_boxes_num=config.max_detection_boxes_num,
                        use_gl=True, gl_cfg=config.gl_cfg, 
                        strides=config.strides, limit_range=config.limit_range)
    
    batch_size = 4
    h, w = 64, 64
    inputs = [torch.rand(batch_size, in_channel, h // (2 ** i), w // (2 ** i)) for i in range(5)]
    cls_logits, reg_preds = gfl_head(inputs)
    print(len(cls_logits))
    print(len(reg_preds))
    for preds in reg_preds:
        print(preds.shape)
    
    m = 10
    gt_boxes = torch.rand((batch_size, m, 4)) * 16
    classes = (torch.rand((batch_size, m)) * (config.class_num)).long()

    losses = gfl_head.loss(inputs=[[cls_logits, reg_preds], gt_boxes, classes])

    for loss in losses:
        print(loss.shape)
    
    scores, classes, boxes = gfl_head.inference([cls_logits, reg_preds])
    print('inference')
    print(scores.shape)
    print(classes.shape)
    print(boxes.shape)

    
