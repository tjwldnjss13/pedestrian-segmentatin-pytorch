import time
import numpy as np
import torch
import torch.nn as nn

from model.backbone import *
from utils.pytorch_util import *

# For test
import cv2 as cv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


class RPN(nn.Module):
    def __init__(self, in_channel, feature_size):
        super(RPN, self).__init__()
        if isinstance(feature_size, int):
            self.size = (feature_size, feature_size)
        else:
            self.size = feature_size

        self.backbone = Backbone()
        # self.anchors = np.array([[1.73145, 1.3221], [4.00944, 3.19275], [8.09892, 5.05587], [4.84053, 9.47112], [10.0071, 11.2364]])
        self.anchors = self._make_anchor()
        self.anc_boxes = self._make_anchor_box().cuda()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True)
        )
        self.conv_reg = nn.Sequential(
            nn.Conv2d(in_channel, 4 * len(self.anchors), 1),
            nn.Sigmoid()
        )
        self.conv_cls = nn.Sequential(
            nn.Conv2d(in_channel, len(self.anchors), 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_anchor(self):
        scale = [4., 8., 16.]
        ratio = [.5, 1, 2]

        anc = np.zeros((len(scale) * len(ratio), 2))
        i = 0
        for s in scale:
            for r in ratio:
                h = s / np.sqrt(r)
                w = s * np.sqrt(r)
                anc[i] = [h, w]
                i += 1

        return anc

    def _make_anchor_center(self):
        y = torch.linspace(0, self.size[0] - 1, self.size[0]).reshape(self.size[0], 1).repeat(1, self.size[1]).reshape(self.size) + .5
        x = torch.linspace(0, self.size[0] - 1, self.size[0]).repeat(self.size[0], 1) + .5

        cen = torch.cat([y.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)

        return cen

    def _make_anchor_box(self):
        anc_centers = self._make_anchor_center()
        anc_boxes = torch.zeros(*self.size, 4 * len(self.anchors))
        for i in range(len(self.anchors)):
            anc_boxes[..., i * 4] = anc_centers[..., 0]
            anc_boxes[..., i * 4 + 1] = anc_centers[..., 1]
            anc_boxes[..., i * 4 + 2] = self.anchors[i, 0]
            anc_boxes[..., i * 4 + 3] = self.anchors[i, 1]

        anc_boxes = convert_box_from_yxhw_to_yxyx(anc_boxes.reshape(-1, 4))
        # valid_idx = (anc_boxes[..., 0] > 0) & (anc_boxes[..., 1] > 0) & (anc_boxes[..., 2] < self.size[0]) & (anc_boxes[..., 3] < self.size[1])
        # anc_boxes = anc_boxes[valid_idx]

        return anc_boxes

    def _calculate_ious_anchor_ground_truth(self, ground_truth):
        # t_s = time.time()
        # print(f'rpn._calculate_ious_anchor_ground_truth() start')

        N_anc = len(self.anc_boxes)
        N_gt = len(ground_truth)

        #######################################################################################
        # ious_anc_gt = torch.zeros(N_anc, N_gt)
        #
        # for i in range(N_anc):
        #     for j in range(N_gt):
        #         ious_anc_gt[i, j] = calculate_iou(self.anc_boxes[i], ground_truth[j])
        #######################################################################################

        for i in range(N_gt):
            if i == 0:
                ious_anc_gt = calculate_ious(self.anc_boxes, ground_truth[i]).unsqueeze(-1)
            else:
                ious_anc_gt = torch.cat([ious_anc_gt, calculate_ious(self.anc_boxes, ground_truth[i]).unsqueeze(-1)], dim=-1)

        # t_e = time.time()
        # print(f'rpn._calculate_ious_anchor_ground_truth() end : {t_e - t_s:.3f}')

        return ious_anc_gt

    def test(self, ground_truth):
        self._sample_anchor_box(ground_truth)


    def _sample_anchor_box(self, ground_truth):
        device = ground_truth.device

        # t_s = time.time()
        # print(f'rpn._sample_anchor_box() start')

        ious_anc_gt = self._calculate_ious_anchor_ground_truth(ground_truth)
        val_max_iou_anc_gt = torch.max(ious_anc_gt, dim=0).values
        idx_max_iou_anc_gt = torch.max(ious_anc_gt, dim=0).indices

        pos_iou_idx = (ious_anc_gt > .5)
        for i in range(len(ground_truth)):
            pos_iou_idx[idx_max_iou_anc_gt[i], i] = 1

        pos_anc = torch.cat([*[self.anc_boxes[(pos_iou_idx[..., i] == 1)] for i in range(len(ground_truth))]], dim=0)
        neg_anc = torch.cat([*[self.anc_boxes[(pos_iou_idx[..., i] == 0)] for i in range(len(ground_truth))]], dim=0)
        anc_gt_match = torch.cat(
            [*[torch.Tensor([i for _ in range(pos_iou_idx[..., i].sum())]) for i in range(len(ground_truth))]], dim=0)

        # N_neg_anc = 256 - len(pos_anc)
        # neg_anc_idx = torch.randperm(len(neg_anc))[:N_neg_anc].to(device)
        # neg_anc = neg_anc[neg_anc_idx]

        pos_anc_idx = torch.cat([*[torch.where(pos_iou_idx[..., i] == 1)[0] for i in range(len(ground_truth))]], dim=0)
        neg_anc_idx = torch.cat([*[torch.where(pos_iou_idx[..., i] == 0)[0] for i in range(len(ground_truth))]], dim=0)
        anc_idx = torch.cat([pos_anc_idx, neg_anc_idx], dim=0)

        anc_box = torch.cat([pos_anc, neg_anc], dim=0)
        anc_label = torch.cat([torch.ones(len(pos_anc)), torch.zeros(len(neg_anc))], dim=0)
        anc_gt_match = torch.cat([anc_gt_match, torch.zeros(len(neg_anc))], dim=0)
        gt_per_anc = ground_truth[anc_gt_match.long()]

        anc_box = convert_box_from_yxyx_to_yxhw(anc_box)

        anc_box = anc_box.cuda()
        anc_label = anc_label.cuda()
        gt_per_anc = gt_per_anc.cuda()
        anc_idx = anc_idx.cuda()

        # Normalize anchor box coordinates
        anc_box /= 28

        # t_e = time.time()
        # print(f'rpn._sample_anchor_box() end : {t_e - t_s:.3f}')

        return anc_box, anc_label, gt_per_anc, anc_idx

    def forward(self, x, ground_truth=None):
        # ground truth는 (28, 28) feature size에 맞게 yxyx로 조정되어져 있음

        # t_s = time.time()
        # print('rpn.forward() start')

        _, _, _, x = self.backbone(x)

        x = self.conv(x)
        reg = self.conv_reg(x)
        cls = self.conv_cls(x)

        reg = reg.reshape(reg.shape[0], 4, -1).permute(0, 2, 1)
        cls = cls.reshape(cls.shape[0], -1)

        if ground_truth is not None:
            anc_box, anc_label, gt_per_anc, anc_idx = self._sample_anchor_box(ground_truth)
            reg = reg[0, anc_idx]
            cls = cls[0, anc_idx]

            return reg, cls, anc_box, anc_label, gt_per_anc

        else:
            return reg, cls


class BoxRegModule(nn.Module):
    def __init__(self):
        super(BoxRegModule, self).__init__()

    def forward(self, box, anchor_box):
        # t_s = time.time()
        # print(f'BoxRegModule() start')

        cy, cx, h, w = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        cy_anc, cx_anc, h_anc, w_anc = anchor_box[..., 0], anchor_box[..., 1], anchor_box[..., 2], anchor_box[..., 3]

        cy_t, cx_t = (cy - cy_anc) / h_anc, (cx - cx_anc) / w_anc
        h_t, w_t = torch.log(h / h_anc + 1e-9), torch.log(w / w_anc + 1e-9)

        cy_t, cx_t, h_t, w_t = cy_t.unsqueeze(-1), cx_t.unsqueeze(-1), h_t.unsqueeze(-1), w_t.unsqueeze(-1)

        # t_e = time.time()
        # print(f'BoxRegModule() start : {t_e - t_s:.3f}')

        return torch.cat([cy_t, cx_t, h_t, w_t], dim=-1)


if __name__ == '__main__':
    import cv2 as cv
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open('../sample/FudanPed00007.png')
    img = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])(img).unsqueeze(0).cuda()

    gt = torch.Tensor([[69, 112, 346, 218], [76, 378, 377, 529], [108, 317, 192, 347]]).cuda()
    size = (381, 539)
    in_size = 448
    feat_size = 28

    gt_feat = gt / in_size * feat_size

    rpn = RPN(256, feat_size).cuda()
    rpn.test(gt_feat)


    # For visualization
    # for i, box in enumerate(anc_box[idx_max_iou_anc_gt]):
    #     img = cv.putText(img, f'{i+1}, {val_max_iou_anc_gt[i]:.2f}', (box[1], box[0]), cv.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), thickness=3)
    #     img = cv.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 0, 255), thickness=3)
    #
    # i = 0
    # for box in anc_box:
    #     img_temp = cv.rectangle(img.copy(), (box[1], box[0]), (box[3], box[2]), (0, 255, 0), thickness=2)
    #
    #     # print(f'{i+1} of {len(anc_box)}')
    #     if i == 1197:
    #         print('anc_box', anc_box[i])
    #         print('gt', gt[2])
    #         print('ious_anc_gt', ious_anc_gt[i, 2])
    #         plt.imshow(img_temp)
    #         plt.show()
    #
    #     i += 1

    # 891




























