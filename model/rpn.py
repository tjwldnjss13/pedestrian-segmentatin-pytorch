import time
import numpy as np
import torch
import torch.nn as nn

from model.backbone import *
from utils.pytorch_util import *
from loss import *

# For test
import cv2 as cv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image


class RPN(nn.Module):
    def __init__(self, in_channels, device=torch.device('cuda:0')):
        super(RPN, self).__init__()
        self.device = device
        self.train_rpn_only = False

        # self.backbone = Backbone()
        self.anc_boxes, self.valid_idx = self._make_anchor_box()
        self.roi_cls_threshold = .9

        self.conv = Conv(in_channels, in_channels, 3, 1, 1)
        self.conv_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4 * 9, 1),
            nn.Sigmoid()
        )
        self.conv_cls = nn.Sequential(
            nn.Conv2d(in_channels, 9, 1),
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
        scale = [64, 128, 256]
        ratio = [.5, 1, 2]

        anc = np.zeros((len(scale) * len(ratio), 2))
        i = 0
        for s in scale:
            for r in ratio:
                h = s / np.sqrt(r)
                w = s * np.sqrt(r)
                anc[i] = [h, w]
                i += 1

        anc /= 448

        return anc

    def _make_anchor_center(self):
        y = torch.linspace(0, 27, 28).view(28, 1).repeat(1, 28).view(28, 28) + .5
        x = torch.linspace(0, 27, 28).repeat(28, 1) + .5

        cen = torch.cat([y.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)
        cen /= 28

        return cen

    def _make_anchor_box(self):
        ancs = self._make_anchor()
        anc_centers = self._make_anchor_center()
        anc_boxes = torch.zeros(28, 28, 4 * 9)
        for i in range(9):
            anc_boxes[..., i * 4] = anc_centers[..., 0]
            anc_boxes[..., i * 4 + 1] = anc_centers[..., 1]
            anc_boxes[..., i * 4 + 2] = ancs[i, 0]
            anc_boxes[..., i * 4 + 3] = ancs[i, 1]

        anc_boxes = convert_box_from_yxhw_to_yxyx(anc_boxes.view(-1, 4))
        valid_idx = (anc_boxes[..., 0] > 0) & (anc_boxes[..., 1] > 0) & (anc_boxes[..., 2] < 28) & (anc_boxes[..., 3] < 28)
        anc_boxes = anc_boxes[valid_idx]

        return anc_boxes, valid_idx

    def _calculate_ious_anchor_ground_truth(self, ground_truth):
        # t_s = time.time()
        # print(f'rpn._calculate_ious_anchor_ground_truth() start')

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
                # ious_anc_gt = calculate_ious(self.anc_boxes, ground_truth[i]).unsqueeze(-1)
                ious_anc_gt = calculate_ious(self.anc_boxes.to(self.device), ground_truth[i]).unsqueeze(-1)
            else:
                # ious_anc_gt = torch.cat([ious_anc_gt, calculate_ious(self.anc_boxes, ground_truth[i]).unsqueeze(-1)], dim=-1)
                ious_anc_gt = torch.cat([ious_anc_gt, calculate_ious(self.anc_boxes.to(self.device), ground_truth[i]).unsqueeze(-1)], dim=-1)

        # t_e = time.time()
        # print(f'rpn._calculate_ious_anchor_ground_truth() end : {t_e - t_s:.3f}')

        return ious_anc_gt

    def _sample_anchor_box(self, ground_truth):

        ious_anc_gt = self._calculate_ious_anchor_ground_truth(ground_truth)
        pos_iou_idx = ious_anc_gt > .5
        for i in range(len(ground_truth)):
            if torch.sum(pos_iou_idx[..., i]) == 0:
                max_idx = torch.where(ious_anc_gt[..., i] == torch.max(ious_anc_gt[..., i]))[0]
                pos_iou_idx[max_idx] = False
                pos_iou_idx[max_idx, i] = True

        error_idx = torch.where(torch.sum(pos_iou_idx, dim=-1) > 1)[0]
        if len(error_idx) > 0:
            _, max_idx = torch.max(ious_anc_gt[error_idx], -1)
            pos_iou_idx_error = torch.zeros(pos_iou_idx[error_idx].shape).type(torch.bool).to(self.device)
            idx1 = torch.linspace(0, len(max_idx) - 1, len(max_idx)).type(torch.long)
            idx = (idx1, max_idx)
            pos_iou_idx_error[idx] = True
            pos_iou_idx[error_idx] = pos_iou_idx_error

        pos_iou_idx_global = torch.sum(pos_iou_idx, dim=-1).type(torch.bool)

        pos_anc_idx = torch.where(pos_iou_idx_global == True)
        neg_anc_idx = torch.where(pos_iou_idx_global == False)
        N_pos = len(pos_anc_idx[0])
        N_neg = len(neg_anc_idx[0])
        neg_idx_range = torch.randperm(N_neg)[:256-N_pos]
        anc_idx = torch.cat([pos_anc_idx[0], neg_anc_idx[0][neg_idx_range]])
        anc_cls_label = torch.cat([torch.ones(N_pos), torch.zeros(256-N_pos)])

        gt_per_anc_idx = torch.where(pos_iou_idx[pos_anc_idx[0]] == 1)[-1]

        anc_cls_label = anc_cls_label.to(self.device)
        gt_per_anc_idx = gt_per_anc_idx.to(self.device)

        return anc_idx, anc_cls_label, gt_per_anc_idx

    def _parametrize_box(self, box, anchor_box):
        cy, cx, h, w = box[..., 0], box[..., 1], box[..., 2], box[..., 3]
        cy_anc, cx_anc, h_anc, w_anc = anchor_box[..., 0], anchor_box[..., 1], anchor_box[..., 2], anchor_box[..., 3]

        cy_t, cx_t = (cy - cy_anc) / h_anc, (cx - cx_anc) / w_anc
        h_t, w_t = torch.log(h / h_anc + 1e-9), torch.log(w / w_anc + 1e-9)

        cy_t, cx_t, h_t, w_t = cy_t.unsqueeze(-1), cx_t.unsqueeze(-1), h_t.unsqueeze(-1), w_t.unsqueeze(-1)

        # t_e = time.time()
        # print(f'BoxRegModule() start : {t_e - t_s:.3f}')

        return torch.cat([cy_t, cx_t, h_t, w_t], dim=-1)

    def forward(self, x, gt=None):
        # ground truth는 (28, 28) feature size에 맞게 yxyx로 조정되어져 있음
        # reg layer의 roi들은 (28, 28) feature size에 대해 normalize 되어져 있음 (0~1)

        # t_s = time.time()
        # print('rpn.forward() start')

        if self.train_rpn_only:
            _, _, _, x = self.backbone(x)

        x = self.conv(x)
        reg = self.conv_reg(x)
        cls = self.conv_cls(x)

        reg = reg.view(reg.shape[0], 4, -1).permute(0, 2, 1).squeeze()[self.valid_idx]
        cls = cls.view(cls.shape[0], -1).squeeze()[self.valid_idx]

        if gt is not None:
            anc_idx, gt_cls, gt_per_anc_idx = self._sample_anchor_box(gt)
            rois = reg[anc_idx][gt_cls == 1]
            reg = self._parametrize_box(rois, self.anc_boxes[anc_idx][gt_cls == 1].to(self.device)).type(torch.float64)
            cls = cls[anc_idx]

            gt = convert_box_from_yxyx_to_yxhw(gt[gt_per_anc_idx])
            gt_reg = self._parametrize_box(gt, self.anc_boxes[anc_idx][gt_cls == 1].to(self.device))

            loss = rpn_loss(reg, cls, gt_reg, gt_cls)

            anc_box = self.anc_boxes[anc_idx][gt_cls == 1]

            return rois, loss, anc_box, gt_per_anc_idx
        else:
            reg = reg[0]
            cls = cls[0]

            cls_mask = cls > self.roi_cls_threshold
            reg = reg[cls_mask].view(-1, 4)

            return reg


if __name__ == '__main__':
    import cv2 as cv
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    from PIL import Image

    from torchsummary import summary
    model = RPN().cuda()
    model.train_rpn_only = True
    img = np.zeros((448, 448, 3))

    scale = [64, 128, 256]
    ratio = [.5, 1, 2]

    anc = np.zeros((len(scale) * len(ratio), 2))
    i = 0
    for s in scale:
        for r in ratio:
            h = s / np.sqrt(r)
            w = s * np.sqrt(r)
            anc[i] = [h, w]
            i += 1

    anc = np.array(anc / 2, dtype=np.int64)
    print(anc)
    cx, cy = 224, 224
    for i in range(len(anc)):
        pt1 = (int(cx - anc[i, 1]), int(cy - anc[i, 0]))
        pt2 = (int(cx + anc[i, 1]), int(cy + anc[i, 0]))
        img = cv.rectangle(img.copy(), pt1, pt2, (0, 255, 0), thickness=3)
    cv.circle(img, (cx, cy), 3, (255, 0, 0), -1)

    plt.imshow(img)
    plt.show()

    # img = Image.open('../sample/FudanPed00007.png')
    # img = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])(img).unsqueeze(0).cuda()
    #
    # gt = torch.Tensor([[69, 112, 346, 218], [76, 378, 377, 529], [108, 317, 192, 347]]).cuda()
    # size = (381, 539)
    # in_size = 448
    # feat_size = 28
    #
    # gt[..., 0] /= size[0]
    # gt[..., 1] /= size[1]
    # gt[..., 2] /= size[0]
    # gt[..., 3] /= size[1]
    #
    # rpn = RPN().cuda()
    # rpn.train_rpn_only = True
    # reg, cls, anc_box, anc_label, gt_per_anc, gt_idx_per_anc = rpn(img, gt)
    # print(gt_idx_per_anc)


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




























