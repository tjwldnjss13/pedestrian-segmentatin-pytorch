import torch
import numpy as np

from utils.pytorch_util import *


def make_anchor():
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


def make_anchor_center():
    y = torch.linspace(0, 27, 28).view(28, 1).repeat(1, 28).view(28, 28) + .5
    x = torch.linspace(0, 27, 28).repeat(28, 1) + .5

    cen = torch.cat([y.unsqueeze(-1), x.unsqueeze(-1)], dim=-1)
    cen /= 28

    return cen


def make_anchor_box():
    ancs = make_anchor()
    anc_centers = make_anchor_center()
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


def calculate_ious_anchor_ground_truth(anchor_box, ground_truth):
    N_gt = len(ground_truth)

    for i in range(N_gt):
        if i == 0:
            # ious_anc_gt = calculate_ious(self.anc_boxes, ground_truth[i]).unsqueeze(-1)
            ious_anc_gt = calculate_ious(anchor_box, ground_truth[i]).unsqueeze(-1)
        else:
            # ious_anc_gt = torch.cat([ious_anc_gt, calculate_ious(self.anc_boxes, ground_truth[i]).unsqueeze(-1)], dim=-1)
            ious_anc_gt = torch.cat([ious_anc_gt, calculate_ious(anchor_box, ground_truth[i]).unsqueeze(-1)], dim=-1)

    # t_e = time.time()
    # print(f'rpn._calculate_ious_anchor_ground_truth() end : {t_e - t_s:.3f}')

    return ious_anc_gt


def sample_anchor_box(anchor_box, ground_truth):
    ious_anc_gt = calculate_ious_anchor_ground_truth(anchor_box, ground_truth)
    pos_iou_idx = ious_anc_gt > .5
    for i in range(len(ground_truth)):
        if torch.sum(pos_iou_idx[..., i]) == 0:
            max_idx = torch.where(ious_anc_gt[..., i] == torch.max(ious_anc_gt[..., i]))[0]
            pos_iou_idx[max_idx] = False
            pos_iou_idx[max_idx, i] = True

    error_idx = torch.where(torch.sum(pos_iou_idx, dim=-1) > 1)[0]
    if len(error_idx) > 0:
        _, max_idx = torch.max(ious_anc_gt[error_idx], -1)
        pos_iou_idx_error = torch.zeros(pos_iou_idx[error_idx].shape).type(torch.bool)
        idx1 = torch.linspace(0, len(max_idx) - 1, len(max_idx)).type(torch.long)
        idx = (idx1, max_idx)
        pos_iou_idx_error[idx] = True
        pos_iou_idx[error_idx] = pos_iou_idx_error

    pos_iou_idx_global = torch.sum(pos_iou_idx, dim=-1).type(torch.bool)

    pos_anc_idx = torch.where(pos_iou_idx_global == True)
    neg_anc_idx = torch.where(pos_iou_idx_global == False)
    N_pos = len(pos_anc_idx[0])
    N_neg = len(neg_anc_idx[0])
    neg_idx_range = torch.randperm(N_neg)[:256 -N_pos]
    anc_idx = torch.cat([pos_anc_idx[0], neg_anc_idx[0][neg_idx_range]])
    anc_cls_label = torch.cat([torch.ones(N_pos), torch.zeros(256 -N_pos)])

    gt_per_anc_idx = torch.where(pos_iou_idx[pos_anc_idx[0]] == 1)[-1]

    # anc_idx = anc_idx.to(self.device)
    # anc_cls_label = anc_cls_label.to(self.device)
    # gt_per_anc_idx = gt_per_anc_idx.to(self.device)

    return anc_idx, anc_cls_label, gt_per_anc_idx