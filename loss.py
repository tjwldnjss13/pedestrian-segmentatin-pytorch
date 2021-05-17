import torch
import torch.nn as nn

from model.rpn import *


class RPNLoss(nn.Module):
    def __init__(self, N_cls, N_reg, lambda_reg):
        super(RPNLoss, self).__init__()
        self.N_cls = N_cls
        self.N_reg = N_reg
        self.lambda_reg = lambda_reg
        self.box_reg_mod = BoxRegModule()

    def cross_entropy_loss(self, predict, target):
        loss_temp = -(target * torch.log2(predict + 1e-9) + (1 - target) * torch.log2(1 - predict + 1e-9))

        return loss_temp.sum()

    def smooth_l1_loss(self, predict, target):
        n = torch.abs(predict - target)
        cond = n < 1
        losses = torch.where(cond, .5 * n ** 2, n - .5)

        return losses

    def custom_reg_loss(self, predict_reg, target_reg):
        loss_temp = self.smooth_l1_loss(predict_reg, target_reg)

        return loss_temp.sum()

    def forward(self, predict_reg, predict_cls, anchor_box, anchor_label, ground_truth_per_anchor_box):
        pred_reg = predict_reg
        pred_cls = predict_cls
        anc_box = anchor_box
        anc_label = anchor_label
        gt_per_anc = ground_truth_per_anchor_box

        gt_per_anc /= 28

        # print('pred_reg : ', pred_reg)
        # print('gt_per_anc : ', gt_per_anc)

        self.N_cls = pred_cls.shape[0]

        pred_reg = self.box_reg_mod(pred_reg, anc_box)
        gt_reg = self.box_reg_mod(gt_per_anc, anc_box)

        # print('pred_reg : ', pred_reg)
        # print('gt_reg : ', gt_reg)

        N_pos = anc_label.sum().int()
        pos_pred_reg = pred_reg[:N_pos]
        pos_gt_reg = gt_reg[:N_pos]

        loss_cls = self.cross_entropy_loss(pred_cls, anc_label) / self.N_cls
        loss_reg = self.lambda_reg * self.custom_reg_loss(pos_pred_reg, pos_gt_reg) / self.N_reg

        loss = loss_cls + loss_reg

        return loss



