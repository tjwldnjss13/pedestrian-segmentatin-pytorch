import torch
import torch.nn as nn

from model.rpn import *
from dataset.panet_target import *


def test_cross_entropy_loss(predict_box, target_box):
    loss_box = -(target_box * torch.log2(predict_box + 1e-9) + (1 - target_box) * torch.log2(1 - predict_box + 1e-9)) / len(predict_box)
    # loss_mask = -(target_mask * torch.log2(predict_mask + 1e-9) + (1 - target_mask) * torch.log2(1 - predict_mask + 1e-9)) / len(predict_mask)
    # loss = loss_box.sum() + loss_mask.sum()

    return loss_box.mean(), torch.Tensor([1]).cuda(), torch.Tensor([1]).cuda(), torch.Tensor([1]).cuda()


def rpn_loss(predict_reg, predict_cls, target_reg, target_cls):
    N_cls = 1
    N_reg = 28 ** 2
    lambda_reg = 1

    def smooth_l1_loss(predict, target):
        n = torch.abs(predict - target)
        cond = n < 1
        losses = torch.where(cond, .5 * n ** 2, n - .5)

        return losses.mean()

    loss_cls = F.binary_cross_entropy(predict_cls, target_cls) / N_cls
    loss_reg = lambda_reg * smooth_l1_loss(predict_reg, target_reg) / N_reg

    loss = loss_cls + loss_reg

    return loss


def panet_loss(predict_box, predict_mask, target_box, target_mask):
    loss_box = F.binary_cross_entropy(predict_box, target_box)
    loss_mask = F.binary_cross_entropy(predict_mask, target_mask)

    return loss_box, loss_mask


class PANetLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def cross_entropy_loss(self, predict, target):
        loss = -(target * torch.log2(predict + 1e-9) + (1 - target) * torch.log2(1 - predict + 1e-9))

        return loss.mean()

    def forward(self, predict_box, predict_mask, predict_reg_rpn, predict_cls_rpn,
                ground_truth_box, ground_truth_mask, anchor_box, anchor_label, ground_truth_per_anchor_idx):
        print('panet loss start')

        # for i in range(len(predict_box)):
        #     print(predict_box[i].detach().cpu().numpy(), target_box[i].detach().cpu().numpy())

        tar_box, tar_mask = generate_panet_target(ground_truth_box, ground_truth_mask, ground_truth_per_anchor_idx)
        loss_box = self.cross_entropy_loss(predict_box, tar_box)
        loss_mask = self.cross_entropy_loss(predict_mask, tar_mask)

        loss = loss_box + loss_mask

        print('panet loss end')

        return loss, loss_rpn, loss_box, loss_mask






























