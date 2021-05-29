import torch

from utils.pytorch_util import *


def generate_panet_target_2(ground_truth_box, ground_truth_mask, ground_truth_idx):
    N_gt = len(ground_truth_box)

    print(ground_truth_box)

    rois = convert_box_from_yxhw_to_yxyx(rois)

    for i in range(N_gt):
        if i == 0:
            ious_roi_gt = calculate_ious(rois, ground_truth_box[i]).unsqueeze(-1)
        else:
            ious_roi_gt = torch.cat([ious_roi_gt, calculate_ious(rois, ground_truth_box[i]).unsqueeze(-1)], dim=-1)

    is_pos_ious = ious_roi_gt > .5
    print(ious_roi_gt)
    exit()

    return ious_roi_gt


def generate_panet_target(bbox, mask_instance, mask_idx_per_roi):
    mask_idx_per_roi = mask_idx_per_roi.type(torch.long)

    device = bbox.device
    num_rois = len(mask_idx_per_roi)
    bbox = convert_box_from_yxyx_to_yxhw(bbox)

    target_box = torch.ones(num_rois, 5).to(device)
    target_box[..., 0:4] = bbox[mask_idx_per_roi]

    target_mask = mask_instance[mask_idx_per_roi].to(device)

    return target_box, target_mask