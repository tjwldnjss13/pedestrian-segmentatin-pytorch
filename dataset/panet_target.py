import torch

from utils.pytorch_util import *


def generate_panet_target(bbox, mask_instance, mask_idx_per_roi):
    mask_idx_per_roi = mask_idx_per_roi.type(torch.long)

    device = bbox.device
    num_rois = len(mask_idx_per_roi)
    bbox = convert_box_from_yxyx_to_yxhw(bbox)

    target_box = torch.ones(num_rois, 5).to(device)
    target_box[..., 0:4] = bbox[mask_idx_per_roi]

    target_mask = mask_instance[mask_idx_per_roi].type(torch.float).to(device)

    return target_box, target_mask