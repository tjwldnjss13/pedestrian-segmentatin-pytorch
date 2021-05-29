import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cutorch

from torchvision.ops import nms

from model.conv import *
from model.rpn import *
from model.adaptive_feature_pooling import *
from utils.nms import *
from utils.util import *
from utils.panet_util import *
from loss import *


class FullyConnectedFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.roi_size = 14
        self.conv_layer_1 = nn.Sequential(
            *[Conv(in_channels, in_channels, 3, 1, 1) for _ in range(3)]
        )
        self.conv_layer_2 = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            Deconv(in_channels, in_channels, 3, 2, 1, 1),
            nn.Conv2d(in_channels, 1, 1)
        )
        self.conv_layer_fc = nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            Conv(in_channels, 1, 1),
        )
        self.fc_layer = nn.Linear(self.roi_size ** 2, (self.roi_size * 2) ** 2)
        # self.flat_layer = nn.Conv2d(self.roi_size ** 2, (self.roi_size * 2) ** 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_conv1 = self.conv_layer_1(x)
        x_conv2 = self.conv_layer_2(x_conv1)

        x_fc = self.conv_layer_fc(x_conv1)
        x_fc = x_fc.view(x_fc.shape[0], -1)
        x_fc = self.fc_layer(x_fc)
        x_fc = x_fc.reshape(x_fc.shape[0], 1, self.roi_size * 2, self.roi_size * 2)

        x = x_conv2 + x_fc
        x = self.sigmoid(x)

        return x


class BoxPredictModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.roi_size = 14
        self.conv1 = Conv(in_channels, in_channels, 3, 2, 1)
        self.box_branch = nn.Sequential(
            nn.Linear(in_channels * (self.roi_size // 2) ** 2, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.box_branch(x)

        return x


class PANet(nn.Module):
    def __init__(self, device=torch.device('cuda:0')):
        super(PANet, self).__init__()
        self.device = device
        self.roi_size = 14

        self.backbone = Backbone()
        self.rpn = RPN(512, device)
        # self.rpn.load_state_dict(torch.load('../pretrain/rpn_final.pth'))
        # self.afp = AdaptiveFeaturePooling()
        self.fcf = FullyConnectedFusion(512)
        self.box_branch = BoxPredictModule(512)
        self.threshold_nms = .7

        self.loss_func = PANetLoss()

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

    def forward(self, x, ground_truth_box=None, ground_truth_mask=None):
        """
        :param x: Tensor of image, [1, channels, height, width]
        :param ground_truth: Normalized Tensor, [num ground truth, (y1, x1, y2, x2)]
        :return:
        """
        # N2, N3, N4, N5 = self.backbone(x)
        N5 = self.backbone(x)

        train_f = ground_truth_box is not None or ground_truth_mask is not None

        if train_f:
            rois, loss_rpn, anc_box, gt_idx = self.rpn(N5, ground_truth_box)
            
            # 이거 때문에 backward 속도 개느림
            # rois_afp = self.afp([N2, N3, N4, N5], rois)
            # rois_afp = adaptive_feature_pooling([N2, N3, N4, N5], rois)
            rois_afp = adaptive_feature_pooling([N5], rois)

            pred_box = self.box_branch(rois_afp)
            pred_mask = self.fcf(rois_afp)

            tar_box, tar_mask = generate_panet_target(ground_truth_box, ground_truth_mask, gt_idx)
            loss_box, loss_mask = panet_loss(pred_box, pred_mask, tar_box, tar_mask)

            return pred_box, pred_mask, loss_box, loss_mask, loss_rpn

        else:
            reg_rpn = self.rpn(N5)
            reg_rpn = reg_rpn.squeeze()
            rois_afp = self.afp([N5], reg_rpn)
            pred_box = []
            pred_mask = []
            for i, roi in enumerate(rois_afp):
                # print(f'{i + 1} / {len(rois_afp)}')
                pred_box.append(self.box_branch(roi.view(-1)).unsqueeze(0))
                pred_mask.append(self.fcf(roi.unsqueeze(0)).squeeze().unsqueeze(0))
            pred_box = torch.cat(pred_box, dim=0)
            pred_mask = torch.cat(pred_mask, dim=0)

            return pred_box, pred_mask


        # idx_nms = nms(reg_rpn, cls_rpn, self.threshold_nms)
        # reg_rpn = reg_rpn[idx_nms]
        # cls_rpn = cls_rpn[idx_nms]
        # if train_f:
        #     anc_box_rpn = anc_box_rpn[idx_nms]
        #     anc_label_rpn = anc_label_rpn[idx_nms]
        #     gt_per_anc_rpn = gt_per_anc_rpn[idx_nms]




if __name__ == '__main__':
    def get_n_params(model):
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def get_model_size_in_mb(model):
        mods = list(model.modules())
        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            sizes = []
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        total_bits = 0
        for i in range(len(sizes)):
            s = sizes[i]
            bits = np.prod(np.array(s)) * bits
            total_bits += bits

        return total_bits

    from torchsummary import summary
    model = PANet().cuda()
    print(get_n_params(model))

#     from dataset.penn_fudan_dataset import *
#
#     model = PANet().cuda()
#
#     root = 'D://DeepLearningData/PennFudanPed/Train/'
#     img_size = (448, 448)
#     transform_img = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
#     transform_mask = transforms.Compose([transforms.Resize(img_size, interpolation=Image.NEAREST),
#                                          transforms.ToTensor()])
#     dset = PennFudanDataset(root, transform_img, transform_mask)
#     img, mask, mask_inst, ann = dset[0]
#     img = img.unsqueeze(0).cuda()
#     bbox = ann['bounding_boxes'].cuda()
#
#     print('bbox # :', len(bbox))
#
#     pred_box, pred_mask, reg_rpn, cls_rpn, anc_box_rpn, anc_label_rpn, gt_per_anc_rpn = model(img, bbox)
#     anc_box_rpn = anc_box_rpn[anc_label_rpn == 1]
#     gt_per_anc_rpn = gt_per_anc_rpn[anc_label_rpn == 1]
#
#     anc_box_rpn = convert_box_from_yxhw_to_yxyx(anc_box_rpn)
#     gt_per_anc_rpn = convert_box_from_yxhw_to_yxyx(gt_per_anc_rpn)
#
#     for i in range(len(anc_box_rpn)):
#         print(anc_box_rpn[i].detach().cpu().numpy(), gt_per_anc_rpn[i].detach().cpu().numpy(), calculate_ious(anc_box_rpn[i], gt_per_anc_rpn[i]))





























