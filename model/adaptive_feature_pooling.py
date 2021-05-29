import torch
import torch.nn as nn
import torchvision.ops as ops


class RoIAlign(nn.Module):
    def __init__(self):
        super(RoIAlign, self).__init__()
        self.roi_size = 14

    def _bilinear_interpolation(self, image, x, y):
        h, w = image.shape[-2:]

        y1 = torch.floor(y).type(torch.long)
        y2 = y1 + 1
        x1 = torch.floor(x).type(torch.long)
        x2 = x1 + 1

        y1 = torch.clamp(y1, 0, h - 1)
        x1 = torch.clamp(x1, 0, w - 1)
        y2 = torch.clamp(y2, 0, h - 1)
        x2 = torch.clamp(x2, 0, w - 1)

        # y_dists_norm = torch.Tensor([y - y1, y2 - y])
        # x_dists_norm = torch.Tensor([x - x1, x2 - x])
        y_dists_norm = [y - y1, y2 - y]
        x_dists_norm = [x - x1, x2 - x]

        p = image[..., y1, x1] * x_dists_norm[1] + image[..., y1, x2] * x_dists_norm[0]
        q = image[..., y2, x1] * x_dists_norm[1] + image[..., y2, x2] * x_dists_norm[0]

        dest = p * y_dists_norm[1] + q * y_dists_norm[0]

        return dest

    def forward(self, feature, rois):
        """
        :param feature: Tensor, [num batches, channels, height, width]
        :param rois: Tensor, [num rois, (y1, x1, y2, x2)]
        :return aligned_feature: Tensor, [num rois, channels, height, width]
        """
        device = feature.device

        aligned_feature = torch.zeros(len(rois), feature.shape[-3], self.roi_size, self.roi_size)

        h_rois = rois[..., 2] - rois[..., 0]
        w_rois = rois[..., 3] - rois[..., 1]

        strides_h = h_rois / self.roi_size
        strides_w = w_rois / self.roi_size

        # for i, roi in enumerate(rois):
        #     stride_h, stride_w = strides_h[i], strides_w[i]
        #     for m in range(self.roi_size):
        #         for n in range(self.roi_size):
        #             y_base = roi[0] + stride_h * m
        #             x_base = roi[1] + stride_w * n
        #
        #             pnt_sample_y1 = y_base + stride_h / 3
        #             pnt_sample_y2 = y_base + 2 * stride_h / 3
        #             pnt_sample_x1 = x_base + stride_w / 3
        #             pnt_sample_x2 = x_base + 2 * stride_w / 3
        #
        #             dest_1 = self._bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y1).unsqueeze(-1)
        #             dest_2 = self._bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y2).unsqueeze(-1)
        #             dest_3 = self._bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y1).unsqueeze(-1)
        #             dest_4 = self._bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y2).unsqueeze(-1)
        #
        #             dest = torch.max(torch.cat([dest_1, dest_2, dest_3, dest_4], dim=-1))
        #             aligned_feature[i, ..., m, n] = dest

        for m in range(self.roi_size):
            for n in range(self.roi_size):
                y_base = rois[..., 0] + strides_h * m
                x_base = rois[..., 1] + strides_w * n

                pnts_sample_y1 = y_base + strides_h / 3
                pnts_sample_y2 = y_base + 2 * strides_h / 3
                pnts_sample_x1 = x_base + strides_w / 3
                pnts_sample_x2 = x_base + 2 * strides_w / 3

                dest_1 = self._bilinear_interpolation(feature, pnts_sample_x1, pnts_sample_y1).unsqueeze(-1)
                dest_2 = self._bilinear_interpolation(feature, pnts_sample_x1, pnts_sample_y2).unsqueeze(-1)
                dest_3 = self._bilinear_interpolation(feature, pnts_sample_x2, pnts_sample_y1).unsqueeze(-1)
                dest_4 = self._bilinear_interpolation(feature, pnts_sample_x2, pnts_sample_y2).unsqueeze(-1)

                dest = torch.max(torch.cat([dest_1, dest_2, dest_3, dest_4], dim=-1))
                aligned_feature[..., m, n] = dest

        return aligned_feature.to(device)


def adaptive_feature_pooling(scaled_features, rois):
    base_size = 28
    roi_size = 14

    first_f = True
    for i in range(len(scaled_features) - 1, -1, -1):
        rois = rois * 2 ** i * base_size
        rois_pool_temp = ops.roi_align(input=scaled_features[i], boxes=[rois], output_size=roi_size)
        if first_f:
            rois_pool = rois_pool_temp
            first_f = False
        else:
            rois_pool = torch.maximum(rois_pool, rois_pool_temp)
            del rois_pool_temp

    return rois_pool



class AdaptiveFeaturePooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_size = 28
        # self.roi_align = RoIAlign()
        self.roi_align = ops.roi_align
        self.roi_size = 14

    def forward(self, scaled_features, rois):
        """
        :param scaled_features: list of Tensor, [N2(Biggest), N3, N4, N5(Smallest)] where N# is Tensor of [num batches, channels, height, width]
        :param rois: Tensor normalized (0~1), [num rois, (y1, x1, y2, x2)]
        :return rois_max: Tensor, [num rois, channels, height, width]
        """

        first_f = True
        for i in range(len(scaled_features) - 1, -1, -1):
            rois = rois * 2 ** i * self.base_size
            rois_pool_temp = self.roi_align(input=scaled_features[i], boxes=[rois], output_size=self.roi_size)
            if first_f:
                rois_pool = rois_pool_temp
                first_f = False
            else:
                rois_pool = torch.maximum(rois_pool, rois_pool_temp)

        return rois_pool




def test():
    def bilinear_interpolation(image, x, y):
        h, w = image.shape[-2:]

        y1 = torch.floor(y).type(torch.long)
        y2 = y1 + 1
        x1 = torch.floor(x).type(torch.long)
        x2 = x1 + 1

        y1 = torch.clamp(y1, 0, h - 1)
        x1 = torch.clamp(x1, 0, w - 1)
        y2 = torch.clamp(y2, 0, h - 1)
        x2 = torch.clamp(x2, 0, w - 1)

        # y_dists_norm = torch.Tensor([y - y1, y2 - y])
        # x_dists_norm = torch.Tensor([x - x1, x2 - x])

        y_dists_norm = [y - y1, y2 - y]
        x_dists_norm = [x - x1, x2 - x]

        p = image[..., y1, x1] * x_dists_norm[1] + image[..., y1, x2] * x_dists_norm[0]
        q = image[..., y2, x1] * x_dists_norm[1] + image[..., y2, x2] * x_dists_norm[0]

        dest = p * y_dists_norm[1] + q * y_dists_norm[0]

        return dest

    def roi_align(feature, rois):
        aligned_feature = torch.zeros(len(rois), feature.shape[-3], 14, 14)

        h_rois = rois[..., 2] - rois[..., 0]
        w_rois = rois[..., 3] - rois[..., 1]

        strides_h = h_rois / 14
        strides_w = w_rois / 14

        for m in range(14):
            for n in range(14):
                y_base = rois[..., 0] + strides_h * m
                x_base = rois[..., 1] + strides_w * n

                pnts_sample_y1 = y_base + strides_h / 3
                pnts_sample_y2 = y_base + 2 * strides_h / 3
                pnts_sample_x1 = x_base + strides_w / 3
                pnts_sample_x2 = x_base + 2 * strides_w / 3

                dest_1 = bilinear_interpolation(feature, pnts_sample_x1, pnts_sample_y1).unsqueeze(-1)
                dest_2 = bilinear_interpolation(feature, pnts_sample_x1, pnts_sample_y2).unsqueeze(-1)
                dest_3 = bilinear_interpolation(feature, pnts_sample_x2, pnts_sample_y1).unsqueeze(-1)
                dest_4 = bilinear_interpolation(feature, pnts_sample_x2, pnts_sample_y2).unsqueeze(-1)

                dest = torch.max(torch.cat([dest_1, dest_2, dest_3, dest_4], dim=-1))
                aligned_feature[..., m, n] = dest


        # for i, roi in enumerate(rois):
        #     stride_h, stride_w = strides_h[i], strides_w[i]
        #     for m in range(7):
        #         for n in range(7):
        #             y_base = roi[0] + stride_h * m
        #             x_base = roi[1] + stride_w * n
        #
        #             pnt_sample_y1 = y_base + stride_h / 3
        #             pnt_sample_y2 = y_base + 2 * stride_h / 3
        #             pnt_sample_x1 = x_base + stride_w / 3
        #             pnt_sample_x2 = x_base + 2 * stride_w / 3
        #
        #             dest_1 = bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y1).unsqueeze(-1)
        #             dest_2 = bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y2).unsqueeze(-1)
        #             dest_3 = bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y1).unsqueeze(-1)
        #             dest_4 = bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y2).unsqueeze(-1)
        #
        #             dest = torch.max(torch.cat([dest_1, dest_2, dest_3, dest_4], dim=-1))
        #             aligned_feature[i, ..., m, n] = dest

        return aligned_feature

    img = torch.Tensor([1, 2, 3, 4, 5]).repeat(5, 1).repeat(3, 1, 1)
    rois = torch.Tensor([1, 1, 3, 4]).repeat(10000, 1)

    print('img.shape :', img.shape)
    print('rois.shape :', rois.shape)

    aligned_feature = roi_align(img, rois)
    print('aligned_feature.shape :', aligned_feature.shape)



if __name__ == '__main__':
    test()