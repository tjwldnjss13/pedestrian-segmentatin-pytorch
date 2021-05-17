import torch
import torch.nn as nn


class RoIAlign(nn.Module):
    def __init__(self):
        super(RoIAlign, self).__init__()
        self.roi_size = 7

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

        y_dists_norm = torch.Tensor([y - y1, y2 - y])
        x_dists_norm = torch.Tensor([x - x1, x2 - x])

        p = image[..., y1, x1] * x_dists_norm[1] + image[..., y1, x2] * x_dists_norm[0]
        q = image[..., y2, x1] * x_dists_norm[1] + image[..., y2, x2] * x_dists_norm[0]

        dest = p * y_dists_norm[1] + q * y_dists_norm[0]

        return dest

    def forward(self, feature, rois):
        aligned_feature = torch.zeros(len(rois), feature.shape[-3], 7, 7)

        h_rois = rois[..., 2] - rois[..., 0]
        w_rois = rois[..., 3] - rois[..., 1]

        strides_h = h_rois / 7
        strides_w = w_rois / 7

        for i, roi in enumerate(rois):
            stride_h, stride_w = strides_h[i], strides_w[i]
            for m in range(7):
                for n in range(7):
                    y_base = roi[0] + stride_h * m
                    x_base = roi[1] + stride_w * n

                    pnt_sample_y1 = y_base + stride_h / 3
                    pnt_sample_y2 = y_base + 2 * stride_h / 3
                    pnt_sample_x1 = x_base + stride_w / 3
                    pnt_sample_x2 = x_base + 2 * stride_w / 3

                    dest_1 = self._bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y1).unsqueeze(-1)
                    dest_2 = self._bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y2).unsqueeze(-1)
                    dest_3 = self._bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y1).unsqueeze(-1)
                    dest_4 = self._bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y2).unsqueeze(-1)

                    dest = torch.max(torch.cat([dest_1, dest_2, dest_3, dest_4], dim=-1))
                    aligned_feature[i, ..., m, n] = dest

        return aligned_feature

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

        y_dists_norm = torch.Tensor([y - y1, y2 - y])
        x_dists_norm = torch.Tensor([x - x1, x2 - x])

        p = image[..., y1, x1] * x_dists_norm[1] + image[..., y1, x2] * x_dists_norm[0]
        q = image[..., y2, x1] * x_dists_norm[1] + image[..., y2, x2] * x_dists_norm[0]

        dest = p * y_dists_norm[1] + q * y_dists_norm[0]

        return dest

    def roi_align(feature, rois):
        aligned_feature = torch.zeros(len(rois), feature.shape[-3], 7, 7)

        h_rois = rois[..., 2] - rois[..., 0]
        w_rois = rois[..., 3] - rois[..., 1]

        strides_h = h_rois / 7
        strides_w = w_rois / 7

        for i, roi in enumerate(rois):
            stride_h, stride_w = strides_h[i], strides_w[i]
            for m in range(7):
                for n in range(7):
                    y_base = roi[0] + stride_h * m
                    x_base = roi[1] + stride_w * n

                    pnt_sample_y1 = y_base + stride_h / 3
                    pnt_sample_y2 = y_base + 2 * stride_h / 3
                    pnt_sample_x1 = x_base + stride_w / 3
                    pnt_sample_x2 = x_base + 2 * stride_w / 3

                    dest_1 = bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y1).unsqueeze(-1)
                    dest_2 = bilinear_interpolation(feature, pnt_sample_x1, pnt_sample_y2).unsqueeze(-1)
                    dest_3 = bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y1).unsqueeze(-1)
                    dest_4 = bilinear_interpolation(feature, pnt_sample_x2, pnt_sample_y2).unsqueeze(-1)

                    dest = torch.max(torch.cat([dest_1, dest_2, dest_3, dest_4], dim=-1))
                    aligned_feature[i, ..., m, n] = dest

        return aligned_feature

    img = torch.Tensor([1, 2, 3, 4, 5]).repeat(5, 1).repeat(3, 1, 1)
    rois = torch.Tensor([1, 1, 3, 4]).repeat(2, 1)

    print('img.shape :', img.shape)
    print('rois.shape :', rois.shape)

    aligned_feature = roi_align(img, rois)
    for f in aligned_feature:
        print(f)



if __name__ == '__main__':
    test()