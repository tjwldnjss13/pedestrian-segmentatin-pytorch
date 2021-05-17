import torch
import cv2 as cv
import numpy as np
import torchvision.transforms as transforms

from PIL import Image

# For test
import matplotlib.pyplot as plt
from dataset.augment import *
from dataset.penn_fudan_dataset import *


class RandomGaussianNoise(object):
    def __init__(self, mean=0., std=.1, probability=.5):
        self.mean = mean
        self.std = std
        self.prob = probability

    def __call__(self, tensor):
        if np.random.rand() < self.prob:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            return tensor

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}'


def rotate2d(image, bounding_box, angle):
    """
    :param image: Tensor, [channel, height, width]
    :param bounding_box: Tensor, [num bounding box, (y_min, x_min, y_max, x_max)]
    :param angle: int
    :return: img_rotate, bbox_rotate
    """
    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    bbox = bounding_box.numpy()

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))

    h_rotate, w_rotate, _ = img_rotate.shape
    x_ctr_rotate, y_ctr_rotate = int(w_rotate / 2), int(h_rotate / 2)

    theta = angle * np.pi / 180
    w_dif, h_dif = int((w_rotate - w) / 2), int((h_rotate - h) / 2)

    bbox_rotate_list = []
    if len(bbox) > 0:
        theta *= -1
        for i in range(len(bbox)):
            # x0, y0, x2, y2 = bbox[i]
            y0, x0, y2, x2 = bbox[i]
            x1, y1, x3, y3 = x2, y0, x0, y2

            # img = cv.circle(img, (x0, y0), 5, (0, 0, 255), thickness=2)
            # img = cv.circle(img, (x1, y1), 5, (0, 0, 255), thickness=2)
            # img = cv.circle(img, (x2, y2), 5, (0, 0, 255), thickness=2)
            # img = cv.circle(img, (x3, y3), 5, (0, 0, 255), thickness=2)
            #
            # img = cv.putText(img, '0', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img = cv.putText(img, '1', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img = cv.putText(img, '2', (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img = cv.putText(img, '3', (x3, y3), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            #
            # plt.imshow(img)
            # plt.show()

            x0, y0, x1, y1 = x0 + w_dif, y0 + h_dif, x1 + w_dif, y1 + h_dif
            x2, y2, x3, y3 = x2 + w_dif, y2 + h_dif, x3 + w_dif, y3 + h_dif

            x0_rot = int((((x0 - x_ctr_rotate) * np.cos(theta)) - ((y0 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y0_rot = int((((x0 - x_ctr_rotate) * np.sin(theta)) + ((y0 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x1_rot = int((((x1 - x_ctr_rotate) * np.cos(theta)) - ((y1 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y1_rot = int((((x1 - x_ctr_rotate) * np.sin(theta)) + ((y1 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x2_rot = int((((x2 - x_ctr_rotate) * np.cos(theta)) - ((y2 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y2_rot = int((((x2 - x_ctr_rotate) * np.sin(theta)) + ((y2 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
            x3_rot = int((((x3 - x_ctr_rotate) * np.cos(theta)) - ((y3 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
            y3_rot = int((((x3 - x_ctr_rotate) * np.sin(theta)) + ((y3 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))

            # img_rotate = cv.circle(img_rotate, (x0_rot, y0_rot), 5, (0, 0, 255), thickness=2)
            # img_rotate = cv.circle(img_rotate, (x1_rot, y1_rot), 5, (0, 0, 255), thickness=2)
            # img_rotate = cv.circle(img_rotate, (x2_rot, y2_rot), 5, (0, 0, 255), thickness=2)
            # img_rotate = cv.circle(img_rotate, (x3_rot, y3_rot), 5, (0, 0, 255), thickness=2)
            #
            # img_rotate = cv.putText(img_rotate, '0', (x0_rot, y0_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img_rotate = cv.putText(img_rotate, '1', (x1_rot, y1_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img_rotate = cv.putText(img_rotate, '2', (x2_rot, y2_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            # img_rotate = cv.putText(img_rotate, '3', (x3_rot, y3_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
            #
            # plt.imshow(img_rotate)
            # plt.show()


            x_min, y_min = int(min(x0_rot, x1_rot, x2_rot, x3_rot)), int(min(y0_rot, y1_rot, y2_rot, y3_rot))
            x_max, y_max = int(max(x0_rot, x1_rot, x2_rot, x3_rot)), int(max(y0_rot, y1_rot, y2_rot, y3_rot))

            bbox_rotate_list.append([y_min, x_min, y_max, x_max])

    h_rot, w_rot, _ = img_rotate.shape
    h_ratio, w_ratio = h_og / h_rot, w_og / w_rot

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)

    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)
    bbox_rotate = torch.as_tensor(bbox_rotate_list).type(dtype=torch.float64)

    if len(bbox_rotate) > 0:
        bbox_rotate[:, 0] *= h_ratio
        bbox_rotate[:, 1] *= w_ratio
        bbox_rotate[:, 2] *= h_ratio
        bbox_rotate[:, 3] *= w_ratio

    return img_rotate, bbox_rotate


def rotate2d_with_mask(image, mask):
    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    mask = mask.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    angle = np.random.randint(-30, 30)

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))
    mask_rotate = cv.warpAffine(mask, mat, (bound_w, bound_h))

    h_rotate, w_rotate, _ = img_rotate.shape

    h_rot, w_rot, _ = img_rotate.shape

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)
    mask_rotate = cv.resize(mask_rotate, (w_og, h_og), interpolation=cv.INTER_NEAREST)

    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)
    mask_rotate = torch.as_tensor(mask_rotate).type(dtype=torch.float64)

    return img_rotate, mask_rotate


def rotate2d_augmentation(image, bounding_box=None, mask=None):
    additional = []

    _, h_og, w_og = image.shape
    img = image.permute(1, 2, 0).numpy()
    mask = mask.permute(1, 2, 0).numpy()
    h, w, _ = img.shape
    x_ctr, y_ctr = int(w / 2), int(h / 2)

    angle = np.random.randint(-30, 30)

    mat = cv.getRotationMatrix2D((x_ctr, y_ctr), angle, 1)
    abs_cos = abs(mat[0, 0])
    abs_sin = abs(mat[0, 1])

    bound_w = int(h * abs_sin + w * abs_cos)
    bound_h = int(h * abs_cos + w * abs_sin)

    mat[0, 2] += bound_w / 2 - x_ctr
    mat[1, 2] += bound_h / 2 - y_ctr

    img_rotate = cv.warpAffine(img, mat, (bound_w, bound_h))

    if bounding_box is not None:
        bbox = bounding_box

        h_rotate, w_rotate, _ = img_rotate.shape
        x_ctr_rotate, y_ctr_rotate = int(w_rotate / 2), int(h_rotate / 2)

        theta = angle * np.pi / 180
        w_dif, h_dif = int((w_rotate - w) / 2), int((h_rotate - h) / 2)

        bbox_rotate_list = []
        if len(bbox) > 0:
            theta *= -1
            for i in range(len(bbox)):
                # x0, y0, x2, y2 = bbox[i]
                y0, x0, y2, x2 = bbox[i]
                x1, y1, x3, y3 = x2, y0, x0, y2

                # img = cv.circle(img, (x0, y0), 5, (0, 0, 255), thickness=2)
                # img = cv.circle(img, (x1, y1), 5, (0, 0, 255), thickness=2)
                # img = cv.circle(img, (x2, y2), 5, (0, 0, 255), thickness=2)
                # img = cv.circle(img, (x3, y3), 5, (0, 0, 255), thickness=2)
                #
                # img = cv.putText(img, '0', (x0, y0), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img = cv.putText(img, '1', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img = cv.putText(img, '2', (x2, y2), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img = cv.putText(img, '3', (x3, y3), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                #
                # plt.imshow(img)
                # plt.show()

                x0, y0, x1, y1 = x0 + w_dif, y0 + h_dif, x1 + w_dif, y1 + h_dif
                x2, y2, x3, y3 = x2 + w_dif, y2 + h_dif, x3 + w_dif, y3 + h_dif

                x0_rot = int(
                    (((x0 - x_ctr_rotate) * np.cos(theta)) - ((y0 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y0_rot = int(
                    (((x0 - x_ctr_rotate) * np.sin(theta)) + ((y0 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
                x1_rot = int(
                    (((x1 - x_ctr_rotate) * np.cos(theta)) - ((y1 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y1_rot = int(
                    (((x1 - x_ctr_rotate) * np.sin(theta)) + ((y1 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
                x2_rot = int(
                    (((x2 - x_ctr_rotate) * np.cos(theta)) - ((y2 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y2_rot = int(
                    (((x2 - x_ctr_rotate) * np.sin(theta)) + ((y2 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))
                x3_rot = int(
                    (((x3 - x_ctr_rotate) * np.cos(theta)) - ((y3 - y_ctr_rotate) * np.sin(theta)) + x_ctr_rotate))
                y3_rot = int(
                    (((x3 - x_ctr_rotate) * np.sin(theta)) + ((y3 - y_ctr_rotate) * np.cos(theta)) + y_ctr_rotate))

                # img_rotate = cv.circle(img_rotate, (x0_rot, y0_rot), 5, (0, 0, 255), thickness=2)
                # img_rotate = cv.circle(img_rotate, (x1_rot, y1_rot), 5, (0, 0, 255), thickness=2)
                # img_rotate = cv.circle(img_rotate, (x2_rot, y2_rot), 5, (0, 0, 255), thickness=2)
                # img_rotate = cv.circle(img_rotate, (x3_rot, y3_rot), 5, (0, 0, 255), thickness=2)
                #
                # img_rotate = cv.putText(img_rotate, '0', (x0_rot, y0_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img_rotate = cv.putText(img_rotate, '1', (x1_rot, y1_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img_rotate = cv.putText(img_rotate, '2', (x2_rot, y2_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                # img_rotate = cv.putText(img_rotate, '3', (x3_rot, y3_rot), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0))
                #
                # plt.imshow(img_rotate)
                # plt.show()

                x_min, y_min = int(min(x0_rot, x1_rot, x2_rot, x3_rot)), int(min(y0_rot, y1_rot, y2_rot, y3_rot))
                x_max, y_max = int(max(x0_rot, x1_rot, x2_rot, x3_rot)), int(max(y0_rot, y1_rot, y2_rot, y3_rot))

                bbox_rotate_list.append([y_min, x_min, y_max, x_max])

        h_rot, w_rot, _ = img_rotate.shape
        h_ratio, w_ratio = h_og / h_rot, w_og / w_rot

        bbox_rotate = torch.as_tensor(bbox_rotate_list).type(dtype=torch.float64)

        if len(bbox_rotate) > 0:
            bbox_rotate[:, 0] *= h_ratio
            bbox_rotate[:, 1] *= w_ratio
            bbox_rotate[:, 2] *= h_ratio
            bbox_rotate[:, 3] *= w_ratio

        additional.append(bbox_rotate)

    if mask is not None:
        mask_rotate = cv.warpAffine(mask, mat, (bound_w, bound_h))
        mask_rotate = cv.resize(mask_rotate, (w_og, h_og), interpolation=cv.INTER_NEAREST)
        mask_rotate = torch.as_tensor(mask_rotate).type(dtype=torch.float64)
        additional.append(mask_rotate)

    img_rotate = cv.resize(img_rotate, (w_og, h_og), interpolation=cv.INTER_CUBIC)
    img_rotate = torch.as_tensor(img_rotate).permute(2, 0, 1)

    if len(additional) == 0:
        return img_rotate
    else:
        return img_rotate, additional


def horizontal_flip(image, bounding_box):
    """
    :param image: Tensor, [channel, height, width]
    :param bounding_box: Tensor, [num bounding box, (y_min, x_min, y_max, x_max)]
    :return:
    """
    img_flip = transforms.RandomHorizontalFlip(1)(image)
    _, h, w = img_flip.shape

    bbox = bounding_box
    for i in range(len(bbox)):
        bbox[i, 1], bbox[i, 3] = w - bbox[i, 3], w - bbox[i, 1]

    return img_flip, bbox


def horizontal_flip_with_mask(image, mask):
    img_flip = transforms.RandomHorizontalFlip(1)(image)
    mask_flip = transforms.RandomHorizontalFlip(1)(mask)

    return img_flip, mask_flip


def horizontal_flip_augmentation(image, bounding_box=None, mask=None):
    additional = []

    img_flip = transforms.RandomHorizontalFlip(1)(image)
    if bounding_box is not None:
        bbox = bounding_box
        bbox_flip = torch.zeros(bbox.shape)
        w = image.shape[-1]
        bbox_flip[..., 0] = bbox[..., 0]
        bbox_flip[..., 1] = w - bbox[..., 3]
        bbox_flip[..., 2] = bbox[..., 2]
        bbox_flip[..., 3] = w - bbox[..., 1]
        additional.append(bbox_flip)
    if mask is not None:
        mask_flip = transforms.RandomHorizontalFlip(1)(mask)
        additional.append(mask_flip)

    if len(additional) == 0:
        return img_flip
    else:
        return img_flip, additional



def shift_with_mask(image, mask):
    dist_y, dist_x = [np.random.randint(-10, 10) for _ in range(2)]

    img_shift = torch.roll(image, shifts=(dist_y, dist_x), dims=(1, 2))
    mask_shift = torch.roll(mask, shifts=(dist_y, dist_x), dims=(1, 2))

    return img_shift, mask_shift


def shift_augmentation(image, bounding_box=None, mask=None):
    additional = []

    dist_y, dist_x = [np.random.randint(-10, 10) for _ in range(2)]

    img_shift = torch.roll(image, shifts=(dist_y, dist_x), dims=(1, 2))
    if bounding_box is not None:
        bbox_shift = bounding_box
        bbox_shift[..., 0] += dist_y
        bbox_shift[..., 1] += dist_x
        bbox_shift[..., 2] += dist_y
        bbox_shift[..., 3] += dist_x
        additional.append(bbox_shift)
    if mask is not None:
        mask_shift = torch.roll(mask, shifts=(dist_y, dist_x), dims=(1, 2))
        additional.append(mask_shift)

    if len(additional) == 0:
        return image
    else:
        return img_shift, additional


def rotate_test():
    root = 'D://DeepLearningData/PennFudanPed/Train/'
    img_size = (448, 448)
    transform_img = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    transform_mask = transforms.Compose([transforms.Resize(img_size, interpolation=Image.NEAREST),
                                         transforms.ToTensor()])
    dset = PennFudanDataset(root, transform_img, transform_mask)
    img, mask, ann = dset[0]
    bbox = ann['bounding_boxes']
    h = ann['height']
    w = ann['width']
    bbox[..., 0] *= 448. / h
    bbox[..., 1] *= 448. / w
    bbox[..., 2] *= 448. / h
    bbox[..., 3] *= 448. / w

    # img_rot, mask_rot = rotate2d_with_mask(img, mask)
    # img_flip, mask_flip = horizontal_flip_with_mask(img, mask)
    img_rot, [bbox_rot, mask_rot] = rotate2d_augmentation(img, bbox, mask)
    img_rot = img_rot.permute(1, 2, 0).numpy()
    # img_rot, bbox_rot = rotate2d(img, bbox, 30)
    # img_rot = img_rot.permute(1 ,2, 0).numpy()

    print(img_rot.dtype)

    for box in bbox_rot:
        cv.rectangle(img_rot, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 3)

    plt.subplot(121)
    plt.imshow(img_rot)
    plt.subplot(122)
    plt.imshow(mask_rot.squeeze(0))
    plt.show()


def flip_test():
    root = 'D://DeepLearningData/PennFudanPed/Train/'
    img_size = (448, 448)
    transform_img = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    transform_mask = transforms.Compose([transforms.Resize(img_size, interpolation=Image.NEAREST),
                                         transforms.ToTensor()])
    dset = PennFudanDataset(root, transform_img, transform_mask)
    img, mask, ann = dset[0]
    bbox = ann['bounding_boxes']
    h = ann['height']
    w = ann['width']
    bbox[..., 0] *= 448. / h
    bbox[..., 1] *= 448. / w
    bbox[..., 2] *= 448. / h
    bbox[..., 3] *= 448. / w

    img_flip, [bbox_flip, mask_flip] = horizontal_flip_augmentation(img, bbox, mask)
    img_flip = img_flip.permute(1, 2, 0).numpy()

    for box in bbox_flip:
        img_flip = cv.rectangle(img_flip.copy(), (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 3)

    plt.subplot(121)
    plt.imshow(img_flip)
    plt.subplot(122)
    plt.imshow(mask_flip.squeeze(0))
    plt.show()


def shift_test():
    root = 'D://DeepLearningData/PennFudanPed/Train/'
    img_size = (448, 448)
    transform_img = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    transform_mask = transforms.Compose([transforms.Resize(img_size, interpolation=Image.NEAREST),
                                         transforms.ToTensor()])
    dset = PennFudanDataset(root, transform_img, transform_mask)
    img, mask, ann = dset[0]
    bbox = ann['bounding_boxes']
    h = ann['height']
    w = ann['width']
    bbox[..., 0] *= 448. / h
    bbox[..., 1] *= 448. / w
    bbox[..., 2] *= 448. / h
    bbox[..., 3] *= 448. / w

    img_shift, [bbox_shift, mask_shift] = shift_augmentation(img, bbox, mask)
    img_shift = img_shift.permute(1, 2, 0).numpy()

    for box in bbox_shift:
        img_shift = cv.rectangle(img_shift.copy(), (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 3)

    plt.subplot(121)
    plt.imshow(img_shift)
    plt.subplot(122)
    plt.imshow(mask_shift.squeeze(0))
    plt.show()



if __name__ == '__main__':
    shift_test()


























