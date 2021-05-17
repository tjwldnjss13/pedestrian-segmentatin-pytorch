import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


def custom_collate_fn(batch):
    img = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    ann = [item[2] for item in batch]

    return img, mask, ann


def adjust_bounding_box(bounding_box, image_size, target_size):
    bbox = bounding_box
    h_img, w_img = image_size
    h_feat, w_feat = target_size

    bbox[..., 0] *= h_feat / h_img
    bbox[..., 1] *= w_feat / w_img
    bbox[..., 2] *= h_feat / h_img
    bbox[..., 3] *= w_feat / w_img

    return bbox


class PennFudanDataset(data.Dataset):
    def __init__(self, root, transform_image=None, transform_mask=None, transform_augment=None):
        super(PennFudanDataset, self).__init__()
        self.root = root
        self.transform_img = transform_image if transform_image is not None else transforms.ToTensor()
        self.transform_mask = transform_mask if transform_mask is not None else transforms.ToTensor()
        self.transform_aug = transform_augment
        self.img_list = None
        self.mask_list = None
        self.ann_list = None
        self._get_data_list()

    def __getitem__(self, idx):
        img_pth = self.img_list[idx]
        mask_pth = self.mask_list[idx]
        ann_pth = self.ann_list[idx]

        img = Image.open(img_pth)
        mask = Image.open(mask_pth)
        ann = self._get_annotation(ann_pth)

        img = self.transform_img(img)
        mask = self.transform_mask(mask) * 255

        idx = mask > 1
        mask[idx] = 1

        aug = self.transform_aug
        if aug is not None:
            if isinstance(aug, list):
                for func in aug:
                    img, mask = func(img, mask)
            else:
                img, mask = aug(img, mask)

        return img, mask, ann

    def __len__(self):
        return len(self.img_list)

    def _get_data_list(self):
        img_list = []
        mask_list = []
        ann_list = []

        img_dir = os.path.join(self.root, 'PNGImages')
        mask_dir = os.path.join(self.root, 'PedMasks')
        ann_dir = os.path.join(self.root, 'Annotation')

        img_fn_list = os.listdir(img_dir)
        mask_fn_list = os.listdir(mask_dir)
        ann_fn_list = os.listdir(ann_dir)

        num_data = len(img_fn_list)

        for i in range(num_data):
            img_fn = img_fn_list[i]
            mask_fn = mask_fn_list[i]
            ann_fn = ann_fn_list[i]

            img_pth = os.path.join(img_dir, img_fn)
            mask_pth = os.path.join(mask_dir, mask_fn)
            ann_pth = os.path.join(ann_dir, ann_fn)

            img_list.append(img_pth)
            mask_list.append(mask_pth)
            ann_list.append(ann_pth)

        self.img_list = img_list
        self.mask_list = mask_list
        self.ann_list = ann_list

    def _get_annotation(self, annotation_path):
        with open(annotation_path, 'r') as f:
            lines = [line.strip().split() for line in f if not line.startswith('#')]

        h = int(lines[1][-3])
        w = int(lines[1][-5])
        num_objs = int(lines[3][5])
        bboxes = []
        for i in range(num_objs):
            idx = 5 + i * 4
            ymin = int(lines[idx][-4].strip('(').strip(')').strip(','))
            xmin = int(lines[idx][-5].strip('(').strip(')').strip(','))
            ymax = int(lines[idx][-1].strip('(').strip(')').strip(','))
            xmax = int(lines[idx][-2].strip('(').strip(')').strip(','))
            bboxes.append([ymin, xmin, ymax, xmax])

        bboxes = torch.as_tensor(bboxes, dtype=torch.float64)
        bboxes = adjust_bounding_box(bboxes, (448, 448), (28, 28))

        ann_dict = {
            'height': h,
            'width': w,
            'num_objects': num_objs,
            'bounding_boxes': bboxes
        }

        return ann_dict





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from dataset.augment import *

    root = 'D://DeepLearningData/PennFudanPed/'
    img_size = (448, 448)
    transform_img = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    transform_mask = transforms.Compose([transforms.Resize(img_size, interpolation=Image.NEAREST),
                                         transforms.ToTensor()])
    dset = PennFudanDataset(root, transform_img, transform_mask)
    img, mask, ann = dset[0]

    img_rot, mask_rot = rotate2d_with_mask(img, mask, -30)
    img_flip, mask_flip = horizontal_flip_with_mask(img, mask)

    plt.subplot(121)
    plt.imshow(img_flip.permute(1, 2, 0))
    plt.subplot(122)
    plt.imshow(mask_flip.squeeze(0))
    plt.show()






















