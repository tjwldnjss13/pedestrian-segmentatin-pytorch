import os
import time
import argparse
import torch
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image

from dataset.penn_fudan_dataset import *
from dataset.augment import *
from model.rpn import *
from loss import *
from utils.pytorch_util import *
from utils.util import *


def get_dataset():
    root = 'D://DeepLearningData/PennFudanPed/'
    root_train = os.path.join(root, 'Train')
    root_val = os.path.join(root, 'Validation')

    transform_img = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    transform_mask = transforms.Compose(
        [transforms.Resize((448, 448), interpolation=Image.NEAREST), transforms.ToTensor()])

    aug_shift = shift_with_mask
    aug_flip_shift = [horizontal_flip_with_mask, shift_with_mask]
    aug_rot = rotate2d_with_mask
    aug_flip_rot = [horizontal_flip_with_mask, rotate2d_with_mask]
    aug_shift_rot = [shift_with_mask, rotate2d_with_mask]
    aug_flip_shift_rot = [horizontal_flip_with_mask, shift_with_mask, rotate2d_with_mask]

    dset_og = PennFudanDataset(root_train, transform_img, transform_mask)
    dset_shift = [PennFudanDataset(root_train, transform_img, transform_mask, aug_shift) for _ in range(10)]
    dset_flip_shift = [PennFudanDataset(root_train, transform_img, transform_mask, aug_flip_shift) for _ in range(10)]
    dset_rot = [PennFudanDataset(root_train, transform_img, transform_mask, aug_rot) for _ in range(10)]
    dset_flip_rot = [PennFudanDataset(root_train, transform_img, transform_mask, aug_flip_rot) for _ in range(10)]
    dset_shift_rot = [PennFudanDataset(root_train, transform_img, transform_mask, aug_shift_rot) for _ in range(10)]
    dset_flip_shift_rot = [PennFudanDataset(root_train, transform_img, transform_mask, aug_flip_shift_rot) for _ in range(10)]

    dset_train = ConcatDataset([*dset_shift, *dset_flip_shift, *dset_rot, *dset_flip_rot, *dset_shift_rot, *dset_flip_shift_rot])
    dset_val = PennFudanDataset(root_val, transform_img, transform_mask)

    return dset_train, dset_val, custom_collate_fn


def adjust_learning_rate(optimizer, epoch):
    for p in optimizer.param_groups:
        if epoch < 20:
            p['lr'] = .0001
        elif epoch < 40:
            p['lr'] = .00001
        else:
            p['lr'] = .000001


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=.0001)
    parser.add_argument('--num_epoch', type=int, required=False, default=50)
    parser.add_argument('--weight_decay', type=float, required=False, default=.0005)
    parser.add_argument('--momentum', type=float, required=False, default=.9)

    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr
    num_epoch = args.num_epoch
    weight_decay = args.weight_decay
    momentum = args.momentum

    train_dset, val_dset, collate_fn = get_dataset()

    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = RPN(256, 28).to(device)
    state_dict_pth = './pretrain/22epoch_0.0004651loss.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = RPNLoss(1, 28 ** 2, 1)

    train_loss_list = []
    val_loss_list = []

    t_start = time.time()
    for e in range(num_epoch):
        model.train()
        t_start_train = time.time()
        train_losses = 0
        N_data = 0

        adjust_learning_rate(optimizer, e + 1)
        for i, (img, mask, ann) in enumerate(train_loader):
            # t_s = time.time()
            # print(f'train start-------------------------------------------------')
            N_data += 1

            print(f'[{e+1}/{num_epoch}] ', end='')
            print(f'{N_data}/{len(train_dset)}  ', end='')

            x = img[0].unsqueeze(0).to(device)
            y_bbox = ann[0]['bounding_boxes'].to(device)

            optimizer.zero_grad()
            reg, cls, anc_box, anc_label, gt_per_anc = model(x, y_bbox)
            loss = loss_func(reg, cls, anc_box, anc_label, gt_per_anc)
            loss.backward()
            optimizer.step()

            t_mid = time.time()

            h, m, s = time_calculator(t_mid - t_start)

            train_losses += loss.detach().cpu().item()

            print(f'<loss> {loss.detach().cpu().item():10.9f}  <loss_avg> {train_losses / N_data:10.9f}  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        train_losses /= N_data
        train_loss_list.append(train_losses)

        val_losses = 0
        N_dat = 0
        for i, (img, mask, ann) in enumerate(val_loader):
            N_data += 1

            x = img[0].unsqueeze(0).to(device)
            y_bbox = ann[0]['bounding_boxes'].to(device)

            reg, cls, anc_box, anc_label, gt_per_anc = model(x, y_bbox)
            loss = loss_func(reg, cls, anc_box, anc_label, gt_per_anc)

            val_losses += loss.cpu().item()

        val_losses /= N_data

        print(f'<val_loss> {val_losses:10.9f}')
        val_loss_list.append(val_losses)

        save_pth = f'./save/{e+1}epoch_{val_loss_list[-1]:.7f}loss.pth'
        torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(num_epoch)]
    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='train')
    plt.plot(x_axis, val_loss_list, 'b-', label='val')
    plt.title('Loss')
    plt.show()































