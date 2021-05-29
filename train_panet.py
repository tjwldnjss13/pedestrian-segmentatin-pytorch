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
from dataset.panet_target import *
from model.panet import *
from loss import *
from utils.pytorch_util import *
from utils.rpn_util import *
from utils.util import *


def get_dataset():
    root = 'D://DeepLearningData/PennFudanPed/'
    root_train = os.path.join(root, 'Train')
    root_val = os.path.join(root, 'Validation')

    transform_img_train = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    transform_img_val = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    transform_mask = transforms.Compose(
        [transforms.Resize((448, 448), interpolation=Image.NEAREST), transforms.ToTensor()])

    aug_flip = horizontal_flip_augmentation
    aug_shift = shift_augmentation
    aug_flip_shift = [horizontal_flip_augmentation, shift_augmentation]
    aug_rot = rotate2d_augmentation
    aug_flip_rot = [horizontal_flip_augmentation, rotate2d_augmentation]
    aug_shift_rot = [shift_augmentation, rotate2d_augmentation]
    aug_flip_shift_rot = [horizontal_flip_augmentation, shift_augmentation, rotate2d_augmentation]

    dset_og = PennFudanDataset(root_train, transform_img_train, transform_mask)
    dset_flip = PennFudanDataset(root_train, transform_img_train, transform_mask, aug_flip)
    dset_shift = [PennFudanDataset(root_train, transform_img_train, transform_mask, aug_shift) for _ in range(10)]
    dset_flip_shift = [PennFudanDataset(root_train, transform_img_train, transform_mask, aug_flip_shift) for _ in range(10)]
    dset_rot = [PennFudanDataset(root_train, transform_img_train, transform_mask, aug_rot) for _ in range(10)]
    dset_flip_rot = [PennFudanDataset(root_train, transform_img_train, transform_mask, aug_flip_rot) for _ in range(30)]
    dset_shift_rot = [PennFudanDataset(root_train, transform_img_train, transform_mask, aug_shift_rot) for _ in range(30)]
    dset_flip_shift_rot = [PennFudanDataset(root_train, transform_img_train, transform_mask, aug_flip_shift_rot) for _ in range(30)]

    dset_train = ConcatDataset([*dset_shift, *dset_flip_shift, *dset_rot, *dset_flip_rot, *dset_shift_rot, *dset_flip_shift_rot])
    # dset_train = ConcatDataset([dset_og, dset_flip, *dset_rot, *dset_flip_rot])
    dset_val = PennFudanDataset(root_val, transform_img_val, transform_mask)

    return dset_train, dset_val, custom_collate_fn


def adjust_learning_rate(optimizer, epoch):
    for p in optimizer.param_groups:
        if epoch == 30:
            p['lr'] *= .1
        elif epoch == 60:
            p['lr'] *= .1


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--lr', type=float, required=False, default=.0001)
    parser.add_argument('--num_epoch', type=int, required=False, default=30)
    parser.add_argument('--weight_decay', type=float, required=False, default=.005)
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

    model = PANet(device).to(device)
    anc_box, valid_idx = make_anchor_box()

    state_dict_pth = None
    # state_dict_pth = './pretrain/7epoch_1.5891003loss.pth'
    if state_dict_pth is not None:
        model.load_state_dict(torch.load(state_dict_pth))
    anc_box = model.rpn.anc_boxes

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = PANetLoss()
    # loss_func = test_cross_entropy_loss

    train_loss_list = []
    train_loss_rpn_list = []
    train_loss_box_list = []
    train_loss_mask_list = []

    val_loss_list = []
    val_loss_rpn_list = []
    val_loss_box_list = []
    val_loss_mask_list = []

    t_start = time.time()
    for e in range(num_epoch):
        model.train()
        t_start_train = time.time()
        train_losses = 0
        train_losses_rpn = 0
        train_losses_box = 0
        train_losses_mask = 0
        N_data = 0

        # adjust_learning_rate(optimizer, e + 1)
        for i, (img, _, mask_inst, ann) in enumerate(train_loader):
            # t_s = time.time()
            # print(f'train start-------------------------------------------------')
            N_data += 1

            print(f'[{e+1}/{num_epoch}] ', end='')
            print(f'{N_data}/{len(train_dset)}  ', end='')

            x = img[0].unsqueeze(0).to(device)
            y_bbox = ann[0]['bounding_boxes'].to(device)
            y_mask_inst = mask_inst[0].to(device)

            optimizer.zero_grad()

            pred_box, pred_mask, loss_box, loss_mask, loss_rpn = model(x, y_bbox, y_mask_inst)

            loss = loss_box + loss_mask + loss_rpn

            loss.backward()
            optimizer.step()

            t_mid = time.time()

            h, m, s = time_calculator(t_mid - t_start)

            train_losses += loss.detach().cpu().item()
            train_losses_rpn += loss_rpn.detach().cpu().item()
            train_losses_box += loss_box.detach().cpu().item()
            train_losses_mask += loss_mask.detach().cpu().item()

            print(f'<loss> {loss.detach().cpu().item():10.9f} ({train_losses / N_data:10.9f})  ', end='')
            print(f'<loss_rpn> {loss_rpn.detach().cpu().item():10.9f} ({train_losses_rpn / N_data:10.9f})  ', end='')
            print(f'<loss_box> {loss_box.detach().cpu().item():10.9f} ({train_losses_box / N_data:10.9f})  ', end='')
            print(f'<loss_mask> {loss_mask.detach().cpu().item():10.9f} ({train_losses_mask / N_data:10.9f})  ', end='')
            print(f'<time> {int(h):02d}:{int(m):02d}:{int(s):02d}')

        train_losses /= N_data
        train_losses_rpn /= N_data
        train_losses_box /= N_data
        train_losses_mask /= N_data

        train_loss_list.append(train_losses)
        train_loss_rpn_list.append(train_losses_rpn)
        train_loss_box_list.append(train_losses_box)
        train_loss_mask_list.append(train_losses_mask)

        val_losses = 0
        val_losses_rpn = 0
        val_losses_box = 0
        val_losses_mask = 0
        N_data = 0

        with torch.no_grad():
            for i, (img, _, mask_inst, ann) in enumerate(val_loader):
                N_data += 1

                x = img[0].unsqueeze(0).to(device)
                y_bbox = ann[0]['bounding_boxes'].to(device)
                y_mask_inst = mask_inst[0]

                pred_box, pred_mask, loss_box, loss_mask, loss_rpn = model(x, y_bbox, y_mask_inst)

                loss = loss_box + loss_mask + loss_rpn

                val_losses += loss.cpu().item()
                val_losses_rpn += loss_rpn.cpu().item()
                val_losses_box += loss_box.cpu().item()
                val_losses_mask += loss_mask.cpu().item()

        val_losses /= N_data
        val_losses_rpn /= N_data
        val_losses_box /= N_data
        val_losses_mask /= N_data

        print(f'<val_loss> {val_losses:10.9f}  ', end='')
        print(f'<val_loss_rpn> {val_losses_rpn:10.9f}  ', end='')
        print(f'<val_loss_box> {val_losses_box:10.9f}  ', end='')
        print(f'<val_loss_mask> {val_losses_mask:10.9f}')

        val_loss_list.append(val_losses)
        val_loss_rpn_list.append(val_losses_rpn)
        val_loss_box_list.append(val_losses_box)
        val_loss_mask_list.append(val_losses_mask)

        save_pth = f'./save/{e+1}epoch_{val_loss_list[-1]:.7f}loss_{val_loss_rpn_list[-1]:.7f}lossrpn_{val_loss_box_list[-1]:.7f}lossbox_{val_loss_mask_list[-1]:.7f}lossmask.pth'
        torch.save(model.state_dict(), save_pth)

    x_axis = [i for i in range(num_epoch)]

    plt.figure(0)
    plt.plot(x_axis, train_loss_list, 'r-', label='train')
    plt.plot(x_axis, val_loss_list, 'b-', label='val')
    plt.title('Loss')

    plt.figure(1)
    plt.plot(x_axis, train_loss_rpn_list, 'r-', label='train')
    plt.plot(x_axis, val_loss_rpn_list, 'b-', label='val')
    plt.title('RPN Loss')

    plt.figure(2)
    plt.plot(x_axis, train_loss_box_list, 'r-', label='train')
    plt.plot(x_axis, val_loss_box_list, 'b-', label='val')
    plt.title('Box Loss')

    plt.figure(3)
    plt.plot(x_axis, train_loss_mask_list, 'r-', label='train')
    plt.plot(x_axis, val_loss_mask_list, 'b-', label='val')
    plt.title('Mask Loss')

    plt.show()































