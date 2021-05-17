import torch
import torchvision.transforms as transforms

from PIL import Image

from model.rpn import *


def inference_rpn(model, image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize((448, 448)), transforms.ToTensor()])
    img = transform(img).unsqueeze(0).cuda()

    model.eval()
    out = model(img)

    reg, cls = out[:2]
    # print(reg.shape)
    # print(cls.shape)


if __name__ == '__main__':
    img_pth = './sample/FudanPed00007.png'
    model = RPN(256, 28).cuda()
    state_dict_pth = './pretrain/rpn_final.pth'
    model.load_state_dict(torch.load(state_dict_pth))

    inference_rpn(model, img_pth)











