import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import models_mae


def show_image(image, title=''):
    # image is [H, W, 3]
    # assert image.shape[2] == 3
    # plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255 , 0, 255).int())
    # plt.imshow(torch.clip((image -torch.min(image))/(torch.max(image)+torch.min(image)) * 255 , 0, 255).int())
    image = np.array(image)
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model, imggt):
    x = torch.tensor(img)
    gt = torch.tensor(imggt)

    # make it a batch-like
    x = x.unsqueeze(dim=0).unsqueeze(dim=-1).repeat(1,1,1,3)
    x = torch.einsum('nhwc->nchw', x)

    gt = gt.unsqueeze(dim=0).unsqueeze(dim=-1).repeat(1,1,1,3)
    gt = torch.einsum('nhwc->nchw', gt)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.25)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)
    gt = torch.einsum('nchw->nhwc', gt)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    diff = abs(gt-im_paste)

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 6, 1)
    show_image(x[0], "original")
    print(f'rawmax{torch.max(x[0])}---rawmin{torch.min(x[0])}')

    plt.subplot(1, 6, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 6, 3)
    show_image(y[0], "reconstruction")
    print(f'reconmax{torch.max(y[0])}---reconmin{torch.min(y[0])}')

    plt.subplot(1, 6, 4)
    show_image(im_paste[0], "reconstruction + visible")
    print(f'recon&rawmax{torch.max(im_paste[0])}---recon&rawmin{torch.min(im_paste[0])}')

    plt.subplot(1, 6, 5)
    show_image(diff[0], "difference")
    print(f'diffmax{torch.max(diff[0])}---diffmin{torch.min(diff[0])}')
    plt.subplot(1, 6, 6)
    show_image(gt[0], "original")
    print(f'gtmax{torch.max(gt[0])}---gtmin{torch.min(gt[0])}')

    plt.show()

img = Image.open('/mnt/yangzhou/dataset/matterport/test/raw_depth/q9vSo1VnCiC/resize_1e1e9007512a4bc79d5d2f1da5de91c6_d1_0.png')
imggt = Image.open("/mnt/yangzhou/dataset/matterport/test/GT_depth/q9vSo1VnCiC/resize_1e1e9007512a4bc79d5d2f1da5de91c6_d1_0_mesh_depth.png")

img = img.resize((320,320))
imggt = imggt.resize((320,320))
img = np.array(img) 
imggt = np.array(imggt) / 4000

chkpt_dir = '/mnt/yangzhou/output_dirs_dema25/checkpoint-55.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

# make random mask reproducible (comment out to make it change)
torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae,imggt)