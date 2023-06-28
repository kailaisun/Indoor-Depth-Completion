import os
import torch
import numpy as np
from math import exp
from torch.autograd import Variable
import open3d as o3d
import cv2

import matplotlib.pyplot as plt
from PIL import Image
import models_mae_full as models_mae
from util import pytorch_ssim


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda')
    # print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'], strict=False)
    # # msg = model.load_state_dict(checkpoint, strict=False)
    # state_dict = model.state_dict()
    # for k in ['head.weight', 'head.bias']:
    #     if k in checkpoint['model'] and checkpoint['model'].shape != state_dict[k].shape:
    #         print(f"Removing key {k} from pretrained checkpoint")
    # print(msg)
    return model

def save_image(image, title=''):
    image = np.array(image)
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    gt = x[:,4:5,:,:]

    # run MAE
    model = model.float()
    loss, y, mask = model(x.float(), mask_ratio=0)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # # visualize the mask
    # mask = mask.detach()
    # mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *4)  # (N, H*W, p*p*3)
    # mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)
    gt = torch.einsum('nchw->nhwc', gt)

    # masked image
    # im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    # im_paste = x * (1 - mask) + y * mask

    

    # diff = abs(gt-im_paste) *60000/4000
    # diff_ = torch.where(gt == 0 , torch.zeros(gt.shape).type(diff.dtype), diff)
    # loss = diff_ ** 2 
    # loss = loss[0,:,:,3]
    # loss = loss.sum()  # [N, L], mean loss per patch
    # loss = loss  / torch.nonzero(gt[0,:,:,3]).sum()
    # loss = torch.sqrt(loss).data.numpy()
    # # diff_raw = abs(gt-x)
    # diff_ = diff_[0,:,:,3]
    # me = diff_.sum() / torch.nonzero(gt[0,:,:,3]).sum().data.numpy()

    
    gt_np = np.array(gt[0,:,:,-1])*60000/4000
    pred_np = np.array(y[0,:,:,3])*60000/4000
    # gtzero = (gt_np == 0.0)
    # pred_np[gtzero] = 0.0
    # gt_np[gtzero] = 0.0
    pred_np = np.where(gt_np <= 0.0, 0.0, pred_np)
    rmse = (gt_np-pred_np)**2
    mean = np.abs(gt_np-pred_np)
    pred = torch.Tensor(pred_np[np.newaxis,np.newaxis,:,:])
    target = torch.Tensor(gt_np[np.newaxis,np.newaxis,:,:])
    ss = ssim_helper(target, pred).data.numpy()
    # # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]

    # plt.subplot(2, 4, 1)
    # save_image(x[0,:,:,3], "raw")
    # print(f'rawmax{torch.max(x[0,:,:,3])}---rawmin{torch.min(x[0,:,:,3])}')

    # plt.subplot(2, 4, 2)
    # save_image(im_masked[0,:,:,3], "raw_masked")

    # plt.subplot(2, 4, 3)
    # save_image(y[0,:,:,3], "rec")
    # print(f'reconmax{torch.max(y[0,:,:,3])}---reconmin{torch.min(y[0,:,:,3])}')

    # plt.subplot(2, 4, 4)
    # save_image(im_paste[0,:,:,3], "rec + raw")
    # print(f'recon&rawmax{torch.max(im_paste[0,:,:,3])}---recon&rawmin{torch.min(im_paste[0,:,:,3])}')

    # plt.subplot(2, 4, 5)
    # save_image(diff_[0,:,:,3], "diff_rec")
    # print(f'diffrecmax{torch.max(diff[0,:,:,3])}---diffrecmin{torch.min(diff[0,:,:,3])}')\
    
    # plt.subplot(2, 4, 6)
    # save_image(diff_raw[0,:,:,3], "diff_raw")
    # print(f'diffrawmax{torch.max(diff_raw[0,:,:,3])}---diffrawmin{torch.min(diff_raw[0,:,:,3])}')

    # plt.subplot(2, 4, 7)
    # save_image(gt[0,:,:,3], "original")
    # print(f'gtmax{torch.max(gt[0,:,:,3])}---gtmin{torch.min(gt[0,:,:,3])}')
    if ss<0.6:
        rgb2file(np.uint8((np.array(img[:,:,0:3]))*255),f'debug/{ss}_color.jpg')
        depth2file(np.uint16(pred_np*4000),'debug/pred.png')
        depth2file(np.uint16(gt_np*4000),'debug/gt.png')
        rgbd2pcd(f'debug/{ss}_color.jpg','debug/pred.png',f'debug/{ss}_pred.ply')
        rgbd2pcd(f'debug/{ss}_color.jpg','debug/gt.png',f'debug/{ss}_gt.ply')

    return rmse,mean,ss

    # plt.show()

def rgb2file(img,color_p):
    img = Image.fromarray(img)
    img.save(color_p)

def depth2file(img,depth_p):
    cv2.imwrite(depth_p, img)   

def rgbd2pcd(color_p,depth_p,name):
    color = o3d.io.read_image(color_p)
    depth = o3d.io.read_image(depth_p)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, depth_scale=10000.0, depth_trunc=6.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # print("Radius oulier removal")
    # pcd, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    o3d.io.write_point_cloud(name,pcd)
    print(name)

test_path='/mnt/yangzhou/dataset/matterport/npy/test_full/'
# gt_path='/mnt/yangzhou/dataset/matterport/test/GT_depth/'
rm_path='/home/yangzhou/mae_color/rm.txt'

chkpt_dir = "/mnt/yangzhou/finetune/checkpoint-10.pth"
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')
ssim_helper = pytorch_ssim.SSIM(11)

torch.manual_seed(0)
print('MAE with pixel reconstruction:')

rmse = []
mae = []
ssip=[]
rm = []
test_list = os.listdir(test_path)

# with open(rm_path, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip('\n').split(' ')
#         rm.append(line[0])
  
for i,scene in enumerate(test_list):
    scene_path = os.path.join(test_path,scene)
    frames = os.listdir(scene_path)
    for j,frame in enumerate(frames):
        frame_path = os.path.join(scene_path,frame)
        if frame_path in rm:
            print(frame_path)
            continue

        img_np = np.load(frame_path)
        depth = img_np[:,:,3]
        depth = Image.fromarray(depth)
        depth = depth.resize((224,224),Image.NEAREST)
        tg = img_np[:,:,4]
        tg = Image.fromarray(tg)
        tg = tg.resize((224,224),Image.NEAREST)

        color = Image.fromarray(np.uint8((img_np[:,:,:-1])*255))
        color = color.resize((224,224),Image.NEAREST)
        color = np.array(color)
        
        img_np =  np.concatenate((color/255,np.expand_dims(depth,axis=-1),np.expand_dims(tg,axis=-1)),axis=-1)

        rmse_, mae_, ssip_ = run_one_image(img_np, model_mae)
        rmse.append(rmse_)
        mae.append(mae_)
        ssip.append(ssip_)
        if ssip_ < 0.6:
            with open("debug/test.txt","a+") as f:
                f.write(f"{ssip_}\t{frame_path}\n")
        print(f'{i}/{len(test_list)}\t{j}/{len(frames)}\trmse:{rmse_}\tmae:{mae_}\tssip:{ssip_}')

rmse_final = np.sqrt(np.mean(rmse))
mae_final = np.mean(mae)
ssip_final = np.mean(ssip)
print(f'rmse:{rmse_final}\nmae:{mae_final}\nssip:{ssip_final}')

