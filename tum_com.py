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
    checkpoint = torch.load(chkpt_dir, map_location={'cuda:7': 'cuda:0'})
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

    # x tensor (b,h,w,5) 补全只用到前4c,最后一c为gt
    # run MAE
    model = model.float()
    loss, y, mask = model(x.float(), mask_ratio=0)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)
 
    #kinect 15000 max 20000~4米

    pred_np = np.array(y[0,:,:,3])*30000
    return pred_np

'''
    pred_np = np.where(gt_np <= 0.0, 0.0, pred_np)
    rmse = (gt_np-pred_np)**2
    mean = np.abs(gt_np-pred_np)
    pred = torch.Tensor(pred_np[np.newaxis,np.newaxis,:,:])
    target = torch.Tensor(gt_np[np.newaxis,np.newaxis,:,:])
    ss = ssim_helper(target, pred).data.numpy()

    if ss<0.6:
        rgb2file(np.uint8((np.array(img[:,:,0:3]))*255),f'debug/{ss}_color.jpg')
        depth2file(np.uint16(pred_np*4000),'debug/pred.png')
        depth2file(np.uint16(gt_np*4000),'debug/gt.png')
        rgbd2pcd(f'debug/{ss}_color.jpg','debug/pred.png',f'debug/{ss}_pred.ply')
        rgbd2pcd(f'debug/{ss}_color.jpg','debug/gt.png',f'debug/{ss}_gt.ply')

    return rmse,mean,ss
'''

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
    o3d.io.write_point_cloud(name,pcd)
    print(name)

ssim_helper = pytorch_ssim.SSIM(11)

# 定义源文件夹和目标文件夹路径
source_folder = "/mnt/yangzhou/dataset/icl/lr1/ndepth/"
color_folder = "/mnt/yangzhou/dataset/icl/lr1/rgb/"
target_folder = "/mnt/yangzhou/dataset/icl/lr1/ndepth_com/"
chkpt_dir = "/mnt/yangzhou/finetune/checkpoint-28.pth"

model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')


# 如果目标文件夹不存在，就创建它
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

i = 0
n = len(os.listdir(source_folder))

# 遍历源文件夹中的每个文件
for file_name in os.listdir(source_folder):
    # 只处理灰度图
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        # 读取图像
        depth_path = os.path.join(source_folder, file_name)
        color_path = os.path.join(color_folder, file_name)
        depth = cv2.imread(depth_path,cv2.IMREAD_ANYDEPTH)
        # depth = Image.open(depth_path)
        color = Image.open(color_path)
        # 处理图像 
        # depth = np.array(depth) / 30000
        # depth = Image.fromarray(depth)
        depth = np.float32(depth/30000)
        depth = cv2.resize(depth, (224,224), interpolation = cv2.INTER_AREA)
        # depth = depth.resize((224,224),Image.NEAREST)
        color = color.resize((224,224),Image.NEAREST)
        color = np.array(color)

        img_np =  np.concatenate((color/255,np.expand_dims(depth,axis=-1),np.expand_dims(depth,axis=-1)),axis=-1)
        torch.manual_seed(0)
        pred = run_one_image(img_np, model_mae)
        # pred = Image.fromarray(pred)
        # pred = pred.resize((640,480),Image.NEAREST)
        pred = cv2.resize(pred, (640,480), interpolation = cv2.INTER_AREA)
        pred = np.uint16(pred)
        
        
        # 保存处理后的图像到目标文件夹中
        target_path = os.path.join(target_folder, file_name)
        i += 1
        print(f'{i}/{n}#{target_path}')
        cv2.imwrite(target_path, pred)

'''
ndepth
vertices 4986598
mean 0.13844856444533932 m
median 0.053973 m
std 0.20035621356932085 m
min 0.0 m
max 1.106677 m

ndepth_c
vertices 21776161
mean 0.08666213017446014 m
median 0.057554 m
std 0.10160359791353529 m
min 0.0 m
max 1.100818 m
'''