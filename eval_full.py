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


def get_args_parser():
    parser = argparse.ArgumentParser('Evaluation', add_help=False)
    parser.add_argument('--data_path', default='/npy/test_full/', type=str,
                        help='test dataset path')
    parser.add_argument('--checkpoint', default='/checkpoint-finetune.pth',
                        help='finetune from checkpoint')

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cuda')
    # print(checkpoint['model'])
    model.load_state_dict(checkpoint['model'], strict=False)
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
    
    x = torch.einsum('nchw->nhwc', x)
    gt = torch.einsum('nchw->nhwc', gt)
    
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

    if ss>0.6:
        rgb2file(np.uint8((np.array(img[:,:,0:3]))*255),f'examples/{ss}_color.jpg')
        depth2file(np.uint16(pred_np*4000),'examples/pred.png')
        depth2file(np.uint16(gt_np*4000),'examples/gt.png')
        rgbd2pcd(f'examples/{ss}_color.jpg','examples/pred.png',f'examples/{ss}_pred.ply')
        rgbd2pcd(f'examples/{ss}_color.jpg','examples/gt.png',f'examples/{ss}_gt.ply')

    return rmse,mean,ss

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

def main(args):
    test_path = args.data_path
    chkpt_dir = args.checkpoint

    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')
    ssim_helper = pytorch_ssim.SSIM(11)

    torch.manual_seed(0)
    print('MAE with pixel reconstruction:')

    rmse = []
    mae = []
    ssip=[]
    test_list = os.listdir(test_path)
  
    for i,scene in enumerate(test_list):
        scene_path = os.path.join(test_path,scene)
        frames = os.listdir(scene_path)
        for j,frame in enumerate(frames):
            frame_path = os.path.join(scene_path,frame)
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
            

rmse_final = np.sqrt(np.mean(rmse))
mae_final = np.mean(mae)
ssip_final = np.mean(ssip)
print(f'rmse:{rmse_final}\nmae:{mae_final}\nssip:{ssip_final}')

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)

