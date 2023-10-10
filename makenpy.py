import shutil 
import os
import numpy as np
from PIL import Image

# colordatapath = '/mnt/yangzhou/dataset/matterport/test/color'
# depthdatapath = '/mnt/yangzhou/dataset/matterport/test/raw_depth'
# newdatapath = '/mnt/yangzhou/dataset/matterport/npy/test'

# paths = os.listdir(colordatapath)
# for path in paths:
#     color_subpath = os.path.join(colordatapath,path)
#     depth_subpath = os.path.join(depthdatapath,path)
#     imgnames = os.listdir(color_subpath)
#     newpath = os.path.join(newdatapath,path)
#     if not os.path.exists(newpath):
#         os.makedirs(newpath)
#     for imgname in imgnames:
#         color_imgpath = os.path.join(color_subpath,imgname)
#         color_img = Image.open(color_imgpath)
#         color_npy = np.array(color_img) / 255
#         depth_imgpath = os.path.join(depth_subpath,imgname)
#         # depth_img = Image.open(depth_imgpath[:-8]+'d'+depth_imgpath[-7:-4]+'_mesh_depth.png')
#         depth_img = Image.open(depth_imgpath[:-8]+'d'+depth_imgpath[-7:-4]+'.png')
#         depth_npy = np.array(depth_img) / 60000
#         depth_npy = np.expand_dims(depth_npy,axis=-1)
#         rgbd_npy = np.concatenate((color_npy,depth_npy),axis=-1)
#         np.save(os.path.join(newdatapath,os.path.join(path,imgname[:-4]+'.npy')),rgbd_npy)
#         print(os.path.join(newdatapath,os.path.join(path,imgname[:-4]+'.npy')))

colordatapath = "/mnt/yangzhou/dataset/matterport/test/color/"
depthdatapath = "/mnt/yangzhou/dataset/matterport/test/raw_depth/"
gtdatapath = "/mnt/yangzhou/dataset/matterport/test/GT_depth/"
newdatapath = "/mnt/yangzhou/dataset/matterport/npy/test_full"

paths = os.listdir(colordatapath)
for path in paths:
    color_subpath = os.path.join(colordatapath,path)
    depth_subpath = os.path.join(depthdatapath,path)
    gt_subpath = os.path.join(gtdatapath,path)
    imgnames = os.listdir(color_subpath)
    newpath = os.path.join(newdatapath,path)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for imgname in imgnames:
        color_imgpath = os.path.join(color_subpath,imgname)
        color_img = Image.open(color_imgpath)
        color_npy = np.array(color_img) / 255
        depth_imgpath = os.path.join(depth_subpath,imgname)
        gt_imgpath = os.path.join(gt_subpath,imgname)
        gt_img = Image.open(gt_imgpath[:-8]+'d'+depth_imgpath[-7:-4]+'_mesh_depth.png')
        gt_npy = np.array(gt_img) / 60000
        gt_npy = np.expand_dims(gt_npy,axis=-1)
        depth_img = Image.open(depth_imgpath[:-8]+'d'+depth_imgpath[-7:-4]+'.png')
        depth_npy = np.array(depth_img) / 60000
        depth_npy = np.expand_dims(depth_npy,axis=-1)
        rgbd_npy = np.concatenate((color_npy,depth_npy,gt_npy),axis=-1)
        np.save(os.path.join(newdatapath,os.path.join(path,imgname[:-4]+'.npy')),rgbd_npy)
        print(os.path.join(newdatapath,os.path.join(path,imgname[:-4]+'.npy')))