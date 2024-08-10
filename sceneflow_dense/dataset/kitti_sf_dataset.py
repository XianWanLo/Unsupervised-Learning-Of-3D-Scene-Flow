import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import scipy
import os
import cv2
import math
import random
import json
import csv
import pickle
import os.path as osp
from .dataset_utils.augmentation import padding_and_cropping, interpolation, direct_cropping,depth2pc,generate_depth_map,new_interpolation
from utils.inverse_warp import build_block_mask, build_angle_sky
from utils.filters import FilterR
from glob import glob


import re
import cv2
import sys
import logging
import numpy as np
import torch.utils.data
import torch.distributed as dist
from matplotlib.colors import hsv_to_rgb
import custom_transforms


def init_logging(filename=None, debug=False):
    logging.root = logging.RootLogger('DEBUG' if debug else 'INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def dist_reduce_sum(value, n_gpus):
    if n_gpus <= 1:
        return value
    tensor = torch.Tensor([value]).cuda()
    dist.all_reduce(tensor)
    return tensor


def copy_to_device(inputs, device, non_blocking=True):
    if isinstance(inputs, list):
        inputs = [copy_to_device(item, device, non_blocking) for item in inputs]
    elif isinstance(inputs, dict):
        inputs = {k: copy_to_device(v, device, non_blocking) for k, v in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to(device=device, non_blocking=non_blocking)
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))
    return inputs


def size_of_batch(inputs):
    if isinstance(inputs, list):
        return size_of_batch(inputs[0])
    elif isinstance(inputs, dict):
        return size_of_batch(list(inputs.values())[0])
    elif isinstance(inputs, torch.Tensor):
        return inputs.shape[0]
    else:
        raise TypeError('Unknown type: %s' % str(type(inputs)))


def load_fpm(filename):
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception('Malformed PFM header.')

        scale = float(f.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
        else:
            endian = '>'  # big-endian

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)

    return data


def load_flow(filepath):
    with open(filepath, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert (202021.25 == magic), 'Invalid .flo file: incorrect magic number'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow = np.fromfile(f, np.float32, count=2 * w * h).reshape([h, w, 2])

    return flow


def load_flow_png(filepath, scale=64.0):
    # for KITTI which uses 16bit PNG images
    # see 'https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py'
    # The -1 is here to specify not to change the image depth (16bit), and is compatible
    # with both OpenCV2 and OpenCV3
    flow_img = cv2.imread(filepath, -1)
    flow = flow_img[:, :, 2:0:-1].astype(np.float32)
    mask = flow_img[:, :, 0] > 0
    flow = flow - 32768.0
    flow = flow / scale
    return flow, mask


def save_flow(filepath, flow):
    assert flow.shape[2] == 2
    magic = np.array(202021.25, dtype=np.float32)
    h = np.array(flow.shape[0], dtype=np.int32)
    w = np.array(flow.shape[1], dtype=np.int32)
    with open(filepath, 'wb') as f:
        f.write(magic.tobytes())
        f.write(w.tobytes())
        f.write(h.tobytes())
        f.write(flow.tobytes())


def save_flow_png(filepath, flow, mask=None, scale=64.0):
    assert flow.shape[2] == 2
    assert np.abs(flow).max() < 32767.0 / scale
    flow = flow * scale
    flow = flow + 32768.0

    if mask is None:
        mask = np.ones_like(flow)[..., 0]
    else:
        mask = np.float32(mask > 0)

    flow_img = np.concatenate([
        mask[..., None],
        flow[..., 1:2],
        flow[..., 0:1]
    ], axis=-1).astype(np.uint16)

    cv2.imwrite(filepath, flow_img)


def load_disp_png(filepath):
    array = cv2.imread(filepath, -1)
    valid_mask = array > 0
    disp = array.astype(np.float32) / 256.0
    disp[np.logical_not(valid_mask)] = -1.0
    
    return disp, valid_mask


def save_disp_png(filepath, disp, mask=None):
    if mask is None:
        mask = disp > 0
    disp = np.uint16(disp * 256.0)
    disp[np.logical_not(mask)] = 0
    cv2.imwrite(filepath, disp)


def load_calib(filepath):
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('P_rect_02'):
                proj_mat = line.split()[1:]
                proj_mat = [float(param) for param in proj_mat]
                proj_mat = np.array(proj_mat, dtype=np.float32).reshape(3, 4)
                assert proj_mat[0, 1] == proj_mat[1, 0] == 0
                assert proj_mat[2, 0] == proj_mat[2, 1] == 0
                assert proj_mat[0, 0] == proj_mat[1, 1]
                assert proj_mat[2, 2] == 1

    return proj_mat





def viz_optical_flow(flow, max_flow=512):
    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)

    image_h = np.mod(angle / (2 * np.pi) + 1, 1)
    image_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    image_v = np.ones_like(image_s)

    image_hsv = np.stack([image_h, image_s, image_v], axis=2)
    image_rgb = hsv_to_rgb(image_hsv)
    image_rgb = np.uint8(image_rgb * 255)

    return image_rgb





def resize_sparse_image(data, valid, ht1, wd1):
    ht, wd, dim = data.shape
    data = data
    valid = valid > 0.5

    coords = np.meshgrid(np.arange(wd), np.arange(ht))
    coords = np.stack(coords, axis=-1)

    coords0 = coords[valid]
    coords1 = coords0 * [ht1/float(ht), wd1/float(wd)]
    data1 = data[valid]

    xx = np.round(coords1[:,0]).astype(np.int32)
    yy = np.round(coords1[:,1]).astype(np.int32)

    v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
    xx = xx[v]
    yy = yy[v]

    data_resized = np.zeros([ht1, wd1, 2], dtype=np.float32)
    valid_resize = np.zeros([ht1, wd1], dtype=np.float32)

    data_resized[yy, xx] = data1[v]
    valid_resize[yy, xx] = 1.0

    return data_resized, valid_resize


def resize_sparse_flow_map(flow, target_h,target_w):
    curr_h, curr_w = flow.shape[:2]

    coords = np.meshgrid(np.arange(curr_w), np.arange(curr_h))
    coords = np.stack(coords, axis=-1).astype(np.float32)

    mask = flow[..., -1] > 0
    coords0, flow0 = coords[mask], flow[mask][:, :2]

    scale_ratio_w = (target_w - 1) / (curr_w - 1)
    scale_ratio_h = (target_h - 1) / (curr_h - 1)

    #coords1 = coords0 * [scale_ratio_w, scale_ratio_h]
    #flow1 = flow0 * [scale_ratio_w, scale_ratio_h]

    coords1 = coords0 * [scale_ratio_h, scale_ratio_w]
    flow1 = flow0 * [scale_ratio_h, scale_ratio_w]

    xx = np.round(coords1[:, 0]).astype(np.int32)
    yy = np.round(coords1[:, 1]).astype(np.int32)

    valid = (xx >= 0) & (xx < target_w) & (yy >= 0) & (yy < target_h)
    xx, yy, flow1 = xx[valid], yy[valid], flow1[valid]

    flow_resized = np.zeros([target_h, target_w, 2], dtype=np.float32)
    flow_resized[yy, xx, :2] = flow1
    #flow_resized[yy, xx, 2:] = 1.0

    valid_resized  = np.zeros([target_h, target_w], dtype=np.float32)
    valid_resized[yy, xx] = 1.0

    return flow_resized,valid_resized

def outliers_mask(pc1_n3,pc2_n3):

    n,_ = pc1_n3.shape
    data1 = FilterR(pc1_n3)
    _,index1 = data1.fliter_radius(pc1_n3)      # 返回哪些点不是离群点的索引
    non_outliers_pc1 = np.zeros((n,1))
    non_outliers_pc1[index1] = 1

    data2 = FilterR(pc2_n3)
    _,index2 = data2.fliter_radius(pc2_n3)      # 返回哪些点不是离群点的索引
    non_outliers_pc2 = np.zeros((n,1))
    non_outliers_pc2[index2] = 1

    return (non_outliers_pc1*non_outliers_pc2)





class KITTI(torch.utils.data.Dataset):
    def __init__(self, data_root, train = True):
        assert os.path.isdir(data_root)
        # assert cfgs.split in ['training200', 'training160', 'training40']

        self.root_dir = os.path.join(data_root, 'training')
        self.train = train
        self.crop = 80

        if train:
            self.indices = [i for i in range(200) if i % 5 != 0]
            #self.indices = np.arange(150)
        else:
            self.indices = [i for i in range(200) if i % 5 == 0]
            #self.indices = np.arange(150)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        
        np.random.seed(23333)

        #print(index)
        index = self.indices[i]

        proj_mat = load_calib(os.path.join(self.root_dir, 'calib_cam_to_cam', '%06d.txt' % index))
        f, cx, cy = proj_mat[0, 0], proj_mat[0, 2], proj_mat[1, 2]

        image1 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_10.png' % index))[..., ::-1]
        image2 = cv2.imread(os.path.join(self.root_dir, 'image_2', '%06d_11.png' % index))[..., ::-1]
        flow_2d, flow_2d_mask = load_flow_png(os.path.join(self.root_dir, 'flow_occ', '%06d_10.png' % index))
        disp1, mask1 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_0', '%06d_10.png' % index))
        disp2, mask2 = load_disp_png(os.path.join(self.root_dir, 'disp_occ_1', '%06d_10.png' % index))
        disp1_dense, mask1_dense = load_disp_png(os.path.join(self.root_dir, 'disp_ganet_training', '%06d_10.png' % index))
        disp2_dense, mask2_dense = load_disp_png(os.path.join(self.root_dir, 'disp_ganet_training', '%06d_11.png' % index))
        depth1 = np.clip((0.54*f / disp1), a_min=0.001, a_max=None )
        depth2 = np.clip((0.54*f / disp2), a_min=0.001, a_max=None )
        depth1_dense = np.clip((0.54*f / disp1_dense), a_min=0.001, a_max=None )
        depth2_dense =np.clip((0.54*f / disp2_dense), a_min=0.001, a_max=None )

        flow_3d = depth2pc(depth2, fx=f, fy=f, cx=cx, cy=cy, flow=flow_2d) - depth2pc(depth1, fx=f, fy=f, cx=cx, cy=cy)
        
        pc1 = depth2pc(depth1_dense,fx=f, fy=f, cx=cx, cy=cy)
        pc2 = depth2pc(depth2_dense, fx=f, fy=f, cx=cx, cy=cy)

        valid = np.logical_and(np.logical_and(mask1, mask2), flow_2d_mask)
        valid = np.logical_and(np.logical_and(disp1>0, disp2>0) , valid)
        
        new_h = 256 
        new_w = 832    

        #^ 4/23 根据Sequence folder数据处理      
        #image1 = scipy.misc.imresize(image1, (new_h, new_w))
        #image2 = scipy.misc.imresize(image2, (new_h, new_w))
        #flow_2d = scipy.misc.imresize(flow_2d, (new_h, new_w))

        image1,image2,valid,flow_2d,flow_3d, depth1,depth2,depth1_dense,depth2_dense,pc1,pc2 = padding_and_cropping(image1,image2,valid,flow_2d,flow_3d, depth1,depth2,depth1_dense,depth2_dense,self.crop,pc1,pc2)
        intrinsics = np.array([f, f, cx, cy-self.crop])

        #^ 原论文方法尺度缩小
        image1,image2,depth1, depth2, valid,flow_2d,flow_3d ,pc1,pc2,intrinsics = new_interpolation(image1,image2,depth1,depth2 ,depth1_dense,depth2_dense, flow_2d, valid, new_h, new_w, intrinsics)

        #^ 图片尺度缩小
        #image1,image2,depth1, depth2, valid,flow_2d,flow_3d ,pc1,pc2,intrinsics = interpolation(image1,image2,depth1,depth2,depth1_dense, depth2_dense ,valid, flow_2d,flow_3d ,pc1,pc2, new_h, new_w, intrinsics)

        #^ 图片直接裁剪
        #image1,image2,depth1, depth2, valid,flow_2d,flow_3d,pc1,pc2,intrinsics =  direct_cropping(image1,image2,depth1,depth2 ,valid, flow_2d, flow_3d,pc1,pc2, new_h, new_w, intrinsics)
        
        normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
        
        [image1,image2], intrinsics = normalize([image1,image2], intrinsics)

        fx, fy, cx, cy = intrinsics   
        intrinsic_mat = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0,  1]])      

        inv_intrinsic_mat =  np.linalg.inv(intrinsic_mat)             

        #non_outliers_mask = outliers_mask(pc1.reshape(-1,3), pc2.reshape(-1,3))
        #non_outliers_mask = non_outliers_mask.reshape(new_h,new_w)

        #near_mask = np.logical_and(pc1[..., 2] < 35.0, pc2[..., 2] < 35.0)
        #not_ground = np.logical_and(pc1[..., 1] < 1.15, pc2[..., 1] < 1.15)

        #non_remove_mask = np.logical_and(near_mask, not_ground)       
        #remove_mask = np.logical_not(non_remove_mask * non_outliers_mask) 
        #remove_mask = np.logical_not(non_remove_mask) 
        
        #valid[remove_mask] = False
        
        return image1, image2, pc1,pc2,depth1,depth2, flow_3d, intrinsic_mat, inv_intrinsic_mat, valid


if __name__ == '__main__':
    train_dataset = KITTI(data_root = "args.data", train = True)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        pin_memory=False)
    dataitem = iter(train_loader).next()
    pc1, pc2, flow, intrinsics, occ_mask = dataitem
    print(pc1.shape)
    print(pc2.shape)
    print(flow.shape)
    print(occ_mask.shape)

#! Cropping  -- 12/18 
        #x0 = 200
        #y0 = 50

        #new_h = 256
        #new_w = 832 

        #image1 = image1[y0:y0+new_h , x0:x0+new_w, : ]          # [H W 3]
        #image2 = image2[y0:y0+new_h , x0:x0+new_w, : ]
        #flow_2d = flow_2d[y0:y0+new_h , x0:x0+new_w, : ]        # [H W 2]
        #disp1 = disp1[y0:y0+new_h , x0:x0+new_w]                # [H W]
        #disp2 = disp2[y0:y0+new_h , x0:x0+new_w]
        #disp1_dense = disp1_dense[y0:y0+new_h , x0:x0+new_w]    # [H W]
        #disp2_dense = disp2_dense[y0:y0+new_h , x0:x0+new_w]
        #valid = valid[y0:y0+new_h , x0:x0+new_w]                # [H W]

        #cx -= x0
        #cy -= y0
#! Cropping end -- 12/18