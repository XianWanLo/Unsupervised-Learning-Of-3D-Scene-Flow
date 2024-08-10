import numpy as np
import torch
import torch.nn.functional as F
import cv2
import scipy
from collections import Counter


def sub2ind(matrixSize, rowSub, colSub):
            m, n = matrixSize
            return rowSub * (n-1) + colSub - 1


def generate_depth_map(depth, new_h, new_w, valid=None):
    
    h,w = depth.shape

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    #mask = valid
    #xx, yy , depth0 = xx[mask], yy[mask], depth[mask]
    xx, yy , depth0 = xx.reshape(-1), yy.reshape(-1), depth.reshape(-1)

    scale_ratio_w = (new_w-1) / (w-1)
    scale_ratio_h = (new_h-1) / (h-1)

    xx = np.round(xx*scale_ratio_w).astype(np.int32)
    yy = np.round(yy*scale_ratio_h).astype(np.int32)

    valid = (xx >= 0) & (xx < new_w) & (yy >= 0) & (yy < new_h)
    xx, yy, depth0 = xx[valid], yy[valid], depth0[valid]

    #xx, yy, depth = xx[val_inds], yy[val_inds], depth[val_inds]
    
    # project to image
    new_depth = np.zeros([new_h, new_w], dtype=np.float32)
    new_depth[yy, xx] = depth0

    # find the duplicate points and choose the closest depth
    #inds = sub2ind(new_depth.shape, yy, xx)
    #dupe_inds = [item for item, count in Counter(inds).items() if count > 1]

    #for dd in dupe_inds:
        #print("ya")
    #    pts = np.where(inds == dd)[0]
    #    print(pts)
    #    x_loc = int(xx[pts[0]])
    #    y_loc = int(yy[pts[0]])
    #    new_depth[y_loc, x_loc] = depth0[pts].min()
    
    #new_depth[new_depth < 0] = 0

    return new_depth


def resize_sparse_flow_map(flow, valid, target_h,target_w):
    
    curr_h, curr_w, C = flow.shape

    xx = np.tile(np.arange(curr_w, dtype=np.float32)[None, :], (curr_h, 1))
    yy = np.tile(np.arange(curr_h, dtype=np.float32)[:, None], (1, curr_w))

    #mask = valid
    mask = valid
    xx, yy, flow0 = xx[mask], yy[mask], flow[mask][:, :]

    scale_ratio_w = (target_w-1) / (curr_w-1)
    scale_ratio_h = (target_h-1) / (curr_h-1)

    flow1 = flow0

    xx = np.round(xx*scale_ratio_w ).astype(np.int32)
    yy = np.round(yy*scale_ratio_h ).astype(np.int32)

    valid = (xx >= 0) & (xx < target_w) & (yy >= 0) & (yy < target_h)
    xx, yy, flow1 = xx[valid], yy[valid], flow1[valid]

    flow_resized = np.zeros([target_h, target_w, C], dtype=np.float32)
    flow_resized[yy, xx, :] = flow1
    #flow_resized[yy, xx, 2:] = 1.0

    valid_resized  = np.zeros([target_h, target_w], dtype=np.float32)
    valid_resized[yy, xx] = 1.0

    return flow_resized,valid_resized


def depth2pc(depth, fx, fy, cx, cy, flow=None):
    h, w = depth.shape
    depth = depth

    xx = np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1))
    yy = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))

    if flow is None:
        x = (xx - cx) * depth / fx
        y = (yy - cy) * depth / fy
    else:
        x = (xx - cx + flow[..., 0]) * depth / fx
        y = (yy - cy + flow[..., 1]) * depth / fy

    pc = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        depth[:, :, None],
    ], axis=-1)

    return pc


def project_pc2image(pc, image_h, image_w, f, cx=None, cy=None, clip=True):
    pc_x, pc_y, depth = pc[..., 0], pc[..., 1], pc[..., 2]

    cx = (image_w - 1) / 2 if cx is None else cx
    cy = (image_h - 1) / 2 if cy is None else cy

    image_x = cx + (f / depth) * pc_x
    image_y = cy + (f / depth) * pc_y

    if clip:
        return np.concatenate([
            np.clip(image_x[..., None], a_min=0, a_max=image_w - 1),
            np.clip(image_y[..., None], a_min=0, a_max=image_h - 1),
        ], axis=-1)
    else:
        return np.concatenate([
            image_x[..., None],
            image_y[..., None]
        ], axis=-1)

def zero_padding(inputs, pad_h, pad_w):
    input_dim = len(inputs.shape)
    assert input_dim in [2, 3]

    if input_dim == 2:
        inputs = inputs[..., None]

    h, w, c = inputs.shape
    assert h <= pad_h and w <= pad_w

    result = np.zeros([pad_h, pad_w, c], dtype=inputs.dtype)
    result[:h, :w, :c] = inputs

    if input_dim == 2:
        result = result[..., 0]

    return result

def padding_and_cropping(img1,img2,valid,flow_2d,flow_3d, depth1,depth2,depth1_dense,depth2_dense,crop,pc1,pc2):

    padding_h, padding_w = 376, 1242

    img1 = zero_padding(img1, padding_h, padding_w)
    img2 = zero_padding(img2, padding_h, padding_w)
    valid = zero_padding(valid, padding_h, padding_w)
    flow_2d = zero_padding(flow_2d, padding_h, padding_w)
    flow_3d = zero_padding(flow_3d, padding_h, padding_w)
    depth1 = zero_padding(depth1, padding_h, padding_w)
    depth2 = zero_padding(depth2, padding_h, padding_w)
    depth1_dense = zero_padding(depth1_dense, padding_h, padding_w)
    depth2_dense = zero_padding(depth2_dense, padding_h, padding_w)
    pc1 = zero_padding(pc1, padding_h, padding_w)
    pc2 = zero_padding(pc2, padding_h, padding_w)

    img1 = img1[crop:]
    img2 = img2[crop:]
    depth1 = depth1[crop:]
    depth2 = depth2[crop:]
    depth1_dense = depth1_dense[crop:]
    depth2_dense = depth2_dense[crop:]
    flow_2d = flow_2d[crop:]
    flow_3d = flow_3d[crop:]
    valid = valid[crop:]
    pc1 = pc1[crop:]
    pc2 = pc2[crop:]

    return img1,img2,valid,flow_2d,flow_3d,depth1,depth2,depth1_dense,depth2_dense,pc1,pc2


def direct_cropping(image1,image2,depth1,depth2 ,valid, flow_2d,flow_3d,pc1,pc2, new_h, new_w, intrinsics):

    x0 = 100
    y0 = 0

    x1 = x0 + new_w
    y1 = y0 + new_h

    image1 = image1[y0:y1 , x0:x1, :].copy()        # [H W 3]
    image2 = image2[y0:y1 , x0:x1, :].copy()
    flow_2d = flow_2d[y0:y1 , x0:x1, :].copy()
    flow_3d = flow_3d[y0:y1 , x0:x1, :].copy()
    pc1 = pc1[y0:y1 , x0:x1, :].copy()
    pc2 = pc2[y0:y1 , x0:x1, :].copy()
    depth1 = depth1[y0:y1 , x0:x1].copy() 
    depth2 = depth2[y0:y1 , x0:x1].copy() 
    valid = valid[y0:y1 , x0:x1].copy()                  # [H W]
    
    image1 = np.ascontiguousarray(image1.transpose([2, 0, 1]))      # 3 x H x W
    image2 = np.ascontiguousarray(image2.transpose([2, 0, 1]))
    flow_2d = np.ascontiguousarray(flow_2d.transpose([2, 0, 1]))    # 3 x H x W
    flow_3d = np.ascontiguousarray(flow_3d.transpose([2, 0, 1]))
    pc1 = np.ascontiguousarray(pc1.transpose([2, 0, 1]))
    pc2 = np.ascontiguousarray(pc2.transpose([2, 0, 1]))

    intrinsics -= np.array([0, 0, x0, y0])                                  # new intrinsic params     

    return image1,image2,depth1, depth2, valid,flow_2d,flow_3d,pc1,pc2,intrinsics



def new_interpolation(image1,image2,depth1,depth2 ,depth1_dense, depth2_dense,flow_2d, valid, new_h, new_w, intrinsics):

    zoom_y = (new_h) / (image1.shape[0])
    zoom_x = (new_w) / (image1.shape[1])

    fx,fy,cx,cy = intrinsics
    image1 = scipy.misc.imresize(image1, (new_h, new_w))
    image2 = scipy.misc.imresize(image2, (new_h, new_w))

    new_depth1 = generate_depth_map(depth1, new_h, new_w, valid)
    new_depth2 = generate_depth_map(depth2, new_h, new_w, valid)
    new_depth1_dense = generate_depth_map(depth1_dense, new_h, new_w)
    new_depth2_dense = generate_depth_map(depth2_dense, new_h, new_w)

    flow_2d, valid = resize_sparse_flow_map(flow_2d, valid, new_h,new_w) 
    flow_2d[:,:, 0] *= zoom_x
    flow_2d[:,:, 1] *= zoom_y

    intrinsics *= np.array([zoom_x, zoom_y, zoom_x, zoom_y])
    fx,fy,cx,cy = intrinsics

    flow_3d = depth2pc(new_depth2, fx=fx, fy=fy, cx=cx, cy=cy, flow=flow_2d) - depth2pc(new_depth1, fx=fx, fy=fy, cx=cx, cy=cy)
    pc1 = depth2pc(new_depth1_dense, fx=fx, fy=fy, cx=cx, cy=cy)
    pc2 = depth2pc(new_depth2_dense, fx=fx, fy=fy, cx=cx, cy=cy)

    #pc1[:,:, 0] *= zoom_x
    #pc1[:,:, 1] *= zoom_y

    #pc2[:,:, 0] *= zoom_x
    #pc2[:,:, 1] *= zoom_y

    #valid = np.logical_and(np.logical_and(new_depth1 > 0.002, new_depth2 > 0.002) , valid)
    #remove_mask = np.logical_not(valid)

    #pc1[remove_mask,:] = np.array([0.0, 0.0, 0.0])
    #pc2[remove_mask,:] = np.array([0.0, 0.0, 0.0])
    #flow_3d[remove_mask,:] = np.array([0.0, 0.0, 0.0])

    image1 = torch.from_numpy(image1).float().permute(2,0,1).contiguous()
    image2 = torch.from_numpy(image2).float().permute(2,0,1).contiguous()
    depth1 = torch.from_numpy(depth1).contiguous()
    depth2 = torch.from_numpy(depth2).contiguous()
    depth1_dense = torch.from_numpy(depth1_dense).contiguous()
    depth2_dense = torch.from_numpy(depth2_dense).contiguous()
    flow_2d = torch.from_numpy(flow_2d).float().permute(2,0,1).contiguous()
    flow_3d = torch.from_numpy(flow_3d).float().permute(2,0,1).contiguous()
    pc1 = torch.from_numpy(pc1).float().permute(2,0,1).contiguous()
    pc2 = torch.from_numpy(pc2).float().permute(2,0,1).contiguous()
    valid = torch.from_numpy(valid).float()

    #valid = torch.from_numpy(valid).float()
    #valid = F.interpolate(valid[None][None], [new_h, new_w], mode='nearest')[0,0]
    #valid = valid.cpu().detach().numpy()

    return image1,image2, new_depth1, new_depth2,valid,flow_2d,flow_3d ,pc1,pc2 ,intrinsics



def interpolation(image1,image2,depth1,depth2 ,depth1_dense, depth2_dense,valid, flow_2d, flow_3d ,pc1,pc2, new_h, new_w, intrinsics):

    zoom_y = (new_h) / (image1.shape[0])
    zoom_x = (new_w) / (image1.shape[1])

    image1 = torch.from_numpy(image1).float().permute(2,0,1).contiguous()
    image2 = torch.from_numpy(image2).float().permute(2,0,1).contiguous()
    depth1 = torch.from_numpy(depth1).contiguous()
    depth2 = torch.from_numpy(depth2).contiguous()
    depth1_dense = torch.from_numpy(depth1_dense).contiguous()
    depth2_dense = torch.from_numpy(depth2_dense).contiguous()
    flow_2d = torch.from_numpy(flow_2d).float().permute(2,0,1).contiguous()
    flow_3d = torch.from_numpy(flow_3d).float().permute(2,0,1).contiguous()
    pc1 = torch.from_numpy(pc1).float().permute(2,0,1).contiguous()
    pc2 = torch.from_numpy(pc2).float().permute(2,0,1).contiguous()
    valid = torch.from_numpy(valid).float()

    image1 = F.interpolate(image1[None], [new_h, new_w], mode='bilinear',align_corners=True)[0]
    image2 = F.interpolate(image2[None], [new_h, new_w], mode='bilinear',align_corners=True)[0]
    depth1 = F.interpolate(depth1[None][None], [new_h, new_w], mode='bilinear',align_corners=True)[0,0]
    depth2 = F.interpolate(depth2[None][None], [new_h, new_w], mode='bilinear',align_corners=True)[0,0]
    depth1_dense = F.interpolate(depth1_dense[None][None], [new_h, new_w], mode='bilinear',align_corners=True)[0,0]
    depth2_dense = F.interpolate(depth2_dense[None][None], [new_h, new_w], mode='bilinear',align_corners=True)[0,0]
    flow_2d = F.interpolate(flow_2d[None], [new_h, new_w], mode='bilinear',align_corners=True)[0]
    flow_3d = F.interpolate(flow_3d[None], [new_h, new_w],mode='bilinear',align_corners=True)[0]
    pc1 = F.interpolate(pc1[None], [new_h, new_w], mode='bilinear',align_corners=True)[0]
    pc2 = F.interpolate(pc2[None], [new_h, new_w], mode='bilinear',align_corners=True)[0]
    
    valid = F.interpolate(valid[None][None], [new_h, new_w], mode='nearest')[0,0]
    #valid = F.interpolate(valid[None][None], [new_h, new_w],  mode='bilinear',align_corners=True)[0,0]
    #valid = valid > 0.7

    image1 = image1.cpu().detach().numpy()
    image2 = image2.cpu().detach().numpy()
    depth1 = depth1.cpu().detach().numpy()
    depth2 = depth2.cpu().detach().numpy()
    depth1_dense = depth1_dense.cpu().detach().numpy()
    depth2_dense = depth2_dense.cpu().detach().numpy()
    flow_2d = flow_2d.cpu().detach().numpy()
    flow_3d = flow_3d.cpu().detach().numpy()
    valid = valid.cpu().detach().numpy()
    pc1 = pc1.cpu().detach().numpy()
    pc2 = pc2.cpu().detach().numpy()

    intrinsics *= np.array([zoom_x, zoom_y, zoom_x, zoom_y])
    fx,fy,cx,cy = intrinsics

    flow_2d = np.ascontiguousarray(flow_2d.transpose([1, 2, 0]))
    flow_2d = flow_2d * np.array([zoom_x, zoom_y])
    #flow_2d[..., 0] *= zoom_x
    #flow_2d[..., 1] *= zoom_y

    flow_3d = depth2pc(depth2, fx=fx, fy=fy, cx=cx, cy=cy, flow=flow_2d) - depth2pc(depth1, fx=fx, fy=fy, cx=cx, cy=cy)
    pc1 = depth2pc(depth1_dense, fx=fx, fy=fy, cx=cx, cy=cy)
    pc2 = depth2pc(depth2_dense, fx=fx, fy=fy, cx=cx, cy=cy)
    
    #flow_3d[:,:, 0] *= zoom_x
    #flow_3d[:,:, 1] *= zoom_y

    #pc1[:,:, 0] *= zoom_x
    #pc1[:,:, 1] *= zoom_y

    #pc2[:,:, 0] *= zoom_x
    #pc2[:,:, 1] *= zoom_y

    flow_3d = np.ascontiguousarray(flow_3d.transpose([2,0,1]))      # H x W x 3
    pc1 = np.ascontiguousarray(pc1.transpose([2,0,1]))      # H x W x 3
    pc2 = np.ascontiguousarray(pc2.transpose([2,0,1]))      # H x W x 3


    return image1,image2,depth1, depth2, valid,flow_2d,flow_3d ,pc1,pc2 ,intrinsics