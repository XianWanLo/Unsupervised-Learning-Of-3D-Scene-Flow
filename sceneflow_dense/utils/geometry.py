import numpy as np
import os
import os.path as osp
import torch 

HEIGHT = 256 
WIDTH = 832

def resize_intrinsic(intrinsic, new_h, new_w, ori_h, ori_w):

    Sx = new_w / ori_w
    Sy = new_h / ori_h

    b = intrinsic.shape[0]

    scale = torch.tensor([[Sx, 0,0],
                         [0, Sy,0],
                         [0, 0, 1]]).unsqueeze(0).repeat(b,1,1).cuda()        #(1,3,3)
    
    new_intrinsic = scale.bmm(intrinsic.float())
    new_inv_intrinsic = torch.inverse(new_intrinsic)

    return new_intrinsic, new_inv_intrinsic


def get_batch_2d_flow(pc1, pc2, predicted_pc2, intrinsics , new_h, new_w, ori_h, ori_w):
    
    #^ 3/16 修改
    w_ratio = ori_w / new_w
    h_ratio = ori_h / new_h

    #w_ratio = 1
    #h_ratio = 1

    fx = intrinsics[0,0,0] / w_ratio
    cxs = intrinsics[0,0,2] / w_ratio

    fy = intrinsics[0,1,1] / h_ratio
    cys = intrinsics[0,1,2] / h_ratio

    print("fx:",fx)
    print("cxs:",cxs)
    print("fy:",fy)
    print("cys:",cys)

    constx = 0
    consty = 0
    constz = 0

    px1, py1 = project_3d_to_2d(pc1, fx=fx, fy=fy, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2, py2 = project_3d_to_2d(predicted_pc2, fx=fx, fy=fy, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2_gt, py2_gt = project_3d_to_2d(pc2, fx=fx, fy=fy, cx=cxs, cy=cys,
                                        constx=constx, consty=consty, constz=constz)
   
    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt


def project_3d_to_2d(pc, fx=479, fy=479, cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * fx + cx * pc[..., 2] + constx) / (pc[..., 2] + constz + 10e-10)
    y = (pc[..., 1] * fy + cy * pc[..., 2] + consty) / (pc[..., 2] + constz + 10e-10)

    return x, y


