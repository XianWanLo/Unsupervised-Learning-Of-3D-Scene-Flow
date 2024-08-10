import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.external_utils import devide_by_index
#from pointconv_util import PointConv, PointConvD, PointWarping, UpsampleFlow, PointConvFlow
from .point_conv import index_points_gather as index_points, index_points_group, Conv1d, square_distance
import time


def curvature(pc):  #作用是保持变形过程中相邻顶点之间的相对位置
    # pc: B 3 N
    pc = pc.permute(0, 2, 1)
    
    #square_distance:用来在ball query过程中确定每一个点距离采样点的距离
    #返回的是两组点之间两两的欧几里德距离，即N×M的矩阵 [B, N, M]
    sqrdist = square_distance(pc, pc)
    
    # torch.topk(input, k, dim=None, largest=True, sorted=True, out=None)
    # 沿给定dim维度（-1：按列）返回input中 k 个最小值和索引
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(pc, kidx)  #B N K 3
    pc_curvature = torch.sum(grouped_pc - pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3


def computeChamfer(pc1, pc2):  #作用是限制网格顶点的具体位置
    
    # Chamfer distance:倒角距离：两个点云之间的距离
    #该距离较大，则说明两组点云区别较大；如果距离较小，则说明重建效果较好。
    # 一般来说，该距离用作3D重建网络的损失函数
    '''
    pc1: B 3 N
    pc2: B 3 M
    '''
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    
    sqrdist12 = square_distance(pc1, pc2) # B N M

    #chamferDist
    #第一项代表S1中任意一点x 到S2 的最小距离之和，按行取值
    dist1, _ = torch.topk(sqrdist12, 1, dim = -1, largest=False, sorted=False)
    #第二项则表示S2 中任意一点y 到S1 的最小距离之和
    dist2, _ = torch.topk(sqrdist12, 1, dim = 1, largest=False, sorted=False)
    dist1 = dist1.squeeze(2)
    dist2 = dist2.squeeze(1)

    return dist1, dist2


def curvatureWarp(pc, warped_pc):

    warped_pc = warped_pc.permute(0, 2, 1)
    pc = pc.permute(0, 2, 1)
    sqrdist = square_distance(pc, pc)
    _, kidx = torch.topk(sqrdist, 10, dim = -1, largest=False, sorted=False) # B N 10 3
    grouped_pc = index_points_group(warped_pc, kidx)
    pc_curvature = torch.sum(grouped_pc - warped_pc.unsqueeze(2), dim = 2) / 9.0
    return pc_curvature # B N 3


def computeSmooth(pc1, pred_flow):
    '''
    pc1: B 3 N
    pred_flow: B 3 N
    '''
    pc1 = pc1.permute(0, 2, 1)
    pred_flow = pred_flow.permute(0, 2, 1)
    sqrdist = square_distance(pc1, pc1) # B N N

    #Smoothness: assume predicted sf at local regions = sf at p 
    _, kidx = torch.topk(sqrdist, 9, dim = -1, largest=False, sorted=False)
    grouped_flow = index_points_group(pred_flow, kidx) # B N 9 3
    diff_flow = torch.norm(grouped_flow - pred_flow.unsqueeze(2), dim = 3).sum(dim = 2) / 8.0

    return diff_flow


def interpolateCurvature(pc1, pc2, pc2_curvature):
    '''
    pc1: B 3 N
    pc2: B 3 M
    pc2_curvature: B 3 M
    '''

    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)
    pc2_curvature = pc2_curvature

    sqrdist12 = square_distance(pc1, pc2) # B N M
    #输出距离和坐标
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False)
    grouped_pc2_curvature = index_points_group(pc2_curvature, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm  # B N 5

    inter_pc2_curvature = torch.sum(weight.view(B, N, 5, 1) * grouped_pc2_curvature, dim = 2)
    return inter_pc2_curvature



def ChamferCurvature_2048(pc1, pc2, flow,mask):

    b = pc1.shape[0]

    mask = mask.reshape(b,-1)

    index1 = np.zeros((b,3000))
    index2 = np.zeros((b,3000))

    for i in range(b):

        f = (mask[i, :] > 0).nonzero()
        f = f.view(-1)
        n_f = f.cpu().numpy()
        np.random.shuffle(n_f)
        index1[i, :] = n_f[:3000]
        np.random.shuffle(n_f)
        index2[i, :] = n_f[:3000]

    cur_pc1 =  devide_by_index(pc1,index1).contiguous()
    cur_pc2 =  devide_by_index(pc2,index2).contiguous()
    cur_flow =  devide_by_index(flow,index1).contiguous()

    #compute curvature: 输出PC2 的相邻点关系
    cur_pc2_curvature = curvature(cur_pc2)

    # PC1 warped = PC1 + sceneflow
    cur_pc1_warp = cur_pc1 + cur_flow
    dist1, dist2 = computeChamfer(cur_pc1_warp, cur_pc2)
   
    #laplacian coordinate vector：输出PC1 warped 的相邻点关系
    moved_pc1_curvature = curvatureWarp(cur_pc1, cur_pc1_warp)

    #CHAMFER LOSS ：PC1 warped 和 PC 2 的同一点距离 
    chamferLoss = dist1.sum(dim = 1).mean() + dist2.sum(dim = 1).mean()

    #SMOOTHNESS LOSS ：PC1 warped 和 PC2 的场景流
    smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

    #interpolated laplacian coordinate vector ：输出interpolated PC1 warped 的相邻点关系
    inter_pc2_curvature = interpolateCurvature(cur_pc1_warp, cur_pc2, cur_pc2_curvature)
    
    #LAPLACIAN REGULARIZATION LOSS 
    curvatureLoss = torch.sum((inter_pc2_curvature - moved_pc1_curvature) ** 2, dim = 2).sum(dim = 1).mean()

    return chamferLoss, curvatureLoss


def interpolateFea(pc1, pc2, fea2):
    '''
    pc1: B 3 N
    pc2: B 3 M
    fea2: B M 3
    '''
    B, _, N = pc1.shape
    pc1 = pc1.permute(0, 2, 1)
    pc2 = pc2.permute(0, 2, 1)

    sqrdist12 = square_distance(pc1, pc2) # B N M
    dist, knn_idx = torch.topk(sqrdist12, 5, dim = -1, largest=False, sorted=False) # B N 5
    grouped_fea2 = index_points_group(fea2, knn_idx) # B N 5 3
    norm = torch.sum(1.0 / (dist + 1e-8), dim = 2, keepdim = True)
    weight = (1.0 / (dist + 1e-8)) / norm  # B N 5
    inter_fea2 = torch.sum(weight.view(B, N, 5, 1) * grouped_fea2, dim = 2)
    return inter_fea2 # B N 3


def rgbConsistencyLoss(pc1,pc2,fea1,fea2,flow):
    fea1 = fea1.permute(0,2,1)
    fea2 = fea2.permute(0,2,1)
    pred_pc2 = pc1 + flow  #B C N
    inter_pc2_fea = interpolateFea(pred_pc2,pc2,fea2)
    rgbLoss = (inter_pc2_fea - fea1).abs().mean()
    return rgbLoss


def devide_by_index(data, index):
    b = data.size()[0]
    new = data.clone()

    new = new[..., : index.shape[1]]
    
    for i in range(b):
        temp_data = data[i,:]
        temp_index = index[i,:]
        temp_data = temp_data[...,temp_index]
        new[i,...] = temp_data
    
    return new
