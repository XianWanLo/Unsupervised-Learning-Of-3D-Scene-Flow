import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from ops_pytorch.fused_conv_select.fused_conv_select_k import fused_conv_select_k
from time import time 
from model.point_conv_pytorch import fast_gather, warppingProject
from utils.inverse_warp import BackprojectDepth,set_by_index
from torch.autograd import Variable

start_ = time()
end_ = time()

print( " time = ",(end_ - start_))


def computeChamfer_seed(pc1_warp_bn3, pc2_bn3, idx_fetching, intrinsic, height, width ,valid):
    '''
    pc1: B N 3
    pc2: B N 3
    '''

    B = pc1_warp_bn3.shape[0]

    #pc2_warped_bn3 = pc2_to_pc1_warped(pc1_warp_bn3, pc2_bn3, intrinsic, height, width, valid)     # project pc2 to pc1-indexed pc1_warp
    
    pc1_warp = pc1_warp_bn3.reshape(B,height,width,3)       #[B H W 3]
    pc2 = pc2_bn3.reshape(B,height,width,3)                 #[B H W 3]
    pc2_warped = pc2_bn3.reshape(B,height,width,3)       #[B H W 3]
    
    H = pc1_warp.shape[1]
    W = pc1_warp.shape[2]

    npoints = H*W
    kernel_size_H = 20         # bigger ? 
    kernel_size_W = 40
    distance = 100
    K = 1   # smaller
    flag_copy = 0

    kernel_size = kernel_size_H * kernel_size_W
    random_hw = (torch.arange(kernel_size)).cuda().int()

    select_b_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_h_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_w_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    #valid_idx = torch.zeros(B, npoints, kernel_size, 1).cuda().float() 
    #valid_in_dis_idx = torch.zeros(B, npoints, kernel_size, 1).cuda().float() 
    select_mask = torch.zeros(B, npoints, K, 1).cuda().float() 
    
    idx_n2 = get_hw_idx(B, H, W)
    idx_hw = idx_n2.contiguous()

    #^ 3/19日修改 idx_fetching 为 idx_n2 ,结果变差！！ 错误！ 改回来

    #np.savetxt('./../cloud/CHAM_pc1_warp_bn3.txt',(pc1_warp_bn3).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    #np.savetxt('./../cloud/pc2_bn3.txt',(pc2_bn3).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    #np.savetxt('./../cloud/CHAM_pc2_warped.txt',(pc2_warped).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    #valid_bhw1 = valid.unsqueeze(-1).reshape(B,H,W,-1)
    #pc1_warp_valid = pc1_warp*valid_bhw1
    #pc2_valid = pc2*valid_bhw1
    #pc2_warped_valid = pc2_warped*valid_bhw1

    with torch.no_grad():
        select_b_idx, select_h_idx, select_w_idx, select_mask = fused_conv_select_k\
            (pc2_warped, pc1_warp, idx_hw, idx_hw, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, 0, float(distance), 1, 1, select_b_idx, select_h_idx, select_w_idx, select_mask, H, W)

    feature_surround = fast_gather(pc1_warp, select_h_idx, select_w_idx)
    gather_feature_surround = feature_surround.reshape(B, npoints,K, 3)
    gather_feature_surround = gather_feature_surround * (select_mask.detach())  

    #print(select_mask.sum())
    #np.savetxt('./../cloud/CHAM_1_gather_feature_surround.txt',(gather_feature_surround).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    gather_feature_center = pc2_warped.reshape(B, npoints, 3).unsqueeze(2).repeat(1,1,K,1)
    #gather_feature_center_mask = gather_feature_center * (select_mask.detach())  

    #np.savetxt('./../cloud/CHAM_1_gather_feature_center.txt',(gather_feature_center).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    #np.savetxt('./../cloud/gather_feature_center_mask.txt',(gather_feature_center_mask).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    #print("gather_feature_center:",gather_feature_center.shape)
    
    xyz_diff_1 = gather_feature_surround - gather_feature_center     # (b, npoints, K, 3)   #! 在pc2坐标上
    
    #^ Euclidean difference ---- square of xyz2(Q) - xyz1(M)
    euc_diff_1 = torch.sqrt(torch.sum(torch.square(xyz_diff_1), dim=-1 , keepdim=True) + 1e-20 )      # (b, npoints, K, 1)
    euc_diff_1 = euc_diff_1.reshape(B,npoints,1)

    #print("chamfer euc_diff_1:",euc_diff_1.mean(dim=1))
    # -----------------------------------------------------------------------------------------------------
    select_b_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_h_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_w_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    #valid_idx = torch.zeros(B, npoints, kernel_size, 1).cuda().float() 
    #valid_in_dis_idx = torch.zeros(B, npoints, kernel_size, 1).cuda().float() 
    select_mask = torch.zeros(B, npoints, K, 1).cuda().float() 

    with torch.no_grad():
        select_b_idx, select_h_idx, select_w_idx, select_mask = fused_conv_select_k\
            (pc1_warp, pc2, idx_hw, idx_fetching, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, 0, float(distance), 1, 1, select_b_idx, select_h_idx, select_w_idx, select_mask, H, W)

    feature_surround = fast_gather(pc2, select_h_idx, select_w_idx)
    gather_feature_surround = feature_surround.reshape(B, npoints,K, 3)
    gather_feature_surround = gather_feature_surround *(select_mask.detach())  
    #np.savetxt('./../cloud/CHAM_2_gather_feature_surround.txt',(gather_feature_surround).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    gather_feature_center = pc1_warp.reshape(B, npoints,3).unsqueeze(2).repeat(1,1,K,1)
    #np.savetxt('./../cloud/CHAM_2_gather_feature_center.txt',(gather_feature_center).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    xyz_diff_2 = gather_feature_surround - gather_feature_center  # (b, npoints, K 3)           #! 在pc1坐标上
    euc_diff_2 = torch.sqrt(torch.sum(torch.square(xyz_diff_2), dim=-1 , keepdim=True) + 1e-20 )      # (b, npoints, K ,1)
    
    euc_diff_2 = euc_diff_2.reshape(B,npoints,1)
    #print("chamfer euc_diff_2:",euc_diff_2.mean(dim=1))

    return euc_diff_1, euc_diff_2
    #return  euc_diff_2



def compute_curvature_seed(pc1_bn3, pc1_warp_bn3, pc2_bn3, idx_fetching, height, width, valid):

    B = pc1_bn3.shape[0]

    pc1 = pc1_bn3.reshape(B,height,width,3)                 #[B H W 3]
    pc1_warp = pc1_warp_bn3.reshape(B,height,width,3)       #[B H W 3]
    pc2 = pc2_bn3.reshape(B,height,width,3)                 #[B H W 3]

    H = pc1.shape[1]
    W = pc1.shape[2]
    
    npoints = H*W
    kernel_size_H = 20
    kernel_size_W = 40      # bigger? 
    distance = 100
    K = 10
    flag_copy = 0

    kernel_size =  kernel_size_H * kernel_size_W
    random_hw = (torch.arange(kernel_size)).cuda().int()

    select_b_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_h_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_w_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_mask = torch.zeros(B, npoints, K, 1).cuda().float() 
    
    idx_n2 = get_hw_idx(B, H, W)     # b n 2

    #valid_bhw1 = valid.unsqueeze(-1).reshape(B,H,W,-1)
    #pc1_valid = pc1*valid_bhw1
    #pc1_warp_valid = pc1_warp*valid_bhw1
    #pc2_valid = pc2*valid_bhw1
 
    # ----------------------------------------------------------
    with torch.no_grad():
        select_b_idx, select_h_idx, select_w_idx, valid_mask = fused_conv_select_k\
            (pc1, pc1, idx_n2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, 0, float(distance), 1, 1, select_b_idx, select_h_idx, select_w_idx, select_mask,H, W)
    #total_idx = select_h_idx * W + select_w_idx# (b, npoints, K, 1)
    #total_idx = total_idx.reshape(B, npoints*K, 1)   
    
    extract_pc1_warp = fast_gather(pc1_warp, select_h_idx, select_w_idx)        # (B npoints nsample 1 3)
    extract_pc1_warp = extract_pc1_warp.reshape(B, npoints, K, 3)       
    extract_pc1_warp = extract_pc1_warp * (valid_mask.detach()) 
    #print("extract_pc1_warp:",extract_pc1_warp.shape)
    #np.savetxt('./../cloud/extract_pc1_warp.txt',(extract_pc1_warp).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    center_pc1_warp = pc1_warp.reshape(B, npoints,3).unsqueeze(2).repeat(1, 1, K, 1)            #!在pc1坐标上
    #np.savetxt('./../cloud/center_pc1_warp.txt',(center_pc1_warp).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    #laplace_pc1_warp = extract_pc1_warp - center_pc1_warp                              # (b, npoints, K, 3)
    #laplace_pc1_warp = laplace_pc1_warp.sum(dim = 2) / 9.0                             # (b, npoints, 3)
    laplace_pc1_warp = (extract_pc1_warp - center_pc1_warp)                             # (b, npoints, K, 3)
    laplace_pc1_warp = laplace_pc1_warp.sum(dim = 2) / 9.0                              # (b, npoints, 3)
    
    #np.savetxt('./../cloud/laplace_pc1_warp.txt',(laplace_pc1_warp).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    #print("laplace_pc1_warp:",laplace_pc1_warp.mean(dim=1))
    # ---------------------------------------------------------------------------------------
    #start_cur2 = time()

    select_b_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_h_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_w_idx = torch.zeros(B, npoints, K, 1).cuda().long()
    select_mask = torch.zeros(B, npoints, K, 1).cuda().float() 

    with torch.no_grad():
        select_b_idx, select_h_idx, select_w_idx, valid_mask = fused_conv_select_k\
            (pc2, pc2, idx_n2, idx_n2, random_hw, H, W, npoints, kernel_size_H, kernel_size_W, K, 0, float(distance), 1, 1, select_b_idx, select_h_idx, select_w_idx, select_mask, H, W)
    
    extract_pc2 = fast_gather(pc2, select_h_idx, select_w_idx)
    extract_pc2 = extract_pc2.reshape(B, npoints, K, 3) 
    extract_pc2 = extract_pc2 * (valid_mask.detach())
    #np.savetxt('./../cloud/extract_pc2.txt',(extract_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    center_pc2 = pc2.reshape(B,npoints,3).unsqueeze(2).repeat(1, 1, K, 1)               #! 在pc2坐标上
    #np.savetxt('./../cloud/center_pc2.txt',(center_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    laplace_pc2 = extract_pc2 - center_pc2 # (b,npoints, K, 3)
    laplace_pc2 = laplace_pc2.sum(dim=2) / 9.0 # (b, npoints, 3)

    #print("laplace_pc2:",laplace_pc2.mean(dim=1))

    laplace_pc2 = laplace_pc2.reshape(B,H,W,3)
    #np.savetxt('./../cloud/laplace_pc2.txt',(laplace_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    #end_other2 = time()
    #print( "    other2 time = ",(end_other2 - start_other2))

    # --------------------------------------------------------------------------------------------------
    #start_cur3 = time()
    
    K_NEW = 5
    kernel_size_H_new = 20
    kernel_size_W_new = 40
    kernel_size_new = kernel_size_H_new*kernel_size_W_new
    random_hw_new = (torch.arange(kernel_size_new)).cuda().int()

    select_b_idx = torch.zeros(B, npoints, K_NEW, 1).cuda().long()
    select_h_idx = torch.zeros(B, npoints, K_NEW, 1).cuda().long()
    select_w_idx = torch.zeros(B, npoints, K_NEW, 1).cuda().long()
    #valid_idx = torch.zeros(B, npoints, kernel_size_new, 1).cuda().float() 
    #valid_in_dis_idx = torch.zeros(B, npoints, kernel_size_new, 1).cuda().float() 
    select_mask = torch.zeros(B, npoints, K_NEW, 1).cuda().float() 

    with torch.no_grad():
        select_b_idx, select_h_idx, select_w_idx, valid = fused_conv_select_k\
            (pc1_warp, pc2, idx_n2, idx_fetching, random_hw_new, H, W, npoints, kernel_size_H_new, kernel_size_W_new, K_NEW, 0, float(distance), 1, 1, select_b_idx, select_h_idx, select_w_idx, select_mask, H, W)
    
    extract_interpolate_pc2 = fast_gather(pc2, select_h_idx, select_w_idx)
    extract_interpolate_pc2 = extract_interpolate_pc2.reshape(B, npoints, K_NEW, 3) 
    extract_interpolate_pc2 = extract_interpolate_pc2 * (valid.detach())
    #np.savetxt('./../cloud/extract_interpolate_pc2.txt',(extract_interpolate_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    center_interpolate_pc2 = pc1_warp.reshape(B, npoints,3).unsqueeze(2).repeat(1, 1, K_NEW, 1) 
    #np.savetxt('./../cloud/center_interpolate_pc2.txt',(center_interpolate_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    xyz_distance = extract_interpolate_pc2 - center_interpolate_pc2        # (b, npoints, K, 3)
    euc_distance = torch.sqrt(torch.sum(torch.square(xyz_distance), dim=-1 , keepdim=True) + 1e-20 )    # (b, npoints, K,1)
    #print("curv euc_distance:",(euc_distance.sum(dim=2)/(K-1)).mean())

    norm = torch.sum(1.0 / (euc_distance + 1e-8), dim = 2, keepdims = True)              # (b,npoints, K, 1)
    weight = (1.0 / (euc_distance + 1e-8)) / norm                           # (b, npoints, K, 1)

    # ------------------
    grouped_laplace_pc2 = fast_gather(laplace_pc2, select_h_idx, select_w_idx)
    grouped_laplace_pc2 = grouped_laplace_pc2.reshape(B, npoints, K_NEW, 3)
    #np.savetxt('./../cloud/grouped_laplace_pc2.txt',(grouped_laplace_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
    
    laplace_interpolate_pc2 = grouped_laplace_pc2 * weight.reshape(B, npoints, K_NEW, 1)            # (b, npoints, K, 1)
    #np.savetxt('./../cloud/laplace_interpolate_pc2_first.txt',(laplace_interpolate_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    laplace_interpolate_pc2 = torch.sum(laplace_interpolate_pc2,dim=2)                  # (b, npoints, 3)
    #np.savetxt('./../cloud/laplace_interpolate_pc2.txt',(laplace_interpolate_pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

    laplace_loss = laplace_interpolate_pc2 - laplace_pc1_warp       # (b, npoints, 3)

    #print("laplace_loss:",laplace_loss.mean(dim=1))

    #end_other3 = time()
    #print( "    other3 time = ",(end_other3 - start_other3))

    return laplace_loss

HEIGHT = 256 
WIDTH = 832


def pc2_to_pc1_warped(warped_pc1, pc2, intr, h, w , valid):
    
    """warped_pc1: [B N 3]
       pc2 : [B N 3]
       intr: [B 3 3]
    """

    b = warped_pc1.shape[0]
    warped_pc1_b3n = warped_pc1.permute(0,2,1)

    pcoords = intr.bmm(warped_pc1_b3n)          #! pc1坐标上的pc1_warped
    X_ = pcoords[:, 0]
    Y_ = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)
    #Z = warped_pc1_b3n[:, 2:3, :].clamp(min=1e-3)

    u = torch.clamp((X_ / Z), 0, w - 1)
    v = torch.clamp((Y_ / Z), 0, h - 1)

    #^ 二维像素阵列 分别除 w 和 h ,得到归一化像素平面 [0-1, 0-1]
    U_norm = (2*u / (w - 1) - 1)      # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    V_norm = (2*v / (h - 1) - 1)

    #? 3/30新加的
    #grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w), requires_grad=False).type_as(U_norm).expand(b, h, w)  # [bs, H, W]
    #grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w), requires_grad=False).type_as(V_norm).expand(b, h, w)  # [bs, H, W]
    #grid_x = 2 * (grid_x / (w - 1.0) - 0.5)
    #grid_y = 2 * (grid_y / (h - 1.0) - 0.5)

    #grid_x_flat = grid_x.view(b, -1)
    #grid_y_flat = grid_y.view(b, -1)

    #grid_x_flat = torch.where(valid, U_norm, grid_x_flat)
    #grid_y_flat = torch.where(valid, V_norm, grid_y_flat)

    #grid_x = grid_x_flat.view(b, h, w)
    #grid_y = grid_y_flat.view(b, h, w)

    #grid_tf = torch.stack([grid_x, grid_y], dim=2).reshape(b,h,w,2).cuda()     
    grid_tf = torch.stack([U_norm, V_norm], dim=2).reshape(b,h,w,2).cuda()   

    #^ 因为形状有变，pc2_depth 和 inverse intrinsic也需要重新处理，而不是用原始的new_fw_depth 和 inv_intrinsics  (pc2 [BN3] -> pc2_depth [B1HW])
    pc2_depth = pc2[:,:,2].reshape(b,h,w).unsqueeze(1)      # [B N 1] --> [B 1 H W]
    
    #^ depth consistency   
    pro_depth = torch.nn.functional.grid_sample(pc2_depth, grid_tf, padding_mode='border',align_corners=True)
    
    #^ 深度[z] 加上 二维像素阵列 [u,v]  ， 乘以逆内参后得到 三维点云 [x,y,z]
    pro_xyz = BackprojectDepth(pro_depth, torch.inverse(intr))        # 三维点云 = 逆参矩阵 * 像素深度     [B 3 H W]
    pro_xyz_flatten = (pro_xyz).view(b, 3, -1)
    pc2_warped = pro_xyz_flatten.permute(0,2,1).contiguous()

    return pc2_warped


def multiScaleChamferSmoothCurvature(cur_pc1, cur_pc2, cur_flow ,intrinsic, height, width, valid):
    
    b, _ , n = cur_pc1.shape
    #mask_bn1 = mask.reshape(b,-1).unsqueeze(-1)     # [b n 1]                                                          
    
    cur_pc1_warp = cur_pc1 + cur_flow       # [B N 3] 
    
    #^ 3/17 修改 --> 由于fused_conv里要在warped pc1上找 pc2， 所以idx要变，需要是投影warped pc1的index
    pc1_warp_projectIndexes = warppingProject(cur_pc1_warp, intrinsic, height, width)
    #start_chamfer = time()
            
    chamfer_dist1,chamfer_dist2 = computeChamfer_seed(cur_pc1_warp, cur_pc2 ,pc1_warp_projectIndexes, intrinsic, height, width, valid)
    #chamfer_dist_valid = (chamfer_dist1 + chamfer_dist2)* (mask.reshape(b,-1,1))        #(b,n,1)
    #chamferLoss = chamfer_dist_valid.sum(dim=0).sum(dim=0)      #(1)
    chamferLoss1 = (chamfer_dist1.reshape(b,-1)[valid]).sum()      #(1)
    chamferLoss2 = (chamfer_dist2.reshape(b,-1)[valid]).sum()      #(1)
    chamferLoss = chamferLoss2 + chamferLoss1
     
    #start_curv = time()
    curvature = compute_curvature_seed(cur_pc1, cur_pc1_warp, cur_pc2, pc1_warp_projectIndexes, height, width,valid)
    curvature_masked = curvature[valid]                                   #(n,3)
    curvatureLoss = torch.sum((curvature_masked)**2,dim = -1).sum()       # (1)
    
    #end_curv = time()
    #smoothnessLoss = computeSmooth(cur_pc1, cur_flow).sum(dim = 1).mean()

    return chamferLoss, curvatureLoss



RV_WEIGHT = 0.2
DZ_WEIGHT = 250.0


def loss_fn(flow2d_est, flow2d_rev, flow_gt, valid_mask, gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    N = len(flow2d_est)
    loss = 0.0

    for i in range(N):
        w = gamma**(N - i - 1)
        fl_rev = flow2d_rev[i]

        fl_est, dz_est = flow2d_est[i].split([2,1], dim=-1)
        fl_gt, dz_gt = flow_gt.split([2,1], dim=-1)

        loss += w * (valid_mask * (fl_est - fl_gt).abs()).mean()
        loss += w * DZ_WEIGHT * (valid_mask * (dz_est - dz_gt).abs()).mean()
        loss += w * RV_WEIGHT * (valid_mask * (fl_rev - fl_gt).abs()).mean()

    epe_2d = (fl_est - fl_gt).norm(dim=-1)
    epe_2d = epe_2d.view(-1)[valid_mask.view(-1)]

    epe_dz = (dz_est - dz_gt).norm(dim=-1)
    epe_dz = epe_dz.view(-1)[valid_mask.view(-1)]

    metrics = {
        'epe2d': epe_2d.mean().item(),
        'epedz': epe_dz.mean().item(),
        '1px': (epe_2d < 1).float().mean().item(),
        '3px': (epe_2d < 3).float().mean().item(),
        '5px': (epe_2d < 5).float().mean().item(),
    }

    return loss, metrics


 

def get_hw_idx(B, out_H, out_W, stride_H = 1, stride_W = 1):

    H_idx = torch.reshape(torch.arange(0, out_H * stride_H, stride_H, device = "cuda", dtype = torch.int), [1, -1, 1, 1]).expand(B, out_H, out_W, 1)
    W_idx = torch.reshape(torch.arange(0, out_W * stride_W, stride_W, device = "cuda", dtype = torch.int), [1, 1, -1, 1]).expand(B, out_H, out_W, 1)

    idx_n2 = torch.cat([H_idx, W_idx], dim = -1).reshape(B, -1, 2)

    return idx_n2
