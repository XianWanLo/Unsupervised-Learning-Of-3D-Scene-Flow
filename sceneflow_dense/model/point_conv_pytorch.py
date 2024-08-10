import torch
import torch.nn as nn

import torch.nn.functional as F
import numpy as np
from Pointnet2.pointnet2_utils_pytorch import three_nn,three_interpolate
from Pointnet2.utils_pytorch import Conv1d,Conv2d
#from ops_pytorch.fused_conv_select_add import fused_conv_select_k_add
from ops_pytorch.fused_conv_select.fused_conv_select_k import fused_conv_select_k
from ops_pytorch.gpu_threenn_sample.no_sort_knn import no_sort_knn
from utils.geometry import resize_intrinsic



def gather_nd_torch(params, indices):
   
    indices = indices.type(torch.int64)
    gathered = params[list(indices.T)]
    gathered = gathered.permute(2,1,0,3)
    
    return gathered


class cutoff_mask (nn.Module):
    def __init__ (self, percent):
        super(cutoff_mask,self).__init__()
    
        self.percent = percent

    def forward(self,batch, height, width):
    
        random_base = torch.nn.init.uniform_(torch.cuda.FloatTensor(batch, height, width, 1), 0, 1).repeat(1, 1, 1, 3)
        percent_mask = random_base < self.percent
        cutoff_mask = percent_mask.float()
    
        return cutoff_mask

        
def fast_gather(pointCloud, indexOfHeight, indexOfWidth):
    batch, height, width, feature = pointCloud.shape

    neighborIndex = indexOfHeight * width + indexOfWidth
    inputIndex = neighborIndex.reshape(batch, -1)

    gatherFeatureProj = pointCloud.reshape(batch, -1, feature)

    # https://blog.csdn.net/cpluss/article/details/90260550
    gatherFeatureProj = torch.gather(gatherFeatureProj, 1, inputIndex.unsqueeze(-1).repeat(1, 1, feature))

    return gatherFeatureProj # bnc


def fast_select(pointCloud, indexOfHeight, indexOfWidth):
    batch, height, width, feature = pointCloud.shape

    neighborIndex = indexOfHeight * width + indexOfWidth # （B, h, w)
    inputIndex = neighborIndex.reshape(batch, -1)

    gatherFeatureProj = pointCloud.reshape(batch, -1, feature)
    
    _, out_h, out_w = indexOfHeight.shape

    # https://blog.csdn.net/cpluss/article/details/90260550
    gatherFeatureProj = torch.gather(gatherFeatureProj, 1, inputIndex.unsqueeze(-1).repeat(1, 1, feature))
    gatherFeatureProj = torch.reshape(gatherFeatureProj, [batch, out_h, out_w, -1])
    return gatherFeatureProj




#def warppingProject(pc, fx, fy, cx, cy, constx, consty, constz, out_h, out_w):
    """ 
    To project the pc to the 2D plane
    
    Args:
        @pc: (B, N, 3)    
    """

    B = pc.shape[0]
    pc_w = torch.round((pc[..., 0] * fx + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)).int()
    pc_h = torch.round((pc[..., 1] * fy + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)).int()
    width = torch.clamp((torch.reshape(pc_w, (B, -1, 1))), 0, out_w - 1)  # (B, N, 1)
    height = torch.clamp((torch.reshape(pc_h, (B, -1, 1))), 0, out_h - 1)  # (B, N, 1)
    indices = torch.cat([height, width], -1)

    #print("cx * pc[..., 2]",(cx * pc[..., 2]).shape)
    #print("pc_h",pc_h.shape)

    #print("width",width.shape)
    #print("height",height.shape)
    #print("indices",height.shape)

    return indices.contiguous()


HEIGHT = 256 
WIDTH = 832

def warppingProject(pc, intrinsic, out_h, out_w):
    """ 
    To project the pc to the 2D plane
    
    Args:
        @pc: (B, N, 3)    
    """

    B = pc.shape[0]

    pcoord = intrinsic.bmm(pc.permute(0,2,1))       # [b 3 n]
    x_idx = pcoord[:, 0] 
    y_idx = pcoord[:, 1]                        # [b n]
    z_idx = pc[..., 2].clamp(min=1e-10)

    u_proj = (x_idx/z_idx).unsqueeze(-1).int()
    v_proj = (y_idx/z_idx).unsqueeze(-1).int()
    
    width = torch.clamp(u_proj, 0, out_w - 1)  # (B, N, 1)
    height = torch.clamp(v_proj, 0, out_h - 1)  # (B, N, 1)

    indices = torch.cat([height, width], -1)

    #print("cx * pc[..., 2]",(cx * pc[..., 2]).shape)
    #print("pc_h",pc_h.shape)

    #print("width",width.shape)
    #print("height",height.shape)
    #print("indices",height.shape)

    return indices.contiguous()



class WarpingLayers(nn.Module):
    
    " add coarse dense flow into original coordinates"
    def __init__(self, kernel_size1, kernel_size2, nsample, nsample_q, in_channels, mlp1, mlp2, is_training, bn_decay, bn=True,  distance1=5, distance2=5 ) -> None:
        super(WarpingLayers, self).__init__()
        self.cost = CostVolume(kernel_size1, kernel_size2, nsample, nsample_q, \
        1, 1, distance1, [in_channels, in_channels], mlp1, mlp2, is_training, bn_decay, bn=True, pooling='max', knn=True, \
        corr_func='elementwise_product', distance2 = 100)
    
    def forward(self, upsampled_flow, xyz1_norm_proj, xyz1_raw_proj, xyz1_feature_proj, xyz2_norm_proj, xyz2_feature_proj,
                intrinsics, constxs, constys, constzs, out_h, out_w, H_input, W_input):
        """ first warp the F1 point cloud, then execute the inter- and intra- feature fusion step
        
        Args:
            @upsampled_flow: (B, N, 3)
            @xyz1_norm: (B, N, 3)
            @xyz1_raw: (B, N, 3)
            @xyz1_feature: (B, N, C)
            @xyz2_norm_proj: (B, out_h, out_w, 3)
            @xyz2_feature: (B, out_h, out_w, C)
            
        """
        batch, H, W, C = xyz1_feature_proj.shape
        
        assert H == out_h or W == out_w, "size mismatch"
        
        xyz1_raw = torch.reshape(xyz1_raw_proj, (batch, -1, 3))
        xyz1_norm = torch.reshape(xyz1_norm_proj, (batch, -1, 3))
        now_xyz_raw = xyz1_raw + upsampled_flow
        now_xyz_norm = xyz1_norm + upsampled_flow

        new_intrinsic, _ =  resize_intrinsic(intrinsics, out_h, out_w, H_input, W_input)
        projectIndexes = warppingProject(now_xyz_raw, new_intrinsic, out_h, out_w)
    
        # invalidMask = (torch.all(torch.eq(now_xyz_raw, 0), dim=-1, keepdim=True)).repeat([1, 1, 2])  # (B H*W 3)
        # maskedProjectIndexes = torch.where(invalidMask, torch.ones_like(projectIndexes) * -1, projectIndexes) # (B N 2)
        # np.savetxt("test", maskedProjectIndexes[1].detach().cpu().numpy())
        now_xyz_norm_proj = now_xyz_norm.reshape(batch, out_h, out_w, 3)
        # idx_hw = get_hw_idx(batch, out_h, out_w).contiguous()  ###  B N 2
        # pi_feat_f1_new, qi_xyz_grouped, qi_h_idx, qi_w_idx = self.inter_cost(now_xyz_norm_proj, xyz2_norm_proj, xyz2_feature_proj, now_xyz_norm, xyz1_feature, idx_hw, maskedProjectIndexes)
        # fusedFeature = self.intra_cost(now_xyz_norm_proj, now_xyz_norm, xyz1_feature, pi_feat_f1_new, idx_hw)
        fusedFeature = self.cost(now_xyz_norm_proj, xyz2_norm_proj, xyz1_feature_proj, xyz2_feature_proj, projectIndexes)

        #invalidMask = (torch.all(torch.eq(xyz1_raw, 0), dim=-1, keepdim=True)).repeat([1, 1, 2])  # (B H*W 2)
        #maskedProjectIndexes = torch.where(invalidMask, torch.ones_like(projectIndexes, dtype = torch.int32) * -1, projectIndexes) # (B N 2)
        #maskedProjectIndexes = maskedProjectIndexes.contiguous()
        #now_xyz_norm_proj = now_xyz_norm.reshape(batch, out_h, out_w, 3)
        #fusedFeature = self.cost(now_xyz_norm_proj, xyz2_norm_proj, xyz1_feature_proj, xyz2_feature_proj, maskedProjectIndexes)

        return fusedFeature
        


def get_hw_idx(B, out_H, out_W, stride_H = 1, stride_W = 1):

    H_idx = torch.reshape(torch.arange(0, out_H * stride_H, stride_H, device = "cuda", dtype = torch.int), [1, -1, 1, 1]).expand(B, out_H, out_W, 1)
    W_idx = torch.reshape(torch.arange(0, out_W * stride_W, stride_W, device = "cuda", dtype = torch.int), [1, 1, -1, 1]).expand(B, out_H, out_W, 1)

    idx_n2 = torch.cat([H_idx, W_idx], dim = -1).reshape(B, -1, 2)

    return idx_n2



class CostVolume(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, nsample, nsample_q, \
        stride_H, stride_W, distance, in_channels, mlp1, mlp2, is_training, bn_decay, bn=True, pooling='max', knn=True, \
        corr_func='concat', distance2 = 100):
        super(CostVolume,self).__init__()

        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.nsample = nsample
        self.nsample_q = nsample_q
        self.in_channels = in_channels[0] + in_channels[1] + 10
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.corr_func = corr_func
        self.distance1 = distance
        self.distance2 = distance2
        self.stride_H = stride_H
        self.stride_W = stride_W
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        self.mlp2_convs_new = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp1):
            self.mlp1_convs.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=self.bn))
            self.in_channels = num_out_channel

        self.pi_encoding = Conv2d(10,mlp1[-1],[1,1],stride=[1,1], bn=self.bn)

        self.in_channels = 2*mlp1[-1]     
        for j, num_out_channel in enumerate(mlp2):
            self.mlp2_convs.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=self.bn))
            self.in_channels = num_out_channel

        self.pc_encoding = Conv2d(10,mlp1[-1], [1,1],stride=[1,1], bn=self.bn)

        self.in_channels = 2 * mlp1[-1] + in_channels[1]
        for j,num_out_channel in enumerate(mlp2):
            self.mlp2_convs_new.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=self.bn))
            self.in_channels = num_out_channel


    def forward(self, warped_xyz1_proj, xyz2_proj, points1_proj, points2_proj, idx_fetching = None):

        B = warped_xyz1_proj.shape[0]
        H = warped_xyz1_proj.shape[1]
        W = warped_xyz1_proj.shape[2]
        C = points2_proj.shape[3]

        warped_xyz1 = warped_xyz1_proj.reshape(B, -1, 3)
        points1 = points1_proj.reshape(B, -1, points1_proj.shape[-1])

        kernel_total_q = self.kernel_size2[0] * self.kernel_size2[1]

        random_HW_q = torch.arange(0, kernel_total_q, device = "cuda", dtype = torch.int)
        idx_n2 = get_hw_idx(B, H, W, self.stride_H, self.stride_W)
        idx_hw = idx_n2.contiguous()

        #! 有warped的pc1要使用新的index 

        if idx_fetching is None:
            idx_fetching = idx_hw.detach().clone()

        # Initialize
        select_b_idx = torch.zeros(B, H*W, self.nsample_q, 1, device = 'cuda').long().detach()             # (B N nsample_q 1)
        select_h_idx = torch.zeros(B, H*W, self.nsample_q, 1, device = 'cuda').long().detach()
        select_w_idx = torch.zeros(B, H*W, self.nsample_q, 1, device = 'cuda').long().detach()

        #valid_idx = torch.zeros(B, H*W, kernel_total_q, 1, device = 'cuda').float().detach()
        #valid_in_dis_idx = torch.zeros(B, H*W, kernel_total_q, 1, device = 'cuda').float().detach()
        select_mask = torch.zeros(B, H*W, self.nsample_q, 1, device = 'cuda').float().detach()
        
        
        with torch.no_grad():
        # Sample QNN of (M neighbour points from sampled n points in PC1) in PC2
            select_b_idx, select_h_idx, select_w_idx, valid_mask = fused_conv_select_k\
                ( warped_xyz1_proj, xyz2_proj, idx_hw, idx_fetching, random_HW_q, H, W, H * W, self.kernel_size2[0], self.kernel_size2[1],\
                    self.nsample_q,  0, self.distance2, 1, 1, select_b_idx, select_h_idx, select_w_idx, select_mask, H, W)
            
        neighbor_idx = select_h_idx * W + select_w_idx    
        neighbor_idx = neighbor_idx.reshape(B, -1)
        xyz2_bn3 = xyz2_proj.reshape(B, -1, 3)
        points2_bn3 = points2_proj.reshape(B, -1, C)
        
        qi_xyz_grouped = torch.gather(xyz2_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, 3))
        qi_points_grouped = torch.gather(points2_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, C))

        # B N M 3 idx

        # Output 3D coordinates xyz2(Q) of QNN of M nearest neighbours of sampled n points in PC2
        # qi_xyz_grouped = xyz2_proj[qi_xyz_points_b_idx, qi_xyz_points_h_idx, qi_xyz_points_w_idx, : ]    

        qi_xyz_grouped = qi_xyz_grouped.reshape(B, H*W, self.nsample_q, 3) 
        qi_xyz_grouped = qi_xyz_grouped * (valid_mask.detach())                            # (B N nsample_q 3)
    
        # Output features f2(Q) of QNN of the M nearest neighbours of sampled n points in PC2
        # qi_points_grouped = points2_proj[qi_xyz_points_b_idx, qi_xyz_points_h_idx, qi_xyz_points_w_idx, : ]
        qi_points_grouped = qi_points_grouped.reshape(B, H*W, self.nsample_q, C)
        qi_points_grouped = qi_points_grouped * (valid_mask.detach())                                

        # Output 3D coordinates xyz1(M) of M nearest neighbours of sampled n points in PC1
        pi_xyz_expanded = (torch.unsqueeze(warped_xyz1, 2)).expand(B, H*W, self.nsample_q, 3)     # (B N nsample_q 3)
        
        # Output features f1(M) of M nearest neighbours of sampled n points in PC1
        pi_points_expanded = (torch.unsqueeze(points1, 2)).expand(B, H*W, self.nsample_q, points1.shape[-1])      #(B N nsample_q C)

        # xyz2(Q) - xyz1(M)
        pi_xyz_diff = qi_xyz_grouped - pi_xyz_expanded         # (B N nsample_q 3)
        
        # Euclidean difference ---- square of xyz2(Q) - xyz1(M)
        pi_euc_diff = torch.sqrt(torch.sum(torch.square(pi_xyz_diff), dim=-1 , keepdim=True) + 1e-20 )      # (B N nsample_q 1)

        # 3D Euclidean Space Information --- (3D coordinates xyz1(M) /// 3D coordinates xyz2(Q) /// xyz2(Q)-xyz1(M) /// square of xyz2(Q)-xyz1(M) )
        pi_xyz_diff_concat = torch.cat([pi_xyz_expanded, qi_xyz_grouped, pi_xyz_diff, pi_euc_diff], dim=-1)    # (B N nsample_q 3+3+3+1)
        pi_xyz_diff_concat_aft_mask = pi_xyz_diff_concat ##### mask 
        
        # Concatenate ( f2(Q) /// f1(M) /// Euclidean Space )
        pi_feat_diff = torch.cat([pi_points_expanded, qi_points_grouped],dim=-1)     # (B N nsample_q 2C)
        pi_feat1_concat = torch.cat([pi_xyz_diff_concat, pi_feat_diff], dim=-1)     # (B N nsample_q 2C+10)
        pi_feat1_concat_aft_mask = pi_feat1_concat 

        pi_feat1_new_reshape = torch.reshape(pi_feat1_concat_aft_mask, [B, H*W, self.nsample_q, -1])        # (B N nsample_q 2C+10)

        pi_xyz_diff_concat_reshape = torch.reshape(pi_xyz_diff_concat_aft_mask, [B, H*W, self.nsample_q, -1])       # (B N nsample_q 10)

        # First Flow Embedding h(Q) --- MLP(f2//f1//Eu)
        for i,conv in enumerate(self.mlp1_convs):
            pi_feat1_new_reshape = conv(pi_feat1_new_reshape)    

        # Point Position Encoding --- FC(Eu) 
        pi_xyz_encoding = self.pi_encoding(pi_xyz_diff_concat_reshape)

        # Concatenate ( FC(Eu)// h(Q) )
        pi_concat = torch.cat([pi_xyz_encoding, pi_feat1_new_reshape], dim = 3)

        # First Attentive Weight w(Q) --- MLP( FC(Eu)// h(Q) )
        for j,conv in enumerate(self.mlp2_convs):
            pi_concat = conv(pi_concat)

        valid_mask_bool = torch.eq(valid_mask, torch.ones_like(valid_mask))
        WQ_mask = valid_mask_bool.expand(B, H*W, self.nsample_q, pi_concat.shape[-1])                  #  B N K MLP[-1]     
        pi_concat_mask = torch.where(WQ_mask, pi_concat, torch.ones_like(pi_concat) * (-1e10)) 
        WQ = F.softmax(pi_concat_mask, dim=2)

        # First Attentive Flow Embedding e(M) ---  w(Q) * h(Q)
        pi_feat1_new_reshape = WQ * pi_feat1_new_reshape
        pi_feat1_new_reshape_bnc = torch.sum(pi_feat1_new_reshape, dim=2, keepdim=False)    # b, n, mlp1[-1]

        pi_feat1_new = torch.reshape(pi_feat1_new_reshape_bnc, [B, H, W, -1])  # project the selected central points to bhwc

    ##########################################################################################################################
        
        kernel_total_p = self.kernel_size1[0] * self.kernel_size1[1]

        random_HW_p = torch.arange(0,kernel_total_p, device = "cuda", dtype = torch.int)
        # nsample = kernel_size1[0] * kernel_size1[1]

        select_b_idx = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').long().detach()           # (B N nsample 1)
        select_h_idx = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').long().detach()
        select_w_idx = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').long().detach()

        #valid_idx = torch.zeros(B, H*W, kernel_total_p, 1, device = 'cuda').float().detach()
        #valid_in_dis_idx = torch.zeros(B, H*W, kernel_total_p, 1, device = 'cuda').float().detach()
        select_mask = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').float().detach()
        
        with torch.no_grad():
        # # Sample QNN of (M neighbour points from sampled n points in PC1) in PC1
            select_b_idx, select_h_idx, select_w_idx, valid_mask2 = \
                fused_conv_select_k( warped_xyz1_proj, warped_xyz1_proj, idx_hw, idx_hw, random_HW_p, H, W, H * W, self.kernel_size1[0], self.kernel_size1[1], 
                                        self.nsample,  0, self.distance1, 1, 1,select_b_idx, select_h_idx, select_w_idx,  select_mask,H,W
                )   
        
        C = pi_feat1_new.shape[3]

        neighbor_idx = select_h_idx * W + select_w_idx    
        neighbor_idx = neighbor_idx.reshape(B, -1)
        warped_xyz_bn3 = warped_xyz1_proj.reshape(B, -1, 3)
        pi_points_bn3 = pi_feat1_new.reshape(B, -1, C)
        
        pc_xyz_grouped = torch.gather(warped_xyz_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, 3))
        pc_points_grouped = torch.gather(pi_points_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, C))
        
        # Output 3D coordinates xyz1(M) of M neighbours of sampled n points in e(M) 
        # pc_points_grouped = pi_feat1_new[pc_xyz_points_b_idx, pc_xyz_points_h_idx, pc_xyz_points_w_idx, :]      # (B N nsample 1 C)
        pc_points_grouped = pc_points_grouped.reshape(B, H*W, self.nsample, C)                                   # (B N nsample C)
        pc_points_grouped= pc_points_grouped * (valid_mask2.detach())

        # Output features f1(M) of M neighbours of sampled n points in PC1
        # pc_xyz_grouped = warped_xyz1_proj[pc_xyz_points_b_idx, pc_xyz_points_h_idx, pc_xyz_points_w_idx, :]     # (B N nsample 1 3)
        pc_xyz_grouped = pc_xyz_grouped.reshape(B, H*W, self.nsample, 3)                                         # (B N nsample 3)
        pc_xyz_grouped= pc_xyz_grouped * (valid_mask2.detach())

        pc_xyz_new = torch.unsqueeze(warped_xyz1, dim = 2).expand(B,H*W,self.nsample,3)             # (B N nsample 3)
        pc_points_new = torch.unsqueeze(points1, dim = 2).expand(B,H*W,self.nsample,points1.shape[-1])              # (B N nsample C)

        # 3D Euclidean space information Eu ( xyz1 // xyz1(M) // xyz1-xyz1(M) // square of xyz1-xyz1(M) )
        pc_xyz_diff = pc_xyz_grouped - pc_xyz_new                                                               # (B N nsample 3
        pc_euc_diff = torch.sqrt(torch.sum(torch.square(pc_xyz_diff), dim=-1, keepdim=True) + 1e-20)          # (B N nsample 1)
        pc_xyz_diff_concat = torch.cat([pc_xyz_new, pc_xyz_grouped, pc_xyz_diff, pc_euc_diff], dim=-1)         # (B N nsample 3+3+3+1)

        # Point Position Encoding --- FC(Eu)
        pc_xyz_encoding = self.pc_encoding(pc_xyz_diff_concat)
    
        pc_concat = torch.cat([pc_xyz_encoding, pc_points_new, pc_points_grouped], dim = -1)
        pc_concat = pc_concat * (valid_mask2.detach())
        
        # Second Attentive Weight w(M) --- MLP( FC(Eu)// f1(M) // xyz1(M) )
        for j,conv in enumerate(self.mlp2_convs_new):
            pc_concat = conv(pc_concat)

        valid_mask2_bool = torch.eq(valid_mask2, torch.ones_like(valid_mask2)) 
        WP_mask = valid_mask2_bool.expand(B, H*W, self.nsample, pc_concat.shape[-1])   ####   B N K MLP[-1]    #####################  
        pc_concat_mask = torch.where(WP_mask, pc_concat, torch.ones_like(pc_concat) * (-1e10)) 
        WP = F.softmax(pc_concat_mask,dim=2)   #####  b, npoints, nsample, mlp[-1]

        # Final Attentive Flow Embedding --- w(M) * xyz1(M) 
        pc_feat1_new = WP * pc_points_grouped
        pc_feat1_new = torch.sum(pc_feat1_new, dim=2, keepdim=False)           # (b n self.mlp2[-1])


        return  pc_feat1_new 



class set_upconv_module(nn.Module):
    def __init__(self,kernel_size, nsample, in_channels, mlp, mlp2, is_training, bn_decay=None, bn=True, pooling='max', radius=None, knn=True, stride_H=1, stride_W=1, distance=5):
        super(set_upconv_module,self).__init__()
        
        """[上采样]

        Args:
            xyz1_proj ([type]): [description]
            xyz2_proj ([type]): [description]
            feat2_proj ([type]): [description]
            xyz1 ([type]): [frame 1 with dense points BN3]
            points1 ([type]): [frame 1 with dense points BNC]
            kernel_size ([type]): [description]
            nsample ([type]): [description]
            mlp ([type]): [description]
            mlp2 ([type]): [description]
            is_training (bool): [description]
            bn_decay ([type], optional): [description]. Defaults to None.
            bn (bool, optional): [description]. Defaults to True.
            pooling (str, optional): [description]. Defaults to 'max'.
            radius ([type], optional): [description]. Defaults to None.
            knn (bool, optional): [description]. Defaults to True.
        Returns:
            BNC feature
        """
        self.kernel_size = kernel_size
        self.nsample = nsample
        self.mlp = mlp
        self.mlp2 = mlp2
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.radius = radius
        self.knn = knn
        self.stride_h = stride_H
        self.stride_w = stride_W
        self.distance = distance

        self.last_channel = in_channels[-1] + 3  # LAST CHANNEL = C+3
        self.mlp_conv = nn.ModuleList()
        self.mlp2_conv = nn.ModuleList()

        if mlp is not None:
            for i,num_out_channel in enumerate(mlp):
                self.mlp_conv.append(Conv2d(self.last_channel,num_out_channel,[1,1],stride=[1,1], bn=True ))
                self.last_channel = num_out_channel

        if len(mlp) is not 0:
            self.last_channel = mlp[-1] + in_channels[0]
        else:
            self.last_channel = self.last_channel + in_channels[0]

        if mlp2 is not None:
            for i,num_out_channel in enumerate(mlp2):
                self.mlp2_conv.append(Conv2d(self.last_channel,num_out_channel,[1,1],stride=[1,1], bn=True))
                self.last_channel = num_out_channel

        #  xyz1 dense points         #  xyz2 sparse points (queried)

    def forward(self,xyz1_proj, xyz2_proj, points1_proj, feat2_proj):
        
        """
        xyz1_proj : xyz of n dense points  (B H1 W1 3)
        xyz2_proj : xyz of n' sparse points  (B H2 W2 3)
        feat2_proj : flow embedding f' of n' sparse points  (B H2 W2 C)
        xyz1 : [frame 1 with xyz of dense points BN3]
        points1 : [frame 1 with features f of dense points BNC]
        
        """

        #print("------------------- UpConv Start ------------------")

        B = xyz1_proj.shape[0]
        H = xyz1_proj.shape[1]
        W = xyz1_proj.shape[2]
        C = feat2_proj.shape[3]

        SMALL_H = xyz2_proj.shape[1]
        SMALL_W = xyz2_proj.shape[2]

        xyz1 = xyz1_proj.reshape(B, -1, 3)
        points1 = points1_proj.reshape(B, -1, points1_proj.shape[-1])
        
        idx_n2 = get_hw_idx(B, H, W, 1, 1)
        idx_hw = idx_n2.contiguous()    
                                   ###  B N 2
        kernel_total = self.kernel_size[0] * self.kernel_size[1]
        random_HW = torch.arange(kernel_total, device = "cuda", dtype = torch.int)

        select_b_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()       # B N n_sample 1
        select_h_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()
        select_w_idx = torch.zeros(B, H*W, self.nsample, 1).cuda().long().detach()

        #valid_idx = torch.zeros(B, H*W, kernel_total, 1).cuda().float().detach()         # B N kernel_total 1
        #valid_in_dis_idx = torch.zeros(B, H*W, kernel_total, 1).cuda().float()
        select_mask = torch.zeros(B, H*W, self.nsample, 1).cuda().float().detach()       # B N n_sample 1

        with torch.no_grad():

        # output the KNN of n dense points in n' sparse points (skip connection) 
            xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, valid_mask = fused_conv_select_k(
            xyz1_proj, xyz2_proj, idx_hw, idx_hw, random_HW, H, W, H*W, 
            self.kernel_size[0], self.kernel_size[1], self.nsample, 1, self.distance,
            self.stride_h, self.stride_w, select_b_idx, select_h_idx, select_w_idx, select_mask,SMALL_H,SMALL_W
            )

        # output the xyz1(K) of KNN points of dense n points 
        xyz1_up_grouped = xyz2_proj [xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, :]       # (B h w nsample 1 3)
        xyz1_up_grouped = xyz1_up_grouped.reshape(B, H*W, self.nsample, 3)                                                   # (B npoints nsample 3)
        xyz1_up_grouped = xyz1_up_grouped * (valid_mask.detach())
        #print("xyz1_up_grouped:",xyz1_up_grouped.shape)
        
        # output the feature f1(K) of KNN points of dense n points
        xyz1_up_points_grouped = feat2_proj [xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, :]           # (B npoints nsample 1 in_channel[-1])           
        xyz1_up_points_grouped = xyz1_up_points_grouped.reshape(B, H*W, self.nsample, C)                                                 # (B npoints nsample in_channel[-1])
        xyz1_up_points_grouped = xyz1_up_points_grouped * (valid_mask.detach())

        xyz1_expanded = torch.unsqueeze(xyz1, 2).expand(B, H*W, self.nsample, 3)      # (B, H*W, nsample, 3)
        
        #  Concatenate ( xyz1(K)-xyz1 // f1(K) )
        xyz1_diff = xyz1_up_grouped - xyz1_expanded                                 # (B, H*W, nsample, 3)
        xyz1_concat = torch.cat([xyz1_diff, xyz1_up_points_grouped], dim = -1)     # (B, H*W, nsample, in_channel[-1]+3)
        xyz1_concat_aft_mask = xyz1_concat
        xyz1_concat_aft_mask_reshape = torch.reshape(xyz1_concat_aft_mask, [B, H*W, self.nsample, -1])  # B, npoint1, nsample, in_channel[-1]+3
        
        # Point Feature Embedding -- shared MLP 
        for i,conv in enumerate(self.mlp_conv):
            xyz1_concat_aft_mask_reshape = conv(xyz1_concat_aft_mask_reshape)       # (B, H*W, nsample, mlp[-1])
     
       # Pooling in Local Regions -- max pooling (gather K points feature to 1)
        if self.pooling == 'max':
            xyz1_up_feat = torch.max(xyz1_concat_aft_mask_reshape, dim=2, keepdim=False)[0]       # (B, H*W,  mlp[-1])
        if self.pooling == 'avg':
            xyz1_up_feat = torch.mean(xyz1_concat_aft_mask_reshape, dim=2, keepdim=False)

 #############################   mlp2  ##########################

        xyz1_up_feat_concat_feat1 = torch.cat([xyz1_up_feat, points1], dim = -1)       # B  H*W  mlp[-1]+in_channel[0] 
        xyz1_up_feat_concat_feat1 = torch.unsqueeze(xyz1_up_feat_concat_feat1, 2)   # B  H*W  1 mlp[-1]+in_channel[0]  

        # Further Processing --- another MLP
        for i,conv in enumerate(self.mlp2_conv):
            xyz1_up_feat_concat_feat1 = conv(xyz1_up_feat_concat_feat1)     # B  H*W  1  mlp2[-1]

        xyz1_up_feat_concat_feat1 = torch.squeeze(xyz1_up_feat_concat_feat1, 2) # B  H*W  mlp2[-1]

        #print("xyz1_up_feat_concat_feat1:",xyz1_up_feat_concat_feat1.shape)
        #print("------------------- UpConv End ------------------")

        return xyz1_up_feat_concat_feat1


class gpu_fp_model_knn(nn.Module):
    def __init__(self, input_channel, mlp, is_training, bn_decay, kernel_size, stride_h, stride_w, distance, bn=True, last_mlp_activation=True, n_sample=3) -> None:
        super(gpu_fp_model_knn, self).__init__()
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.last_mlp_activation = last_mlp_activation
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.distance = distance
        self.nsample = n_sample
        self.kernel_size = kernel_size
        self.input_channel = input_channel
        
        self.mlp_conv = nn.ModuleList()
        
        for i, num_out_channel in enumerate(mlp):
            if i == len(mlp) - 1 and not(last_mlp_activation):
                activation_fn = False
            else:
                activation_fn = True
            self.mlp_conv.append(Conv2d(self.input_channel, num_out_channel, [1, 1], bn=bn, activation_fn=activation_fn))
            self.input_channel = num_out_channel
    
    def forward(self, xyz1_proj, xyz2_proj, feat1_proj, feat2_proj):
        B, H, W, _ = xyz1_proj.shape

        xyz1_bnc = torch.reshape(xyz1_proj, (B, -1, 3))
        idx_hw = get_hw_idx(B, H, W).contiguous()                        
        kernel_total = self.kernel_size[0] * self.kernel_size[1]
        random_HW = torch.arange(kernel_total, device = "cuda", dtype = torch.int)
   
        select_b_idx = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').long().detach()      
        select_h_idx = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').long().detach() 
        select_w_idx = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').long().detach() 

        select_mask = torch.zeros(B, H*W, self.nsample, 1, device = 'cuda').float().detach()
        
        xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, valid_mask = no_sort_knn(
            xyz1_proj, xyz2_proj, idx_hw, random_HW, H, W, H * W,
            self.kernel_size[0], self.kernel_size[1], self.nsample, 1,
            self.distance, self.stride_h, self.stride_w, 
            select_b_idx, select_h_idx, select_w_idx, select_mask)
        
        xyz1_up_grouped = fast_gather(xyz2_proj, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx)
        # xyz1_up_grouped = xyz2_proj [xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, :]       # (B npoints nsample 1 3)
        xyz1_up_grouped = xyz1_up_grouped.reshape(B, H*W, self.nsample, 3)                                                   # (B npoints nsample 3)
        xyz1_up_grouped = xyz1_up_grouped * (valid_mask.detach())
        
        xyz1_up_points_grouped = fast_gather(feat2_proj, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx)
        # xyz1_up_points_grouped = feat2_proj [xyz1_up_xyz_points_b_idx, xyz1_up_xyz_points_h_idx, xyz1_up_xyz_points_w_idx, :]           # (B npoints nsample 1 in_channel[-1])           
        xyz1_up_points_grouped = xyz1_up_points_grouped.reshape(B, H*W, self.nsample, -1)                                                 # (B npoints nsample in_channel[-1])
        xyz1_up_points_grouped = xyz1_up_points_grouped * (valid_mask.detach())
        
        xyz1_expanded = torch.unsqueeze(xyz1_bnc, 2).repeat(1, 1, self.nsample, 1)     
        dist = torch.sum(torch.square(xyz1_up_grouped - xyz1_expanded), -1)

        dist = torch.where(dist < 1e-10, torch.ones_like(dist) * 1e-10, dist)
        # dist[dist < 1e-10] = 1e-10
        norm = torch.sum((1. / dist), axis=2, keepdims=True)
        norm = norm.repeat(1, 1, 3)
        weight = (1. / dist) / norm
        
        weight = torch.unsqueeze(weight, 3)
        weighted_new_points_f1 = torch.sum(xyz1_up_points_grouped * weight, 2)
        
        if feat1_proj is not None:
            feat1_bnc = torch.reshape(feat1_proj, [B, -1, feat1_proj.shape[-1]])
            new_points = torch.cat([feat1_bnc, weighted_new_points_f1], -1)
        else:
            new_points = weighted_new_points_f1

        new_points1 = torch.unsqueeze(new_points, 2)
        for i, conv in enumerate(self.mlp_conv):
            new_points1 = conv(new_points1)
            
        new_points1 = torch.squeeze(new_points1, 2)
        return new_points1
    


class flow_predictor(nn.Module):
    def __init__(self, in_channels,mlp, is_training, bn_decay, bn=True ):
        super(flow_predictor,self).__init__()
        
        #if in_channels[2] is not None:
        #    self.in_channels = in_channels[0] + in_channels[1] + in_channels[2]
        #else:
        #     self.in_channels = in_channels[0] + in_channels[1]

        self.in_channels = in_channels
        
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.mlp_conv = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp):
            self.mlp_conv.append(Conv2d(self.in_channels, num_out_channel, [1, 1], stride=[1, 1], bn=bn))
            self.in_channels = num_out_channel
            

    def forward(self,points_f1,upsampled_feat,cost_volume):

        '''
                    Input:
                        points_f1: (b,n,c1)
                        upsampled_feat: (b,n,c2)
                        cost_volume: (b,n,c3)

                    Output:
                        points_concat:(b,n,mlp[-1])
                '''
        if upsampled_feat is not None:
            points_concat = torch.cat([points_f1, cost_volume, upsampled_feat], -1)  # b,n,c1+c2+c3
        else:
            points_concat = torch.cat([points_f1, cost_volume], -1)

        points_concat = torch.unsqueeze(points_concat, 2)  # B,n,1,c1+c2+c3

        for i, conv in enumerate(self.mlp_conv):
            points_concat = conv(points_concat)

        points_concat = torch.squeeze(points_concat, 2)

        return points_concat



class pointnet_fp_module(nn.Module):

    def __init__( self,in_channels,mlp, is_training, bn_decay,  bn=True,last_mlp_activation=True):
        super(pointnet_fp_module,self).__init__()

        self.in_channels = in_channels
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.last_mlp_activation = last_mlp_activation
        self.mlp_conv = nn.ModuleList()

        for i, num_out_channel in enumerate(mlp):

            if i == len(mlp)-1 and not(last_mlp_activation):
                activation_fn = None
            else:
                activation_fn = True

            self.mlp_conv.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1], bn=bn,activation_fn = activation_fn))
            self.in_channels = num_out_channel

    
    def forward(self,xyz1, xyz2, points1, points2):
    
        """ PointNet Feature Propogation (FP) Module

        Args:
            xyz1: (batch_size, ndataset1, 3)    xyz of  dense points in PC1 
            xyz2: (batch_size, ndataset2, 3)    xyz of  sparse points in PC1
            points1: (batch_size, ndataset1, nchannel1)     
            points2: (batch_size, ndataset2, nchannel2)     sparse flow in PC1

        Returns:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
        """

        #print("----------------------- FEATURE PROPOGATION Start -----------------------")

        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()

        dist, idx = three_nn(xyz1, xyz2)
        dist[dist < 1e-10] = 1e-10
        norm = torch.sum((1.0 / dist), dim=2, keepdim=True)
        norm = norm.repeat(1, 1, 3)
        weight = (1.0 / dist) / norm
        points2 = points2.permute(0,2,1).contiguous()

        # Output coarse dense flows from 3NN of coarse sparse flow 
        interpolated_points = three_interpolate(points2, idx, weight)
        interpolated_points = interpolated_points.permute(0,2,1)

        if points1 is not None:
            new_points1 = torch.cat([interpolated_points, points1],dim=2)  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points           # B,ndataset1,nchannel2
        
        new_points1 = torch.unsqueeze(new_points1, 2)       # B ndataset1  1 nchannel1+nchannel2 or nchannel2       

        # Point Feature Embedding --- shared MLP 
        for i,conv in enumerate(self.mlp_conv):
            new_points1 = conv(new_points1)
        
        new_points1 = torch.squeeze(new_points1, 2)  # B,ndataset1,mlp[-1]

        #print("----------------------- FEATURE PROPOGATION End ------------------------")

        return new_points1          



class PointNetSaModule(nn.Module):

    def __init__(self,  K_sample, kernel_size, H, W, stride_H, stride_W, distance, in_channels, mlp, is_training, bn_decay,
                           bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):

        super(PointNetSaModule,self).__init__()

        self.K_sample = K_sample
        self.kernel_size = kernel_size
        self.H = H; self.W = W
        self.stride_H = stride_H; self.stride_W = stride_W
        self.distance = distance
        self.in_channels = in_channels + 3
        self.mlp = mlp
        self.is_training = is_training
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw
        self.mlp_convs = nn.ModuleList()

        for i,num_out_channel in enumerate(mlp):
            self.mlp_convs.append(Conv2d(self.in_channels,num_out_channel,[1,1],stride=[1,1],bn=bn))
            self.in_channels = num_out_channel


    def forward(self, xyz_proj, points_proj, xyz_sampled_proj):
        
        B = xyz_proj.shape[0]
        H = xyz_proj.shape[1]
        W = xyz_proj.shape[2]
        C = points_proj.shape[3]

        h = xyz_sampled_proj.shape[1]
        w = xyz_sampled_proj.shape[2]
        
        idx_n2 = get_hw_idx(B, out_H = self.H, out_W = self.W, stride_H = self.stride_H, stride_W = self.stride_W) ## b -1 2

        kernel_total = self.kernel_size[0] * self.kernel_size[1]
    
        n_sampled = idx_n2.shape[1]    #  n_sampled = h*w

        random_HW = (torch.arange(self.kernel_size[0] * self.kernel_size[1])).int().cuda()
        
        ##randperm 
        
        #################   fused_conv   default  padding = 'same'

        select_b_idx = torch.zeros(B, n_sampled, self.K_sample, 1, device = 'cuda').long().detach()       # (B, n_sampled, K_sampled, 1)
        select_h_idx = torch.zeros(B, n_sampled, self.K_sample, 1, device = 'cuda').long().detach()
        select_w_idx = torch.zeros(B, n_sampled, self.K_sample, 1, device = 'cuda').long().detach()

        #valid_idx = torch.zeros(B, n_sampled, kernel_total, 1, device = 'cuda').float().detach()                  # (B, n_sampled, H*W, 1)    
        #valid_in_dis_idx = torch.zeros(B, n_sampled, kernel_total, 1, device = 'cuda').float().detach()
        valid_mask = torch.zeros(B, n_sampled, self.K_sample, 1, device = 'cuda').float().detach()     # (B, n_sampled, K_sampled, 1)

        idx_n2_part = idx_n2.contiguous()   # (B N 2)

        with torch.no_grad():
        # Sample n' points from input n points 
            select_b_idx, select_h_idx, select_w_idx, valid_mask = \
            fused_conv_select_k(xyz_proj, xyz_proj, idx_n2_part, idx_n2_part, random_HW, H, W, n_sampled, self.kernel_size[0], self.kernel_size[1], self.K_sample, 0, self.distance,\
                1, 1, select_b_idx, select_h_idx, select_w_idx, valid_mask, H, W)
            
        neighbor_idx = select_h_idx * W + select_w_idx    
        neighbor_idx = neighbor_idx.reshape(B, -1)
        xyz_bn3 = xyz_proj.reshape(B, -1, 3)
        points_bn3 = points_proj.reshape(B, -1, C)
        
        new_xyz_group = torch.gather(xyz_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, 3))
        new_points_group = torch.gather(points_bn3, 1, neighbor_idx.unsqueeze(-1).repeat(1, 1, C))
        
        # Directly output 3D coordinate xyz of KNN of sampled point p' with mask
        # new_xyz_group = xyz_proj[select_b_idx, select_h_idx, select_w_idx, : ]          # (B, n_sampled, K_sampled, 1, 3) 
        new_xyz_group = new_xyz_group.reshape(B,n_sampled,self.K_sample,3)              # (B, n_sampled, K_sampled, 3)
        new_xyz_group = new_xyz_group * (valid_mask.detach())                                   # (B, n_sampled, K_sampled, 3)

        # new_points_group = points_proj[select_b_idx, select_h_idx, select_w_idx, : ]    # (B, n_sampled, K_sampled, 1, 3) 
        new_points_group = new_points_group.reshape(B,n_sampled,self.K_sample,C)        # (B, n_sampled, K_sampled, C)
        new_points_group = new_points_group * (valid_mask.detach())                                 # (B, n_sampled, K_sampled, C)
        
        #  Directly output 3D coordinate xyz' of sampled point p' with added values
        new_xyz_proj = xyz_sampled_proj                              # (B, h, w, 3) 
       
        new_xyz = new_xyz_proj.reshape(B,-1,3)                                          # B, n_sampled, 3
        new_xyz_expand = torch.unsqueeze(new_xyz,2).expand(B, h*w ,self.K_sample,3)         # B, n_sampled, K_sampled, 3
        
        xyz_diff = new_xyz_group - new_xyz_expand                                           # (B, n_sampled, K_sampled, 3)
        new_points_group_concat = torch.cat([xyz_diff, new_points_group], dim = -1)        # (B, n_sampled, K_sampled, C+3) 
        
        # Point Feature Embedding -- shared MLP  
        for i,conv in enumerate(self.mlp_convs):
            new_points_group_concat = conv(new_points_group_concat)                         # (B, n_sampled, K_sampled, mlp[-1])              

        # Pooling in Local Regions -- max pooling (gather K points feature to 1)
        if self.pooling=='max':
            new_points_group_concat = torch.max(new_points_group_concat, dim=2, keepdim=True)[0]      # (B, n_sampled, 1, mlp[-1])

        elif self.pooling=='avg':
            new_points_group_concat = torch.mean(new_points_group_concat, dim=2, keepdim=True)     # (B, n_sampled, 1, mlp[-1])

        points_down_sample = torch.squeeze(new_points_group_concat, 2)           # (B, n_sampled, mlp2[-1])
        points_down_sample_proj = torch.reshape(points_down_sample, [B, h, w, -1])           # (B, n_sampled, mlp2[-1])
        
        return points_down_sample, points_down_sample_proj


    