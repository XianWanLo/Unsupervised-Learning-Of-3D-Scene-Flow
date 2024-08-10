import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from utils.inverse_warp import BackprojectDepth


def DepthConsistencyLoss(sf_constant,sf_new,intrinsic,inv_intrinsic,depth2, height,width ,weight, img1, img2, valid_mask):
        """
        sf_constant: b 3 n
        sf_new: b 3 n
        valid_mask : b h w 
        """
        b = sf_new.shape[0]        
        H = height 
        W = width 

        pcoords = intrinsic.bmm(sf_new)         # [b 3 212992]     #TODO摄像机2下的像素坐标（齐次）* XYZ    (3D相机坐标变换到2D齐次图像坐标)
            
        X = pcoords[:, 0]                       # [b 212992]
        Y = pcoords[:, 1]                   
        Z = pcoords[:, 2].clamp(min=1e-3)   

        X_norm = 2 * (X / Z) / (W - 1) - 1      # [b 212992]
        Y_norm = 2 * (Y / Z) / (H - 1) - 1 
        Z = sf_constant[:, 2:3, :].clamp(min=1e-3) 
        
        grid_x = X_norm.reshape(b,H,W).cpu()
        grid_y = Y_norm.reshape(b,H,W).cpu()
        grid_tf = torch.stack((grid_x, grid_y), dim=3).cuda()          # [b 235 832 2]
        
        mask_x_low = torch.BoolTensor(grid_x>-1).byte()
        mask_x_high = torch.BoolTensor(grid_x<1).byte()
        mask_y_low = torch.BoolTensor(grid_y>-1).byte()
        mask_y_high = torch.BoolTensor(grid_y<1).byte()
        mask = (mask_x_low * mask_x_high * mask_y_low * mask_y_high).cuda()
        mask = mask.view(b,1,H,W).repeat(1,3,1,1)

        #TODO interpolated depth map (双线性插值)
        pro_depth = torch.nn.functional.grid_sample(depth2, grid_tf, padding_mode='border')   #[b 1 235 832]
        pro_xyz = BackprojectDepth(pro_depth, inv_intrinsic)                    # [b 3 235 832]
        pro_xyz_masked = pro_xyz * mask 
        pro_xyz_flatten = pro_xyz_masked.view(b, 3, -1)                         # [b 3 212992]   

        pro_img = torch.nn.functional.grid_sample(img2, grid_tf, padding_mode='border')
        pro_img_masked = pro_img * mask
        pro_img_flatten = pro_img_masked.view(b, 3, -1)

        tgt_img_masked = img1 * mask
        tgt_flatten = tgt_img_masked.view(b, 3, -1)

        #final_mask = valid_mask * (mask.reshape(b,3,-1))        # b 3 n 
        
        depth_consistency = abs(pro_xyz_flatten - sf_constant)           # b 3 n
        color_consistency = abs(pro_img_flatten - tgt_flatten)
        
        #valid_num = final_mask.sum(dim=1).sum(dim=1).mean()

        valid = (valid_mask>0).reshape(-1)      # bxn=N

        depth_consistency_n3 = (depth_consistency.permute(0,2,1).reshape(-1,3))[valid,:]       
        color_consistency_n3 = (color_consistency.permute(0,2,1).reshape(-1,3))[valid,:]

        print(depth_consistency_n3.shape)

        loss4 = depth_consistency_n3.sum(dim=1).mean()* weight    
        loss9 = color_consistency_n3.sum(dim=1).mean()* weight
        
        #print("loss4:",loss4)
        #print("loss9:",loss9)

        return loss4, loss9 
