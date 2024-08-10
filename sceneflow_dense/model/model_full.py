import os
import sys
import math
import time
import torch
import torch.nn as nn
from .point_conv_pytorch import PointNetSaModule, cutoff_mask,set_upconv_module, CostVolume, flow_predictor,WarpingLayers, gpu_fp_model_knn, fast_select,fast_gather,PointNetSaModule,warppingProject
from Pointnet2.utils_pytorch import Conv1d
from utils.geometry import resize_intrinsic

MAX_SQUARE_DIST = 1225
MAX_DEPTH_LIMIT = 90.0

HEIGHT = 256 
WIDTH = 832

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# efficient stride based sceneflow model
class ESBFlow(nn.Module):
    def __init__(self,
                 H_input,
                 W_input,
                 is_training,
                 bn_decay=None):
        super(ESBFlow, self).__init__()
        #####   initialize the parameters (distance  &  stride ) ######
        self.training = is_training
        self.bn_decay = bn_decay
        self.H_input = H_input
        self.W_input = W_input
        # 270 x 480
        #! 256 x 832 

        #self.Down_conv_dis = [3.0, 3.0, 6.0, 9.0, 9.0, 15.0]
        #self.Up_conv_dis = [3.0, 6.0, 9.0]
        #self.Cost_volume_dis = [10.0, 10.0, 10.0]
        #drop_prob = 0.1
        self.Down_conv_dis = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        self.Up_conv_dis = [100.0, 100.0, 100.0]
        self.Cost_volume_dis = [100.0, 100.0, 100.0]

        self.stride_H_list = [ 2, 2, 2, 2, 1]
        self.stride_W_list = [ 4, 2, 2, 2, 1]
        
        n_times = len(self.stride_H_list)
        
        self.out_H_list = [math.ceil(self.H_input / self.stride_H_list[0])]
        self.out_W_list = [math.ceil(self.W_input / self.stride_W_list[0])]


        for i in range(1, n_times):
            self.out_H_list.append(math.ceil(self.out_H_list[i - 1] / self.stride_H_list[i]))
            self.out_W_list.append(math.ceil(self.out_W_list[i - 1] / self.stride_W_list[i]))


        # 90 x 120 --- pholy
        #! 128 x 208
        self.layer0 = PointNetSaModule(K_sample = 32, kernel_size = [12, 20], distance = self.Down_conv_dis[0], in_channels = 3,
                                       mlp = [8, 8, 16], H = self.out_H_list[0], W = self.out_W_list[0], 
                                       stride_H = self.stride_H_list[0], stride_W = self.stride_W_list[0],
                                       is_training = self.training, bn_decay = self.bn_decay)
        # 45 x 60
        #! 64 x 104
        self.layer1 = PointNetSaModule(K_sample = 24, kernel_size = [12, 20], distance = self.Down_conv_dis[1], in_channels = 16,
                                       mlp = [16, 16, 32], H = self.out_H_list[1], W = self.out_W_list[1], 
                                       stride_H = self.stride_H_list[1], stride_W = self.stride_W_list[1],   
                                        is_training=self.training, bn_decay = self.bn_decay)
        # out : 27 x 30
        #! 32 x 52
        self.layer2 = PointNetSaModule(K_sample = 16, kernel_size = [12, 20], distance = self.Down_conv_dis[2], in_channels=32,
                                       mlp=[32, 32, 64], H = self.out_H_list[2], W = self.out_W_list[2], 
                                       stride_H = self.stride_H_list[2], stride_W = self.stride_W_list[2], 
                                       is_training=self.training, bn_decay= self.bn_decay)
        # out: 13 x 15
        #! 16 x 26
        self.layer3_2 = PointNetSaModule(K_sample = 16, kernel_size = [10, 18], distance = self.Down_conv_dis[3], in_channels=64,
                                         mlp=[64, 64, 128], H = self.out_H_list[3], W = self.out_W_list[3], 
                                         stride_H = self.stride_H_list[3], stride_W = self.stride_W_list[3], 
                                         is_training=self.training, bn_decay=self.bn_decay)
        # out: 13 x 15
        #! 16 x 26
        self.layer3_1 = PointNetSaModule(K_sample = 16, kernel_size = [10, 18], distance = self.Down_conv_dis[3], in_channels=64,
                                         mlp=[64, 64, 128], H = self.out_H_list[3], W = self.out_W_list[3], 
                                         stride_H = self.stride_H_list[3], stride_W = self.stride_W_list[3], 
                                         is_training=self.training, bn_decay=self.bn_decay)
        # out: 13 x 15
        #! 16 x 26
        self.layer4_1 = PointNetSaModule(K_sample = 16, kernel_size = [10, 18], distance = self.Down_conv_dis[4], in_channels=128,
                                         mlp=[128, 128, 256], H = self.out_H_list[4], W = self.out_W_list[4], 
                                         stride_H = self.stride_H_list[4], stride_W = self.stride_W_list[4], 
                                         is_training=self.training, bn_decay=self.bn_decay)
        # in:45 x 60
        #! 32 x 52
        self.cost_volume1 = CostVolume(kernel_size1 = [14, 26], kernel_size2 = [22, 40] , nsample = 4, nsample_q= 32, 
                                       stride_H = 1, stride_W = 1, distance = self.Cost_volume_dis[2],
                                       in_channels = [64, 64],
                                       mlp1=[128, 64, 64], mlp2=[128, 64], is_training=self.training, bn_decay=self.bn_decay,
                                       bn=True, pooling='max', knn=True, corr_func='concat')

        #! 16 x 26
        # upconv3: in_channels = layer3 + layer4
        self.upconv3 = set_upconv_module(kernel_size=[10, 18], 
                                        stride_H=self.stride_H_list[4], stride_W=self.stride_W_list[4], 
                                        nsample=8, distance=self.Up_conv_dis[2], 
                                        in_channels=[128, 256], 
                                        mlp=[128, 128, 256],  mlp2=[256], is_training=self.training, 
                                        bn_decay=bn_decay, knn=True)
        self.conv3_1 = Conv1d(256, 3)
        # warping: in_channels= layer3
        self.warping3 = WarpingLayers(kernel_size1=[10, 18], kernel_size2=[14, 26], nsample=4, nsample_q=8, in_channels=128, mlp1=[256, 128, 128], 
                                      mlp2=[256, 128], is_training=is_training, bn_decay=bn_decay, bn=True, distance1=100, distance2=100)
        # flow_pred: in_channels = layer3 + upconv3 + warping3
        self.flow_pred3 = flow_predictor(in_channels=128 + 256 + 128, mlp=[256, 128, 128], is_training=self.training, bn_decay=bn_decay)
        self.conv3_2 = Conv1d(128, 3)
        
        #self.drop3 = nn.Dropout(drop_prob)
        
        #! 32 x 52
        self.upconv2 = set_upconv_module(kernel_size = [12, 20],
                                        stride_H = self.stride_H_list[3], stride_W = self.stride_W_list[3], \
                                        nsample=8, distance = self.Up_conv_dis[1],
                                        in_channels=[64, 128],
                                        mlp=[128, 64, 64], mlp2=[64], is_training=self.training,
                                        bn_decay=bn_decay, knn=True)
        self.fp2 = gpu_fp_model_knn(3, [], is_training, bn_decay, [12,20], self.stride_H_list[3], self.stride_W_list[3], 100)
        self.warping2 = WarpingLayers(kernel_size1=[12, 20], kernel_size2=[22, 40], nsample=4, nsample_q=8, in_channels=64, mlp1=[128, 64, 64], 
                                mlp2=[128, 64], is_training=is_training, bn_decay=bn_decay, bn=True, distance1=100, distance2=100)
        self.flow_pred2 = flow_predictor(in_channels= 64 + 64 + 64, mlp=[128, 64, 64], is_training=self.training, bn_decay=bn_decay)
        self.conv2 = Conv1d(64, 3)
        #self.drop2 = nn.Dropout(drop_prob)
        
        #! 64 x 104
        self.upconv1 = set_upconv_module(kernel_size = [12, 20],
                                        stride_H = self.stride_H_list[2], stride_W = self.stride_W_list[2], \
                                        nsample=8, distance = self.Up_conv_dis[0],
                                        in_channels=[32, 64],
                                        mlp=[128, 64, 64], mlp2=[64], is_training=self.training,
                                        bn_decay=bn_decay, knn=True)
        self.fp1 = gpu_fp_model_knn(3, [], is_training, bn_decay, [12,20], self.stride_H_list[2], self.stride_W_list[2], 100)
        self.warping1 = WarpingLayers(kernel_size1=[12, 20], kernel_size2=[22, 40], nsample=4, nsample_q=8, in_channels=32, mlp1=[64, 32, 32], 
                                mlp2=[64, 32], is_training=is_training, bn_decay=bn_decay, bn=True, distance1=100, distance2=100)
        self.flow_pred1 = flow_predictor(in_channels=32 + 64 + 32, mlp=[128, 64, 64], is_training=self.training, bn_decay=bn_decay)
        self.conv1 = Conv1d(64, 3)
        #self.drop1 = nn.Dropout(drop_prob)
        
        # self.fp0_1 = gpu_fp_model_knn(160, [256, 256], is_training, bn_decay, [9, 13], self.stride_H_list[1], self.stride_W_list[1], 100)
        # self.conv0_1 = Conv1d(256, 128)
        # self.fp0_2 = gpu_fp_model_knn(3, [], is_training, bn_decay, [9, 13], self.stride_H_list[1], self.stride_W_list[1], 100)
        # self.conv0_2 = Conv1d(128, 3)

        #! 128 x 208
        self.upconv0 = set_upconv_module(kernel_size = [12, 20],
                                        stride_H = self.stride_H_list[1], stride_W = self.stride_W_list[1], \
                                        nsample=8, distance = self.Up_conv_dis[0],
                                        in_channels=[16, 64],
                                        mlp=[64, 64, 32], mlp2=[32], is_training=self.training,
                                        bn_decay=bn_decay, knn=True)
        self.fp0 = gpu_fp_model_knn(3, [], is_training, bn_decay, [12,20], self.stride_H_list[1], self.stride_W_list[1], 100)
        self.warping0 = WarpingLayers(kernel_size1=[12,20], kernel_size2=[22, 40], nsample=4, nsample_q=8, in_channels=16, mlp1=[32, 16, 16], 
                                mlp2=[32, 16], is_training=is_training, bn_decay=bn_decay, bn=True, distance1=100, distance2=100)
        self.flow_pred0 = flow_predictor(in_channels=16 + 32 + 16, mlp=[64, 32], is_training=self.training, bn_decay=bn_decay)
        self.conv0 = Conv1d(32, 3)
        #self.drop0 = nn.Dropout(drop_prob)

        self.fp0_1 = gpu_fp_model_knn(3 + 32, [64, 64], is_training, bn_decay, [25, 39], self.stride_H_list[0], self.stride_W_list[0], 100)
        self.conv0_1 = Conv1d(64, 32)
        self.fp0_2 = gpu_fp_model_knn(3, [], is_training, bn_decay, [25, 39], self.stride_H_list[0], self.stride_W_list[0], 100)
        self.conv0_2 = Conv1d(32, 3)

        
    def forward(self, pos1_bhw3, pos2_bhw3, color1, color2, intrinsics, occ_mask , label=None, train=False, warped=False):
        
        """
        pos,color : B H W 3
        intrinsics: 
        occ_mask : B H W 1
        label :
        
        """
        batch_size = pos1_bhw3.shape[0]

        constxs, constys, constzs = (0,0,0)
        
        #depth1_mask = (pos1_bhw3[..., 2] < MAX_DEPTH_LIMIT).unsqueeze(-1).expand(-1, -1, -1, 3)
        #depth2_mask = (pos2_bhw3[..., 2] < MAX_DEPTH_LIMIT).unsqueeze(-1).expand(-1, -1, -1, 3)
        #input_xyz_f1_raw = torch.where(depth1_mask, pos1_bhw3, torch.zeros_like(pos1_bhw3))
        #input_xyz_f2_raw = torch.where(depth2_mask, pos2_bhw3, torch.zeros_like(pos2_bhw3))  
        
        input_xyz_f1_raw = pos1_bhw3
        input_xyz_f2_raw = pos2_bhw3
        
        #print("mask",(pos1_bhw3[..., 2] < MAX_DEPTH_LIMIT).sum())
        #occ_mask = occ_mask.unsqueeze(-1)
        
        #mean_pc1 = torch.mean(input_xyz_f1_raw, dim=1, keepdims=True)
        #mean_pc1 = torch.mean(mean_pc1, dim=2, keepdims=True)
        #pos1_bhw3_norm = pos1_bhw3 - mean_pc1
        #pos2_bhw3_norm = pos2_bhw3 - mean_pc1

        input_xyz_f1_norm = (input_xyz_f1_raw).detach().clone() # [B H W 3]
        input_xyz_f2_norm = input_xyz_f2_raw.detach().clone()
        
        #! 这里应该放 RGB / RGB+XYZ 
        #input_points_proj_f1 = torch.where(depth1_mask, color1, torch.zeros_like(color1))
        #input_points_proj_f2 = torch.where(depth2_mask, color2, torch.zeros_like(color2))
        input_points_proj_f1 = input_xyz_f1_norm.detach().clone()
        input_points_proj_f2 = input_xyz_f2_norm.detach().clone()

        #input_xyz_f1_bn3 = input_xyz_f1_raw.reshape(batch_size, -1, 3)
        #input_xyz_f2_bn3 = input_xyz_f2_raw.reshape(batch_size, -1, 3)

        #color_proj_f1 = color1.reshape(batch_size, -1, 3)
        #color_proj_f2 = color2.reshape(batch_size, -1, 3)

        _, l0_h_idx, l0_w_idx = get_selected_idx(batch_size, self.out_H_list[0], self.out_W_list[0], self.stride_H_list[0], self.stride_W_list[0])
        _, l1_h_idx, l1_w_idx = get_selected_idx(batch_size, self.out_H_list[1], self.out_W_list[1], self.stride_H_list[1], self.stride_W_list[1])
        _, l2_h_idx, l2_w_idx = get_selected_idx(batch_size, self.out_H_list[2], self.out_W_list[2], self.stride_H_list[2], self.stride_W_list[2])
        _, l3_h_idx, l3_w_idx = get_selected_idx(batch_size, self.out_H_list[3], self.out_W_list[3], self.stride_H_list[3], self.stride_W_list[3])
        _, l4_h_idx, l4_w_idx = get_selected_idx(batch_size, self.out_H_list[4], self.out_W_list[4], self.stride_H_list[4], self.stride_W_list[4])

        if label is not None:
            #input_label = torch.where(depth1_mask, label, torch.zeros_like(label)) 
            input_label = label
            l0_label_proj = fast_select(input_label, l0_h_idx, l0_w_idx)
            l0_label = l0_label_proj.reshape(batch_size, -1, 3)
            l1_label_proj = fast_select(l0_label_proj, l1_h_idx, l1_w_idx)
            l1_label = l1_label_proj.reshape(batch_size, -1, 3)
            l2_label_proj = fast_select(l1_label_proj, l2_h_idx, l2_w_idx)
            l2_label = l2_label_proj.reshape(batch_size, -1, 3)
            l3_label_proj = fast_select(l2_label_proj, l3_h_idx, l3_w_idx)
            l3_label = l3_label_proj.reshape(batch_size, -1, 3)
            
            #label_list = [input_label_bn3, l0_label, l1_label, l2_label, l3_label]
            label_list = [ l0_label, l1_label, l2_label, l3_label]

        ##################  the l0 select xyz, label    #################################
        l0_xyz_proj_f1 = fast_select(input_xyz_f1_norm, l0_h_idx, l0_w_idx)
        l0_xyz_proj_f1_raw = fast_select(input_xyz_f1_raw, l0_h_idx, l0_w_idx)
        l0_xyz_proj_f2 = fast_select(input_xyz_f2_norm, l0_h_idx, l0_w_idx)
        l0_xyz_proj_f2_raw = fast_select(input_xyz_f2_raw, l0_h_idx, l0_w_idx)
        #l0_label_proj = fast_select(input_label, l0_h_idx, l0_w_idx)
        
        l0_occ_mask = fast_select(occ_mask, l0_h_idx, l0_w_idx)
                
        #l0_label = l0_label_proj.reshape(batch_size, -1, 3)
        l0_xyz_f1_raw = l0_xyz_proj_f1_raw.reshape(batch_size, -1, 3)
        l0_xyz_f2_raw = l0_xyz_proj_f2_raw.reshape(batch_size, -1, 3)

        #! RGB1 & RGB2 也需要进行 select ,以便后续color consistency loss能计算
        l0_color_proj_f1 = fast_select(color1, l0_h_idx, l0_w_idx)
        l0_color_f1 = l0_color_proj_f1.reshape(batch_size, -1, 3)
        l0_color_proj_f2 = fast_select(color2, l0_h_idx, l0_w_idx)
        l0_color_f2 = l0_color_proj_f2.reshape(batch_size, -1, 3)

       ##################  the l1 select xyz, label    ##################################
        l1_xyz_proj_f1 = fast_select(l0_xyz_proj_f1, l1_h_idx, l1_w_idx)
        l1_xyz_proj_f1_raw = fast_select(l0_xyz_proj_f1_raw, l1_h_idx, l1_w_idx)
        l1_xyz_proj_f2 = fast_select(l0_xyz_proj_f2, l1_h_idx, l1_w_idx)
        l1_xyz_proj_f2_raw = fast_select(l0_xyz_proj_f2_raw, l1_h_idx, l1_w_idx)
        #l1_label_proj = fast_select(l0_label_proj, l1_h_idx, l1_w_idx)
       
        l1_occ_mask = fast_select(l0_occ_mask, l1_h_idx, l1_w_idx)
        
        #l1_label = l1_label_proj.reshape(batch_size, -1, 3)
        l1_xyz_f1_raw = l1_xyz_proj_f1_raw.reshape(batch_size, -1, 3)
        l1_xyz_f2_raw = l1_xyz_proj_f2_raw.reshape(batch_size, -1, 3)
        
        ##################  the l2 select xyz, label    ###################

        l2_xyz_proj_f1 = fast_select(l1_xyz_proj_f1, l2_h_idx, l2_w_idx)
        l2_xyz_proj_f1_raw = fast_select(l1_xyz_proj_f1_raw, l2_h_idx, l2_w_idx)
        l2_xyz_proj_f2 = fast_select(l1_xyz_proj_f2, l2_h_idx, l2_w_idx)
        l2_xyz_proj_f2_raw = fast_select(l1_xyz_proj_f2_raw, l2_h_idx, l2_w_idx)
        #l2_label_proj = fast_select(l1_label_proj, l2_h_idx, l2_w_idx)
         
        l2_occ_mask = fast_select(l1_occ_mask, l2_h_idx, l2_w_idx)
        
        #l2_label = l2_label_proj.reshape(batch_size, -1, 3)
        l2_xyz_f1_raw = l2_xyz_proj_f1_raw.reshape(batch_size, -1, 3)
        l2_xyz_f2_raw = l2_xyz_proj_f2_raw.reshape(batch_size, -1, 3)

        ##################  the l3 select xyz, label    ###################

        l3_xyz_proj_f1 = fast_select(l2_xyz_proj_f1, l3_h_idx, l3_w_idx)
        l3_xyz_proj_f1_raw = fast_select(l2_xyz_proj_f1_raw, l3_h_idx, l3_w_idx)
        l3_xyz_proj_f2 = fast_select(l2_xyz_proj_f2, l3_h_idx, l3_w_idx)
        l3_xyz_proj_f2_raw = fast_select(l2_xyz_proj_f2_raw, l3_h_idx, l3_w_idx)
        #l3_label_proj = fast_select(l2_label_proj, l3_h_idx, l3_w_idx)
        
        l3_occ_mask = fast_select(l2_occ_mask, l3_h_idx, l3_w_idx)
        
        #l3_label = l3_label_proj.reshape(batch_size, -1, 3)
        l3_xyz_f1_raw = l3_xyz_proj_f1_raw.reshape(batch_size, -1, 3)
        l3_xyz_f2_raw = l3_xyz_proj_f2_raw.reshape(batch_size, -1, 3)
        
        ##################  the l4 select bhw3 xyz  #######################
         
        l4_xyz_proj_f1 = fast_select(l3_xyz_proj_f1, l4_h_idx, l4_w_idx)
        
        ################################ Frame 1 ######################################################
        
        l0_points_f1, l0_points_proj_f1 = self.layer0(input_xyz_f1_norm, input_points_proj_f1, l0_xyz_proj_f1)
        l1_points_f1, l1_points_proj_f1 = self.layer1(l0_xyz_proj_f1, l0_points_proj_f1, l1_xyz_proj_f1)
        l2_points_f1, l2_points_proj_f1 = self.layer2(l1_xyz_proj_f1, l1_points_proj_f1, l2_xyz_proj_f1)

        ################################ Frame 2 ######################################################

        _, l0_points_proj_f2 = self.layer0(input_xyz_f2_norm, input_points_proj_f2, l0_xyz_proj_f2)
        _, l1_points_proj_f2 = self.layer1(l0_xyz_proj_f2, l0_points_proj_f2, l1_xyz_proj_f2)
        _, l2_points_proj_f2 = self.layer2(l1_xyz_proj_f2, l1_points_proj_f2, l2_xyz_proj_f2)
        _, l3_points_proj_f2 = self.layer3_2(l2_xyz_proj_f2, l2_points_proj_f2, l3_xyz_proj_f2)

        ############################### first cost_volume #############################################
        
        if warped == True:
            
            #fxs, fys, cxs, cys = intrinsics
            #N = self.out_H_list[2]*self.out_W_list[2]

            #fxs = (fxs / (WIDTH/self.out_W_list[2])).unsqueeze(-1).repeat(1, N)     # extend from original code
            #cxs = (cxs / (WIDTH/self.out_W_list[2])).unsqueeze(-1).repeat(1, N)
            #fys = (fys / (HEIGHT/self.out_H_list[2])).unsqueeze(-1).repeat(1, N)
            #cys = (cys / (HEIGHT/self.out_H_list[2])).unsqueeze(-1).repeat(1, N)

            new_intrinsic, _ = resize_intrinsic(intrinsics, self.out_H_list[2], self.out_W_list[2] ,HEIGHT, WIDTH)
            idx_fetching = warppingProject(l2_xyz_proj_f1.reshape(batch_size, -1, 3), new_intrinsic, self.out_H_list[2], self.out_W_list[2])
        
        else:
            idx_fetching = None

        l2_points_f1_new = self.cost_volume1(l2_xyz_proj_f1, l2_xyz_proj_f2, l2_points_proj_f1, l2_points_proj_f2, idx_fetching)
        l2_points_f1_new_proj = torch.reshape(l2_points_f1_new,[batch_size, self.out_H_list[2], self.out_W_list[2], -1])

        ############################### Layer3 & 4 Rectification #####################################

        l3_points_f1, l3_points_proj_f1 = self.layer3_1(l2_xyz_proj_f1, l2_points_f1_new_proj, l3_xyz_proj_f1)
        _, l4_points_proj_f1 = self.layer4_1(l3_xyz_proj_f1, l3_points_proj_f1, l4_xyz_proj_f1)


        ############################### Layer 3 Upconv #############################################
        
        l3_points_f1_new = self.upconv3(l3_xyz_proj_f1, l4_xyz_proj_f1, l3_points_proj_f1, l4_points_proj_f1)
        l3_flow_coarse = self.conv3_1(l3_points_f1_new)  
        l3_cost_volume = self.warping3(l3_flow_coarse, l3_xyz_proj_f1, l3_xyz_proj_f1_raw, l3_points_proj_f1, 
                                       l3_xyz_proj_f2, l3_points_proj_f2, intrinsics, 
                                       constxs, constys, constzs, self.out_H_list[3], self.out_W_list[3],self.H_input,self.W_input)

        l3_flow_finer = self.flow_pred3(l3_points_f1, l3_points_f1_new, l3_cost_volume)
        l3_flow_finer_proj = torch.reshape(l3_flow_finer, (batch_size, self.out_H_list[3], self.out_W_list[3], -1))
        #l3_flow_det = self.drop3(self.conv3_2(l3_flow_finer))
        l3_flow_det = self.conv3_2(l3_flow_finer)
        l3_flow = l3_flow_coarse + l3_flow_det
        l3_flow_proj = torch.reshape(l3_flow, (batch_size, self.out_H_list[3], self.out_W_list[3], 3))
        
        ############################### Layer 2 Upconv #############################################

        l2_points_f1_new = self.upconv2(l2_xyz_proj_f1, l3_xyz_proj_f1, l2_points_proj_f1, l3_flow_finer_proj)
        l2_flow_coarse = self.fp2(l2_xyz_proj_f1, l3_xyz_proj_f1, None, l3_flow_proj)
        l2_cost_volume = self.warping2(l2_flow_coarse, l2_xyz_proj_f1, l2_xyz_proj_f1_raw, l2_points_proj_f1,
                                       l2_xyz_proj_f2, l2_points_proj_f2, intrinsics,
                                       constxs, constys, constzs, self.out_H_list[2], self.out_W_list[2], self.H_input,self.W_input)
        
        l2_flow_finer = self.flow_pred2(l2_points_f1, l2_points_f1_new, l2_cost_volume)
        l2_flow_finer_proj = torch.reshape(l2_flow_finer, (batch_size, self.out_H_list[2], self.out_W_list[2], -1))
        #l2_flow_det = self.drop2(self.conv2(l2_flow_finer))
        l2_flow_det = self.conv2(l2_flow_finer)
        l2_flow = l2_flow_coarse + l2_flow_det
        l2_flow_proj = torch.reshape(l2_flow, (batch_size, self.out_H_list[2], self.out_W_list[2], 3))
      
        ############################### Layer 1 Upconv #############################################

        l1_points_f1_new = self.upconv1(l1_xyz_proj_f1, l2_xyz_proj_f1,l1_points_proj_f1, l2_flow_finer_proj)
        l1_flow_coarse = self.fp1(l1_xyz_proj_f1, l2_xyz_proj_f1, None, l2_flow_proj)
        l1_cost_volume = self.warping1(l1_flow_coarse, l1_xyz_proj_f1, l1_xyz_proj_f1_raw, l1_points_proj_f1, 
                                       l1_xyz_proj_f2, l1_points_proj_f2, intrinsics,
                                       constxs, constys, constzs, self.out_H_list[1], self.out_W_list[1], self.H_input,self.W_input)
        
        l1_flow_finer = self.flow_pred1(l1_points_f1, l1_points_f1_new, l1_cost_volume)
        l1_flow_finer_proj = torch.reshape(l1_flow_finer, (batch_size, self.out_H_list[1], self.out_W_list[1], -1))
        #l1_flow_det = self.drop1(self.conv1(l1_flow_finer))
        l1_flow_det = self.conv1(l1_flow_finer)
        l1_flow = l1_flow_coarse + l1_flow_det 
        l1_flow_proj = torch.reshape(l1_flow, (batch_size, self.out_H_list[1], self.out_W_list[1], 3))
        
        ############################### Layer 0 Flow #############################################
        
        l0_points_f1_new = self.upconv0(l0_xyz_proj_f1, l1_xyz_proj_f1, l0_points_proj_f1, l1_flow_finer_proj)
        l0_flow_coarse = self.fp0(l0_xyz_proj_f1, l1_xyz_proj_f1, None, l1_flow_proj)
        l0_cost_volume = self.warping0(l0_flow_coarse, l0_xyz_proj_f1, l0_xyz_proj_f1_raw, l0_points_proj_f1, 
                                       l0_xyz_proj_f2, l0_points_proj_f2, intrinsics,
                                       constxs, constys, constzs, self.out_H_list[0], self.out_W_list[0],self.H_input,self.W_input)
        
        l0_flow_finer = self.flow_pred0(l0_points_f1, l0_points_f1_new, l0_cost_volume)
        l0_flow_finer_proj = torch.reshape(l0_flow_finer, (batch_size, self.out_H_list[0], self.out_W_list[0], -1))
        #l0_flow_det = self.drop0(self.conv0(l0_flow_finer))
        l0_flow_det = self.conv0(l0_flow_finer)
        l0_flow = l0_flow_coarse + l0_flow_det
        l0_flow_proj = torch.reshape(l0_flow, (batch_size, self.out_H_list[0], self.out_W_list[0], 3))
        

        ########################### Full flow #################################################################
        
        full_feat_f1 = self.fp0_1(input_xyz_f1_norm, l0_xyz_proj_f1, input_points_proj_f1, l0_flow_finer_proj)
        net = self.conv0_1(full_feat_f1)
        full_flow_coarse = self.fp0_2(input_xyz_f1_norm, l0_xyz_proj_f1, None, l0_flow_proj)
        full_flow_det = self.conv0_2(net)
        full_flow = full_flow_det + full_flow_coarse


        pc1_sample = [input_xyz_f1_raw, l0_xyz_f1_raw, l1_xyz_f1_raw, l2_xyz_f1_raw, l3_xyz_f1_raw]
        pc2_sample = [input_xyz_f2_raw, l0_xyz_f2_raw, l1_xyz_f2_raw, l2_xyz_f2_raw, l3_xyz_f2_raw]
        flow_list = [ full_flow, l0_flow, l1_flow, l2_flow, l3_flow]
        mask_list = [ occ_mask.reshape(batch_size, -1), l0_occ_mask.reshape(batch_size, -1), l1_occ_mask.reshape(batch_size, -1), l2_occ_mask.reshape(batch_size, -1), l3_occ_mask.reshape(batch_size, -1)]
        color12_list = [l0_color_f1,l0_color_f2]

        valid = [torch.logical_and(torch.all(torch.ne(pc1, 0), dim = -1), occ) for (pc1, occ) in zip(pc1_sample, mask_list)]

        #pc1_sample = [l0_xyz_f1_raw.permute(0,2,1), l1_xyz_f1_raw.permute(0,2,1), l2_xyz_f1_raw.permute(0,2,1), l3_xyz_f1_raw.permute(0,2,1)]
        #pc2_sample = [l0_xyz_f2_raw.permute(0,2,1), l1_xyz_f2_raw.permute(0,2,1), l2_xyz_f2_raw.permute(0,2,1), l3_xyz_f2_raw.permute(0,2,1)]
        #occ_mask_col = [l0_occ_mask.reshape(batch_size,1,-1), l1_occ_mask.reshape(batch_size,1,-1), l2_occ_mask.reshape(batch_size,1,-1), l3_occ_mask.reshape(batch_size,1,-1)]

        #flow_list = [l0_flow.permute(0,2,1), l1_flow.permute(0,2,1), l2_flow.permute(0,2,1), l3_flow.permute(0,2,1)]
        #re_label = [l0_label, l1_label, l2_label, l3_label]

        height_list = [self.H_input, self.out_H_list[0], self.out_H_list[1], self.out_H_list[2], self.out_H_list[3], self.out_H_list[4] ] 
        width_list  = [self.W_input, self.out_W_list[0], self.out_W_list[1], self.out_W_list[2], self.out_W_list[3], self.out_W_list[4] ]
        
        if train:
            return flow_list, valid , pc1_sample, pc2_sample , height_list, width_list ,color12_list
        else:
            return flow_list, valid, pc1_sample ,pc2_sample, height_list, width_list , label_list


def multiScaleLoss(pred_flows, gt_flows, valid_mask, alpha=[0.02, 0.04, 0.08, 0.16]):

    """ 
    To calculate multi-scale loss
 
    Args:
        @ pred_flows: list, (B, N, 3)
        @ gt_flows: list, (B, N, 3)
        @ valid_mask: list, (B, N)
    """
    num_scale = len(pred_flows)

    # generate GT list and masks
    total_loss = torch.zeros(1, device = "cuda", requires_grad = True)
    
    # epe3d = torch.sum((flow3d_est - flow3d)**2, -1).sqrt()
    for i in range(num_scale):
        diff_flow = torch.norm(pred_flows[i] - gt_flows[i], dim = 2) # (B, N)
        masked_diff_flow = torch.sum(diff_flow * valid_mask[i], dim = 1) # (B, )
        
        # valid_num = torch.sum(valid_mask[i].float(), dim=-1) #(B, )
        # loss_items = torch.where(valid_num > 1, masked_diff_flow / valid_num, torch.zeros_like(valid_num))
        # loss_item = loss_items.mean()
        
        loss_item = masked_diff_flow.mean()
        total_loss = total_loss + alpha[i] * loss_item

    return total_loss



def get_selected_idx(batch, out_H: int, out_W: int, stride_H: int, stride_W: int):
    """According to given stride and output size, return the corresponding selected points

    Args:
        array (tf.Tensor): [any array with shape (B, H, W, 3)]
        stride_h (int): [stride in height]
        stride_w (int): [stride in width]
        out_h (int): [height of output array]
        out_w (int): [width of output array]
    Returns:
        [tf.Tensor]: [shape (B, outh, outw, 3) indices]
    """
    
    # torch.arange(start, end, step=1, out=None):返回一个1维张量，长度为(end-start)/step，以step`为步长的一组序列值。
    
    select_h_idx = torch.arange(0, out_H * stride_H, stride_H).cuda()
    select_w_idx = torch.arange(0, out_W * stride_W, stride_W).cuda()
    height_indices = (torch.reshape(select_h_idx, (1, -1, 1))).expand(batch, out_H, out_W)  # b out_H out_W
    width_indices = (torch.reshape(select_w_idx, (1, 1, -1))).expand(batch, out_H, out_W)  # b out_H out_W
    padding_indices = torch.reshape(torch.arange(batch), (-1, 1, 1)).expand(batch, out_H, out_W)  # b out_H out_W

    return padding_indices, height_indices, width_indices
