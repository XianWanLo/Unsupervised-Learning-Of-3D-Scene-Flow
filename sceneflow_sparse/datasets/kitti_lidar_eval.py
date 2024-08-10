import sys, os
import os.path as osp
import numpy as np
import torch


__all__ = ['KITTIlidar']


class LIDAR_KITTI():
    def __init__(self, root):
        
        self.files = []
        self.root = root
        self.input_features = 'absolute_coords'
        self.num_points = 8192
        self.voxel_size = 0.10
        self.remove_ground = True 
        self.dataset = 'LidarKITTI_ME'
        self.only_near_points = True
        self.phase = 'test'
        self.DATA_FILES = {
        'train': './configs/datasets/lidar_kitti/test.txt',
        'val': './configs/datasets/lidar_kitti/test.txt',
        'test': './configs/datasets/lidar_kitti/test.txt'
    }
        self.randng = np.random.RandomState()
        self.device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu') 

        self.augment_data = False

        #print("Loading the subset {} from {}".format(self.phase,self.root))

        subset_names = open(self.DATA_FILES[self.phase]).read().split()

        for name in subset_names:
            self.files.append(name)

    def __getitem__(self, idx):
        file = os.path.join(self.root,self.files[idx])
        #print(file)
        file_name = file.replace(os.sep,'/').split('/')[-1]
        
        # Load the data
        data = np.load(file)
        pc_1 = data['pc1']
        pc_2 = data['pc2']

        if 'pose_s' in data:
            pose_1 = data['pose_s']
        else:
            pose_1 = np.eye(4)

        if 'pose_t' in data:
            pose_2 = data['pose_t']
        else:
            pose_2 = np.eye(4)

        if 'sem_label_s' in data:
            labels_1 = data['sem_label_s']
        else:
            labels_1 = np.zeros(pc_1.shape[0])


        if 'sem_label_t' in data:
            labels_2 = data['sem_label_t']
        else:
            labels_2 = np.zeros(pc_2.shape[0])

        if 'flow' in data:
            flow = data['flow']
        else:
            flow = np.zeros_like(pc_1)

        # Remove the ground and far away points
        # In stereoKITTI the direct correspondences are provided therefore we remove,
        # if either of the points fullfills the condition (as in hplflownet, flot, ...)

        if self.dataset in ["SemanticKITTI_ME", 'LidarKITTI_ME', "WaymoOpen_ME"]:
            if self.remove_ground:
                if self.phase == 'test':
                    is_not_ground_s = (pc_1[:, 1] > -1.4)
                    is_not_ground_t = (pc_2[:, 1] > -1.4)

                    pc_1 = pc_1[is_not_ground_s,:]
                    labels_1 = labels_1[is_not_ground_s]
                    flow = flow[is_not_ground_s,:]

                    pc_2 = pc_2[is_not_ground_t,:]
                    labels_2 = labels_2[is_not_ground_t]

            if self.only_near_points:
                is_near_s = (pc_1[:, 2] < 35)
                is_near_t = (pc_2[:, 2] < 35)

                pc_1 = pc_1[is_near_s,:]
                labels_1 = labels_1[is_near_s]
                flow = flow[is_near_s,:]

                pc_2 = pc_2[is_near_t,:]
                labels_2 = labels_2[is_near_t]


        # Sample n points for evaluation before the voxelization
        # If less than desired points are available just consider the maximum
        if pc_1.shape[0] > self.num_points:
            idx_1 = np.random.choice(pc_1.shape[0], self.num_points, replace=False)
        else:
            idx_1 = np.concatenate((np.arange(pc_1.shape[0]), np.random.choice(pc_1.shape[0], self.num_points - pc_1.shape[0], replace=True)), axis=-1)
            #idx_1 = np.random.choice(pc_1.shape[0], pc_1.shape[0], replace=False)

        if pc_2.shape[0] > self.num_points:
            idx_2 = np.random.choice(pc_2.shape[0], self.num_points, replace=False)
        else:
            idx_2 = np.concatenate((np.arange(pc_2.shape[0]), np.random.choice(pc_2.shape[0], self.num_points - pc_2.shape[0], replace=True)), axis=-1)
            #idx_2 = np.random.choice(pc_2.shape[0], pc_2.shape[0], replace=False)

        pc_1_eval = pc_1[idx_1,:]
        flow_eval = flow[idx_1,:]
        #labels_1_eval = labels_1[idx_1]

        pc_2_eval = pc_2[idx_2,:]
        #labels_2_eval = labels_2[idx_2]

        color1 = np.zeros([self.num_points, 3])
        color2 = np.zeros([self.num_points, 3])
        mask = np.ones([self.num_points])

        return pc_1_eval, pc_2_eval, color1, color2, flow_eval, mask

    
    def __len__(self):
        return len(self.files)