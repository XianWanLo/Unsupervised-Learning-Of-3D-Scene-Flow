from datetime import date
import sys
import cmd_args
import numpy as np
import importlib
from time import time 
from dataset.kitti_sf_dataset import KITTI
from tensorboardX import SummaryWriter
import os
import csv
import custom_transforms
import torch
import model
import torch.backends.cudnn as cudnn
from utils.inverse_warp import pixel2cam, build_sky_mask, BackprojectDepth, build_block_mask, build_groud_mask, build_angle_sky, build_outliers_mask
from utils.external_utils import AverageMeter,pose_vec2mat
from utils.evaluation_utils import evaluate_2d,evaluate_3d
from utils.geometry import get_batch_2d_flow
from model.model_pholy import ESBFlow
from graph import BHW3_to_plot



def compute_depth(disp_net, tgt_img, ref_img):

    tgt_depth = [1 / disp for disp in disp_net(tgt_img)]
    ref_depth = [1 / disp for disp in disp_net(ref_img)]

    return tgt_depth, ref_depth



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

n_iter = 0

def main():
    global args, n_iter
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    print(args)
    best_loss = float('inf')
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    today = date.today().strftime('%m_%d_%Y')

    log_dir = args.test_save_path + '/' +'eval_' + today
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('=> will save everything to {}'.format(log_dir))

    with open(log_dir + '/eval.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([' eva_epe3d', ' eva_acc_3d_1',' eva_acc_3d_1', ' outliers_3d', ' eva_epe2d', ' eva_acc2d'])

    torch.manual_seed(args.seed)
    os.system('cp %s %s' % ('evaluate.py', log_dir))
    os.system('cp %s %s' % ('config.yaml', log_dir))

    test_set = KITTI(data_root = args.test_data_path, train = True)
    
    print("len set: ", len(test_set))
    print('{} samples found in test data'.format(len(test_set)))
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=True
    )

    # create model ------------------- (scene net + disp net + pose net)
    scene_net = ESBFlow(256,832,args.is_training).cuda()
    #scene_net = ESBFlow(376, 1242, is_training=False).cuda()

    disp_net = getattr(model, args.depth_name)(args.resnet_layers, 0).cuda()
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)
    disp_net.eval()

    pose_net = getattr(model, args.pose_name)(18, False).cuda()
    weights = torch.load(args.pretrained_posenet)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()
    
    if args.resume_eval:

        model_path = args.load_eval_model_path + '/scene_model.newest.t7' 
        state = torch.load(model_path)
        scene_net.load_state_dict(state['state_dict'])   
        print("=> resuming from checkpoint %s " % model_path)   
        print("load resume model successfully! ----------------")

    cudnn.benchmark = True
    
    if args.multi_gpu:
        scene_net = torch.nn.DataParallel(scene_net)
  
    epe_pre, acc1_pre, acc2_pre,outlier_pre, epe2d, acc2d = eval_KITTI_train(scene_net, test_loader,disp_net,pose_net)
                
    with open(log_dir +'/eval.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([epe_pre, acc1_pre, acc2_pre, outlier_pre, epe2d, acc2d])


LAYER_IDX = 0
ODO_HEIGHT = 256 
ODO_WIDTH = 832
KITTI_HEIGHT = 376 
KITTI_WIDTH = 1242

def eval_KITTI_train(scene_net, test_loader,disp_net,pose_net ):

    # 3D metrics
    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    
    # 2D metrics
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    model = scene_net.eval()
    disp_net.eval()
    pose_net.eval()

    for i, data in enumerate(test_loader, 0):
        
        # pc1, pc2, flow, intrinsics, sampled_index = data
        image1_b3hw, image2_b3hw, pc1_b3hw,pc2_b3hw, depth1,depth2, flow_gt_b3hw, intrinsic, inv_intrinsic, valid = data
        
        # move to cuda
        image1_b3hw = image1_b3hw.cuda(non_blocking = True).float()
        image2_b3hw = image2_b3hw.cuda(non_blocking = True).float()
        pc1_b3hw = pc1_b3hw.cuda(non_blocking = True).float()
        pc2_b3hw = pc2_b3hw.cuda(non_blocking = True).float()
        depth1 = depth1.cuda(non_blocking = True).float()
        depth2 = depth2.cuda(non_blocking = True).float()
        flow_gt_b3hw = flow_gt_b3hw.cuda(non_blocking = True).float()
        intrinsic = intrinsic.cuda(non_blocking = True).float()
        inv_intrinsic = inv_intrinsic.cuda(non_blocking = True).float()
        valid = valid.cuda(non_blocking = True)   

        #print("image1_b3hw:",image1_b3hw.shape)
        #print("flow_gt_b3hw:",flow_gt_b3hw.shape)
        #print("intrinsic:",intrinsic.shape)
        
        #BHW3_to_plot(pc1,pc2,flow,'try')

        #^ newly added 
    
        #depth, ref_depths = compute_depth(disp_net, image1_b3hw, image2_b3hw)
        
        #depth1 = depth[0]
        #depth2 = ref_depths[0]

        pose_predict = pose_net(image1_b3hw, image2_b3hw)
        pose_mat = pose_vec2mat(pose_predict)
        pose_inv_predict = pose_net(image2_b3hw, image1_b3hw)

        #pc1_b3hw = BackprojectDepth(depth1.unsqueeze(1), inv_intrinsic)
        #pc2_b3hw = BackprojectDepth(depth2.unsqueeze(1), inv_intrinsic)

        sky_mask1 = build_sky_mask(depth1)  # [b h w]
        sky_mask2 = build_sky_mask(depth2)
        ground_mask1 = build_groud_mask(pc1_b3hw, 1.4)
        ground_mask2 = build_groud_mask(pc2_b3hw, 1.4)
        #angle_sky_mask = build_angle_sky(pc1, velo2cam_ten.type_as(pc1), cam2velo_ten.type_as(pc1))
        #block_mask1 = build_block_mask(image1_b3hw, image2_b3hw, depth1.unsqueeze(1), depth2.unsqueeze(1), pose_predict, intrinsic, inv_intrinsic).type_as(sky_mask1)
        #block_mask2 = build_block_mask(image2_b3hw, image1_b3hw, depth2.unsqueeze(1), depth1.unsqueeze(1), pose_inv_predict, intrinsic, inv_intrinsic).type_as(sky_mask1)
        outliers_mask = build_outliers_mask(pc1_b3hw,pc2_b3hw)

        #! change mask
        #total_mask = sky_mask1 * sky_mask2 * ground_mask1 * ground_mask2 *outliers_mask *block_mask1 * block_mask2 
        total_mask = sky_mask1 * sky_mask2 * ground_mask1 * ground_mask2 *outliers_mask * valid

        #^ 坐标系变换: pc1 点云坐标系 【 X(前)，Y(左)，Z(上)】 到 pc1_new 相机坐标系 【X(右)，Y(下)，Z(前)】     

        #if i ==0:
        #    np.savetxt('./../cloud/pc1_transform.txt',(pc1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #    np.savetxt('./../cloud/pc2_transform.txt',(pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        
        pc1 = pc1_b3hw.permute(0,2,3,1).contiguous()                                # [B H W 3]
        pc2 = pc2_b3hw.permute(0,2,3,1).contiguous()                                # [B H W 3]
        image1 = image1_b3hw.permute(0,2,3,1).contiguous()
        image2 = image2_b3hw.permute(0,2,3,1).contiguous()
        flow_gt = flow_gt_b3hw.permute(0,2,3,1).contiguous()

        print("pc1:",pc1.shape)
        #print("image1:",image1.shape)
        
        flow3d_list, mask_list, pc1_sample ,pc2_sample ,height_list, width_list, flow3d_gt_list = model(pc1, pc2, image1, image2, intrinsic, total_mask.unsqueeze(-1), flow_gt, False, False )
        
        #print("pc1_sample:",pc1_sample[0].shape)

        #np.savetxt('./../cloud/pc1_03_10.txt',(pc1_sample[0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #np.savetxt('./../cloud/pc2_02_27.txt',(pc2_sample[0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        if i ==0:
            np.savetxt('./../cloud/pc1_eval.txt',(pc1_sample[0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
            np.savetxt('./../cloud/pc1_warped_eval.txt',(pc1_sample[0] + flow3d_list[0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
            np.savetxt('./../cloud/pc2_eval.txt',(pc2_sample[0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

        flow3d_pred_np =  (flow3d_list[0])               # [b n 3]
        flow3d_label_np = flow3d_gt_list[0]              # [b n 3] 
        pc1_3d_np = (pc1_sample[0])                      # [b n 3] 
        mask = (mask_list[0])                            # [b n]  

        intrinsic_np = intrinsic.detach().cpu().numpy()
        pred_sf = flow3d_pred_np[mask].detach().cpu().numpy()
        sf_np = flow3d_label_np[mask].detach().cpu().numpy()
        pc1_np = pc1_3d_np[mask].detach().cpu().numpy()

        if i ==0:
            np.savetxt('./../cloud/pc1_masked_eval.txt',(pc1_np).reshape(-1,3),fmt='%f %f %f')
            np.savetxt('./../cloud/pc1_warped_masked_eval.txt',(pc1_np + pred_sf).reshape(-1,3),fmt='%f %f %f')
            np.savetxt('./../cloud/pc2_masked_eval.txt',(pc1_np+ sf_np).reshape(-1,3),fmt='%f %f %f')
        
        if pc1_np.shape[0] == 0:
            print("continue")
            continue
        # if i >= 20:
        #     break
        # print("Valid num of points: {} / {}".format(pc1_np.shape[0], valid_mask[0].shape[1]))
        # print("Max flow %f" % np.max(np.sum(sf_np ** 2, axis = -1)))
        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        print(epe3ds.avg[0])

        #^ 由于第一层output的pc不是原始大小，而是128 x 208 ，所以intrinsic 要进行resize

        # 2D evaluation metrics
        flow_pred_2d, flow_gt_2d = get_batch_2d_flow(pc1_np, pc1_np + sf_np, pc1_np + pred_sf, intrinsic_np , height_list[0], width_list[0],ODO_HEIGHT, ODO_WIDTH)
        EPE2D, acc2d = evaluate_2d(flow_pred_2d, flow_gt_2d)

        print('pre eval mean EPE 3D:' , (EPE3D))
        print('pre eval mean acc3d_1:', (acc3d_strict))
        print('pre eval mean acc3d_2 :', (acc3d_relax))
        print('pre eval mean outlier3d :', (outlier))
        print('pre eval mean EPE 2D :', (EPE2D))
        print('pre eval mean acc2d :', (acc2d))

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

        print("end")
        
    print('TOTAL EPE 3D:' , (epe3ds.avg[0]))
    print('TOTAL acc3d_1:', (acc3d_stricts.avg[0]))
    print('TOTAL acc3d_2 :', (acc3d_relaxs.avg[0]))
    print('TOTAL outlier3d :', (outliers.avg[0]))
    print('TOTAL EPE 2D :', ( epe2ds.avg[0]))
    print('TOTAL acc2d :', (acc2ds.avg[0]))
    print("Evaluating Finished")

    return epe3ds.avg[0], acc3d_stricts.avg[0], acc3d_relaxs.avg[0], outliers.avg[0] , epe2ds.avg[0], acc2ds.avg[0]




if __name__ == '__main__':
    print("run evaluate.py, start here: ---------")
    main()