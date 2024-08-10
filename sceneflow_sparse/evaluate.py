import sys
import cmd_args
import numpy as np
import argparse
import yaml 
from omegaconf import DictConfig
from datasets.kitti_data import SceneflowDataset
from datasets.kitti_lidar_eval import LIDAR_KITTI
from datasets.Argoverse import ArgoverseSceneFlowDataset
from datasets.NuScene import NuScenesSceneFlowDataset
from tensorboardX import SummaryWriter
import os
import csv
import custom_transforms
import torch
import models
import torch.backends.cudnn as cudnn
from itertools import chain
from utils.inverse_warp import pixel2cam, build_sky_mask, BackprojectDepth, build_block_mask, build_groud_mask, build_angle_sky
from utils.external_util import devide_by_index, set_by_index, AverageMeter, readlines, tensor2array, cam2cam, write_ply, pose_vec2mat, scale_con
from torch.autograd import Variable
import transforms


n_iter = 0

def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f * -1.0 + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y


def get_2d_flow(pc1, pc2, predicted_pc2, paths=None):
    if paths == None:
        px1, py1 = project_3d_to_2d(pc1)
        px2, py2 = project_3d_to_2d(predicted_pc2)
        px2_gt, py2_gt = project_3d_to_2d(pc2)
        flow_x = (px2 - px1).cpu().detach().numpy()
        flow_y = (py2 - py1).cpu().detach().numpy()

        flow_x_gt = (px2_gt - px1).cpu().detach().numpy()
        flow_y_gt = (py2_gt - py1).cpu().detach().numpy()

    else:
        focallengths = []
        cxs = []
        cys = []
        constx = []
        consty = []
        constz = []
        for path in paths:
            fname = os.path.split(path)[-1]
            calib_path = os.path.join(
                os.path.dirname(__file__),
                'utils',
                'calib_cam_to_cam',
                fname + '.txt')
            with open(calib_path) as fd:
                lines = fd.readlines()
                P_rect_left = \
                    np.array([float(item) for item in
                              [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                             dtype=np.float32).reshape(3, 4)
                focallengths.append(-P_rect_left[0, 0])
                cxs.append(P_rect_left[0, 2])
                cys.append(P_rect_left[1, 2])
                constx.append(P_rect_left[0, 3])
                consty.append(P_rect_left[1, 3])
                constz.append(P_rect_left[2, 3])
        focallengths = np.array(focallengths)[:, None, None]
        cxs = np.array(cxs)[:, None, None]
        cys = np.array(cys)[:, None, None]
        constx = np.array(constx)[:, None, None]
        consty = np.array(consty)[:, None, None]
        constz = np.array(constz)[:, None, None]

        px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                          constx=constx, consty=consty, constz=constz)

        flow_x = (px2 - px1)
        flow_y = (py2 - py1)

        flow_x_gt = (px2_gt - px1)
        flow_y_gt = (py2_gt - py1)

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)

    return flow_pred, flow_gt


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def main():
    global args, n_iter
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    print(args)
    #best_loss = float('inf')
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    log_dir = args.save_path + '/' + 'eval_' + args.name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('=> will save everything to {}'.format(log_dir))

    with open(log_dir + '/eval_kitti.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([ 'eva_epe', 'eva_acc1', 'eva_acc2', 'outlier'])
    
    with open(log_dir + '/eval_lidar_kitti.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([ 'eva_epe', 'eva_acc1', 'eva_acc2', 'outlier'])

    with open(log_dir + '/eval_argoverse.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([ 'eva_epe', 'eva_acc1', 'eva_acc2', 'outlier'])

    with open(log_dir + '/eval_nuscenes.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([ 'eva_epe', 'eva_acc1', 'eva_acc2', 'outlier'])
        

    torch.manual_seed(args.seed)
    os.system('cp %s %s' % ('train.py', log_dir))
    os.system('cp %s %s' % ('config.yaml', log_dir))
    depth_weight_path = os.path.join(args.load_weights_folder, "dispnet_model_best.pth.tar")
    pose_weight_path = os.path.join(args.load_weights_folder, "exp_pose_model_best.pth.tar")

    # load train data -------------------------
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    
    #^ KITTI --------------------------

    test_set_kitti = SceneflowDataset(args.test_data_path, npoints=args.index_num, train=False)
    
    test_loader_kitti = torch.utils.data.DataLoader(
        test_set_kitti, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=True
    )

    print("len set in stereoKITTI: ", len(test_set_kitti))
    #print('{} samples found in stereoKITTI: '.format(len(test_set_kitti)))

    #^ LIDAR KITTI --------------------------
    
    test_set_lidar_kitti = LIDAR_KITTI(root = args.test_lidar_kitti_path)

    test_loader_lidar_kitti = torch.utils.data.DataLoader(
                                test_set_lidar_kitti,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                pin_memory=False,
                                drop_last=True
                            )
    
    print("len set in lidarKITTI: ", len(test_set_lidar_kitti))
    #print('{} samples found in lidarKITTI: '.format(len(test_loader_lidar_kitti)))

    #^ ARGOVERSE --------------------------
    
    test_set_argoverse = ArgoverseSceneFlowDataset(root = args.test_argoverse_path)

    test_loader_argoverse = torch.utils.data.DataLoader(
                                test_set_argoverse,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                pin_memory=False,
                                drop_last=True
                            )
                    
    print("len set in Argoverse: ", len(test_set_argoverse))
    #print('{} samples found in Argoverse:'.format(len(test_loader_argoverse)))

    #^ NuScenes --------------------------
    
    test_set_nuscenes = NuScenesSceneFlowDataset(root = args.test_nuscenes_path)

    test_loader_nuscenes = torch.utils.data.DataLoader(
                                test_set_nuscenes,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                num_workers=args.workers,
                                pin_memory=False,
                                drop_last=True
                            )
                    
    print("len set in Nuscenes: ", len(test_set_nuscenes))
    #print('{} samples found in Nuscenes:'.format(len(test_loader_nuscenes)))
    

    # create model -------------------
    scene_net = getattr(models, args.flownet_name)(args.flag_big_radius, args.kernel_shape, args.layers, args.is_training).cuda()

    disp_net = getattr(models, args.depth_name)(args.resnet_layers, 0).cuda()
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)
    disp_net.eval()

    pose_net = getattr(models, args.pose_name)(18, False).cuda()
    weights = torch.load(args.pretrained_posenet)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()

    if args.resume:
        print("=> resuming from checkpoint %s " % args.resume_name)
        resume_path = args.resume_name
        print("load model from path: ", resume_path)
        state = torch.load(resume_path)
        scene_net.load_state_dict(state['state_dict'])   
        print("load resume model successfully! ----------------")

    #cudnn.benchmark = True

    if args.multi_gpu:
        scene_net = torch.nn.DataParallel(scene_net)

    parameters = chain(scene_net.parameters())

    if True:

        # eval kitti
        epe, acc1, acc2,outlier = eval_kitti(scene_net, test_loader_kitti)
        print(' ---- EVALUATION ON KITTI ----')
        print("epe3d: ", epe)
        print("acc3d strict: ", acc1)
        print("acc3d relax: ", acc2)
        print("outlier: ", outlier)

        with open(log_dir + '/eval_kitti.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epe, acc1, acc2, outlier])


        # eval lidar kitti
        epe, acc1, acc2,outlier = eval_lidar(scene_net, test_loader_lidar_kitti)
        print(' ---- EVALUATION ON LIDAR KITTI ----')
        print("epe3d: ", epe)
        print("acc3d strict: ", acc1)
        print("acc3d relax: ", acc2)
        print("outlier: ", outlier)

        with open(log_dir + '/eval_lidar_kitti.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epe, acc1, acc2, outlier])
        

        # eval Argoverse
        #epe, acc1, acc2,outlier = eval_lidar(scene_net, test_loader_argoverse)
        print(' ---- EVALUATION ON ARGOVERSE ----')
        print("epe3d: ", epe)
        print("acc3d strict: ", acc1)
        print("acc3d relax: ", acc2)
        print("outlier: ", outlier)

        with open(log_dir + '/eval_argoverse.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epe, acc1, acc2, outlier])

        # eval Nuscene
        #epe, acc1, acc2,outlier = eval_lidar(scene_net, test_loader_nuscenes)
        print(' ---- EVALUATION ON NUSCENES ----')
        print("epe3d: ", epe)
        print("acc3d strict: ", acc1)
        print("acc3d relax: ", acc2)
        print("outlier: ", outlier)

        with open(log_dir + '/eval_nuscenes.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([epe, acc1, acc2, outlier])
    

def eval_kitti(scene_net, test_loader):
    global args
    scene_net.eval()
    epe_ori = AverageMeter()
    acc1_ori = AverageMeter()
    acc2_ori = AverageMeter()
    outliers = AverageMeter()

    for i, (pos1, pos2, color1, color2, flow, mask) in enumerate(test_loader):
        pos1 = Variable(pos1.cuda()).permute(0, 2, 1).float()
        pos2 = Variable(pos2.cuda()).permute(0, 2, 1).float()
        color1 = Variable(color1.cuda()).permute(0, 2, 1).float()
        color2 = Variable(color2.cuda()).permute(0, 2, 1).float()
        b, c, n = pos1.size()
        flow = Variable(flow.cuda()).permute(0, 2, 1).float()

        trans = torch.tensor([[0, -1, 0],
                              [0, 0, -1],
                              [1, 0, 0]])
        trans_mat = trans.unsqueeze(0).repeat(b, 1, 1).type_as(pos1)
        pos1 = trans_mat.bmm(pos1)
        pos2 = trans_mat.bmm(pos2)
        flow = trans_mat.bmm(flow).permute(0, 2, 1)

        pred_sfs, _, _ = scene_net(pos1, pos2, color1, color2)  # [b 2048 3]
        pred_sf = pred_sfs[0].permute(0, 2, 1)
        pre = pred_sf.cpu().detach().numpy()
        tar = flow.cpu().detach().numpy()[:, :args.num_points, :]

        if i == 10:
            np.savetxt('./../cloud/finals/pc1_eval_kitti.txt',(pos1).permute(0, 2, 1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
            np.savetxt('./../cloud/finals/pc1_warped_eval_kitti.txt',(pos1.permute(0, 2, 1)+flow).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
            np.savetxt('./../cloud/finals/pc2_eval_kitti.txt',(pos2).permute(0, 2, 1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f') 

        l2_norm = np.linalg.norm(np.abs(tar - pre) + 1e-20, axis=-1)
        EPE3D_ori = l2_norm.mean()

        sf_norm = np.linalg.norm(tar, axis=-1)
        relative_err = l2_norm / (sf_norm + 1e-4)

        acc3d_strict_ori = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
        acc3d_relax_ori = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
        outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()
        epe_ori.update(EPE3D_ori, args.batch_size)
        acc1_ori.update(acc3d_strict_ori, args.batch_size)
        acc2_ori.update(acc3d_relax_ori, args.batch_size)
        outliers.update(outlier, args.batch_size)


    return epe_ori.avg[0], acc1_ori.avg[0], acc2_ori.avg[0],outliers.avg[0]


def eval_lidar(scene_net, test_loader , ):
    global args
    scene_net.eval()
    epe_ori = AverageMeter()
    acc1_ori = AverageMeter()
    acc2_ori = AverageMeter()
    outliers = AverageMeter()

    for i, (pos1, pos2, color1, color2, flow, mask) in enumerate(test_loader):

        pos1 = Variable(pos1.cuda()).permute(0, 2, 1).float()
        pos2 = Variable(pos2.cuda()).permute(0, 2, 1).float()
        color1 = Variable(color1.cuda()).permute(0, 2, 1).float()
        color2 = Variable(color2.cuda()).permute(0, 2, 1).float()
        b, c, n = pos1.size()
        flow = Variable(flow.cuda()).float()

        pred_sfs, _, _ = scene_net(pos1, pos2, color1, color2)  # [b 2048 3]
        pred_sf = pred_sfs[0].permute(0, 2, 1)
        pre = pred_sf.cpu().detach().numpy()
        tar = flow.cpu().detach().numpy()[:, :args.num_points, :]

        if i == 10:
            np.savetxt('./../cloud/finals/pc1_eval_lidar.txt',(pos1).permute(0, 2, 1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
            np.savetxt('./../cloud/finals/pc1_warped_eval_lidar.txt',(pos1.permute(0, 2, 1)+flow).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
            np.savetxt('./../cloud/finals/pc2_eval_lidar.txt',(pos2).permute(0, 2, 1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f') 


        l2_norm = np.linalg.norm(np.abs(tar - pre) + 1e-20, axis=-1)
        EPE3D_ori = l2_norm.mean()

        sf_norm = np.linalg.norm(tar, axis=-1)
        relative_err = l2_norm / (sf_norm + 1e-4)

        acc3d_strict_ori = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
        acc3d_relax_ori = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
        outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()
        epe_ori.update(EPE3D_ori, args.batch_size)
        acc1_ori.update(acc3d_strict_ori, args.batch_size)
        acc2_ori.update(acc3d_relax_ori, args.batch_size)
        outliers.update(outlier, args.batch_size)


    return epe_ori.avg[0], acc1_ori.avg[0], acc2_ori.avg[0],outliers.avg[0]


if __name__ == '__main__':
    print("run train.py, start here: ---------")
    main()