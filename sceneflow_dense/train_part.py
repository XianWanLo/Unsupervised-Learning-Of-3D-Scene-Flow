from datetime import date
import sys
import cmd_args
import numpy as np
from dataset.sequence_folders import SequenceFolder
from dataset.kitti_sf_dataset import KITTI
from torch.utils.tensorboard import SummaryWriter
import os
import csv
import custom_transforms
import torch
import model
import torch.backends.cudnn as cudnn
from itertools import chain
from utils.inverse_warp import pixel2cam, build_sky_mask, BackprojectDepth, build_block_mask, build_groud_mask, build_angle_sky
from utils.external_utils import devide_by_index, set_by_index, AverageMeter, readlines, tensor2array, cam2cam, write_ply, pose_vec2mat, scale_con
from utils.evaluation_utils import evaluate_2d,evaluate_3d
from utils.geometry import get_batch_2d_flow, resize_intrinsic
from torch.autograd import Variable
from model.model_pholy import ESBFlow
from model.point_conv_pytorch import warppingProject
from multi_loss import multiScaleChamferSmoothCurvature
from loss_function.loss_2048 import ChamferCurvature_2048
from loss_function.depth_consistency import DepthConsistencyLoss
import torch.nn.functional as F


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


def testOnce(scene_net, test_loader):
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


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1 / disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1 / disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def robust_l1_per_pix(x, q=0.5, eps=1e-2, compute_type=False, q2=0.5, eps2=1e-2):
    if compute_type:
        x = torch.pow((x.pow(2) + eps), q)
    else:
        x = torch.pow((x.abs() + eps2), q2)
    return x

def augmentation():
    anglex = np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(np.float32) * np.pi / 4.0
    angley = np.clip(0.05 * np.random.randn(), -0.1, 0.1).astype(np.float32) * np.pi / 4.0
    anglez = np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(np.float32) * np.pi / 4.0
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                   [0, cosx, -sinx],
                   [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                   [0, 1, 0],
                   [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                   [sinz, cosz, 0],
                   [0, 0, 1]])

    R_trans = Rx.dot(Ry).dot(Rz)
    xx = np.clip(0.1 * np.random.randn(), -0.2, 0.2).astype(np.float32)
    yy = np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(np.float32)
    zz = np.clip(0.5 * np.random.randn(), -1, 1).astype(np.float32)
    shift = np.array([[xx], [yy], [zz]])

    return R_trans, shift


def main():

    global args, n_iter
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    print(args)
    best_loss = float('inf')
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    today = date.today().strftime('%m_%d_%Y')

    log_dir = args.save_path + '/' + 'test_image_' + today
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print('=> will save everything to {}'.format(log_dir))

    training_writer = SummaryWriter(log_dir)

    with open(log_dir + '/_eval.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow([ 'eva_epe_3d   ', 'eva_acc3d_1   ', 'eva_acc3d_2   ', 'outlier3d   ','eva_epe_2d   ', 'eva_acc2d_1   ' ])

    torch.manual_seed(args.seed)
    os.system('cp %s %s' % ('train_all.py', log_dir))
    os.system('cp %s %s' % ('config.yaml', log_dir))
    os.system('cp %s %s' % ('multi_loss.py', log_dir))
    depth_weight_path = os.path.join(args.load_weights_folder, "dispnet_model_best.pth.tar")
    pose_weight_path = os.path.join(args.load_weights_folder, "exp_pose_model_best.pth.tar")

    # load train data -------------------------
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])
    
    # transforms.Normalize RGB彩色图的数据归一化
    # 先做ArrayToTensor再做Normalize:把RGB数据范围缩到【0，1】，再用Normalize归一到【-2，2.44】
    train_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])
    train_transform_1 = custom_transforms.Compose([
        custom_transforms.ArrayToTensor()
    ])
    
    train_set = SequenceFolder(
        args.data,
        transform=train_transform,
        transform_1=train_transform_1,
        seed=args.seed,
        train=True,
        sequence_length=args.sequence_length
    )
    #train_set实际上就是一堆图片

    print("train data len set: ", len(train_set))
    print('{} samples found in train data'.format(len(train_set)))

    n_quarter = 5000
    n_left_train = len(train_set) - n_quarter
    
    train_part_dataset,_ = torch.utils.data.random_split(train_set, [n_quarter,n_left_train])
    print("train data len set used :",len(train_part_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # load test data  -----------------
    
    #test_set_256 = SequenceFolder(
    #    args.data,
    #    transform=train_transform,
    #    transform_1=train_transform_1,
    #    seed=args.seed,
    #    train=False,
    #    sequence_length=args.sequence_length
    #)

    test_set_sceneflow = KITTI(data_root = args.test_data_path)

    test_set = test_set_sceneflow

    print("len set: ", len(test_set))
    print('{} samples found in test data'.format(len(test_set)))

    #n_quarter = 800
    #n_left_test = len(test_set) - n_quarter
    #test_part_dataset,_ = torch.utils.data.random_split(test_set, [n_quarter,n_left_test])
    #print("test data len set used :",len(test_part_dataset))
   
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, drop_last=True
    )

    # create model ------------------- (scene net + disp net + pose net)
    #scene_net = getattr(model, args.flownet_name)(args.flag_big_radius, args.kernel_shape, args.layers, args.is_training).cuda()
    scene_net = ESBFlow(256,832,args.is_training).cuda()
    disp_net = getattr(model, args.depth_name)(args.resnet_layers, 0).cuda()
    print("=> using pre-trained weights for DispResNet")
    weights = torch.load(args.pretrained_disp)
    disp_net.load_state_dict(weights['state_dict'], strict=False)
    disp_net.eval()

    pose_net = getattr(model, args.pose_name)(18, False).cuda()
    weights = torch.load(args.pretrained_posenet)
    pose_net.load_state_dict(weights['state_dict'], strict=False)
    pose_net.eval()
    
    #flownet_name:Each_layer_Output
    #depth_name:DispResNet
    #pose_name:PoseResNet

    if args.resume:

        model_path = args.load_model_path + '/scene_model.newest.t7' 
        state = torch.load(model_path)
        scene_net.load_state_dict(state['state_dict'])   
        print("=> resuming from checkpoint %s " % model_path)   
        print("load resume model successfully! ----------------")

    cudnn.benchmark = True

    if args.multi_gpu:
        scene_net = torch.nn.DataParallel(scene_net)

    parameters = chain(scene_net.parameters())
    optimizer = torch.optim.Adam(parameters, args.lr,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay
                                 )

    if args.evaluate:  #只需要evaluate
        epe, acc1, acc2,outlier, epe2d , acc2d = eval_kitti_sceneflow(scene_net, test_loader)  #先跑一次without mask的测试
        print("epe without mask: ", epe)
        print("acc strict without mask: ", acc1)
        print("acc relax without mask: ", acc2)

        with open(log_dir + '/log_ori.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([0, epe, acc1, acc2, outlier])

        with open(log_dir + '/log_pre2.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([0, epe, acc1, acc2, outlier])

        with open(log_dir + '/log_l0.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([0, epe, acc1, acc2, outlier])
   
    else: #evaluate + train
        print("do the training + evaluate part:")
        
        for epoch in range(args.epochs):
            
            train_part_dataset,_ = torch.utils.data.random_split(train_set, [n_quarter,n_left_train])
            print("train data len set used :",len(train_part_dataset))

            train_loader = torch.utils.data.DataLoader(
                train_part_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
            
            # TRAIN
            print("current epoch: ", epoch)
            train_loss = train(train_loader, disp_net, pose_net, scene_net, optimizer, training_writer)
            training_writer.add_scalar("epoch train loss", train_loss, epoch)
            
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({'epoch': epoch + 1, 'state_dict': scene_net.state_dict()},log_dir + '/model.best.t7')
            torch.save({'epoch': epoch + 1, 'state_dict': scene_net.state_dict()}, log_dir + '/scene_model.newest.t7')

            with open(log_dir + '/train_loss.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([train_loss])

            #EVALUATE
            epe3d_pre2, acc1_pre2, acc2_pre2,outlier_pre2, epe2d , acc2d = eval_kitti_sceneflow(scene_net, test_loader)

            with open(log_dir + '/_eval.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter='\t')
                    writer.writerow([epe3d_pre2, acc1_pre2, acc2_pre2, outlier_pre2, epe2d , acc2d])


HEIGHT = 256 
WIDTH = 832       

def train(train_loader, disp_net, pose_net, scene_net, optimizer, training_writer=None):
    global n_iter, args
    disp_net.eval()
    pose_net.eval()
    scene_net.train()

    losses = AverageMeter(precision=4)

    for i, (tgt_img, ref_imgs, intrinsics, inv_intrinsics, velo2cam,cam2velo, ori_tgt_img, ori_ref_imgs, depth_gt) in enumerate(train_loader):

        b  = tgt_img.shape[0]
        tgt_img = tgt_img.to(device)  # [b 3 h w]
        ref_imgs = [img.to(device) for img in ref_imgs]  #  [b 3 h w]
        intrinsic = intrinsics.to(device)  # [b 3 3]
        inv_intrinsic = inv_intrinsics.to(device)  # [b 3 3]

        velo2cam_ten = velo2cam.to(device)
        cam2velo_ten = cam2velo.to(device)
        
        ori_tgt_img = ori_tgt_img.to(device)  # [b 3 h w]
        ori_ref_imgs = [img.to(device) for img in ori_ref_imgs]  # list 2 [b 3 h w]
        b, _, h, w = tgt_img.size()

        depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        depth = depth[0]

        fw_depth = ref_depths[1][0]
        bw_depth = ref_depths[0][0]

        pose_predict = pose_net(tgt_img, ref_imgs[1])
        pose_mat = pose_vec2mat(pose_predict)
        pose_inv_predict = pose_net(ref_imgs[1], tgt_img)

        # remove_sky
        new_pred_depth, new_fw_depth, pose_mat = scale_con(depth_gt, depth.squeeze(1), fw_depth.squeeze(1), pose_mat)
        sky_mask1 = build_sky_mask(new_pred_depth.squeeze(1))  # [b h w]
        sky_mask2 = build_sky_mask(new_fw_depth.squeeze(1))
        pc1_cam1 = BackprojectDepth(new_pred_depth, inv_intrinsic)
        pc2_cam2 = BackprojectDepth(new_fw_depth, inv_intrinsic)
        ground_mask1 = build_groud_mask(pc1_cam1)
        ground_mask2 = build_groud_mask(pc2_cam2)
        angle_sky_mask = build_angle_sky(pc1_cam1, velo2cam_ten.type_as(pc1_cam1), cam2velo_ten.type_as(pc1_cam1))
        block_mask1 = build_block_mask(tgt_img, ref_imgs[1], depth, fw_depth, pose_predict, intrinsic, inv_intrinsic).type_as(sky_mask1)
        block_mask2 = build_block_mask(ref_imgs[1], tgt_img, fw_depth, depth, pose_inv_predict, intrinsic, inv_intrinsic).type_as(sky_mask1)

        training_writer.add_image('sky_mask', tensor2array(sky_mask1.data[0].cpu(), max_value=1, colormap='bone'), n_iter)
        training_writer.add_image('block_mask_1', tensor2array(block_mask1.data[0].cpu(), max_value=1, colormap='bone'), n_iter)
        training_writer.add_image('block_mask_2', tensor2array(block_mask2.data[0].cpu(), max_value=1, colormap='bone'), n_iter)

        #total_mask = sky_mask1* ground_mask1 * block_mask1 *  angle_sky_mask.type_as(ground_mask1)
        total_mask = sky_mask1 * sky_mask2 * ground_mask1 * ground_mask2 * block_mask1 * block_mask2  * angle_sky_mask.type_as(ground_mask2)
        training_writer.add_image('total_mask', tensor2array(total_mask.data[0].cpu(), max_value=1, colormap='bone'), n_iter)

        valid_batch = total_mask.reshape(b,-1).sum(1) < 3000
        
        print("total_mask numbers:",total_mask.reshape(b,-1).sum(1))

        if valid_batch.sum().item() > 0:
            continue

        total_mask_bhw1 = total_mask.unsqueeze(-1).contiguous()                 # [B H W 1]

        pc1_cam1_bhw3 = (pc1_cam1.permute(0,2,3,1)).contiguous()        # [B H W 3]
        rgb1_bhw3 = (tgt_img.permute(0,2,3,1)).contiguous()             # [B H W 3]
        #ori_rgb1_bhw3 = ori_tgt_img.permute(0,2,3,1).float().contiguous()

        pc2_cam2_bhw3 = (pc2_cam2.permute(0,2,3,1)).contiguous()        # [B H W 3]
        rgb2_bhw3 = (ref_imgs[1].permute(0,2,3,1)).contiguous()         # [B H W 3]
        #ori_rgb2_bhw3 = ori_ref_imgs[1].permute(0,2,3,1).float().contiguous()

        #pc1_warp_idx_fetching = warppingProject(pc1_cam1_bhw3.reshape(b,-1,3), fxs, fys, cxs, cys, 0, 0, 0, height, width)
        
        #! pos1_bhw3, pos2_bhw3, fea1, fea2, intrinsics, occ_mask , label
        over_all_flows, mask_list , pc1, pc2, h_list, w_list, color12_list = scene_net(pc1_cam1_bhw3, pc2_cam2_bhw3, rgb1_bhw3, rgb2_bhw3, intrinsic, total_mask_bhw1 , None, True, False)  # [b h w 3]

        #if i ==0:
        #    np.savetxt('./../cloud/pc1_trained.txt',(pc1[0][mask_list[0]>0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #    np.savetxt('./../cloud/pc1_warped_trained.txt',((pc1[0]+over_all_flows[0])[mask_list[0]>0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #    np.savetxt('./../cloud/pc2_trained.txt',(pc2[0][mask_list[0]>0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #    break

        # do the random static part first
        R_trans, shift = augmentation()
        R_trans = torch.from_numpy(R_trans).type_as(pose_mat)
        shift = torch.from_numpy(shift).type_as(pose_mat)

        batch_R = R_trans.unsqueeze(0).repeat(b, 1, 1)
        batch_T = shift.unsqueeze(0).repeat(b, 1, 1)

        pose_R = pose_mat[:, :, :3]  # [b 3 3]
        pose_T = pose_mat[:, :, 3:]  # [b 3 1]
        new_R = pose_R.bmm(batch_R)
        new_T = pose_R.bmm(batch_T) + pose_T

        random_RT = torch.cat([new_R, new_T], axis=-1)
        pc1_cam2_random = cam2cam(pc1_cam1.reshape(b,3,-1), random_RT)
        static_random_gt = pc1_cam2_random - pc1_cam1.reshape(b,3,-1)
        pc1_cam2_random_bhw3 = pc1_cam2_random.permute(0,2,1).reshape(b,h,w,3).float().contiguous()   # [B H W 3]

        dynamic_sfs, _, pc1_cam2_list,_ ,_ ,_, _ = scene_net(pc1_cam2_random_bhw3, pc2_cam2_bhw3, rgb1_bhw3, rgb2_bhw3, intrinsic, total_mask_bhw1 ,None, True , True)
        
        weight_list = [0.16, 0.08, 0.04, 0.02]    

        for level_id, overall_sf in enumerate(over_all_flows):
            
            h = h_list[level_id]
            w = w_list[level_id]
            valid = (mask_list[level_id] > 0)            # [B N]
            
            #^ 由于图片大小改变，所以内参矩阵 intrinsic 和 inv_intrinsic 都需要改变 --- 3/14
            new_intr,new_inv_intr = resize_intrinsic(intrinsic, h, w, HEIGHT, WIDTH)

            if level_id == 0:

                #~ Depth Consistency Loss1 : loss computed from overall sceneflow
                #! warped_pc1 坐标上
                
                warped_pc1 = pc1[level_id] + overall_sf         # [B N 3]
                warped_pc1_b3n = warped_pc1.permute(0,2,1)      # [B 3 N]
                
                #^ 相机三维点云 [X,Y,Z] 投影到 相机二维像素阵列 [u,v] = [Fx*X/Z+Cx, Fy*Y/Z+Cy] 
                pcoords = new_intr.bmm(warped_pc1_b3n)      #! pc1坐标上的pc1_warped
                X_ = pcoords[:, 0]
                Y_ = pcoords[:, 1]
                Z = pcoords[:, 2].clamp(min=1e-3)

                u = torch.clamp((X_ / Z), 0, w - 1)
                v = torch.clamp((Y_ / Z), 0, h - 1)

                #^ 二维像素阵列 分别除 w 和 h ,得到归一化像素平面 [0-1, 0-1]
                U_norm = (2*u / (w - 1) - 1).cpu()      # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
                V_norm = (2*v / (h - 1) - 1).cpu()

                grid_tf = torch.stack([U_norm, V_norm], dim=2).reshape(b,h,w,2).cuda()     
                #Z = warped_pc1_b3n[:, 2:3, :].clamp(min=1e-3)

                mask_x_low = torch.BoolTensor(U_norm>-1).byte()
                mask_x_high = torch.BoolTensor(U_norm<1).byte()
                mask_y_low = torch.BoolTensor(V_norm>-1).byte()
                mask_y_high = torch.BoolTensor(V_norm<1).byte()
                mask = (mask_x_low * mask_x_high * mask_y_low * mask_y_high).cuda()
                mask = mask.view(b,1,h,w)

                #^ 因为形状有变，pc2_depth 和 inverse intrinsic也需要重新处理，而不是用原始的new_fw_depth 和 inv_intrinsics  (pc2 [BN3] -> pc2_depth [B1HW])
                pc2_depth = (pc2[level_id])[:,:,2].reshape(b,h,w).unsqueeze(1)      # [B N 1] --> [B 1 H W]
                #print("pc2_depth:",pc2_depth.shape)

                #^ depth consistency   
                pro_depth = torch.nn.functional.grid_sample(pc2_depth, grid_tf, padding_mode='border',align_corners=True)
                
                #^ 深度[z] 加上 二维像素阵列 [u,v]  ， 乘以逆内参后得到 三维点云 [x,y,z]
                pro_xyz = BackprojectDepth(pro_depth, new_inv_intr)        # 三维点云 = 逆参矩阵 * 像素深度     [B 3 H W]
                pro_xyz_flatten = (pro_xyz*mask).view(b, 3, -1)

                if i ==10:
                    np.savetxt('./../cloud/pro_xyz_1.txt',(pro_xyz_flatten.permute(0,2,1)).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
                    np.savetxt('./../cloud/warped_pc1.txt',(warped_pc1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

                depth_con1_n3 = (abs(pro_xyz_flatten - warped_pc1_b3n).reshape(b,-1,3))[valid]         
                
                #pro_depth_masked = pro_depth*mask               #[B 1 H W]
                
                depth_con1_loss = depth_con1_n3.mean() * weight_list[level_id] 
                depth_loss = depth_con1_loss

                #^ color consistency 
                #rgb1_flatten = color12_list[0].permute(0,2,1)                       # [B 3 N]
                #rgb2 = color12_list[1].permute(0,2,1).reshape(b,-1,h,w)             # [B 3 H W]

                #pro_img = torch.nn.functional.grid_sample(rgb2, grid_tf, padding_mode='border')
                #pro_img_flatten = (pro_img * mask).view(b, 3, -1)           # [B 3 N]
                
                #pro_img_flatten = (pro_img_flatten - rgb1_flatten)*valid              
                #color_con1_loss = pro_img_flatten.abs().mean() * weight_list[level_id]    
                
                #color_loss = color_con1_loss
                
                #print(loss4.shape)
                #print("Level ",level_id)
                #print("depth loss:",depth_con1_loss)

                #~ Depth consistency Loss2 : loss computed from dynamic sf 
                #! pc1_RT 坐标上 
                new_warped_pc1 = pc1_cam2_list[level_id] + dynamic_sfs[level_id]    # [B N 3]       #! pc1三维坐标上
                new_warped_pc1_b3n = new_warped_pc1.permute(0,2,1)                  # [B 3 N]

                new_pcoords = new_intr.bmm(new_warped_pc1_b3n)  # [b 3 n]   #![三维点云投影到二维平面]
                X_ = new_pcoords[:, 0]           #! pc1_warped二维坐标上的3D值
                Y_ = new_pcoords[:, 1]
                Z = new_pcoords[:, 2].clamp(min=1e-3)       

                u = torch.clamp((X_ / Z), 0, w - 1)
                v = torch.clamp((Y_ / Z), 0, h - 1)

                #^ 二维像素阵列 分别除 w 和 h ,得到归一化像素平面 [0-1, 0-1]
                U_norm = (2*u / (w - 1) - 1).cpu()      # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
                V_norm = (2*v / (h - 1) - 1).cpu()
                grid_tf = torch.stack([U_norm, V_norm], dim=2).reshape(b,h,w,2).cuda()     

                mask_x_low = torch.BoolTensor(U_norm>-1).byte()
                mask_x_high = torch.BoolTensor(U_norm<1).byte()
                mask_y_low = torch.BoolTensor(V_norm>-1).byte()
                mask_y_high = torch.BoolTensor(V_norm<1).byte()
                mask = (mask_x_low * mask_x_high * mask_y_low * mask_y_high).cuda()
                mask = mask.view(b,1,h,w)

                #^ depth consistency    
                pro_depth = torch.nn.functional.grid_sample(pc2_depth, grid_tf, padding_mode='border',align_corners=True)
                #^ 深度[z] 加上 二维像素阵列 [u,v]  ， 乘以逆内参后得到 三维点云 [x,y,z]
                pro_xyz = BackprojectDepth(pro_depth, new_inv_intr)        # 三维点云 = 逆参矩阵 * 像素深度     [B 3 H W]
                pro_xyz_flatten = (pro_xyz*mask).view(b, 3, -1) 
                
                if i ==10:
                    #[mask_list[0]>0]
                    np.savetxt('./../cloud/pro_xyz_2.txt',(pro_xyz_flatten.permute(0,2,1)[mask_list[0]>0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
                    np.savetxt('./../cloud/new_warped_pc1.txt',(new_warped_pc1[mask_list[0]>0]).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

                depth_con2_n3 = (abs(pro_xyz_flatten - warped_pc1_b3n).permute(0,2,1))[valid]         
                depth_con2_loss = depth_con2_n3.mean() * weight_list[level_id]  

                depth_loss +=  depth_con2_loss

                #^ color consistency 
                #rgb1_flatten = color12_list[0].permute(0,2,1)                       # [B 3 N]
                #rgb2 = color12_list[1].permute(0,2,1).reshape(b,-1,h,w)             # [B 3 H W]

                #pro_img = torch.nn.functional.grid_sample(rgb2, grid_tf, padding_mode='border')
                #pro_img_flatten = (pro_img * mask).view(b, 3, -1)           # [B 3 N]
                
                #pro_img_flatten = (pro_img_flatten - rgb1_flatten)*valid              
                #color_con2_loss = pro_img_flatten.abs().mean() * weight_list[level_id]    
                
                #color_loss += color_con2_loss

            #~Dynamic-static Consistency Loss: sfo -(sfs+sfd)
            static_random_gt  = pc1_cam2_list[level_id] - pc1[level_id]                     # [b 各层点数 3] 
            diff_sdo_sf = abs(overall_sf - dynamic_sfs[level_id] - static_random_gt)        # [b n 3]
            diff_b3n_masked = torch.norm(diff_sdo_sf + 1e-20, dim=2).reshape(b,-1)[valid]   #[n]
            diff_sdo_sf_sum = diff_b3n_masked.mean() * weight_list[level_id]     

            #diff_b3n_masked = diff_sdo_sf.permute(0,2,1) * valid                             # [b 3 n]
            #diff_sdo_sf_sum = torch.norm(diff_b3n_masked + 1e-20, dim=1).mean() * weight_list[level_id]

            chamfer_loss, curvature_loss = multiScaleChamferSmoothCurvature(pc1[level_id], pc2[level_id], overall_sf, new_intr,h ,w , mask_list[level_id])
            valid_num = mask_list[level_id].reshape(b,-1).sum(dim=-1).mean()          # mean of valid number across all batches
            
            chamfer_loss = (chamfer_loss / valid_num)* weight_list[level_id]
            curvature_loss = (curvature_loss / valid_num) * weight_list[level_id]

            if level_id == 0:
                dynamic_loss = diff_sdo_sf_sum
                loss7 = chamfer_loss
                loss8 = curvature_loss
            else:
                dynamic_loss += diff_sdo_sf_sum
                loss7 += chamfer_loss
                loss8 += curvature_loss

        print("overall depth consistency loss1:",depth_con1_loss)
        print("overall depth consistency loss2:",depth_con2_loss)
        #print("overall color consistency loss:",color_con1_loss)
        #print("dynamic depth consistency loss:",loss41)
        #print("dynamic color consistency loss:",loss91)
        print("dynamic + static loss:",dynamic_loss)
        print("chamfer_loss:",loss7)
        print("curvature_loss:",loss8)

        training_writer.add_scalar("loss4: Depth Consistency Loss:", depth_con1_loss, n_iter)
        training_writer.add_scalar("loss41: Depth Consistency Loss of new flow:", depth_con2_loss, n_iter)
        training_writer.add_scalar("loss5: overall static dynamic Loss:", dynamic_loss, n_iter)
        training_writer.add_scalar("loss7: chamfer_loss:", loss7, n_iter)
        training_writer.add_scalar("loss8: curvature_loss:", loss8, n_iter)

        
        #loss = args.loss_weight_dc * (depth_con1_loss + depth_con2_loss) + args.loss_weight_ods * loss5 + args.chamfer_loss_weight * loss7 + args.curvature_loss_weight * loss8
        loss =  args.loss_weight_dc * depth_loss + args.loss_weight_ods * dynamic_loss + args.chamfer_loss_weight * loss7 + args.curvature_loss_weight * loss8
    
        loss = loss.float()
        print("loss:",loss)

        losses.update(loss.item())
        #losses.update(loss)

        for p in optimizer.param_groups:
            temp_lr = args.lr * (args.decay_rate ** (n_iter * args.batch_size / args.decay_step))
            temp_lr = max(0.0001, temp_lr)
            p['lr'] = temp_lr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_iter += 1

    print("losses.avg[0]:",losses.avg[0])
    return losses.avg[0]


def eval_kitti_sceneflow(scene_net, test_loader):
    
    scene_net.eval()
    
    # 3D metrics
    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    
    # 2D metrics
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    for i, data in enumerate(test_loader, 0):
        
        model = scene_net.eval()
        # pc1, pc2, flow, intrinsics, sampled_index = data
        pc1, pc2, color1, color2, flow, intrinsics, occ_mask = data
        
        # move to cuda
        b = pc1.shape[0]
        pc1 = pc1.cuda(non_blocking = True)
        pc2 = pc2.cuda(non_blocking = True)
        color1 = color1.cuda(non_blocking = True)
        color2 = color2.cuda(non_blocking = True)
        flow = flow.cuda(non_blocking = True)
        intrinsics = intrinsics.cuda(non_blocking = True)
        occ_mask = occ_mask.cuda(non_blocking = True)    
        #print("intrinsics:",intrinsics.shape)

        flow3d_list, mask_list, pc1_sample ,pc2_sample ,height_list, width_list, flow3d_gt_list = scene_net(pc1, pc2, color1, color2, intrinsics[0], occ_mask.unsqueeze(-1), flow, False )
        #print("pc1_sample:",pc1_sample[0].shape)

        #np.savetxt('./../cloud/pc1_02_27.txt',(pc1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #np.savetxt('./../cloud/pc2_02_27.txt',(pc2).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

        #np.savetxt('./../cloud/pc1_masked_02_27.txt',(pc1_sample[0]).permute(0,2,1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #np.savetxt('./../cloud/pc1_warped_masked_02_27.txt',(pc1_sample[0] + flow3d_list[0]).permute(0,2,1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')
        #np.savetxt('./../cloud/pc2_masked_02_27.txt',(pc2_sample[0]).permute(0,2,1).reshape(-1,3).cpu().detach().numpy(),fmt='%f %f %f')

        flow3d_pred_np =  (flow3d_list[0])               # [b n 3]
        flow3d_label_np = flow3d_gt_list[0]              # [b n 3] 
        pc1_3d_np = (pc1_sample[0])                      # [b n 3] 
        mask = (mask_list[0])              # [b n]  

        intrinsic_np = intrinsics.detach().cpu().numpy()
        pred_sf = flow3d_pred_np[mask].detach().cpu().numpy()
        sf_np = flow3d_label_np[mask].detach().cpu().numpy()
        pc1_np = pc1_3d_np[mask].detach().cpu().numpy()
        
        if pc1_np.shape[0] == 0:
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

        # 2D evaluation metrics
        flow_pred_2d, flow_gt_2d = get_batch_2d_flow(pc1_np, pc1_np + sf_np, pc1_np + pred_sf, intrinsic_np, height_list[0], width_list[0])
        EPE2D, acc2d = evaluate_2d(flow_pred_2d, flow_gt_2d)

        print('pre eval mean EPE 3D:' , (EPE3D))
        print('pre eval mean acc3d_1:', (acc3d_strict))
        print('pre eval mean acc3d_2 :', (acc3d_relax))
        print('pre eval mean outlier3d :', (outlier))
        print('pre eval mean EPE 2D :', (EPE2D))
        print('pre eval mean acc2d :', (acc2d))

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)

    print("Evaluating Finished")

    return epe3ds.avg[0], acc3d_stricts.avg[0], acc3d_relaxs.avg[0], outliers.avg[0] , epe2ds.avg[0], acc2ds.avg[0]



if __name__ == '__main__':
    print("run train.py, start here: ---------")
    main()