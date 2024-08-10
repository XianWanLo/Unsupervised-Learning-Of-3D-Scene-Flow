import sys
import os
import torch
import numpy as np
import torch.utils.data
import pprint
from dataset.kitti_sf_dataset import KITTI
import cmd_args
from utils import AverageMeter
from utils.geometry import get_batch_2d_flow
from utils.evaluation_utils import evaluate_2d, evaluate_3d
from model.model_pytorch import ESBFlow

MAX_FLOW = 0.3
LAYER_IDX = 0
def main():

    torch.backends.cudnn.benchmark = False

    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])
    print('args: {}'.format(pprint.pformat(args)))
    print("Current PID:", os.getpid())



    model = ESBFlow(H_input=args.height,
                    W_input=args.width,
                    is_training=False)

    val_dataset = KITTI(args.data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_worker,
        pin_memory=True)

    assert os.path.exists(args.load_path), "unknown load_path"

    print('load model from %s' % args.load_path)
    
        
    state_dict = torch.load(args.load_path)
    
    model.load_state_dict(state_dict)
    

    model.cuda()
    model = model.eval()

    # 3D metrics
    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()
    
    # 2D metrics
    epe2ds = AverageMeter()
    acc2ds = AverageMeter()

    for i, data in enumerate(val_loader, 0):
        
        pc1, pc2, flow, intrinsics, occ_mask = data
        pc1 = pc1.cuda(non_blocking = True)
        pc2 = pc2.cuda(non_blocking = True)
        flow = flow.cuda(non_blocking = True)
        intrinsics = intrinsics.cuda(non_blocking = True)
        occ_mask = occ_mask.cuda(non_blocking = True) 
        
        #! 需要将 pc1,pc2, flow, mask 从 376,1242 转换成 256, 832

        pred_flow, label_flow, valid_mask, pc1_sample = model(pc1, pc2, flow, intrinsics, occ_mask)

        pc1_np = pc1_sample[LAYER_IDX][valid_mask[LAYER_IDX]].cpu().numpy()
        intrinsic_np = intrinsics.cpu().numpy()
        sf_np = label_flow[LAYER_IDX][valid_mask[LAYER_IDX]].detach().cpu().numpy()
        pred_sf = pred_flow[LAYER_IDX][valid_mask[LAYER_IDX]].detach().cpu().numpy()
        
        if pc1_np.shape[0] == 0:
            continue

        # print("Valid num of points: {} / {}".format(pc1_np.shape[0], valid_mask[0].shape[1]))
        # print("Max flow %f" % np.max(np.sum(sf_np ** 2, axis = -1)))
        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

        # 2D evaluation metrics
        flow_pred_2d, flow_gt_2d = get_batch_2d_flow(pc1_np, pc1_np + sf_np, pc1_np + pred_sf, intrinsic_np)
        EPE2D, acc2d = evaluate_2d(flow_pred_2d, flow_gt_2d)

        epe2ds.update(EPE2D)
        acc2ds.update(acc2d)


    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
               'EPE2D {epe2d_.avg:.4f}\t'
               'ACC2D {acc2d_.avg:.4f}'.format(epe3d_=epe3ds,
                                               acc3d_s=acc3d_stricts,
                                               acc3d_r=acc3d_relaxs,
                                               outlier_=outliers,
                                               epe2d_=epe2ds,
                                               acc2d_=acc2ds))

    print(res_str)

    print("Evaluating Finished")




if __name__ == '__main__':
    main()
