import torch
import torch.nn.functional as F
import numpy as np
from utils.csrc import k_nearest_neighbor

MIN_DEPTH = 0.05


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


def remove_zeros_b3n(pc1,pc2,pred,label):
    """
    Args:
        pred (np.ndarray): [predicted flow with shape (B, 3, N)]
        label (np.ndarray): [flow label with shape (B, N, 3)]
    Returns:
        Tuple(predicted flow without zeros points, label flow without zeros points) both shape (N', 3)
    """

    #mask = np.all(np.equal(label,np.zeros_like(label)),axis=-1)
    
    pc1 = pc1.permute(0,2,1)            # B N 3
    pc2 = pc2.permute(0,2,1)            # B N 3
    pred = pred.permute(0,2,1)          # B N 3
    
    mask = (label == 0).all(axis=-1)    # B N

    rectified_pc1 = pc1[~mask]          # N 3 
    rectified_pc2 = pc2[~mask]          # N 3 
    rectified_pred = pred[~mask]        # N 3

    rectified_pc1 = rectified_pc1.unsqueeze(0).permute(0,2,1)       # 1 3 N
    rectified_pc2 = rectified_pc2.unsqueeze(0).permute(0,2,1)       # 1 3 N
    rectified_pred = rectified_pred.unsqueeze(0).permute(0,2,1)     # 1 3 N

    return rectified_pc1, rectified_pc2, rectified_pred




def remove_zeros(pred, label):
    """
    Args:
        pred (np.ndarray): [predicted flow with shape (B, N, 3)]
        label (np.ndarray): [flow label with shape (B, N, 3)]
    Returns:
        Tuple(predicted flow without zeros points, label flow without zeros points) both shape (N', 3)
    """
    
    #mask = np.all(np.equal(label,np.zeros_like(label)),axis=-1)
    mask = (label == 0).all(axis=-1)   

    rectified_pred = pred[~mask]
    rectified_flow = label[~mask]


    return rectified_pred, rectified_flow



def project(Xs, intrinsics):
    """ Pinhole camera projection """
    X, Y, Z = Xs.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[:,None,None].unbind(dim=-1)

    x = fx * (X / Z) + cx
    y = fy * (Y / Z) + cy
    d = 1.0 / Z

    coords = torch.stack([x, y, d], dim=-1)
    return coords



def inv_project(depths, intrinsics):
    """ Pinhole camera inverse-projection """

    ht, wd = depths.shape[-2:]
    
    fx, fy, cx, cy = \
        intrinsics[:,None,None].unbind(dim=-1)

    y, x = torch.meshgrid(
        torch.arange(ht).to(depths.device).float(), 
        torch.arange(wd).to(depths.device).float())

    X = depths * ((x - cx) / fx)
    Y = depths * ((y - cy) / fy)
    Z = depths

    return torch.stack([X, Y, Z], dim=-1)



def induced_flow(Ts, depth, intrinsics):
    """ Compute 2d and 3d flow fields """

    X0 = inv_project(depth, intrinsics)
    X1 = Ts * X0

    x0 = project(X0, intrinsics)
    x1 = project(X1, intrinsics)

    flow2d = x1 - x0
    flow3d = X1 - X0

    valid = (X0[...,-1] > MIN_DEPTH) & (X1[...,-1] > MIN_DEPTH)
    return flow2d, flow3d, valid.float()



def backproject_flow3d(flow2d, depth0, depth1, intrinsics):
    """ compute 3D flow from 2D flow + depth change """

    ht, wd = flow2d.shape[0:2]

    fx, fy, cx, cy = \
        intrinsics[None].unbind(dim=-1)
    
    y0, x0 = torch.meshgrid(
        torch.arange(ht).to(depth0.device).float(), 
        torch.arange(wd).to(depth0.device).float())

    x1 = x0 + flow2d[...,0]
    y1 = y0 + flow2d[...,1]

    X0 = depth0 * ((x0 - cx) / fx)
    Y0 = depth0 * ((y0 - cy) / fy)
    Z0 = depth0

    X1 = depth1 * ((x1 - cx) / fx)
    Y1 = depth1 * ((y1 - cy) / fy)
    Z1 = depth1

    flow3d = torch.stack([X1-X0, Y1-Y0, Z1-Z0], dim=-1)
    return flow3d



def batch_indexing_channel_first(batched_data: torch.Tensor, batched_indices: torch.Tensor):
    """
    :param batched_data: [batch_size, C, N]
    :param batched_indices: [batch_size, I1, I2, ..., Im]
    :return: indexed data: [batch_size, C, I1, I2, ..., Im]
    """
    def product(arr):
        p = 1
        for i in arr:
            p *= i
        return p
    assert batched_data.shape[0] == batched_indices.shape[0]
    batch_size, n_channels = batched_data.shape[:2]
    indices_shape = list(batched_indices.shape[1:])
    batched_indices = batched_indices.reshape([batch_size, 1, -1])
    batched_indices = batched_indices.expand([batch_size, n_channels, product(indices_shape)])
    result = torch.gather(batched_data, dim=2, index=batched_indices.to(torch.int64))
    result = result.view([batch_size, n_channels] + indices_shape)
    return result


def knn_interpolation(input_xyz, input_features, query_xyz, k=3):
    """
    :param input_xyz: 3D locations of input points, [batch_size, 3, n_inputs]
    :param input_features: features of input points, [batch_size, n_features, n_inputs]
    :param query_xyz: 3D locations of query points, [batch_size, 3, n_queries]
    :param k: k-nearest neighbor, int
    :return interpolated features: [batch_size, n_features, n_queries]
    """
    knn_indices = k_nearest_neighbor(input_xyz, query_xyz, k)  # [batch_size, n_queries, 3]
    knn_xyz = batch_indexing_channel_first(input_xyz, knn_indices)  # [batch_size, 3, n_queries, k]
    knn_dists = torch.linalg.norm(knn_xyz - query_xyz[..., None], dim=1).clamp(1e-8)  # [bs, n_queries, k]
    knn_weights = 1.0 / knn_dists  # [bs, n_queries, k]
    knn_weights = knn_weights / torch.sum(knn_weights, dim=-1, keepdim=True)  # [bs, n_queries, k]
    knn_features = batch_indexing_channel_first(input_features, knn_indices)  # [bs, n_features, n_queries, k]
    interpolated = torch.sum(knn_features * knn_weights[:, None, :, :], dim=-1)  # [bs, n_features, n_queries]

    return interpolated


def backwarp_3d(xyz1, xyz2, flow12, k=3):
    """
    :param xyz1: 3D locations of points1, [batch_size, 3, n_points]
    :param xyz2: 3D locations of points2, [batch_size, 3, n_points]
    :param flow12: scene flow, [batch_size, 3, n_points]
    :param k: k-nearest neighbor, int
    """
    xyz1_warp = xyz1 + flow12
    flow21 = knn_interpolation(xyz1_warp, -flow12, query_xyz=xyz2, k=k)
    xyz2_warp = xyz2 + flow21
    return 
    
