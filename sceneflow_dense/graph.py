import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import coo_matrix
CMAP = 'jet'

def gray2rgb(im, cmap= CMAP):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    # print('rgba: ','\n' , rgba_img)
    rgb_img = np.delete(rgba_img, 3, 1) # (array, obj, axis)
    # print('rgb: ','\n', rgb_img)
    return rgb_img


def normalize_depth_for_display(depth, pc=98, crop_percent=0, normalizer=None, reciprocal=False, cmap=CMAP):
    """Converts a depth map to an RGB image."""
    # Convert to disparity.
    if reciprocal:
        disp = 1.0 / (depth + 1e-6)
    else:
        disp = (depth + 1e-6)

    if normalizer is not None:
        disp /= normalizer
    else:
        disp /= (np.percentile(disp, pc) + 1e-6) #percentile: from the minimum to the maximum
    
    disp = np.clip(disp, 0, 1)
    '''
    clip: For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1
    '''
    disp = gray2rgb(disp, cmap=cmap)

    keep_h = int(disp.shape[0] * (1 - crop_percent))
    disp = disp[:keep_h]

    return disp


def plot_graph(pc1,filename,height=540,width=960):
    
    r = np.linalg.norm((pc1[0, : ,:2]), axis=-1)

    pc1_x = np.reshape(np.tile(np.expand_dims(np.arange(height), axis=-1), [1, width]), [-1])
    pc1_y = np.reshape(np.tile(np.expand_dims(np.arange(width), axis=0), [height, 1]), [-1])

    # pc1_x, pc1_y, depth = project_3d_to_2d(pc1)
    disp = normalize_depth_for_display(r)

    plt.scatter(pc1_y, pc1_x, c=disp, s = 1, marker=',')
    plt.axis('equal')

    plt.savefig(filename, dpi = 600)



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



def project_pc_to_2d_image(pc, out_h=540, out_w=960, f=-1050., cx=479.5, cy=269.5, constx=0., consty=0., constz=0., 
                           flow=None, pc2=None, pc_norm=None, pc2_norm=None):
    
    batch_size, num, _ = pc.shape
    
    #tf.constant -> torch.Tensor
    #f = torch.Tensor(f)
    #cx = torch.Tensor(cx)
    #cy = torch.Tensor(cy)
    #constx = torch.Tensor(constx)
    #consty = torch.Tensor(consty)
    #constz = torch.Tensor(constz)

    euclidean1 = torch.sqrt(torch.sum(torch.pow(pc, 2), 2)).view(batch_size, -1)
    euclidean2 = torch.sqrt(torch.sum(torch.pow(pc2, 2), 2)).view(batch_size, -1)


    for i in range(batch_size):
        
        # --------------------------------------- PC 1 ------------------------------------------
        current_pc = pc[i]      #(N 3)
        
        euclideani_1 = euclidean1[i, :].view(1, -1)     # (1 N)

        # crop the 2D image 
        width = torch.clamp((torch.reshape((current_pc[..., 0] * f + cx * current_pc[..., 2] + constx) / (current_pc[..., 2] + constz), (-1, 1))).int(), 0, out_w - 1)  # (N, 1)
        height = torch.clamp((torch.reshape((current_pc[..., 1] * f + cy * current_pc[..., 2] + consty) / (current_pc[..., 2] + constz), (-1, 1))).int(), 0, out_h - 1)  # (N, 1)
        indices = torch.cat([height, width], -1)        # (N 2)

        unique_hw, unique_idx = torch.unique((indices[:, 0] * out_w + indices[:, 1]),return_inverse=True)        # (比N少) 

        outputs = coo_matrix((torch.squeeze(1.0 / euclideani_1, 0).cuda().cpu(), (unique_idx.long().cuda().cpu(), torch.arange(0, euclideani_1.size()[1]).long().cuda().cpu())),  shape=(unique_hw.size()[0], euclideani_1.size()[1])).max(1)
        outputs = torch.squeeze(torch.from_numpy(outputs.toarray()).cuda())

        zuixiaojuli = torch.unsqueeze(torch.gather(1.0 / outputs, 0, torch.squeeze(unique_idx).cuda()), 0).float() 
        mask1_distance = torch.where(zuixiaojuli==euclideani_1,torch.ones_like(zuixiaojuli).cuda(), torch.zeros_like(zuixiaojuli).cuda()).view(-1)     # N 

        
        # ---------------------------------------- PC 2 ---------------------------------------------------
        current_pc2 = pc2[i]
        
        euclideani_2 = euclidean2[i, :].view(1, -1)

        width2 = torch.clamp((torch.reshape((current_pc2[..., 0] * f + cx * current_pc2[..., 2] + constx) / (current_pc2[..., 2] + constz), (-1, 1))).int(), 0, out_w - 1)  # (N, 1)
        height2 = torch.clamp((torch.reshape((current_pc2[..., 1] * f + cy * current_pc2[..., 2] + consty) / (current_pc2[..., 2] + constz), (-1, 1))).int(), 0, out_h - 1)  # (N, 1)
        indices2 = torch.cat([height2, width2], -1)

        unique_hw2, unique_idx2 = torch.unique((indices2[:, 0] * out_w + indices2[:, 1]),return_inverse=True)
        
        outputs = coo_matrix((torch.squeeze(1.0 / euclideani_2, 0).cuda().cpu(), (unique_idx2.long().cuda().cpu(), torch.arange(0, euclideani_2.size()[1] ).long().cuda().cpu())),  shape=(unique_hw2.size()[0], euclideani_2.size()[1])).max(1)
        outputs = torch.squeeze(torch.from_numpy(outputs.toarray()).cuda())
        
        zuixiaojuli = torch.unsqueeze(torch.gather(1.0 / outputs, 0, torch.squeeze(unique_idx2).cuda()), 0).float()
        mask2_distance = torch.where(zuixiaojuli==euclideani_2,torch.ones_like(zuixiaojuli).cuda(), torch.zeros_like(zuixiaojuli).cuda()).view(-1)
   

    # ------------------------------------- COMBINE PC 1&2 -------------------------------------- 

        final_mask = (mask1_distance * mask2_distance).unsqueeze(1)  # 0 = masked, 1 = available
        
        current_flow = flow[i]
        
        PC_idx_unique = (current_pc * final_mask).float()
        PC_idx_unique2 = (current_pc2 * final_mask).float()
        flow_idx_unique = (current_flow * final_mask).float()

        # the index for 2D image 
        indices = indices.long()               
        indices2 = indices2.long()
        
        # putting xyz values into indexed points at 2D image 
        current_image = torch.zeros(out_h, out_w, 3).cuda()
        current_image[indices[:,0],indices[ :,1], : ] = PC_idx_unique.float()
        current_image = current_image.reshape(1, out_h, out_w, 3)

        current_image2 = torch.zeros(out_h, out_w, 3).cuda()
        current_image2[indices2[:,0],indices2[ :,1], : ] = PC_idx_unique2.float()
        current_image2 = current_image2.reshape(1, out_h, out_w, 3)

        if flow is not None:
            
            current_flow = flow[i]
            current_label = torch.zeros(out_h, out_w, 3).cuda()
            current_label[indices[:,0],indices[ :,1], : ] = flow_idx_unique.float()
            current_label = current_label.reshape(1, out_h, out_w, 3)

        
        if i == 0:
            final_image = current_image        
            final_image2 = current_image2  
            
            if flow is not None:
                final_label = current_label
            
        else:
            
            final_image = torch.cat([final_image, current_image], 0)
            final_image2 = torch.cat([final_image2, current_image2], 0)
            
            if flow is not None:
                final_label = torch.cat([final_label, current_label], 0)
            
    if flow is not None:
        return final_image, final_image2, final_label
    else:
        return final_image, final_image2

def original_BN3_to_plot(pc1,pc2,flow,filename):
      
    pc1_try,pc2_try,flow_try = project_pc_to_2d_image(pc=pc1, pc2=pc2, flow=flow )

    pc1_try = pc1_try[0, : , : ].reshape(1,-1,3)
    pc2_try = pc2_try[0, : , : ].reshape(1,-1,3)
    flow_try = flow_try[0, : , : ].reshape(1,-1,3)

    plot_graph(pc1_try.cpu().detach().numpy(),'%s - pc1.png'%(filename))
    plot_graph((pc1_try+flow_try).cpu().detach().numpy(),'%s - pc1+flow.png'%(filename))
    plot_graph(pc2_try.cpu().detach().numpy(),'%s - pc2.png'%(filename))


def BHW3_to_plot(pc1,pc2,flow,filename):
            
    h = pc1.shape[1]
    w = pc1.shape[2]
    
    pc1_try = pc1[0,:,:,:].reshape(1,-1,3)
    pc2_try = pc2[0,:,:,:].reshape(1,-1,3)
    flow_try = flow[0,:,:,:].reshape(1,-1,3)

    plot_graph(pc1_try.cpu().detach().numpy(),'%s - pc1.png'%(filename),height=h,width=w)
    plot_graph((pc1_try+flow_try).cpu().detach().numpy(),'%s - pc1+flow.png'%(filename),height=h,width=w)
    plot_graph(pc2_try.cpu().detach().numpy(),'%s - pc2.png'%(filename),height=h,width=w)