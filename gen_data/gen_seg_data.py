import os 
import numpy as np
from openpoints.models.layers import furthest_point_sample
import torch 
from tqdm import tqdm
import sys 

data_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_front"
motion_dir = "/mnt/disk3/zihao/20220610data/InterRecon/motion_split"

save_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_seg"

N = 4096

def get_dis(p1, p2):
    return np.sum((p1 - p2) ** 2)

def fps_sample(xyz, seg, num_points):
    xyz = torch.from_numpy(xyz)[None,...].cuda()
    seg = torch.from_numpy(seg)[None,...].cuda()
    fps_idx = furthest_point_sample(xyz, num_points)
    fps_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, 3))
    fps_seg = torch.gather(seg, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, 1))
    fps_xyz = fps_xyz.cpu().numpy()[0]
    fps_seg = fps_seg.cpu().numpy()[0]
    return fps_xyz, fps_seg

def work(cate, file_name):
    shape_id, sid, pid, fid = file_name[:-4].split('_')
    
    pcs_start = np.loadtxt(f"{data_dir}/{cate}/{shape_id}_{sid}_{pid}_f00.pts").astype(np.float32)
    pcs_end = np.loadtxt(f"{data_dir}/{cate}/{shape_id}_{sid}_{pid}_f09.pts").astype(np.float32)
    
    pidt = int(pid[1:])
    
    pcs_start_xyz = pcs_start[:, :3]
    pcs_end_xyz = pcs_end[:, :3]
    
    pc_start_seg = pcs_start[:, -1:].astype(np.int32)
    pc_end_seg = pcs_end[:, -1:].astype(np.int32)
    
    pcs_start_xyz, pc_start_seg = fps_sample(pcs_start_xyz, pc_start_seg, N)
    pcs_end_xyz,  pc_end_seg = fps_sample(pcs_end_xyz,  pc_end_seg, N)
    
    gt_start_seg = np.zeros_like(pc_start_seg).astype(np.int32)
    gt_start_seg[pc_start_seg == pidt] = 1

    gt_end_seg = np.zeros_like(pc_end_seg).astype(np.int32)
    gt_end_seg[...] = 2
    gt_end_seg[pc_end_seg == pidt] = 1
    
    for i in range(N):
        if gt_start_seg[i] == 0:
            dist = np.sum((pcs_end_xyz - pcs_start_xyz[i:i+1,:])**2, axis=1)
            dist[gt_end_seg[:,0]==1] = 1000
            gt_end_seg[np.argsort(dist)[:8]] = 0
            
    gt_start = np.concatenate([pcs_start_xyz, pc_start_seg], axis=1)
    gt_end = np.concatenate([pcs_end_xyz, gt_end_seg], axis=1)
    
    np.savetxt(f"{save_dir}/{cate}/{shape_id}_{sid}.pts", gt_start)
    np.savetxt(f"{save_dir}/{cate}/{shape_id}_{sid}_{pid}.pts", gt_end)
    

if __name__ == '__main__':
    cate = sys.argv[-2]
    file_name = sys.argv[-1]
    work(cate, file_name)