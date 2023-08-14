import numpy as np
import torch
import os
from tqdm import tqdm
import json 

motion_dir = "/mnt/disk3/zihao/20220610data/InterRecon/motion_split"

data_dir_incom = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_com/"
data_dir_com = "/mnt/disk3/zihao/20220610data/InterRecon/obj_iter_part_pts/"

save_dir_incom = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_com_norm3/"
save_dir_com = "/mnt/disk3/zihao/20220610data/InterRecon/obj_iter_part_pts_norm3/"

def norm_one(pcs):
    pcds_cat = pcs
    x_max = np.max(pcds_cat[:, 0])
    y_max = np.max(pcds_cat[:, 1])
    z_max = np.max(pcds_cat[:, 2])
    x_min = np.min(pcds_cat[:, 0])
    y_min = np.min(pcds_cat[:, 1])
    z_min = np.min(pcds_cat[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)
    
    pcs[:, 0] = (pcs[:, 0] - x_mid) / scale
    pcs[:, 1] = (pcs[:, 1] - y_mid) / scale
    pcs[:, 2] = (pcs[:, 2] - z_mid) / scale
    
    return pcs

def norm_two(pcs1, pcs2):
    pcs1 = norm_one(pcs1)
    pcs2 = norm_one(pcs2)
    return pcs1, pcs2
    
for cate in os.listdir(data_dir_incom):
    print("cate: ", cate)
    os.makedirs(f"{save_dir_incom}/{cate}", exist_ok=True)
    os.makedirs(f"{save_dir_com}/{cate}", exist_ok=True)
    static_map = {}
    for file_name in os.listdir(f"{data_dir_com}/{cate}"):
        shape_id, sid, pid, fid, rpid = file_name[:-4].split('_')
        if not os.path.exists(f"{motion_dir}/{cate}/{shape_id}_{sid}.json"):
            continue
        refpart = 0
        with open(f"{motion_dir}/{cate}/{shape_id}_{sid}.json" , "r") as json_fp:
            info = json.load(json_fp)
            refpart = info['refpart']
        if fid == 'f00' and int(rpid) == refpart:
            static_map[f"{shape_id}_{sid}"] = file_name
            
    for file_name in tqdm(os.listdir(f"{data_dir_incom}/{cate}")):
        if 'p' in file_name[:-4]:
            shape_id, sid, pid = file_name[:-4].split('_')
            com_path = f"{data_dir_com}/{cate}/{shape_id}_{sid}_{pid}_f09_{pid[1:]}.pts"
            if os.path.exists(com_path):
                pcds = np.loadtxt(f"{data_dir_incom}/{cate}/{file_name}").astype(np.float32)
                if pcds.shape[0] < 100:
                    continue
                pcds_com = np.loadtxt(com_path).astype(np.float32)
                pcds, pcds_com = norm_two(pcds[:, :3], pcds_com)
                
                np.savetxt(f"{save_dir_incom}/{cate}/{file_name}", pcds)
                np.savetxt(f"{save_dir_com}/{cate}/{file_name}", pcds_com)
        else:
            shape_id, sid = file_name[:-4].split('_')
            if f"{shape_id}_{sid}"  in static_map:
                com_path = f"{data_dir_com}/{cate}/"+ static_map[f"{shape_id}_{sid}"]
                pcds = np.loadtxt(f"{data_dir_incom}/{cate}/{file_name}").astype(np.float32)
                pcds_com = np.loadtxt(com_path).astype(np.float32)
                pcds, pcds_com = norm_two(pcds[:, :3], pcds_com)
                
                np.savetxt(f"{save_dir_incom}/{cate}/{file_name}", pcds)
                np.savetxt(f"{save_dir_com}/{cate}/{file_name}", pcds_com)