import os 
import numpy as np 
from tqdm import tqdm
import json 

seg_data_dir = "data/seg"
motion_dir = "data_raw/motion"

com_data_dir = "/home/ubuntu/zihao/dev/Interaction-Driven-Reconstruction/data/com"

for cate in os.listdir(seg_data_dir):
    data_map = {}
    os.makedirs(f"{com_data_dir}/{cate}", exist_ok=True)
    print(f"save in {com_data_dir}/{cate}")
    for file_name in tqdm(os.listdir(f"{seg_data_dir}/{cate}")):
        if 'p' not in file_name[:-4]:
            shape_id, sid = file_name[:-4].split('_')
            pcs = np.loadtxt(f"{seg_data_dir}/{cate}/{file_name}").astype(np.float32)
            refpart = 0
            if not os.path.exists(f"{motion_dir}/{cate}/{shape_id}_{sid}.json"):
                continue
            with open(f"{motion_dir}/{cate}/{shape_id}_{sid}.json" , "r") as json_fp:
                info = json.load(json_fp)
                refpart = info['refpart']
            pcs_static = pcs[pcs[:,-1]==refpart]
            used_key = f"{shape_id}_{sid}"
            if used_key not in data_map:
                data_map[used_key] = pcs_static
            else:
                data_map[used_key] = np.concatenate([data_map[used_key], pcs_static], axis=0)
        else:
            pcs = np.loadtxt(f"{seg_data_dir}/{cate}/{file_name}").astype(np.float32)
            shape_id, sid, pid = file_name[:-4].split('_')
            used_key = f"{shape_id}_{sid}"
            static_more = pcs[pcs[:, -1] == 2]
            pcs_move = pcs[pcs[:, -1] == 1]
            np.savetxt(f"{com_data_dir}/{cate}/{shape_id}_{sid}_{pid}.pts", pcs_move)
            
            if used_key not in data_map:
                data_map[used_key] = static_more
            else:
                data_map[used_key] = np.concatenate([data_map[used_key], static_more], axis=0)
                
    for static_key in tqdm(data_map.keys()):
        np.savetxt(f"{com_data_dir}/{cate}/{static_key}.pts", data_map[static_key])
    