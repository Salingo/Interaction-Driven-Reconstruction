import h5py
import json 
import numpy as np 
import os 

BASE_DIR = "/mnt/disk3/zihao/20220610data/InterRecon"
CATE_LIST = ['dishwasher','kitchenpot','microwave','oven','refrige','storage','table','trashcan']
MOTION_DIR = f"{BASE_DIR}/motion"
SPLIT_MOTION_DIR = f"{BASE_DIR}/motion_split"

for cate in CATE_LIST:
    os.makedirs(f"{SPLIT_MOTION_DIR}/{cate}",exist_ok=True)

    h5_file = h5py.File(f"{MOTION_DIR}/{cate}_motion.h5", 'r')
    name_list = h5_file["name"][:]
    name_list = [name.decode('utf-8').split("_p")[0] for name in name_list]
    valid_list = h5_file["valid"][:].astype(np.int32)
    partnum_list = h5_file["partnum"][:].astype(np.int32)
    movpart_list = h5_file["movpart"][:].astype(np.int32)
    axisdir_list = h5_file["axisdir"][:].astype(np.float32)
    axispos_list = h5_file["axispos"][:].astype(np.float32)
    refpart_list = h5_file["refpart"][:].astype(np.int32)
    type_list = h5_file["type"][:].astype(np.int32)

    len_data = len(name_list)
    for i in range(len_data):
        name = name_list[i]
        shape_id, sid = name.split('_')
        idmap = {}
        info = {}
        part_num = int(partnum_list[i])
        for j in range(part_num-1):
            idmap[str(movpart_list[i][j])] = j
        info['name'] = name
        info['axispos'] = axispos_list[i].tolist()[:part_num-1]
        info['axisdir'] = axisdir_list[i].tolist()[:part_num-1]
        info['partnum'] = part_num
        info['movepart'] = movpart_list[i].tolist()[:part_num]
        info['idmap'] = idmap
        info['refpart'] = int(refpart_list[i][0])
        if cate == 'kitchenpot' and shape_id == '102085':
            info['refpart'] = 1
        info['type'] = type_list[i].tolist()[:part_num-1]
        info['valid'] = valid_list[i].tolist()[:part_num-1]
        json_save_path = f"{SPLIT_MOTION_DIR}/{cate}/{name}.json" 
        with open(json_save_path, "w") as outfile:
            json.dump(info, outfile)
    h5_file.close()