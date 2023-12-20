import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import os
import json 

# CATE_LIST = ['dishwasher', 'kitchenpot', 'microwave', 'oven', 'refrige', 'storage', 'table', 'trashcan']
CATE_LIST = ['dishwasher', 'microwave', 'oven', 'refrige', 'storage', 'table', 'trashcan']
# CATE_LIST =['trashcan']
class SegDataset(data.Dataset):
    def __init__(self, data_dir, motion_dir, data_index_path, use_joint=True):
        self.cate_shape_list_dict = {}
        self.pm_path_list = []
        self.ps_path_list = []
        self.data_dir = data_dir
        self.pid_list = []
        self.rid_list = []
        self.use_joint = use_joint
        
        if use_joint:
            self.joint_list = []
        
        need_map = set()
        with open(data_index_path, "r") as fp:
            for line in fp.readlines():
                cate, shape_id = line.rstrip().split(" ")
                need_map.add(f"{cate}{shape_id}")

        for cate in CATE_LIST:
            for file_name in os.listdir(f"{self.data_dir}/{cate}"):
                if 'p' not in file_name[:-4]:
                    continue
                shape_id, sid, pid = file_name[:-4].split('_')
                info_key = f"{cate}{shape_id}"
                if info_key in need_map:
                    with open(f"{motion_dir}/{cate}/{shape_id}_{sid}.json" , "r") as json_fp:
                        info = json.load(json_fp)
                        idmap = info['idmap']
                        valid = info['valid']
                        if valid[idmap[str(int(pid[1:]))]] == 1:
                            self.pm_path_list.append(f"{cate}/{file_name}")
                            self.ps_path_list.append(f"{cate}/{shape_id}_{sid}.pts")
                            self.pid_list.append(int(pid[1:]))
                            # load motion 
                            if use_joint:
                                joint_info = np.zeros((1, 7),dtype=np.float32)
                                joint_info[0, 0] = info['type'][idmap[str(int(pid[1:]))]]
                                joint_info[0, 1:4] = info['axispos'][idmap[str(int(pid[1:]))]]
                                joint_info[0, 4:] = info['axisdir'][idmap[str(int(pid[1:]))]]
                                self.joint_list.append(joint_info)
                                
    def __getitem__(self, index):
        pcs_start_full = np.loadtxt(f"{self.data_dir}/{self.ps_path_list[index]}").astype(np.float32)
        pcs_end_full = np.loadtxt(f"{self.data_dir}/{self.pm_path_list[index]}").astype(np.float32)

        pc_start_xyz = pcs_start_full[:, :3].astype(np.float32)
        pc_end_xyz = pcs_end_full[:, :3].astype(np.float32)
        
        pc_start_xyz = pc_start_xyz + np.random.randn(pc_start_xyz.shape[0], 3) * 0.008
        pc_end_xyz = pc_end_xyz + np.random.randn(pc_end_xyz.shape[0], 3) * 0.008
        pc_start_xyz = pc_start_xyz.astype(np.float32)
        pc_end_xyz = pc_end_xyz.astype(np.float32)

        pc_start_seg = pcs_start_full[:, -1:].astype(np.int32)
        pc_end_seg = pcs_end_full[:, -1:].astype(np.int32)

        gt_start_seg = np.zeros_like(pc_start_seg).astype(np.int32)
        gt_start_seg[pc_start_seg == self.pid_list[index]] = 1
        pc_start_seg = gt_start_seg
        
        # gt_end_seg = np.zeros_like(pc_end_seg).astype(np.int32)
        # gt_end_seg[...] = 2
        # gt_end_seg[pc_end_seg == self.pid_list[index]] = 1
        
        pcs_cat = np.concatenate([pc_start_xyz, pc_end_xyz], axis=0)
        pcs_cat = torch.from_numpy(pcs_cat).float()
        feat_start = torch.ones((3, pc_start_xyz.shape[0])).float()
        feat_end = torch.zeros((3, pc_start_xyz.shape[0])).float()
        feat = torch.cat([feat_start, feat_end], dim=1)
        data = {'pos': pcs_cat, 'x': feat}
        
        pc_start_xyz = torch.from_numpy(pc_start_xyz).float()
        pc_end_xyz = torch.from_numpy(pc_end_xyz).float()
        gt_start_seg = torch.from_numpy(pc_start_seg).long()
        gt_end_seg = torch.from_numpy(pc_end_seg).long()


        joint_info = None 
        if self.use_joint:
            joint_info = torch.from_numpy(self.joint_list[index])

        return data, pc_start_xyz, pc_end_xyz, gt_start_seg, gt_end_seg, self.pid_list[index], self.pm_path_list[index], joint_info

    def __len__(self):
        return len(self.ps_path_list)

