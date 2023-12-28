import os.path

import numpy as np
import torch
import torch.utils.data as data
import json 

CATE_LIST = ['dishwasher', 'microwave', 'oven', 'refrige', 'storage', 'table', 'trashcan']

class ComDataset(data.Dataset):

    def __init__(
        self, 
        data_dir, 
        com_data_dir, 
        motion_dir, 
        data_index_path, 
        use_joint=True
    ):
        self.cate_shape_list_dict = {}
        self.data_list = []
        self.data_dir = data_dir
        self.com_data_dir = com_data_dir
        self.motion_list = []
        self.use_joint = use_joint
        
        if use_joint:
            self.joint_list = []
            
        need_map = set()
        with open(data_index_path, "r") as fp:
            for line in fp.readlines():
                cate, shape_id = line.rstrip().split(" ")
                need_map.add(f"{cate}{shape_id}")
        for cate in CATE_LIST:
            for file_name in os.listdir(f"{data_dir}/{cate}"):
                info_list = file_name[:-4].split('_')
                shape_id = info_list[0]
                sid = info_list[1]
                info_key = f"{cate}{shape_id}"
                if info_key in need_map:
                    with open(f"{motion_dir}/{cate}/{shape_id}_{sid}.json" , "r") as json_fp:
                        info = json.load(json_fp)
                        idmap = info['idmap']
                        valid = info['valid']
                        if len(info_list) != 2:
                            pid = info_list[2]
                            if valid[idmap[str(int(pid[1:]))]] != 1:
                                continue
                        self.data_list.append(f"{cate}/{file_name}")
                        if use_joint:
                            joint_info = np.zeros((1, 7),dtype=np.float32)
                            if len(info_list) != 2:
                                pid = info_list[2]
                                joint_info[0, 0] = info['type'][idmap[str(int(pid[1:]))]]
                                joint_info[0, 1:4] = info['axispos'][idmap[str(int(pid[1:]))]]
                                joint_info[0, 4:] = info['axisdir'][idmap[str(int(pid[1:]))]]
                            self.joint_list.append(joint_info)
                  
    def __getitem__(self, index):
        pcds = np.loadtxt(f"{self.data_dir}/{self.data_list[index]}").astype(np.float32)
        pcds_com = np.loadtxt(f"{self.com_data_dir}/{self.data_list[index]}").astype(np.float32)
        np.random.shuffle(pcds)
        pcds = torch.from_numpy(pcds) - 0.5
        pcds_com = torch.from_numpy(pcds_com) - 0.5

        while pcds.shape[0]<2048:
            pcds = torch.cat([pcds, pcds], dim=0)
        pcds = pcds[:2048, :]

        joint_info = None
        if self.use_joint:
            joint_info = self.joint_list[index]
            joint_info = torch.from_numpy(joint_info)
            
        return pcds, pcds_com, self.data_list[index], joint_info

    def __len__(self):
        return len(self.data_list)