import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import os
import json

from scipy.linalg import expm, norm

def randomRotateZ(pcs_dir):
    axis = np.array([0, 1, 0], dtype=np.float32)
    rotate_angle = np.random.uniform(-np.pi, np.pi)
    R = expm(np.cross(np.eye(3), axis / norm(axis) * rotate_angle))
    pcs = pcs_dir[:, :3]
    dire = pcs_dir[:, 3:]
    pcs_dir[:, :3] = np.dot(pcs, R)
    pcs_dir[:, 3:] = np.dot(dire, R)
    return pcs_dir


class ActionDataset(data.Dataset):

    def __init__(
        self, 
        data_dir, 
        motion_dir, 
        index_path, 
        cate_list,
        random_rotate=True, 
        outline_num=0
    ):

        self.data_dir = data_dir
        self.motion_dir = motion_dir
        self.index_path = index_path
        self.cate_list = cate_list
        self.random_rotate = random_rotate
        self.outline_num = outline_num
        
        self.data = []
        self.refpart_id = []
        need_map = set()
        with open(index_path, "r") as fp:
            for line in fp.readlines():
                cate, shape_id = line.rstrip().split(" ")
                need_map.add(f"{cate}{shape_id}")

        for cate in self.cate_list:
            for file_name in os.listdir(f"{self.data_dir}/{cate}"):
                shape_id, sid, pid, fid = file_name.split('_')
                info_key = f"{cate}{shape_id}"
                if info_key in need_map:
                    self.data.append(f"{self.data_dir}/{cate}/{file_name}")
                    motion_json_path = f"{self.motion_dir}/{cate}/{shape_id}_{sid}.json"
                    json_fp = open(motion_json_path, "r")
                    info = json.load(json_fp)
                    json_fp.close()
                    self.refpart_id.append(info['refpart'])



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path = self.data[index]
        pcs_dir_score = np.loadtxt(file_path, delimiter=',').astype(np.float32)
        # np.random.shuffle(pcs_dir_score)
        
        if self.outline_num != 0:
            outline_points = np.random.rand(self.outline_num, 3)
            pcs_dir_score[:self.outline_num, :3] = outline_points 
            
        pcs_dir_score[:, :3] = pcs_dir_score[:, :3] - 0.5
        # pcs_dir_score[:,:3] += np.random.normal(0, self.noise, size=pcs_dir_score[:,:3].shape)

        if self.random_rotate:
            pcs_dir_score[:, :6] = randomRotateZ(pcs_dir_score[:, :6])

        pcs_dir_score = torch.from_numpy(pcs_dir_score)
        return pcs_dir_score, file_path, self.refpart_id[index]
