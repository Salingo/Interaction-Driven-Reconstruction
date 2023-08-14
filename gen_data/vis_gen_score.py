import numpy as np
import os
import os,sys
import json
import random
import pandas as pd
import math
import matplotlib.pyplot as plt

MAX_LINK_NUM = 30

BASE_DIR = "/mnt/disk3/zihao/20220610data/InterRecon"
MOTION_DIR = "/mnt/disk3/zihao/20220610data/InterRecon/motion_split"
PCS_DIR = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_front"

DATA_DIR = "/mnt/disk3/zihao/dev/pointnet_tem/data"
SAVE_DIR = "/mnt/disk3/zihao/dev/pointnet_tem/data/score_front"

# ALL_CATE_LIST = ['dishwasher','kitchenpot','microwave','oven','refrige','storage','table','trashcan']
ALL_CATE_LIST = ['table']
SAMPLE_PER_PART = 1
COE_FRICTION = 0.3 
DEGREE_RANGE = 45
GEN_REF_PART = False

def generate_random_vector():
    v = np.random.randn(3)
    length = np.linalg.norm(v)
    unit_v = v / length
    return unit_v

def generate_vector(u):
    # Normalize the vector
    u = u / np.linalg.norm(u)

    # Generate two orthogonal vectors
    v1 = np.cross(u, np.array([1, 0, 0]))
    if np.linalg.norm(v1) < 1e-6:
        v1 = np.cross(u, np.array([0, 1, 0]))
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(u, v1)

    # Choose random angles
    degree_val = np.pi * DEGREE_RANGE / 360
    theta = np.random.uniform(-degree_val, degree_val)
    phi = np.random.uniform(0, 2*np.pi)

    # Compute the new vector
    v = np.cos(theta) * u + np.sin(theta) * (np.cos(phi) * v1 + np.sin(phi) * v2)

    return v 

# def get_probs(d_list):
#     di = [[d,i] for i, d in enumerate(d_list)]
#     di = sorted(di, key=lambda x:x[0])
    
#     n = len(di)
#     n1 = int(n/3)
#     n2 = n - 2*n1

#     probs = [0 for i in range(n)]

#     for i in range(n1):
#         probs[i] = 0.3 / n1
#         probs[i+n1] = 0.1 / n1 
#     for i in range(n2): 
#         probs[2*n1+i] = 0.6/n2

#     rp = [0 for i in range(n)]
#     for i in range(n):
#         rp[di[i][1]] = probs[i]

#     return rp 

def work(cate='trashcan', file_name='102210_s04_p02_f00.pts', sample_per_part=SAMPLE_PER_PART, coe_friction=COE_FRICTION):
    shape_id, sid, pid, fid = file_name.split("_") # 46452 s00 p00 f00
    # use files :
    pc_pts_path = f"{PCS_DIR}/{cate}/{file_name}"
    motion_json_path = f"{MOTION_DIR}/{cate}/{shape_id}_{sid}.json" 
    # save in :
    os.makedirs(f"{SAVE_DIR}/{cate}", exist_ok=True)
    save_path = f"{SAVE_DIR}/{cate}/{file_name}.pts"
    # load motion 
    json_fp = open(motion_json_path, "r")
    info = json.load(json_fp)
    json_fp.close()
    # load point clouds
    pcs = np.loadtxt(pc_pts_path).astype(np.float32) # (16384, 7) :px py pz nx ny nz part_id
    
    # split the indice by part_id
    # part_num = info['partnum']
    part_num = int(np.max(pcs[:,-1])) + 1
    refpart = info['refpart']
    idmap = info['idmap']
    valid = info['valid']
    indices = [[] for i in range(part_num)]
    for i in range(len(pcs)):
        indices[int(pcs[i][-1])].append(i)
    # get the average normal of the part.
    average_normal = [[0,0,0] for i in range(part_num)]
    part_point_num = [0 for i in range(part_num)]
    for item in pcs:
        part_point_num[int(item[-1])] += 1
        for i in range(3):
            average_normal[int(item[-1])][i] += item[3+i]
    for part_id in range(part_num):
        if part_point_num[part_id] != 0:
            for i in range(3):
                average_normal[part_id][i] /= part_point_num[part_id]

    # gen_data
    gen_data = [[] for i in range(part_num)]
    for i in range(part_num):
        if np.linalg.norm(average_normal[i]) < 1e-6:
            for index in indices[i]:
                p = pcs[index][:3]
                v = generate_random_vector()
                gen_data[i].append([p[0],p[1],p[2],v[0],v[1],v[2],0])
            continue
        if i == refpart:
            for index in indices[i]:
                p = pcs[index][:3]
                v = generate_random_vector()
                gen_data[i].append([p[0],p[1],p[2],v[0],v[1],v[2],0])
        elif str(i) in idmap.keys() and valid[idmap[str(i)]] == 1 :
            axispos = info['axispos'][idmap[str(i)]]
            axisdir = info['axisdir'][idmap[str(i)]]
            if info['type'][idmap[str(i)]] == 0:
                for index in indices[i]:
                    index = random.choice(indices[i])
                    print(index)
                    p = pcs[index][:3]
                    d = pcs[index][3:6]
                    score_list_ = []
                    d_list_ = []
                    mx_val_ = -1000
                    mi_val_ = 1000
                    for sample_id in range(25):
                        v = generate_vector(d)
                        cos_va = np.dot(v,axisdir)
                        sin_va = np.sqrt(1-cos_va**2)
                        dis = np.linalg.norm(np.cross(p-axispos,axisdir))
                        ### ADD some thing to:
                        # score = sin_va * d - coe_friction * cos_va
                        axispos_p = p - axispos
                        axispos_p /= np.linalg.norm(axispos_p)
                        cos__ = np.dot(axispos_p, axisdir)
                        help_d = np.array(axisdir) * cos__
                        help_d = help_d - axispos_p
                        action2out = np.dot(help_d, v) > 0

                        score = sin_va * dis - coe_friction * cos_va
                        print(score)
                        if action2out:
                            score -= coe_friction * cos_va
                        ####
                        # score = sin_va * dis - coe_friction * cos_va
                        score_list_.append(score)
                        d_list_.append(v)
                        mx_val_ = max(mx_val_, score)
                        mi_val_ = min(mi_val_, score)
                    score_list_ = [(score-mi_val_)/(mx_val_-mi_val_) for score in score_list_]
                    x = pcs[:, 0]
                    y = pcs[:, 1]
                    z = pcs[:, 2]
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.axis([0, 1, 0, 1])
                    plt.axis('off')
                    ax.scatter(z, x, y)
                    for i_155 in range(25):
                        color = np.array([score_list_[i_155], 1, 1])
                        ax.quiver(pcs[index][2], pcs[index][0], pcs[index][1],
                                  d_list_[i_155][2], d_list_[i_155][0], d_list_[i_155][1],
                                  color=plt.cm.RdBu(color),length=0.6)
                    plt.show()
                    plt.close()
                    exit(0)
                    gen_data[i].append([p[0],p[1],p[2],v[0],v[1],v[2], score])
                tmp_arr = np.array(gen_data[i])
                min_val = min(tmp_arr[:,-1])
                max_val = max(tmp_arr[:,-1])
                tmp_arr = np.array(gen_data[i])
                tmp_arr[:,-1] = (tmp_arr[:,-1] - min_val) / (max_val - min_val)
                gen_data[i] = tmp_arr.tolist()
            else:
                for index in indices[i]:
                    p = pcs[index][:3]
                    d = pcs[index][3:6]
                    v = generate_vector(d)
                    cos_va = np.dot(v, axisdir)
                    sin_va = np.sqrt(1 - cos_va**2)
                    score = cos_va - coe_friction * sin_va
                    gen_data[i].append([p[0],p[1],p[2],v[0],v[1],v[2],score])
                tmp_arr = np.array(gen_data[i])
                min_val = min(tmp_arr[:,-1])
                max_val = max(tmp_arr[:,-1])
                tmp_arr[:,-1] = (tmp_arr[:,-1] - min_val) / (max_val - min_val)
                gen_data[i] = tmp_arr.tolist()
        else:
            for index in indices[i]:
                p = pcs[index][:3]
                v = generate_random_vector()
                gen_data[i].append([p[0],p[1],p[2],v[0],v[1],v[2],0])
                
    all_gen_data = sum(gen_data, [])
    for i in range(len(all_gen_data)):
        score = all_gen_data[i][-1]
        if math.isnan(score):
            all_gen_data[i][-1] = 0
    df = pd.DataFrame(all_gen_data)
    df.to_csv(save_path, index=False, header=False)


if __name__ == '__main__':
    # cate = sys.argv[-2]
    # file_name = sys.argv[-1]
    # work(cate, file_name)
    work()



