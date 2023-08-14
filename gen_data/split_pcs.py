import os
import numpy as np

data_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_front"
save_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_front_split"

for cate in os.listdir(data_dir):
    os.makedirs(f"{save_dir}/{cate}", exist_ok=True)
    for file in os.listdir(f"{data_dir}/{cate}"):
        full_path = f"{data_dir}/{cate}/{file}"
        pcs_full = np.loadtxt(full_path).astype(np.float32)
        part_num = np.max(pcs_full[:,-1]) + 1
        part_pcs = [[] for i in range(int(part_num))]
        for i in range(pcs_full.shape[0]):
            part_id = int(pcs_full[i, -1])
            part_pcs[part_id].append(pcs_full[i, :3])
        for i in range(int(part_num)):
            if len(part_pcs[i]) > 100:
                np.savetxt(f"{save_dir}/{cate}/{file[:-4]}_{i:02}.pts", np.array(part_pcs[i]))