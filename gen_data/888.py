import os 
import shutil

data_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_front"

save_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_seg"

# for cate in os.listdir(data_dir):
    # had_map = set()
    # for filename in os.listdir(f"{data_dir}/{cate}"):
    #     if filename[-5] =='0':
    #         shape_id, sid, pid, fid = filename[:-4].split("_")
    #         had_key = f"{shape_id}_{sid}"
    #         if had_key in had_map:
    #             continue
    #         had_map.add(had_key)
    #         shutil.copy2(f"{data_dir}/{cate}/{filename}", f"{save_dir}/{cate}/{shape_id}_{sid}.pts")
    #         print(cate, filename)
    
for cate in os.listdir(save_dir):
    for filename in os.listdir(f"{save_dir}/{cate}"):
        print(cate, filename)
        if 'p' not in filename[:-4]:
            os.remove(f"{save_dir}/{cate}/{filename}")