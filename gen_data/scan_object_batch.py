'''
Usage:
python scan_batch_shapenet.py <num_process> <dataset_dir> <output_dir> <save_rgbd_image> <save_pc_per_view> <save_pc_complete> <pc_per_view_size> <pc_complete_size>

Example:
Scan all the models in specific categories in ShapeNetCore directory:
python scan_batch_shapenet.py 8 F:/datasets/ShapeNetCore.v2 ./output 1 1 1 2048 16384
'''

import os
import sys
import subprocess
from functools import partial
from multiprocessing.dummy import Pool
import time

blender_command = "/opt/blender292/blender"
scan_file = "/mnt/disk3/zihao/dev/pointnet_tem/gen_data/scan_object.py"

num_process = 48
dataset_dir = "/mnt/disk3/zihao/20220610data/InterRecon/obj_iter"
out_dir = "/mnt/disk3/zihao/20220610data/InterRecon/pc_vscan_iter_front"
save_rgbd_image = True
save_pc_per_view = True
save_pc_complete = True
pc_per_view_size = 4096
pc_complete_size = 16384

category_list = ['dishwasher','microwave','oven','refrige','table','trashcan', 'storage']
# category_list = ['storage']

devnull = open(os.devnull, 'w')

def get_time_str(t):
    t = int(t)
    s = t%60
    m = int(t/60)%60
    h = int(t/3600)
    return f"{h}:{m}:{s}"

if __name__ == '__main__':
    commands = []
    # category_list.reverse()
    # print('=== Rendering %d models on %d workers ===' % (len(commands), num_process))
    cnt = 0
    for cate in category_list:
        # print(f"Work in {cate}")
        os.makedirs(f"{out_dir}/{cate}", exist_ok=True)
        for file_name in os.listdir(f"{dataset_dir}/{cate}"):
            object_tile = file_name.split('.')[0]
            save_dir = f"{out_dir}/{cate}/{object_tile}"
            obj_path = f"{dataset_dir}/{cate}/{file_name}"
            if os.path.exists(save_dir + ".pts"):
                print("exits", save_dir+'.pts')
                cnt += 1
                continue
            commands.append([blender_command, '-b', '-P', scan_file, obj_path,
                             save_dir, object_tile, '0', '0', '1', '4096', '16384'])
    commands.reverse()
            # commands.append(f"{blender_command} -b -P {scan_file} {obj_path} {save_dir} {object_tile} 0 0 1 4096 16384")
    pool = Pool(num_process)
    print('=== Exits :', cnt)
    print('=== Rendering %d models on %d workers ===' % (len(commands), num_process))
    start_time = time.time()
    # stdout = devnull, stderr = devnull
    for idx, completed in enumerate(pool.imap(partial(subprocess.run), commands)):
        pp = (idx + 1)/len(commands)
        end_time = time.time()
        used_time = end_time - start_time
        need_time = used_time * (1-pp) / pp
        print("Done: ", commands[idx])
        print(f'===> Done>({idx + 1} / {len(commands)}) {100*pp:.2f}% [{get_time_str(used_time)}|{get_time_str(need_time)}]  {(used_time/(idx+1))}/s')

