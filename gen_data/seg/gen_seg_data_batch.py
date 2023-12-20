import os
import sys
import subprocess
from functools import partial
from multiprocessing.dummy import Pool
import time 

data_dir = "./data_raw/pc_vscan_iter_front"
save_dir = "./data/seg2"

num_process = 48
CATE_LIST = ['dishwasher', 'microwave', 'oven', 'refrige', 'storage', 'table', 'trashcan']

gen_py_file_path = "./gen_data/action/gen_score.py"
python_command = "python"

if __name__ == '__main__':
    commands = []
    for cate in CATE_LIST:
        used_map = set()
        os.makedirs(f"{save_dir}/{cate}", exist_ok=True)
        for file_name in os.listdir(f"{data_dir}/{cate}"):
            if file_name[-5] == '9':
                continue
            shape_id, sid, pid, fid = file_name[:-4].split('_')
            used_key = f"{shape_id}{sid}"
            if used_key in used_map:
                continue
            used_map.add(used_key)
            commands.append([python_command, gen_py_file_path, cate, file_name])

    start_time = time.time()
    pool = Pool(num_process)
    for idx, completed in enumerate(pool.imap(partial(subprocess.run), commands)):
        pp = (idx + 1)/len(commands)
        end_time = time.time()
        used_time = end_time - start_time 
        need_time = used_time * (1-pp) / pp 
        print(f'Done>({idx + 1} / {len(commands)}) {100*pp:.2f}% [ {used_time:.1f}s | {need_time:.1f}s]')