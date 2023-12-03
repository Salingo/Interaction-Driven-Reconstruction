import os
import sys
import subprocess
from functools import partial
from multiprocessing.dummy import Pool
import time 
from tqdm import tqdm

SAVE_DIR = "../../data/action"
PC_DATA_DIR = "../../data_raw/pc_vscan_iter_front"

python_command = "python"
gen_py_file_path = "./gen_score.py"

category_list = ['dishwasher','microwave','oven','refrige','table','trashcan', 'storage']
# category_list = ['trashcan']
num_process = 48

if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)
    commands = []
    had_obj = []
    for cate in category_list:
        os.makedirs(SAVE_DIR + "/" + cate, exist_ok = True)
        for folder_name in tqdm(os.listdir(f"{PC_DATA_DIR}/{cate}")):
            shape_id, sid, pid, fid = folder_name[:-4].split('_')
            info_key = f"{shape_id}{sid}"
            if fid[-1] == '0' and info_key not in had_obj:
                had_obj.append(info_key)
                commands.append([python_command, gen_py_file_path, cate, folder_name])

    start_time = time.time()
    pool = Pool(num_process)
    for idx, completed in enumerate(pool.imap(partial(subprocess.run), commands)):
        pp = (idx + 1)/len(commands)
        end_time = time.time()
        used_time = end_time - start_time 
        need_time = used_time * (1-pp) / pp 
        print(f'Done>({idx + 1} / {len(commands)}) {100*pp:.2f}% [ {used_time:.1f}s | {need_time:.1f}s]')