import os
import pyrootutils
import numpy as np 

from src.utils.pcs_utils import visual

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def vis_data(path):
    pcs = np.loadtxt(path, delimiter=',')
    pcs[:3,:] -= 0.5
    visual(pcs[:5000,:3], pcs[:5000,-1:],pcs[2000:2100,:6])
    

# vis_data("oven", "7120_s01_p00_f00.pts")
vis_data("/mnt/disk3/zihao/dev/lightning-hydra-template-2.0.2/data/action/trashcan/10584_s07_p01_f00.pts")