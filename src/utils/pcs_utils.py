import torch
import matplotlib.pyplot as plt
import numpy as np


def visual(pcs, pcs_seg=None, joint_list=None, save_path=None):
    x = pcs[:, 0]
    y = pcs[:, 1]
    z = pcs[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.axis([-0.5,0.5,-0.5,0.5,-0.5,0.5])
    # origin = [0], [0], [0]
    # ax.quiver(*origin, [1], [0], [0], color='r', length=0.5, arrow_length_ratio=0.1)
    # ax.quiver(*origin, [0], [1], [0], color='g', length=0.5, arrow_length_ratio=0.1)
    # ax.quiver(*origin, [0], [0], [1], color='b', length=0.5, arrow_length_ratio=0.1)
    plt.axis('off')
    if pcs_seg is not None:
        ax.scatter(z,x,y,c=pcs_seg)
    else:
        ax.scatter(z,x,y)
    if joint_list is not None:
        for joint in joint_list:
            ax.quiver(joint[2],joint[0],joint[1],joint[2+3],joint[0+3],joint[1+3],color='r', length=0.6)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def generate_random_vector():
    v = np.random.rand(3)
    length = np.linalg.norm(v)
    unit_v = v / length
    return torch.from_numpy(unit_v).float()
