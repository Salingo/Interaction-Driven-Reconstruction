import torch
# from openpoints.cpp.chamfer_dist import ChamferFunction
from openpoints.models.layers import furthest_point_sample
from src.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
import torch.nn as nn
from src.emd.emd_module import emdModule
from openpoints.cpp.emd import emd

def fps_subsample(xyz, num_points):
    fps_idx = furthest_point_sample(xyz, num_points)
    fps_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, 3))
    return fps_xyz

chamfer_dist = chamfer_3DDist()


def chamfer(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    return torch.mean(d1) + torch.mean(d2)

def chamfer_sqrt(p1, p2):
    d1, d2, _, _ = chamfer_dist(p1, p2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def chamfer_single_side(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def chamfer_single_side_sqrt(pcd1, pcd2):
    d1, d2, _, _ = chamfer_dist(pcd1, pcd2)
    d1 = torch.clamp(d1, min=1e-9)
    d2 = torch.clamp(d2, min=1e-9)
    d1 = torch.mean(torch.sqrt(d1))
    return d1

emdLoss = emd()

def get_loss(pcds_pred, partial, gt, sqrt=True):
    """loss function
    Args
        pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
    """
    if sqrt:
        CD = chamfer_sqrt
        PM = chamfer_single_side_sqrt
    else:
        CD = chamfer
        PM = chamfer_single_side

    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = CD(Pc, gt_c)
    cd1 = CD(P1, gt_1)
    cd2 = CD(P2, gt_2)
    cd3 = CD(P3, gt)

    partial_matching = PM(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3 + partial_matching) * 1e3
    losses = [cdc, cd1, cd2, cd3, partial_matching]
    return loss_all, losses, [gt_2, gt_1, gt_c]


def get_loss2(pcds_pred, partial, gt, sqrt=True):
    Pc, P1, P2, P3 = pcds_pred

    gt_2 = fps_subsample(gt, P2.shape[1])
    gt_1 = fps_subsample(gt_2, P1.shape[1])
    gt_c = fps_subsample(gt_1, Pc.shape[1])

    cdc = emdLoss(Pc, gt_c)
    cd1 = emdLoss(P1, gt_1)
    cd2 = emdLoss(P2, gt_2)
    cd3 = emdLoss(P3, gt)

    # partial_matching = emdLoss(partial, P3)

    loss_all = (cdc + cd1 + cd2 + cd3) * 1e3
    losses = [cdc, cd1, cd2, cd3, cd3]
    return loss_all, losses, [gt_2, gt_1, gt_c]