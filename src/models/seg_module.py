from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from openpoints.models.layers import furthest_point_sample
import os
import numpy as np
from src.utils.pcs_utils import *

from torchmetrics import JaccardIndex
from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.classification import MulticlassJaccardIndex

def fps_sample(xyz, seg, num_points):
    fps_idx = furthest_point_sample(xyz, num_points)
    fps_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, 3))
    fps_seg = torch.gather(seg, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, 1))
    return fps_xyz, fps_seg

def save_seg(save_path, xyz, seg):
    np.savetxt(save_path, np.concatenate([xyz, seg], axis=1))

class SegLitModule(LightningModule):

    def __init__(
            self,
            net,
            num_points: int,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            save_dir: str,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.save_dir = save_dir
        self.num_points = num_points
        self.net = net

        # loss function
        self.criterion_start = torch.nn.MSELoss()
        self.criterion_end = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_start_seg_acc = Accuracy(task="multiclass", num_classes=2)
        self.train_end_seg_acc = Accuracy(task="multiclass", num_classes=3)

        self.val_start_seg_acc = Accuracy(task="multiclass", num_classes=2)
        self.val_end_seg_acc = Accuracy(task="multiclass", num_classes=3)

        self.test_start_seg_acc = Accuracy(task="multiclass", num_classes=2)
        self.test_end_seg_acc = Accuracy(task="multiclass", num_classes=3)

        self.test_start_seg_jaccard1 = JaccardIndex(num_classes=2, task="multiclass", average="macro")
        self.test_end_seg_jaccard1 = JaccardIndex(num_classes=3, task="multiclass", average="macro")

        # self.test_start_seg_jaccard_static = JaccardIndex(num_classes=2, task="binary")
        # self.test_start_seg_jaccard2 = JaccardIndex(num_classes=2, task="multiclass")
        # self.test_end_seg_jaccard2 = JaccardIndex(num_classes=3, task="multiclass")
        
        # self.test_start_seg_jaccard3 = JaccardIndex(num_classes=2, task="binary")
        # self.test_end_seg_jaccard3 = JaccardIndex(num_classes=3, task="multiclass")
        
        
        # self.test_start_ap1 = Precision(num_classes=2, task="multiclass")
        # self.test_end_ap1 = Precision(num_classes=3, task="multiclass")
        
        # self.test_start_ap2 = Precision(num_classes=2, task="multiclass", average="macro")
        # self.test_end_ap2 = Precision(num_classes=3, task="multiclass", average="macro")
        
        # for averaging loss across batches
        self.train_seg_loss = MeanMetric()
        self.val_seg_loss = MeanMetric()
        self.test_seg_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_start_seg_acc_best = MaxMetric()
        self.val_end_seg_acc_best = MaxMetric()

        self.vis_epoch = 0

    def forward(self, pcs_start: torch.Tensor, pcs_end: torch.Tensor, joint_info: torch.Tensor):
        return self.net(pcs_start, pcs_end, joint_info)

    def forward(self, data: torch.Tensor, joint_info: torch.Tensor):
        return self.net(data, joint_info)
    
    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_start_seg_acc_best.reset()
        self.val_end_seg_acc_best.reset()

    def model_step(self, batch: Any, visualize: bool = False, output=False):
        # gt_start_xyz, gt_end_xyz, gt_start_seg, gt_end_seg, pid_list, data_path_list, joint_info = batch
        data, gt_start_xyz, gt_end_xyz, gt_start_seg, gt_end_seg, pid_list, data_path_list, joint_info = batch
        # pred_logit_start_seg, pred_logit_end_seg = self.forward(gt_start_xyz, gt_end_xyz, joint_info)
        pred_logit_start_seg, pred_logit_end_seg = self.forward(data, joint_info)
        
        # New loss
        viewed_pred_logit_start_seg = pred_logit_start_seg.view(-1, 2).contiguous()
        viewed_gt_start_seg = gt_start_seg.view(-1).contiguous()
        viewed_pred_logit_end_seg = pred_logit_end_seg.view(-1, 3).contiguous()
        viewed_gt_end_seg = gt_end_seg.view(-1).contiguous()

        loss_start = F.nll_loss(viewed_pred_logit_start_seg, viewed_gt_start_seg.long())
        loss_end = F.nll_loss(viewed_pred_logit_end_seg, viewed_gt_end_seg.long())
        
        # loss_start = self.criterion_start(pred_seg_start, gt_start_seg.float())
        # loss_end = self.criterion_end(pred_seg_end, gt_end_seg.float())

        loss = 0.8*loss_start +  0.2*loss_end

        pred_start_seg = torch.argmax(viewed_pred_logit_start_seg, dim=1)
        pred_end_seg = torch.argmax(viewed_pred_logit_end_seg, dim=1)
        
        if visualize or output:
            with torch.no_grad():
                B = gt_start_xyz.shape[0]
                pred_seg_start = pred_start_seg.reshape(B, -1, 1).detach()
                pred_seg_end = pred_end_seg.reshape(B, -1, 1).detach()
                gt_seg_start = gt_start_seg.reshape(B, -1, 1)
                gt_seg_end = gt_end_seg.reshape(B, -1, 1)
                
                if visualize:
                    gt_start_ = torch.cat([gt_start_xyz, gt_seg_start], dim=2).cpu().numpy()
                    gt_end_ = torch.cat([gt_end_xyz, gt_seg_end], dim=2).cpu().numpy()
                    pred_start = torch.cat([gt_start_xyz, pred_seg_start], dim=2).cpu().numpy()
                    pred_end = torch.cat([gt_end_xyz, pred_seg_end], dim=2).cpu().numpy()
                    for i in range(B):
                        cate, file_name = data_path_list[i].split('/')
                        shape_id, sid, pid = file_name[:-4].split('_')
                        save_diri = f"{self.save_dir}/visual/{self.vis_epoch}/{cate}/{shape_id}_{sid}_{pid}"
                        os.makedirs(save_diri, exist_ok=True)
                        np.savetxt(f"{save_diri}/gt_start.pts", gt_start_[i], fmt="%.6f")
                        np.savetxt(f"{save_diri}/gt_end.pts", gt_end_[i], fmt="%.6f")
                        np.savetxt(f"{save_diri}/pred_start.pts", pred_start[i], fmt="%.6f")
                        np.savetxt(f"{save_diri}/pred_end.pts", pred_end[i], fmt="%.6f")
                        
                if output:
                    pred_seg_start_np = pred_seg_start.cpu().numpy()
                    pred_seg_end_np = pred_seg_end.cpu().numpy()
                    gt_start_xyz_np = gt_start_xyz.cpu().numpy()
                    gt_end_xyz_np = gt_end_xyz.cpu().numpy()
                    for i in range(B):
                        cate, file_name = data_path_list[i].split('/')
                        shape_id, sid, pid = file_name[:-4].split('_')
                        save_diri = f"{self.save_dir}/output/{self.vis_epoch}/{cate}/{shape_id}_{sid}"
                        os.makedirs(save_diri, exist_ok=True)

                        gt_start_xyzi = gt_start_xyz_np[i]
                        gt_end_xyzi = gt_end_xyz_np[i]
                        pred_seg_starti = pred_seg_start_np[i]
                        pred_seg_endi = pred_seg_end_np[i]
                        
                        pred_move = gt_end_xyzi[pred_seg_endi[:,0]==1]
                        np.savetxt(f"{save_diri}/{pid}.pts", pred_move, fmt="%.6f")
                        pred_static_more = gt_end_xyzi[pred_seg_endi[:,0]==2]
                        np.savetxt(f"{save_diri}/{pid}_more.pts", pred_static_more, fmt="%.6f")
                        
                        if pred_static_more.shape[0] > 2:
                            if os.path.exists(f"{save_diri}/more.pts"):
                                pcs_more = np.loadtxt(f"{save_diri}/more.pts").astype(np.float32)
                                np.savetxt(f"{save_diri}/more.pts", np.concatenate([pcs_more, pred_static_more],axis=0), fmt="%.6f")
                            else:
                                np.savetxt(f"{save_diri}/more.pts", pred_static_more, fmt="%.6f")
                            
                        if os.path.exists(f"{save_diri}/static.pts"):
                            pcs_static = np.loadtxt(f"{save_diri}/static.pts").astype(np.float32)
                            pcs_static[pred_seg_starti[:,0]==1, -1] = 1
                            np.savetxt(f"{save_diri}/static.pts", pcs_static, fmt="%.6f")
                            np.savetxt(f"{save_diri}/only_static.pts", pcs_static[pcs_static[:,-1]==0][:,:3], fmt="%.6f")
                        else:
                            pcs_static = np.concatenate([gt_start_xyzi, pred_seg_starti],axis=1)
                            np.savetxt(f"{save_diri}/static.pts", pcs_static, fmt="%.6f")
                            np.savetxt(f"{save_diri}/only_static.pts", pcs_static[pcs_static[:,-1]==0][:,:3], fmt="%.6f")
                            
        return loss, pred_start_seg, pred_end_seg, viewed_gt_start_seg, viewed_gt_end_seg

    def training_step(self, batch: Any, batch_idx: int):
        loss,  pred_seg_start, pred_seg_end, gt_start_seg, gt_end_seg = self.model_step(batch)

        # update and log metrics
        self.train_seg_loss(loss)

        self.train_start_seg_acc(pred_seg_start, gt_start_seg)
        self.train_end_seg_acc(pred_seg_end, gt_end_seg)

        self.log("train/seg_loss", self.train_seg_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/start_seg_acc", self.train_start_seg_acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/end_seg_acc", self.train_end_seg_acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, pred_seg_start, \
            pred_seg_end, gt_start_seg, gt_end_seg = self.model_step(batch, visualize=True)

        # update and log metrics
        self.val_seg_loss(loss)

        self.val_start_seg_acc(pred_seg_start, gt_start_seg)
        self.val_end_seg_acc(pred_seg_end, gt_end_seg)

        self.log("val/seg_loss", self.val_seg_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/start_seg_acc", self.val_start_seg_acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/end_seg_acc", self.val_end_seg_acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.vis_epoch += 1
        # torch.save(self.net.state_dict(), f'{self.save_dir}/{self.vis_epoch}_model.pth')

    def test_step(self, batch: Any, batch_idx: int):
        loss,  pred_seg_start, pred_seg_end, \
            gt_start_seg, gt_end_seg = self.model_step(batch)
 
        # update and log metrics
        self.train_seg_loss(loss)

        self.test_start_seg_acc(pred_seg_start, gt_start_seg)
        self.test_end_seg_acc(pred_seg_end, gt_end_seg)
        
        self.test_start_seg_jaccard1(pred_seg_start, gt_start_seg)
        self.test_end_seg_jaccard1(pred_seg_end, gt_end_seg)

        # pred_seg_start = torch.ones_like(pred_seg_start) - pred_seg_start
        # gt_start_seg = torch.ones_like(gt_start_seg) - gt_start_seg 
        
        # self.test_start_seg_jaccard_static(pred_seg_start, gt_start_seg)
        
        # self.test_start_seg_jaccard2(pred_seg_start, gt_start_seg)
        # self.test_end_seg_jaccard2(pred_seg_end, gt_end_seg)
        
        # self.test_start_seg_jaccard3(pred_seg_start, gt_start_seg)
        # self.test_end_seg_jaccard3(pred_seg_end, gt_end_seg)
        
        # self.test_start_ap1(pred_seg_start, gt_start_seg)
        # self.test_end_ap1(pred_seg_end, gt_end_seg)
        
        # self.test_start_ap2(pred_seg_start, gt_start_seg)
        # self.test_end_ap2(pred_seg_end, gt_end_seg)
        
        # self.log("test/seg_loss", self.test_seg_loss, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/start_seg_acc", self.test_start_seg_acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/end_seg_acc", self.test_end_seg_acc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        # self.log("test/iou_start_static", self.test_start_seg_jaccard1, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/iou_start", self.test_start_seg_jaccard1, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/iou_end", self.test_end_seg_jaccard1, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        
        # self.log("test/start_iou2", self.test_start_seg_jaccard2, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("test/end_iou2", self.test_end_seg_jaccard2, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        
        # self.log("test/start_iou3", self.test_start_seg_jaccard3, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("test/end_iou3", self.test_end_seg_jaccard3, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        
        # self.log("test/start_ap1", self.test_start_ap1, on_step=True,
        #             on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("test/end_ap1", self.test_end_ap1, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        
        # self.log("test/start_ap2", self.test_start_ap2, on_step=True,
        #             on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("test/end_ap2", self.test_end_ap2, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("test/start_ap2", self.test_start_ap2, on_step=True,
        #             on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log("test/end_ap2", self.test_end_ap2, on_step=True,
        #          on_epoch=True, prog_bar=True, sync_dist=True)     
        # return loss or backpropagation will fail
        return loss

    def on_test_epoch_end(self):
        self.vis_epoch += 1

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/end_seg_acc",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


