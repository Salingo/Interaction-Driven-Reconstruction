from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import os
import numpy as np
from src.utils.seed_former_loss_utils import fps_subsample, get_loss

def save_pts(save_path, pcs_partial, pcs_complete):
    # print(pcs_partial.shape)
    # print(pcs_complete.shape)
    pcs_partial_np = pcs_partial.cpu().numpy()
    pcs_partial_np0 = np.concatenate([pcs_partial_np, np.zeros((pcs_partial_np.shape[0], 1))], axis=1)
    pcs_complete_np = pcs_complete.cpu().numpy()
    pcs_complete_np1 = np.concatenate([pcs_complete_np, np.ones((pcs_complete_np.shape[0], 1))], axis=1)
    pcs_full = np.concatenate([pcs_partial_np0, pcs_complete_np1], axis=0)
    np.savetxt(save_path, pcs_full)

class ComLitModule(LightningModule):

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

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_cd_pc = MeanMetric()
        self.val_cd_pc = MeanMetric()
        self.test_cd_pc = MeanMetric()

        self.train_cd_p1 = MeanMetric()
        self.val_cd_p1 = MeanMetric()
        self.test_cd_p1 = MeanMetric()

        self.train_cd_p1 = MeanMetric()
        self.val_cd_p1 = MeanMetric()
        self.test_cd_p1 = MeanMetric()

        self.train_cd_p2 = MeanMetric()
        self.val_cd_p2 = MeanMetric()
        self.test_cd_p2 = MeanMetric()

        self.train_cd_p3 = MeanMetric()
        self.val_cd_p3 = MeanMetric()
        self.test_cd_p3 = MeanMetric()

        self.train_partial = MeanMetric()
        self.val_partial = MeanMetric()
        self.test_partial = MeanMetric()

        self.vis_epoch = 0

    def forward(self, pcs_partial: torch.Tensor, joint: torch.Tensor):
        return self.net(pcs_partial, joint)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def model_step(self, batch: Any, visualize: bool = False):
        pcs_partial, pcs_gt, data_path_list, joint_list = batch

        # pcs_partial = fps_subsample(pcs_partial, self.num_points)
        # pcs_gt = fps_subsample(pcs_gt, self.num_points)

        
        pcs_pred_list = self.forward(pcs_partial, joint_list)

        loss_total, losses, gts = get_loss(pcs_pred_list, pcs_partial, pcs_gt)

        if visualize:
            for i in range(pcs_partial.shape[0]):
                cate, file_name = data_path_list[i].split('/')
                save_diri = f"{self.save_dir}/visualize/epoch_{self.vis_epoch}/{cate}/{file_name[:-4]}"
                os.makedirs(save_diri, exist_ok=True)
                save_pts(f"{save_diri}/gt.pts", pcs_partial[i], pcs_gt[i])
                for j in range(len(pcs_pred_list)):
                    save_pts(f"{save_diri}/pred_{j}.pts", pcs_partial[i], pcs_pred_list[j][i])

        return loss_total, losses, gts

    def training_step(self, batch: Any, batch_idx: int):

        loss_total, losses, gts = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss_total)
        self.train_cd_pc(losses[0] * 1e3)
        self.train_cd_p1(losses[1] * 1e3)
        self.train_cd_p2(losses[2] * 1e3)
        self.train_cd_p3(losses[3] * 1e3)
        self.train_partial(losses[4] * 1e3)

        self.log("train/loss", self.train_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cd_pc", self.train_cd_pc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cd_p1", self.train_cd_p1, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cd_p2", self.train_cd_p2, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cd_p3", self.train_cd_p3, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cd_partial", self.train_partial, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss_total

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss_total, losses, gts = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss_total)
        self.val_cd_pc(losses[0] * 1e3)
        self.val_cd_p1(losses[1] * 1e3)
        self.val_cd_p2(losses[2] * 1e3)
        self.val_cd_p3(losses[3] * 1e3)
        self.val_partial(losses[4] * 1e3)

        self.log("val/loss", self.val_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cd_pc", self.val_cd_pc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cd_p1", self.val_cd_p1, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cd_p2", self.val_cd_p2, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cd_p3", self.val_cd_p3, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cd_partial", self.val_partial, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss_total

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.vis_epoch += 1

    def test_step(self, batch: Any, batch_idx: int):
        loss_total, losses, gts = self.model_step(batch, visualize=True)

        # update and log metrics
        self.test_loss(loss_total)
        self.test_cd_pc(losses[0] * 1e3)
        self.test_cd_p1(losses[1] * 1e3)
        self.test_cd_p2(losses[2] * 1e3)
        self.test_cd_p3(losses[3] * 1e3)
        self.test_partial(losses[4] * 1e3)

        torch.save(self.net.state_dict(), os.path.join(self.save_dir, 'model.pth'))
          
        self.log("test/loss", self.test_loss, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cd_pc", self.test_cd_pc, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cd_p1", self.test_cd_p1, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cd_p2", self.test_cd_p2, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cd_p3", self.test_cd_p3, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test/cd_partial", self.test_partial, on_step=True,
                 on_epoch=True, prog_bar=True, sync_dist=True)

        return loss_total

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
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


