from typing import Any, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.trainer.trainer import Trainer
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from openpoints.models.layers import furthest_point_sample
from datetime import datetime
import os
import numpy as np
from src.utils.pcs_utils import *
import json
from src.models.components.action_network import *

FM = 1E3

class ActionLitModule(LightningModule):

    def __init__(
        self,
        net : Network,
        net_critic: NetworkCritic,
        train_critic: bool,
        num_points: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        save_dir: str,
        critic_weight_path: str,
        random_vector_num: int = 100,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.save_dir = save_dir
        self.num_points = num_points
        self.train_critic = train_critic
        self.critic_weight_path = critic_weight_path
        self.random_vector_num = random_vector_num

        self.net = net
        self.net_critic = net_critic
            
        self.critic_criterion = nn.MSELoss()
        self.actor_criterion = nn.CosineEmbeddingLoss()
        self.score_criterion = nn.MSELoss()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # loss of Action Scoring Module
        self.train_loss_critic = MeanMetric()
        self.val_loss_critic = MeanMetric()
        self.test_loss_critic = MeanMetric()
        
        if not train_critic:
            self.train_loss_actor = MeanMetric()
            self.val_loss_actor = MeanMetric()
            self.test_loss_actor = MeanMetric()

            self.train_loss_score = MeanMetric()
            self.val_loss_score = MeanMetric()
            self.test_loss_score = MeanMetric()

        # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()
        self.vis_epoch = 1
        self.save_epoch = 1
    
    def forward(self, pcs: torch.Tensor, pcs_dir: torch.Tensor):
        pass

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.val_loss.reset()
        self.val_loss_critic.reset()
        
        if not self.train_critic:
            self.val_loss_actor.reset()
            self.val_loss_score.reset()
            
            assert(self.critic_weight_path!="")
            self.load_state_dict(torch.load(self.critic_weight_path)['state_dict'], strict=False)
            self.net_critic.requires_grad_(False)

    def visualize_critic(self, pcs, pcs_dir, file_path_list, random_sample_num = 10):
        # mean_score = torch.zeros(pcs.shape[0], pcs.shape[1], 1).cuda()
        # for _ in range(random_sample_num):
        #     pcs_dir0 = pcs_dir.clone()
        #     pcs_dir0[...,3:] = generate_random_vector()
        #     logits0 = self.net_critic(pcs, pcs_dir0)
        #     mean_score += logits0
        # mean_score /= random_sample_num
        
        mean_score, critic_d = self.net_critic.critic_best(pcs, pcs_dir)
        
        # mean_score.unsqueeze_(2)
        pcs_n = pcs.detach().cpu().numpy()
        critic_dn = critic_d.cpu().numpy()
        mean_score_n = mean_score.cpu().numpy()
        
        # print(mean_score.shape)
        for i in range(pcs.shape[0]):
            
            file_path = file_path_list[i]
            cate, file_name = file_path.split('/')[-2:]
            
            pcsi = pcs_n[i]
            scorei = mean_score_n[i]
            critic_di = critic_dn[i]
            critic_actioni = np.concatenate([pcsi,critic_di],axis=1)
            
            # print(pcsi.shape, scorei.shape)
            
            save_diri = f"{self.save_dir}/visualize/epoch_{self.vis_epoch}/{cate}/{file_name}"
            os.makedirs(save_diri, exist_ok=True)
            visual(pcsi, scorei, joint_list= critic_actioni[:10], save_path=save_diri + "/pcs_score.png")
            
            pcs_scorei = np.concatenate([pcsi, scorei], axis=1)
            np.savetxt(save_diri + f"/pcs_score.pts", pcs_scorei)
    
    def visualize_actor(self, pcs, pcs_dir, file_path_list):
        B, N, _ = pcs.shape
        pred_actor, pred_score = self.net(pcs, pcs_dir)

        mean_score, critic_d = self.net_critic.critic_best(pcs, pcs_dir)

        max_indices = torch.argmax(pred_score, dim=1)
        best_p = torch.gather(pcs, dim=1, index=max_indices.unsqueeze(-1).expand(-1, -1, 3)).squeeze(dim=1)
        best_d = torch.gather(pred_actor, dim=1, index=max_indices.unsqueeze(-1).expand(-1, -1, 3)).squeeze(dim=1)

        best_action = torch.cat([best_p, best_d], dim=1).cpu().numpy()

        pcs_n = pcs.detach().cpu().numpy()
        score_n = pred_score.cpu().numpy()
        mean_score_n = mean_score.cpu().numpy()
        critic_dn = critic_d.cpu().numpy()

        for i in range(B):
            file_path = file_path_list[i]
            cate, file_name = file_path.split('/')[-2:]
            file_name = file_name[:-4]
            save_diri_critic = f"{self.save_dir}/visualize/epoch_{self.vis_epoch}/critic/{cate}"
            save_diri_action = f"{self.save_dir}/visualize/epoch_{self.vis_epoch}/action/{cate}"
            save_diri_score = f"{self.save_dir}/visualize/epoch_{self.vis_epoch}/score/{cate}"
            save_diri_pts = f"{self.save_dir}/visualize/epoch_{self.vis_epoch}/pts/{cate}"

            os.makedirs(save_diri_critic, exist_ok=True)
            os.makedirs(save_diri_action, exist_ok=True)
            os.makedirs(save_diri_score, exist_ok=True)
            os.makedirs(save_diri_pts, exist_ok=True)

            pcsi = pcs_n[i]
            scorei = score_n[i]
            mean_scorei = mean_score_n[i]
            critic_di = critic_dn[i]
            critic_actioni = np.concatenate([pcsi,critic_di],axis=1)
            visual(pcsi, mean_scorei, joint_list= critic_actioni[:10], save_path=f"{save_diri_critic}/{file_name}.png")
            visual(pcsi, scorei, joint_list= [best_action[i]],save_path= f"{save_diri_action}/{file_name}.png")
            visual(pcsi, scorei, save_path= f"{save_diri_score}/{file_name}.png")
            np.savetxt(f"{save_diri_pts}/{file_name}.pts", np.concatenate([pcsi, scorei], axis=1))

    def model_step(self, batch: Any, visualize: bool = False):
        point_clouds, file_path_list, ref_id = batch
        
        fps_idx = furthest_point_sample(point_clouds[:, :, :3].contiguous(), self.num_points)
        fps_pcs = torch.gather(point_clouds, 1, fps_idx.unsqueeze(-1).long().expand(-1, -1, point_clouds.shape[-1]))
        
        pcs = fps_pcs[:, :, :3].contiguous()
        pcs_dir = fps_pcs[:, :, :6].contiguous()
        score = fps_pcs[:, :, 6].unsqueeze(2).contiguous()

        if visualize:
            with torch.no_grad():
                if self.train_critic:
                    self.visualize_critic(pcs, pcs_dir, file_path_list)
                else:
                    self.visualize_actor(pcs, pcs_dir,file_path_list)

        if self.train_critic:
            pred_critic = self.net_critic(pcs, pcs_dir)
            # * ADD something here
            # loss_critic = self.critic_criterion(pred_critic, score)
            loss_critic = (pred_critic-score)**2
            loss_wight = torch.ones_like(loss_critic)
            mask = fps_pcs[:, :, -1] == ref_id.unsqueeze(1).unsqueeze(2).repeat(1, self.num_points, 1)[:, :, 0]
            loss_wight[mask] = 0.18
            loss_critic *= loss_wight
            loss_critic = torch.mean(loss_critic) * FM
            loss = loss_critic
            return loss, loss_critic, None, None
        else:
            pred_actor, pred_score = self.net(pcs, pcs_dir)
            mean_score, best_d = self.net_critic.critic_best(pcs, pcs_dir)
            loss_actor = torch.mean(1 - F.cosine_similarity(pred_actor, best_d, dim=2)) * FM
            loss_score = self.score_criterion(pred_score, mean_score) * FM
            loss = loss_actor + loss_score

            return loss, None, loss_actor, loss_score

    def training_step(self, batch: Any, batch_idx: int):
        
        loss, loss_critic, loss_actor, loss_score = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.train_critic:
            self.train_loss_critic(loss_critic)
            self.log("train/loss_critic", self.train_loss_critic, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.train_loss_actor(loss_actor)
            self.log("train/loss_actor", self.train_loss_actor, on_step=True, on_epoch=True, prog_bar=True)
            self.train_loss_score(loss_score)
            self.log("train/loss_score", self.train_loss_score, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass
        # os.makedirs(self.save_dir+"/pths", exist_ok=True)
        # torch.save(self.net.critic_layer.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_critic.pth")
        # torch.save(self.net.pointNeXt.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_pointNeXt.pth")
        # if not self.train_critic:
        #     torch.save(self.net.actor_layer.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_actor.pth")
        #     torch.save(self.net.score_layer.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_score.pth")
        # self.save_epoch += 1

    def validation_step(self, batch: Any, batch_idx: int):
        loss, loss_critic, loss_actor, loss_score = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.train_critic:
            self.val_loss_critic(loss_critic)
            self.log("val/loss_critic", self.val_loss_critic, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.val_loss_actor(loss_actor)
            self.log("val/loss_actor", self.val_loss_actor, on_step=True, on_epoch=True, prog_bar=True)
            self.val_loss_score(loss_score)
            self.log("val/loss_score", self.val_loss_score, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_end(self):
        # acc = self.val_acc.compute()  # get current val acc
        # self.val_acc_best(acc)  # update best so far val acc
        # # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # # otherwise metric would be reset by lightning after each epoch
        # self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.vis_epoch += 1

    def test_step(self, batch: Any, batch_idx: int):
        loss, loss_critic, loss_actor, loss_score = self.model_step(batch, visualize=True)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.train_critic:
            self.test_loss_critic(loss_critic)
            self.log("test/loss_critic", self.test_loss_critic, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.test_loss_actor(loss_actor)
            self.log("test/loss_actor", self.test_loss_actor, on_step=True, on_epoch=True, prog_bar=True)
            self.test_loss_score(loss_score)
            self.log("test/loss_score", self.test_loss_score, on_step=True, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_test_epoch_end(self):
        self.vis_epoch += 1
        # os.makedirs(self.save_dir+"/pths", exist_ok=True)
        # torch.save(self.net.critic_layer.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_critic.pth")
        # torch.save(self.net.pointNeXt.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_pointNeXt.pth")
        # if not self.train_critic:
        #     torch.save(self.net.actor_layer.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_actor.pth")
        #     torch.save(self.net.score_layer.state_dict(), f"{self.save_dir}/pths/{self.save_epoch}_score.pth")
        # self.save_epoch += 1

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
                    "monitor": "train/loss_epoch",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


