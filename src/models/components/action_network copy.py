import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from src.utils.pcs_utils import *


class Network(nn.Module):
    def __init__(
            self,
            pointNeXt_cfg: str,
            mlp_dim_critic: int = 64,
            mlp_dim_actor: int = 64,
            mlp_dim_score: int = 64,
            random_direction_num: int = 100,
    ):
        super(Network, self).__init__()

        self.rd_cnt = random_direction_num

        cfg = EasyConfig()
        cfg.load(pointNeXt_cfg, recursive=True)
        self.pointNeXt = build_model_from_cfg(cfg.model)
        feat_dim = cfg.model.encoder_args.width

        self.score_layer = nn.Sequential(
            nn.Linear(feat_dim + 3, mlp_dim_score),
            nn.ReLU(),
            nn.Linear(mlp_dim_score, 1),
            nn.Sigmoid()
        )

        self.actor_layer = nn.Sequential(
            nn.Linear(feat_dim + 3, mlp_dim_actor),
            nn.ReLU(),
            nn.Linear(mlp_dim_actor, 3)
        )

        self.critic_layer = nn.Sequential(
            nn.Linear(feat_dim + 6, mlp_dim_critic),
            nn.ReLU(),
            nn.Linear(mlp_dim_critic, 1),
            nn.Sigmoid()
        )

    def forward_score(self, pcs_feat, pcs):
        whole_queries = torch.cat([pcs_feat, pcs], dim=-1)
        pred_score = self.score_layer(whole_queries)
        return pred_score

    def forward_actor(self, pcs_feat, pcs):
        whole_queries = torch.cat([pcs_feat, pcs], dim=-1)  # B x N x (F+3)
        pred_actor = F.normalize(self.actor_layer(whole_queries), dim=-1)  # B x N x 3
        return pred_actor

    def forward_critic(self, pcs_feat, pcs_dir):
        whole_queries = torch.cat([pcs_feat, pcs_dir], dim=-1)
        pred_critic = self.critic_layer(whole_queries)
        return pred_critic

    def critic_best(self, pcs_feat, pcs):
        B, N, feat_dim = pcs_feat.shape  # B x N x F
        feat_pcs = torch.cat([pcs_feat, pcs], dim=-1)  # B x N x (F+3)
        feat_pcs = feat_pcs.repeat(1, self.rd_cnt, 1)  # B x (Nxrdn) x (F+3)

        rd = np.random.randn(B, N * self.rd_cnt, 3)
        rd /= np.linalg.norm(rd, axis=2, keepdims=True)
        rd = torch.from_numpy(rd).to(feat_pcs.device).float()

        whole_queries = torch.cat([feat_pcs, rd], dim=-1)

        pred_critic = self.critic_layer(whole_queries)  # B x (Nxrgn) x 1

        pred_critic = pred_critic.reshape(B, self.rd_cnt, N).transpose(2, 1)  # B x N x rgn
        rd = rd.reshape(B, self.rd_cnt, N, 3).transpose(2, 1)  # B x N x rgn x 3

        mean_score = torch.mean(pred_critic, dim=2)
        max_score, indices = torch.max(pred_critic, dim=2)
        best_d = rd[torch.arange(B)[:, None, None], torch.arange(N)[None, :, None], indices[:, :, None], :].squeeze(2)

        return mean_score, best_d

    def forward(self, pcs, pcs_dir):
        pcs_feat = self.pointNeXt(pcs)
        pcs_feat = pcs_feat.permute(0, 2, 1).contiguous()  # B x N x F

        max_score, best_d = self.critic_best(pcs_feat, pcs)

        # pred_critic = self.forward_critic(pcs_feat, pcs_dir)
        pred_actor = self.forward_actor(pcs_feat, pcs)
        pred_score = self.forward_score(pcs_feat, pcs)
        return (max_score, best_d), pred_actor, pred_score


class NetworkCritic(nn.Module):
    def __init__(
            self,
            pointNeXt_cfg: str,
            mlp_dim_critic: int = 64,
    ):
        super(NetworkCritic, self).__init__()

        cfg = EasyConfig()
        cfg.load(pointNeXt_cfg, recursive=True)
        self.pointNeXt = build_model_from_cfg(cfg.model)
        feat_dim = cfg.model.encoder_args.width

        self.critic_layer = nn.Sequential(
            nn.Linear(feat_dim + 6, mlp_dim_critic),
            nn.ReLU(),
            nn.Linear(mlp_dim_critic, 1),
            nn.Sigmoid()
        )

    def forward(self, pcs, pcs_dir):
        pcs_feat = self.pointNeXt(pcs)
        pcs_feat = pcs_feat.permute(0, 2, 1).contiguous()  # B x N x F
        whole_queries = torch.cat([pcs_feat, pcs_dir], dim=-1)
        pred_critic = self.critic_layer(whole_queries)

        return pred_critic


# from torchinfo import summary

# if __name__ == '__main__':
#     model = Network("/mnt/disk3/zihao/dev/pointnet_tem/model_cfg/pointNext.yaml")
#     model = model.cuda()


#     summary(model, input_size=[(8, 4096, 3), (8, 4096, 6)])
