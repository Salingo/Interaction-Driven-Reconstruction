import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from openpoints.models import build_model_from_cfg
from openpoints.utils import EasyConfig
from src.utils.seed_former_utils import MLP_Res

class SegNet(nn.Module):
    def __init__(self, pointNeXt_cfg, num_points=4096, mlp_dim=256, use_joint=True):
        super().__init__()
        self.num_points = num_points
        self.use_joint = use_joint
        
        cfg = EasyConfig()
        cfg.load(pointNeXt_cfg, recursive=True)
        self.pointNeXt = build_model_from_cfg(cfg.model)
        feat_dim = cfg.model.encoder_args.width

        if use_joint:
            self.mlp_start = MLP_Res(feat_dim+7, mlp_dim, out_dim=2)
            self.mlp_end = MLP_Res(feat_dim+7, mlp_dim, out_dim=3)
        else:
            self.mlp_start = MLP_Res(feat_dim, mlp_dim, out_dim=2)
            self.mlp_end = MLP_Res(feat_dim, mlp_dim, out_dim=3)
            
    def forward(self, pcs_start, pcs_end, joint_info = None):
        B, N, C = pcs_start.shape
        pcs = torch.cat([pcs_start, pcs_end], dim=1)  # B, 2*N, C
        feat = self.pointNeXt(pcs)
        if self.use_joint:
            joint_info = joint_info.transpose(2, 1).contiguous()
            joint_info = joint_info.repeat(1, 1, 2*N)
            #print(joint_info.shape)
            #print(feat.shape)
            feat = torch.cat([feat, joint_info], dim=1) # B, C+7, 2*N
        feat_start, feat_end = torch.split(feat, [self.num_points, self.num_points], dim=2)
        seg_start = F.log_softmax(self.mlp_start(feat_start).transpose(2, 1).contiguous(), dim=-1)
        seg_end = F.log_softmax(self.mlp_end(feat_end).transpose(2, 1).contiguous(), dim=-1)
        return seg_start, seg_end
    
    def forward(self, data, joint_info = None):
        pcs = data['pos']
        B, N, C = pcs.shape
        feat = self.pointNeXt(data)
        if self.use_joint:
            joint_info = joint_info.transpose(2, 1).contiguous()
            joint_info = joint_info.repeat(1, 1, 2*N)
            #print(joint_info.shape)
            #print(feat.shape)
            feat = torch.cat([feat, joint_info], dim=1) # B, C+7, 2*N
        feat_start, feat_end = torch.split(feat, [self.num_points, self.num_points], dim=2)
        seg_start = F.log_softmax(self.mlp_start(feat_start).transpose(2, 1).contiguous(), dim=-1)
        seg_end = F.log_softmax(self.mlp_end(feat_end).transpose(2, 1).contiguous(), dim=-1)
        return seg_start, seg_end
    
    
class SegNetWoMotion(nn.Module):
    def __init__(self, pointNeXt_cfg, num_points=4096, mlp_dim=256, use_joint=True):
        super().__init__()
        self.num_points = num_points
        
        cfg = EasyConfig()
        cfg.load(pointNeXt_cfg, recursive=True)
        self.pointNeXt = build_model_from_cfg(cfg.model)
        feat_dim = cfg.model.encoder_args.width

        self.mlp_start = MLP_Res(feat_dim, mlp_dim, out_dim=2)
        self.mlp_end = MLP_Res(feat_dim, mlp_dim, out_dim=3)
            
    def forward(self, pcs_start, pcs_end, joint_info = None):
        B, N, C = pcs_start.shape
        pcs = torch.cat([pcs_start, pcs_end], dim=1)  # B, 2*N, C
        feat = self.pointNeXt(pcs)
        
        feat_start, feat_end = torch.split(feat, [self.num_points, self.num_points], dim=2)
        seg_start = F.log_softmax(self.mlp_start(feat_start).transpose(2, 1).contiguous(), dim=-1)
        seg_end = F.log_softmax(self.mlp_end(feat_end).transpose(2, 1).contiguous(), dim=-1)
        return seg_start, seg_end

class SegNetBaseline(nn.Module):
    def __init__(self, pointNeXt_cfg, num_points=4096, mlp_dim=256, use_joint=False):
        super().__init__()
        self.num_points = num_points
        self.use_joint = use_joint
        
        cfg = EasyConfig()
        cfg.load(pointNeXt_cfg, recursive=True)
        self.pointNeXt_start = build_model_from_cfg(cfg.model)
        self.pointNeXt_end = build_model_from_cfg(cfg.model)
        feat_dim = cfg.model.encoder_args.width

        self.ww = nn.Linear(feat_dim,feat_dim)
        
        self.mlp_start = MLP_Res(feat_dim+1, mlp_dim, out_dim=2)
        self.mlp_end = MLP_Res(feat_dim+1, mlp_dim, out_dim=3)
            
    def forward(self, pcs_start, pcs_end, joint_info = None):
        # B, N, C = pcs_start.shape
        pcs = pcs_start['pos']
        pcs_start = pcs[:, :self.num_points, :].contiguous()
        pcs_end = pcs[:, self.num_points:, :].contiguous()
        feat_start = self.pointNeXt_start(pcs_start).transpose(2, 1).contiguous()
        feat_end = self.pointNeXt_end(pcs_end).transpose(2, 1).contiguous()
        
        
        w_start = self.ww(feat_start) # B N source_dim
        w_end = self.ww(feat_end) # B N source_dim
        w_end_t = w_end.transpose(2, 1).contiguous()  # B source_dim N
        feat_source = torch.matmul(w_start,w_end_t) # B N N

        feat_source = F.relu(feat_source)
        source_start = torch.sum(feat_source, dim=2, keepdim=True)  # B N 1
        source_end = torch.sum(feat_source, dim=1, keepdim=True)  # B 1 N
        source_end = source_end.transpose(2, 1).contiguous() # B N 1

        feat_start = torch.cat([feat_start,source_start],dim=2) # B N 1+feat_dim
        feat_end = torch.cat([feat_end,source_end],dim=2) # B N 1+feat_dim

        # Conv Network
        feat_start = feat_start.transpose(2, 1).contiguous() 
        feat_end = feat_end.transpose(2, 1).contiguous()
        seg_start = self.mlp_start(feat_start)
        seg_end = self.mlp_end(feat_end)
        seg_start = seg_start.transpose(2, 1).contiguous()
        seg_end = seg_end.transpose(2, 1).contiguous()
        seg_start = F.log_softmax(seg_start, dim=-1) # B N 2
        seg_end = F.log_softmax(seg_end, dim=-1) # B N 2

        return seg_start, seg_end 


from torchinfo import summary

if __name__ == '__main__':
    model = SegNet("/mnt/disk3/zihao/dev/pointnet_tem/model_cfg/pointNext_seg.yaml")
    model = model.cuda()


    summary(model, input_size=[(8, 4096, 3), (8, 4096, 3), (8,1,7)])