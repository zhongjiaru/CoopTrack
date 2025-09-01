import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Linear
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox

class LatentTransformation(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 head=16,
                 rot_dims=6,
                 trans_dims=3,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 inf_pc_range=[0, -51.2, -5.0, 102.4, 51.2, 3.0],
                 ):
        super(LatentTransformation, self).__init__()
        self.embed_dims = embed_dims
        self.head = head
        self.rot_dims = rot_dims
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range

        rot_final_dim = int((embed_dims / head) * (embed_dims / head) * head)
        trans_final_dim = embed_dims

        layers = []
        dims = [rot_dims, embed_dims, embed_dims, rot_final_dim]
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.rot_mlp = nn.Sequential(*layers)

        layers = []
        dims = [trans_dims, embed_dims, embed_dims, trans_final_dim]
        for i in range(len(dims) - 1):
            layers.append(Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.ReLU())
        self.trans_mlp = nn.Sequential(*layers)

        self.feat_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.embed_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.motion_input_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )

        self.feat_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.embed_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
        self.motion_output_proj = nn.Sequential(
            Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU()
        )
    
    def continuous_rot(self, rot):
        ret = rot[:, :2].clone()
        ret = ret.reshape(1, -1)
        return ret

    def fill_tensor(self, original_tensor):
        h = self.head
        k = self.embed_dims // self.head
        d = self.embed_dims
        
        blocks_indices = torch.arange(h, device=original_tensor.device) * k
        
        offset = torch.arange(k, device=original_tensor.device)
        rows_offset, cols_offset = torch.meshgrid(offset, offset)
        
        base_rows = blocks_indices.view(h, 1, 1)
        global_rows = base_rows + rows_offset.unsqueeze(0)
        base_cols = blocks_indices.view(h, 1, 1)
        global_cols = base_cols + cols_offset.unsqueeze(0)
        
        all_rows = global_rows.reshape(-1)
        all_cols = global_cols.reshape(-1)
        data = original_tensor.view(-1)
        
        target_tensor = torch.zeros((d, d), device=original_tensor.device)
        target_tensor[all_rows, all_cols] = data
        
        return target_tensor

    def transform_pts(self, points, transformation):
        # relative -> absolute (in inf pc range)
        locs = points.clone()
        locs[:, 0:1] = (locs[:, 0:1] * (self.inf_pc_range[3] - self.inf_pc_range[0]) + self.inf_pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (self.inf_pc_range[4] - self.inf_pc_range[1]) + self.inf_pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (self.inf_pc_range[5] - self.inf_pc_range[2]) + self.inf_pc_range[2])

        # transformation
        locs = torch.cat((locs, torch.ones_like(locs[..., :1])), -1).unsqueeze(-1)
        locs = torch.matmul(transformation, locs).squeeze(-1)[..., :3]
        
        # filter
        mask = (self.pc_range[0] <= locs[:, 0]) & (locs[:, 0] <= self.pc_range[3]) & \
                    (self.pc_range[1] <= locs[:, 1]) & (locs[:, 1] <= self.pc_range[4])
        locs = locs[mask]
        # absolute -> relative (in veh pc range)
        locs[..., 0:1] = (locs[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
        locs[..., 1:2] = (locs[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
        locs[..., 2:3] = (locs[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])

        return locs, mask
    
    def transform_boxes(self, pred_boxes, transformation):
        device = transformation.device
        transformation = transformation.cpu()
        pred_boxes = pred_boxes.cpu()

        pred_boxes = denormalize_bbox(pred_boxes, self.inf_pc_range)
        pred_boxes = LiDARInstance3DBoxes(pred_boxes, 9)
        rot = transformation[:3, :3].T
        trans = transformation[:3, 3:4].reshape(1, 3)
        pred_boxes.rotate(rot)
        pred_boxes.translate(trans)
        pred_boxes = normalize_bbox(pred_boxes.tensor, self.pc_range)
        pred_boxes = pred_boxes.to(device)
        
        return pred_boxes

    def forward(self, instances, veh2inf_rt):
        calib_inf2veh = np.linalg.inv(veh2inf_rt.cpu().numpy().T)
        calib_inf2veh = instances['ref_pts'].new_tensor(calib_inf2veh)
        rot = calib_inf2veh[:3, :3].clone()
        trans = calib_inf2veh[:3, 3:4].clone()

        if self.rot_dims == 6:
            con_rot = self.continuous_rot(rot)
            assert con_rot.size(1) == 6
        trans = trans.reshape(1, -1)

        rot_para = self.rot_mlp(con_rot)
        trans_para = self.trans_mlp(trans)
        rot_mat = self.fill_tensor(rot_para)
        instances['ref_pts'], mask = self.transform_pts(instances['ref_pts'], calib_inf2veh)
        instances['query_feats'] = instances['query_feats'][mask]
        instances['query_embeds'] = instances['query_embeds'][mask]
        instances['cache_motion_feats'] = instances['cache_motion_feats'][mask]
        instances['pred_boxes'] = instances['pred_boxes'][mask]

        identity_query_feats = instances['query_feats'].clone()
        identity_query_embeds = instances['query_embeds'].clone()
        identity_cache_motion_feats = instances['cache_motion_feats'].clone()

        instances['query_feats'] = self.feat_output_proj((self.feat_input_proj(instances['query_feats']) @ rot_mat.T + trans_para) + identity_query_feats)
        instances['query_embeds'] = self.embed_output_proj((self.embed_input_proj(instances['query_embeds']) @ rot_mat.T + trans_para) + identity_query_embeds)
        instances['cache_motion_feats'] = self.motion_output_proj((self.motion_input_proj(instances['cache_motion_feats']) @ rot_mat.T + trans_para) + identity_cache_motion_feats)
        
        instances['pred_boxes'] = self.transform_boxes(instances['pred_boxes'], calib_inf2veh)
        return instances

        