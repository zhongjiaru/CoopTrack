# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
""" Spatial-temporal Reasoning Module
"""
import os
import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ..dense_heads.track_head_plugin import Instances
from mmcv.cnn import Conv2d, Linear
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import build_transformer
from .pf_utils import time_position_embedding, xyz_ego_transformation, normalize, denormalize
from scipy.optimize import linear_sum_assignment
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox

color_mapping = [
    np.array([1.0, 0.078, 0.576]), # 鲜艳的粉色
    np.array([1.0, 1.0, 0.0]),   # 鲜艳的黄色
    np.array([1.0, 0.647, 0.0]), # 鲜艳的橙色
    np.array([0.502, 0.0, 0.502]), # 鲜艳的紫色
    np.array([0.0, 1.0, 1.0]),   # 鲜艳的青色
    np.array([1.0, 0.0, 1.0]),   # 鲜艳的洋红色
    np.array([0.0, 1.0, 0.502]), # 鲜艳的青绿色
    np.array([1.0, 0.843, 0.0]),  # 鲜艳的金色
    np.array([0.0, 0.8, 0.0]),
    np.array([0.4, 0.0, 0.8]),   # 鲜艳的蓝色
]

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

class SpatialTemporalReasoner(nn.Module):
    def __init__(self, 
                 history_reasoning=True,
                 future_reasoning=True,
                 embed_dims=256, 
                 hist_len=3, 
                 fut_len=4,
                 num_reg_fcs=2,
                 code_size=10,
                 num_classes=10,
                 pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                 hist_temporal_transformer=None,
                 fut_temporal_transformer=None,
                 spatial_transformer=None,
                 is_motion=False,
                 is_cooperation=False,
                 learn_match=False,
                 veh_thre=0.05):
        super(SpatialTemporalReasoner, self).__init__()

        self.embed_dims = embed_dims
        self.hist_len = hist_len
        self.fut_len = fut_len
        self.num_reg_fcs = num_reg_fcs
        self.pc_range = pc_range

        self.num_classes = num_classes
        self.code_size = code_size

        self.history_reasoning = history_reasoning
        self.future_reasoning = future_reasoning
        self.is_motion = is_motion
        self.is_cooperation = is_cooperation
        self.learn_match = learn_match
        self.veh_thre = veh_thre
        self.hist_temporal_transformer = hist_temporal_transformer
        self.fut_temporal_transformer = fut_temporal_transformer
        self.spatial_transformer = spatial_transformer

        self.init_params_and_layers()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, track_instances, inf_track_instances=None, sample_idx=None):
        # 1. Prepare the spatial-temporal features
        track_instances = self.frame_shift(track_instances)

        affinity = None
        # 2. History reasoning
        if self.history_reasoning:
            track_instances = self.forward_history_reasoning(track_instances)
            if inf_track_instances or self.learn_match:
                track_instances, affinity = self.aggregation(track_instances, inf_track_instances, sample_idx)
            track_instances = self.forward_history_refine(track_instances)

        # 3. Future reasoning
        if self.future_reasoning:
            track_instances = self.forward_future_reasoning(track_instances)
            track_instances = self.forward_future_prediction(track_instances)
        return track_instances, affinity

    def forward_history_reasoning(self, track_instances: Instances):
        """Using history information to refine the current frame features
        """
        if len(track_instances) == 0:
            return track_instances
        
        valid_idxes = (track_instances.hist_padding_masks[:, -1] == 0)
        embed = track_instances.cache_query_feats[valid_idxes]

        if len(embed) == 0:
            return track_instances
        
        hist_embed = track_instances.hist_embeds[valid_idxes]
        hist_padding_mask = track_instances.hist_padding_masks[valid_idxes]

        ts_pe = time_position_embedding(hist_embed.shape[0], self.hist_len, 
                                        self.embed_dims, hist_embed.device)
        ts_pe = self.ts_query_embed(ts_pe)
        temporal_embed = self.hist_temporal_transformer(
            target=embed[:, None, :], 
            x=hist_embed, 
            query_embed=ts_pe[:, -1:, :],
            pos_embed=ts_pe,
            query_key_padding_mask=hist_padding_mask[:, -1:],
            key_padding_mask=hist_padding_mask)

        hist_pe = track_instances.cache_query_embeds[valid_idxes, None, :]
        spatial_embed = self.spatial_transformer(
            target=temporal_embed.transpose(0, 1),
            x=temporal_embed.transpose(0, 1),
            query_embed=hist_pe.transpose(0, 1),
            pos_embed=hist_pe.transpose(0, 1),
            query_key_padding_mask=hist_padding_mask[:, -1:].transpose(0, 1),
            key_padding_mask=hist_padding_mask[:, -1:].transpose(0, 1))[0]

        if self.is_motion:
            mot_embed = track_instances.cache_motion_feats[valid_idxes]
            hist_mot_embed = track_instances.hist_motion_embeds[valid_idxes]
            hist_padding_mask = track_instances.hist_padding_masks[valid_idxes]

            ts_pe = time_position_embedding(hist_mot_embed.shape[0], self.hist_len, 
                                            self.embed_dims, hist_mot_embed.device)
            ts_pe = self.ts_query_embed(ts_pe)
            mot_embed = self.hist_motion_transformer(
                target=mot_embed[:, None, :], 
                x=hist_mot_embed, 
                query_embed=ts_pe[:, -1:, :],
                pos_embed=ts_pe,
                query_key_padding_mask=hist_padding_mask[:, -1:],
                key_padding_mask=hist_padding_mask)[:, 0, :]
            track_instances.cache_motion_feats[valid_idxes] = mot_embed.clone()
            track_instances.hist_motion_embeds[valid_idxes, -1] = mot_embed.clone().detach()
        
        track_instances.cache_query_feats[valid_idxes] = spatial_embed.clone()
        track_instances.hist_embeds[valid_idxes, -1] = spatial_embed.clone().detach()
        return track_instances
    
    def forward_history_refine(self, track_instances: Instances):
        """Refine localization and classification"""
        if len(track_instances) == 0:
            return track_instances
        
        valid_idxes = (track_instances.hist_padding_masks[:, -1] == 0)
        embed = track_instances.cache_query_feats[valid_idxes]

        if len(embed) == 0:
            return track_instances
        
        """Classification"""
        logits = self.track_cls(track_instances.cache_query_feats[valid_idxes])
        # track_instances.hist_logits[valid_idxes, -1, :] = logits.clone()
        track_instances.cache_logits[valid_idxes] = logits.clone()
        track_instances.cache_scores = logits.sigmoid().max(dim=-1).values

        """Localization"""
        if self.is_motion:
            deltas = self.track_reg(track_instances.cache_motion_feats[valid_idxes])
        else:
            deltas = self.track_reg(track_instances.cache_query_feats[valid_idxes])
        reference = inverse_sigmoid(track_instances.cache_ref_pts[valid_idxes].clone())
        deltas[..., [0, 1, 4]] += reference
        deltas[..., [0, 1, 4]] = deltas[..., [0, 1, 4]].sigmoid()

        track_instances.cache_ref_pts[valid_idxes] = deltas[..., [0, 1, 4]].clone()
        # track_instances.hist_xyz[valid_idxes, -1, :] = deltas[..., [0, 1, 4]].clone()
        deltas[..., [0, 1, 4]] = denormalize(deltas[..., [0, 1, 4]], self.pc_range)
        track_instances.cache_bboxes[valid_idxes, :] = deltas
        # track_instances.hist_bboxes[valid_idxes, -1, :] = deltas.clone()
        return track_instances

    def forward_future_reasoning(self, track_instances: Instances):
        hist_embeds = track_instances.hist_embeds
        hist_padding_masks = track_instances.hist_padding_masks
        ts_pe = time_position_embedding(hist_embeds.shape[0], self.hist_len + self.fut_len, 
                                        self.embed_dims, hist_embeds.device)
        ts_pe = self.ts_query_embed(ts_pe)
        fut_embeds = self.fut_temporal_transformer(
            target=torch.zeros_like(ts_pe[:, self.hist_len:, :]),
            x=hist_embeds,
            query_embed=ts_pe[:, self.hist_len:],
            pos_embed=ts_pe[:, :self.hist_len],
            key_padding_mask=hist_padding_masks)
        track_instances.fut_embeds = fut_embeds
        return track_instances
    
    def forward_future_prediction(self, track_instances):
        """Predict the future motions"""
        motion_predictions = self.future_reg(track_instances.fut_embeds)
        track_instances.cache_motion_predictions = motion_predictions
        return track_instances

    def aggregation(self, veh_instances, inf_instances, sample_idx):
        # aggregate cache_query_feats, cache_motion_feats, cache_ref_pts
        # 1. association
        veh_ref_pts = veh_instances.cache_ref_pts.clone()
        inf_ref_pts = inf_instances.cache_ref_pts.clone()
        veh_abs_pts = self._loc_denorm(veh_ref_pts, self.pc_range)
        inf_abs_pts = self._loc_denorm(inf_ref_pts, self.pc_range)

        mask = veh_instances.cache_scores > self.veh_thre
        
        affinity = None
        if not self.learn_match:
            # rule-based match
            veh_idx, inf_idx, _ = self._query_matching(inf_abs_pts, veh_abs_pts, mask)
        else:
            # learning match
            veh_idx, inf_idx, affinity = self._learn_matching(inf_instances, veh_instances, inf_abs_pts, veh_abs_pts, mask)
            if self.training:
                veh_idx, inf_idx, _ = self._query_matching(inf_abs_pts, veh_abs_pts, mask)

        # 2. aggregation
        fused_instances = self._query_fusion(inf_instances, veh_instances, inf_idx, veh_idx)
        res_instances = self._query_complementation(inf_instances, veh_instances, inf_idx, veh_idx, fused_instances)

        res_instances.cache_query_feats = self.cross_domain_query(torch.cat([res_instances.cache_query_feats, res_instances.cache_query_embeds], dim=-1))
        res_instances.cache_motion_feats = self.cross_domain_motion(torch.cat([res_instances.cache_motion_feats, res_instances.cache_query_embeds], dim=-1))
        return res_instances, affinity

    def _query_fusion(self, inf, veh, inf_idx, veh_idx):
        """
        Query fusion: 
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP
        
        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """
        matched_veh = veh[veh_idx]
        matched_inf = inf[inf_idx]
        matched_veh.cache_query_feats = self.fuse_feats(torch.cat([matched_veh.cache_query_feats, matched_inf.cache_query_feats], dim=-1))
        matched_veh.cache_motion_feats = self.fuse_motion(torch.cat([matched_veh.cache_motion_feats, matched_inf.cache_motion_feats], dim=-1))
        matched_veh.cache_query_embeds = self.fuse_embed(torch.cat([matched_veh.cache_query_embeds, matched_inf.cache_query_embeds], dim=-1))
        matched_veh.cache_ref_pts = (matched_veh.cache_ref_pts + matched_inf.cache_ref_pts) / 2.0
        return matched_veh

    def _query_complementation(self, inf, veh, inf_accept_idx, veh_accept_idx, fused):
        """
        Query complementation: replace low-confidence vehicle-side query with unmatched inf-side query

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_accept_idx: idxs of matched instances
        """
        veh_num = len(veh)
        inf_num = len(inf)

        mask = torch.ones(veh_num, dtype=bool)
        mask[veh_accept_idx] = False
        unmatched_veh = veh[mask]

        mask = torch.ones(inf_num, dtype=bool)
        mask[inf_accept_idx] = False
        unmatched_inf = inf[mask]
        res_instances = Instances((1, 1))
        res_instances = Instances.cat([res_instances, fused])
        res_instances = Instances.cat([res_instances, unmatched_inf])

        select_num = veh_num - inf_num
        _, topk_indexes = torch.topk(unmatched_veh.cache_scores, select_num, dim=0)
        res_instances = Instances.cat([res_instances, unmatched_veh[topk_indexes]])

        return res_instances
    
    def _loc_denorm(self, ref_pts, pc_range):
        """
        normalized (x,y,z) ---> absolute (x,y,z) in global coordinate system
        """
        locs = ref_pts.clone()

        locs[:, 0:1] = (locs[:, 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
        locs[:, 1:2] = (locs[:, 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
        locs[:, 2:3] = (locs[:, 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2])

        return locs
    
    def _learn_matching(self, inf, veh, inf_pts, veh_pts, mask):
        # filter veh
        if self.training:
            indices = torch.arange(len(veh))
            filter_indices = indices[mask]
            veh = veh[mask]

        inf_query = inf.cache_query_feats.clone()
        inf_motion = inf.cache_motion_feats.clone()
        inf_ref_pts = inf.cache_ref_pts.clone()
        veh_query = veh.cache_query_feats.clone()
        veh_motion = veh.cache_motion_feats.clone()
        veh_ref_pts = veh.cache_ref_pts.clone()
        # 1. construct graph
        # 1.1 prepare nodes
        inf_nodes = self.get_node_inf(torch.cat([inf_query, inf_motion], dim=-1))
        veh_nodes = self.get_node_veh(torch.cat([veh_query, veh_motion], dim=-1))

        # 1.2 prepare edges
        dis = veh_ref_pts.unsqueeze(1) - inf_ref_pts
        edges = self.get_edge(dis)

        # 2. generate affine matrix
        q = torch.matmul(veh_nodes, self.wq)
        k = torch.matmul(inf_nodes, self.wk)
        e = torch.matmul(edges, self.we)

        attn = torch.matmul(q, k.transpose(0, 1)).unsqueeze(-1)
        attn = attn + e
        attn = self.ffn(attn).squeeze(-1)

        if self.training:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), attn
        # 3. output associated idxes
        affinity = self.sigmoid(attn)
        if affinity.shape[0] == 0 or affinity.shape[1] == 0:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), attn
        
        # affinity: close to 1 means the objects are same
        norm_diff_aff = 1 - affinity.detach().cpu()
        diff = norm_diff_aff

        cost = diff.numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        cost_mask = cost[row_ind, col_ind] < 1e5
        veh_accept_idx = row_ind[cost_mask]
        inf_accept_idx = col_ind[cost_mask]
        if self.training:
            veh_accept_idx = filter_indices[veh_accept_idx]
        return torch.tensor(veh_accept_idx), torch.tensor(inf_accept_idx), attn

    def _query_matching(self, inf_ref_pts, veh_ref_pts, mask):
        """
        inf_ref_pts: [..., 3] (xyz)
        veh_ref_pts: [..., 3] (xyz)
        veh_pred_dims: [..., 3] (dx, dy, dz)
        """
        inf_nums = inf_ref_pts.shape[0]
        veh_nums = veh_ref_pts.shape[0]
        cost_matrix = np.ones((veh_nums, inf_nums)) * 1e6

        veh_ref_pts_np = veh_ref_pts.detach().cpu().numpy()
        inf_ref_pts_np = inf_ref_pts.cpu().numpy()

        veh_ref_pts_repeat = np.repeat(veh_ref_pts_np[:, np.newaxis], inf_nums, axis=1)
        distances = np.sqrt(np.sum((veh_ref_pts_repeat - inf_ref_pts_np)**2, axis=2))

        mask = distances > 3.5
        cost_matrix[~mask] = distances[~mask]
        
        idx_veh, idx_inf = linear_sum_assignment(cost_matrix)

        cost_mask = cost_matrix[idx_veh, idx_inf] < 1e5
        veh_accept_idx = idx_veh[cost_mask]
        inf_accept_idx = idx_inf[cost_mask]

        return veh_accept_idx, inf_accept_idx, cost_matrix
    
    def _gen_asso_label(self, gt_boxes, inf_boxes, veh_boxes, sample_idx):
        with torch.no_grad():
            inf_num = inf_boxes.shape[0]
            veh_num = veh_boxes.shape[0]
            device = veh_boxes.device
            label = torch.zeros((veh_num, inf_num), dtype=torch.long, device=device)
            gt_boxes = gt_boxes[0].tensor.to(device)
            gt_boxes = normalize_bbox(gt_boxes, self.pc_range)
            
            def hungarian_matching(box1, box2, thre=5.0):
                box1_nums = box1.shape[0]
                box2_nums = box2.shape[0]
                cost_matrix = np.ones((box1_nums, box2_nums)) * 1e6

                box1_np = box1.cpu().numpy()
                box2_np = box2.cpu().numpy()
                box1_np = box1_np[:, 0:2]
                box2_np = box2_np[:, 0:2]
                box1_repeat = np.repeat(box1_np[:, np.newaxis], box2_nums, axis=1)
                distances = np.sqrt(np.sum((box1_repeat - box2_np)**2, axis=2))

                mask = distances > 3.0
                cost_matrix[~mask] = distances[~mask]
                
                idx_row, idx_col = linear_sum_assignment(cost_matrix)

                cost_mask = cost_matrix[idx_row, idx_col] < 1e5
                row_accept_idx = idx_row[cost_mask]
                col_accept_idx = idx_col[cost_mask]
                return row_accept_idx, col_accept_idx

            # associate veh and gt
            matched_veh_indices, mathced_gt_indices_from_v = hungarian_matching(veh_boxes, gt_boxes)
            # associate inf and gt
            matched_inf_indices, mathced_gt_indices_from_i = hungarian_matching(inf_boxes, gt_boxes)

            from collections import defaultdict
            gt_to_veh = defaultdict(list)
            for veh_idx, gt_idx in zip(matched_veh_indices, mathced_gt_indices_from_v):
                gt_to_veh[gt_idx].append(veh_idx)

            gt_to_inf = defaultdict(list)
            for inf_idx, gt_idx in zip(matched_inf_indices, mathced_gt_indices_from_i):
                gt_to_inf[gt_idx].append(inf_idx)

            pairs = []
            for gt_idx in gt_to_veh:
                if gt_idx in gt_to_inf:
                    for veh_idx in gt_to_veh[gt_idx]:
                        for inf_idx in gt_to_inf[gt_idx]:
                            label[veh_idx, inf_idx] = 1
                            pairs.append((veh_idx, inf_idx, gt_idx))

        return label
    
    def init_params_and_layers(self):
        # Modules for history reasoning
        if self.history_reasoning:
            # temporal transformer
            self.hist_temporal_transformer = build_transformer(self.hist_temporal_transformer)
            self.spatial_transformer = build_transformer(self.spatial_transformer)
            if self.is_motion:
                self.hist_motion_transformer = copy.deepcopy(self.hist_temporal_transformer)
            # classification refinement from history
            cls_branch = []
            for _ in range(self.num_reg_fcs):
                cls_branch.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch.append(nn.LayerNorm(self.embed_dims))
                cls_branch.append(nn.ReLU(inplace=True))
            cls_branch.append(Linear(self.embed_dims, self.num_classes))
            self.track_cls = nn.Sequential(*cls_branch)
    
            # localization refinement from history
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.LayerNorm(self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(Linear(self.embed_dims, self.code_size))
            self.track_reg = nn.Sequential(*reg_branch)
        
        # Modules for future reasoning
        if self.future_reasoning:
            # temporal transformer
            self.fut_temporal_transformer = build_transformer(self.fut_temporal_transformer)

            # regression head
            reg_branch = []
            for _ in range(self.num_reg_fcs):
                reg_branch.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch.append(nn.LayerNorm(self.embed_dims))
                reg_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(Linear(self.embed_dims, 3))
            self.future_reg = nn.Sequential(*reg_branch)
        
        if self.future_reasoning or self.history_reasoning:
            self.ts_query_embed = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        if self.is_cooperation:
            self.fuse_feats = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.fuse_motion = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.fuse_embed = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.cross_domain_query = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )
            self.cross_domain_motion = nn.Sequential(
                nn.Linear(self.embed_dims*2, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims)
            )

            if self.learn_match:
                self.get_node_veh = nn.Sequential(
                    nn.Linear(self.embed_dims*2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.get_node_inf = nn.Sequential(
                    nn.Linear(self.embed_dims*2, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.get_edge = nn.Sequential(
                    nn.Linear(3, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.get_dummy_node = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, self.embed_dims)
                )
                self.get_dummy_pts = nn.Sequential(
                    nn.Linear(3, self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, 3)
                )
                self.wq = nn.Parameter(torch.Tensor(self.embed_dims, self.embed_dims))
                self.wk = nn.Parameter(torch.Tensor(self.embed_dims, self.embed_dims))
                self.we = nn.Parameter(torch.Tensor(self.embed_dims, self.embed_dims))
                torch.nn.init.kaiming_uniform_(self.wq, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.wk, a=math.sqrt(5))
                torch.nn.init.kaiming_uniform_(self.we, a=math.sqrt(5))
                self.ffn = nn.Sequential(
                    nn.Linear(self.embed_dims, self.embed_dims),
                    nn.LayerNorm(self.embed_dims),
                    nn.ReLU(),
                    nn.Linear(self.embed_dims, 1)
                )
                self.sigmoid = nn.Sigmoid()
        return
    
    def sync_pos_embedding(self, track_instances: Instances, mlp_embed: nn.Module = None):
        """Synchronize the positional embedding across all fields"""
        if mlp_embed is not None:
            track_instances.query_embeds = mlp_embed(pos2posemb3d(track_instances.ref_pts))
            track_instances.hist_position_embeds = mlp_embed(pos2posemb3d(track_instances.hist_xyz))
        return track_instances
    
    def update_ego(self, track_instances: Instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        Args:
            track_instances: objects
            l2g0: a [4x4] matrix for current frame lidar-to-global transformation 
            l2g1: a [4x4] matrix for target frame lidar-to-global transformation
        Return:
            transformed track_instances (inplace)
        """
        # TODO: orientation of the bounding boxes
        """1. Current states"""
        ref_points = track_instances.ref_pts.clone()
        physical_ref_points = xyz_ego_transformation(ref_points, l2g_r1, l2g_t1, l2g_r2, l2g_t2, self.pc_range,
                                                     src_normalized=True, tgt_normalized=False)
        track_instances.pred_boxes[..., [0, 1, 4]] = physical_ref_points.clone()
        track_instances.ref_pts = normalize(physical_ref_points, self.pc_range)
        
        """2. History states"""
        inst_num = len(track_instances)
        hist_ref_xyz = track_instances.hist_xyz.clone().view(inst_num * self.hist_len, 3)
        physical_hist_ref = xyz_ego_transformation(hist_ref_xyz, l2g_r1, l2g_t1, l2g_r2, l2g_t2, self.pc_range,
                                                   src_normalized=True, tgt_normalized=False)
        physical_hist_ref = physical_hist_ref.reshape(inst_num, self.hist_len, 3)
        track_instances.hist_bboxes[..., [0, 1, 4]] = physical_hist_ref
        track_instances.hist_xyz = normalize(physical_hist_ref, self.pc_range)
        
        """3. Future states"""
        inst_num = len(track_instances)
        fut_ref_xyz = track_instances.fut_xyz.clone().view(inst_num * self.fut_len, 3)
        physical_fut_ref = xyz_ego_transformation(fut_ref_xyz, l2g_r1, l2g_t1, l2g_r2, l2g_t2, self.pc_range,
                                                   src_normalized=True, tgt_normalized=False)
        physical_fut_ref = physical_fut_ref.reshape(inst_num, self.fut_len, 3)
        track_instances.fut_bboxes[..., [0, 1, 4]] = physical_fut_ref
        track_instances.fut_xyz = normalize(physical_fut_ref, self.pc_range)

        return track_instances
    
    def update_reference_points(self, track_instances, time_deltas, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction
           Used for next frame
        """
        if use_prediction:
            # inference mode, use multi-step forecasting to modify reference points
            if tracking:
                motions = track_instances.motion_predictions[:, 0, :2].clone()
                reference_points = track_instances.ref_pts.clone()
                motions[:, 0] /= (self.pc_range[3] - self.pc_range[0])
                motions[:, 1] /= (self.pc_range[4] - self.pc_range[1])
                reference_points[..., :2] += motions.clone().detach()
                track_instances.ref_pts = reference_points
            # training mode, single-step prediction
            else:
                track_instances.ref_pts = track_instances.fut_xyz[:, 0, :].clone()
                track_instances.pred_boxes = track_instances.fut_bboxes[:, 0, :].clone()
        else:
            velos = track_instances.pred_boxes[..., 8:10].clone()
            reference_points = track_instances.ref_pts.clone()
            velos[:, 0] /= (self.pc_range[3] - self.pc_range[0])
            velos[:, 1] /= (self.pc_range[4] - self.pc_range[1])
            reference_points[..., :2] += (velos * time_deltas).clone().detach()
            track_instances.ref_pts = reference_points
        return track_instances
    
    def frame_shift(self, track_instances: Instances):
        """Shift the information for the newest frame before spatial-temporal reasoning happens. 
           Pay attention to the order.
        """
        device = track_instances.query_feats.device
        
        """1. History reasoining"""
        # embeds
        track_instances.hist_embeds = track_instances.hist_embeds.clone()
        track_instances.hist_embeds = torch.cat((
            track_instances.hist_embeds[:, 1:, :], track_instances.cache_query_feats[:, None, :]), dim=1)
        # padding masks
        track_instances.hist_padding_masks = torch.cat((
            track_instances.hist_padding_masks[:, 1:], 
            torch.zeros((len(track_instances), 1), dtype=torch.bool, device=device)), 
            dim=1)
        # positions
        track_instances.hist_xyz = torch.cat((
            track_instances.hist_xyz[:, 1:, :], track_instances.cache_ref_pts[:, None, :]), dim=1)
        # positional embeds
        track_instances.hist_position_embeds = torch.cat((
            track_instances.hist_position_embeds[:, 1:, :], track_instances.cache_query_embeds[:, None, :]), dim=1)
        # bboxes
        track_instances.hist_bboxes = torch.cat((
            track_instances.hist_bboxes[:, 1:, :], track_instances.cache_bboxes[:, None, :]), dim=1)
        # logits
        track_instances.hist_logits = torch.cat((
            track_instances.hist_logits[:, 1:, :], track_instances.cache_logits[:, None, :]), dim=1)
        # scores
        track_instances.hist_scores = torch.cat((
            track_instances.hist_scores[:, 1:], track_instances.cache_scores[:, None]), dim=1)
        # motion features
        track_instances.hist_motion_embeds = track_instances.hist_motion_embeds.clone()
        track_instances.hist_motion_embeds = torch.cat((
            track_instances.hist_motion_embeds[:, 1:, :], track_instances.cache_motion_feats[:, None, :]), dim=1)
        """2. Temporarily load motion predicted results as final results"""
        if self.future_reasoning:
            track_instances.ref_pts = track_instances.fut_xyz[:, 0, :].clone()
            track_instances.pred_boxes = track_instances.fut_bboxes[:, 0, :].clone()
            track_instances.scores = track_instances.fut_scores[:, 0].clone() + 0.01
            track_instances.pred_logits = track_instances.fut_logits[:, 0, :].clone()

        """3. Future reasoning"""
        # TODO: shift the future information, useful for occlusion handling
        track_instances.motion_predictions = torch.cat((
            track_instances.motion_predictions[:, 1:, :], torch.zeros_like(track_instances.motion_predictions[:, 0:1, :])), dim=1)
        track_instances.fut_embeds = torch.cat((
            track_instances.fut_embeds[:, 1:, :], torch.zeros_like(track_instances.fut_embeds[:, 0:1, :])), dim=1)
        track_instances.fut_padding_masks = torch.cat((
            track_instances.fut_padding_masks[:, 1:], torch.ones_like(track_instances.fut_padding_masks[:, 0:1]).bool()), dim=1)
        track_instances.fut_xyz = torch.cat((
            track_instances.fut_xyz[:, 1:, :], torch.zeros_like(track_instances.fut_xyz[:, 0:1, :])), dim=1)
        track_instances.fut_position_embeds = torch.cat((
            track_instances.fut_position_embeds[:, 1:, :], torch.zeros_like(track_instances.fut_position_embeds[:, 0:1, :])), dim=1)
        track_instances.fut_bboxes = torch.cat((
            track_instances.fut_bboxes[:, 1:, :], torch.zeros_like(track_instances.fut_bboxes[:, 0:1, :])), dim=1)
        track_instances.fut_logits = torch.cat((
            track_instances.fut_logits[:, 1:, :], torch.zeros_like(track_instances.fut_logits[:, 0:1, :])), dim=1)
        track_instances.fut_scores = torch.cat((
            track_instances.fut_scores[:, 1:], torch.zeros_like(track_instances.fut_scores[:, 0:1])), dim=1)
        return track_instances