#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
import copy
import math
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmdet.models import build_loss
from einops import rearrange
from mmdet.models.utils.transformer import inverse_sigmoid
from ..dense_heads.track_head_plugin import Instances, RunTimeTracker
from ..modules import CrossAgentSparseInteraction
import mmcv,os
import torch.nn.functional as F
import numpy as np
from ..modules import SpatialTemporalReasoner, MotionExtractor, LatentTransformation
from ..modules import pos2posemb3d
from torchvision.ops import sigmoid_focal_loss

def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result

@DETECTORS.register_module()
class CoopTrack(MVXTwoStageDetector):
    """
    CoopTrack
    """
    def __init__(
        self, 
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        pc_range=None,
        inf_pc_range=None,
        post_center_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        runtime_tracker=None,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
        is_cooperation=False,
        read_track_query_file_root=None,
        drop_rate = 0,
        save_track_query=False,
        save_track_query_file_root='',
        seq_mode=False,
        batch_size=1,
        spatial_temporal_reason=None,
        motion_prediction_ref_update=True,
        if_update_ego=True,
        train_det=False,
        fp_ratio=0.3,
        random_drop=0.1,
        shuffle=False,
        is_motion=False,
        asso_loss_cfg=None,
    ):
        super(CoopTrack, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )

        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
        )
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.inf_pc_range = inf_pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        
        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        self.query_feat_embedding = nn.Embedding(self.num_query, self.embed_dims)
        nn.init.zeros_(self.query_feat_embedding.weight)

        self.runtime_tracker = RunTimeTracker(
            **runtime_tracker
        ) 

        self.criterion = build_loss(loss_cfg)
        # for test memory
        self.scene_token = None
        self.timestamp = None
        self.prev_bev = None
        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.prev_pos = 0
        self.prev_angle = 0
        
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder

        # spatial-temporal reasoning
        self.STReasoner = SpatialTemporalReasoner(**spatial_temporal_reason)
        self.hist_len = self.STReasoner.hist_len
        self.fut_len = self.STReasoner.fut_len
        
        self.motion_prediction_ref_update=motion_prediction_ref_update
        self.if_update_ego = if_update_ego
        self.train_det = train_det
        self.fp_ratio = fp_ratio
        self.random_drop = random_drop
        self.shuffle = shuffle
        self.is_motion = is_motion
        if self.is_motion:
            self.MotionExtractor = MotionExtractor(embed_dims=embed_dims,
                                                   mlp_channels=(3, 64, 64, 256))

        # cross-agent query interaction
        self.is_cooperation = is_cooperation
        self.read_track_query_file_root = read_track_query_file_root
        if self.is_cooperation:
            self.crossview_alignment = LatentTransformation(embed_dims=embed_dims,
                                                            head=16,
                                                            rot_dims=6,
                                                            trans_dims=3,
                                                            pc_range=pc_range,
                                                            inf_pc_range=inf_pc_range
                                                            )
            if self.STReasoner.learn_match:
                self.asso_loss_focal = asso_loss_cfg['loss_focal']
        self.drop_rate = drop_rate

        self.save_track_query = save_track_query
        self.save_track_query_file_root = save_track_query_file_root

        self.bev_embed_linear = nn.Linear(embed_dims, embed_dims)
        self.bev_pos_linear = nn.Linear(embed_dims, embed_dims)

        self.seq_mode = seq_mode
        if self.seq_mode:
            self.batch_size = batch_size
            self.test_flag = False
            # for stream train memory
            self.train_prev_infos = {
                'scene_token': [None] * self.batch_size,
                'prev_timestamp': [None] * self.batch_size,
                'prev_bev': [None] * self.batch_size,
                'track_instances': [None] * self.batch_size,
                'l2g_r_mat': [None] * self.batch_size,
                'l2g_t': [None] * self.batch_size,
                'prev_pos': [0] * self.batch_size,
                'prev_angle': [0] * self.batch_size,
            }
            
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        
    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_forecasting_locs=None,
                      gt_forecasting_masks=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      #for coop
                      veh2inf_rt=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        if self.test_flag: #for interval evaluation
            self.reset_memory()
            self.test_flag = False
        losses = dict()
        if self.seq_mode:
            losses_track = self.forward_track_stream_train(img, gt_bboxes_3d, gt_labels_3d, gt_inds, gt_forecasting_locs, gt_forecasting_masks,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs)
        else:
            NotImplementedError
            # losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_inds,
            #                                             l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt)
        losses.update(losses_track)
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)
        return losses
    
    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     #for coop
                     veh2inf_rt=None,
                     **kwargs
                    ):
        """Test function
        """
        # import ipdb;ipdb.set_trace()
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp, veh2inf_rt, **kwargs)
        
        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding']
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update(result_track[i])
        return result
    
    def reset_memory(self):
        self.train_prev_infos['scene_token'] = [None] * self.batch_size
        self.train_prev_infos['prev_timestamp'] = [None] * self.batch_size
        self.train_prev_infos['prev_bev'] = [None] * self.batch_size
        self.train_prev_infos['track_instances'] = [None] * self.batch_size
        self.train_prev_infos['l2g_r_mat'] = [None] * self.batch_size
        self.train_prev_infos['l2g_t'] = [None] * self.batch_size
        self.train_prev_infos['prev_pos'] = [0] * self.batch_size
        self.train_prev_infos['prev_angle'] = [0] * self.batch_size

    def extract_img_feat(self, img, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        assert img.dim() == 5
        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B//len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        device = self.reference_points.weight.device
        
        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        track_instances.ref_pts = reference_points.clone()
        track_instances.query_embeds = query_embeds.clone()
        track_instances.query_feats = self.query_feat_embedding.weight.clone()
        
        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)
        
        """Current frame information"""
        # classification scores
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # bounding boxes
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), self.pts_bbox_head.code_size), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        
        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        track_instances.cache_bboxes = torch.zeros(
            (len(track_instances), self.pts_bbox_head.code_size), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_ref_pts = reference_points.clone()
        track_instances.cache_query_embeds = query_embeds.clone()
        track_instances.cache_query_feats = self.query_feat_embedding.weight.clone()
        track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)
        track_instances.cache_motion_feats = torch.zeros_like(query_embeds)

        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)
        # motion features
        track_instances.hist_motion_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        
        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)
        
        return track_instances
    
    def _init_inf_tracks(self, inf_dict):
        # import pdb;pdb.set_trace()
        track_instances = Instances((1, 1))
        device = inf_dict['ref_pts'].device
        
        """Detection queries"""
        # reference points, query embeds, and query targets (features)
        track_instances.ref_pts = inf_dict['ref_pts'].clone()
        track_instances.query_embeds = inf_dict['query_embeds'].clone()
        track_instances.query_feats = inf_dict['query_feats'].clone()
        track_instances.cache_motion_feats = inf_dict['cache_motion_feats'].clone()
        track_instances.pred_boxes = inf_dict['pred_boxes'].clone()
        """Tracking information"""
        # id for the tracks
        track_instances.obj_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # matched gt indexes, for loss computation
        track_instances.matched_gt_idxes = torch.full(
            (len(track_instances),), -1, dtype=torch.long, device=device)
        # life cycle management
        track_instances.disappear_time = torch.zeros(
            (len(track_instances), ), dtype=torch.long, device=device)
        track_instances.track_query_mask = torch.zeros(
            (len(track_instances), ), dtype=torch.bool, device=device)
        
        """Current frame information"""
        # classification scores
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # track scores, normally the scores for the highest class
        track_instances.scores = torch.zeros(
            (len(track_instances)), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        # motion prediction, not normalized
        track_instances.motion_predictions = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        
        """Cache for current frame information, loading temporary data for spatial-temporal reasoining"""
        track_instances.cache_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device)
        # track_instances.cache_bboxes = torch.zeros(
        #     (len(track_instances), self.pts_bbox_head.code_size), dtype=torch.float, device=device)
        track_instances.cache_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device)
        track_instances.cache_ref_pts = inf_dict['ref_pts'].clone()
        track_instances.cache_query_embeds = inf_dict['query_embeds'].clone()
        track_instances.cache_query_feats = inf_dict['query_feats'].clone()
        track_instances.cache_motion_predictions = torch.zeros_like(track_instances.motion_predictions)
        track_instances.cache_bboxes = inf_dict['pred_boxes'].clone()
        """History Reasoning"""
        # embeddings
        track_instances.hist_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.hist_padding_masks = torch.ones(
            (len(track_instances), self.hist_len), dtype=torch.bool, device=device)
        # positions
        track_instances.hist_xyz = torch.zeros(
            (len(track_instances), self.hist_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.hist_position_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.hist_bboxes = torch.zeros(
            (len(track_instances), self.hist_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.hist_logits = torch.zeros(
            (len(track_instances), self.hist_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.hist_scores = torch.zeros(
            (len(track_instances), self.hist_len), dtype=torch.float, device=device)
        # motion features
        track_instances.hist_motion_embeds = torch.zeros(
            (len(track_instances), self.hist_len, self.embed_dims), dtype=torch.float32, device=device)
        
        """Future Reasoning"""
        # embeddings
        track_instances.fut_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # padding mask, follow MultiHeadAttention, 1 indicates padded
        track_instances.fut_padding_masks = torch.ones(
            (len(track_instances), self.fut_len), dtype=torch.bool, device=device)
        # positions
        track_instances.fut_xyz = torch.zeros(
            (len(track_instances), self.fut_len, 3), dtype=torch.float, device=device)
        # positional embeds
        track_instances.fut_position_embeds = torch.zeros(
            (len(track_instances), self.fut_len, self.embed_dims), dtype=torch.float32, device=device)
        # bboxes
        track_instances.fut_bboxes = torch.zeros(
            (len(track_instances), self.fut_len, 10), dtype=torch.float, device=device)
        # logits
        track_instances.fut_logits = torch.zeros(
            (len(track_instances), self.fut_len, self.num_classes), dtype=torch.float, device=device)
        # scores
        track_instances.fut_scores = torch.zeros(
            (len(track_instances), self.fut_len), dtype=torch.float, device=device)
        
        # follow the vehicle setting
        track_instances.cache_query_feats = track_instances.query_feats.clone()
        track_instances.cache_ref_pts = track_instances.ref_pts.clone()
        track_instances.cache_query_embeds = track_instances.query_embeds.clone()
        
        track_instances.hist_padding_masks = torch.cat((
            track_instances.hist_padding_masks[:, 1:], 
            torch.zeros((len(track_instances), 1), dtype=torch.bool, device=device)), 
            dim=1)
        track_instances.hist_embeds = torch.cat((
            track_instances.hist_embeds[:, 1:, :], track_instances.cache_query_feats[:, None, :]), dim=1)
        track_instances.hist_xyz = torch.cat((
            track_instances.hist_xyz[:, 1:, :], track_instances.cache_ref_pts[:, None, :]), dim=1)
        # positional embeds
        track_instances.hist_position_embeds = torch.cat((
            track_instances.hist_position_embeds[:, 1:, :], track_instances.cache_query_embeds[:, None, :]), dim=1)
        track_instances.hist_motion_embeds = torch.cat((
            track_instances.hist_motion_embeds[:, 1:, :], track_instances.cache_motion_feats[:, None, :]), dim=1)
        return track_instances

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.reference_points.weight.device
        track_instances = Instances((1, 1))

        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)

        track_instances.matched_gt_idxes = copy.deepcopy(tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(tgt_instances.disappear_time)

        track_instances.scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.track_scores = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device
        )
        track_instances.iou = torch.zeros(
            (len(track_instances),), dtype=torch.float, device=device
        )
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        return track_instances.to(device)

    def get_history_bev(self, imgs_queue, img_metas_list):
        """
        Get history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()
        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_img_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev, _ = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, 
                    img_metas=img_metas, 
                    prev_bev=prev_bev)
        self.train()
        return prev_bev

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev)
        
        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)
        
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    def load_detection_output_into_cache(self, track_instances: Instances, out):
        """ Load output of the detection head into the track_instances cache (inplace)
        """
        with torch.no_grad():
            track_scores = out['all_cls_scores'][-1, :].sigmoid().max(dim=-1).values
        track_instances.cache_scores = track_scores.clone()
        track_instances.cache_logits = out['all_cls_scores'][-1].clone()
        track_instances.cache_query_feats = out['query_feats'][-1].clone()
        track_instances.cache_ref_pts = out['ref_pts'].clone()
        track_instances.cache_bboxes = out['all_bbox_preds'][-1].clone()
        track_instances.cache_query_embeds = self.query_embedding(pos2posemb3d(track_instances.cache_ref_pts))
        return track_instances
    
    def frame_summarization(self, track_instances, tracking=False):
        """ Load the results after spatial-temporal reasoning into track instances
        """
        # inference mode
        if tracking:
            active_mask = (track_instances.cache_scores >= self.runtime_tracker.record_threshold)
            # print(f"update instance: {track_instances.obj_idxes[active_mask]}")
            # active_mask = (track_instances.cache_scores >= 0.0)
        # training mode
        else:
            track_instances.pred_boxes = track_instances.cache_bboxes.clone()
            track_instances.pred_logits = track_instances.cache_logits.clone()
            track_instances.scores = track_instances.cache_scores.clone()
            active_mask = (track_instances.cache_scores >= 0.0)

        track_instances.pred_logits[active_mask] = track_instances.cache_logits[active_mask]
        track_instances.scores[active_mask] = track_instances.cache_scores[active_mask]
        track_instances.pred_boxes[active_mask] = track_instances.cache_bboxes[active_mask]
        ref_pts = track_instances.ref_pts.clone()
        ref_pts[active_mask] = track_instances.cache_ref_pts[active_mask]
        track_instances.ref_pts = ref_pts
        query_embeds = track_instances.query_embeds.clone()
        query_embeds[active_mask] = track_instances.cache_query_embeds[active_mask]
        track_instances.query_embeds = query_embeds
        query_feats = track_instances.query_feats.clone()
        query_feats[active_mask] = track_instances.cache_query_feats[active_mask]
        track_instances.query_feats = query_feats
        track_instances.motion_predictions[active_mask] = track_instances.cache_motion_predictions[active_mask]

        if self.STReasoner.future_reasoning:
            motion_predictions = track_instances.motion_predictions[active_mask]
            track_instances.fut_xyz[active_mask] = track_instances.ref_pts[active_mask].clone()[:, None, :].repeat(1, self.fut_len, 1)
            track_instances.fut_bboxes[active_mask] = track_instances.pred_boxes[active_mask].clone()[:, None, :].repeat(1, self.fut_len, 1)
            motion_add = torch.cumsum(motion_predictions.clone().detach(), dim=1)
            motion_add_normalized = motion_add.clone()
            motion_add_normalized[..., 0] /= (self.pc_range[3] - self.pc_range[0])
            motion_add_normalized[..., 1] /= (self.pc_range[4] - self.pc_range[1])
            track_instances.fut_xyz[active_mask, :, 0] += motion_add_normalized[..., 0]
            track_instances.fut_xyz[active_mask, :, 1] += motion_add_normalized[..., 1]
            track_instances.fut_bboxes[active_mask, :, 0] += motion_add[..., 0]
            track_instances.fut_bboxes[active_mask, :, 1] += motion_add[..., 1]
        return track_instances
    
    def loss_single_batch(self, gt_bboxes_3d, gt_labels_3d, gt_inds, pred_dict):
        # init gt instances!
        gt_instances_list = []
        device = self.reference_points.weight.device
        gt_instances = Instances((1, 1))
        boxes = gt_bboxes_3d[0].tensor.to(device)
        # normalize gt bboxes here!
        boxes = normalize_bbox(boxes, self.pc_range)
        gt_instances.boxes = boxes
        gt_instances.labels = gt_labels_3d[0]
        gt_instances.obj_ids = gt_inds[0]
        gt_instances_list.append(gt_instances)
        self.criterion.initialize_for_single_clip(gt_instances_list)

        output_classes = pred_dict['all_cls_scores']
        output_coords = pred_dict['all_bbox_preds']
        track_instances = pred_dict['track_instances']
        with torch.no_grad():
            track_scores = output_classes[-1].sigmoid().max(dim=-1).values

        # Step-1 Update track instances with current prediction
        # [nb_dec, num_query, xxx]
        nb_dec = output_classes.size(0)

        # the track id will be assigned by the matcher.
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances) for i in range(nb_dec - 1)
        ]
        track_instances_list.append(track_instances)
        
        single_out = {}
        for i in range(nb_dec):
            track_instances_tmp = track_instances_list[i]

            track_instances_tmp.scores = track_scores
            track_instances_tmp.pred_logits = output_classes[i]  # [300, num_cls]
            track_instances_tmp.pred_boxes = output_coords[i]  # [300, box_dim]

            single_out["track_instances"] = track_instances_tmp
            track_instances_tmp, matched_indices = self.criterion.match_for_single_frame(
                single_out, i, if_step=(i == (nb_dec - 1))
            )
        
        return track_instances_tmp

    def forward_loss_prediction(self, 
                                active_track_instances,
                                gt_trajs,
                                gt_traj_masks,
                                instance_inds,
                                img_metas):
        active_gt_trajs, active_gt_traj_masks = list(), list()
        obj_idx_to_gt_idx = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(
            instance_inds.detach().cpu().numpy().tolist())}

        active_gt_trajs = torch.ones_like(active_track_instances.motion_predictions)
        active_gt_trajs[..., -1] = 0.0
        active_gt_traj_masks = torch.zeros_like(active_gt_trajs)[..., 0]

        for track_idx, id in enumerate(active_track_instances.obj_idxes):
            cpu_id = id.cpu().numpy().tolist()
            if cpu_id not in obj_idx_to_gt_idx.keys():
                continue
            index = obj_idx_to_gt_idx[cpu_id]
            traj = gt_trajs[index:index+1, :self.fut_len + 1, :]

            gt_motion = traj[:, torch.arange(1, self.fut_len + 1)] - traj[:, torch.arange(0, self.fut_len)]
            active_gt_trajs[track_idx: track_idx + 1] = gt_motion
            active_gt_traj_masks[track_idx: track_idx + 1] = \
                gt_traj_masks[index: index+1, 1: self.fut_len + 1] * gt_traj_masks[index: index+1, : self.fut_len]
        
        loss_dict = self.criterion.loss_prediction(active_gt_trajs[..., :2],
                                                   active_gt_traj_masks,
                                                   active_track_instances.cache_motion_predictions[..., :2])
        return loss_dict

    def update_reference_points(self, track_instances, time_delta=None, use_prediction=True, tracking=False):
        """Update the reference points according to the motion prediction/velocities
        """
        track_instances = self.STReasoner.update_reference_points(
            track_instances, time_delta, use_prediction, tracking)
        return track_instances
    
    def update_ego(self, track_instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2):
        """Update the ego coordinates for reference points, hist_xyz, and fut_xyz of the track_instances
           Modify the centers of the bboxes at the same time
        """
        track_instances = self.STReasoner.update_ego(track_instances, l2g_r1, l2g_t1, l2g_r2, l2g_t2)
        return track_instances
    
    @auto_fp16(apply_to=("img", "prev_bev"))
    def _forward_single_frame_train_bs(
        self,
        img,
        img_metas,
        track_instances,
        prev_img,
        prev_img_metas,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        prev_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        gt_inds=None,
        gt_forecasting_locs=None,
        gt_forecasting_masks=None,
        **kwargs,
    ):
        """
        Perform forward only on one frame. Called in  forward_train
        Warnning: Only Support BS=1
        Args:
            img: shape [B, num_cam, 3, H, W]
            if l2g_r2 is None or l2g_t2 is None:
                it means this frame is the end of the training clip,
                so no need to call velocity update
        """
        assert self.batch_size == len(track_instances)
        for i in range(self.batch_size):    
            prev_active_track_instances = track_instances[i]

            if prev_active_track_instances is None:
                track_instances[i] = self._generate_empty_tracks()
            else:
                prev_active_track_instances = self.update_reference_points(prev_active_track_instances,
                                                                                time_delta[i],
                                                                                use_prediction=self.motion_prediction_ref_update,
                                                                                tracking=False)
                if self.if_update_ego:
                    prev_active_track_instances = self.update_ego(prev_active_track_instances, 
                                                                l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i])
                prev_active_track_instances = self.STReasoner.sync_pos_embedding(prev_active_track_instances, self.query_embedding)
            
                empty_track_instances = self._generate_empty_tracks()
                full_length = len(empty_track_instances)
                active_length = len(prev_active_track_instances)
                if active_length > 0:
                    random_index = torch.randperm(full_length)
                    selected = random_index[:full_length-active_length]
                    empty_track_instances = empty_track_instances[selected]
                out_track_instances = Instances.cat([empty_track_instances, prev_active_track_instances])
                track_instances[i] = out_track_instances
            if self.shuffle:
                # shuffle the instances
                shuffle_index = torch.randperm(len(track_instances[i]))
                # print(len(shuffle_index))
                track_instances[i] = track_instances[i][shuffle_index]
        
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            query_feats=torch.stack([ins.query_feats for ins in track_instances]),
            query_embeds=torch.stack([ins.query_embeds for ins in track_instances]),
            ref_points=torch.stack([ins.ref_pts for ins in track_instances]),
            img_metas=img_metas,
        )

        output_classes = det_output["all_cls_scores"] # [num_layers, bs, num_query, num_cls]
        output_coords = det_output["all_bbox_preds"] #[num_layers, bs, num_query, num_dim]
        last_ref_pts = det_output["last_ref_points"] #[bs, num_query, 3]
        query_feats = det_output["query_feats"] #[num_layers, bs, num_query, embed_dims]

        losses = {}
        out = {
            'track_instances': [],
            'bev_embed': bev_embed,
            'bev_pos': bev_pos,
        }
        avg_factors = {}
        for j in range(self.batch_size):
            # 0. get infos of current batch
            cur_loss = dict()
            cur_track_instances = track_instances[j]
            cur_out = {
                'all_cls_scores': output_classes[:, j, :, :],
                'all_bbox_preds': output_coords[:, j, :, :],
                'ref_pts': last_ref_pts[j, :, :],
                'query_feats': query_feats[:, j, :, :],
            }
            # 1. Record the information into the track instances cache
            cur_track_instances = self.load_detection_output_into_cache(cur_track_instances, cur_out)
            cur_out['track_instances'] = cur_track_instances
            
            # 2. loss for detection
            if not self.is_cooperation:
                cur_track_instances = self.loss_single_batch(gt_bboxes_3d[j],
                                    gt_labels_3d[j],
                                    gt_inds[j],
                                    cur_out)
                cur_loss.update(self.criterion.losses_dict)
            # extract motion feature
            if self.is_motion:
                cur_track_instances = self.MotionExtractor(cur_track_instances, img_metas[j])

            inf_instances = None
            if self.is_cooperation:
                inf_dcit = {
                    'query_feats': kwargs['query_feats'][j][0],
                    'query_embeds': kwargs['query_embeds'][j][0],
                    'cache_motion_feats': kwargs['cache_motion_feats'][j][0],
                    'ref_pts': kwargs['ref_pts'][j][0],
                    'pred_boxes': kwargs['pred_boxes'][j][0],
                }
                if inf_dcit['query_feats'].shape[0] > 0:
                    inf_dcit = self.crossview_alignment(inf_dcit, veh2inf_rt[j])
                    inf_instances = self._init_inf_tracks(inf_dcit)
                if self.STReasoner.learn_match:
                    mask = cur_track_instances.cache_scores > self.STReasoner.veh_thre
                    veh_boxes = cur_track_instances[mask].cache_bboxes.clone()
                    inf_boxes = inf_instances.cache_bboxes.clone()
                    asso_label = self.STReasoner._gen_asso_label(gt_bboxes_3d[j], inf_boxes, veh_boxes, img_metas[j]['sample_idx'])
            # 3. Spatial-temporal reasoning
            cur_track_instances, affinity = self.STReasoner(cur_track_instances, inf_instances)

            if self.is_cooperation:
                cur_out = dict()
                cur_out = {
                'all_cls_scores': cur_track_instances.cache_logits[None, :, :],
                'all_bbox_preds': cur_track_instances.cache_bboxes[None, :, :],
                'track_instances': cur_track_instances
                }
                cur_track_instances = self.loss_single_batch(gt_bboxes_3d[j],
                                   gt_labels_3d[j],
                                   gt_inds[j],
                                   cur_out)
                prefix = 'fused_'
                cur_loss.update({prefix + key: value for key, value in self.criterion.losses_dict.items()})
                if self.STReasoner.learn_match:
                    # compute the affine loss (use Focal loss)
                    if affinity.shape[0] == 0 or affinity.shape[1] == 0 or torch.all(asso_label.eq(0)):
                        loss_focal = torch.tensor(0.0, requires_grad=True).to(affinity.device)
                    else:
                        affinity = affinity.view(-1, 1)
                        target = asso_label.view(-1, 1).float()
                        loss_focal = self.asso_loss_focal['loss_weight'] * sigmoid_focal_loss(
                                        affinity, target, alpha=self.asso_loss_focal['alpha'], 
                                        gamma=self.asso_loss_focal['gamma'], reduction='mean'
                                    )
                    cur_loss.update({'asso_loss': loss_focal,
                                     'asso_avg_factor': torch.tensor([1.0], device=affinity.device)})

            # if self.STReasoner.history_reasoning and not self.is_cooperation:
            if self.STReasoner.history_reasoning:
                loss_hist = self.criterion.loss_mem_bank(gt_bboxes_3d[j],
                                                         gt_labels_3d[j],
                                                         gt_inds[j],
                                                         cur_track_instances)
                cur_loss.update(loss_hist)
            if self.STReasoner.future_reasoning:
                active_mask = (cur_track_instances.obj_idxes >= 0)
                loss_fut = self.forward_loss_prediction(cur_track_instances[active_mask],
                                                        gt_forecasting_locs[j][0],
                                                        gt_forecasting_masks[j][0],
                                                        gt_inds[j][0],
                                                        img_metas[j])
                cur_loss.update(loss_fut)
            # 4. Prepare for next frame
            cur_track_instances = self.frame_summarization(cur_track_instances, tracking=False)
            active_mask = self.runtime_tracker.get_active_mask(cur_track_instances, training=True)
            cur_track_instances.track_query_mask[active_mask] = True
            active_track_instances = cur_track_instances[active_mask]
            # import pdb;pdb.set_trace()
            if self.random_drop > 0.0:
                active_track_instances = self._random_drop_tracks(active_track_instances)
            if self.fp_ratio > 0.0:
                active_track_instances = self._add_fp_tracks(cur_track_instances, active_track_instances)
            out['track_instances'].append(active_track_instances)
            for key, value in cur_loss.items():
                if 'loss' not in key:
                    continue
                af_key = key.replace('loss', 'avg_factor')
                avg_factor = cur_loss[af_key]
                if key not in losses:
                    losses[key] = value * avg_factor
                    avg_factors[af_key] = avg_factor
                else:
                    new_value = losses[key] + value * avg_factor
                    losses[key] = new_value
                    new_avg_factor = avg_factors[af_key] + avg_factor
                    avg_factors[af_key] = new_avg_factor
        
        for key, value in losses.items():
            af_key = key.replace('loss', 'avg_factor')
            avg_factor = avg_factors[af_key]
            losses[key] = value / avg_factor

        return out, losses

    def _random_drop_tracks(self, track_instances: Instances) -> Instances:
        drop_probability = self.random_drop
        if drop_probability > 0 and len(track_instances) > 0:
            keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
            track_instances = track_instances[keep_idxes]
        return track_instances
    
    def _add_fp_tracks(self, track_instances: Instances,
                       active_track_instances: Instances) -> Instances:
        """
        self.fp_ratio is used to control num(add_fp) / num(active)
        """
        inactive_instances = track_instances[track_instances.obj_idxes < 0]

        # add fp for each active track in a specific probability.
        fp_prob = torch.ones_like(
            active_track_instances.scores) * self.fp_ratio
        selected_active_track_instances = active_track_instances[
            torch.bernoulli(fp_prob).bool()]
        num_fp = len(selected_active_track_instances)

        if len(inactive_instances) > 0 and num_fp > 0:
            if num_fp >= len(inactive_instances):
                fp_track_instances = inactive_instances
            else:
                # randomly select num_fp from inactive_instances
                # fp_indexes = np.random.permutation(len(inactive_instances))
                # fp_indexes = fp_indexes[:num_fp]
                # fp_track_instances = inactive_instances[fp_indexes]

                # v2: select the fps with top scores rather than random selection
                fp_indexes = torch.argsort(inactive_instances.scores)[-num_fp:]
                fp_track_instances = inactive_instances[fp_indexes]

            merged_track_instances = Instances.cat(
                [active_track_instances, fp_track_instances])
            return merged_track_instances

        return active_track_instances
    
    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        # result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][result_dict['bbox_index']][result_dict['mask']]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][result_dict['bbox_index']][result_dict['mask']]
        return result_dict
    

    def forward_track_stream_train(self,
                                    img,
                                    gt_bboxes_3d,
                                    gt_labels_3d,
                                    gt_inds,
                                    gt_forecasting_locs,
                                    gt_forecasting_masks,
                                    l2g_t,
                                    l2g_r_mat,
                                    img_metas,
                                    timestamp,
                                    veh2inf_rt,
                                    **kwargs):
        """Forward funciton
        Args:
        Returns:
        """
        assert self.batch_size == img.size(0)
        # import ipdb;ipdb.set_trace()
        time_delta = [None] * self.batch_size
        l2g_r1 = [None] * self.batch_size
        l2g_t1 = [None] * self.batch_size
        l2g_r2 = [None] * self.batch_size
        l2g_t2 = [None] * self.batch_size
        for i in range(self.batch_size):
            tmp_pos = copy.deepcopy(img_metas[i][0]['can_bus'][:3])
            tmp_angle = copy.deepcopy(img_metas[i][0]['can_bus'][-1])
            if img_metas[i][0]['scene_token'] != self.train_prev_infos['scene_token'][i] or \
                timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i] > 0.5 or \
                timestamp[i][0] < self.train_prev_infos['prev_timestamp'][i]:
                # the first sample of each scene is truncated
                self.train_prev_infos['track_instances'][i] = None
                self.train_prev_infos['prev_bev'][i] = None
                time_delta[i], l2g_r1[i], l2g_t1[i], l2g_r2[i], l2g_t2[i] = None, None, None, None, None
                img_metas[i][0]['can_bus'][:3] = 0
                img_metas[i][0]['can_bus'][-1] = 0
            else:
                time_delta[i] = timestamp[i][0] - self.train_prev_infos['prev_timestamp'][i]
                assert time_delta[i] > 0
                l2g_r1[i] = self.train_prev_infos['l2g_r_mat'][i]
                l2g_t1[i] = self.train_prev_infos['l2g_t'][i]
                l2g_r2[i] = l2g_r_mat[i][0]
                l2g_t2[i] = l2g_t[i][0]
                img_metas[i][0]['can_bus'][:3] -= self.train_prev_infos['prev_pos'][i]
                img_metas[i][0]['can_bus'][-1] -= self.train_prev_infos['prev_angle'][i]
            
            # update prev_infos
            # timestamp[0][0]: the first 0 is batch, the second 0 is num_frame
            self.train_prev_infos['scene_token'][i] = img_metas[i][0]['scene_token']
            self.train_prev_infos['prev_timestamp'][i] = timestamp[i][0]
            self.train_prev_infos['l2g_r_mat'][i] = l2g_r_mat[i][0]
            self.train_prev_infos['l2g_t'][i] = l2g_t[i][0]
            self.train_prev_infos['prev_pos'][i] = tmp_pos
            self.train_prev_infos['prev_angle'][i] = tmp_angle

        prev_bev = torch.stack([bev if isinstance(bev, torch.Tensor) 
                                    else torch.zeros([self.pts_bbox_head.bev_h*self.pts_bbox_head.bev_w, self.pts_bbox_head.in_channels]).to(img.device)
                                    for bev in self.train_prev_infos['prev_bev']])
        if self.train_det:
            track_instances = [None for i in range(self.batch_size)]
        else:
            track_instances = self.train_prev_infos['track_instances']

        img_single = torch.stack([img_[0] for img_ in img], dim=0)
        img_metas_single = [copy.deepcopy(img_metas[i][0]) for i in range(self.batch_size)]
        frame_res, losses = self._forward_single_frame_train_bs(
            img_single,
            img_metas_single,
            track_instances,
            None, # prev_img
            None, # prev_img_metas
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            # all_query_embeddings,
            # all_matched_idxes,
            # all_instances_pred_logits,
            # all_instances_pred_boxes,
            veh2inf_rt,
            prev_bev=prev_bev,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            gt_forecasting_locs=gt_forecasting_locs,
            gt_forecasting_masks=gt_forecasting_masks,
            **kwargs,
        )
        track_instances = frame_res["track_instances"]

        bev_embed = frame_res['bev_embed'].detach().clone()
        for i in range(self.batch_size):
            self.train_prev_infos['prev_bev'][i] = bev_embed[:, i, :]
            self.train_prev_infos['track_instances'][i] = track_instances[i].detach_and_clone()
        return losses

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
        veh2inf_rt=None,
        sample_idx=None,
        **kwargs
    ):
        """
        img: B, num_cam, C, H, W = img.shape
        """
        prev_active_track_instances = track_instances
        if prev_active_track_instances is None:
            track_instances = self._generate_empty_tracks()
        else:
            prev_active_track_instances = self.update_reference_points(prev_active_track_instances,
                                                                       time_delta.type(torch.float),
                                                                       use_prediction=self.motion_prediction_ref_update,
                                                                       tracking=True)
            if self.if_update_ego:
                prev_active_track_instances = self.update_ego(prev_active_track_instances, 
                                                                l2g_r1[0], l2g_t1[0], l2g_r2[0], l2g_t2[0])
            prev_active_track_instances = self.STReasoner.sync_pos_embedding(prev_active_track_instances, self.query_embedding)
            
            empty_track_instances = self._generate_empty_tracks()
            full_length = len(empty_track_instances)
            active_length = len(prev_active_track_instances)
            if active_length > 0:
                random_index = torch.randperm(full_length)
                selected = random_index[:full_length-active_length]
                empty_track_instances = empty_track_instances[selected]
            out_track_instances = Instances.cat([empty_track_instances, prev_active_track_instances])
            track_instances = out_track_instances

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)

        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            query_feats=track_instances.query_feats[None, :, :],
            query_embeds=track_instances.query_embeds[None, :, :],
            ref_points=track_instances.ref_pts[None, :, :],
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"].clone()
        output_coords = det_output["all_bbox_preds"].clone()
        last_ref_pts = det_output["last_ref_points"].clone()
        query_feats = det_output["query_feats"].clone()

        out = {
            "all_cls_scores": output_classes[:, 0, :, :],
            "all_bbox_preds": output_coords[:, 0, :, :],
            "ref_pts": last_ref_pts[0, :, :],
            "query_feats": query_feats[:, 0, :, :],
        }

        track_instances = self.load_detection_output_into_cache(track_instances, out)
        out['track_instances'] = track_instances

        # extract motion features
        if self.is_motion:
            track_instances = self.MotionExtractor(track_instances, img_metas)
        
        inf_instances = None
        # import pdb;pdb.set_trace()
        if self.is_cooperation:
            inf_dcit = {
                'query_feats': kwargs['query_feats'][0][0],
                'query_embeds': kwargs['query_embeds'][0][0],
                'cache_motion_feats': kwargs['cache_motion_feats'][0][0],
                'ref_pts': kwargs['ref_pts'][0][0],
                'pred_boxes': kwargs['pred_boxes'][0][0],
            }
            if inf_dcit['query_feats'].shape[0] > 0:
                inf_dcit = self.crossview_alignment(inf_dcit, veh2inf_rt[0])
                inf_instances = self._init_inf_tracks(inf_dcit)
        # Spatial-temporal Reasoning
        track_instances, _ = self.STReasoner(track_instances, inf_instances, sample_idx)
        track_instances = self.frame_summarization(track_instances, tracking=True)
        out['all_cls_scores'][-1] = track_instances.pred_logits
        out['all_bbox_preds'][-1] = track_instances.pred_boxes

        if self.STReasoner.future_reasoning:
            # motion forecasting has the shape of [num_query, T, 2]
            out['all_motion_forecasting'] = track_instances.motion_predictions.clone()
        else:
            out['all_motion_forecasting'] = None

        # assign ids
        active_mask = (track_instances.scores > self.runtime_tracker.threshold)
        for i in range(len(track_instances)):
            if track_instances.obj_idxes[i] < 0:
                track_instances.obj_idxes[i] = self.runtime_tracker.current_id 
                self.runtime_tracker.current_id += 1
                if active_mask[i]:
                    track_instances.track_query_mask[i] = True
        out['track_instances'] = track_instances
        # output track results
        active_index = (track_instances.scores >= self.runtime_tracker.output_threshold)    # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))

        next_instances = self.runtime_tracker.update_active_tracks(track_instances, active_mask)
        out["track_instances"] = next_instances
        out.update(self._det_instances2results(out, img_metas))
        out["track_obj_idxes"] = track_instances.obj_idxes
        out["bev_embed"] = bev_embed
        return out
    
    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
        veh2inf_rt=None,
        **kwargs,
    ):
        """only support bs=1 and sequential input"""

        bs = img.size(0)
        tmp_pos = copy.deepcopy(img_metas[0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0]['can_bus'][-1])
        """ init track instances for first frame """
        if (
            self.test_track_instances is None
            or img_metas[0]["scene_token"] != self.scene_token
            or timestamp - self.timestamp > 0.5
            or timestamp < self.timestamp
        ):
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = None
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
            img_metas[0]['can_bus'][:3] = 0
            img_metas[0]['can_bus'][-1] = 0
            self.runtime_tracker.empty()
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t
            img_metas[0]['can_bus'][:3] -= self.prev_pos
            img_metas[0]['can_bus'][-1] -= self.prev_angle
            # print(time_delta, img_metas[0]['can_bus'][:3], img_metas[0]['can_bus'][-1])
        """ get time_delta and l2g r/t infos """
        """ update frame info for next frame"""
        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat
        self.prev_pos = tmp_pos
        self.prev_angle = tmp_angle

        """ predict and update """
        prev_bev = self.prev_bev
        if self.train_det:
            track_instances = None
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
            veh2inf_rt,
            img_metas[0]['sample_idx'],
            **kwargs
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]

        self.test_track_instances = track_instances
                
        results = [dict()]
        get_keys = ["track_bbox_results", "boxes_3d_det", "scores_3d_det", "labels_3d_det",
                    "boxes_3d", "scores_3d", "labels_3d", "track_scores", "track_ids"]
        results[0].update({k: frame_res[k] for k in get_keys})

        if self.save_track_query:
            tensor_to_cpu = torch.zeros(1)
            save_path = os.path.join(self.save_track_query_file_root, img_metas[0]['sample_idx'] +'.pkl')
            track_instances = track_instances.to(tensor_to_cpu)
            mmcv.dump(track_instances, save_path)

        return results
    
    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )
        bboxes_dict = self.pts_bbox_head.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[[bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]]
        )
        return result_dict

    def _det_instances2results(self, pred_dict, img_metas):
        """
        Outs:
        active_instances. keys:
        - 'pred_logits':
        - 'pred_boxes': normalized bboxes
        - 'scores'
        - 'obj_idxes'
        out_dict. keys:
            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
            - track_ids
            - tracking_score
        """
        # import pdb;pdb.set_trace()
        cls_score = pred_dict['all_cls_scores'].clone()
        bbox_preds = pred_dict['all_bbox_preds'].clone()
        scores = cls_score[-1].sigmoid().max(dim=-1).values
        obj_idxes = torch.ones_like(scores)
        bbox_dict = dict(
            cls_scores=cls_score[-1],
            bbox_preds=bbox_preds[-1],
            track_scores=scores,
            obj_idxes=obj_idxes,
        )
        bboxes_dict = self.pts_bbox_head.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )

        return result_dict_det

