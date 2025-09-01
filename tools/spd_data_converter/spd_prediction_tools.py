# ------------------------------------------------------------------------
# Copyright (c) 2023 toyota research instutute.
# ------------------------------------------------------------------------
# Modified from FutureDet (https://github.com/neeharperi/FutureDet)
# ------------------------------------------------------------------------
import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from itertools import tee
from copy import deepcopy


def get_forecasting_annotations(instance_token_mappings,
                                lidar_ego_global_infos,
                                annotations,
                                forecasting_length):
    """Acquire the trajectories for each box
        instance_token_mappings: dict, instance_token: list()
        lidar_ego_global_infos: dict, sample_token: dict
        annotations: contains certain frame's annotations, i.e. many boxes
    """
    forecast_annotations = []
    forecast_boxes = []   
    forecast_trajectory_type = []
    forecast_visibility_mask = []
    
    # loop every instance
    for t, annotation in annotations.items():
        tracklet_box = []
        tracklet_annotation = []
        tracklet_visiblity_mask = []
        tracklet_trajectory_type = []
        timestamps = []

        instance_token = annotation['instance_token']
        instance_anno_list = instance_token_mappings[instance_token]
        anno_token_mappings = {a['annotation']['token']: a for a in instance_anno_list}

        visibility = True
        for step in range(forecasting_length):
            anno_token = annotation['token']
            sample_token = anno_token_mappings[anno_token]['sample_token']

            box = Box(center = [annotation["3d_location"]['x'], annotation["3d_location"]['y'],
                                annotation["3d_location"]['z']],
                      size = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                              annotation['3d_dimensions']['h']],
                      orientation = Quaternion(axis=[0, 0, 1], radians=annotation['rotation']),
                      velocity = annotation['gt_velocity'].tolist() + [0], # x, y, and pad 0 for z
                      name = annotation["type"],
                      token = annotation["token"])
            
            if step == 0:
                # the first frame
                tgt_lidar2ego_rot = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
                tgt_lidar2ego_trans = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
                tgt_ego2global_rot = lidar_ego_global_infos[sample_token]['ego2global_rotation']       
                tgt_ego2global_trans = lidar_ego_global_infos[sample_token]['ego2global_translation']
            else:
                # subsequent frames
                src_lidar2ego_rot = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
                src_lidar2ego_trans = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
                src_ego2global_rot = lidar_ego_global_infos[sample_token]['ego2global_rotation']       
                src_ego2global_trans = lidar_ego_global_infos[sample_token]['ego2global_translation']
                
                # move to tgt frame
                box.rotate(Quaternion(src_lidar2ego_rot.tolist()))
                box.translate(src_lidar2ego_trans)
                box.rotate(Quaternion(src_ego2global_rot.tolist()))
                box.translate(src_ego2global_trans)

                box.translate(-tgt_ego2global_trans)
                box.rotate(Quaternion(tgt_ego2global_rot.tolist()).inverse)
                box.translate(-tgt_lidar2ego_trans)
                box.rotate(Quaternion(tgt_lidar2ego_rot.tolist()).inverse)

            tracklet_box.append(box)
            tracklet_annotation.append(annotation)
            tracklet_visiblity_mask.append(visibility)
            timestamps.append(anno_token_mappings[anno_token]['timestamp'])

            next_token = annotation['next']
            if next_token != '':
                annotation = anno_token_mappings[next_token]['annotation']
            else:
                # if the trajectory cannot be prolonged anymore,
                # use the last one to pad and set the visibility flag
                annotation = annotation
                visibility = False

        time = [get_time(src, dst) for src, dst in window(timestamps, 2)]
        tracklet_trajectory_type = trajectory_type(tracklet_box, time, forecasting_length) # same as FutureDet

        forecast_boxes.append(tracklet_box)
        forecast_annotations.append(tracklet_annotation)
        forecast_trajectory_type.append(forecasting_length * [tracklet_trajectory_type])
        forecast_visibility_mask.append(tracklet_visiblity_mask)
    return forecast_boxes, forecast_annotations, forecast_visibility_mask, forecast_trajectory_type


def window(iterable, size):
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)

    return zip(*iters)

def get_time(src_time, dst_time):
    time_last = 1e-6 * src_time
    time_first = 1e-6 * dst_time
    time_diff = time_first - time_last

    return time_diff 


def center_distance(gt_box, pred_box) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.center[:2]) - np.array(gt_box.center[:2]))


def trajectory_type(boxes, time, timesteps=7, past=False):
    target = boxes[-1]
    
    static_forecast = deepcopy(boxes[0])

    linear_forecast = deepcopy(boxes[0])
    vel = linear_forecast.velocity[:2]
    disp = np.sum(list(map(lambda x: np.array(list(vel) + [0]) * x, time)), axis=0)

    if past:
        linear_forecast.center = linear_forecast.center - disp

    else:
        linear_forecast.center = linear_forecast.center + disp
    
    if center_distance(target, static_forecast) < max(target.wlh[0], target.wlh[1]):
        # return "static"
        return 0

    elif center_distance(target, linear_forecast) < max(target.wlh[0], target.wlh[1]):
        # return "linear"
        return 1

    else:
        # return "nonlinear"
        return 2