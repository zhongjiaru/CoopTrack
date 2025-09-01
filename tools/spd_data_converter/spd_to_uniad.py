import mmcv
import numpy as np
import os
from collections import OrderedDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.prediction import PredictHelper
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.datasets import NuScenesDataset

import os.path as osp
import argparse
import json
import random
import string
from tqdm import tqdm
import uuid
from scipy.linalg import polar
from tools.spd_data_converter.spd_prediction_tools import get_forecasting_annotations

to_remove_list_veh = ['015389', '015390', '015391', '015392', '015393', '015394', '015395', '015396', '015397', '015398', '015399', '004394', '004395', '004396', '004397', '004398', '004399', 
             '004400', '004401', '004402', '004403', '004404', '004405', '004406', '004407', '004408', '004409', '004411', '004412', '004413', '002060', '002061', '002062', '002063', 
             '002064', '013006', '013083', '013084', '013085', '013086', '013087', '013088', '013089', '013090', '013091', '013092', '013093', '013094', '013155', '013156', '013157', 
             '013158', '013159', '013160', '013161', '013162', '013163', '013164', '013165', '013166', '013167', '013168', '013169', '013170', '013171', '013172', '013173', '013175',
             '013186', '013187', '013188', '013189', '013206', '013207', '013208', '013209', '012263', '012264', '012265', '012266', '012268', '012269', '012270', '012271', '012272', 
             '012273', '012274', '012275', '012276', '012277', '012278', '012279', '012280', '012281', '012282', '012283', '012284', '012285', '012286', '012287', '012288', '012289', '012290', 
             '003211', '003212', '003213',
             '003214', '003215', '003216', '003217', '003218', '003219', '003220', '003221', '003222', '003223', '003224', '003225', '003226', '003227', '003228',
             '003229', '003230', '004262', 
             '004263', '004264', '004265', '004266', '004381', '004382', '004383', '004386', '004388', '004389', '011118', '011119', '011120', '011121', '011122', '011123', '011124', 
             '011125', '011126', '011127', '011128', '011129', '011130', '011131', '011132', '011133', '011134', '011135', '011136', '011137', '011138', '011139', '011140', '011141', 
             '011142', '011143', '011144', '011145', '011146', '011147', '011148', '011149', '011150', '005596', '005597', '005598', '005599', '005600', '005601', '005602', '005603', 
             '005604', '005605', '005606', '005607', '005608', '005609', '005610', '004845', '004846', '004847', '004848', '004849', '004850', '004851', '004852', '006600', '006602', 
             '006603', '006604', '006605', '006606', '006607', '006608', '006609', '006610', '006611', '006612', '006613', '006614', '006615', '006616', '006617', '006618', '005538', 
             '005539', '005540', '005541', '005542', '005543', '005544', '005545', '014466', '014467', '014468', '008626', '008627', '008628', '008629', '008630', '008631', '008632', 
             '008633', '008634', '008757', '008758', '008759', '008760', '014551', '014552', '014553', '014554', '014555', '014556', '014570', '014571', '014572', '014573', '014574', 
             '014575', '014576', '014577', '006378', '006379', '006380', '006381', '006382', '006383', '006384', '006385', '006386', '006387', '006388', '006389', '006390', '006391', 
             '006392', '006393', '006394', '006395', '006396', '000368', '000369', '000370', '000372', '000373', '000375', '000377', '000378', '000379', '000380', '000381', '000382', 
             '000383', '000384', '000385', '000386', '000387', '000388', '000390', '000391', '000392', '000393', '000394', '000395', '000396', '000400', '000401', '003593', '003594', '003595', '003596', 
             '003597', '003598', '003599', '003600', '003601', '003602', '003603', '003604', '007457', '007458', '007459', '007460', '007461', '007462', '007463', '007464', '007465', 
             '007466', '007467', '007468', '007469', '007470', '007471', '007472', '007473', '007474', '007475', '007476', '007477', '007478', '007479', '007480', '007481', '007482', '009103', 
             '009104', '009105', '009106', '009107']

to_remove_list_coop = ['003211', '003212', '003213', '003214', '003215', '003216', '003217', '003218', '003219', '003220', '003221', '003222', '003223', '003224', '003225', '003226', '003227', '003228', '003229', '003230', 
 '004394', '004395', '004396', '004397', '004398', '004399', '004400', '004401', '004402', '004403', '004404', '004405', '004406', '004407', '004408', '004409', '004410', '004411', '004412', '004413', 
 '007457', '007458', '007459', '007460', '007461', '007462', '007463', '007464', '007465', '007466', '007467', '007468', '007469', '007470', '007471', '007472', '007473', '007474', '007475', '007476', '007477', '007478', '007479', '007480', '007481', '007482', 
 '013006', '013083', '013084', '013085', '013086', '013087', '013088', '013089', '013090', '013091', '013092', '013093', '013094', '013206', '013207', '013208', '013209']

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R


from math import pi

class Box3D():
    def __init__(self):
        self.center = None
        self.wlh = None
        self.orientation_yaw_pitch_roll = None
        self.name = None
        self.token = None
        self.instance_token = None
        self.track_id = None
        self.prev_token = None
        self.next_token = None
        self.timestamp = None
        self.visibility = None
        self.gt_velocity = None
        self.prev = None
        self.next = None


def get_cam_intr(calib_path):
    try:
        intr = np.array(load_json(calib_path)['P']).reshape(3, 4)[:, :3]
    except:
        intr = np.array(load_json(calib_path)['cam_K']).reshape(3, 3)

    return intr


def mul_matrix(rotation_1, translation_1, rotation_2, translation_2):
    rotation_1 = np.matrix(rotation_1)
    translation_1 = np.matrix(translation_1).reshape(3, 1)
    rotation_2 = np.matrix(rotation_2)
    translation_2 = np.matrix(translation_2).reshape(3, 1)

    rotation = rotation_2 * rotation_1
    translation = rotation_2 * translation_1 + translation_2
    rotation = np.array(rotation)
    translation = np.array(translation).reshape(3)

    return rotation, translation


visibility_mappings = {
    0: 4,
    1: 3,
    2: 2,
    3: 1
}

## UniV2X TODO: remapping
## UniV2X TODO: modify the related code in UniAD and nuScenes
class_names_nuscenes_mappings = {
    'Car': 'car',
    'Truck': 'car',
    'Van': 'car',
    'Bus': 'car',
    'Motorcyclist': 'bicycle',
    'Cyclist': 'bicycle',
    'Tricyclist': 'bicycle',
    'Barrowlist': 'bicycle',
    'Pedestrian': 'pedestrian',
    'TrafficCone': 'traffic_cone',
    'car': 'car',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'traffic_cone': 'traffic_cone'
}

def create_spd_infos_coop(root_path,
                          out_path,
                          v2x_side,
                          split_path,
                          can_bus_root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          split_part='train',
                          flag_save=True,
                          forecasting=False,
                          forecasting_length=13):
    """Create info file of spd dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'vehicle-side'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    out_path = osp.join(out_path, v2x_side)

    ## Step 0: load neccesary data
    coop_data_info_path = osp.join(root_path, 'cooperative/data_info.json')
    coop_data_infos = load_json(coop_data_info_path)
    # filter invalid frames (without vaild annos)
    coop_data_infos = [item for item in coop_data_infos if item['vehicle_frame'] not in to_remove_list_coop]
    print(len(coop_data_infos))

    veh_data_info_path = osp.join(root_path, 'vehicle-side/data_info.json')
    veh_data_infos = load_json(veh_data_info_path)

    inf_data_info_path = osp.join(root_path, 'infrastructure-side/data_info.json')
    inf_data_infos = load_json(inf_data_info_path)

    split_data_path = split_path    
    split_data = load_json(split_data_path)

    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    train_spd_infos = []
    val_spd_infos = []
    spd_infos = []

    ## Generate  sample_info_mappings, secene_frame_mappings, total_annotations, instance_token_mappings
    sample_infos, sample_info_mappings = _generate_sample_infos_coop(coop_data_infos,veh_data_infos,inf_data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations_coop(root_path,coop_data_infos,sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    #get lidar2ego info
    lidar_ego_global_infos = get_lidar_ego_global_infos(osp.join(root_path,'vehicle-side'),veh_data_infos,v2x_side)
    
    ## interpolate boxes for unvisible objects    
    ## UniV2X TODO: optimize the linear interpolation
    ## UniV2X TODO: visibility type??
    total_annotations =  _generate_unvisible_annotations("cooperative",sample_info_mappings,secene_frame_mappings,instance_token_mappings,total_annotations,lidar_ego_global_infos)

    ##update instance_token_mappings
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    ## add velocity and prev/next, update total_annotations and instance_token_mappings
    ## UniV2X TODO: optimize the velocity calculation
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos)

    for coop_data_info in tqdm(coop_data_infos):        
        ## Step 1: build basic information
        veh_frame_id = coop_data_info['vehicle_frame']
        inf_frame_id = coop_data_info['infrastructure_frame']
        
        sample_token = veh_frame_id
        sample_info = sample_info_mappings[sample_token]

        assert veh_frame_id == sample_info['token']
        assert inf_frame_id == sample_info['token_inf']

        info = {
            'token':sample_info['token'],
            'frame_idx': sample_info['frame_idx'], 
            'scene_token': sample_info['scene_token'],
            'location': sample_info['location'],
            'timestamp': sample_info['timestamp'],
            'prev': sample_info['prev'],
            'next': sample_info['next'],
            'token_inf':sample_info['token_inf'],
            'timestamp_inf': sample_info['timestamp_inf'],
            'system_error_offset':sample_info['system_error_offset']
        }

        veh_data_info = get_single_sample_info(veh_frame_id,veh_data_infos)
        inf_data_info = get_single_sample_info(inf_frame_id,inf_data_infos) 

        info['lidar_path'] = veh_data_info['pointcloud_path'].replace('pcd','bin')

        ##veh
        info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
        info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
        info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
        info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

        ##inf
        info['lidar2ego_rotation_inf'] = np.array(list(Quaternion(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))))
        info['lidar2ego_translation_inf'] = np.array([0, 0, 0]).reshape(3)

        calib_virtuallidar2global_path = osp.join(root_path, 'infrastructure-side', inf_data_info['calib_virtuallidar_to_world_path'])
        calib_virtuallidar2global = load_json(calib_virtuallidar2global_path)
        ## UniV2X TODO: check the matrix transformation with Quaternion

        virtuallidar2word_rotation = np.array(calib_virtuallidar2global['rotation'])
        approx_rotation_matrix = iterative_closest_point(virtuallidar2word_rotation)

        info['ego2global_rotation_inf'] = np.array(list(Quaternion(matrix=approx_rotation_matrix)))
        info['ego2global_translation_inf'] = np.array(calib_virtuallidar2global['translation']).reshape(3)

        ## UniV2X TODO: check the correction
        ##coop
        veh_l2e_r = np.array(Quaternion(info['lidar2ego_rotation']).rotation_matrix)
        veh_l2e_t = np.array(info['lidar2ego_translation']).reshape(3)
        veh_e2g_r = np.array(Quaternion(info['ego2global_rotation']).rotation_matrix)
        veh_e2g_t = np.array(info['ego2global_translation']).reshape(3)

        inf_e2g_r = np.array(calib_virtuallidar2global['rotation'])
        inf_e2g_t = np.array(calib_virtuallidar2global['translation']).reshape(3)
        inf_l2e_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        inf_l2e_t = np.array([0, 0, 0]).reshape(3)

        err_offset = np.array([sample_info['system_error_offset']['delta_x'], sample_info['system_error_offset']['delta_y'],0])
        r = ((veh_l2e_r.T @ veh_e2g_r.T) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
        t = (-err_offset @ veh_l2e_r.T @ veh_e2g_r.T + veh_l2e_t @ veh_e2g_r.T + veh_e2g_t) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
        t -= inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) + \
                inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T)
        
        info['VehLidar2InfLidar_rotation'], info['VehLidar2InfLidar_translation'] = r,t

        ##
        info['cams'] = {}
        camera_type = 'VEHICLE_CAM_FRONT'
        info['cams'][camera_type] = {}
        info['cams'][camera_type]['data_path'] = veh_data_info['image_path']

        calib_lidar2cam_path = osp.join(root_path, 'vehicle-side', veh_data_info['calib_lidar_to_camera_path'])
        calib_lidar2cam = load_json(calib_lidar2cam_path)
        info['cams'][camera_type]['lidar2cam_rotation'] = np.array(calib_lidar2cam['rotation'])
        info['cams'][camera_type]['lidar2cam_translation'] = np.array(calib_lidar2cam['translation'])

        ## UniV2X TODO: check the correction
        cam2lidar_r = np.linalg.inv(calib_lidar2cam['rotation'])
        cam2lidar_t = - np.array(calib_lidar2cam['translation']).reshape(1, 3) @ cam2lidar_r.T
        ## UniV2X TODO: check the matrix transformation with Quaternion
        info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
        info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(3)

        ## UniV2X TODO: check the correction
        cam2ego_r, cam2ego_t = mul_matrix(cam2lidar_r, cam2lidar_t,
                                                            Quaternion(info['lidar2ego_rotation']).rotation_matrix, np.array(info['lidar2ego_translation']))
        ## UniV2X TODO: check the matrix transformation with Quaternion
        info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
        info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3) 

        calib_cam_intrinsic_path = osp.join(root_path, 'vehicle-side', veh_data_info['calib_camera_intrinsic_path'])
        calib_cam_intrinsic = get_cam_intr(calib_cam_intrinsic_path)
        info['cams'][camera_type]['cam_intrinsic'] = calib_cam_intrinsic

        # UniV2X TODO: complete this part
        info['sweeps'] = {}
        info['can_bus'] = np.zeros(18)

        ## Step 3: build annotation information
        annotations = total_annotations[sample_token]

        boxes = []
        for anno_token in annotations.keys():
            annotation = annotations[anno_token]
            box3d = Box3D()
            box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'],
                                            annotation['3d_location']['z']]
            box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                                            annotation['3d_dimensions']['h']]
            box3d.orientation_yaw_pitch_roll = annotation['rotation']
            box3d.name = annotation['type']
            box3d.token = annotation['token']
            box3d.instance_token = annotation['instance_token']
            box3d.track_id = int(annotation['track_id'])
            box3d.timestamp = float(sample_info['timestamp'])
            box3d.visibility = visibility_mappings[annotation['occluded_state']]
            box3d.gt_velocity = annotation['gt_velocity']
            box3d.prev = annotation['prev']
            box3d.next = annotation['next']

            boxes.append(box3d)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll
                            for b in boxes]).reshape(-1, 1)
        
        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        names = np.array([b.name for b in boxes])
        instance_tokens = np.array([b.instance_token for b in boxes])
        instance_inds = np.array([b.track_id for b in boxes])
        box_tokens = np.array([b.token for b in boxes])
        timestamps = np.array([b.timestamp for b in boxes])
        visibility_tokens = np.array([b.visibility for b in boxes])
        gt_velocity = np.array([b.gt_velocity for b in boxes])
        prev_anno_tokens = np.array([b.prev for b in boxes])
        next_anno_tokens = np.array([b.next for b in boxes])

        # TODO: complete this part
        valid_flag = np.array([True for b in boxes])
        num_lidar_pts = np.array([1 for b in boxes])

        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_ins_tokens'] = instance_tokens
        info['gt_inds'] = instance_inds
        info['anno_tokens'] = box_tokens
        info['valid_flag'] = valid_flag
        info['num_lidar_pts'] = num_lidar_pts
        info['timestamps'] = timestamps
        info['visibility_tokens'] = visibility_tokens
        info['gt_velocity'] = gt_velocity
        info['prev_anno_tokens'] = prev_anno_tokens
        info['next_anno_tokens'] = next_anno_tokens

        if forecasting:
            fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(instance_token_mappings,
                                                                               lidar_ego_global_infos,
                                                                               annotations, 
                                                                               forecasting_length)
            locs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in fboxes]
            tokens = [np.array([b.token for b in boxes]) for boxes in fboxes]
            info['forecasting_locs'] = np.array(locs)
            info['forecasting_tokens'] = np.array(tokens)
            info['forecasting_masks'] = np.array(fmasks)
            info['forecasting_types'] = np.array(ftypes)
            
        ## Step X: save spd infos
        if veh_data_info['sequence_id'] in train_scenes:
            train_spd_infos.append(info)
        elif veh_data_info['sequence_id'] in val_scenes:
            val_spd_infos.append(info)
        spd_infos.append(info)

    if flag_save:
        metadata = dict(version=version)
        data = dict(infos=train_spd_infos, metadata=metadata)
        info_path = osp.join(out_path,
                            '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)

        data['infos'] = val_spd_infos
        info_val_path = osp.join(out_path,
                                '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)

    return total_annotations, sample_info_mappings, spd_infos

def get_lidar_ego_global_infos(root_path,data_infos,v2x_side):
    lidar_ego_global_infos = {}
    for data_info in tqdm(data_infos):
        ## Step 1: build basic information
        sample_token = data_info['frame_id']
        lidar_ego_global_infos[sample_token] = {}

        if v2x_side == 'infrastructure-side':
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(list(Quaternion(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))))
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array([0, 0, 0]).reshape(3)

            calib_virtuallidar2global_path = osp.join(root_path, data_info['calib_virtuallidar_to_world_path'])
            calib_virtuallidar2global = load_json(calib_virtuallidar2global_path)

            virtuallidar2word_rotation = np.array(calib_virtuallidar2global['rotation'])
            approx_rotation_matrix = iterative_closest_point(virtuallidar2word_rotation)

            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(list(Quaternion(matrix=approx_rotation_matrix)))
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(calib_virtuallidar2global['translation']).reshape(3)
        else:
            calib_lidar2ego_path = osp.join(root_path, data_info['calib_lidar_to_novatel_path'])
            calib_lidar2ego = load_json(calib_lidar2ego_path)
            lidar_ego_global_infos[sample_token]['lidar2ego_rotation'] = np.array(list(Quaternion(matrix=np.array(calib_lidar2ego['transform']['rotation']))))
            lidar_ego_global_infos[sample_token]['lidar2ego_translation'] = np.array(calib_lidar2ego['transform']['translation']).reshape(3)

            calib_ego2global_path = osp.join(root_path, data_info['calib_novatel_to_world_path'])
            calib_ego2global = load_json(calib_ego2global_path)
            lidar_ego_global_infos[sample_token]['ego2global_rotation'] = np.array(list(Quaternion(matrix=np.array(calib_ego2global['rotation']))))
            lidar_ego_global_infos[sample_token]['ego2global_translation'] = np.array(calib_ego2global['translation']).reshape(3)
    
    return lidar_ego_global_infos

def cal_ego_velocity(data_infos,sample_info_mappings,lidar_ego_global_infos):
    #{'sample_token': [vx,xy]}
    ego_velocity = {}
    for data_info in tqdm(data_infos):
        sample_token = data_info['frame_id']
        cur_loc = lidar_ego_global_infos[sample_token]['ego2global_translation']
        cur_timestamp = float(sample_info_mappings[sample_token]['timestamp']) / 1e6

        next_sample_token = sample_info_mappings[sample_token]['next']
        if next_sample_token != '':
            next_loc = lidar_ego_global_infos[next_sample_token]['ego2global_translation']
            next_timestamp = float(sample_info_mappings[next_sample_token]['timestamp']) / 1e6
            ego_velocity[sample_token] = (next_loc - cur_loc) / (next_timestamp - cur_timestamp)
        else:
            ego_velocity[sample_token] = [0,0]
    
    return ego_velocity


def create_spd_infos(root_path,
                     out_path,
                     v2x_side,
                     split_path,
                     can_bus_root_path,
                     info_prefix,
                     version='v1.0-trainval',
                     max_sweeps=10,
                     split_part='train',
                     flag_save=True,
                     forecasting=False,
                     forecasting_length=13):
    """Create info file of spd dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'vehicle-side'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    root_path = osp.join(root_path, v2x_side)
    out_path = osp.join(out_path, v2x_side)
    ## Step 0: load neccesary data
    data_info_path = osp.join(root_path, 'data_info.json')
    split_data_path = split_path

    data_infos = load_json(data_info_path)
    split_data = load_json(split_data_path)
    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']

    if v2x_side == 'vehicle-side':
        data_infos = [item for item in data_infos if item['frame_id'] not in to_remove_list_veh]
        print(len(data_infos))
    train_spd_infos = []
    val_spd_infos = []
    spd_infos = []

    ## Generate  sample_info_mappings, secene_frame_mappings, total_annotations, instance_token_mappings
    sample_infos, sample_info_mappings = _generate_sample_infos(data_infos)
    secene_frame_mappings = _get_secene_frame_mappings(sample_info_mappings)
    total_annotations = _get_total_annotations(root_path, data_infos, sample_info_mappings)
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    #get lidar2ego info
    lidar_ego_global_infos = get_lidar_ego_global_infos(root_path,data_infos,v2x_side)

    ## interpolate boxes for unvisible objects    
    ## UniV2X TODO: optimize the linear interpolation
    ## UniV2X TODO: visibility type??
    total_annotations = _generate_unvisible_annotations(v2x_side, sample_info_mappings, secene_frame_mappings,
                                                        instance_token_mappings, total_annotations, lidar_ego_global_infos)

    ##update instance_token_mappings
    instance_token_mappings = _get_instance_token_mappings(total_annotations, sample_info_mappings)

    # #cal ego_velocity
    # ego_velocity = cal_ego_velocity(data_infos,sample_info_mappings,lidar_ego_global_infos)

    ## add velocity and prev/next, update total_annotations and instance_token_mappings    
    ## UniV2X TODO: optimize the velocity calculation
    total_annotations, instance_token_mappings = _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos)

    #gen_infos
    camera_type = 'VEHICLE_CAM_FRONT'
    for data_info in tqdm(data_infos):
        ## Step 1: build basic information
        sample_token = data_info['frame_id']
        sample_info = sample_info_mappings[sample_token]

        info = {
            'token': sample_info['token'],
            'frame_idx': sample_info['frame_idx'],
            'scene_token': sample_info['scene_token'],
            'location': sample_info['location'],
            'timestamp': sample_info['timestamp'],
            'prev': sample_info['prev'],
            'next': sample_info['next'],
        }

        ## Step 2: build sensor data infomation
        info['lidar_path'] = data_info['pointcloud_path'].replace('pcd', 'bin')

        info['lidar2ego_rotation'] = lidar_ego_global_infos[sample_token]['lidar2ego_rotation']
        info['lidar2ego_translation'] = lidar_ego_global_infos[sample_token]['lidar2ego_translation']
        info['ego2global_rotation'] = lidar_ego_global_infos[sample_token]['ego2global_rotation']
        info['ego2global_translation'] = lidar_ego_global_infos[sample_token]['ego2global_translation']

        info['cams'] = {}
        info['cams'][camera_type] = {}
        info['cams'][camera_type]['data_path'] = data_info['image_path']

        key_calib_lidar2cam = 'calib_lidar_to_camera_path'
        if v2x_side == 'infrastructure-side':
            key_calib_lidar2cam = 'calib_virtuallidar_to_camera_path'                

        calib_lidar2cam_path = osp.join(root_path, data_info[key_calib_lidar2cam])
        calib_lidar2cam = load_json(calib_lidar2cam_path)
        info['cams'][camera_type]['lidar2cam_rotation'] = np.array(calib_lidar2cam['rotation'])
        info['cams'][camera_type]['lidar2cam_translation'] = np.array(calib_lidar2cam['translation'])

        ## UniV2X TODO: check the correction
        cam2lidar_r = np.linalg.inv(calib_lidar2cam['rotation'])
        cam2lidar_t = - np.array(calib_lidar2cam['translation']).reshape(1, 3) @ cam2lidar_r.T
        ## UniV2X TODO: check the matrix transformation with Quaternion
        info['cams'][camera_type]['sensor2lidar_rotation'] = cam2lidar_r
        info['cams'][camera_type]['sensor2lidar_translation'] = cam2lidar_t.reshape(3)

        cam2ego_r, cam2ego_t = mul_matrix(cam2lidar_r, cam2lidar_t,
                                        Quaternion(info['lidar2ego_rotation']).rotation_matrix,
                                        np.array(info['lidar2ego_translation']))
        info['cams'][camera_type]['sensor2ego_rotation'] = cam2ego_r
        info['cams'][camera_type]['sensor2ego_translation'] = cam2ego_t.reshape(3)        

        calib_cam_intrinsic_path = osp.join(root_path, data_info['calib_camera_intrinsic_path'])
        calib_cam_intrinsic = get_cam_intr(calib_cam_intrinsic_path)
        info['cams'][camera_type]['cam_intrinsic'] = calib_cam_intrinsic

        #
        info['sweeps'] = {}
        info['can_bus'] = np.zeros(18)

        ## Step 3: build annotation information
        annotations = total_annotations[sample_token]

        boxes = []
        for anno_token in annotations.keys():
            annotation = annotations[anno_token]
            box3d = Box3D()
            box3d.center = [annotation['3d_location']['x'], annotation['3d_location']['y'],
                            annotation['3d_location']['z']]
            box3d.wlh = [annotation['3d_dimensions']['w'], annotation['3d_dimensions']['l'],
                            annotation['3d_dimensions']['h']]
            box3d.orientation_yaw_pitch_roll = annotation['rotation']
            box3d.name = annotation['type']
            box3d.token = annotation['token']
            box3d.instance_token = annotation['instance_token']
            box3d.track_id = int(annotation['track_id'])
            box3d.timestamp = float(sample_info['timestamp'])
            box3d.visibility = visibility_mappings[annotation['occluded_state']]
            box3d.gt_velocity = annotation['gt_velocity']
            box3d.prev = annotation['prev']
            box3d.next = annotation['next']

            boxes.append(box3d)

        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array([b.orientation_yaw_pitch_roll
                            for b in boxes]).reshape(-1, 1)

        gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
        names = np.array([b.name for b in boxes])
        instance_tokens = np.array([b.instance_token for b in boxes])
        instance_inds = np.array([b.track_id for b in boxes])
        box_tokens = np.array([b.token for b in boxes])
        timestamps = np.array([b.timestamp for b in boxes])
        visibility_tokens = np.array([b.visibility for b in boxes])
        gt_velocity = np.array([b.gt_velocity for b in boxes])
        prev_anno_tokens = np.array([b.prev for b in boxes])
        next_anno_tokens = np.array([b.next for b in boxes])

        # TODO: complete this part
        valid_flag = np.array([True for b in boxes])
        num_lidar_pts = np.array([1 for b in boxes])

        info['gt_boxes'] = gt_boxes
        info['gt_names'] = names
        info['gt_ins_tokens'] = instance_tokens
        info['gt_inds'] = instance_inds
        info['anno_tokens'] = box_tokens
        info['valid_flag'] = valid_flag
        info['num_lidar_pts'] = num_lidar_pts
        info['timestamps'] = timestamps
        info['visibility_tokens'] = visibility_tokens
        info['gt_velocity'] = gt_velocity
        info['prev_anno_tokens'] = prev_anno_tokens
        info['next_anno_tokens'] = next_anno_tokens

        if forecasting:
            fboxes, fannotations, fmasks, ftypes = get_forecasting_annotations(instance_token_mappings,
                                                                               lidar_ego_global_infos,
                                                                               annotations, 
                                                                               forecasting_length)
            locs = [np.array([b.center for b in boxes]).reshape(-1, 3) for boxes in fboxes]
            tokens = [np.array([b.token for b in boxes]) for boxes in fboxes]
            info['forecasting_locs'] = np.array(locs)
            info['forecasting_tokens'] = np.array(tokens)
            info['forecasting_masks'] = np.array(fmasks)
            info['forecasting_types'] = np.array(ftypes)

        ## Step X: save spd infos
        if data_info['sequence_id'] in train_scenes:
            train_spd_infos.append(info)
        elif data_info['sequence_id'] in val_scenes:
            val_spd_infos.append(info)
        spd_infos.append(info)

    if flag_save:
        metadata = dict(version=version)
        data = dict(infos=train_spd_infos, metadata=metadata)
        info_path = osp.join(out_path,
                                '{}_infos_temporal_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)

        data['infos'] = val_spd_infos
        info_val_path = osp.join(out_path,
                                    '{}_infos_temporal_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


    return total_annotations, sample_info_mappings, spd_infos


def gen_token(*args):
    token_name = ''
    for value in args:
        token_name += str(value)
    token = uuid.uuid3(uuid.NAMESPACE_DNS, token_name)
    return str(token)


def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data


def write_json(data, path):
    with open(path, mode="w") as f:
        json.dump(data, f, indent=2)

def get_single_sample_info(frame_id, data_infos):
    sample_info = {}
    for data in data_infos:
        if data['frame_id'] == frame_id:
            sample_info = data
            break
    return sample_info

def _generate_sample_infos_coop(coop_data_infos,veh_data_infos,inf_data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    veh_sample_mappings = {}
    inf_sample_mappings = {}  
    coop_sample_mappings = {}  
    scene_data_dict = {}

    for data_info in coop_data_infos:
        veh_frame_id = data_info['vehicle_frame']
        inf_frame_id = data_info['infrastructure_frame']

        assert data_info['vehicle_sequence'] == data_info['infrastructure_sequence']

        veh_sample_mappings[veh_frame_id] = get_single_sample_info(veh_frame_id,veh_data_infos)
        inf_sample_mappings[inf_frame_id] = get_single_sample_info(inf_frame_id,inf_data_infos) 
        coop_sample_mappings[veh_frame_id] = data_info

        scene_token = data_info['vehicle_sequence']
        if scene_token not in scene_data_dict.keys():
            scene_data_dict[scene_token] = []

        scene_data_dict[scene_token].append(veh_frame_id)

    sample_infos = []
    for scene_token in scene_data_dict.keys():
        scene_data_dict[scene_token].sort()

        for idx in range(len(scene_data_dict[scene_token])):
            info = {}
            if idx == 0:
                info['prev'] = ''
            else:
                info['prev'] = scene_data_dict[scene_token][idx - 1]
            
            if idx == len(scene_data_dict[scene_token]) - 1:
                info['next'] = ''
            else:
                info['next'] = scene_data_dict[scene_token][idx + 1]
            
            veh_frame_id = scene_data_dict[scene_token][idx]
            inf_frame_id = coop_sample_mappings[veh_frame_id]['infrastructure_frame']

            veh_sample_info = veh_sample_mappings[veh_frame_id]
            inf_sample_info = inf_sample_mappings[inf_frame_id]
            coop_sample_info = coop_sample_mappings[veh_frame_id]

            info['token'] = veh_sample_info['frame_id']
            info['timestamp'] = float(veh_sample_info['pointcloud_timestamp'])
            info['image_timestamp'] = float(veh_sample_info['image_timestamp'])
            info['scene_token'] = veh_sample_info['sequence_id']
            info['location'] = veh_sample_info['intersection_loc']
            info['frame_idx'] = idx

            info['token_inf'] = inf_sample_info['frame_id']
            info['timestamp_inf'] = float(inf_sample_info['pointcloud_timestamp'])
            info['image_timestamp_inf'] = float(inf_sample_info['image_timestamp'])

            info['system_error_offset'] = coop_sample_info['system_error_offset']

            sample_infos.append(info)
    
    sample_info_mappings = {}
    for sample_info in sample_infos:
        sample_token = sample_info['token']
        sample_info_mappings[sample_token] = sample_info
    
    return sample_infos, sample_info_mappings

def _generate_sample_infos(data_infos):
    """Get the prev and next sample token for a given `sample_data_token`.
    Args:
        data_infos (list): data_infos loaded from data_info.json file.
    Return:
        list[dict]: List of sample info
        dict: mapping sample token to sample info   
    """
    sample_mappings = {}
    scene_data_dict = {}
    for data_info in data_infos:
        sample_token = data_info['frame_id']
        sample_mappings[sample_token] = data_info

        scene_token = data_info['sequence_id']
        if scene_token not in scene_data_dict.keys():
            scene_data_dict[scene_token] = []

        scene_data_dict[scene_token].append(sample_token)

    sample_infos = []
    for scene_token in scene_data_dict.keys():
        scene_data_dict[scene_token].sort()

        for idx in range(len(scene_data_dict[scene_token])):
            info = {}
            if idx == 0:
                info['prev'] = ''
            else:
                info['prev'] = scene_data_dict[scene_token][idx - 1]

            if idx == len(scene_data_dict[scene_token]) - 1:
                info['next'] = ''
            else:
                info['next'] = scene_data_dict[scene_token][idx + 1]

            sample_token = scene_data_dict[scene_token][idx]
            sample_info = sample_mappings[sample_token]
            info['token'] = sample_info['frame_id']
            info['timestamp'] = float(sample_info['pointcloud_timestamp'])
            info['image_timestamp'] = float(sample_info['image_timestamp'])
            info['scene_token'] = sample_info['sequence_id']
            info['location'] = sample_info['intersection_loc']
            info['frame_idx'] = idx

            sample_infos.append(info)

    sample_info_mappings = {}
    for sample_info in sample_infos:
        sample_token = sample_info['token']
        sample_info_mappings[sample_token] = sample_info

    return sample_infos, sample_info_mappings

def _get_total_annotations_coop(root_path,data_infos,sample_info_mappings):
    total_annotations = {}
    for data_info in data_infos:
        sample_token = data_info['vehicle_frame']
        scene_token = sample_info_mappings[sample_token]['scene_token']

        annotation_path = osp.join(root_path, 'cooperative/label', sample_token+'.json')
        annotations = load_json(annotation_path)
        total_annotations[sample_token] = {}
        for annotation in annotations:
            anno_token = annotation['token']
            annotation["instance_token"] = gen_token(annotation['track_id'],scene_token)
            annotation['type'] = class_names_nuscenes_mappings[annotation['type']]
            total_annotations[sample_token][anno_token] = annotation
    return total_annotations

def _get_total_annotations(root_path, data_infos, sample_info_mappings):
    total_annotations = {}
    for data_info in data_infos:
        sample_token = data_info['frame_id']
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        timestamp = sample_info_mappings[sample_token]['timestamp']

        annotation_path = osp.join(root_path, data_info['label_lidar_std_path'])
        annotations = load_json(annotation_path)
        total_annotations[sample_token] = {}
        for annotation in annotations:
            anno_token = annotation['token']
            annotation["instance_token"] = gen_token(annotation['track_id'], scene_token)
            annotation['type'] = class_names_nuscenes_mappings[annotation['type']]
            total_annotations[sample_token][anno_token] = annotation
    return total_annotations


def _generate_unvisible_annotations(source_name, sample_info_mappings, secene_frame_mappings, instance_token_mappings,
                                    total_annotations,lidar_ego_global_infos):
    """Generate annotations for totally occluded objects and make trajectory complete.
    Args:
        root_path: data root
        data_infos: (list): data_infos loaded from data_info.json file.
    Return:
        dict[dict]: {'sample_token': {'anno_token': }}
    """
    ## Interpolate box   
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        cur_scene_token_end = cur_instance_samples[-1]['scene_token']
        assert cur_scene_token == cur_scene_token_end

        for ii in range(len(cur_instance_samples) - 1):
            cur_frame_idx = cur_instance_samples[ii]['frame_idx'] + 1
            while cur_frame_idx != cur_instance_samples[ii + 1]['frame_idx']:
                # linear interpolation
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']
                rot_ii_0 = cur_instance_samples[ii]['annotation']['rotation']
                rot_ii_1 = cur_instance_samples[ii + 1]['annotation']['rotation']

                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']

                cur_sample_token = secene_frame_mappings[(cur_scene_token, cur_frame_idx)]
                # cur_time_stamp = sample_data_mappings[cur_sample_token]['pointcloud_timestamp']
                cur_timestamp = sample_info_mappings[cur_sample_token]['timestamp']

                # if cur_timestamp == '1626155888.384136':
                #     cur_timestamp = cur_timestamp

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']
                
                cur_loc = loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp,
                                                   lidar_ego_global_infos[sample_token_0],lidar_ego_global_infos[sample_token_1],
                                                   lidar_ego_global_infos[cur_sample_token])
                cur_rot = rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp)
                cur_anno_token = gen_token(source_name, cur_sample_token, str(cur_loc['x']), str(cur_loc['y']), str(cur_loc['z']))

                cur_instance_sample_anno = {
                    "token": cur_anno_token,
                    "type": cur_instance_samples[ii]['annotation']['type'],
                    "track_id": cur_instance_samples[ii]['annotation']['track_id'],
                    "truncated_state": 0,
                    "occluded_state": 3,
                    "3d_dimensions": cur_instance_samples[ii]['annotation']['3d_dimensions'],
                    "3d_location": cur_loc,
                    "rotation": cur_rot,
                    "instance_token": instance_token
                }

                total_annotations[cur_sample_token][cur_anno_token] = cur_instance_sample_anno

                cur_frame_idx = cur_frame_idx + 1

    return total_annotations


def loc_linear_interpolation(loc_ii_0, loc_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp, \
                             lidar_ego_global_info_0,lidar_ego_global_info_1,cur_lidar_ego_global_info):
    """Use linear interpolation to estimate the 3d location for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    #cvt to global
    center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
    # lidar2ego
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['lidar2ego_rotation']).rotation_matrix, center_0)   \
                        + np.array(lidar_ego_global_info_0['lidar2ego_translation'])
    # ego2global
    center_0 = np.dot(Quaternion(lidar_ego_global_info_0['ego2global_rotation']).rotation_matrix, center_0)  \
                        + np.array(lidar_ego_global_info_0['ego2global_translation'])

    #cvt to global
    center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])
    # lidar2ego
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['lidar2ego_rotation']).rotation_matrix, center_1)   \
                        + np.array(lidar_ego_global_info_1['lidar2ego_translation'])
    # ego2global
    center_1 = np.dot(Quaternion(lidar_ego_global_info_1['ego2global_rotation']).rotation_matrix, center_1)  \
                        + np.array(lidar_ego_global_info_1['ego2global_translation'])
    
    #global interpolation
    cur_center = (center_1 - center_0) / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + center_0

    #cur sesor data interpolation
    cur_ego2global_translation = cur_lidar_ego_global_info['ego2global_translation']
    cur_ego2global_rotation = Quaternion(cur_lidar_ego_global_info['ego2global_rotation'])

    # cur_ego2global_translation = (lidar_ego_global_info_1['ego2global_translation'] - lidar_ego_global_info_0['ego2global_translation'])  / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + lidar_ego_global_info_0['ego2global_translation']
    # yaw_0 = Quaternion(lidar_ego_global_info_0['ego2global_rotation']).yaw_pitch_roll[0]
    # yaw_1 = Quaternion(lidar_ego_global_info_1['ego2global_rotation']).yaw_pitch_roll[0]

    # diff = yaw_1 - yaw_0 + pi
    # if diff < 0:
    #     diff = diff + pi
    # elif diff > 2*pi:
    #     diff = diff -3*pi
    # else:
    #     diff = diff - pi

    # cur_yaw = diff / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + yaw_0 
    # if cur_yaw < -pi:
    #     cur_yaw = cur_yaw + 2*pi
    # if cur_yaw > 2*pi:
    #     cur_yaw = cur_yaw - 2*pi
    
    # cur_ego2global_rotation = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=cur_yaw)

    #global to ego, ego to lidar
    global2ego_r = np.linalg.inv(cur_ego2global_rotation.rotation_matrix)
    global2ego_t = - np.array(cur_ego2global_translation).reshape(1, 3) @ global2ego_r.T
    global2ego_t = global2ego_t.reshape(3)

    ego2lidar_r = np.linalg.inv(Quaternion(cur_lidar_ego_global_info['lidar2ego_rotation']).rotation_matrix)
    ego2lidar_t =  - np.array(cur_lidar_ego_global_info['lidar2ego_translation']).reshape(1, 3) @ ego2lidar_r.T
    ego2lidar_t = ego2lidar_t.reshape(3)

    # global2ego
    cur_center = np.dot(global2ego_r, cur_center) + global2ego_t 
    # ego2lidar
    cur_center = np.dot(ego2lidar_r, cur_center) + ego2lidar_t 
                       
    cur_loc = {}
    # for key in loc_ii_0.keys():
    #     cur_loc[key] = (loc_ii_1[key] - loc_ii_0[key]) / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + \
    #                    loc_ii_0[key]
    cur_loc['x'] = cur_center[0]
    cur_loc['y'] = cur_center[1]
    cur_loc['z'] = cur_center[2]

    return cur_loc


def rot_linear_interpolation(rot_ii_0, rot_ii_1, timestamp_ii_0, timestamp_ii_1, cur_timestamp):
    """Use linear interpolation to estimate the rotation for occluded objects.
    """
    timestamp_ii_0 = float(timestamp_ii_0) / 1e6
    timestamp_ii_1 = float(timestamp_ii_1) / 1e6
    cur_timestamp = float(cur_timestamp) / 1e6

    # cur_rot  = (rot_ii_1 - rot_ii_0) / (timestamp_ii_1 - timestamp_ii_0) * (cur_timestamp - timestamp_ii_0) + rot_ii_0

    time_ratio = (cur_timestamp - timestamp_ii_0) / (timestamp_ii_1 - timestamp_ii_0)
    diff = rot_ii_1 - rot_ii_0 + pi
    if diff < 0:
        diff = diff + pi
    elif diff > 2*pi:
        diff = diff -3*pi
    else:
        diff = diff - pi
    
    cur_rot = time_ratio * diff + rot_ii_0
    if cur_rot < -pi:
        cur_rot = cur_rot + 2*pi
    if cur_rot > 2*pi:
        cur_rot = cur_rot - 2*pi
    
    return cur_rot

def _add_annotation_velocity_prev_next(total_annotations, instance_token_mappings, lidar_ego_global_infos):
    """Generate velocity and prev/next token for annotations.
    Args:
        total_annotations: added occluded annotations
        sample_info_mappings
        data_infos
    """
    ## Generate Velocity and Successors
    for instance_token in instance_token_mappings.keys():
        cur_instance_samples = instance_token_mappings[instance_token]
        cur_scene_token = cur_instance_samples[0]['scene_token']
        
        for ii in range(len(cur_instance_samples)):
            if ii == 0:
                prev_anno_token = ''
            else:
                prev_anno_token = cur_instance_samples[ii - 1]['annotation']['token']

            if ii == len(cur_instance_samples) - 1:
                next_anno_token = ''
            else:
                next_anno_token = cur_instance_samples[ii + 1]['annotation']['token']

            if ii == len(cur_instance_samples) - 1:
                gt_velocity = np.array([0, 0])
            else:
                loc_ii_0 = cur_instance_samples[ii]['annotation']['3d_location']
                loc_ii_1 = cur_instance_samples[ii + 1]['annotation']['3d_location']

                sample_token_0 = cur_instance_samples[ii]['sample_token']
                sample_token_1 = cur_instance_samples[ii+1]['sample_token']

                #cvt to global
                center_0 = np.array([loc_ii_0['x'], loc_ii_0['y'], loc_ii_0['z']])
                center_1 = np.array([loc_ii_1['x'], loc_ii_1['y'], loc_ii_1['z']])

                # lidar2ego
                # center_0 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix, center_0)   \
                #                     + np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation'])
                center_1 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_1]['lidar2ego_rotation']).rotation_matrix, center_1)   \
                                    + np.array(lidar_ego_global_infos[sample_token_1]['lidar2ego_translation'])

                # ego2global
                # center_0 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix, center_0)  \
                #                     + np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation'])
                center_1 = np.dot(Quaternion(lidar_ego_global_infos[sample_token_1]['ego2global_rotation']).rotation_matrix, center_1)  \
                                    + np.array(lidar_ego_global_infos[sample_token_1]['ego2global_translation']) 
                
                global2ego_r0 = np.linalg.inv(Quaternion(lidar_ego_global_infos[sample_token_0]['ego2global_rotation']).rotation_matrix)
                global2ego_t0 = - np.array(lidar_ego_global_infos[sample_token_0]['ego2global_translation']).reshape(1, 3) @ global2ego_r0.T
                global2ego_t0 = global2ego_t0.reshape(3)

                ego2lidar_r0 = np.linalg.inv(Quaternion(lidar_ego_global_infos[sample_token_0]['lidar2ego_rotation']).rotation_matrix)
                ego2lidar_t0 = - np.array(lidar_ego_global_infos[sample_token_0]['lidar2ego_translation']).reshape(1, 3) @ ego2lidar_r0.T
                ego2lidar_t0 = ego2lidar_t0.reshape(3)                

                #ego2lidar
                center_1 = np.dot(global2ego_r0,center_1) + global2ego_t0  
                center_1 = np.dot(ego2lidar_r0,center_1) + ego2lidar_t0                         

                #time_delta
                timestamp_ii_0 = cur_instance_samples[ii]['timestamp']
                timestamp_ii_1 = cur_instance_samples[ii + 1]['timestamp']
                timestamp_ii_0 = float(timestamp_ii_0) / 1e6
                timestamp_ii_1 = float(timestamp_ii_1) / 1e6
                time_delta = timestamp_ii_1 - timestamp_ii_0

                # gt_velocity_dict = {}
                # for key in loc_ii_0.keys():
                #     gt_velocity_dict[key] = (loc_ii_1[key] - loc_ii_0[key]) / (timestamp_ii_1 - timestamp_ii_0)
                # gt_velocity = [gt_velocity_dict['x'], gt_velocity_dict['y']]
                gt_velocity = (center_1 - center_0) / time_delta
                gt_velocity = gt_velocity[:2]

            instance_token_mappings[instance_token][ii]['annotation']['gt_velocity'] = gt_velocity
            instance_token_mappings[instance_token][ii]['annotation']['prev'] = prev_anno_token
            instance_token_mappings[instance_token][ii]['annotation']['next'] = next_anno_token

    return total_annotations, instance_token_mappings

def _get_secene_frame_mappings(sample_info_mappings):
    secene_frame_mappings = {}
    for sample_token in sample_info_mappings.keys():
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        secene_frame_mappings[(scene_token, frame_idx)] = sample_token

    return secene_frame_mappings


def _get_instance_token_mappings(total_annotations, sample_info_mappings):
    instance_token_mappings = {}

    for sample_token in total_annotations.keys():
        annotations = total_annotations[sample_token]
        scene_token = sample_info_mappings[sample_token]['scene_token']
        frame_idx = sample_info_mappings[sample_token]['frame_idx']
        timestamp = sample_info_mappings[sample_token]['timestamp']

        for anno_token in annotations.keys():
            annotation = annotations[anno_token]
            instance_token = annotation["instance_token"]
            if instance_token not in instance_token_mappings.keys():
                instance_token_mappings[instance_token] = []
            instance_token_mappings[instance_token].append({
                'scene_token': scene_token,
                'frame_idx': frame_idx,
                'sample_token': sample_token,
                'timestamp': timestamp,
                'annotation': annotation})

    # sorted by frame_idx, for downstream usage
    for instance_token in instance_token_mappings.keys():
        sorted(instance_token_mappings[instance_token], key=lambda annotation: annotation['frame_idx'])

    return instance_token_mappings


def generate_json_maps_files(data_root, version='v1.0-mini'):
    json_types = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                  'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

    # if not os.path.exists(osp.join(data_root, version)):
    #     os.mkdir(osp.join(data_root, version))

    # for json_type in json_types:
    #     json_file_data = []
    #     json_file_path = osp.join(data_root, version, json_type+'.json')
    #     write_json(json_file_data, json_file_path)

    import shutil
    if not os.path.exists(osp.join(data_root, version)):
        tmp_nuscenes_json_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/v1.0-mini'
        shutil.copytree(tmp_nuscenes_json_root, osp.join(data_root, version))

    if not os.path.exists(osp.join(data_root, 'maps')):
        tmp_nuscenes_map_root = '/data/ad_sharing/datasets/nuScenes/nuScenes_v1.0-mini/maps'
        shutil.copytree(tmp_nuscenes_map_root, osp.join(data_root, 'maps'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--save-root', type=str, default="./data/infos/V2X-Seq-SPD-Example")
    parser.add_argument('--split-file', type=str, default="./data/split_datas/cooperative-split-data-spd.json")
    parser.add_argument('--v2x-side', type=str, default="vehicle-side")
    parser.add_argument('--version', type=str, default="v1.0-trainval")
    parser.add_argument('--info-prefix', type=str, default="spd")
    parser.add_argument('--forecasting', action='store_true', default=False, help='prepare trajectory forecasting data')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    v2x_side = args.v2x_side
    data_root = args.data_root
    save_root = args.save_root
    split_path = args.split_file
    can_bus_root_path = ''
    info_prefix = args.info_prefix
    forecasting = args.forecasting

    print(data_root)
    print(save_root)
    print(v2x_side)
    if forecasting:
        print('prepare trajectory forecasting data')
    else:
        print('not prepare trajectory forecasting data')

    if v2x_side == 'cooperative':
        # generate_json_maps_files(data_root, version=v2x_side)
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos_coop(data_root,
                            save_root,
                            v2x_side,
                            split_path, 
                            can_bus_root_path,
                            info_prefix,
                            version=args.version,
                            max_sweeps=10,
                            forecasting=forecasting)     
    else:   
        # generate_json_maps_files(data_root, version=v2x_side)
        total_annotations, sample_info_mappings, spd_infos = create_spd_infos(data_root,
                                                                            save_root,
                                                                            v2x_side,
                                                                            split_path,
                                                                            can_bus_root_path,
                                                                            info_prefix,
                                                                            version=args.version,
                                                                            max_sweeps=10,
                                                                            forecasting=forecasting)

