import mmcv
import argparse
import os
import glob
import cv2
import copy
from nuscenes.nuscenes import NuScenes
from PIL import Image
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from typing import Tuple, List, Iterable
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.eval.common.data_classes import EvalBoxes, EvalBox
# from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.tracking.data_classes import TrackingBox, TrackingConfig
from nuscenes.eval.detection.utils import category_to_detection_name
from transform_box_veh2inf import veh2inf_convert, read_json


cams = ['CAM_FRONT',
 'CAM_FRONT_RIGHT',
 'CAM_BACK_RIGHT',
 'CAM_BACK',
 'CAM_BACK_LEFT',
 'CAM_FRONT_LEFT']

color_mapping = [
    np.array([1.0, 0.0, 0.0]),   # 鲜艳的红色
    np.array([1.0, 0.078, 0.576]), # 鲜艳的粉色
    np.array([0.0, 0.0, 1.0]),   # 鲜艳的蓝色
    np.array([1.0, 1.0, 0.0]),   # 鲜艳的黄色
    np.array([1.0, 0.647, 0.0]), # 鲜艳的橙色
    np.array([0.502, 0.0, 0.502]), # 鲜艳的紫色
    np.array([0.0, 1.0, 1.0]),   # 鲜艳的青色
    np.array([1.0, 0.0, 1.0]),   # 鲜艳的洋红色
    np.array([0.0, 1.0, 0.502]), # 鲜艳的青绿色
    np.array([1.0, 0.843, 0.0])  # 鲜艳的金色
]

import numpy as np
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from matplotlib import rcParams

from nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes.utils.data_classes import PointCloud
from scipy.linalg import polar

class CustomLidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str) -> 'CustomLidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = np.fromfile(file_name, dtype=np.float32)
        points = scan.reshape((-1, 4))[:, :cls.nbr_dims()]
        return cls(points.T)

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R

def visualize_sample(nusc: NuScenes,
                     sample_token: str,
                     gt_boxes: EvalBoxes,
                     pred_boxes: EvalBoxes,
                     nsweeps: int = 1,
                     conf_th: float = 0.15,
                     eval_range: list = [-53.0, -53.0, 53.0, 53.0],
                     verbose: bool = True,
                     savepath: str = None,
                     ax=None) -> None:
    """
    Visualizes a sample from BEV with annotations and detection results.
    :param nusc: NuScenes object.
    :param sample_token: The nuScenes sample token.
    :param gt_boxes: Ground truth boxes grouped by sample.
    :param pred_boxes: Prediction grouped by sample.
    :param nsweeps: Number of sweeps used for lidar visualization.
    :param conf_th: The confidence threshold used to filter negatives.
    :param eval_range: Range in meters beyond which boxes are ignored.
    :param verbose: Whether to print to stdout.
    :param savepath: If given, saves the the rendering here instead of displaying.
    """
    # Retrieve sensor & pose records.
    sample_rec = nusc.get('sample', sample_token)
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    # Get boxes.
    boxes_gt_global = gt_boxes[sample_token]
    boxes_est_global = pred_boxes[sample_token]

    # Map GT boxes to lidar.
    boxes_gt = boxes_to_sensor(boxes_gt_global, pose_record, cs_record)

    # Map EST boxes to lidar.
    boxes_est = boxes_to_sensor(boxes_est_global, pose_record, cs_record)

    # Add scores to EST boxes.
    for box_est, box_est_global in zip(boxes_est, boxes_est_global):
        box_est.score = box_est_global.tracking_score
        box_est.tracking_id = box_est_global.tracking_id

    # Get point cloud in lidar frame.
    # pc, _ = CustomLidarPointCloud.from_file_multisweep(nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps)

    # Init axes.
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(9, 9))

    # Show point cloud.
    # points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
    # dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
    # colors = np.minimum(1, dists / eval_range[2])
    # ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)

    # Show ego vehicle.
    ax.plot(0, 0, 'x', color='black')

    # Show GT boxes.
    for box in boxes_gt:
        box.render(ax, view=np.eye(4), colors=('g', 'g', 'g'), linewidth=1)

    # Show EST boxes.
    for box in boxes_est:
        # Show only predictions with a high score.
        assert not np.isnan(box.score), 'Error: Box score cannot be NaN!'
        if box.score >= conf_th:
            c = 'r'
            if hasattr(box, 'tracking_id'): # this is true
                tr_id = box.tracking_id
                c = color_mapping[tr_id % len(color_mapping)]
            box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=1)

    # Limit visible range.
    # axes_limit = eval_range + 3  # Slightly bigger to include boxes that extend beyond the range.
    ax.set_xlim(eval_range[0], eval_range[2])
    ax.set_ylim(eval_range[1], eval_range[3])

    # Show / save plot.
    if verbose:
        print('Rendering sample token %s' % sample_token)
    # plt.title(sample_token)
    if eval_range[0] == -53.0:
        ax.set_xticks([-40, -20, 0, 20, 40])
        ax.set_xticklabels(['-40', '-20', '0', '20', '40',])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_yticklabels(['-40', '-20', '0', '20', '40'])
    else:
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_xticklabels(['20', '40', '60', '80', '100',])
        ax.set_yticks([-40, -20, 0, 20, 40])
        ax.set_yticklabels(['-40', '-20', '0', '20', '40'])
    ax.set_aspect('equal')
    if savepath is not None:
        savepath = savepath + '_bev'
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    # else:
    #     plt.show()

def render_annotation(
        anntoken: str,
        margin: float = 10,
        view: np.ndarray = np.eye(4),
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        out_path: str = 'render.png',
        extra_info: bool = False) -> None:
    """
    Render selected annotation.
    :param anntoken: Sample_annotation token.
    :param margin: How many meters in each direction to include in LIDAR view.
    :param view: LIDAR view point.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param out_path: Optional path to save the rendered figure to disk.
    :param extra_info: Whether to render extra information below camera view.
    """
    ann_record = nusc.get('sample_annotation', anntoken)
    sample_record = nusc.get('sample', ann_record['sample_token'])
    assert 'LIDAR_TOP' in sample_record['data'].keys(), 'Error: No LIDAR_TOP in data, unable to render.'

    # Figure out which camera the object is fully visible in (this may return nothing).
    boxes, cam = [], []
    cams = [key for key in sample_record['data'].keys() if 'CAM' in key]
    all_bboxes = []
    select_cams = []
    for cam in cams:
        _, boxes, _ = nusc.get_sample_data(sample_record['data'][cam], box_vis_level=box_vis_level,
                                           selected_anntokens=[anntoken])
        if len(boxes) > 0:
            all_bboxes.append(boxes)
            select_cams.append(cam)
            # We found an image that matches. Let's abort.
    # assert len(boxes) > 0, 'Error: Could not find image where annotation is visible. ' \
    #                      'Try using e.g. BoxVisibility.ANY.'
    # assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'

    num_cam = len(all_bboxes)

    fig, axes = plt.subplots(1, num_cam + 1, figsize=(18, 9))
    select_cams = [sample_record['data'][cam] for cam in select_cams]
    print('bbox in cams:', select_cams)
    # Plot LIDAR view.
    lidar = sample_record['data']['LIDAR_TOP']
    data_path, boxes, camera_intrinsic = nusc.get_sample_data(lidar, selected_anntokens=[anntoken])
    CustomLidarPointCloud.from_file(data_path).render_height(axes[0], view=view)
    for box in boxes:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=view, colors=(c, c, c))
        corners = view_points(boxes[0].corners(), view, False)[:2, :]
        axes[0].set_xlim([np.min(corners[0, :]) - margin, np.max(corners[0, :]) + margin])
        axes[0].set_ylim([np.min(corners[1, :]) - margin, np.max(corners[1, :]) + margin])
        axes[0].axis('off')
        axes[0].set_aspect('equal')

    # Plot CAMERA view.
    for i in range(1, num_cam + 1):
        cam = select_cams[i - 1]
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(cam, selected_anntokens=[anntoken])
        im = Image.open(data_path)
        axes[i].imshow(im)
        axes[i].set_title(nusc.get('sample_data', cam)['channel'])
        axes[i].axis('off')
        axes[i].set_aspect('equal')
        for box in boxes:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[i], view=camera_intrinsic, normalize=True, colors=(c, c, c))

        # Print extra information about the annotation below the camera view.
        axes[i].set_xlim(0, im.size[0])
        axes[i].set_ylim(im.size[1], 0)

    if extra_info:
        rcParams['font.family'] = 'monospace'

        w, l, h = ann_record['size']
        category = ann_record['category_name']
        lidar_points = ann_record['num_lidar_pts']
        radar_points = ann_record['num_radar_pts']

        sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_record['translation']))

        information = ' \n'.join(['category: {}'.format(category),
                                  '',
                                  '# lidar points: {0:>4}'.format(lidar_points),
                                  '# radar points: {0:>4}'.format(radar_points),
                                  '',
                                  'distance: {:>7.3f}m'.format(dist),
                                  '',
                                  'width:  {:>7.3f}m'.format(w),
                                  'length: {:>7.3f}m'.format(l),
                                  'height: {:>7.3f}m'.format(h)])

        plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

    if out_path is not None:
        plt.savefig(out_path)



def get_sample_data(sample_data_token: str,
                    box_vis_level: BoxVisibility = BoxVisibility.ANY,
                    selected_anntokens=None,
                    use_flat_vehicle_coordinates: bool = False,
                    boxes = None,
                    side='vehicle-side',
                    nusc=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    :param sample_data_token: Sample_data token.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param selected_anntokens: If provided only return the selected annotation.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                         aligned to z-plane in the world.
    :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
    """

    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    # hardcode
    if side in ['vehicle-side', 'cooperative']:
        cs_record = nusc.get('calibrated_sensor', 'a4debdb5-22b2-3269-b7d7-756f1d560c92')
        sensor2ego_rot = iterative_closest_point(np.array(cs_record['rotation']))
        sensor2ego_trans = cs_record['translation']
    elif side == 'infrastructure-side':
        cs_record = nusc.get('calibrated_sensor', '23ef3a7f-ebdc-389f-a831-34b76331632a')
        sensor2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
        sensor2ego_trans = cs_record['translation']
        calib_l2c_path = data_root + side + '/calib/virtuallidar_to_camera/' + sample_data_token + '.json'
        calib_l2c = read_json(calib_l2c_path)
        l2c_rot = np.array(calib_l2c['rotation'])
        appro_l2c_rot = iterative_closest_point(np.array(l2c_rot))
        cs_record = nusc.get('calibrated_sensor', 'eda75990-71f2-387c-b06a-415a923663a9')
        
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    # hardcode
    dir_path = data_path.split('velodyne')[0]
    data_path = dir_path + 'image/' + sample_data_token + '.jpg'

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        # cam_intrinsic_path = data_root + side + '/calib/camera_intrinsic/' + sample_data_token + '.json'
        # cam_intrinsic = np.array(read_json(cam_intrinsic_path)['cam_K']).reshape(3, 3)
        
        # hardcode
        imsize = (1920, 1080)
    else:
        cam_intrinsic = None
        imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    # if selected_anntokens is not None:
    #     boxes = list(map(nusc.get_box, selected_anntokens))
    # else:
    #     boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        if use_flat_vehicle_coordinates:
            # Move box to ego vehicle coord system parallel to world z plane.
            yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
        else:
            # Move box to ego vehicle coord system.
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system.
            box.translate(-np.array(sensor2ego_trans))
            box.rotate(Quaternion(matrix=np.array(sensor2ego_rot)).inverse)
            
            if side == 'infrastructure-side':
                # rotate
                box.center = np.dot(l2c_rot, box.center)
                q = Quaternion(matrix=appro_l2c_rot)
                box.orientation = q * box.orientation
                box.velocity = np.dot(l2c_rot, box.velocity)
                # translate
                box.translate(np.squeeze(np.array(calib_l2c['translation'])))
        if sensor_record['modality'] == 'camera' and not \
                box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
            continue

        box_list.append(box)

    return data_path, box_list, cam_intrinsic



# def get_predicted_data(sample_data_token: str,
#                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
#                        selected_anntokens=None,
#                        use_flat_vehicle_coordinates: bool = False,
#                        pred_anns=None,
#                        side = 'vehicle-side',
#                        nusc=None
#                        ):
#     """
#     Returns the data path as well as all annotations related to that sample_data.
#     Note that the boxes are transformed into the current sensor's coordinate frame.
#     :param sample_data_token: Sample_data token.
#     :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
#     :param selected_anntokens: If provided only return the selected annotation.
#     :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
#                                          aligned to z-plane in the world.
#     :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
#     """

#     # Retrieve sensor & pose records
#     sd_record = nusc.get('sample_data', sample_data_token)
#     # hardcode
#     if side in ['vehicle-side', 'cooperative']:
#         cs_record = nusc.get('calibrated_sensor', 'db7afd30-5a9a-325c-996b-ce1fd8f54739')
#         sensor2ego_rot = iterative_closest_point(np.array(cs_record['rotation']))
#         sensor2ego_trans = cs_record['translation']
#     elif side == 'infrastructure-side':
#         cs_record = nusc.get('calibrated_sensor', '23ef3a7f-ebdc-389f-a831-34b76331632a')
#         sensor2ego_rot = Quaternion(cs_record['rotation']).rotation_matrix
#         sensor2ego_trans = cs_record['translation']
#         calib_l2c_path = data_root + side + '/calib/virtuallidar_to_camera/' + sample_data_token + '.json'
#         calib_l2c = read_json(calib_l2c_path)
#         l2c_rot = np.array(calib_l2c['rotation'])
#         appro_l2c_rot = iterative_closest_point(np.array(l2c_rot))
#         cs_record = nusc.get('calibrated_sensor', 'eda75990-71f2-387c-b06a-415a923663a9')
        
#     sensor_record = nusc.get('sensor', cs_record['sensor_token'])
#     pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

#     data_path = nusc.get_sample_data_path(sample_data_token)
#     # hardcode
#     dir_path = data_path.split('velodyne')[0]
#     data_path = dir_path + 'image/' + sample_data_token + '.jpg'

#     if sensor_record['modality'] == 'camera':
#         cam_intrinsic = np.array(cs_record['camera_intrinsic'])
#         # hardcode
#         imsize = (1920, 1080)
#     else:
#         cam_intrinsic = None
#         imsize = None

#     # Retrieve all sample annotations and map to sensor coordinate system.
#     # if selected_anntokens is not None:
#     #    boxes = list(map(nusc.get_box, selected_anntokens))
#     # else:
#     #    boxes = nusc.get_boxes(sample_data_token)
#     boxes = pred_anns
#     # Make list of Box objects including coord system transforms.
#     box_list = []
#     for box in boxes:
#         if use_flat_vehicle_coordinates:
#             # Move box to ego vehicle coord system parallel to world z plane.
#             yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
#             box.translate(-np.array(pose_record['translation']))
#             box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
#         else:
#             # Move box to ego vehicle coord system.
#             box.translate(-np.array(pose_record['translation']))
#             box.rotate(Quaternion(pose_record['rotation']).inverse)

#             #  Move box to sensor coord system.
#             box.translate(-np.array(sensor2ego_trans))
#             box.rotate(Quaternion(matrix=np.array(sensor2ego_rot)).inverse)

#             if side == 'infrastructure-side':
#                 # rotate
#                 box.center = np.dot(l2c_rot, box.center)
#                 q = Quaternion(matrix=appro_l2c_rot)
#                 box.orientation = q * box.orientation
#                 box.velocity = np.dot(l2c_rot, box.velocity)
#                 # translate
#                 box.translate(np.squeeze(np.array(calib_l2c['translation'])))
                
#         if sensor_record['modality'] == 'camera' and not \
#                 box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
#             continue
#         box_list.append(box)

#     return data_path, box_list, cam_intrinsic

detection_mapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }


def lidar_render(sample_token, data, ax=None, out_path=None, side='vehicle-side', thre=0.0, nusc=None):
    bbox_gt_list = []
    bbox_pred_list = []
    anns = nusc.get('sample', sample_token)['anns']
    for ann in anns:
        content = nusc.get('sample_annotation', ann)
        bbox_gt_list.append(TrackingBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=nusc.box_velocity(content['token'])[:2],
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            tracking_name=content['category_name'],
            tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']),
            tracking_id=content['instance_token']))


    bbox_anns = data['results'][sample_token]
    for content in bbox_anns:
        bbox_pred_list.append(TrackingBox(
            sample_token=content['sample_token'],
            translation=tuple(content['translation']),
            size=tuple(content['size']),
            rotation=tuple(content['rotation']),
            velocity=tuple(content['velocity']),
            ego_translation=(0.0, 0.0, 0.0) if 'ego_translation' not in content
            else tuple(content['ego_translation']),
            num_pts=-1 if 'num_pts' not in content else int(content['num_pts']),
            tracking_name=content['tracking_name'],
            tracking_score=-1.0 if 'tracking_score' not in content else float(content['tracking_score']),
            tracking_id=content['tracking_id']))
    gt_annotations = EvalBoxes()
    pred_annotations = EvalBoxes()
    gt_annotations.add_boxes(sample_token, bbox_gt_list)
    pred_annotations.add_boxes(sample_token, bbox_pred_list)
    print('green is ground truth')
    print('blue is the predited result')
    if side in ['vehicle-side', 'cooperative']:
        eval_range = [-53.0, -53.0, 53.0, 53.0]
    elif side == 'infrastructure-side':
        eval_range = [-3.0, -53.0, 103.0, 53.0]
    visualize_sample(nusc, sample_token, gt_annotations, pred_annotations, conf_th=thre, savepath=out_path, eval_range=eval_range, ax=ax)


def get_color(category_name: str):
    """
    Provides the default colors based on the category names.
    This method works for the general nuScenes categories, as well as the nuScenes detection categories.
    """
    a = ['noise', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker',
     'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
     'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris',
     'movable_object.pushable_pullable', 'movable_object.trafficcone', 'static_object.bicycle_rack', 'vehicle.bicycle',
     'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance',
     'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'flat.driveable_surface',
     'flat.other', 'flat.sidewalk', 'flat.terrain', 'static.manmade', 'static.other', 'static.vegetation',
     'vehicle.ego']
    class_names = [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]
    #print(category_name)
    if category_name == 'bicycle':
        return nusc.colormap['vehicle.bicycle']
    elif category_name == 'construction_vehicle':
        return nusc.colormap['vehicle.construction']
    elif category_name == 'traffic_cone':
        return nusc.colormap['movable_object.trafficcone']

    for key in nusc.colormap.keys():
        if category_name in key:
            return nusc.colormap[key]
    return [0, 0, 0]


def render_sample_data(
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        side='vehicle-side',
        thre=None,
        nusc=None,
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    sample = nusc.get('sample', sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    if ax is None:
        # Create a figure
        fig = plt.figure(figsize=(24, 13))
        gs = gridspec.GridSpec(1, 2, wspace=0.05)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.05)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        axes = [ax1, ax2, ax3]

    # hardcode
    sample_data_token = sample['data']['LIDAR_TOP']
    
    # plot in BEV
    lidar_render(sample_token, pred_data, ax=axes[2], out_path=None, side=side, thre=thre, nusc=nusc)
    
    # plot in image
    boxes_pred = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['detection_name'], token=record['tracking_id']) for record in
                pred_data['results'][sample_token] if record['detection_score'] > thre]
    boxes_gt = nusc.get_boxes(sample_data_token)
    data_path, boxes_pred, camera_intrinsic = get_sample_data(sample_data_token,
                                                                    box_vis_level=box_vis_level, boxes=boxes_pred, side=side, nusc=nusc)
    _, boxes_gt, _ = get_sample_data(sample_data_token, box_vis_level=box_vis_level, boxes=boxes_gt, side=side, nusc=nusc)

    data = Image.open(data_path)
    # Show image.
    axes[0].imshow(data)
    axes[1].imshow(data)

    # Show boxes.
    for box in boxes_pred:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    for box in boxes_gt:
        c = np.array(get_color(box.name)) / 255.0
        box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)

    # Limit visible range.
    axes[0].set_xlim(0, data.size[0])
    axes[0].set_ylim(data.size[1], 0)
    axes[1].set_xlim(0, data.size[0])
    axes[1].set_ylim(data.size[1], 0)

    axes[0].axis('off')
    # axes[0].set_title('PRED: {} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[0].set_aspect('equal')

    axes[1].axis('off')
    # axes[1].set_title('GT:{} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[1].set_aspect('equal')

    if out_path is not None:
        out_path = os.path.join(out_path, sample_token+'.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    else:
        plt.show()
    plt.close()

def render_sample_data_coop(
        sample_token: str,
        with_anns: bool = True,
        box_vis_level: BoxVisibility = BoxVisibility.ANY,
        axes_limit: float = 40,
        ax=None,
        nsweeps: int = 1,
        out_path: str = None,
        underlay_map: bool = True,
        use_flat_vehicle_coordinates: bool = True,
        show_lidarseg: bool = False,
        show_lidarseg_legend: bool = False,
        filter_lidarseg_labels=None,
        lidarseg_preds_bin_path: str = None,
        verbose: bool = True,
        show_panoptic: bool = False,
        pred_data=None,
        side='vehicle-side',
        thre=None,
        veh2inf=None,
        nusc=None,
        nusc_inf=None,
        is_gt=False
      ) -> None:
    """
    Render sample data onto axis.
    :param sample_data_token: Sample_data token.
    :param with_anns: Whether to draw box annotations.
    :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
    :param axes_limit: Axes limit for lidar and radar (measured in meters).
    :param ax: Axes onto which to render.
    :param nsweeps: Number of sweeps for lidar and radar.
    :param out_path: Optional path to save the rendered figure to disk.
    :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
    :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
        aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
        can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
        setting is more correct and rotates the plot by ~90 degrees.
    :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
    :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param verbose: Whether to display the image after it is rendered.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    """
    sample = nusc.get('sample', sample_token)
    # sample = data['results'][sample_token_list[0]][0]
    if ax is None:
        # Create a figure
        fig = plt.figure(figsize=(24, 13))
        gs = gridspec.GridSpec(1, 2, wspace=0.05)
        gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace=0.05)
        ax1 = fig.add_subplot(gs0[0, 0])
        ax2 = fig.add_subplot(gs0[1, 0])
        ax3 = fig.add_subplot(gs[0, 1])
        axes = [ax1, ax2, ax3]

    # hardcode
    sample_data_token = sample['data']['LIDAR_TOP']
    
    # plot in BEV
    lidar_render(sample_token, pred_data, ax=axes[2], out_path=None, side=side, thre=thre, nusc=nusc)
    
    # plot in image for veh
    if is_gt:
        # load cooperative label
        boxes_gt = nusc.get_boxes(sample_data_token)
        data_path, boxes_gt, camera_intrinsic = get_sample_data(sample_data_token, box_vis_level=box_vis_level, boxes=boxes_gt, side=side, nusc=nusc)
    else:
        boxes_pred = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                    name=record['detection_name'], token='predicted') for record in
                pred_data['results'][sample_token] if record['detection_score'] > thre]
        data_path, boxes_pred, camera_intrinsic = get_sample_data(sample_data_token,
                                                                    box_vis_level=box_vis_level, boxes=boxes_pred, side=side, nusc=nusc)
    data = Image.open(data_path)
    axes[0].imshow(data)
    # Show boxes.
    if is_gt:
        for box in boxes_gt:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    else:
        for box in boxes_pred:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[0], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)

    # plot in image for inf
    sample_data_token_inf = veh2inf[sample_data_token]
    if is_gt:
        # load cooperative label, so use nusc rather than nusc_inf
        boxes_gt = nusc.get_boxes(sample_data_token)
        boxes_gt = veh2inf_convert(boxes_gt, data_root, veh2inf, sample_data_token)
        data_path, boxes_gt, camera_intrinsic = get_sample_data(sample_data_token_inf, box_vis_level=box_vis_level, boxes=boxes_gt, side='infrastructure-side', nusc=nusc_inf)
    else:
        # load cooperative prediction
        boxes = [Box(record['translation'], record['size'], Quaternion(record['rotation']),
                        name=record['detection_name'], token='predicted') for record in
                    pred_data['results'][sample_token] if record['detection_score'] > thre]
        boxes_inf = veh2inf_convert(boxes, data_root, veh2inf, sample_data_token)
        data_path, boxes_pred, camera_intrinsic = get_sample_data(sample_data_token_inf,
                                                                    box_vis_level=box_vis_level, boxes=boxes_inf, side='infrastructure-side', nusc=nusc_inf)
    data = Image.open(data_path)
    axes[1].imshow(data)
    if is_gt:
        for box in boxes_gt:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    else:
        for box in boxes_pred:
            c = np.array(get_color(box.name)) / 255.0
            box.render(axes[1], view=camera_intrinsic, normalize=True, colors=(c, c, c), linewidth=1)
    # Limit visible range.
    axes[0].set_xlim(0, data.size[0])
    axes[0].set_ylim(data.size[1], 0)
    axes[1].set_xlim(0, data.size[0])
    axes[1].set_ylim(data.size[1], 0)

    axes[0].axis('off')
    # axes[0].set_title('PRED: {} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[0].set_aspect('equal')

    axes[1].axis('off')
    # axes[1].set_title('GT:{} {labels_type}'.format(
    #     'VEHICLE_CAM_FRONT', labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
    axes[1].set_aspect('equal')

    if out_path is not None:
        out_path = os.path.join(out_path, sample_token+'.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.03, dpi=300)
    else:
        plt.show()
    plt.close()
    
def to_video(folder_path, out_path, fps=4, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, '*.png'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height //
                            downsample), interpolation=cv2.INTER_AREA)
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predroot', default='test/tiny_track_r50_stream_bs8_48epoch_3cls/Sun_Dec_29_18_24_23_2024/results_nusc.json', help='Path to json')
    parser.add_argument('--out_folder', default='result_vis/pf-track', help='Output folder path')
    parser.add_argument('--side', default='vehicle-side', help='side')
    parser.add_argument('--is_gt', default=True, help='plot gt')
    parser.add_argument('--dataroot', default='datasets/V2X-Seq-SPD-Batch-65-10-10761/', help='path to data')
    parser.add_argument('--version', default='v1.0-trainval', help='data version')
    parser.add_argument('--thre', default=0.15, help='filter threshold')
    args = parser.parse_args()

    data_root = args.dataroot
    bevformer_results = mmcv.load(args.predroot)
    # side = 'infrastructure-side' # or 'vehicle-side'
    side = args.side
    root_path = args.out_folder
    sample_token_list = list(bevformer_results['results'].keys())[73:]
    thre = args.thre
    is_gt = args.is_gt
    nusc = NuScenes(version=args.version, dataroot=data_root+side, verbose=True)

    folder_path = os.path.join(root_path, side)
    video_path = os.path.join(root_path, side+'.avi')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    from nuscenes.eval.common.config import config_factory
    cfg = config_factory("tracking_nips_2019")
    if side == 'cooperative':
        inf_root = data_root + 'infrastructure-side'
        nusc_inf = NuScenes(version=args.version, dataroot=inf_root, verbose=True)
        veh2inf = {}
        coop_info = read_json(os.path.join(data_root, 'cooperative/data_info.json'))
        for f in coop_info:
            veh2inf[f['vehicle_frame']] = f['infrastructure_frame']
            veh2inf[f['vehicle_frame']+'offset'] = f['system_error_offset']
            
    for id in range(len(sample_token_list)):
        if side != 'cooperative':
            render_sample_data(sample_token_list[id], pred_data=bevformer_results, out_path=folder_path, side=side, thre=thre, nusc=nusc)
        else:
            render_sample_data_coop(sample_token_list[id], pred_data=bevformer_results, out_path=folder_path, side=side, thre=thre, veh2inf=veh2inf, nusc=nusc, nusc_inf=nusc_inf, is_gt=is_gt)
    to_video(folder_path=folder_path, out_path=video_path)