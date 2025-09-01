import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from nuscenes import NuScenes
from nuscenes.prediction import (PredictHelper,
                                 convert_local_coords_to_global,
                                 convert_global_coords_to_local)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle


def classify_label_type_to_id(label_type: str) -> int:
    """
    Classifies label type to id.
    
    Args:
        label_type (str): The category name of the object.

    Returns:
        int: The corresponding id for the label type.
    """
    # if 'human' in label_type: # pedestrain
    #     return 2
    # if 'movable_object' in label_type: # traffic_cone
    #     return 3
    # if ('vehicle.bicycle' in label_type) or ('vehicle.motorcycle' in label_type): # bicycle
    #     return 1
    # else: # car
    #     return 0
    mapping_dict = {
        'car': 0,
        'bicycle': 1,
        'pedestrian': 2,
        'traffic_cone': 3
    }
    return mapping_dict[label_type]


def k_means_anchors(k: int, future_traj_all: np.ndarray) -> np.ndarray:
    """
    Extracts anchors for multipath/covernet using k-means on train set
    trajectories.
    
    Args:
        k (int): The number of clusters for k-means algorithm.
        future_traj_all (np.ndarray): The array containing all future trajectories.

    Returns:
        np.ndarray: The k anchor trajectories.
    """
    prototype_traj = future_traj_all
    traj_len = prototype_traj.shape[1]
    traj_dim = prototype_traj.shape[2]
    ds_size = future_traj_all.shape[0]
    trajectories = future_traj_all
    clustering = KMeans(n_clusters=k).fit(trajectories.reshape((ds_size, -1)))
    anchors = np.zeros((k, traj_len, traj_dim))
    for i in range(k):
        anchors[i] = np.mean(trajectories[clustering.labels_ == i], axis=0)
    return anchors


def run(num_modes: int = 6, predicted_traj_len: int = 12) -> None:
    """
    Main function to run the script. It calculates anchor trajectories for each object type
    using the k-means algorithm and saves the result as a pickle file.

    Args:
        num_modes (int, optional): The number of clusters for k-means algorithm. Defaults to 6.
        predicted_traj_len (int, optional): The length of the predicted trajectories. Defaults to 12.
    """
    nusc = NuScenes(version='v1.0-trainval',
                    dataroot='datasets/V2X-Seq-SPD-Batch-95-2-3134-rm-ego-car-new/vehicle-side/', verbose=True)
    predict_helper = PredictHelper(nusc)

    all_fut_trajectories = []
    all_sample_tokens = [sample['token'] for sample in nusc.sample]

    grouped_trajectories = [[], [], [], []]
    for sample_token in all_sample_tokens:
        sd_rec = nusc.get('sample', sample_token)
        ann_tokens = sd_rec['anns']
        for ann_token in ann_tokens:
            ann_record = nusc.get('sample_annotation', ann_token)
            label_type = ann_record['category_name']
            type_id = classify_label_type_to_id(label_type)

            instance_token = nusc.get('sample_annotation', ann_token)[
                'instance_token']
            fut_traj_local = predict_helper.get_future_for_agent(
                instance_token, sample_token, seconds=6, in_agent_frame=True)
            if fut_traj_local.shape[0] < predicted_traj_len:
                continue
            grouped_trajectories[type_id].append(fut_traj_local)

    kmeans_anchors = []
    for type_id in range(4):
        if len(grouped_trajectories[type_id]) == 0:
            anchor = np.zeros((num_modes, predicted_traj_len, 2))
            kmeans_anchors.append(anchor)
            continue
        grouped_trajectory = np.stack(grouped_trajectories[type_id])
        kmeans_anchors.append(k_means_anchors(num_modes, grouped_trajectory))
    # kmeans_anchors = np.stack(kmeans_anchors)

    info_dict = {'anchors_all': kmeans_anchors}
    pickle.dump(info_dict, open('data/others/spd_veh_motion_anchor_infos_mode6.pkl', 'wb'))


if __name__ == '__main__':
    run()