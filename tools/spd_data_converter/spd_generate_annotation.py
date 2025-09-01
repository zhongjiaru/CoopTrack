import os
import json
import numpy as np
from rich.progress import track
import argparse
import os


def read_json(path_json):
    with open(path_json, 'r') as load_f:
        data_json = json.load(load_f)
    return data_json


def write_json(data_json, path_json):
    with open(path_json, "w") as dump_f:
        json.dump(data_json, dump_f)


def muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C):
    rotationA2B = np.array(rotationA2B).reshape(3, 3)
    rotationB2C = np.array(rotationB2C).reshape(3, 3)
    rotation = np.dot(rotationB2C, rotationA2B)
    translationA2B = np.array(translationA2B).reshape(3, 1)
    translationB2C = np.array(translationB2C).reshape(3, 1)
    translation = np.dot(rotationB2C, translationA2B) + translationB2C
    return rotation, translation


def rev_matrix(R):
    R = np.matrix(R)
    rev_R = R.I
    rev_R = np.array(rev_R)
    return rev_R


def reverse(rotation, translation):
    rev_rotation = rev_matrix(rotation)
    rev_translation = -np.dot(rev_rotation, translation)
    return rev_rotation, rev_translation


def trans(input_point, translation, rotation):
    translation = np.array(translation).reshape(3, 1)
    rotation = np.array(rotation).reshape(3, 3)
    output_point = np.dot(rotation, input_point.reshape(3, 1)).reshape(3) + np.array(translation).reshape(3)
    return np.array(output_point)


def get_lidar2novatel(path_lidar2novatel):  # vehicle side
    lidar2novatel = read_json(path_lidar2novatel)
    rotation = lidar2novatel["transform"]["rotation"]
    translation = lidar2novatel["transform"]["translation"]
    return rotation, translation


def get_novatel2world(path_novatel2world):  # vehicle side
    novatel2world = read_json(path_novatel2world)
    rotation = novatel2world["rotation"]
    translation = novatel2world["translation"]
    return rotation, translation


def get_virtuallidar2world(path_lidar2world):  # Infrastructure side, lidar to word
    lidar2world = read_json(path_lidar2world)
    rotation = lidar2world["rotation"]
    translation = lidar2world["translation"]
    return rotation, translation


def coord_vehicle_lidar2world(lidar2novatel_path, novatel2world_path):
    rotationA2B, translationA2B = get_lidar2novatel(lidar2novatel_path)
    rotationB2C, translationB2C = get_novatel2world(novatel2world_path)
    new_rotationA2C, new_translationA2C = muilt_coord(rotationA2B, translationA2B, rotationB2C, translationB2C)
    return new_rotationA2C, new_translationA2C


def gen_coop_anno_database(vic_data_path, save_dir):
    data_info = read_json(f'{vic_data_path}/cooperative/data_info.json')
    dict_anno_database = {}
    for i in track(range(len(data_info)-1)):
        veh_frame = data_info[i]["vehicle_frame"]
        frame_info = read_json(f'{vic_data_path}/cooperative/label/{veh_frame}.json')
        lidar2novatel_path = f'{vic_data_path}/vehicle-side/calib/lidar_to_novatel/{veh_frame}.json'
        novatel2world_path = f'{vic_data_path}/vehicle-side/calib/novatel_to_world/{veh_frame}.json'
        rotation, translation = coord_vehicle_lidar2world(lidar2novatel_path, novatel2world_path)
        for obj in frame_info:
            token = obj.pop("token")
            if token == "2b12efc0-c935-3771-aebf-2d01956815e3":
                print(veh_frame)
                continue
            if token in dict_anno_database.keys():
                print("token error:", token)
            dict_anno_database[token] = {}
            dict_anno_database[token].update(obj)
            dict_anno_database[token]["calib_e2g"] = {
                "rotation": rotation.tolist(),
                "translation": translation.tolist()
            }
            dict_anno_database[token]["next"] = ""
            next_veh_frame = data_info[i+1]["vehicle_frame"]
            next_frame_info = read_json(f'{vic_data_path}/cooperative/label/{next_veh_frame}.json')
            m = 0
            for next_obj in next_frame_info:
                if next_obj["track_id"] == obj["track_id"]:
                    m += 1
                    dict_anno_database[token]["next"] = next_obj["token"]
            if m > 1:
                print("next has", m, "objects")

    veh_frame = data_info[len(data_info)-1]["vehicle_frame"]
    frame_info = read_json(f'{vic_data_path}/cooperative/label/{veh_frame}.json')
    lidar2novatel_path = f'{vic_data_path}/vehicle-side/calib/lidar_to_novatel/{veh_frame}.json'
    novatel2world_path = f'{vic_data_path}/vehicle-side/calib/novatel_to_world/{veh_frame}.json'
    rotation, translation = coord_vehicle_lidar2world(lidar2novatel_path, novatel2world_path)
    for obj in frame_info:
        token = obj.pop("token")
        if token in dict_anno_database.keys():
            print("token error:", token)
        dict_anno_database[token] = {}
        dict_anno_database[token].update(obj)
        dict_anno_database[token]["calib_e2g"] = {
            "rotation": rotation.tolist(),
            "translation": translation.tolist()
        }
        dict_anno_database[token]["next"] = ""

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_path = os.path.join(save_dir, "anno_database.json")
    write_json(dict_anno_database, save_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('--data-root', type=str, default="./datasets/V2X-Seq-SPD-Example")
    parser.add_argument('--save-dir', type=str, default="./datasets/V2X-Seq-SPD-Example")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    save_dir = args.save_dir
    gen_coop_anno_database(data_root, save_dir)