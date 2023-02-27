import numpy as np
from modeling.utils.baseline_utils import create_folder
import habitat
from core import cfg
import argparse
import multiprocessing
import os
import json
from build_semantic_BEV_map_large_scale import build_sem_map
from build_occ_map_large_scale import build_occ_map


def build_env(split, scene_with_index, device_id=0):
    # ================================ load habitat env============================================
    print(f'scene_with_index = {scene_with_index}')
    config = habitat.get_config(
        config_paths=cfg.GENERAL.BUILD_MAP_CONFIG_PATH)
    config.defrost()
    config.SIMULATOR.SCENE = f'data/scene_datasets/mp3d/{scene_with_index}/{scene_with_index}.glb'
    config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
    config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
    config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = device_id
    config.freeze()
    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    return env


def build_floor(split, scene_with_index, output_folder, scene_floor_dict):
    # ============================ get a gpu
    device_id = gpu_Q.get()

    # ================ initialize habitat env =================
    env = build_env(split, scene_with_index, device_id=device_id)
    env.reset()

    scene_dict = scene_floor_dict[scene_with_index]
    for floor_id in list(scene_dict.keys()):
        height = scene_dict[floor_id]['y']
        scene_name = f'{scene_with_index}_{floor_id}'

        scene_output_folder = f'{output_folder}/{scene_name}'
        create_folder(scene_output_folder)

        build_sem_map(env, scene_output_folder, height)
        build_occ_map(env, scene_output_folder, height, scene_name, split)

    env.close()

    # ================================ release the gpu============================
    gpu_Q.put(device_id)


def multi_run_wrapper(args):
    """ wrapper for multiprocessor """
    build_floor(args[0], args[1], args[2], args[3])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, default='test')
    args = parser.parse_args()

    cfg.merge_from_file(f'configs/generate_maps.yaml')
    cfg.freeze()

    # ====================== get the available GPU devices ============================
    visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    devices = [int(dev) for dev in visible_devices]

    for device_id in devices:
        for _ in range(cfg.SLURM.PROC_PER_GPU):
            gpu_Q.put(device_id)

    # =============================== basic setup =======================================
    split = args.split
    scene_floor_dict = np.load(
        f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
        allow_pickle=True).item()

    output_folder = f'output/semantic_map/{split}'
    create_folder(output_folder)

    args1 = list(scene_floor_dict.keys())
    with multiprocessing.Pool(processes=cfg.SLURM.NUM_PROCESS) as pool:
        args0 = [split for _ in range(len(args1))]
        args2 = [output_folder for _ in range(len(args1))]
        args3 = [scene_floor_dict for _ in range(len(args1))]
        pool.map(multi_run_wrapper, list(zip(args0, args1, args2, args3)))
        pool.close()
        pool.join()


if __name__ == "__main__":
    gpu_Q = multiprocessing.Queue()
    main()
