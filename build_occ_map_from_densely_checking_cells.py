import numpy as np
from modeling.utils.baseline_utils import create_folder
import habitat
import random
from modeling.utils.baseline_utils import read_map_npy, pose_to_coords, save_occ_map_through_plt
from core import cfg
import json
import os

# =========================================== fix the habitat scene shuffle ===============================
SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

split = 'val'
output_folder = f'output/semantic_map/{split}'
semantic_map_folder = f'output/semantic_map/{split}'

# after testing, using 8 angles is most efficient
theta_lst = [0]
built_scenes = []
cell_size = cfg.SEM_MAP.CELL_SIZE

scene_floor_dict = np.load(
    f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

# ============================= build a grid =========================================
x = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, cell_size)
z = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, cell_size)
xv, zv = np.meshgrid(x, z)
grid_H, grid_W = zv.shape

# =================================analyze json file to get the semantic files =============================
point_filenames = []
sem_filenames = []
with open(f'data/versioned_data/hm3d-1.0/hm3d/hm3d_annotated_basis.scene_dataset_config.json') as f:
    data = json.loads(f.read())
    if split == 'val':
        list_json_dirs = data['scene_instances']['paths']['.json'][103:]
    elif split == 'train':
        list_json_dirs = data['scene_instances']['paths']['.json'][23:103]

    for json_dir in list_json_dirs:
        first_slash = json_dir.find('/')
        second_slash = json_dir.find('/', first_slash+1)

        sem_filename = json_dir[first_slash+1:second_slash]
        point_filename = json_dir[first_slash+7:second_slash]

        point_filenames.append(point_filename)
        sem_filenames.append(sem_filename)


for scene_with_index in sem_filenames:
    print(f'scene = {scene_with_index}')
    config = habitat.get_config(config_paths=cfg.GENERAL.BUILD_MAP_CONFIG_PATH)
    config.defrost()
    config.SIMULATOR.SCENE = f'data/scene_datasets/hm3d/{split}/{scene_with_index}/{scene_with_index[6:]}.basis.glb'
    config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
    config.SIMULATOR.SCENE_DATASET = f'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.freeze()

    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    env.reset()

    scene_dict = scene_floor_dict[scene_with_index]

    # =============================== traverse each floor ===========================
    for floor_id in list(scene_dict.keys()):
        height = scene_dict[floor_id]['y']
        scene_name = f'{scene_with_index}_{floor_id}'

        # =============================== traverse each floor ===========================
        print(f'*****scene_name = {scene_name}***********')

        saved_folder = f'{output_folder}/{scene_name}'
        create_folder(saved_folder, clean_up=False)

        npy_file = f'{saved_folder}/BEV_occupancy_map.npy'
        if os.path.isfile(npy_file):
            print(
                f'!!!!!!!!!!!!!!!!!!!!npy file exists. skip scene {scene_name}')
            continue

        sem_map_npy = np.load(
            f'{semantic_map_folder}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
        _, pose_range, coords_range, WH = read_map_npy(sem_map_npy)

        occ_map = np.zeros((grid_H, grid_W), dtype=int)

        count_ = 0
        # ========================= generate observations ===========================
        for grid_z in range(grid_H):
            for grid_x in range(grid_W):

                x = xv[grid_z, grid_x] + cell_size/2.
                z = zv[grid_z, grid_x] + cell_size/2.
                y = height

                agent_pos = np.array([x, y, z])
                flag_nav = env.is_navigable(agent_pos)

                if flag_nav:
                    x = xv[grid_z, grid_x] + cell_size/2.
                    z = zv[grid_z, grid_x] + cell_size/2.
                    # should be map pose
                    z = -z
                    x_coord, z_coord = pose_to_coords(
                        (x, z), pose_range, coords_range, WH, flag_cropped=False)
                    occ_map[z_coord, x_coord] = 1

        occ_map = occ_map[coords_range[1]:coords_range[3] +
                          1, coords_range[0]:coords_range[2]+1]

        # save the final results
        map_dict = {}
        map_dict['occupancy'] = occ_map
        map_dict['min_x'] = coords_range[0]
        map_dict['max_x'] = coords_range[2]
        map_dict['min_z'] = coords_range[1]
        map_dict['max_z'] = coords_range[3]
        map_dict['min_X'] = pose_range[0]
        map_dict['max_X'] = pose_range[2]
        map_dict['min_Z'] = pose_range[1]
        map_dict['max_Z'] = pose_range[3]
        map_dict['W'] = WH[0]
        map_dict['H'] = WH[1]
        np.save(f'{saved_folder}/BEV_occupancy_map.npy', map_dict)

        # save the final color image
        save_occ_map_through_plt(occ_map, f'{saved_folder}/occ_map.jpg')

        print(f'**********************finished building the occ map!')

    env.close()
