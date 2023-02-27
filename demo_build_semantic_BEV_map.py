import numpy as np
from math import pi
from modeling.utils.baseline_utils import convertInsSegToSSeg, create_folder
import habitat
import habitat_sim
from modeling.utils.build_map_utils import SemanticMap
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import random
from core import cfg
import json
import os

# =========================================== fix the habitat scene shuffle ===============================
SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

split = 'train'
output_folder = f'output/semantic_map/{split}'
# after testing, using 8 angles is most efficient
theta_lst = [0, pi/4, pi/2, pi*3./4, pi, pi*5./4, pi*3./2, pi*7./4]
str_theta_lst = ['000', '090', '180', '270']

# ============================ load scene heights ===================================
scene_floor_dict = np.load(
    f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

# ============================= build a grid =========================================
x = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, 0.3)
z = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, 0.3)
xv, zv = np.meshgrid(x, z)
grid_H, grid_W = zv.shape

print(f'number of semantic scenes = {len(sem_filenames)}')

for scene_name in list(scene_floor_dict.keys()):
    print(f'scene = {scene_name}')
    # ============================ traverse each scene ============================
    config = habitat.get_config(config_paths=cfg.GENERAL.BUILD_MAP_CONFIG_PATH)
    config.defrost()
    config.SIMULATOR.SCENE = f'data/scene_datasets/hm3d/{split}/{scene_name}/{scene_name}.basis.glb'
    config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
    config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'
    config.freeze()

    env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
    env.reset()

    scene_dict = scene_floor_dict[scene_name]

    # =============================== traverse each floor ===========================
    for floor_id in list(scene_dict.keys()):
        height = scene_dict[floor_id]['y']
        scene_name = f'{scene_name}_{floor_id}'

        print(f'*****scene_name = {scene_name}***********')

        saved_folder = f'{output_folder}/{scene_name}'
        create_folder(saved_folder, clean_up=False)

        npy_file = f'{saved_folder}/BEV_semantic_map.npy'
        if os.path.isfile(npy_file):
            print(
                f'!!!!!!!!!!!!!!!!!!!!npy file exists. skip scene {scene_name}')
            continue

        # ============================ get scene ins to cat dict
        scene = env.semantic_annotations()
        ins2cat_dict = {
            int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

        # ================================ Building a map ===============================
        SemMap = SemanticMap(saved_folder)

        count_ = 0
        # ========================= generate observations ===========================
        for grid_z in range(grid_H):
            for grid_x in range(grid_W):
                x = xv[grid_z, grid_x]
                z = zv[grid_z, grid_x]
                y = height
                agent_pos = np.array([x, y, z])

                flag_nav = env.is_navigable(agent_pos)

                if flag_nav:
                    # ==================== traverse theta ======================
                    for idx_theta, theta in enumerate(theta_lst):
                        agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
                            theta, habitat_sim.geo.GRAVITY)
                        observations = env.get_observations_at(
                            agent_pos, agent_rot, keep_agent_at_new_pose=True)
                        rgb_img = observations["rgb"]
                        depth_img = observations['depth'][:, :, 0]
                        InsSeg_img = observations["semantic"]
                        sseg_img = convertInsSegToSSeg(
                            InsSeg_img, ins2cat_dict)

                        # =============================== get agent global pose on habitat env ========================#
                        agent_pos = env.get_agent_state().position
                        agent_rot = env.get_agent_state().rotation
                        heading_vector = quaternion_rotate_vector(
                            agent_rot.inverse(), np.array([0, 0, -1]))
                        phi = cartesian_to_polar(
                            -heading_vector[2], heading_vector[0])[1]
                        angle = phi
                        print(f'agent position = {agent_pos}, angle = {angle}')
                        pose = (agent_pos[0], agent_pos[2], angle)

                        SemMap.build_semantic_map(
                            rgb_img, depth_img, sseg_img, pose, count_)
                        count_ += 1

        SemMap.save_final_map()

    env.close()
