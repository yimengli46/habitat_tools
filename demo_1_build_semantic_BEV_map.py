import numpy as np
from math import pi
from modeling.utils.baseline_utils import convert_insseg_to_sseg, create_folder
import habitat
import habitat_sim
from modeling.utils.build_map_utils import semantic_map
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from core import cfg

# specify on which scene you want to build the map
scene = '2t7WUuJeko7'
height = 0.16325

# specify the output folder
create_folder('output')
create_folder('output/semantic_map')
output_folder = 'output/semantic_map/'
saved_folder = f'{output_folder}/{scene}'
create_folder(saved_folder, clean_up=False)

# using 8 angles is most efficient
theta_lst = [0, pi / 4, pi / 2, pi * 3. / 4,
             pi, pi * 5. / 4, pi * 3. / 2, pi * 7. / 4]

# ============================= initialize a grid =========================================
x = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, 0.3)
z = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, 0.3)
xv, zv = np.meshgrid(x, z)
grid_H, grid_W = zv.shape

print(f'build a semantic map on scene = {scene}')

# =============================== initialize the habitat environment ============================
config = habitat.get_config(
    config_paths='configs/habitat_env/build_map_mp3d.yaml')
config.defrost()
config.SIMULATOR.SCENE = f'data/scene_datasets/mp3d/{scene}/{scene}.glb'
config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/mp3d/mp3d_annotated_basis.scene_dataset_config.json'
config.freeze()

env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
env.reset()

# ============================ get scene ins to cat dict =================================
scene_semantics = env.semantic_annotations()
ins2cat_dict = {
    int(obj.id.split("_")[-1]): obj.category.index() for obj in scene_semantics.objects}

# ================================ Building a map ===============================
sem_map = semantic_map(saved_folder)

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
                rgb_img = observations['rgb']
                depth_img = observations['depth'][:, :, 0]
                insseg_img = observations['semantic']
                # convert instance segmentation to semantic segmentation
                sseg_img = convert_insseg_to_sseg(insseg_img, ins2cat_dict)

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

                sem_map.build_semantic_map(
                    rgb_img, depth_img, sseg_img, pose, count_)
                count_ += 1

sem_map.save_final_map()

env.close()
