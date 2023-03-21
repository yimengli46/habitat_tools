import numpy as np
from math import pi
import habitat
import habitat_sim
from modeling.utils.baseline_utils import read_occ_map_npy, coords_to_pose, wrap_angle, convert_insseg_to_sseg, apply_color_to_map
import random
import matplotlib.pyplot as plt

# specify on which scene you want to build the map
scene = '2t7WUuJeko7'
height = 0.16325

semantic_map_folder = 'output/semantic_map'
random.seed(5)

# load the pre-built occupancy map
occ_map_npy = np.load(
    f'{semantic_map_folder}/{scene}/BEV_occupancy_map.npy', allow_pickle=True).item()
map_data = read_occ_map_npy(occ_map_npy)

occ_map = map_data['occupancy_map']

# =============================== initialize the habitat environment ============================
config = habitat.get_config(
    config_paths='configs/habitat_env/build_map_mp3d.yaml')
config.defrost()
config.SIMULATOR.SCENE = f'data/scene_datasets/mp3d/{scene}/{scene}.glb'
config.SIMULATOR.SCENE_DATASET = 'data/scene_datasets/mp3d/mp3d_annotated_basis.scene_dataset_config.json'
config.freeze()

env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
env.reset()

# get scene insance to category dict
scene = env.semantic_annotations()
ins2cat_dict = {
    int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

# pick a free viewpoint (x, y, yaw) on the map
coords = (90, 45, pi)

# convert to pose in the environment
pose = coords_to_pose(coords, map_data)
X, Z, yaw = pose

# ==================== render the view at various heading angle ================
obs_list = []
for rot in [0, 90, 180, 270]:
    heading_angle = rot / 180 * np.pi
    heading_angle = wrap_angle(heading_angle + yaw)

    agent_rot = habitat_sim.utils.common.quat_from_angle_axis(
        heading_angle, habitat_sim.geo.GRAVITY)
    obs = env.get_observations_at(
        (X, height, Z), agent_rot, keep_agent_at_new_pose=True)
    obs_list.append(obs)

# ======================= stitch the views to form a panorama
rgb_lst, depth_lst, sseg_lst = [], [], []

for idx, obs in enumerate(obs_list):
    # load rgb image, depth and sseg
    rgb_img = obs['rgb']
    depth_img = obs['depth'][:, :, 0]

    insseg_img = obs["semantic"]
    if len(insseg_img.shape) > 2:
        insseg_img = np.squeeze(insseg_img)
    sseg_img = convert_insseg_to_sseg(insseg_img, ins2cat_dict)

    rgb_lst.append(rgb_img)
    depth_lst.append(depth_img)
    sseg_lst.append(sseg_img)

    panorama_rgb = np.concatenate(rgb_lst, axis=1)
    panorama_depth = np.concatenate(depth_lst, axis=1)
    panorama_sseg = np.concatenate(sseg_lst, axis=1)

# visualize the given coordinates
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
ax.imshow(occ_map, cmap='gray')
ax.scatter(x=[coords[0]], y=[coords[1]], marker='o', c='r', s=40)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.tight_layout()
plt.show()


# visualize the rendered panorama
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))
ax[0].imshow(panorama_rgb)
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title("rgb")
ax[1].imshow(apply_color_to_map(panorama_sseg))
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title("sseg")
ax[2].imshow(panorama_depth)
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title("depth")
fig.tight_layout()
plt.show()
