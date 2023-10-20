import numpy as np
import matplotlib.pyplot as plt
import json
import gzip
from math import floor
from modeling.utils.build_map_utils import find_first_nonzero_elem_per_row
from modeling.utils.constants import d3_41_colors_rgb
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, coords_to_pose


def build_semantic_map_via_point_cloud(x, y, z, label_seq, height):
    """
    build the semantic map from the point cloud
    """
    min_X = -50.0
    max_X = 50.0
    min_Z = -50.0
    max_Z = 50.0
    cell_size = 0.05
    min_Y = height - 0.2
    max_Y = height + 2.0
    map_boundary = 5

    x_grid = np.arange(min_X, max_X, cell_size)
    z_grid = np.arange(min_Z, max_Z, cell_size)
    y_grid = np.arange(min_Y, max_Y, cell_size)

    H, W = len(z_grid), len(x_grid)
    min_x_coord = W - 1
    max_x_coord = 0
    min_z_coord = H - 1
    max_z_coord = 0
    max_y_coord = 0
    THRESHOLD_HIGH = len(y_grid)

    four_dim_grid = np.zeros(
        (len(z_grid), len(y_grid) + 1, len(x_grid), 42), dtype=np.int16)  # x, y, z, C

    xyz_points = np.vstack((x, z, y))
    sseg_points = label_seq.squeeze().astype(int)

    # ================= slice the point cloud ====================
    mask_y = np.logical_or(
        xyz_points[1, :] > height - 0.2, xyz_points[1, :] < height + 2.0)

    mask_X = np.logical_and(xyz_points[0, :] > min_X,
                            xyz_points[0, :] < max_X)
    mask_Z = np.logical_and(xyz_points[2, :] > min_Z,
                            xyz_points[2, :] < max_Z)
    mask_sseg = np.logical_and(sseg_points != 0, sseg_points != 17)
    mask_XYZ = np.logical_and.reduce((mask_X, mask_y, mask_Z, mask_sseg))
    xyz_points = xyz_points[:, mask_XYZ]
    sseg_points = sseg_points[mask_XYZ]

    x_coord = np.floor(
        (xyz_points[0, :] - min_X) / cell_size).astype(int)
    y_coord = np.digitize(xyz_points[1, :], y_grid)
    z_coord = (H - 1) - np.floor(
        (xyz_points[2, :] - min_Z) / cell_size).astype(int)

    if x_coord.shape[0] > 0:
        four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1

        # update the weights for the local map
        min_x_coord = min(max(np.min(x_coord) - map_boundary, 0),
                          min_x_coord)
        max_x_coord = max(
            min(np.max(x_coord) + map_boundary, W - 1),
            max_x_coord)
        min_z_coord = min(max(np.min(z_coord) - map_boundary, 0),
                          min_z_coord)
        max_z_coord = max(
            min(np.max(z_coord) + map_boundary, H - 1),
            max_z_coord)

        max_y_coord = max(np.max(y_coord), max_y_coord)

    smaller_four_dim_grid = four_dim_grid[min_z_coord:max_z_coord + 1, 0:THRESHOLD_HIGH,
                                          min_x_coord:max_x_coord + 1, :]

    # argmax over the category axis
    zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
    # swap y dim to the last axis
    zxy_grid = np.swapaxes(zyx_grid, 1, 2)
    L, M, N = zxy_grid.shape
    zxy_grid = zxy_grid.reshape(L * M, N)

    semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
    semantic_map = semantic_map.reshape(L, M)

    map_data = {}
    map_data['semantic_map'] = semantic_map
    map_data['pose_range'] = (min_X, min_Z, max_X, max_Z)
    map_data['coords_range'] = (min_x_coord, min_z_coord, max_x_coord, max_z_coord)
    map_data['wh'] = (W, H)

    return semantic_map, map_data


data_folder = 'data/other_datasets/mp3d_scene_pclouds'
scene_name = '1pXnuDYAj8r'

with np.load(f'{data_folder}/{scene_name}_pcloud.npz') as data:
    x = data['x']
    y = data['y']
    z = data['z']
    label_seq = data['label_seq']

# ================= load the episodes ================
split = 'val_seen'  # 'train'
jsonfilename = f'data/datasets/vln_r2r_mp3d_v1/{split}/{split}.json.gz'
with gzip.open(jsonfilename, 'r') as fin:
    data = json.loads(fin.read().decode('utf-8'))
episodes = data['episodes']

id_epi = 226  # 183
episode = episodes[id_epi]

# ================== build the map ======================
height = episode['start_position'][1]  # 3.514  #

semantic_map, map_data = build_semantic_map_via_point_cloud(
    x, y, z, label_seq, height)

color_semantic_map = apply_color_to_map(semantic_map)

# ============================= visualize the path
path = episode['reference_path']

x, y = [], []
for idx, node in enumerate(path):
    X, Y, Z = node
    # transform world pose to map coordinates
    coords = pose_to_coords((X, -Z), map_data)
    x.append(coords[0])
    y.append(coords[1])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
ax.imshow(color_semantic_map)
ax.scatter(x=x, y=y, c='w', s=30, zorder=3)

# draw the start location
X, Y, Z = episode['start_position']
# transform world pose to map coordinates
coords = pose_to_coords((X, -Z), map_data)

ax.scatter(x=coords[0], y=coords[1], c='b', s=30, zorder=4)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.tight_layout()
plt.show()
