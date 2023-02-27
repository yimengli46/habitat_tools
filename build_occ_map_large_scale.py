import numpy as np
from modeling.utils.baseline_utils import read_map_npy, pose_to_coords, save_occ_map_through_plt
from core import cfg


def build_occ_map(env, saved_folder, height, scene_name, split):
    semantic_map_folder = f'output/semantic_map'

    # after testing, using 8 angles is most efficient
    cell_size = cfg.SEM_MAP.CELL_SIZE

    # ============================= build a grid =========================================
    x = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, cell_size)
    z = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, cell_size)
    xv, zv = np.meshgrid(x, z)
    grid_H, grid_W = zv.shape

    sem_map_npy = np.load(
        f'{semantic_map_folder}/{split}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
    _, pose_range, coords_range, WH = read_map_npy(sem_map_npy)

    occ_map = np.zeros((grid_H, grid_W), dtype=int)

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
