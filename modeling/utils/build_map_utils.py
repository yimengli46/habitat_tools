import numpy as np
import cv2
import matplotlib.pyplot as plt
from .baseline_utils import project_pixels_to_world_coords, apply_color_to_map, save_sem_map_through_plt
from core import cfg


def find_first_nonzero_elem_per_row(mat):
    H, W = mat.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)

    xv[mat == 0] = 0
    min_idx_nonzero_per_row = np.max(xv, axis=1).astype(int)

    yv = yv[:, 0].astype(int)

    result = mat[yv, min_idx_nonzero_per_row]
    return result


""" class used to build semantic maps of the scenes

It takes dense observations of the environment and project pixels to the ground.
"""


class SemanticMap:

    def __init__(self, saved_folder):

        self.scene_name = ''
        self.cell_size = cfg.SEM_MAP.CELL_SIZE
        self.step_size = 1000
        self.map_boundary = 5
        self.detector = None
        self.saved_folder = saved_folder

        self.IGNORED_CLASS = cfg.SEM_MAP.IGNORED_SEM_CLASS  # ceiling class is ignored

        # ==================================== initialize 4d grid =================================
        self.min_X = -cfg.SEM_MAP.WORLD_SIZE
        self.max_X = cfg.SEM_MAP.WORLD_SIZE
        self.min_Z = -cfg.SEM_MAP.WORLD_SIZE
        self.max_Z = cfg.SEM_MAP.WORLD_SIZE
        self.min_Y = 0.0
        self.max_Y = cfg.SENSOR.AGENT_HEIGHT + self.cell_size

        self.x_grid = np.arange(self.min_X, self.max_X, self.cell_size)
        self.z_grid = np.arange(self.min_Z, self.max_Z, self.cell_size)
        self.y_grid = np.arange(self.min_Y, self.max_Y, self.cell_size)

        self.THRESHOLD_LOW = 5
        self.THRESHOLD_HIGH = len(self.y_grid)

        print(f'y_grid = {self.y_grid}')
        print(f'len(y_grid) = {len(self.y_grid)}')
        print(
            f'thresh_low = {self.THRESHOLD_LOW}, thresh_high = {self.THRESHOLD_HIGH}')

        self.four_dim_grid = np.zeros(
            (len(self.z_grid), len(self.y_grid)+1,
             len(self.x_grid), cfg.SEM_MAP.GRID_CLASS_SIZE),
            dtype=np.int16)  # x, y, z, C
        print(f'self.four_dim_grid.shape = {self.four_dim_grid.shape}')

        #assert 1==2

        # ===================================
        self.H, self.W = len(self.z_grid), len(self.x_grid)
        self.min_x_coord = self.W - 1
        self.max_x_coord = 0
        self.min_z_coord = self.H - 1
        self.max_z_coord = 0
        self.max_y_coord = 0

    def build_semantic_map(self, rgb_img, depth_img, sseg_img, pose, step_):
        """ update semantic map with observations rgb_img, depth_img, sseg_img and robot pose."""
        sem_map_pose = (pose[0], -pose[1], -pose[2])  # x, z, theta

        xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, sem_map_pose,
                                                                 gap=2, FOV=90, cx=128, cy=128, resolution_x=256, resolution_y=256, ignored_classes=self.IGNORED_CLASS)

        mask_X = np.logical_and(xyz_points[0, :] > self.min_X,
                                xyz_points[0, :] < self.max_X)
        mask_Z = np.logical_and(xyz_points[2, :] > self.min_Z,
                                xyz_points[2, :] < self.max_Z)
        mask_XYZ = np.logical_and.reduce((mask_X, mask_Z))
        xyz_points = xyz_points[:, mask_XYZ]
        sseg_points = sseg_points[mask_XYZ]

        x_coord = np.floor(
            (xyz_points[0, :] - self.min_X) / self.cell_size).astype(int)
        y_coord = np.digitize(xyz_points[1, :], self.y_grid)
        z_coord = (self.H - 1) - np.floor(
            (xyz_points[2, :] - self.min_Z) / self.cell_size).astype(int)

        if x_coord.shape[0] > 0:
            self.four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1

            # update the weights for the local map
            self.min_x_coord = min(max(np.min(x_coord) - self.map_boundary, 0),
                                   self.min_x_coord)
            self.max_x_coord = max(
                min(np.max(x_coord) + self.map_boundary, self.W - 1),
                self.max_x_coord)
            self.min_z_coord = min(max(np.min(z_coord) - self.map_boundary, 0),
                                   self.min_z_coord)
            self.max_z_coord = max(
                min(np.max(z_coord) + self.map_boundary, self.H - 1),
                self.max_z_coord)

            self.max_y_coord = max(np.max(y_coord), self.max_y_coord)
            print(f'max_y_coord = {self.max_y_coord}')

        if step_ % self.step_size == 0:
            self.get_semantic_map(step_)

    def get_semantic_map(self, step_):
        """ get the built semantic map. """
        smaller_four_dim_grid = self.four_dim_grid[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                                                   self.min_x_coord:self.max_x_coord + 1, :]
        print(f'smaller_four_dim_grid.shape = {smaller_four_dim_grid.shape}')
        # argmax over the category axis
        zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
        # swap y dim to the last axis
        zxy_grid = np.swapaxes(zyx_grid, 1, 2)
        L, M, N = zxy_grid.shape
        zxy_grid = zxy_grid.reshape(L * M, N)

        semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
        semantic_map = semantic_map.reshape(L, M)
        color_semantic_map = apply_color_to_map(semantic_map)

        # plt.imshow(color_semantic_map)
        # plt.show()
        if semantic_map.shape[0] > 0:
            save_sem_map_through_plt(
                color_semantic_map,
                f'{self.saved_folder}/step_{step_}_semantic.jpg')

    def save_final_map(self, ENLARGE_SIZE=5):
        """ save the built semantic map to a figure."""
        smaller_four_dim_grid = self.four_dim_grid[self.min_z_coord:self.max_z_coord + 1, 0:self.THRESHOLD_HIGH,
                                                   self.min_x_coord:self.max_x_coord + 1, :]
        # argmax over the category axis
        zyx_grid = np.argmax(smaller_four_dim_grid, axis=3)
        # swap y dim to the last axis
        zxy_grid = np.swapaxes(zyx_grid, 1, 2)
        L, M, N = zxy_grid.shape
        zxy_grid = zxy_grid.reshape(L * M, N)

        semantic_map = find_first_nonzero_elem_per_row(zxy_grid)
        semantic_map = semantic_map.reshape(L, M)

        map_dict = {}
        map_dict['min_x'] = self.min_x_coord
        map_dict['max_x'] = self.max_x_coord
        map_dict['min_z'] = self.min_z_coord
        map_dict['max_z'] = self.max_z_coord
        map_dict['min_X'] = self.min_X
        map_dict['max_X'] = self.max_X
        map_dict['min_Z'] = self.min_Z
        map_dict['max_Z'] = self.max_Z
        map_dict['W'] = self.W
        map_dict['H'] = self.H
        map_dict['semantic_map'] = semantic_map
        print(f'semantic_map.shape = {semantic_map.shape}')
        np.save(f'{self.saved_folder}/BEV_semantic_map.npy', map_dict)

        semantic_map = cv2.resize(
            semantic_map,
            (int(semantic_map.shape[1] * ENLARGE_SIZE),
             int(semantic_map.shape[0] * ENLARGE_SIZE)),
            interpolation=cv2.INTER_NEAREST)
        color_semantic_map = apply_color_to_map(semantic_map)
        save_sem_map_through_plt(color_semantic_map,
                                 f'{self.saved_folder}/final_semantic_map.jpg')
