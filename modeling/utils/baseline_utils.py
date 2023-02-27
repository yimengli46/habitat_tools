import os
import numpy as np
import numpy.linalg as LA
import math
from math import cos, sin, pi, floor, tan
import matplotlib.pyplot as plt
from .constants import coco_categories_mapping, panopticSeg_mapping, d3_41_colors_rgb, COCO_74_COLORS, colormap
from core import cfg


def minus_theta_fn(previous_theta, current_theta):
    """ compute angle current_theta minus angle previous theta."""
    result = current_theta - previous_theta
    if result < -math.pi:
        result += 2 * math.pi
    if result > math.pi:
        result -= 2 * math.pi
    return result


def plus_theta_fn(previous_theta, current_theta):
    """ compute angle current_theta plus angle previous theta."""
    result = current_theta + previous_theta
    if result < -math.pi:
        result += 2 * math.pi
    if result > math.pi:
        result -= 2 * math.pi
    return result


def project_pixels_to_camera_coords(sseg_img,
                                    current_depth,
                                    current_pose,
                                    gap=2,
                                    FOV=90,
                                    cx=320,
                                    cy=240,
                                    resolution_x=640,
                                    resolution_y=480,
                                    ignored_classes=[]):
    """Project pixels in sseg_img into camera frame given depth image current_depth and camera pose current_pose.

XYZ = K.inv((u, v))
"""
    # camera intrinsic matrix
    FOV = 79
    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    # first compute the rotation and translation from current frame to goal frame
    # then compute the transformation matrix from goal frame to current frame
    # thransformation matrix is the camera2's extrinsic matrix
    tx, tz, theta = current_pose
    R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                  [-sin(theta), 0, cos(theta)]])
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # build the point matrix
    x = range(0, resolution_x, gap)
    y = range(0, resolution_y, gap)
    xv, yv = np.meshgrid(np.array(x), np.array(y))
    Z = current_depth[yv.flatten(),
                      xv.flatten()].reshape(yv.shape[0], yv.shape[1])
    points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
    points_4d[:, :, 0] = xv
    points_4d[:, :, 1] = yv
    points_4d[:, :, 2] = Z
    points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1))  # 4 x N

    # apply intrinsic matrix
    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    # transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
    print('points_4d.shape = {}'.format(points_4d.shape))
    points_3d = points_4d[:3, :]
    print('points_3d.shape = {}'.format(points_3d.shape))

    # pick x-row and z-row
    sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

    # ignore some classes points
    for c in ignored_classes:
        good = (sseg_points != c)
        sseg_points = sseg_points[good]
        points_3d = points_3d[:, good]

    return points_3d, sseg_points.astype(int)


def project_pixels_to_world_coords(sseg_img,
                                   current_depth,
                                   current_pose,
                                   gap=2,
                                   FOV=79,
                                   cx=320,
                                   cy=240,
                                   theta_x=0.0,
                                   resolution_x=640,
                                   resolution_y=480,
                                   ignored_classes=[],
                                   sensor_height=cfg.SENSOR.SENSOR_HEIGHT):
    """Project pixels in sseg_img into world frame given depth image current_depth and camera pose current_pose.

(u, v) = KRT(XYZ)
"""

    # camera intrinsic matrix
    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    # first compute the rotation and translation from current frame to goal frame
    # then compute the transformation matrix from goal frame to current frame
    # thransformation matrix is the camera2's extrinsic matrix
    tx, tz, theta = current_pose

    R_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])
    # used when I tilt the camera up/down
    R_x = np.array([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)],
                    [0, sin(theta_x), cos(theta_x)]])
    R = R_y.dot(R_x)
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # build the point matrix
    x = range(0, resolution_x, gap)
    y = range(0, resolution_y, gap)
    xv, yv = np.meshgrid(np.array(x), np.array(y))
    Z = current_depth[yv.flatten(),
                      xv.flatten()].reshape(yv.shape[0], yv.shape[1])
    points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
    points_4d[:, :, 0] = xv
    points_4d[:, :, 1] = yv
    points_4d[:, :, 2] = Z
    points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1))  # 4 x N

    # apply intrinsic matrix
    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    # transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
    points_3d = transformation_matrix.dot(points_4d)

    # reverse y-dim and add sensor height
    points_3d[1, :] = points_3d[1, :] * -1 + sensor_height

    # ignore some artifacts points with depth == 0
    depth_points = current_depth[yv.flatten(), xv.flatten()].flatten()
    good = np.logical_and(depth_points > cfg.SENSOR.DEPTH_MIN,
                          depth_points < cfg.SENSOR.DEPTH_MAX)

    points_3d = points_3d[:, good]

    # pick x-row and z-row
    sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()
    sseg_points = sseg_points[good]

    # ignore some classes points

    for c in ignored_classes:
        good = (sseg_points != c)
        sseg_points = sseg_points[good]
        points_3d = points_3d[:, good]

    return points_3d, sseg_points.astype(int)


def convertInsSegToSSeg(InsSeg, ins2cat_dict):
    """convert instance segmentation image InsSeg (generated by Habitat Simulator) into Semantic segmentation image SSeg,
    given the mapping from instance to category ins2cat_dict.
    """
    ins_id_list = list(ins2cat_dict.keys())
    SSeg = np.zeros(InsSeg.shape, dtype=np.int16)
    for ins_id in ins_id_list:
        SSeg = np.where(InsSeg == ins_id, ins2cat_dict[ins_id], SSeg)
    return SSeg


def convertMaskRCNNToSSeg(detectron2_npy, H=480, W=640, det_thresh=0.5):
    """convert Detectron2 detected instance segmentation image InsSeg into Semantic segmentation image SSeg,
given the mapping from instance to category ins2cat_dict.
"""

    SSeg = np.zeros((H, W), dtype=np.int32)  # 15 semantic categories
    idxs = list(range(len(detectron2_npy['classes'])))

    for j in idxs[::-1]:
        class_idx = detectron2_npy['classes'][j]
        score = detectron2_npy['scores'][j]

        if class_idx in list(
                coco_categories_mapping.keys()) and score > det_thresh:
            idx = coco_categories_mapping[
                class_idx] + 1  # first class has index 0
            obj_mask = detectron2_npy['masks'][j]
            SSeg = np.where(obj_mask, idx, SSeg)
    return SSeg


def convertPanopSegToSSeg(PanopSeg, id2cat_dict):
    """ convert panoptic segmentation image PanopSeg into Semantic Segmentation image PanopSeg"""
    SSeg = np.zeros(PanopSeg.shape, dtype=np.int32)
    for cat_id in list(panopticSeg_mapping.keys()):
        mapped_cat_id = panopticSeg_mapping[cat_id]
        SSeg = np.where(PanopSeg == cat_id, mapped_cat_id, SSeg)

    return SSeg


# if # of classes is <= 41, flag_small_categories is True
def apply_color_to_map(semantic_map, type_categories='MP3D'):
    """ convert semantic map semantic_map into a colorful visualization color_semantic_map"""
    assert len(semantic_map.shape) == 2
    if type_categories == 'MP3D':
        COLOR = d3_41_colors_rgb
        num_classes = 41
    elif type_categories == 'COCO':
        COLOR = COCO_74_COLORS
        num_classes = 74
    elif type_categories == 'LVIS':
        COLOR = colormap(rgb=True)
        num_classes = 300

    H, W = semantic_map.shape
    color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
    for i in range(num_classes):
        if type_categories == 'MP3D' or type_categories == 'COCO':
            color_semantic_map[semantic_map == i] = COLOR[i]
        else:
            color_semantic_map[semantic_map == i] = COLOR[i % len(COLOR), 0:3]
    return color_semantic_map


def create_folder(folder_name, clean_up=False):
    """ create folder with directory folder_name.

    If the folder exists before creation, setup clean_up to True to remove files in the folder.
    """
    flag_exist = os.path.isdir(folder_name)
    if not flag_exist:
        print('{} folder does not exist, so create one.'.format(folder_name))
        os.makedirs(folder_name)
    else:
        print('{} folder already exists, so do nothing.'.format(folder_name))
        if clean_up:
            os.system('rm {}/*.png'.format(folder_name))
            os.system('rm {}/*.npy'.format(folder_name))
            os.system('rm {}/*.jpg'.format(folder_name))


def read_map_npy(map_npy):
    """ read saved semantic map numpy file infomation."""
    min_x = map_npy['min_x']
    max_x = map_npy['max_x']
    min_z = map_npy['min_z']
    max_z = map_npy['max_z']
    min_X = map_npy['min_X']
    max_X = map_npy['max_X']
    min_Z = map_npy['min_Z']
    max_Z = map_npy['max_Z']
    W = map_npy['W']
    H = map_npy['H']
    semantic_map = map_npy['semantic_map']
    return semantic_map, (min_X, min_Z, max_X, max_Z), (min_x, min_z, max_x,
                                                        max_z), (W, H)


def read_occ_map_npy(map_npy):
    """ read saved occupancy map numpy file infomation."""
    min_x = map_npy['min_x']
    max_x = map_npy['max_x']
    min_z = map_npy['min_z']
    max_z = map_npy['max_z']
    min_X = map_npy['min_X']
    max_X = map_npy['max_X']
    min_Z = map_npy['min_Z']
    max_Z = map_npy['max_Z']
    occ_map = map_npy['occupancy']
    W = map_npy['W']
    H = map_npy['H']
    return occ_map, (min_X, min_Z, max_X, max_Z), (min_x, min_z, max_x,
                                                   max_z), (W, H)


def semanticMap_to_binary(sem_map):
    """ convert semantic map to type 'int8' """
    sem_map.astype('uint8')
    sem_map[sem_map != 2] = 0
    sem_map[sem_map == 2] = 255
    return sem_map


def get_class_mapper(dataset='gibson'):
    """ generate the mapping from category to category idx for dataset Gibson 'gibson' and MP3D dataset as 'mp3d'"""
    class_dict = {}
    if dataset == 'mp3d':
        categories = ['void', 'wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa', 'bed',
                      'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool', 'towel', 'mirror', 'tv_monitor',
                      'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting', 'beam', 'railing', 'shelving', 'blinds', 'gym_equipment',
                      'seating', 'board_panel', 'furniture', 'appliances', 'clothes', 'objects', 'misc']
    elif dataset == 'gibson':
        categories = list(
            np.load(f'{cfg.PF.SEMANTIC_PRIOR_PATH}/all_objs_list.npy',
                    allow_pickle=True))
    class_dict = {v: k + 1 for k, v in enumerate(categories)}
    return class_dict


def pxl_coords_to_pose(coords,
                       pose_range,
                       coords_range,
                       WH,
                       cell_size=cfg.SEM_MAP.CELL_SIZE,
                       flag_cropped=True):
    """convert cell location 'coords' on the map to pose (X, Z) in the habitat environment"""
    x, y = coords
    min_X, min_Z, max_X, max_Z = pose_range
    min_x, min_z, max_x, max_z = coords_range

    if flag_cropped:
        X = (x + cell_size/2 + min_x) * cell_size + min_X
        Z = (WH[0] - (y + cell_size/2 + min_z)) * cell_size + min_Z
    else:
        X = (x + cell_size/2) * cell_size + min_X
        Z = (WH[0] - (y + cell_size/2)) * cell_size + min_Z
    return (X, Z)


def pose_to_coords(cur_pose,
                   pose_range,
                   coords_range,
                   WH,
                   cell_size=cfg.SEM_MAP.CELL_SIZE,
                   flag_cropped=True):
    """convert pose (X, Z) in the habitat environment to the cell location 'coords' on the map"""
    tx, tz = cur_pose[:2]

    if flag_cropped:
        x_coord = floor((tx - pose_range[0]) / cell_size - coords_range[0])
        z_coord = floor((WH[0] - (tz - pose_range[1]) / cell_size) -
                        coords_range[1])
    else:
        x_coord = floor((tx - pose_range[0]) / cell_size)
        z_coord = floor(WH[0] - (tz - pose_range[1]) / cell_size)

    return (x_coord, z_coord)


def save_sem_map_through_plt(img, name):
    """ save the figure img at directory 'name' using matplotlib"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(name)
    plt.close()


def save_occ_map_through_plt(img, name):
    """ save the figure img at directory 'name' using matplotlib"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(name)
    plt.close()
