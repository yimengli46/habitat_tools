import os
import numpy as np
import numpy.linalg as LA
from math import cos, sin, pi, floor, tan
import matplotlib.pyplot as plt
from .constants import d3_41_colors_rgb, MP3D_CATEGORIES, colormap
from core import cfg


def wrap_angle(angle):
    """Wrap the angle to be from -pi to pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


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
    """
    Project pixels in sseg_img into camera frame given depth image current_depth and camera pose current_pose.
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
    points_3d = points_4d[:3, :]

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
    """
    Project pixels in sseg_img into world frame given depth image current_depth and camera pose current_pose.
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


def convert_insseg_to_sseg(insseg, ins2cat_dict):
    """
    convert instance segmentation image InsSeg (generated by Habitat Simulator) into Semantic segmentation image SSeg,
    given the mapping from instance to category ins2cat_dict.
    """
    ins_id_list = list(ins2cat_dict.keys())
    sseg = np.zeros(insseg.shape, dtype=np.int16)
    for ins_id in ins_id_list:
        sseg = np.where(insseg == ins_id, ins2cat_dict[ins_id], sseg)
    return sseg


# if # of classes is <= 41, flag_small_categories is True
def apply_color_to_map(semantic_map, dataset='MP3D'):
    """ convert semantic map semantic_map into a colorful visualization color_semantic_map"""
    assert len(semantic_map.shape) == 2
    if dataset == 'MP3D':
        COLOR = d3_41_colors_rgb
        num_classes = 41
    elif dataset == 'HM3D':
        COLOR = colormap(rgb=True)
        num_classes = 300
    else:
        raise NotImplementedError(
            f"Dataset {dataset} not currently supported.")

    H, W = semantic_map.shape
    color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
    for i in range(num_classes):
        if dataset == 'MP3D':
            color_semantic_map[semantic_map == i] = COLOR[i]
        elif dataset == 'HM3D':
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


def read_sem_map_npy(map_npy):
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
    map_data = {}
    map_data['semantic_map'] = semantic_map
    map_data['pose_range'] = (min_X, min_Z, max_X, max_Z)
    map_data['coords_range'] = (min_x, min_z, max_x, max_z)
    map_data['wh'] = (W, H)
    return map_data


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
    map_data = {}
    map_data['occupancy_map'] = occ_map
    map_data['pose_range'] = (min_X, min_Z, max_X, max_Z)
    map_data['coords_range'] = (min_x, min_z, max_x, max_z)
    map_data['wh'] = (W, H)
    return map_data


def get_class_mapper(dataset='gibson'):
    """
    generate the mapping from category to category idx for dataset Gibson 'gibson' and MP3D dataset as 'mp3d'
    """
    class_dict = {}
    if dataset == 'mp3d':
        categories = MP3D_CATEGORIES
    else:
        raise NotImplementedError(
            f"Dataset '{dataset}' not currently supported.")
    class_dict = {v: k + 1 for k, v in enumerate(categories)}
    return class_dict


def coords_to_pose(coords,
                   map_data,
                   cell_size=cfg.SEM_MAP.CELL_SIZE,
                   flag_cropped=True):
    """convert cell location 'coords' on the map to pose (X, Z) in the habitat environment"""
    x, y = coords[:2]

    pose_range = map_data['pose_range']
    coords_range = map_data['coords_range']
    wh = map_data['wh']
    min_X, min_Z, max_X, max_Z = pose_range
    min_x, min_z, max_x, max_z = coords_range

    if flag_cropped:
        X = (x + cell_size / 2 + min_x) * cell_size + min_X
        Z = (wh[0] - (y + cell_size / 2 + min_z)) * cell_size + min_Z
    else:
        X = (x + cell_size / 2) * cell_size + min_X
        Z = (wh[0] - (y + cell_size / 2)) * cell_size + min_Z

    if len(coords) == 3:
        map_yaw = coords[2]
        yaw = -map_yaw
        return (X, Z, yaw)
    else:
        return (X, Z)


def pose_to_coords(cur_pose,
                   map_data,
                   cell_size=cfg.SEM_MAP.CELL_SIZE,
                   flag_cropped=True):
    """
    convert pose (X, Z) in the habitat environment to the cell location 'coords' on the map.
    """
    tx, tz = cur_pose[:2]
    tz = -tz

    pose_range = map_data['pose_range']
    coords_range = map_data['coords_range']
    wh = map_data['wh']

    if flag_cropped:
        x_coord = floor((tx - pose_range[0]) / cell_size - coords_range[0])
        z_coord = floor((wh[0] - (tz - pose_range[1]) / cell_size) -
                        coords_range[1])
    else:
        x_coord = floor((tx - pose_range[0]) / cell_size)
        z_coord = floor(wh[0] - (tz - pose_range[1]) / cell_size)

    if len(cur_pose) == 3:
        yaw = cur_pose[2]
        map_yaw = -yaw
        return (x_coord, z_coord, map_yaw)
    else:
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
