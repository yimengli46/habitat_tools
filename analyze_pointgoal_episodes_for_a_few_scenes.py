import numpy as np
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import gzip
import json
import quaternion as qt
from sklearn.mixture import GaussianMixture

split = 'val'  # 'val', 'train'
saved_folder = 'output/scene_height_distribution'

scene_start_y_dict = {}
scene_height_dict = {}

pointnav_foldername = f'data/datasets/pointnav_hm3d_v1/{split}/content'
'''
point_filenames = ['vLpv2VX547B', 'qk9eeNeR4vw', 'oEPjPNSPmzL', 'gmuS7Wgsbrx', 'ixTj1aTMup2'] 
#['TEEsavR23oF', 'HaxA7YrQdEC', 'wcojb4TFT35', 'k1cupFYWXJ6', 'BHXhpBwSMLh']
sem_filenames = ['00800-TEEsavR23oF', '00801-HaxA7YrQdEC', '00802-wcojb4TFT35', '00803-k1cupFYWXJ6', 
    '00804-BHXhpBwSMLh']
'''


def confirm_nComponents(X, top=1):
    bics = []
    min_bic = 0
    counter = 1
    # at most 3 floors is considered
    maximum_nComponents = min(3, top)
    print(f'max_nC = {maximum_nComponents}')
    opt_bic = 1
    # test the AIC/BIC metric between 1 and 10 components
    for i in range(1, maximum_nComponents + 1):
        gmm = GaussianMixture(n_components=counter, max_iter=1000,
                              random_state=0, covariance_type='full').fit(X)
        bic = gmm.bic(X)
        bics.append(bic)
        if bic < min_bic or min_bic == 0:
            min_bic = bic
            opt_bic = counter

        counter += 1
    return opt_bic


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
        second_slash = json_dir.find('/', first_slash + 1)

        sem_filename = json_dir[first_slash + 1:second_slash]
        point_filename = json_dir[first_slash + 7:second_slash]

        point_filenames.append(point_filename)
        sem_filenames.append(sem_filename)

# ======================================== analyze pointgoal nav episodes =================================
for filename in point_filenames:
    print(f'filename is {filename}.')
    with gzip.open(f'{pointnav_foldername}/{filename}.json.gz', 'rb') as f:
        data = json.loads(f.read())
        episodes = data['episodes']

        # ============================= collect the start point y of each scene =========================
        for episode in episodes:
            scene_id = episode['scene_id']

            pos_slash = scene_id.rfind('/')
            pos_dot = scene_id.rfind('.')-6
            episode_scene = scene_id[pos_slash+1:pos_dot]

            start_pose_y = episode['start_position'][1]

            if episode_scene in scene_start_y_dict:
                scene_start_y_dict[episode_scene].append(start_pose_y)
            else:
                scene_start_y_dict[episode_scene] = [start_pose_y]

# ================================= decide number of floors of each scene ===========================
scene_floor_dict = {}
for idx_scene, scene_with_index in enumerate(sem_filenames):
    height_lst = scene_start_y_dict[scene_with_index[6:]]
    values, counts = np.unique(height_lst, return_counts=True)

    num_floors = confirm_nComponents(
        np.array(height_lst).reshape(-1, 1), top=len(counts))
    print(f'num_floors = {num_floors}')
    print(f'values = {values}, counts = {counts}')

    gm = GaussianMixture(n_components=num_floors).fit(
        np.array(height_lst).reshape(-1, 1))
    sorted_weight_idx = np.argsort(np.array(gm.weights_))[::-1]
    peak_heights = gm.means_[sorted_weight_idx]

    # ================================ summarize the y values of each scene =========================
    scene_floor_dict[scene_with_index] = {}

    count_floor = 0
    for idx_height, height in enumerate(peak_heights):
        # check if within 0.2m of previous heights
        flag = False
        for i in range(0, idx_height):
            prev_height = peak_heights[i]
            if abs(prev_height - height) <= 0.2:
                flag = True

        if not flag:
            scene_floor_dict[scene_with_index][count_floor] = {}
            scene_floor_dict[scene_with_index][count_floor]['y'] = height.item()
            count_floor += 1

    print(f'{scene_floor_dict[scene_with_index]}')

    print(f'------------------------------------------------')


# ===============================assign episodes to each floor =============================
gap_thresh = 0.2
for scene_with_index in sem_filenames:
    print(f'filename is {scene_with_index}.')
    with gzip.open(f'{pointnav_foldername}/{scene_with_index[6:]}.json.gz', 'rb') as f:
        data = json.loads(f.read())
        episodes = data['episodes']

        for episode in episodes:
            scene_id = episode['scene_id']

            pos_slash = scene_id.rfind('/')
            pos_dot = scene_id.rfind('.')-6

            start_pose_y = episode['start_position'][1]

            for idx_floor in list(scene_floor_dict[scene_with_index].keys()):
                floor_y = scene_floor_dict[scene_with_index][idx_floor]['y']
                if abs(start_pose_y - floor_y) < gap_thresh:
                    x = episode['start_position'][0]
                    z = episode['start_position'][2]

                    a, b, c, d = episode['start_rotation']
                    agent_rot = qt.quaternion(a, b, c, d)
                    heading_vector = quaternion_rotate_vector(
                        agent_rot.inverse(), np.array([0, 0, -1]))
                    phi = round(
                        cartesian_to_polar(-heading_vector[2], heading_vector[0])[1], 4)

                    pose = (x, start_pose_y, z, phi)

                    if 'start_pose' in scene_floor_dict[scene_with_index][idx_floor]:
                        scene_floor_dict[scene_with_index][idx_floor]['start_pose'].append(
                            pose)
                    else:
                        scene_floor_dict[scene_with_index][idx_floor]['start_pose'] = [
                            pose]

                    break
# '''

np.save(f'{saved_folder}/{split}_scene_floor_dict.npy', scene_floor_dict)
