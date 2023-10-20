import numpy as np
import matplotlib.pyplot as plt
import cv2
from modeling.utils.baseline_utils import apply_color_to_map, read_sem_map_npy, read_occ_map_npy, wrap_angle, coords_to_pose
from skimage.morphology import skeletonize
import sknw
from skimage.draw import line
import math
import networkx as nx
import bz2
import _pickle as cPickle


def prune_skeleton_graph(skeleton_G):
    dict_node_numEdges = {}
    for edge in skeleton_G.edges():
        u, v = edge
        for node in [u, v]:
            if node in dict_node_numEdges:
                dict_node_numEdges[node] += 1
            else:
                dict_node_numEdges[node] = 1
    to_prune_nodes = []
    for k, v in dict_node_numEdges.items():
        if v < 2:
            to_prune_nodes.append(k)
    skeleton_G_pruned = skeleton_G.copy()
    skeleton_G_pruned.remove_nodes_from(to_prune_nodes)
    return skeleton_G_pruned


ENLARGE_SIZE = 1
GAP = 20
THRESH_GAP_BETWEEN_RED_NODES = 20
GAP_ANGLE = np.pi / 6.


scene_name = '8WUmhLawc2A'
semantic_map_folder = f'output/semantic_map/{scene_name}'

# ======================================== load the semantic map =======================================
sem_map_npy = np.load(f'{semantic_map_folder}/BEV_semantic_map.npy', allow_pickle=True).item()
sem_map_data = read_sem_map_npy(sem_map_npy)
sem_map = sem_map_data['semantic_map']

occ_map_npy = np.load(f'{semantic_map_folder}/BEV_occupancy_map.npy', allow_pickle=True).item()
occ_map_data = read_occ_map_npy(occ_map_npy)
occ_map = occ_map_data['occupancy_map']

H, W = sem_map.shape

semantic_occupancy_map = cv2.resize(
    sem_map, (int(W * ENLARGE_SIZE), int(H * ENLARGE_SIZE)), interpolation=cv2.INTER_NEAREST)

# ================== colorize the semantic map and merge with occupancy map ==================
color_semantic_map = apply_color_to_map(semantic_occupancy_map)
enlarged_occ_map = cv2.resize(
    occ_map, (W * ENLARGE_SIZE, H * ENLARGE_SIZE), interpolation=cv2.INTER_NEAREST)
# turn free space into white
color_semantic_map[enlarged_occ_map > 0] = np.ones(3) * 255

plt.imshow(color_semantic_map)
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
ax.imshow(color_semantic_map)
waypoints_coords = np.zeros((2, 0), dtype=np.int16)

# ================================ compute the skeleton ==========================
skeleton = skeletonize(enlarged_occ_map)
graph = sknw.build_sknw(skeleton)
graph = prune_skeleton_graph(graph)

edges_nodes = np.zeros((0, 2), dtype=np.int16)
for (s, e) in graph.edges():
    ps = graph[s][e]['pts']
    ps_sparse = ps[GAP:ps.shape[0]:GAP, :]
    edges_nodes = np.concatenate((edges_nodes, ps_sparse))
print(f'collect {edges_nodes.shape[0]} nodes from edges.')
edges_nodes = edges_nodes.transpose()

# draw node by o
nodes = graph.nodes()
ps = np.array([nodes[i]['o'] for i in nodes]).transpose()

node_coords = np.concatenate((ps, edges_nodes), axis=1)[
    [1, 0], :].astype(np.int32)

# ======================== remove red nodes that are close to themselves =============
mask = np.ones(node_coords.shape[1], dtype=bool)
for i in range(node_coords.shape[1]):
    current_node_coords = node_coords[:, i:i + 1]
    dist = np.sqrt(((current_node_coords - node_coords)**2).sum(axis=0))
    current_mask = dist > THRESH_GAP_BETWEEN_RED_NODES
    mask[i + 1:] = current_mask[i + 1:] & mask[i + 1:]

node_coords = node_coords[:, mask]
print(
    f'after removing red nodes too close together: node_coords.shape = {node_coords.shape}')

all_node_coords = np.concatenate((node_coords, waypoints_coords), axis=1)
print(f'In total, we have {all_node_coords.shape[1]} nodes.')

# remove duplicates
nodes_set = set()
all_node_coords = all_node_coords.tolist()
all_node_coords = list(zip(all_node_coords[0], all_node_coords[1]))
for i in range(len(all_node_coords)):
    node = all_node_coords[i]
    nodes_set.add((node[0], node[1]))

all_node_coords = np.zeros((2, len(nodes_set)), dtype=np.int32)
for i, node in enumerate(nodes_set):
    all_node_coords[:, i] = node

# ============================ build edges ===============================
# traverse every pair of nodes, if there are no obstacle between them, add an edge
edges = []

for i in range(all_node_coords.shape[1]):
    for j in range(i + 1, all_node_coords.shape[1]):
        source_node = all_node_coords[:, i:i + 1]
        end_node = all_node_coords[:, j:j + 1]
        # check obstacle between source node and end node
        rr_line, cc_line = line(
            source_node[1, 0], source_node[0, 0], end_node[1, 0], end_node[0, 0])
        line_vals = enlarged_occ_map[rr_line, cc_line]
        if np.all(line_vals):
            edges.append([i, j])
            edges.append([j, i])

# =================== go through each node, check if edge have close angle ================
for i in range(all_node_coords.shape[1]):
    unwanted_edges = []
    current_node_is_source_edges = []
    for edge in edges:
        a, b = edge
        if a == i:
            current_node_is_source_edges.append(edge)

    # check if the edge angle is close
    num_edges = len(current_node_is_source_edges)
    if num_edges > 1:
        dists = []
        for edge in current_node_is_source_edges:
            a, b = edge
            a = all_node_coords[:, a:a + 1]
            b = all_node_coords[:, b:b + 1]
            dist = (a[1, 0] - b[1, 0])**2 + (a[0, 0] - b[0, 0])**2
            dists.append(dist)

        # sort the edges
        dists = np.array(dists)
        edge_idxs = np.argsort(dists)
        # traverse from the short edge to the long edges
        for j, edge_i0 in enumerate(edge_idxs[:-1]):
            for edge_i1 in edge_idxs[j + 1:]:
                a1, b1 = current_node_is_source_edges[edge_i0]
                a2, b2 = current_node_is_source_edges[edge_i1]
                a1_node = all_node_coords[:, a1:a1 + 1]
                b1_node = all_node_coords[:, b1:b1 + 1]
                a2_node = all_node_coords[:, a2:a2 + 1]
                b2_node = all_node_coords[:, b2:b2 + 1]
                angle1 = math.atan2(
                    b1_node[1, 0] - a1_node[1, 0], b1_node[0, 0] - a1_node[0, 0])
                angle2 = math.atan2(
                    b2_node[1, 0] - a2_node[1, 0], b2_node[0, 0] - a2_node[0, 0])
                angle_diff = abs(wrap_angle(angle1 - angle2))
                if angle_diff <= GAP_ANGLE:
                    unwanted_edges.append([a2, b2])
                    num_edges -= 1

    for unwanted_edge in unwanted_edges:
        if unwanted_edge in edges:
            edges.remove(unwanted_edge)

# === go through each node, if node is not in any edge, find the nearest neighbor and connect to it ===
mask_known = semantic_occupancy_map > 0
for i in range(all_node_coords.shape[1]):
    current_node_edges = []

    for edge in edges:
        a, b = edge
        if a == i or b == i:
            current_node_edges.append(edge)

    if len(current_node_edges) == 0:
        current_node = all_node_coords[:, i:i + 1]
        dist = np.sqrt(((current_node - all_node_coords)**2).sum(axis=0))
        node_idxs = np.argsort(dist)
        # go through each node
        for node_idx in node_idxs:
            if node_idx == i:
                continue
            rr_line, cc_line = line(
                current_node[1, 0], current_node[0, 0],
                all_node_coords[:, node_idx:node_idx + 1][1, 0],
                all_node_coords[:, node_idx:node_idx + 1][0, 0])
            line_vals = mask_known[rr_line, cc_line]
            if np.all(line_vals):
                edges.append([i, node_idx])
                break

# ===================== build a connected component, remove the dangling nodes ====================
G = nx.Graph()
for edge in edges:
    G.add_edge(edge[0], edge[1])

largest_cc = list(sorted(nx.connected_components(G), key=len, reverse=True)[0])
all_node_coords = all_node_coords[:, largest_cc]

new_edges = []
for edge in edges:
    a, b = edge
    if a in largest_cc:
        node_i0 = largest_cc.index(a)
        node_i1 = largest_cc.index(b)
        if [node_i0, node_i1] not in new_edges:
            new_edges.append([node_i0, node_i1])
            new_edges.append([node_i1, node_i0])

edges = new_edges

# ================== draw edges ===================
x = all_node_coords[0, :].flatten()
y = all_node_coords[1, :].flatten()
edges = np.array(edges)
ax.plot(x[edges.T], y[edges.T], linestyle='-',
        color='y', markerfacecolor='red', marker='o', zorder=1)
ax.scatter(x=all_node_coords[0, :],
           y=all_node_coords[1, :], c='red', s=30, zorder=2)


ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.tight_layout()
plt.show()

# ===================================== save the graph
all_node_poses = np.zeros((2, all_node_coords.shape[1]), dtype=np.float32)
for i in range(all_node_coords.shape[1]):
    node_coords = all_node_coords[:, i]
    pose = coords_to_pose(node_coords, occ_map_data)
    all_node_poses[:, i] = (pose[0], pose[1])

edges_with_heading = []
for edge in edges:
    source_node = all_node_coords[:, edge[0]]
    end_node = all_node_coords[:, edge[1]]
    heading = math.atan2(
        end_node[1] - source_node[1], end_node[0] - source_node[0])
    edges_with_heading.append([edge[0], edge[1], heading])


graph_data = {}
graph_data['node_coords'] = all_node_coords  # 2 x N
graph_data['node_pose'] = all_node_poses  # 2 x N
graph_data['edges'] = edges_with_heading

with bz2.BZ2File(f'output/graph_{scene_name}.pbz2', 'w') as fp:
    cPickle.dump(
        graph_data,
        fp
    )
