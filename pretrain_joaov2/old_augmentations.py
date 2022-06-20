import time
from csv import writer
import numpy as np
import torch_geometric.utils as tg_utils
import torch


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        name = func.__name__.upper() if func.__name__ != '<lambda>' else 'IDENTITY'
        print(f'{name}: took {end - start} seconds for graph with {args[0].x.size()[0]} nodes.')
        with open('times.csv', 'a', newline='') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow([name, args[0].x.size()[0], end - start])
            f_object.close()
        return result
    return wrapper


# @timer
def node_drop(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    drop_num = int(node_num * aug_ratio)

    idx_perm = np.random.permutation(node_num)
    idx_nondrop = idx_perm[drop_num:].tolist()
    idx_nondrop.sort()
    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape
    return data


# @timer
def subgraph(data, aug_ratio):
    G = tg_utils.to_networkx(data)

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    sub_num = int(node_num * (1-aug_ratio))

    idx_sub = [np.random.randint(node_num, size=1)[0]]
    idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

    while len(idx_sub) <= sub_num:
        if len(idx_neigh) == 0:
            idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
            idx_neigh = set([np.random.choice(idx_unsub)])
        sample_node = np.random.choice(list(idx_neigh))

        idx_sub.append(sample_node)
        idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

    idx_nondrop = idx_sub
    idx_nondrop.sort()

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index.long(), relabel_nodes=True, num_nodes=node_num)
    # edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[idx_nondrop]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape
    return data


# @timer
def edge_pert(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    pert_num = int(edge_num * aug_ratio)

    edge_index = data.edge_index[:, np.random.choice(edge_num, (edge_num - pert_num), replace=False)]

    idx_add = np.random.choice(node_num, (2, pert_num))
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_add[0], idx_add[1]] = 1
    adj[np.arange(node_num), np.arange(node_num)] = 0
    edge_index = adj.nonzero(as_tuple=False).t()

    data.edge_index = edge_index
    return data


# @timer
def attr_mask(data, aug_ratio):
    node_num, _ = data.x.size()
    mask_num = int(node_num * aug_ratio)
    _x = data.x.clone()

    # token = data.x.float().mean(dim=0)
    token = (data.x * 1.0).mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    # _x[idx_mask] = token.long()
    _x[idx_mask] = token
    data.x = _x
    return data


def id(x, y):
    return x
