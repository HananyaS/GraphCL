import os
import time
import os.path as osp
import shutil
from itertools import repeat

from csv import writer
import numpy as np
import networkx as nx
import torch
import torch_geometric.utils as tg_utils
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Batch
from torch_geometric.io import read_tu_data

import random
import pandas as pd
from collections import Counter

import community.community_louvain as community_louvain
from node2vec import Node2Vec


# from new_augmentations import *
# from old_augmentations import *


# tudataset adopted from torch_geometric==1.1.0
class TUDatasetExt(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets, *.e.g.* "IMDB-BINARY",
    "REDDIT-BINARY" or "PROTEINS", collected from the `TU Dortmund University
    <http://graphkernels.cs.tu-dortmund.de>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name <http://graphkernels.cs.tu-dortmund.de>`_ of
            the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        use_node_attr (bool, optional): If :obj:`True`, the dataset will
            contain additional continuous node features (if present).
            (default: :obj:`False`)
    """

    url = 'https://ls11-www.cs.uni-dortmund.de/people/morris/' \
          'graphkerneldatasets'

    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 use_node_attr=False,
                 processed_filename='data.pt',
                 new_aug=False,
                 combined=True,
                 prob_comb=True,
                 prob_comb_mode='all_aug',
                 dataset=None):
        if prob_comb:
            assert prob_comb_mode in ['all_aug', 'louvain', 'embeddings']

        self.name = name
        self.processed_filename = processed_filename

        self.set_aug_mode('none')
        if prob_comb:
            if prob_comb_mode == 'all_aug':
                self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, id, remove_by_louvain,
                                      remove_by_embedding]
            elif prob_comb_mode == 'louvain':
                self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, id, remove_by_louvain]
            else:
                self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, id, remove_by_embedding]

        elif combined:
            self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, remove_by_louvain, id]
        elif new_aug:
            # self.augmentations = [remove_by_embedding, lambda x, y:x]
            self.augmentations = [remove_by_louvain, id]
        else:
            self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, id]
        self.set_aug_ratio(0.2)

        num_aug = len(self.augmentations)
        self.set_aug_prob(np.ones(num_aug ** 2) / (num_aug ** 2))
        super(TUDatasetExt, self).__init__(root, transform, pre_transform,
                                           pre_filter)

        if dataset is None:
            self.data, self.slices = torch.load(self.processed_paths[0])

        else:
            self.data, self.slices = dataset.data, dataset.slices

        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]

    @property
    def num_node_labels(self):
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            if self.data.x[:, i:].sum().item() == self.data.x.size(0):
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def raw_file_names(self):
        names = ['A']
        # names = ['A', 'graph_indicator']
        return [f'{name}.txt' for name in names]
        # return ['{}_{}.txt'.format(self.name, name) for name in names]

    @property
    def processed_file_names(self):
        return self.processed_filename

    @property
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0][0].num_node_features

    def download(self):
        path = download_url('{}/{}.zip'.format(self.url, self.name), self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(self.root, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx)[0] for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx)[0] for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def set_aug_mode(self, aug_mode='none'):
        self.aug_mode = aug_mode

    def set_aug_ratio(self, aug_ratio=0.2):
        self.aug_ratio = aug_ratio

    def set_aug_prob(self, prob):
        if prob.ndim == 2:
            prob = prob.reshape(-1)
        self.aug_prob = prob

    def get(self, idx):
        data, data1, data2 = self.data.__class__(), self.data.__class__(), self.data.__class__()
        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes, data1.num_nodes, data2.num_nodes = self.data.__num_nodes__[idx], self.data.__num_nodes__[
                idx], self.data.__num_nodes__[idx]

        for key in self.data.keys:
            if key in self.slices.keys():
                item, slices = self.data[key], self.slices[key]
                if torch.is_tensor(item):
                    s = list(repeat(slice(None), item.dim()))
                    s[self.data.__cat_dim__(key,
                                            item)] = slice(slices[idx],
                                                           slices[idx + 1])
                else:
                    s = slice(slices[idx], slices[idx + 1])
                data[key], data1[key], data2[key] = item[s], item[s], item[s]

        num_augmentations = len(self.augmentations)

        # pre-defined augmentations
        if self.aug_mode == 'none':
            n_aug1, n_aug2 = num_augmentations - 1, num_augmentations - 1
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(num_augmentations ** 2, 1)[0]
            n_aug1, n_aug2 = n_aug // num_augmentations, n_aug % num_augmentations
        elif self.aug_mode == 'sample':
            n_aug = np.random.choice(num_augmentations ** 2, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug // num_augmentations, n_aug % num_augmentations

        data1 = self.augmentations[n_aug1](data1, self.aug_ratio)
        data2 = self.augmentations[n_aug2](data2, self.aug_ratio)

        return data, data1, data2


# def timer(func):
#     def wrapper(*args, **kwargs):
#         start = time.time()
#         result = func(*args, **kwargs)
#         end = time.time()
#         name = func.__name__.upper() if func.__name__ != '<lambda>' else 'IDENTITY'
#         print(f'{name}: took {end - start} seconds for graph with {args[0].x.size()[0]} nodes.')
#         with open('times.csv', 'a', newline='') as f_object:
#             writer_object = writer(f_object)
#             writer_object.writerow([name, args[0].x.size()[0], end - start])
#             f_object.close()
#         return result
#     return wrapper
#
#
# # @timer
# def node_drop(data, aug_ratio):
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     drop_num = int(node_num * aug_ratio)
#
#     idx_perm = np.random.permutation(node_num)
#     idx_nondrop = idx_perm[drop_num:].tolist()
#     idx_nondrop.sort()
#
#     edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
#
#     data.x = data.x[idx_nondrop]
#     data.edge_index = edge_index
#     data.__num_nodes__, _ = data.x.shape
#     return data
#
#
# # @timer
# def subgraph(data, aug_ratio):
#     G = tg_utils.to_networkx(data)
#
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     sub_num = int(node_num * (1-aug_ratio))
#
#     idx_sub = [np.random.randint(node_num, size=1)[0]]
#     idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])
#
#     while len(idx_sub) <= sub_num:
#         if len(idx_neigh) == 0:
#             idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
#             idx_neigh = set([np.random.choice(idx_unsub)])
#         sample_node = np.random.choice(list(idx_neigh))
#
#         idx_sub.append(sample_node)
#         idx_neigh = idx_neigh.union(set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))
#
#     idx_nondrop = idx_sub
#     idx_nondrop.sort()
#
#     edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)
#
#     data.x = data.x[idx_nondrop]
#     data.edge_index = edge_index
#     data.__num_nodes__, _ = data.x.shape
#     return data
#
#
# # @timer
# def edge_pert(data, aug_ratio):
#     node_num, _ = data.x.size()
#     _, edge_num = data.edge_index.size()
#     pert_num = int(edge_num * aug_ratio)
#
#     edge_index = data.edge_index[:, np.random.choice(edge_num, (edge_num - pert_num), replace=False)]
#
#     idx_add = np.random.choice(node_num, (2, pert_num))
#     adj = torch.zeros((node_num, node_num))
#     adj[edge_index[0], edge_index[1]] = 1
#     adj[idx_add[0], idx_add[1]] = 1
#     adj[np.arange(node_num), np.arange(node_num)] = 0
#     edge_index = adj.nonzero(as_tuple=False).t()
#
#     data.edge_index = edge_index
#     return data
#
#
# # @timer
# def attr_mask(data, aug_ratio):
#     node_num, _ = data.x.size()
#     mask_num = int(node_num * aug_ratio)
#     _x = data.x.clone()
#
#     token = data.x.mean(dim=0)
#     idx_mask = np.random.choice(node_num, mask_num, replace=False)
#
#     _x[idx_mask] = token
#     data.x = _x
#     return data


def custom_collate(data_list):
    batch = Batch.from_data_list([d[0] for d in data_list], follow_batch=['edge_index', 'edge_index_neg'])
    batch_1 = Batch.from_data_list([d[1] for d in data_list])
    batch_2 = Batch.from_data_list([d[2] for d in data_list])
    return batch, batch_1, batch_2


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
    sub_num = int(node_num * (1 - aug_ratio))

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

    edge_index, _ = tg_utils.subgraph(idx_nondrop, data.edge_index, relabel_nodes=True, num_nodes=node_num)

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

    token = data.x.mean(dim=0)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)

    _x[idx_mask] = token
    data.x = _x
    return data


def id(x, y):
    return x


# @timer
def remove_by_louvain(data, ignored_arg,
                      min_community_size_ratio: float = .1,
                      removing_ratio_range=(.2, .6)):
    start_time = time.time()
    G = tg_utils.to_networkx(data).to_undirected()

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    # initialize the resolution parameter of Louvain algorithm
    louvain_resolution = 1
    n_tries = 0

    # search for the value of the resolution which leads to an appropriate number of removed nodes
    # higher value causes less separation of the data
    while n_tries < 10:
        n_tries += 1

        partition = community_louvain.best_partition(G, resolution=louvain_resolution)
        threshold = len(G.nodes) * min_community_size_ratio
        counters = dict(Counter(partition.values()).items())
        communities_to_remove = list(filter(lambda x: counters.get(x) < threshold, counters.keys()))
        nodes_to_remain = [k for k in partition.keys() if partition.get(k) not in communities_to_remove]
        nodes_to_remove = list(set(range(G.number_of_nodes())) - set(nodes_to_remain))

        if removing_ratio_range[0] <= 1 - len(nodes_to_remain) / G.number_of_nodes() <= removing_ratio_range[1]:
            break
        elif 1 - len(nodes_to_remain) / G.number_of_nodes() > removing_ratio_range[1]:
            louvain_resolution += .1
        else:
            louvain_resolution -= .1

    if 1 - len(nodes_to_remain) / G.number_of_nodes() < removing_ratio_range[0]:
        while 1 - len(nodes_to_remain) / G.number_of_nodes() < removing_ratio_range[0]:
            # print(f'searching... now got {1 - len(nodes_to_remain) / G.number_of_nodes()}, num communities to remove:'
            #       f' {len(communities_to_remove)}')
            communities_to_remove.append(min(list(set(counters.keys() - set(communities_to_remove))),
                                             key=lambda k: counters[k]))
            nodes_to_remain = [k for k in partition.keys() if partition.get(k) not in communities_to_remove]
            nodes_to_remove = list(set(range(G.number_of_nodes())) - set(nodes_to_remain))

        # print(f'Found! got {1 - len(nodes_to_remain) / G.number_of_nodes()} - {len(communities_to_remove)} communities'
        #       f' removed')

    if 1 - len(nodes_to_remain) / G.number_of_nodes() > removing_ratio_range[1]:
        nodes_to_add = list(random.sample(nodes_to_remove,
                                          int(len(nodes_to_remove) - removing_ratio_range[1] * node_num)))
        nodes_to_remain += nodes_to_add
        # nodes_to_remove = list(set(nodes_to_remove) - set(nodes_to_add))

    nodes_to_remain.sort()

    # print(f'{len(nodes_to_remove) / node_num} of the nodes removed.')

    edge_index, _ = tg_utils.subgraph(nodes_to_remain, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[nodes_to_remain]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape

    # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
    #       f'REMOVE BY LOUVAIN:\t{time.time() - start_time} seconds\n'
    #       f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    return data


# @timer
def remove_by_embedding(data, ignored_arg,
                        n_dimensions: int = 10,
                        dim: int = None):
    start_time = time.time()
    assert not dim or dim in range(n_dimensions)

    G = tg_utils.to_networkx(data).to_undirected()

    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()

    if not dim:
        dim = random.randrange(n_dimensions)
        # print(f'Chosen dimension: {dim}')

    # Generate walks
    node2vec = Node2Vec(G, dimensions=n_dimensions, quiet=True)
    # Learn embedding
    model = node2vec.fit(window=10, min_count=1)

    # get the nodes' embeddings
    node_embeddings = model.wv.vectors

    df_embedding = pd.DataFrame(node_embeddings)

    # center the embeddings
    means = df_embedding.describe().iloc[1]

    for i, col in enumerate(df_embedding.columns):
        df_embedding[col] -= means[i]

    nodes_to_remain = [i for i in range(df_embedding.shape[0]) if df_embedding[df_embedding.columns[dim]][i] >= 0]

    # print(nodes_to_remain)
    # print(f'REMOVE BT EMBEDDING: {1 - len(nodes_to_remain) / df_embedding.shape[0]} of the nodes should be removed.')

    nodes_to_remain.sort()

    edge_index, _ = tg_utils.subgraph(nodes_to_remain, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[nodes_to_remain]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape

    # print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
    #       f'REMOVE BY EMBEDDING:\t{time.time() - start_time} seconds\n'
    #       f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #
    return data


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True,
                 **kwargs):
        super(DataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn=custom_collate, **kwargs)
