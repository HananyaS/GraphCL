import os.path
import pickle

import dill
import numpy as np
import torch
import torch_geometric.loader
import tqdm
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from old_augmentations import *
from new_augmentations import *
from itertools import repeat
from matplotlib import pyplot as plt
import graph_stat as gs


class AD_Dataset(Dataset):
    def __init__(self,
                 name,
                 new_aug=False,
                 combined=True,
                 prob_comb=True,
                 prob_comb_mode='all_aug'):
        if prob_comb:
            assert prob_comb_mode in ['all_aug', 'louvain', 'embeddings']

        self.tu = False
        self.name = name

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
            # self.augmentations = [remove_by_louvain, remove_by_embedding, id]
        else:
            self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, id]
        self.set_aug_ratio(0.2)

        num_aug = len(self.augmentations)
        self.set_aug_prob(np.ones(num_aug ** 2) / (num_aug ** 2))

        self._ds_params = get_ds_params(name)
        feat_path = os.path.join('final_features_pkl', f'{name}.pkl')
        if os.path.isfile(feat_path):
            x = pickle.load(open(feat_path, 'rb'))
        else:
            x = calc_features_for_ds(name)

        x = torch.from_numpy(x)
        # x = torch.from_numpy(x).long()
        y = torch.Tensor(list(self._ds_params['_graph_valid'].values())).long()
        edge_index = convert_edge_index(self._ds_params)

        self.data = Data(x=x, edge_index=edge_index, y=y)

        slices_x = []
        nodes_counter = 0
        for n in self._ds_params['_node_count'].values():
            slices_x.append(nodes_counter)
            nodes_counter += n

        slices_x = torch.Tensor(slices_x).long()

        slices_edge_index = []
        edges_counter = 0
        for e in self._ds_params['_edge_count'].values():
            slices_edge_index.append(edges_counter)
            edges_counter += e

        slices_edge_index = torch.Tensor(slices_edge_index).long()

        slices_y = torch.Tensor(range(slices_x.shape[0])).long()

        self.slices = {
            'x': slices_x,
            'y': slices_y,
            'edge_index': slices_edge_index
        }

        self._indices = None
        self.transform = None

    @property
    def num_features(self):
        return self.data.x.shape[-1]

    @property
    def num_classes(self):
        return torch.unique(self.data.y).shape[0]

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
    def num_node_features(self):
        r"""Returns the number of features per node in the dataset."""
        return self[0][0].num_node_features

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
            data.num_nodes, data1.num_nodes, data2.num_nodes = self.data.__num_nodes__[idx], \
                                                               self.data.__num_nodes__[idx], self.data.__num_nodes__[
                                                                   idx]

        for key in self.data.keys:
            if key in self.slices.keys():
                item, slices = self.data[key], self.slices[key]
                if idx != len(self) - 1:
                    if torch.is_tensor(item):
                        s = list(repeat(slice(None), item.dim()))
                        s[self.data.__cat_dim__(key,
                                                item)] = slice(slices[idx],
                                                               slices[idx + 1])
                    else:
                        s = slice(slices[idx], slices[idx + 1])
                    data[key], data1[key], data2[key] = item[s], item[s], item[s]
                else:
                    if key == 'edge_index':
                        data[key], data1[key], data2[key] = item[:, slices[idx]:], item[:, slices[idx]:], item[:, slices[idx]:]
                    else:
                        data[key], data1[key], data2[key] = item[slices[idx]:], item[slices[idx]:], item[slices[idx]:]

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

    def len(self):
        return self.data.y.shape[0]


def get_ds_params(ds_name):
    with open(f'/home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/ad_pkl_datasets/{ds_name}', 'rb') as dill_f:
        ds_params = dill.load(dill_f)

    ds_params['name'] = ds_name

    nodes_id = {}
    c = 0

    for g, ls in ds_params['_node_lists'].items():
        for n in ls:
            if n not in nodes_id:
                nodes_id[g, n] = c
                c += 1

    ds_params['nodes_id'] = nodes_id
    # print(f"Graph valid equals node count:\t{ds_params['_graph_valid'].keys() == ds_params['_node_count'].keys()}")
    # exit()
    # labels = [ds_params['_graph_valid'][k] for k in ds_params['_node_count']]
    #
    # colors = []
    #
    # for l in labels:
    #     if l:
    #         colors.append('cyan')
    #     else:
    #         colors.append('red')
    #
    # Plot num nodes per graph
    # num_nodes = ds_params['_node_count'].values()
    # x = range(1, len(num_nodes) + 1)
    # plt.scatter(x, num_nodes, c=colors)
    # plt.ylabel('Num Nodes')
    # plt.title('Num Nodes per Graph')
    # plt.show()
    #
    # Plot num edges per graph
    # num_edges = ds_params['_edge_count'].values()
    # x = range(1, len(num_edges) + 1)
    # plt.scatter(x, num_edges, c=colors)
    # plt.ylabel('Num Edges')
    # plt.title('Num Edges per Graph')
    # plt.show()

    return ds_params
    # return AD_Dataset(ds_name, ds_params)


# def convert_edge_index(edges_dict, convert_node_to_idx):
def convert_edge_index(ds_params):
    edges_dict = ds_params['_source']
    convert_node_to_idx = ds_params['nodes_id']
    nodes_count = ds_params['_node_count']

    edge_index = {}
    diff = 0

    for g, edges_g in tqdm.tqdm(edges_dict.items(), desc='iterating over graphs edges'):
        tmp = 0
        for x, y, _ in edges_g:
            x = convert_node_to_idx[g, x] - diff
            y = convert_node_to_idx[g, y] - diff
            tmp += 1

            # edge_index = np.concatenate((edge_index, np.array([x, y])))
            edge_index[g, x, y] = [x, y]

        # diff += tmp
        diff += nodes_count[g]

    # return torch.Tensor(edge_index).T
    return torch.Tensor(list(edge_index.values())).long().T


def calc_features_for_ds(ds):
    # feats = ['degree', 'closeness_centrality']
    # feats = ['degree', 'closeness_centrality', 'motif3']
    feats = ['degree', 'closeness_centrality', 'clustering_coefficient']

    feat_root_path = os.path.join('features_pkl', ds)
    graph_root_path = os.path.join('graph_text_files', ds)

    # generate pickle
    for g in os.listdir(graph_root_path):
        is_exist = True
        for f in feats:
            if not os.path.isfile(os.path.join(feat_root_path, g, f'{f}.pkl')):
                is_exist = False

        if is_exist:
            continue

        g_path = os.path.join(graph_root_path, g)
        out_path = os.path.join('features_pkl', ds, g)

        # calculate the degrees for each graph in the dataset
        ds_params = get_ds_params(ds)
        gs.calc_features_for_graph(g_path, out_path, feats, directed=ds_params['_directed'])

    # all_degrees, all_closeness_centrality = [], []
    all_degrees, all_closeness_centrality, all_clustering_coefficient = [], [], []

    # for g in sorted(os.listdir(graph_root_path), key=lambda g: int(g.split('.')[0])):
    for g in sorted(os.listdir(graph_root_path), key=lambda g: int(g.split('.')[0])):
        for f in feats:
            with open(os.path.join(feat_root_path, g, f'{f}.pkl'), 'rb') as pf:
                feat = pickle.load(pf)

            if f == 'degree':
                if np.std(feat) != 0:
                    feat = (feat - np.mean(feat)) / np.std(feat)

                    for d in feat:
                        all_degrees.append(d[0])

                else:
                    all_degrees += list(np.zeros(len(feat)))

            elif f == 'closeness_centrality':
                feat = np.array(list(feat.features.values()))
                if np.std(feat) != 0:
                    feat = (feat - np.mean(feat)) / np.std(feat)
                else:
                    feat = np.zeros(len(feat))
                all_closeness_centrality += list(feat)
            else:
                feat = np.array(list(feat.features.values()))
                if np.std(feat) != 0:
                    feat = (feat - np.mean(feat)) / np.std(feat)
                else:
                    feat = np.zeros(len(feat))
                all_clustering_coefficient += list(feat)

    # all_features = np.array([all_degrees, all_closeness_centrality]).T
    # all_features = np.array([all_degrees, all_closeness_centrality, all_motifs]).T
    all_features = np.array([all_degrees, all_closeness_centrality, all_clustering_coefficient]).T
    # print(all_features.shape)

    res_root_path = 'final_features_pkl'

    try:
        os.mkdir(res_root_path)
    except:
        ...

    with open(os.path.join(res_root_path, f'{ds}.pkl'), 'wb') as pf:
        pickle.dump(all_features, pf)
        print('Features saved at: ' + os.path.join(res_root_path, ds))

    return all_features


if __name__ == '__main__':
    ds = AD_Dataset('enron')
