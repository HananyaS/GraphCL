import os.path as osp
import re

import torch
from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import degree
import torch_geometric.transforms as T
import torch_geometric.transforms.to_undirected as to_undirected
from feature_expansion import FeatureExpander
from pt_tu_dataset import TUDatasetExt
from ad_dataset import AD_Dataset
from torch_geometric.datasets import UPFD


def get_dataset(name, sparse=True, feat_str="deg+ak3+reall", root=None, new_aug: bool = False, develop: bool = False,
                combined: bool = True, prob_comb: bool = True, prob_comb_mode: str = 'all_aug',
                tu_dataset: bool = False):
    if not tu_dataset:
        dataset = AD_Dataset(name,
                             new_aug=new_aug,
                             combined=combined,
                             prob_comb=prob_comb,
                             prob_comb_mode=prob_comb_mode)

    else:
        if root is None or root == '':
            path = osp.join(osp.expanduser('~'), 'pyG_data', name)
        else:
            path = osp.join(root, name)
        path = '../' + path
        degree = feat_str.find("deg") >= 0
        onehot_maxdeg = re.findall("odeg(\d+)", feat_str)
        onehot_maxdeg = int(onehot_maxdeg[0]) if onehot_maxdeg else None
        k = re.findall("an{0,1}k(\d+)", feat_str)
        k = int(k[0]) if k else 0
        groupd = re.findall("groupd(\d+)", feat_str)
        groupd = int(groupd[0]) if groupd else 0
        remove_edges = re.findall("re(\w+)", feat_str)
        remove_edges = remove_edges[0] if remove_edges else 'none'
        edge_noises_add = re.findall("randa([\d\.]+)", feat_str)
        edge_noises_add = float(edge_noises_add[0]) if edge_noises_add else 0
        edge_noises_delete = re.findall("randd([\d\.]+)", feat_str)
        edge_noises_delete = float(
            edge_noises_delete[0]) if edge_noises_delete else 0
        centrality = feat_str.find("cent") >= 0
        coord = feat_str.find("coord") >= 0

        pre_transform = FeatureExpander(
            degree=degree, onehot_maxdeg=onehot_maxdeg, AK=k,
            centrality=centrality, remove_edges=remove_edges,
            edge_noises_add=edge_noises_add, edge_noises_delete=edge_noises_delete,
            group_degree=groupd).transform

        # dataset = UPFD(path, 'politifact', 'bert', 'test', transform=to_undirected, pre_transform=pre_transform)

        dataset = TUDatasetExt(
            path, name, pre_transform=pre_transform,
            use_node_attr=not develop, processed_filename="data_%s. pt" % feat_str, new_aug=new_aug, combined=combined,
            prob_comb=prob_comb, prob_comb_mode=prob_comb_mode)
            # use_node_attr=not develop, processed_filename='bert/pre_transform.pt', new_aug=new_aug, combined=combined,
            # prob_comb=prob_comb, prob_comb_mode=prob_comb_mode, dataset=dataset)

    dataset.data.edge_attr = None

    return dataset
