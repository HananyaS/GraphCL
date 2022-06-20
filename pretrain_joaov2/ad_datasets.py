from torch_geometric.datasets import UPFD
from torch_geometric.transforms import to_undirected
from torch_geometric.data import InMemoryDataset
from old_augmentations import *
from new_augmentations import *
from tu_dataset import TUDatasetExt

"""
class Dataset(TUDatasetExt):
    def __init__(*args, dataset):
        # if prob_comb:
        #     assert prob_comb_mode in ['all_aug', 'louvain', 'embeddings']
        #
        # self.name = name
        # self.processed_filename = processed_filename
        #
        # self.set_aug_mode('none')
        # if prob_comb:
        #     if prob_comb_mode == 'all_aug':
        #         self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y:x, remove_by_louvain,
        #                               remove_by_embedding]
        #     elif prob_comb_mode == 'louvain':
        #         self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y:x, remove_by_louvain]
        #     else:
        #         self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y:x, remove_by_embedding]
        #
        # elif combined:
        #     self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, remove_by_louvain, lambda x, y:x]
        # elif new_aug:
        #     self.augmentations = [remove_by_embedding, lambda x, y:x]
        # #    self.augmentations = [remove_by_louvain, lambda x, y:x]
        # else:
        #     self.augmentations = [node_drop, subgraph, edge_pert, attr_mask, lambda x, y:x]
        # self.set_aug_ratio(0.2)
        #
        # num_aug = len(self.augmentations)
        # self.set_aug_prob(np.ones(num_aug ** 2) / (num_aug ** 2))
        # super(Dataset, self).__init__(root, transform, pre_transform,
        #                                    pre_filter)
        
        super(Dataset, self).__init__()
        if dataset is not None:
            self.data, self.slices = dataset.data, dataset.slices
        else:
            self.data, self.slices = torch.load(self.processed_paths[0])

        if self.data.x is not None and not use_node_attr:
            self.data.x = self.data.x[:, self.num_node_attributes:]


if __name__ == '__main__':
    train_dataset = UPFD('../NEW_DATASETS', 'politifact', 'bert', 'train', transform=to_undirected)
    val_dataset = UPFD('../NEW_DATASETS', 'politifact', 'bert', 'val', transform=to_undirected)
    test_dataset = UPFD('../NEW_DATASETS', 'politifact', 'bert', 'test', transform=to_undirected)

    print()
"""

