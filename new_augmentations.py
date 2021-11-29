import time
from csv import writer
import random
import pandas as pd
import torch_geometric.utils as tg_utils

from collections import Counter
import community.community_louvain as community_louvain

from node2vec import Node2Vec


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
def remove_by_louvain(data,
                      min_community_size_ratio: float = .1,
                      removing_ratio_range=(.2, .6)):
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
        nodes_to_remove = list(set(nodes_to_remove) - set(nodes_to_add))


    nodes_to_remain.sort()

    # print(f'{len(nodes_to_remove) / node_num} of the nodes removed.')

    edge_index, _ = tg_utils.subgraph(nodes_to_remain, data.edge_index, relabel_nodes=True, num_nodes=node_num)

    data.x = data.x[nodes_to_remain]
    data.edge_index = edge_index
    data.__num_nodes__, _ = data.x.shape

    return data


# @timer
def remove_by_embedding(data,
                        n_dimensions: int = 10,
                        dim: int = None):
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

    return data

