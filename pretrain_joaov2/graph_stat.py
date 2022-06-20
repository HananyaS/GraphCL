import os
import csv
import dill
import numpy as np

from graph_measures.features_for_any_graph import *


def generate_txt_files(ds_params):
    root_dir = 'graph_text_files'
    ds_dir = ds_params['name']

    try:
        os.mkdir(root_dir)
    except:
        ...

    try:
        os.mkdir(os.path.join(root_dir, ds_dir))
    except:
        return

    edges_per_graph = ds_params['_source']
    convert_graph_name_to_idx = ds_params['_dict_id']
    convert_node_to_idx = ds_params['nodes_id']

    for g_name, edges_list in edges_per_graph.items():
        g_name_idx = convert_graph_name_to_idx[g_name]
        file_name = os.path.join(root_dir, ds_dir, f"{g_name_idx}.txt")
        lines = [f'{convert_node_to_idx[g_name, in_node]},{convert_node_to_idx[g_name, out_node]}\n'
                 for in_node, out_node, _ in edges_list]

        with open(file_name, 'w') as f:
            f.writelines(lines)


def calc_features_for_graph(graph_txt_path, out_path, feats, directed=True):
    path = graph_txt_path
    head = out_path  # The path in which one would like to keep the pickled features calculated in the process.
    # More options are shown here. For information about them, refer to the file.
    ftr_calc = FeatureCalculator(path, head, feats, acc=False, directed=directed, gpu=True, device=0, verbose=True)
    ftr_calc.calculate_features()


if __name__ == '__main__':
    datasets = ['enron', 'reality_mining', 'twitter_security', 'sec_repo']
    feats = ['degree']

    avg_node_degree = {}
    headers = [
        'Name',
        'Avg Degree',
        'Num Graphs',
        'Num Outliers',
        'Avg Num Nodes (all)',
        'Avg Num Nodes (normal)',
        'Avg Num Nodes (abnormal)',
        'Avg Num Edges (all)',
        'Avg Num Edges (normal)',
        'Avg Num Edges (abnormal)',
        'Is Directed'
     ]

    all_stats = []

    for ds in datasets:
        all_stat_per_ds = [ds]
        all_deg = []
        for g in os.listdir(os.path.join('features_pkl', ds)):
            with open(os.path.join('features_pkl', ds, g, 'degree.pkl'), 'rb') as f:
                deg_g = pickle.load(f)
                all_deg += list(deg_g.reshape((-1)))

        avg_deg_g = np.mean(all_deg)
        avg_node_degree[ds] = avg_deg_g

        all_stat_per_ds.append(avg_deg_g)  # average degree

        with open(f'/home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/ad_pkl_datasets/{ds}',
                  'rb') as dill_f:
            ds_params = dill.load(dill_f)

        ds_labels = list(ds_params['_graph_valid'].values())

        all_stat_per_ds.append(len(ds_labels))  # num graphs in ds
        all_stat_per_ds.append(len(ds_labels) - sum(ds_labels))  # num outliers in ds

        all_stat_per_ds.append(np.mean(list(ds_params['_node_count'].values())))  # avg num nodes - all ds
        all_stat_per_ds.append(np.mean([n for i, n in enumerate(ds_params['_node_count'].values()) if ds_labels[i]])) # avg num nodes - normal graphs
        all_stat_per_ds.append(np.mean([n for i, n in enumerate(ds_params['_node_count'].values()) if not ds_labels[i]])) # avg num nodes - abnormal graphs

        all_stat_per_ds.append(np.mean(list(ds_params['_edge_count'].values())))  # avg num nodes - all ds
        all_stat_per_ds.append(np.mean([n for i, n in enumerate(ds_params['_edge_count'].values()) if ds_labels[i]]))  # avg num edges - normal graphs
        all_stat_per_ds.append(np.mean([n for i, n in enumerate(ds_params['_edge_count'].values()) if not ds_labels[i]]))  # avg num edges - abnormal graphs

        all_stat_per_ds.append(ds_params['_directed'])

        all_stats.append(all_stat_per_ds)

    with open('ds_stats.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(headers)

        # write multiple rows
        writer.writerows(all_stats)
