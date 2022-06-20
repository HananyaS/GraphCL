import shlex
from math import log10

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def times_all_aug():
    times_df = pd.read_csv('times.csv', names=['Augmentation', 'Num Nodes', 'Duration Time [sec]'])
    times_df = times_df.groupby('Augmentation')
    for aug in times_df.groups:
        times_for_aug = times_df.get_group(aug)
        times_for_aug = times_for_aug.groupby('Num Nodes')
        num_nodes, mean_duration = [], []
        for aug_and_nodes in times_for_aug.groups:
            # print(times_for_aug.get_group(aug_and_nodes)['Duration Time [sec]'].mean())
            num_nodes.append(int(aug_and_nodes))
            mean_duration.append(log10(times_for_aug.get_group(aug_and_nodes)['Duration Time [sec]'].mean()))
            # print(f"For aug {aug} and num nodes {aug_and_nodes}, we got"
            #       f" {times_for_aug.get_group(aug_and_nodes)['Duration Time [sec]'].mean()}")
        plt.plot(num_nodes, mean_duration, label=aug)

    plt.title('Augmentations Duration Times per Number of Nodes')
    plt.xlabel('Num. of Nodes')
    plt.ylabel('Duration Time [log(sec)]')
    plt.legend()
    plt.savefig('plots/times.png')
    plt.show()


def times_for_embedding_aug():
    with open('remove_by_embedding_times.txt', 'r') as f:
        content = f.readlines()

    times_per_ds = {}

    for l in content:
        ds = l.split(' ')[1].split('\t')[0]
        t = float(l.split(' ')[-1])
        if ds in times_per_ds.keys():
            times_per_ds[ds].append(t)
        else:
            times_per_ds[ds] = [t]

    for k, v in times_per_ds.items():
        times_per_ds[k] = np.mean(v)

    plt.bar(times_per_ds.keys(), times_per_ds.values(), width=.3)
    plt.title('Remove by Embedding - Sec per Epoch')
    plt.savefig('plots/remove_by_embedding_times.png')
    plt.show()


if __name__ == '__main__':
    times_for_embedding_aug()
