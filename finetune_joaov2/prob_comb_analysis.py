import csv

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def txt_to_csv(txt_file='prob_comb_results.txt'):
    with open(txt_file, 'r') as f:
        lines = f.readlines()[1:]

    content = np.array([l.split(' ')[1::2] for l in [x.replace('LABEL R', 'LABELR') for x in lines]])

    datasets, modes, alphas, beats, lrs, acc_means, acc_stds = [content[:, i].reshape((-1, 1)) for i in range(7)]
    datasets = np.array([ds[0].replace(',', '') for ds in datasets]).reshape((-1, 1))
    lrs = np.array([lr[0].replace('%', '') for lr in lrs]).reshape((-1, 1))
    beats[modes != 'all_aug'] = None

    content_csv = np.concatenate((datasets, modes, alphas, beats, lrs, acc_means, acc_stds), axis=-1)
    headers = ['Dataset', 'Mode', 'Alpha', 'Beta', 'Label Rate ( % )', 'Acc Mean', 'Acc Std']

    with open('prob_comb_results.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerow(headers)
        writer.writerows(content_csv)


if __name__ == '__main__':
    txt_to_csv()
    df = pd.read_csv('prob_comb_results.csv', keep_date_col=True)
    unique_vals = np.unique(list(zip(df['Dataset'], df['Label Rate ( % )'])), axis=0)
    modes_d = {
        'louvain': 1,
        'embeddings': 2,
        'all_aug': 3
    }

    their_res = {
        ('NCI1', 1): (62.52, 1.16),
        ('NCI1', 10): (74.86, .39),
        ('PROTEINS', 10): (73.31, .48)
    }

    for dataset, label_rate in unique_vals:
        if dataset not in ['NCI1', 'PROTEINS']:
            continue
        label_rate = int(label_rate)
        new_df = df[(df['Dataset'] == dataset) & (df['Label Rate ( % )'] == label_rate)].sort_values(by=['Acc Mean'],
                                                                                                     ascending=False)
        # print()
        # print(new_df)
        acc_means, acc_stds = new_df['Acc Mean'].to_numpy(dtype=float) * 100,\
                              new_df['Acc Std'].to_numpy(dtype=float) * 100
        labels = np.array([f'M {modes_d[mode]}\n{a}\n{b if b != "None" else ""}' for mode, a, b in
                           zip(new_df['Mode'].to_list(), new_df['Alpha'].to_list(), new_df['Beta'].to_list())])

        old_aug_mean, old_aug_std = their_res[dataset, label_rate]
        acc_means = np.concatenate(([old_aug_mean], acc_means))
        acc_stds = np.concatenate(([old_aug_std], acc_stds))
        labels = np.concatenate((['their'], labels))

        fig, ax = plt.subplots()
        ax.bar(labels, acc_means, yerr=acc_stds, alpha=.7, capsize=1)
        ax.set_ylabel('Accuracy')
        ax.set_xticks(labels)
        ax.set_title(f'Accuracy (%) - {dataset}, LR = {label_rate}')
        ax.yaxis.grid(True)
        plt.savefig(f'plots/prob_comb_results/{dataset}_{label_rate}.png')
        plt.show()


    # df_PROTEINS, df_NCI1, df_MUTAG = df[df['Dataset'] == 'PROTEINS'].groupby('Label Rate ( % )'),\
    #                                  df[df['Dataset'] == 'NCI1'].groupby('Label Rate ( % )'),\
    #                                  df[df['Dataset'] == 'MUTAG'].groupby('Label Rate ( % )')
    #
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('Best results for NCI1:')
    # print(df_NCI1)
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # print('Best results for PROTEINS:')
    # print(df_PROTEINS)
