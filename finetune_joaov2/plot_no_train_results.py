import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    modes = ['old_aug', 'new_aug', 'combined']

    with open('no_train_results.txt', 'r') as f:
        content = f.readlines()

    acc_df = pd.DataFrame(columns=['Dataset', 'Prob Mode', 'Aug Mode', 'Accuracy Mean', 'Accuracy Std'])
    acc_dict = {}

    for c in content:
        prob_mode, dataset, aug_mode, acc = c.split('\t')
        prob_mode = prob_mode[:-2]
        dataset = dataset.split(' ')[1]
        aug_mode = aug_mode.split(' ')[1]
        acc_mean, acc_std = float(acc.split(' ')[1:][0]), float(acc.split(' ')[1:][-1])
        acc_dict[dataset, prob_mode, aug_mode] = (acc_mean, acc_std)

    for (dataset, prob_mode, aug_mode), (acc_mean, acc_std) in acc_dict.items():
        acc_df = acc_df.append({
            'Dataset': dataset,
            'Prob Mode': prob_mode,
            'Aug Mode': aug_mode,
            'Accuracy Mean': acc_mean,
            'Accuracy Std': acc_std
        }, ignore_index=True)

    datasets = np.unique(acc_df['Dataset'].values)
    prob_modes = np.unique(acc_df['Prob Mode'].values)
    # aug_modes = np.unique(acc_df['Aug Mode'].values)
    print(acc_df)
    paper_res = {
        'NCI1': (74.86, 0.39),
        'PROTEINS': (73.31, 0.48)
    }
    # exit()

    for ds in datasets:
        # print(ds)
        old_aug_acc_mean = [acc_df['Accuracy Mean'][(acc_df['Dataset'] == ds) & (acc_df['Aug Mode'] == 'old_aug')]
                            for mode in prob_mode][0].values * 100
        old_aug_acc_std = [acc_df['Accuracy Std'][(acc_df['Dataset'] == ds) & (acc_df['Aug Mode'] == 'old_aug')]
                           for mode in prob_mode][0].values * 100

        new_aug_acc_mean = [acc_df['Accuracy Mean'][(acc_df['Dataset'] == ds) & (acc_df['Aug Mode'] == 'new_aug')]
                            for mode in prob_mode][0].values * 100
        new_aug_acc_std = [acc_df['Accuracy Std'][(acc_df['Dataset'] == ds) & (acc_df['Aug Mode'] == 'new_aug')]
                           for mode in prob_mode][0].values * 100

        combined_acc_mean = [acc_df['Accuracy Mean'][(acc_df['Dataset'] == ds) & (acc_df['Aug Mode'] == 'combined')]
                             for mode in prob_mode][0].values * 100
        combined_acc_std = [acc_df['Accuracy Std'][(acc_df['Dataset'] == ds) & (acc_df['Aug Mode'] == 'combined')]
                            for mode in prob_mode][0].values * 100

        x = np.arange(len(prob_modes))  # the label locations
        width = 0.15  # the width of the bars

        paper_acc_mean, paper_acc_std = paper_res[ds]

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, old_aug_acc_mean, width, yerr=old_aug_acc_std, label='Old Augmentations')
        rects2 = ax.bar(x, new_aug_acc_mean, width, yerr=new_aug_acc_std, label='New Augmentations')
        rects3 = ax.bar(x + width, combined_acc_mean, width, yerr=combined_acc_std, label='Combined')
        # ax.bar(.5, paper_acc_mean, width, yerr=paper_acc_std, label='Paper Results')
        # ax.axhline(y=paper_acc_mean, color='r', linestyle='--')

        if ds == 'PROTEINS':
            ax.bar(.45, paper_acc_mean, width, yerr=paper_acc_std, label='Paper Results')
            # ax.axhline(y=paper_acc_mean, color='r', linestyle='--')

            sota_acc_mean, sota_acc_std = 74.17, 0.34

            ax.bar(.6, sota_acc_mean, width, yerr=sota_acc_std, label='SOTA Results', color='brown')
            ax.axhline(y=sota_acc_mean, linestyle='--', c='brown')

        else:
            ax.bar(.5, paper_acc_mean, width, yerr=paper_acc_std, label='Paper Results', color='r')
            ax.axhline(y=paper_acc_mean, linestyle='--', color='r')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy without Additional Training - {ds}')
        ax.set_xticks(x)
        ax.set_xticklabels(prob_modes)
        ax.legend()

        fig.tight_layout()

        plt.savefig(f'plots/no_train/{ds}')
        plt.show()


