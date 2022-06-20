import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    with open('aug_probs.txt', 'r') as f:
        content = f.readlines()

    all_probs = {}

    for l in content:
        tmp = l.split('\t')

        if '~' in l or len(l) == 1:
            continue

        probs = 0

        dataset = tmp[0].split(' ')[-1]
        mode = tmp[1].split(' ')[-1]
        p = tmp[2][:-1].split('\t')[0].split(':')[-1][2:-1].split(' ')

        probs_f = []
        for i, prob in enumerate(p):
            if prob == '':
                continue
            else:
                probs_f.append(float(prob))

        if (dataset, mode) not in all_probs.keys():
            all_probs[dataset, mode] = probs_f
        # else:
        #     all_probs[dataset, mode] = [probs_f]

    # for k, v in all_probs.items():
    #     if len(v) > 1:
    #         avg_value = []
    #         for i in range(len(v[0])):
    #             avg_value.append(numpy.mean([v[j][i] for j in range(len(v))]))
    #             all_probs[k] = avg_value
    #
    # print(all_probs['MUTAG', 'new_aug'])
    # print()

    fig, ax = plt.subplots(1, 2)
    width = 0.2  # the width of the bars
    ds_list = np.unique([ds for ds, _ in all_probs.keys()])
    locate = lambda i, x: [x - width, x, x + width][i]

    for i, mode in enumerate(['new_aug', 'old_aug']):
        labels = [f'{n // int((len(all_probs[ds_list[0], mode]) ** .5))}, {int(n % (len(all_probs[ds_list[0], mode])) ** .5)}'
                  for n in range(len(all_probs[ds_list[0], mode]))]
        x = np.arange(len(labels))  # the label locations

        for j, dataset in enumerate(ds_list):
            ax[i].bar(locate(j, x), all_probs[dataset, mode], width, label=dataset)

        ax[i].set_ylabel('probability')
        ax[i].set_title(f'{mode}' + ' (louvain only)' if mode == 'new_aug' else '')
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(labels)
        ax[i].legend()

    fig.tight_layout()
    plt.savefig('plots/augmentation probabilities.png')
    plt.show()
