import math
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.neighbors import LocalOutlierFactor

from sklearn.manifold import TSNE

import sys
sys.path.append('/home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/pretrain_joaov2/')

import pt_tu_dataset


def calc_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def calc_precision(y_true, y_pred):
    return len(np.where((y_pred == y_true) & y_true)[0]) / len(np.where(y_pred)[0])


def calc_recall(y_true, y_pred):
    return len(np.where((y_pred == y_true) & y_true)[0]) / len(np.where(y_true)[0])


def ad_task(dataset_name,
            prob_comb,
            prob_comb_mode,
            combined,
            new_aug,
            model_func,
            batch_size,
            model_PATH=None, result_PATH=None, result_feat=None):
    '''
    with open(result_PATH, 'r') as f:
        data = f.read().split('\n')[:-1]
    for d in data:
        if result_feat in d:
            assert False
    '''
    if combined:
        aug_txt = 'combined'
    elif new_aug:
        aug_txt = 'new_aug'
    else:
        aug_txt = 'old_aug'

    dataset_obj_path = f'../datasets_obj/{dataset_name}_{prob_comb_mode}.p' if prob_comb\
        else f'../datasets_obj/{dataset_name}_{aug_txt}.p'

    with open(dataset_obj_path, 'rb') as f:
        dataset = pickle.load(f)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    embeddings = torch.Tensor()
    ad_labels = np.array([])

    # # train from scratch
    state_dict = torch.load(model_PATH)
    del state_dict['lin_class.weight']
    del state_dict['lin_class.bias']

    num_aug = max(int(k.split('.')[1]) for k in state_dict.keys() if 'proj_head' in k) + 1
    model = model_func(dataset, num_aug)
    model.load_state_dict(state_dict)

    for i, batch in enumerate(loader):
        batch = batch[0]
        ad_labels = np.concatenate((ad_labels, batch.y))
        out = model(batch)
        embeddings = torch.cat((embeddings, out))

    embeddings = embeddings.detach().numpy()
    ad_labels = 1 - ad_labels.astype(int)

    # num_graphs = len(dataset)
    # num_outliers = num_graphs - sum(dataset.data.y)

    # rand_labels_prob = (num_outliers ** 2 + (num_graphs - num_outliers) ** 2) / (num_graphs ** 2)

    # print(f'Random Labels Accuracy: {rand_labels_prob}')

    outliers_ratio = len(ad_labels[ad_labels == 1]) / len(ad_labels)

    print(f'Outliers ratio: {outliers_ratio}')

    messures = {}

    algs = {
        'Isolation Forest': IsolationForest(contamination=outliers_ratio),
        # 'One-Class SVM': svm.OneClassSVM(nu=outliers_ratio, kernel="rbf", gamma='auto'),
        'Local Outlier Factor': LocalOutlierFactor(contamination=outliers_ratio)
    }

    for alg_name, alg_obj in algs.items():
        y_pred = alg_obj.fit_predict(embeddings)
        y_pred[y_pred == -1] = 0
        y_pred = 1 - y_pred
        precision = calc_precision(ad_labels, y_pred)
        # recall = calc_recall(ad_labels, y_pred)
        messures[alg_name] = np.array([precision])
        # messures[alg_name] = np.array([precision, recall])

    return messures, outliers_ratio


if __name__ == '__main__':
    x_vals, y_vals, true_vals = [], [], []
    p = np.array(range(100)) / 100
    for i in p:
        tmp = []
        for j in range(1000):
            x = np.random.choice([0, 1], 100, p=[i, 1 - i])
            y = np.random.choice([0, 1], 100, p=[i, 1 - i])
            tmp.append(np.mean(x == y))

        x_vals.append(i)
        y_vals.append(np.mean(tmp))
        true_vals.append(i ** 2 + (1 - i) ** 2 + .01)

    plt.plot(x_vals, y_vals)
    plt.plot(x_vals, true_vals)
    plt.show()


    # ~~~~~~~~~~~~~~~~~~ Code for TSNE plot in ad_task ~~~~~~~~~~~~~~~~~~ #
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(embeddings)
    #
    # x, y = tsne_results[:, 0], tsne_results[:, 1]
    # # print(x)
    # # print(y)

    # plt.scatter(x[ad_labels == 0], y[ad_labels == 0], label='Abnormal')
    # plt.scatter(x[ad_labels == 1], y[ad_labels == 1], label='Normal')
    # plt.title('True Labels')
    # plt.savefig('plots/True_Labels')
    # plt.show()

    # plt.scatter(x[outliers_pred == 0], y[outliers_pred == 0], label='Abnormal')
    # plt.scatter(x[outliers_pred == 1], y[outliers_pred == 1], label='Normal')
    # plt.title('Pred Labels')
    # plt.savefig('plots/Pred_Labels')
    # plt.show()

    # print(f'Normalized Accuracy: {normalized_acc}')
