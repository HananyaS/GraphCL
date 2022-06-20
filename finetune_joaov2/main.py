import os
import json
import re
import argparse
import numpy as np
from utils import logger
from datasets import get_dataset
from res_gcn import ResGCN_graphcl
from train_eval import cross_validation_with_val_set
from matplotlib import pyplot as plt
from csv import writer
import warnings
import pandas as pd

from anomaly_detection import ad_task

warnings.filterwarnings("ignore")

os.chdir('/home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/finetune_joaov2')
print('START FINE-TUNING')


str2bool = lambda x: x.lower() == "true"

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="datasets")
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=str, default="0.1")
parser.add_argument('--epoch_select', type=str, default='test_max')
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)
parser.add_argument('--with_eval_mode', type=str2bool, default=True)

parser.add_argument('--dataset', type=str, default="enron")
# parser.add_argument('--dataset', type=str, default="PROTEINS")
parser.add_argument('--n_splits', type=int, default=10)
parser.add_argument('--suffix', type=str, default="0")
parser.add_argument('--model', type=str, default="graphcl")

parser.add_argument('--pretrain_lr', type=str, default="0.003")
parser.add_argument('--pretrain_epoch', type=str, default="100")
parser.add_argument('--pretrain_gamma', type=str, default="0.1")
parser.add_argument('--pretrain_k_folds', type=int, default=5)

parser.add_argument('--new_aug', type=str2bool, default=False)
parser.add_argument('--combine', type=str2bool, default=False)

parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--use_best_params', type=bool, default=True)
parser.add_argument('--comparison_mode', type=str2bool, default=False)
parser.add_argument('--prob_comb', type=str2bool, default=False)
parser.add_argument('--prob_comb_mode', type=str, default='embeddings')
parser.add_argument('--alpha', type=float, default=.7)
parser.add_argument('--beta', type=float, default=None)

parser.add_argument('--classification', type=str2bool, default=False)
parser.add_argument('--train', type=str2bool, default=False)
parser.add_argument('--calc_joao_probs', type=str2bool, default=False)
parser.add_argument('--ad_comp', type=str2bool, default=True)

args = parser.parse_args()


def create_n_filter_triple(dataset, feat_str, net, gfn_add_ak3=False,
                            gfn_reall=True, reddit_odeg10=False,
                            dd_odeg10_ak1=False):
    # Add ak3 for GFN.
    if gfn_add_ak3 and 'GFN' in net:
        feat_str += '+ak3'
    # Remove edges for GFN.
    if gfn_reall and 'GFN' in net:
        feat_str += '+reall'
    # Replace degree feats for REDDIT datasets (less redundancy, faster).
    if reddit_odeg10 and dataset in [
            'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
    # Replace degree and akx feats for dd (less redundancy, faster).
    if dd_odeg10_ak1 and dataset in ['DD']:
        feat_str = feat_str.replace('odeg100', 'odeg10')
        feat_str = feat_str.replace('ak3', 'ak1')
    return dataset, feat_str, net


def get_model_with_default_configs(model_name,
                                   num_feat_layers=args.n_layers_feat,
                                   num_conv_layers=args.n_layers_conv,
                                   num_fc_layers=args.n_layers_fc,
                                   residual=args.skip_connection,
                                   hidden=args.hidden):
    # More default settings.
    res_branch = args.res_branch
    global_pool = args.global_pool
    dropout = args.dropout
    edge_norm = args.edge_norm

    # modify default architecture when needed
    if model_name.find('_') > 0:
        num_conv_layers_ = re.findall('_conv(\d+)', model_name)
        if len(num_conv_layers_) == 1:
            num_conv_layers = int(num_conv_layers_[0])
            print('[INFO] num_conv_layers set to {} as in {}'.format(
                num_conv_layers, model_name))
        num_fc_layers_ = re.findall('_fc(\d+)', model_name)
        if len(num_fc_layers_) == 1:
            num_fc_layers = int(num_fc_layers_[0])
            print('[INFO] num_fc_layers set to {} as in {}'.format(
                num_fc_layers, model_name))
        residual_ = re.findall('_res(\d+)', model_name)
        if len(residual_) == 1:
            residual = bool(int(residual_[0]))
            print('[INFO] residual set to {} as in {}'.format(
                residual, model_name))
        gating = re.findall('_gating', model_name)
        if len(gating) == 1:
            global_pool += "_gating"
            print('[INFO] add gating to global_pool {} as in {}'.format(
                global_pool, model_name))
        dropout_ = re.findall('_drop([\.\d]+)', model_name)
        if len(dropout_) == 1:
            dropout = float(dropout_[0])
            print('[INFO] dropout set to {} as in {}'.format(
                dropout, model_name))
        hidden_ = re.findall('_dim(\d+)', model_name)
        if len(hidden_) == 1:
            hidden = int(hidden_[0])
            print('[INFO] hidden set to {} as in {}'.format(
                hidden, model_name))

    if model_name == 'ResGCN_graphcl':
        def foo(dataset, num_aug):
            return ResGCN_graphcl(dataset=dataset, hidden=hidden, num_feat_layers=num_feat_layers, num_conv_layers=num_conv_layers,
                          num_fc_layers=num_fc_layers, gfn=False, collapse=False,
                          residual=residual, res_branch=res_branch,
                          global_pool=global_pool, dropout=dropout,
                          edge_norm=edge_norm, num_aug=num_aug, classification=args.classification)
    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo


def run_experiment_finetune(model_PATH, result_PATH, result_feat,
                            dataset_feat_net_triple=create_n_filter_triple(args.dataset, 'deg+odeg100', 'ResGCN_graphcl', gfn_add_ak3=True, reddit_odeg10=True, dd_odeg10_ak1=True),
                            get_model=get_model_with_default_configs,
                            tuned_lr: int = args.lr,
                            epochs: int = None, train: bool = False,
                            prob_comb: bool =args.prob_comb, prob_comb_mode: bool = args.prob_comb_mode,
                            combined: bool = args.combine, new_aug: bool = args.new_aug):
    if not epochs:
        epochs = args.epochs

    dataset_name, feat_str, net = dataset_feat_net_triple
    model_func = get_model(net)
    if args.classification:
        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root)
        # if not args.train:
        #     acc = eval_no_training(
        #         dataset,
        #         model_func,
        #         batch_size=args.batch_size,
        #         with_eval_mode=args.with_eval_mode,
        #         model_PATH=model_PATH)
        #     return acc
        # else:
        return cross_validation_with_val_set(
            dataset,
            model_func,
            epochs=epochs,
            batch_size=args.batch_size,
            lr=tuned_lr,
            weight_decay=0,
            epoch_select=args.epoch_select,
            with_eval_mode=args.with_eval_mode,
            logger=logger,
            model_PATH=model_PATH, n_splits=args.n_splits, result_PATH=result_PATH, result_feat=result_feat,
            train_all=train)

    else:
        ad_task(dataset_name,
                prob_comb,
                prob_comb_mode,
                combined,
                new_aug,
                model_func, args.batch_size, model_PATH=model_PATH)


def run_comparison(dataset=args.dataset, lr_vals=[float(args.pretrain_lr)] * 3,
                   gamma_joao_vals=[float(args.pretrain_gamma)] * 3, use_best_params=args.use_best_params):
    if use_best_params:
        try:
            best_params_json_path_old = f'../pretrain_joaov2/best_params/old_aug/{dataset}.json'
            with open(best_params_json_path_old, 'r') as f:
                old_bp = json.load(f)
                old_lr, old_gamma = old_bp['lr'], old_bp['gamma_joao']

            best_params_json_path_new = f'../pretrain_joaov2/best_params/new_aug/{dataset}.json'
            with open(best_params_json_path_new, 'r') as f:
                new_bp = json.load(f)
                new_lr, new_gamma = new_bp['lr'], new_bp['gamma_joao']

            best_params_json_path_comb = f'../pretrain_joaov2/best_params/combined/{dataset}.json'
            with open(best_params_json_path_comb, 'r') as f:
                combined_bp = json.load(f)
                combined_lr, combined_gamma = combined_bp['lr'], combined_bp['gamma_joao']

            lr_vals, gamma_joao_vals = [old_lr, new_lr, combined_lr], [old_gamma, new_gamma, combined_gamma]
        except:
            print("Couldn't find existing best params! Using given arguments")

    model_path_old = '../pretrain_joaov2/weights_joao/old_aug/' + dataset + '_' + str(lr_vals[0]) + '_' +\
                     str(gamma_joao_vals[0]) + '_' + args.suffix + '_patience_' + str(args.patience) + '.pt'

    result_path_old = './results_joao/old_aug/' + dataset + '_' + str(args.n_splits) + '_' + str(args.patience) +\
                      '_' + str(lr_vals[0]) + '.res'

    result_feat_old = str(lr_vals[0]) + '_' + args.pretrain_epoch + '_' + str(gamma_joao_vals[0]) + '_' + args.suffix

    model_path_new = '../pretrain_joaov2/weights_joao/new_aug/' + dataset + '_' + str(lr_vals[1]) + '_' + \
                     str(gamma_joao_vals[1]) + '_' + args.suffix + '_patience_' + str(args.patience) + '.pt'

    result_path_new = './results_joao/new_aug/' + dataset + '_' + str(args.n_splits) + '_' + str(args.patience) + \
                      '_' + str(lr_vals[1]) + '.res'

    result_feat_new = str(lr_vals[1]) + '_' + args.pretrain_epoch + '_' + str(gamma_joao_vals[1]) + '_' + args.suffix

    # model_path_comb = '../pretrain_joaov2/weights_joao/combined/' + dataset + '_' + str(lr_vals[2]) + '_' + \
    #                  str(gamma_joao_vals[2]) + '_' + args.suffix + '_patience_' + str(args.patience) + '.pt'
    #
    # result_path_comb = './results_joao/combined/' + dataset + '_' + str(args.n_splits) + '_' + str(args.patience) + \
    #                   '_' + str(lr_vals[1]) + '.res'
    #
    # result_feat_comb = str(lr_vals[2]) + '_' + args.pretrain_epoch + '_' + str(gamma_joao_vals[2]) + '_' + args.suffix

    print('Calculate accuracy for old augmentations..')
    acc_old_mean, acc_old_std = run_experiment_finetune(model_path_old, result_path_old, result_feat_old,
                                                        tuned_lr=float(lr_vals[0]))

    print('Calculate accuracy for new augmentations..')
    acc_new_mean, acc_new_std = run_experiment_finetune(model_path_new, result_path_new, result_feat_new,
                                                        tuned_lr=float(lr_vals[1]))

    # print('Calculate accuracy for both old and new augmentations..')
    # acc_comb_mean, acc_comb_std = run_experiment_finetune(model_path_comb, result_path_comb, result_feat_comb,
    #                                                     tuned_lr=float(lr_vals[2]))

    x = ['Old Augmentations', 'New Augmentations']
    # x = ['Old Augmentations', 'New Augmentations', 'Combined']
    y = [acc_old_mean * 100, acc_new_mean * 100]
    # y = [acc_old_mean * 100, acc_new_mean * 100, acc_comb_mean * 100]
    yerr = [acc_new_std * 100, acc_new_std * 100]
    # yerr = [acc_new_std * 100, acc_new_std * 100, acc_comb_std * 100]

    fig, ax = plt.subplots()
    ax.bar(x, y, yerr=yerr, alpha=.7, capsize=10)
    ax.set_ylabel('Accuracy')
    ax.set_xticks(x)
    ax.set_title(f'Accuracy (%) - {args.dataset}')
    ax.yaxis.grid(True)
    plt.savefig(f'plots/comparison/{dataset}.png')
    plt.show()

    # with open('results_joao/accuracy.csv', 'a', newline='') as f_object:
    #     writer_object = writer(f_object)
    #     writer_object.writerow([dataset, f'{np.round(acc_old_mean, 4)} +- {np.round(acc_old_std, 4)}',
    #                             f'{np.round(acc_new_mean, 4)} +- {np.round(acc_new_std, 4)},'
    #                             ])
    #                             f' {np.round(acc_comb_mean, 4)} +- {np.round(acc_comb_std, 4)}', 'TBD'])
        # f_object.close()


def run_ad_comp(ds_list, n_repeats=20):
    for ds in ds_list:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'DATASET: {ds.upper()}')
        dataset_name, feat_str, net =\
            create_n_filter_triple(ds, 'deg+odeg100', 'ResGCN_graphcl', gfn_add_ak3=True, reddit_odeg10=True, dd_odeg10_ak1=True)
        model_func = get_model_with_default_configs(net)

        pretrain_lr, pretrain_gamma_joao = .003, 0

        model_path_old = '../pretrain_joaov2/weights_joao/old_aug/' + ds + '_' + str(pretrain_lr) + '_' + \
                         str(pretrain_gamma_joao) + '_' + args.suffix + '_patience_' + str(args.patience) + '.pt'

        model_path_new = '../pretrain_joaov2/weights_joao/new_aug/' + ds + '_' + str(pretrain_lr) + '_' + \
                         str(pretrain_gamma_joao) + '_' + args.suffix + '_patience_' + str(args.patience) + '.pt'

        old_precision, new_precision, old_recall, new_recall = {}, {}, {}, {}

        for r in range(n_repeats):
            print(f'\n##################### ITER {r} - Run Old #####################\n')
            old_measures, outliers_ratio = ad_task(dataset_name, False, args.prob_comb_mode, False, False,
                                                                model_func, args.batch_size, model_PATH=model_path_old)

            print(f'\n##################### ITER {r} - Run New #####################\n')
            new_measures, _ = ad_task(dataset_name, False, args.prob_comb_mode, False, True,
                                    model_func, args.batch_size, model_PATH=model_path_new)

            for k in old_measures.keys():
                if k in old_precision.keys():
                    old_precision[k].append(old_measures[k][0])
                    new_precision[k].append(new_measures[k][0])

                else:
                    old_precision[k] = [old_measures[k][0]]
                    new_precision[k] = [new_measures[k][0]]

        for k in old_precision.keys():
            old_precision[k] = np.mean(old_precision[k])
            new_precision[k] = np.mean(new_precision[k])

            # old_recall[k] = np.mean(old_recall[k])
            # new_recall[k] = np.mean(new_recall[k])

        print(f'Old Precision: {old_precision}')
        print(f'New Precision: {new_precision}')

        # print(f'Old Recall: {old_recall}')
        # print(f'New Recall: {new_recall}')

        # Plot Precision
        old_precision_means = list(old_precision.values())
        new_precision_means = list(new_precision.values())
        x = np.arange(len(old_precision.values()))

        width = 0.4

        plt.clf()

        # plot data in grouped manner of bar type
        plt.bar(x - 0.2, old_precision_means, width, label='Old Augmentations')
        plt.bar(x + 0.2, new_precision_means, width, label='New Augmentations')
        # plt.axhline(y=rand_labels_prob, linestyle='--', label='Random Labels', c='black')

        plt.xticks(x, list(old_precision.keys()))
        plt.xlabel("AD Algorithms")
        plt.ylabel('Precision')
        plt.title(f'Precision for {ds.upper()} - '
                  f'{np.round(outliers_ratio * 100, 4)} % Outliers')

        plt.legend()

        plt.savefig(f'plots/ad_comp/{ds} - precision.png')
        plt.show()

        # Plot Recall
        # old_recall_means = list(old_recall.values())
        # new_recall_means = list(new_recall.values())
        # x = np.arange(len(old_recall.values()))
        #
        # width = 0.4

        # plt.clf()

        # plot data in grouped manner of bar type
        # plt.bar(x - 0.2, old_recall_means, width, label='Old Augmentations')
        # plt.bar(x + 0.2, new_recall_means, width, label='New Augmentations')
        # plt.axhline(y=rand_labels_prob, linestyle='--', label='Random Labels', c='black')

        # plt.xticks(x, list(old_recall.keys()))
        # plt.xlabel("AD Algorithms")
        # plt.ylabel('Recall')
        # plt.title(f'Recall for {ds.upper()} - '
        #           f'{np.round(outliers_ratio * 100, 4)} % Outliers')
        #
        # plt.legend()
        #
        # plt.savefig(f'plots/ad_comp/{ds} - recall.png')
        # plt.show()


if __name__ == '__main__':
    if args.ad_comp:
        ds_list = ['twitter_security']
        # ds_list = ['enron', 'reality_mining', 'sec_repo']
        run_ad_comp(ds_list)
        exit()

    if args.comparison_mode:
        print('Comparison mode is on.')
        run_comparison()

    if args.combine:
        aug_txt = 'combined'
    elif args.new_aug:
        aug_txt = 'new_aug'
    else:
        aug_txt = 'old_aug'

    if args.use_best_params:
        pretrain_lr, pretrain_gamma_joao = .003, 0
    else:
        pretrain_lr, pretrain_gamma_joao = float(args.pretrain_lr), float(args.pretrain_gamma)
        print(f'Using given params - lr = {pretrain_lr}, gamma = {pretrain_gamma_joao}')

    if args.prob_comb:
        if args.prob_comb_mode == 'all_aug':
            model_PATH = f'../pretrain_joaov2/weights_joao/prob_comb/' + args.dataset + '_' + args.prob_comb_mode + '_'\
                         + str(args.alpha) + '_' + str(args.beta) + '.pt'
        else:
            model_PATH = f'../pretrain_joaov2/weights_joao/prob_comb/' + args.dataset + '_' + args.prob_comb_mode + '_'\
                         + str(args.alpha) + '.pt'

    else:
        model_PATH = '../pretrain_joaov2/weights_joao/' + aug_txt + '/' + args.dataset + '_' + str(pretrain_lr) + '_' + \
                     str(pretrain_gamma_joao) + '_' + args.suffix + '_patience_' + str(args.patience) + '.pt'

    if not os.path.isdir(output_dir := f'./results_joao/{aug_txt}'):
        os.mkdir(output_dir)

    result_PATH = './results_joao/' + aug_txt + '/' + args.dataset + '_' + str(args.n_splits) + '_' + str(args.patience) \
                  + '_' + args.lr + '.res'
    result_feat = args.lr + '_' + args.pretrain_epoch + '_' + args.pretrain_gamma + '_' + args.suffix

    if args.classification:
        if args.train:
            acc_mean, acc_std = run_experiment_finetune(model_PATH, result_PATH, result_feat, tuned_lr=float(pretrain_lr), train=True)

            if args.prob_comb:
                try:
                    f = open('prob_comb_results.txt', 'a')
                except:
                    f = open('prob_comb_results.txt', 'w')

                f.write(f'DATASET: {args.dataset}, MODE: {args.prob_comb_mode} alpha: {args.alpha} beta: {args.beta} '
                        f'LABEL RATE: {int(100 / args.n_splits)}%'
                        f' ACC: {acc_mean} +- {acc_std}\n')

                f.close()

            with open('results_joao/accuracy.csv', 'a', newline='') as f_object:
                writer_object = writer(f_object)
                acc_txt = f'{np.round(acc_mean, 4)} +- {np.round(acc_std, 4)}'
                if args.combine:
                    report_line = [args.dataset, '-', '-', acc_txt, 'TBD']
                elif args.new_aug:
                    report_line = [args.dataset, '-', acc_txt, '-', 'TBD']
                else:
                    report_line = [args.dataset, acc_txt, '-', '-', 'TBD']

                writer_object.writerow(report_line)
                f_object.close()

            with open(f'pretrain_tuning.txt', 'a') as f:
                f.write(f'dataset: {args.dataset}   lr: {pretrain_lr}   gamma: {pretrain_gamma_joao} patience: '
                        f'{args.patience} {aug_txt}   acc: {acc_mean}\n')

        else:
            acc_mean, acc_std = run_experiment_finetune(model_PATH, result_PATH, result_feat,
                                                        tuned_lr=float(pretrain_lr), train=False)
            try:
                f = open('no_train_results.txt', 'a')
            except:
                f = open('no_train_results.txt', 'w')

            if args.prob_comb:
                f.write(f'PROB COMB -\tDATASET: {args.dataset}\tMODE: {args.prob_comb_mode}\talpha: {args.alpha}\t'
                        f'beta: {args.beta}\tACC: {acc_mean} ; {acc_std}\n')
            elif args.calc_joao_probs:
                f.write(f'JOAO PROBS -\tDATASET: {args.dataset}\tMODE: {aug_txt}\tACC: {acc_mean} ; {acc_std}\n')
            else:
                f.write(f'FIXED PROBS -\tDATASET: {args.dataset}\tMODE: {aug_txt}\tACC: {acc_mean} ; {acc_std}\n')
            f.close()

    else:
        run_experiment_finetune(model_PATH, result_PATH, result_feat, tuned_lr=float(pretrain_lr))
