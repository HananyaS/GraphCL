import pickle
import re
import argparse
import sys

import matplotlib.pyplot as plt

from datasets import get_dataset
from res_gcn import ResGCN_graphcl, vgae_encoder, vgae_decoder

import experiment_joao
import os
import json
from itertools import product
import warnings


warnings.filterwarnings("ignore")

os.chdir('/home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/pretrain_joaov2')
print('START PRE-TRAINING')

str2bool = lambda x: x.lower() == "true"
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default="datasets")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_layers_feat', type=int, default=1)
parser.add_argument('--n_layers_conv', type=int, default=3)
parser.add_argument('--n_layers_fc', type=int, default=2)
parser.add_argument('--hidden', type=int, default=128)
parser.add_argument('--global_pool', type=str, default="sum")
parser.add_argument('--skip_connection', type=str2bool, default=False)
parser.add_argument('--res_branch', type=str, default="BNConvReLU")
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--edge_norm', type=str2bool, default=True)

parser.add_argument('--lr', type=float, default=0.003)
parser.add_argument('--epochs', type=int, default=100)

# parser.add_argument('--dataset', type=str, default="NCI1")
# parser.add_argument('--dataset', type=str, default="enron")
parser.add_argument('--dataset', type=str, default="sec_repo")
parser.add_argument('--aug_mode', type=str, default="sample")
parser.add_argument('--aug_ratio', type=float, default=0.2)
parser.add_argument('--suffix', type=int, default=0)

parser.add_argument('--model', type=str, default='joao')
parser.add_argument('--gamma_joao', type=float, default=0.1)

parser.add_argument('--comparison_mode', type=str2bool, default=False)
parser.add_argument('--new_aug', type=str2bool, default=False)
parser.add_argument('--combine', type=str2bool, default=False)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--use_best_params', type=str2bool, default=True)
parser.add_argument('--develop', type=str2bool, default=False)
parser.add_argument('--remove_by_embedding', type=str2bool, default=False)
parser.add_argument('--prob_comb', type=str2bool, default=False)
parser.add_argument('--prob_comb_mode', type=str, default='louvain')
parser.add_argument('--alpha', type=float, default=.9)
parser.add_argument('--beta', type=float, default=.5) 
parser.add_argument('--calc_joao_probs', type=str2bool, default=False)

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
        def foo(dataset):
            return ResGCN_graphcl(dataset=dataset, hidden=hidden, num_feat_layers=num_feat_layers,
                                  num_conv_layers=num_conv_layers,
                                  num_fc_layers=num_fc_layers, gfn=False, collapse=False,
                                  residual=residual, res_branch=res_branch,
                                  global_pool=global_pool, dropout=dropout,
                                  edge_norm=edge_norm)

    else:
        raise ValueError("Unknown model {}".format(model_name))
    return foo


def run_experiment_graphcl(dataset_feat_net_triple
                           =create_n_filter_triple(args.dataset, 'deg+odeg100', 'ResGCN_graphcl', gfn_add_ak3=True,
                                                   reddit_odeg10=True, dd_odeg10_ak1=True),
                           get_model=get_model_with_default_configs,
                           save_results=True, lr=args.lr, gamma_joao=args.gamma_joao, new_aug=args.new_aug,
                           combined=args.combine,
                           prob_comb=args.prob_comb, alpha=args.alpha, beta=args.beta, prob_comb_mode=args.prob_comb_mode,
                           calc_joao_probs=args.calc_joao_probs):
    dataset_name, feat_str, net = dataset_feat_net_triple

    if combined:
        aug_txt = 'combined'
    elif new_aug:
        aug_txt = 'new_aug'
    else:
        aug_txt = 'old_aug'

    dataset_obj_path = f'../datasets_obj/{dataset_name}_{prob_comb_mode}.p' if prob_comb\
        else f'../datasets_obj/{dataset_name}_{aug_txt}.p'

    if not os.path.isfile(dataset_obj_path):
        tu_dataset = dataset_name.lower() not in ['enron', 'reality_mining', 'twitter_security', 'sec_repo', 'darpa']

        dataset = get_dataset(
            dataset_name, sparse=True, feat_str=feat_str, root=args.data_root, new_aug=new_aug, develop=args.develop,
            combined=combined, prob_comb=prob_comb, prob_comb_mode=prob_comb_mode, tu_dataset=tu_dataset)

        with open(dataset_obj_path, 'wb') as f:
            print(f'Dataset saved at: {dataset_obj_path}')
            pickle.dump(dataset, f)

    else:
        with open(dataset_obj_path, 'rb') as f:
            dataset = pickle.load(f)

    # print(f'Shapes:')
    # print(f'x: {dataset.data.x.shape}')
    # print(f'x: {dataset.data.y.shape}')
    # print(f'x: {dataset.data.edge_index.shape}')
    # exit()

    if args.develop:
        dataset = dataset[:10]

    model_func = get_model(net)

    loss, val_losses, train_losses = experiment_joao.experiment(dataset, model_func, epochs=args.epochs,
                                                                batch_size=16 if dataset_name == 'twitter_security' else args.batch_size,
                                                                lr=lr,
                                         weight_decay=0, dataset_name=dataset_name, aug_mode=args.aug_mode,
                                         aug_ratio=args.aug_ratio, suffix=args.suffix, gamma_joao=gamma_joao,
                                         new_aug=new_aug, patience=args.patience, save_results=save_results,
                                                  develop=args.develop, combined=combined,
                                                                prob_comb=prob_comb, alpha_p=alpha, beta_p=beta,
                                                                prob_comb_mode=prob_comb_mode,
                                                                calc_joao_probs=calc_joao_probs)

    return loss, val_losses, train_losses


def run_comparison(dataset=args.dataset, lr_vals=[args.lr] * 3, gamma_joao_vals=[args.gamma_joao] * 3,
                   use_best_params=args.use_best_params):
    if use_best_params:
        try:
            with open(f'best_params/old_aug/{dataset}.json', 'r') as f:
                old_bp = json.load(f)
                old_lr, old_gamma = old_bp['lr'], old_bp['gamma_joao']

            with open(f'best_params/new_aug/{dataset}.json', 'r') as f:
                new_bp = json.load(f)
                new_lr, new_gamma = new_bp['lr'], new_bp['gamma_joao']

            with open(f'best_params/combined/{dataset}.json', 'r') as f:
                comb_bp = json.load(f)
                comb_lr, comb_gamma = comb_bp['lr'], comb_bp['gamma_joao']

            lr_vals, gamma_joao_vals = [old_lr, new_lr, comb_lr], [old_gamma, new_gamma, comb_gamma]

        except:
            print("Couldn't find existing best params! Using given arguments")

    print('Calculate loss with old augmentations..')
    min_loss_old, losses_old_aug_val, losses_old_aug_train = run_experiment_graphcl(lr=lr_vals[0], gamma_joao=gamma_joao_vals[0],
                                                          save_results=False, new_aug=False)
    print('Calculate loss with new augmentations..')
    min_loss_new, losses_new_aug_val, losses_new_aug_train = run_experiment_graphcl(lr=lr_vals[1], gamma_joao=gamma_joao_vals[1],
                                                          save_results=False, new_aug=True, combined=False)

    # print('Calculate loss for both old and new augmentations..')
    # min_loss_comb, losses_comb_aug_val, losses_comb_aug_train = run_experiment_graphcl(lr=lr_vals[2], gamma_joao=gamma_joao_vals[2],
    #                                                         save_results=False, combined=True)

    print(f'Min loss for old augmentations: {min_loss_old}')
    print(f'Min loss for new augmentations: {min_loss_new}')
    # print(f'Min loss for all augmentations: {min_loss_comb}')

    plt.plot(range(1, len(losses_old_aug_val) + 1), losses_old_aug_val, label='Old augmentations - val')
    plt.plot(range(1, len(losses_old_aug_train) + 1), losses_old_aug_train, label='Old augmentations - train')

    plt.plot(range(1, len(losses_new_aug_val) + 1), losses_new_aug_val, label='New augmentations - val')
    plt.plot(range(1, len(losses_new_aug_train) + 1), losses_new_aug_train, label='New augmentations - train')

    # plt.plot(range(1, len(losses_comb_aug_val) + 1), losses_comb_aug_val, label='Combined - val')
    # plt.plot(range(1, len(losses_comb_aug_train) + 1), losses_comb_aug_train, label='Combined - train')

    plt.title(f'Losses for old and new augmentations - {dataset}')

    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/comparison/{dataset} - no comb.png')
    plt.show()


def main():
    if args.develop:
        print(f'Develop mode is on (using smaller dataset)')

    if args.comparison_mode:
        print(f'Comparison mode is on.')
        run_comparison()

    else:
        """
        if args.combine:
            aug_txt = 'combined'
        elif args.new_aug:
            aug_txt = 'new_aug'
        else:
            aug_txt = 'old_aug'

        if not os.path.isdir(output_dir := f'output/{aug_txt}'):
            os.mkdir(output_dir)
        if args.use_best_params:
            if args.develop or not\
                    os.path.isfile(f'best_params/{aug_txt}{"/develop" if args.develop else ""}/{args.dataset}.json'):
                # if not args.remove_by_embedding:
                #         # print('Looking for best params.')
                #     # lr = 0.003
                #     # gamma_space = []
                #     # gamma_space = [0.01, 0.1, 1]
                #     # losses_dict = {}
                #
                #     # for gamma in gamma_space:
                #     #     print(f'New trial started - lr = {lr}, gamma = {gamma}')
                #     #     loss, val_losses, train_losses = run_experiment_graphcl(lr=lr, gamma_joao=gamma, save_results=False)
                #     #     plt.plot(range(1, len(val_losses) + 1), val_losses, label=f'lr = {lr}, gamma = {gamma}')
                #     #     losses_dict[(lr, gamma)] = loss
                #     #
                #     # lr, gamma_joao = min(losses_dict.keys(), key=lambda k: losses_dict[k])
                #     lr, gamma_joao = 0.003, None
                #
                #     print(f'Best params found! lr = {lr} ; gamma_joao = {gamma_joao}')
                #
                #     plt.title(f'{args.dataset} - Losses Convergence for Different Hyper-parameters Combinations')
                #     plt.legend()
                # else:
                #     print('Remove by embedding is on, gamma value is ignored')
                # lr, gamma_joao = 0.003, 0

                lr, gamma_joao = 0.003, 0
                best_params_dict = {'lr': lr,
                                    'gamma_joao': gamma_joao}

                if not os.path.isdir(best_params_path := f'best_params/{aug_txt}'):
                    os.mkdir(best_params_path)

                if args.develop:
                    if not os.path.isdir(best_params_path + '/develop'):
                        best_params_path += '/develop'
                        os.mkdir(best_params_path)

                with open(f'{best_params_path}/{args.dataset}.json', 'w') as f:
                    json.dump(best_params_dict, f)

                if not os.path.isdir(best_params_plot_path := f'plots/best_params_search/{aug_txt}'):
                    os.mkdir(best_params_plot_path)

                if args.develop:
                    if not os.path.isdir(best_params_plot_path + '/develop'):
                        best_params_plot_path += '/develop'
                        os.mkdir(best_params_plot_path)

                # if not args.remove_by_embeddings:
                #     plt.savefig(f'{best_params_plot_path}/{args.dataset} - {aug_txt}.png')
                #     plt.show()

            else:
                best_params_json_path = f'best_params/{aug_txt}/{args.dataset}.json'
                with open(best_params_json_path, 'r') as f:
                    best_params = json.load(f)
                lr, gamma_joao = best_params['lr'], best_params['gamma_joao']
                print(f'Using existing params - lr = {lr}, gamma_joao = {gamma_joao}')

        else:
            lr, gamma_joao = args.lr, args.gamma_joao
        """
        lr, gamma_joao = 0.003, 0
        loss = run_experiment_graphcl(lr=lr, gamma_joao=gamma_joao, save_results=True)
        return loss


if __name__ == '__main__':
    # if os.path.isdir('to_delete_ind.txt'):
    #     os.remove('to_delete_ind.txt')
    main()
