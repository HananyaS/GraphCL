import re
import argparse

import matplotlib.pyplot as plt

from datasets import get_dataset
from res_gcn import ResGCN_graphcl, vgae_encoder, vgae_decoder

import experiment_joao
import sys
import os
import shutil
import json
from itertools import product
import warnings


warnings.filterwarnings("ignore")

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

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--dataset', type=str, default="MCF-7")
parser.add_argument('--aug_mode', type=str, default="sample")
parser.add_argument('--aug_ratio', type=float, default=0.2)
parser.add_argument('--suffix', type=int, default=0)

parser.add_argument('--model', type=str, default='joao')
parser.add_argument('--gamma_joao', type=float, default=0.1)

parser.add_argument('--comparison_mode', type=str2bool, default=False)
parser.add_argument('--new_aug', type=str2bool, default=True)
parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--use_best_params', type=str2bool, default=True)
parser.add_argument('--develop', type=str2bool, default=True)

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
                           save_results=True, lr=args.lr, gamma_joao=args.gamma_joao, new_aug=args.new_aug):
    dataset_name, feat_str, net = dataset_feat_net_triple
    dataset = get_dataset(
        dataset_name, sparse=True, feat_str=feat_str, root=args.data_root, new_aug=new_aug, develop=args.develop)

    if args.develop:
        dataset = dataset[:10]

    model_func = get_model(net)

    loss, val_losses = experiment_joao.experiment(dataset, model_func, epochs=args.epochs, batch_size=args.batch_size, lr=lr,
                                         weight_decay=0, dataset_name=dataset_name, aug_mode=args.aug_mode,
                                         aug_ratio=args.aug_ratio, suffix=args.suffix, gamma_joao=gamma_joao,
                                         new_aug=new_aug, patience=args.patience, save_results=save_results,
                                                  develop=args.develop)

    return loss, val_losses


def run_comparison(dataset=args.dataset, lr_vals=[args.lr] * 2, gamma_joao_vals=[args.gamma_joao] * 2,
                   use_best_params=args.use_best_params):
    if use_best_params:
        try:
            with open(f'best_params/old_aug/{dataset}.json', 'r') as f:
                old_bp = json.load(f)
                old_lr, old_gamma = old_bp['lr'], old_bp['gamma_joao']

            with open(f'best_params/new_aug/{dataset}.json', 'r') as f:
                new_bp = json.load(f)
                new_lr, new_gamma = new_bp['lr'], new_bp['gamma_joao']

            lr_vals, gamma_joao_vals = [old_lr, new_lr], [old_gamma, new_gamma]

        except:
            print("Couldn't find existing best params! Using given arguments")

    print('Calculate loss with old augmentations..')
    min_loss_old, losses_old_aug = run_experiment_graphcl(lr=lr_vals[0], gamma_joao=gamma_joao_vals[0],
                                                          save_results=False, new_aug=False)
    print('Calculate loss with new augmentations..')
    min_loss_new, losses_new_aug = run_experiment_graphcl(lr=lr_vals[1], gamma_joao=gamma_joao_vals[1],
                                                          save_results=False, new_aug=True)

    print(f'Min loss for old augmentations: {min_loss_old}\n'
          f'Min loss for new augmentations: {min_loss_new}')

    plt.plot(range(1, len(losses_old_aug) + 1), losses_old_aug, label='Old augmentations')
    plt.plot(range(1, len(losses_new_aug) + 1), losses_new_aug, label='New augmentations')

    plt.title(f'Losses for old and new augmentations - {dataset}')

    plt.xlabel('Num epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots/comparison/{dataset}.png')
    plt.show()


if __name__ == '__main__':
    if args.develop:
        print(f'Develop mode is on (using smaller dataset)')

    if args.comparison_mode:
        print(f'Comparison mode is on.')
        run_comparison()

    else:
        aug_txt = 'new_aug' if args.new_aug else 'old_aug'
        if not os.path.isdir(output_dir := f'output/{aug_txt}'):
            os.mkdir(output_dir)

        if args.use_best_params:
            if args.develop or not os.path.isfile(f'best_params/{aug_txt}{"/develop" if args.develop else ""}/{args.dataset}.json'):
                print('Looking for best params.')
                lr_space = [0.01, 0.001, 0.0001]
                gamma_space = [0.01, 0.1, 1]
                losses_dict = {}

                for lr, gamma in product(lr_space, gamma_space):
                    print(f'New trial started - lr = {lr}, gamma = {gamma}')
                    loss, losses = run_experiment_graphcl(lr=lr, gamma_joao=gamma, save_results=False)
                    plt.plot(range(1, len(losses) + 1), losses, label=f'lr = {lr}, gamma = {gamma}')
                    losses_dict[(lr, gamma)] = loss

                lr, gamma_joao = min(losses_dict.keys(), key=lambda k: losses_dict[k])

                print(f'Best params found! lr = {lr} ; gamma_joao = {gamma_joao}')
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

                plt.title(f'{args.dataset} - Losses Convergence for Different Hyperparameters Combinations')
                plt.legend()

                if not os.path.isdir(best_params_plot_path := f'plots/best_params_search/{aug_txt}'):
                    os.mkdir(best_params_plot_path)

                if args.develop:
                    if not os.path.isdir(best_params_plot_path + '/develop'):
                        best_params_plot_path += '/develop'
                        os.mkdir(best_params_plot_path)

                plt.savefig(f'{best_params_plot_path}/{args.dataset} - {aug_txt}.png')
                plt.show()

                # if not os.path.isfile(nni_path := f'nni_results/{aug_txt}/{args.dataset}.csv'):
                #     raise FileExistsError(f"Best params don't exist! Use NNI first - save results at:\n"
                #                           f"{nni_path}")
                # else:
                #     best_params_df = pd.read_csv(nni_path)
                #     best_trial_num = np.argmin(best_params_df['reward'])
                #     best_lr, best_gamma_joao = \
                #         best_params_df['lr'][best_trial_num], best_params_df['gamma_joao'][best_trial_num]
                #
                #     lr, gamma_joao = best_lr, best_gamma_joao
                #     best_params_dict = {'lr': lr,
                #                         'gamma_joao': gamma_joao}
                #
                #     if not os.path.isdir(best_params_path := f'best_params/{aug_txt}'):
                #         os.mkdir(best_params_path)
                #
                #     with open(f'best_params/{aug_txt}/{args.dataset}.json', 'w') as f:
                #         json.dump(best_params_dict, f)

            else:
                best_params_json_path = f'best_params/{aug_txt}/{args.dataset}.json'
                with open(best_params_json_path, 'r') as f:
                    best_params = json.load(f)
                lr, gamma_joao = best_params['lr'], best_params['gamma_joao']
                print(f'Using existing params - lr = {lr}, gamma_joao = {gamma_joao}')

        else:
            lr, gamma_joao = args.lr, args.gamma_joao

        # outputTmpFile = f'output/tmp/pretrain_output_{args.dataset}_patience_{args.patience}_lr_{lr}_gamma_' \
        #                 f'{gamma_joao}.txt'

        # sys.stdout = open(outputTmpFile, mode='w')

        loss = run_experiment_graphcl(lr=lr, gamma_joao=gamma_joao, save_results=True)

        # shutil.move(outputTmpFile, outputTmpFile.replace('tmp/', aug_txt + '/'))
        # sys.stdout.close()
