from itertools import product
import os

if __name__ == '__main__':
    # datasets = ['DD', 'COLLAB', 'MUTAG']
    datasets = ['NCI1', 'PROTEINS', 'MUTAG']
    modes = ['all_aug', 'louvain', 'embeddings']
    alphas = [.5, .7, .9]
    betas = [.3, .5, .7]
    n_splits = [100, 10]

    # with open('run_prob_comb.sh', 'w') as f:
    for ds in datasets:
        f_name = f'run_prob_comb_{ds}.sh'
        with open(f_name, 'w') as f:
            for mode, alpha in product(modes, alphas):
                if mode == 'all_aug':
                    for beta in betas:
                        pretrain_com = f'python /home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/pretrain_joaov2/main.py' \
                                       f' --dataset {ds} --prob_comb True --prob_comb_mode {mode} --alpha {alpha} --beta {beta}' \
                                       f' --use_best_params True --develop False --comparison_mode False\n'
                        f.write(pretrain_com)

                        for ns in n_splits:
                            if ds == 'PROTEINS' and ns == 100:
                                continue
                            finetune_com = f'python /home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/finetune_joaov2/main.py' \
                                           f' --dataset {ds} --prob_comb True --prob_comb_mode {mode} --alpha {alpha} --beta {beta} --n_splits {ns}\n'
                            f.write(finetune_com)
                else:
                    pretrain_com = f'python /home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/pretrain_joaov2/main.py' \
                                   f' --dataset {ds} --prob_comb True --prob_comb_mode {mode} --alpha {alpha}' \
                                   f' --use_best_params True --develop False --comparison_mode False\n'
                    f.write(pretrain_com)

                    for ns in n_splits:
                        if ds == 'PROTEINS' and ns == 100:
                            continue
                        finetune_com = f'python /home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/finetune_joaov2/main.py' \
                                       f' --dataset {ds} --prob_comb True --prob_comb_mode {mode} --alpha {alpha} --n_splits {ns}\n'
                        f.write(finetune_com)

        path = os.getcwd() + f'/{f_name}'
        mode = os.stat(path).st_mode
        mode |= (mode & 0o444) >> 2    # copy R bits to X
        os.chmod(path, mode)
