from itertools import product
import os

if __name__ == '__main__':
    datasets = ['NCI1', 'PROTEINS', 'MUTAG']
    modes = ['old_aug', 'new_aug', 'combine']
    calc_joao_probs = [False, True]
    # modes = ['all_aug', 'louvain', 'embeddings']
    # alphas = [.5, .7, .9]
    # n_splits = [100, 10]
    # betas = [.3, .5, .7]

    # with open('run_prob_comb.sh', 'w') as f:
    for ds in datasets:
        f_name = f'run_no_ft_training_no_PC_{ds}.sh'
        with open(f_name, 'w') as f:
            for mode, cjp in product(modes, calc_joao_probs):
                if mode == 'old_aug':
                    coms = [False, False]
                elif mode == 'new_aug':
                    coms = [False, True]
                else:
                    coms = [True, True]

                pretrain_com = f'python /home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/pretrain_joaov2/main.py' \
                               f' --dataset {ds} --prob_comb False --combine {coms[0]} --new_aug {coms[1]}' \
                               f' --use_best_params True --develop False --comparison_mode False --calc_joao_probs {cjp}\n'
                f.write(pretrain_com)

                finetune_com = f'python /home/dsi/shacharh/Projects/GraphCL_new/semisupervised_TU/finetune_joaov2/main.py' \
                                   f' --dataset {ds} --prob_comb False --combine {coms[0]} --new_aug {coms[1]}' \
                               f' --classification True --train False --calc_joao_probs {cjp}\n'
                f.write(finetune_com)

        path = os.getcwd() + f'/{f_name}'
        mode = os.stat(path).st_mode
        mode |= (mode & 0o444) >> 2    # copy R bits to X
        os.chmod(path, mode)
