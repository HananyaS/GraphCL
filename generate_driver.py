import argparse
import os
from itertools import product


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--operation', type=str, default='pretrain')
parser.add_argument('--v2', type=str2bool, default='f')
parser.add_argument('--dataset', type=str, default='NCI1')
parser.add_argument('--nohup', type=str2bool, default='f')

args = parser.parse_args()


def get_driver_root(args_dict):
    root = os.getcwd() + '/'
    if args_dict.get('operation') == 'pretrain':
        root += 'pretrain'
    else:
        root += 'finetune'
    if args_dict.get('v2'):
        root += '_joaov2'

    root += '/drivers/'
    return root


def driver_content(args_dict):
    commands = 'cd ../\n'

    dataset = [args_dict.get('dataset')]
    lr = [.01, .001, .0001]
    gamma = [.01, .1, 1]
    epochs = [20, 40, 60, 80, 100]
    epochs += [120, 140, 160, 180, 200] if args_dict.get('v2') else []
    suffixes = range(5)
    nohup = args_dict.get('nohup')

    if args_dict.get('operation') == 'pretrain':
        driver_args = {
            'dataset': dataset,
            'epochs': epochs,
            'lr': lr,
            'gamma_joao': gamma,
            'suffix': suffixes
        }
    else:
        driver_args = {
            'dataset': dataset,
            'pretrain_epochs': epochs,
            'pretrain_lr': lr,
            'pretrain_gamma': gamma,
            'suffix': suffixes,
            'n_splits': [100]
        }

    for args_comb in product(*driver_args.values()):
        command = 'nohup ' if nohup else ''
        command += 'python main.py '
        for i, k in enumerate(driver_args.keys()):
            # pass
            command += f'--{k} {args_comb[i]} '

        command += '&' if nohup else ''
        command += '\n'
        commands += command

    return commands


def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 0o444) >> 2    # copy R bits to X
    os.chmod(path, mode)


if __name__ == '__main__':
    args_dict = vars(args)
    driver_root = get_driver_root(args_dict)
    # print(driver_root)

    if not os.path.isdir(driver_root):
        os.mkdir(driver_root)

    content = driver_content(args_dict)

    driver_name = f'driver_{args.dataset}_joao'
    driver_name += 'v2' if args.v2 else ''
    driver_name += '_nohup' if args.nohup else ''

    print('name: ' + driver_name)
    full_path = driver_root + driver_name
    print(full_path)

    with open(full_path, 'w') as f:
        f.write(content)

    make_executable(full_path)
