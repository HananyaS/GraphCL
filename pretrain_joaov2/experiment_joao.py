import os
import shutil
import time

import torch
from torch.optim import Adam
from pt_tu_dataset import DataLoader
import numpy as np

from utils import print_weights
import math

from matplotlib import pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(dataset, model_func, epochs, batch_size, lr, weight_decay,
               dataset_name=None, aug_mode='uniform', aug_ratio=0.2, suffix=0, gamma_joao=0.1,
               new_aug: bool = False, patience: int = 20, save_results=True, develop: bool = False,
               combined: bool = True, prob_comb: bool = True,  alpha_p: float = .5, beta_p: float = None,
               prob_comb_mode: str = 'all_aug', calc_joao_probs: bool = False):
    num_augmentations = len(dataset.augmentations)
    model = model_func(dataset).to(device)
    print_weights(model)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dataset.set_aug_mode('sample')
    dataset.set_aug_ratio(aug_ratio)
    aug_prob = np.ones(num_augmentations ** 2) / (num_augmentations ** 2)
    dataset.set_aug_prob(aug_prob)

    loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=16)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_losses, train_losses = [], []
    min_loss, early_stopping_counter = math.inf, 0
    epoch = 0

    print(f'Len: {len(dataset)}')

    if combined:
        aug_txt = 'combined'
    elif new_aug:
        aug_txt = 'new_aug'
    else:
        aug_txt = 'old_aug'

    if prob_comb:
        print('\nProb comb mode')
    else:
        print(f'Aug mode: {aug_txt}')

    if not prob_comb:
        if not os.path.isdir(output_dir := f'./weights_joao/{aug_txt}'):
            os.mkdir(output_dir)

        if develop:
            output_dir = output_dir + '/develop'

        if not os.path.isdir(output_dir):
            output_dir = output_dir
            os.mkdir(output_dir)
        weight_tmp_path = output_dir + '/' + dataset_name + '_' + str(lr) + '_' + str(
            gamma_joao) + '_' + str(suffix) + '_patience_' + str(patience) + '_temp.pt'
    else:
        if prob_comb_mode == 'all_aug':
            weight_tmp_path = f'./weights_joao/prob_comb/{dataset.name}_{prob_comb_mode}_{alpha_p}_{beta_p}_temp.pt'
        else:
            weight_tmp_path = f'./weights_joao/prob_comb/{dataset.name}_{prob_comb_mode}_{alpha_p}_temp.pt'

    T = .5
    num_training_batches = math.ceil(.8 * len(dataset) / batch_size) - 1

    print(f'\nTrain with {np.round(num_training_batches * loader.batch_size / len(loader.dataset) * 100, 2)} %'
          f' of the data\n')

    if not calc_joao_probs:
        if prob_comb:
            aug_prob = calc_aug_prob(dataset_name, alpha=alpha_p, beta=beta_p, all_aug=prob_comb_mode == 'all_aug')
        else:
            aug_prob = np.ones(num_augmentations ** 2) / (num_augmentations ** 2)
    else:
        aug_prob = None

    while True:
        epoch += 1
        print(f'epoch No. {epoch}')
        early_stopping_counter += 1
        if prob_comb and prob_comb_mode != 'louvain':
            start = time.time()
        losses, aug_prob = \
            train(loader, model, optimizer, device, gamma_joao, num_augmentations, num_training_batches, T, aug_prob)
        if prob_comb and prob_comb_mode != 'louvain':
            end = time.time()
            print(f'Epoch {epoch} took {end - start} seconds.'
                  f' ({"With louvain" if prob_comb_mode == "all_aug" else "Only embeddings"})')
            with open('embeddings_times_politifact.txt', 'a') as f:
                f.write(f'{"With louvain: " if prob_comb_mode == "all_aug" else "Only embeddings: " }{end - start}\n')

        train_loss, val_loss = losses['train'], losses['val']

        print(f'Train loss:\t{train_loss}\n'
              f'Validation loss:\t{val_loss}')
        print(aug_prob)

        loader.dataset.set_aug_prob(aug_prob)

        val_losses.append(val_loss)
        train_losses.append(train_loss)

        if val_loss < min_loss:
            print(f'min loss changed from {min_loss} to {val_loss} after {early_stopping_counter} iterations.')
            min_loss, early_stopping_counter = val_loss, 0

            if save_results:
                torch.save(model.state_dict(), weight_tmp_path)
                print('Model saved!')

        if develop and epoch >= 5:
            break

        if not develop and early_stopping_counter > patience:
            break

    if save_results:
        shutil.move(weight_tmp_path, weight_tmp_path.replace('_temp', ''))

    print(f'Training finished after {epoch} epochs.\n'
          f'Min loss (on validation) reached: {min_loss}')

    if save_results:
        plot_val_loss(train_losses, val_losses, dataset, patience, new_aug, combined, lr, gamma_joao, T, batch_size,
                      num_training_batches, prob_comb, prob_comb_mode, develop=develop)

    return min_loss, val_losses, train_losses


def calc_loss_lim(last_batch_len, batch_size, num_batches, T):
    return (batch_size * (num_batches - 1) * (math.log(batch_size - 1) - 1/T) +
            last_batch_len * (math.log(last_batch_len - 1) - 1/T)) /\
           (batch_size * (num_batches - 1) + last_batch_len)


def calc_aug_prob(dataset, probs_file: str = 'aug_probs.txt', all_aug=True, alpha=.9, beta=.9):
    with open(probs_file, 'r') as f:
        content = f.readlines()
        if all([not dataset in l for l in content]):
            print('No existing old ')
            num_aug = 7 if all_aug else 6
            marg_p = np.ones(5) / 5 * alpha
            marg_p = np.concatenate(
                (marg_p, np.array([1 - alpha] if not all_aug else
                                          [(1 - alpha) * beta, (1 - alpha) * (1 - beta)])))
            new_p = np.ones(num_aug ** 2)
            for n in range(len(new_p)):
                i, j = n // num_aug, n % num_aug
                new_p[n] = marg_p[i] * marg_p[j]

            return new_p

    prev_probs = []

    for l in content:
        tmp = l.split('\t')

        if '~' in l or len(l) == 1 or len(tmp) < 2:
            continue

        ds = tmp[0].split(' ')[-1]
        mode = tmp[1].split(' ')[-1]

        if ds != dataset or mode != 'old_aug':
            continue

        p = tmp[2][:-1].split('\t')[0].split(':')[-1][2:-1].split(' ')

        for i, prob in enumerate(p):
            if prob == '':
                continue
            else:
                prev_probs.append(float(prob))

        prev_probs = np.array(prev_probs)
        prev_probs = np.resize(prev_probs, (5, 5))

        sum_rows = np.einsum('ij -> i', prev_probs)
        sum_cols = np.einsum('ij -> j', prev_probs)
        break

    if len(prev_probs) == 0:
        prev_probs = np.ones((5, 5)) / 25
        marginal_probs = np.ones(5) * .2

    else:
        marginal_probs = (sum_rows + sum_cols) / 2

    new_marginal_probs = np.concatenate((alpha * marginal_probs, np.array(
        [1 - alpha] if not all_aug else [(1 - alpha) * beta, (1 - alpha) * (1 - beta)])))

    new_aug_count = 2 if all_aug else 1

    new_probs = np.zeros((prev_probs.shape[0] + new_aug_count, prev_probs.shape[0] + new_aug_count))

    for i in range(new_probs.shape[0]):
        for j in range(new_probs.shape[0]):
            if i < 5 and j < 5:
                new_probs[i, j] = alpha ** 2 * (prev_probs[i, j] + prev_probs[j, i]) / 2
            else:
                new_probs[i, j] = new_marginal_probs[i] * new_marginal_probs[j]

    return new_probs.flatten()


def plot_val_loss(train_loss, val_loss, dataset, patience, new_aug, combined, lr, gamma_joao, T, batch_size,
                  num_training_batches, prob_comb, prob_comb_mode, develop: bool = True, show: bool = False, save: bool = True, show_lim=True):
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', c='blue')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', c='orange')

    if show_lim:
        lim_train = calc_loss_lim(batch_size, batch_size, num_training_batches, T)
        lim_val = calc_loss_lim(batch_size, batch_size, len(dataset) / batch_size - num_training_batches, T) \
            if len(dataset) % batch_size == 0 else\
            calc_loss_lim(len(dataset) % batch_size, batch_size, len(dataset) // batch_size - num_training_batches + 1, T)
        print(f'Lim train:\t{lim_train}')
        print(f'Lim val:\t{lim_val}')
        plt.axhline(y=lim_train, linestyle='--', label='training loss limit', c='blue')
        plt.axhline(y=lim_val, linestyle='--', label='validation loss limit', c='orange')

    plt.title(f'Loss - {dataset.name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    if save:
        if not prob_comb:
            if combined:
                aug_txt = 'combined'
            elif new_aug:
                aug_txt = 'new_aug'
            else:
                aug_txt = 'old_aug'
            if not os.path.isdir(output_dir := f'plots/{aug_txt}'):
                os.mkdir(output_dir)

            if develop:
                if not os.path.isdir(output_dir + '/develop'):
                    output_dir += '/develop'
                    os.mkdir(output_dir)

            plt.legend()
            plt.savefig(f'{output_dir}/{dataset.name}_losses_patience_{patience}_lr_{lr}_gamma_{gamma_joao}.png')
        else:
            plt.legend()
            plt.savefig(f'plots/prob_comb/{dataset.name}_{prob_comb_mode}.png')

    if show:
        plt.show()


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, device, gamma_joao, num_augmentations, num_training_batches, T, fixed_aug_prob):
    train_loss = 0
    val_loss = 0

    aug_prob = loader.dataset.aug_prob if fixed_aug_prob is None else fixed_aug_prob
    n_aug = np.random.choice(num_augmentations ** 2, 1, p=aug_prob)[0]
    n_aug1, n_aug2 = n_aug // num_augmentations, n_aug % num_augmentations

    print(f'Chosen Augmentations: {n_aug1}, {n_aug2}')
    len_train, len_val = 0, 0

    for i, (_, data1, data2) in enumerate(loader):
    # for i, x in enumerate(loader):
        # print(data1.batch)
        # print(f'Count data.batch with 0: {torch.unique(data1.batch, return_counts=True)}')
        # print(x)
        # exit()
        train_mode = i < num_training_batches
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward_graphcl(data1, n_aug1)
        # print(f'out1.shape: {out1.shape}')
        # exit()
        out2 = model.forward_graphcl(data2, n_aug2)
        loss = model.loss_graphcl(out1, out2, T=T)

        if train_mode:
            train_loss += loss.item() * num_graphs(data1)
            len_train += num_graphs(data1)
            # print(model.lin_class.weight.grad)
            loss.backward()
            optimizer.step()

        else:
            val_loss += loss.item() * num_graphs(data1)
            len_val += num_graphs(data1)

    losses = {
        'train': train_loss / len_train,
        'val': val_loss / len_val
    }

    if fixed_aug_prob is None:
        aug_prob = joao(loader, model, gamma_joao, num_augmentations)
    return losses, aug_prob


def joao(loader, model, gamma_joao, num_augmentations):
    aug_prob = loader.dataset.aug_prob
    # calculate augmentation loss
    loss_aug = np.zeros(num_augmentations ** 2)

    for n in range(num_augmentations ** 2):
        _aug_prob = np.zeros(num_augmentations ** 2)
        _aug_prob[n] = 1
        loader.dataset.set_aug_prob(_aug_prob)

        n_aug1, n_aug2 = n//num_augmentations, n % num_augmentations

        # for efficiency, we only use around 10% of data to estimate the loss
        count, count_stop = 0, len(loader.dataset)//(loader.batch_size*10) + 1
        with torch.no_grad():
            for _, data1, data2 in loader:
                data1 = data1.to(device)
                data2 = data2.to(device)
                out1 = model.forward_graphcl(data1, n_aug1)
                out2 = model.forward_graphcl(data2, n_aug2)
                loss = model.loss_graphcl(out1, out2)
                loss_aug[n] += loss.item() * num_graphs(data1)
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= (count*loader.batch_size)

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1/num_augmentations ** 2))
    mu_min, mu_max = b.min()-1/num_augmentations ** 2, b.max()-1/num_augmentations ** 2
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b-mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b-mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b-mu, 0)
    aug_prob /= aug_prob.sum()

    return aug_prob

