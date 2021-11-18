import json
import os

import torch
from torch.optim import Adam
from tu_dataset import DataLoader
import numpy as np

from utils import print_weights
import math

from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def experiment(dataset, model_func, epochs, batch_size, lr, weight_decay,
               dataset_name=None, aug_mode='uniform', aug_ratio=0.2, suffix=0, gamma_joao=0.1,
               new_aug: bool = False, patience: int = 20, save_results=True, develop: bool = False):
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

    val_losses = []
    min_loss, early_stopping_counter = math.inf, 0
    epoch = 0

    aug_txt = 'new_aug' if new_aug else 'old_aug'

    if not os.path.isdir(output_dir := f'./weights_joao/{aug_txt}'):
        os.mkdir(output_dir)

    if not os.path.isdir(output_dir + '/develop'):
        output_dir = output_dir + '/develop'
        os.mkdir(output_dir)

    weight_path = output_dir + '/' + dataset_name + '_' + str(lr) + '_' + str(
        gamma_joao) + '_' + str(suffix) + '_patience_' + str(patience) + '.pt'

    while True:
        epoch += 1
        print(f'epoch No. {epoch}')
        early_stopping_counter += 1
        pretrain_loss, aug_prob = train(loader, model, optimizer, device, gamma_joao, num_augmentations)
        print(pretrain_loss, aug_prob)
        loader.dataset.set_aug_prob(aug_prob)

        val_losses.append(pretrain_loss)

        if pretrain_loss < min_loss:
            print(f'min loss changed from {min_loss} to {pretrain_loss} after {early_stopping_counter} iterations.')
            min_loss, early_stopping_counter = pretrain_loss, 0
            if save_results:
                torch.save(model.state_dict(), weight_path)
                print('Model saved!')

        if develop and epoch >= 5:
            break

        if not develop and early_stopping_counter > patience:
            break

    print(f'Training finished after {epoch} epochs.\n'
          f'Min loss reached: {min_loss}')

    best_params = {
        'lr': lr,
        'gamma_joao': gamma_joao
    }

    if save_results:
        # if not os.path.isdir(output_dir := f'best_params/{aug_txt}'):
        #     os.mkdir(output_dir)
        #
        # if develop:
        #     if not os.path.isdir(output_dir + f'/develop'):
        #         output_dir += '/develop'
        #         os.mkdir(output_dir)
        #
        # if not os.path.isfile(f'{output_dir}/{dataset.name}.json'):
        #     with open(f'{output_dir}/{dataset.name}.json', 'w') as f:
        #         json.dump(best_params, f)
        plot_val_loss(val_losses, dataset, patience, new_aug, lr, gamma_joao, develop=develop)

    return min_loss, val_losses


def plot_val_loss(val_loss, dataset, patience, new_aug, lr, gamma_joao, develop: bool = True,
                  show: bool = False, save: bool = True):
    plt.plot(range(1, len(val_loss) + 1), val_loss)
    plt.title(f'Loss - {dataset.name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if save:
        aug_txt = 'new_aug' if new_aug else 'old_aug'
        if not os.path.isdir(output_dir := f'plots/{aug_txt}'):
            os.mkdir(output_dir)

        if develop:
            if not os.path.isdir(output_dir + '/develop'):
                output_dir += '/develop'
                os.mkdir(output_dir)

        plt.savefig(f'{output_dir}/{dataset.name}_losses_patience_{patience}_lr_{lr}_gamma_{gamma_joao}.png')
    if show:
        plt.show()


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(loader, model, optimizer, device, gamma_joao, num_augmentations):
    total_loss = 0

    aug_prob = loader.dataset.aug_prob
    n_aug = np.random.choice(num_augmentations ** 2, 1, p=aug_prob)[0]
    n_aug1, n_aug2 = n_aug // num_augmentations, n_aug % num_augmentations

    print(f'Chosen Augmentations: {n_aug1}, {n_aug2}')

    for i, (_, data1, data2) in enumerate(loader):
        optimizer.zero_grad()
        data1 = data1.to(device)
        data2 = data2.to(device)
        out1 = model.forward_graphcl(data1, n_aug1)
        out2 = model.forward_graphcl(data2, n_aug2)
        loss = model.loss_graphcl(out1, out2)
        loss.backward()
        total_loss += loss.item() * num_graphs(data1)
        optimizer.step()

    aug_prob = joao(loader, model, gamma_joao, num_augmentations)
    return total_loss / len(loader.dataset), aug_prob


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

