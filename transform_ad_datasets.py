import os
import dill
import pickle
import sys
import numpy as np

sys.modules['multi_graph'] = []

if __name__ == '__main__':
    datasets = {}
    root = 'ad_pkl_datasets'

    for ds in os.listdir(root):
        with open(f'{root}/{ds}', 'rb') as pf:
            ds_params = dill.load(pf)

        datasets[ds] = ds_params

        print(f'Params of {ds}:')
        # print(ds_params.keys())
        print(f'Anomaly rate for {ds}:\t'
              f'{100 * (1 - np.mean(list(ds_params.get("_graph_valid").values())))} %')

        print('~~~~~~~~~~~~~~~')
