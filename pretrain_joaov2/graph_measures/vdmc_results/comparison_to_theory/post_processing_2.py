import os
import pickle
import itertools
import numpy as np
import pandas as pd


def aggregate_results():
    # Assuming validation is complete.
    from additional_features import MotifProbability
    from generate_graphs_2 import SIZES, PROB, DIRECTED, NUM_RUNS
    if not os.path.exists("results_2"):
        os.mkdir("results_2")
    for size, is_directed, level in itertools.product(SIZES, DIRECTED, [3, 4]):
        dump_dirname = os.path.join("results_2", f"size{size}_p{PROB}_directed{is_directed}_motif{level}")
        if not os.path.exists(dump_dirname):
            os.mkdir(dump_dirname)
        mp = MotifProbability(size=size, edge_probability=PROB, clique_size=0, directed=is_directed)
        expected_motifs = pd.DataFrame(
            [mp.motif_expected_non_clique_vertex(motif_index=idx) for idx in get_motif_indices(level, is_directed)]).T
        expected_motifs.to_csv(os.path.join(dump_dirname, "expected_motifs.csv"), header=None, index=None)
        dirname = f"size{size}_p{PROB}_directed{is_directed}_runs"
        for run in range(NUM_RUNS):
            res_df = pd.DataFrame(res_to_matrix(
                pickle.load(open(os.path.join(dirname, f"run_{run}", "motifs_gpu", f"motif{level}.pkl"), "rb"))))
            if not os.path.exists(os.path.join(dump_dirname, f"run_{run}")):
                os.mkdir(os.path.join(dump_dirname, f"run_{run}"))
            res_df.to_csv(os.path.join(dump_dirname, f"run_{run}", "motifs.csv"), header=None, index=None)


def res_to_matrix(res):
    ftr = res._features
    if type(ftr) == list:
        return np.array(ftr, dtype=int)
    else:  # Dict
        mx = np.zeros((len(ftr), len(ftr[0]) - 1), dtype=int)
        for i in range(len(ftr)):
            for j in range(len(ftr[0]) - 1):
                mx[i, j] = ftr[i][j]
        return mx


def get_motif_indices(level, directed):
    if directed:
        if level == 3:
            return range(13)
        else:
            return range(13, 212)
    else:
        if level == 3:
            return range(2)
        else:
            return range(2, 8)


if __name__ == "__main__":
    aggregate_results()
