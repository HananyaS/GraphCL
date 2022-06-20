import itertools
import os
import networkx as nx
import pickle

"""
Generate graphs for motif performance evaluation. 
We create NUM_RUNS Erdos-Renyi graphs of each size in SIZES, with edge probabilities such that the average degrees will  
be in AVG_DEG (if directed, we use the probability such that the average total degree will be as required). 
"""

SIZES = [100, 1000, 10000]
AVG_DEG = [3, 10, 30, 100]
DIRECTED = [True, False]
NUM_RUNS = 5

if __name__ == "__main__":
    for size, degree, is_directed in itertools.product(SIZES, AVG_DEG, DIRECTED):
        dirname = "size%s_deg%d_directed%r_runs" % (size, degree, is_directed)
        print(dirname)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        p = min([1, float(degree) / (size - 1) * (0.5 if is_directed else 1)])
        for run in range(NUM_RUNS):
            os.mkdir(os.path.join(dirname, "run_%d" % run))
            # os.mkdir(os.path.join(dirname, "run_%d" % run, "motifs_python"))
            # os.mkdir(os.path.join(dirname, "run_%d" % run, "motifs_cpp"))
            os.mkdir(os.path.join(dirname, "run_%d" % run, "motifs_gpu"))
            g = nx.gnp_random_graph(n=size, p=p, directed=is_directed)
            pickle.dump(g, open(os.path.join(dirname, "run_%d" % run, "gnx.pkl"), "wb"))
