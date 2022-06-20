import itertools
import os
import networkx as nx
import pickle

"""
Generate graphs for motif performance evaluation. 
We create NUM_RUNS Erdos-Renyi graphs of each size in SIZES, with edge probabilities such that the average degrees will  
be in AVG_DEG (if directed, we use the probability such that the average total degree will be as required). 
"""

SIZES = [100, 1000]
PROB = 0.1
DIRECTED = [True, False]
NUM_RUNS = 4

if __name__ == "__main__":
    for size, is_directed in itertools.product(SIZES, DIRECTED):
        dirname = "size{}_p{}_directed{}_runs".format(size, PROB, is_directed)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        for run in range(NUM_RUNS):
            os.mkdir(os.path.join(dirname, "run_{}".format(run)))
            os.mkdir(os.path.join(dirname, "run_{}".format(run), "motifs_cpp"))
            os.mkdir(os.path.join(dirname, "run_{}".format(run), "motifs_gpu"))
            g = nx.gnp_random_graph(n=size, p=PROB, directed=is_directed)
            pickle.dump(g, open(os.path.join(dirname, "run_%d" % run, "gnx.pkl"), "wb"))
