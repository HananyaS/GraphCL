"""
Calculate 3- and 4-motifs for each graph of the generated graphs, using each of the motif calculation codes we have:
In python (CPU only), In C++ using CPU only and in C++ using GPU.
"""
import logging
import os
import sys
import itertools
import pickle
sys.path.append(".")
sys.path.append("..")


def calculate_gpu_one(run, level, size, p, directed):
    from features_infra.graph_features import GraphFeatures
    from features_infra.feature_calculators import FeatureMeta
    from features_algorithms.accelerated_graph_features.motifs import nth_nodes_motif
    from loggers import FileLogger
    feature_meta = {"motif" + str(level): FeatureMeta(nth_nodes_motif(level, gpu=True, device=3), {"m" + str(level)})}
    head_path = os.path.join("size{}_p{}_directed{}_runs".format(size, p, directed), "run_" + str(run))
    dump_path = os.path.join(head_path, "motifs_gpu")
    graph = pickle.load(open(os.path.join(head_path, "gnx.pkl"), "rb"))
    logger = FileLogger("CalculationLogger" + str(level), path=dump_path, level=logging.DEBUG)
    raw_feature = GraphFeatures(gnx=graph, features=feature_meta, dir_path=dump_path, logger=logger)
    raw_feature.build(should_dump=True)


def calculate_gpu_many(runs, level, size, p, directed):
    from multiprocessing import Pool
    from functools import partial
    pool = Pool(runs)
    pool.map(partial(calculate_gpu_one, level=level, size=size, p=p, directed=directed), range(runs))


def run_all_calculations():
    from generate_graphs_2 import SIZES, PROB, DIRECTED, NUM_RUNS
    levels = [3, 4]
    for size, is_directed, level in itertools.product(SIZES, DIRECTED, levels):
        print("Size: {}, Directed: {}, Motif: {}".format(size, is_directed, level))
        calculate_gpu_many(NUM_RUNS, level, size, PROB, is_directed)


if __name__ == '__main__':
    run_all_calculations()
