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


def calculate_for_graph(graph, dir_path, mode, level):
    """
    Calculates the motif on the given graph. Saves both the logger (with calculation time) and the results.
    :param graph: The NetworkX graph on which the calculations are performed.
    :param dir_path: The path to the main directory of the specific graph.
    :param mode: "py" for python-only calculation, "cpp" for C++ with CPU-only and "gpu" for C++ and GPU.
    :param level: 3 or 4, the level of the motif to calculate.
    """

    from features_infra.graph_features import GraphFeatures
    from features_infra.feature_calculators import FeatureMeta
    from loggers import FileLogger

    if mode == "py":
        from features_algorithms.vertices.motifs import nth_nodes_motif
        feature_meta = {
            "motif%s" % level: FeatureMeta(nth_nodes_motif(level), {"m%d" % level})
        }
        dir_path = os.path.join(dir_path, "motifs_python")

    elif mode == "cpp":
        from features_algorithms.accelerated_graph_features.motifs import nth_nodes_motif
        feature_meta = {
            "motif%s" % level: FeatureMeta(nth_nodes_motif(level, gpu=False, device=2), {"m%d" % level})
        }
        dir_path = os.path.join(dir_path, "motifs_cpp")

    else:  # gpu
        from features_algorithms.accelerated_graph_features.motifs import nth_nodes_motif
        feature_meta = {
            "motif%s" % level: FeatureMeta(nth_nodes_motif(level, gpu=True, device=2), {"m%d" % level})
        }
        dir_path = os.path.join(dir_path, "motifs_gpu")

    logger = FileLogger("CalculationLogger%d" % level, path=dir_path, level=logging.DEBUG)
    raw_feature = GraphFeatures(gnx=graph, features=feature_meta, dir_path=dir_path, logger=logger)
    raw_feature.build(should_dump=True)


def calculate_gpu_one(run, level, size, avg_deg, directed):
    from features_infra.graph_features import GraphFeatures
    from features_infra.feature_calculators import FeatureMeta
    from features_algorithms.accelerated_graph_features.motifs import nth_nodes_motif
    from loggers import FileLogger
    feature_meta = {"motif%s" % level: FeatureMeta(nth_nodes_motif(level, gpu=True, device=2), {"m%d" % level})}
    head_path = os.path.join("size%d_deg%d_directed%r_runs" % (size, avg_deg, directed), "run_%d" % run)
    dump_path = os.path.join(head_path, "motifs_gpu")
    graph = pickle.load(open(os.path.join(head_path, "gnx.pkl"), "rb"))
    logger = FileLogger("CalculationLogger%d" % level, path=dump_path, level=logging.DEBUG)
    raw_feature = GraphFeatures(gnx=graph, features=feature_meta, dir_path=dump_path, logger=logger)
    raw_feature.build(should_dump=True)


def calculate_gpu_many(runs, level, size, avg_deg, directed):
    from multiprocessing import Pool
    from functools import partial
    p = Pool(runs)
    p.map(partial(calculate_gpu_one, level=level, size=size, avg_deg=avg_deg, directed=directed), range(runs))


def run_all_calculations():
    from generate_graphs import SIZES, AVG_DEG, DIRECTED, NUM_RUNS
    modes = ["gpu"]  # ["cpp", "gpu"]
    levels = [3, 4]
    for size, degree, is_directed, mode, level in itertools.product(SIZES, AVG_DEG, DIRECTED, modes, levels):
        if mode == "gpu":
            calculate_gpu_many(NUM_RUNS, level, size, degree, is_directed)
        else:
            for run in range(NUM_RUNS):
                dirname = "size%s_deg%d_directed%r_runs" % (size, degree, is_directed)
                dir_path = os.path.join(dirname, "run_%d" % run)
                if not os.path.exists(dir_path):
                    raise ModuleNotFoundError("%s graph of size %d with average degree %d - run %d not found" %
                                              ("Directed" if is_directed else "Undirected", size, degree, run))
                with open(os.path.join(dir_path, "gnx.pkl"), "rb") as g:
                    graph = pickle.load(g)
                    calculate_for_graph(graph.copy(), dir_path, mode, level)


if __name__ == '__main__':
    run_all_calculations()
