

import numpy
import csv
import os
import itertools

if __name__ == '__main__':
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

        with open(os.path.join(dump_dirname, "expected_motifs.csv"), newline='') as expected_motifs_file:
            for row in expected_motifs_file:
                expected_motifs = row
        expected_motifs = list(expected_motifs.split(","))
        for i in range(len(expected_motifs)):
            expected_motifs[i] = float(expected_motifs[i])

        dirname = f"size{size}_p{PROB}_directed{is_directed}_runs"
        motifs = []
        for run in range(NUM_RUNS):
            with open(os.path.join(dump_dirname, f"run_{run}", "motifs.csv"), newline='') as motifs_file:
                reader = csv.reader(motifs_file)
                data = list(reader)
            motifs.append(data)
        motifs = motifs[0]
        for i in range(len(motifs)):
            for j in range(len(motifs[i])):
                motifs[i][j] = int(motifs[i][j])
        average_motifs = [sum(x) / len(x) for x in zip(*motifs)]
        print(
            f"size{size}_p{PROB}_directed{is_directed}_motif{level} expected - founded{[(x1 - x2) for (x1, x2) in zip(expected_motifs, average_motifs)]}")
