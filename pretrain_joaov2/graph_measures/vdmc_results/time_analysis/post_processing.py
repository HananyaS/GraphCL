import os
import itertools
import pandas as pd


def aggregate_results():
    # Assuming validation is complete.
    from generate_graphs import SIZES, AVG_DEG, DIRECTED, NUM_RUNS
    if not os.path.exists("results"):
        os.mkdir("results")
    for size, degree, is_directed in itertools.product(SIZES, AVG_DEG, DIRECTED):
        dirname = f"size{size}_deg{degree}_directed{is_directed}_runs"
        times_elapsed = {#"motif3_python": [], "motif4_python": [],
                         #"motif3_cpp": [], "motif4_cpp": [],
                         "motif3_gpu": [], "motif4_gpu": []}
        for run in range(NUM_RUNS):
            dir_path = os.path.join(dirname, "run_%d" % run)
            for level, mode in itertools.product([3, 4],["gpu"]): #["python", "cpp", "gpu"]):
                if all([size == 10000, degree == 100, level == 4, mode == "python"]):
                    times_elapsed[f"motif{level}_{mode}"].append(float('nan'))  # Too long calculations
                    continue
                path_to_time_log = os.path.join(dir_path, f"motifs_{mode}", f"CalculationLogger{level}.log")
                t = get_calc_times(path_to_time_log)
                times_elapsed[f"motif{level}_{mode}"].append(t)
        times_df = pd.DataFrame(times_elapsed)
        times_df.to_csv(os.path.join("results", f"size{size}_deg{degree}_directed{is_directed}_run_times.csv"),
                        index=None)


def get_calc_times(time_path):  # IN SECONDS
    time_taken = 0
    with open(time_path, "r") as log_file:
        for i, line in enumerate(log_file):
            if i == 1:
                time_str = line.strip().split(" ")[-1].split(":")
                time_taken = 3600 * int(time_str[0]) + 60 * int(time_str[1]) + float(time_str[2])
    return time_taken


if __name__ == "__main__":
    aggregate_results()
