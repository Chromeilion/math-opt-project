import os
import time
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from osap import calc_osap
from utils import generate_data, plot_results, factors


WEAK_SCALE_RATIO = 10
MAX_THREADS = os.cpu_count()-1
STRONG_WORKLOAD = 50
STRONG_MIN_TEAMSIZE = 3
STRONG_MAX_TEAMSIZE = 6
STRONG_N_TEAMS =  11
N_TESTS = 10
N_STUDENTS_GROUP_SCALING = 500


def main():
    times, workload = test_student()
    plot_results(workload, times, f"Student Scaling", "No. Students", "Time", f"./only_student_scaling.png")
    times, workload = test_teams()
    plot_results(workload, times, f"Team Size Scaling (for {N_STUDENTS_GROUP_SCALING} students)", "Team Size", "Time", f"./team_scaling_{N_STUDENTS_GROUP_SCALING}.png")
    times, workload = test_reg()
    plot_results(workload, times, "Student Scaling", "No. Students", "Time", "./scaling.png")
    times_fancy, workload_fancy = test_reg(fancy=True)
    times, workload = test_reg()
    fig, ax = plt.subplots()
    ax.plot(workload_fancy, times_fancy, label="Fancy")
    ax.plot(workload, times,  label="Regular")
    ax.set_xlabel("Workload")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Linear Scaling")
    fig.legend()
    fig.savefig("./scaling.png")
    times_weak, threads_weak = test_weak()
    plot_results(threads_weak, times_weak, 'Weak Scaling', "Time", "Threads", "./weak.png")
    times_strong, threads_strong = test_strong()
    times_strong_inv = 1 / np.array(times_strong)
    plot_results(threads_strong, times_strong, 'Strong Scaling', "Time", "Threads", "./strong.png")
    plot_results(threads_strong, times_strong_inv, 'Strong Scaling Inverted', "Time", "Threads", "./strong_inv.png")

    times_conc, n_conc = test_concurrent()
    plot_results(n_conc, times_conc, 'Concurrent Solver Scaling', "No. Solvers", "Time", "./conc.png")


def test_teams():
    times = []
    workloads = []
    factors_teams = sorted(factors(N_STUDENTS_GROUP_SCALING))
    for n_teams in tqdm.tqdm(factors_teams, desc="Testing team scaling"):
        nodes = list(range(N_STUDENTS_GROUP_SCALING))
        edges = generate_data(N_STUDENTS_GROUP_SCALING)
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=n_teams,
                                 min_team_size=N_STUDENTS_GROUP_SCALING//n_teams,
                                 max_team_size=N_STUDENTS_GROUP_SCALING//n_teams)
        times.append(time_func(N_TESTS, func))
        workloads.append(n_teams)

    return times, workloads


def test_student(fancy: bool = None):
    times = []
    workloads = []
    scale_factor = 50
    peak = 20
    for step in tqdm.tqdm(range(peak, 1, -2), total=peak, desc="Testing linear scaling"):
        workload = scale_factor*step
        nodes = list(range(workload))
        edges = generate_data(workload)
        team_size = workload // 20
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=20,
                                 min_team_size=team_size-10,
                                 max_team_size=team_size+10,
                                 dense=fancy,
                                 dense_fancy=fancy)
        times.append(time_func(N_TESTS, func))
        workloads.append(workload)

    return times, workloads


def test_reg(fancy: bool = None):
    times = []
    workloads = []
    scale_factor = 20
    peak = 30
    for step in tqdm.tqdm(range(peak, 1, -2), total=peak, desc="Testing linear scaling"):
        workload = scale_factor*step
        nodes = list(range(workload))
        edges = generate_data(workload)
        n_teams = workload // 20
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=n_teams,
                                 min_team_size=20,
                                 max_team_size=20+20,
                                 dense=fancy,
                                 dense_fancy=fancy)
        times.append(time_func(N_TESTS, func))
        workloads.append(workload)

    return times, workloads


def time_func(n_tries, func):
    times = []
    for _ in range(n_tries):
        start = time.time()
        func()
        times.append(time.time() - start)
    return np.array(times).mean()


def test_concurrent():
    nodes = list(range(STRONG_WORKLOAD))
    edges = generate_data(STRONG_WORKLOAD)
    times = []
    all_threads = []
    for threads in tqdm.tqdm(range(1, MAX_THREADS+1), total=MAX_THREADS, desc="Testing concurrent scaling"):
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=STRONG_N_TEAMS,
                                 min_team_size=STRONG_MIN_TEAMSIZE,
                                 max_team_size=STRONG_MAX_TEAMSIZE,
                                 n_concurrent=threads)
        times.append(time_func(N_TESTS, func))
        all_threads.append(threads)

    return times, all_threads

def test_strong():
    nodes = list(range(STRONG_WORKLOAD))
    edges = generate_data(STRONG_WORKLOAD)
    times = []
    all_threads = []
    for threads in tqdm.tqdm(range(1, MAX_THREADS+1), total=MAX_THREADS, desc="Testing strong scaling"):
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=STRONG_N_TEAMS,
                                 min_team_size=STRONG_MIN_TEAMSIZE,
                                 max_team_size=STRONG_MAX_TEAMSIZE,
                                 n_threads=threads)
        times.append(time_func(N_TESTS, func))
        all_threads.append(threads)

    return times, all_threads


def test_weak():
    times = []
    all_threads = []
    for threads in tqdm.tqdm(range(1, MAX_THREADS+1), total=MAX_THREADS, desc="Testing weak scaling"):
        workload = WEAK_SCALE_RATIO*threads
        nodes = list(range(workload))
        edges = generate_data(workload)
        n_teams = workload // 5
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=n_teams,
                                 min_team_size=3,
                                 max_team_size=8,
                                 n_threads=threads)
        times.append(time_func(N_TESTS, func))
        all_threads.append(threads)

    return times, all_threads


def plot_results(x, y, title, xlabel, ylabel, saveloc):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(saveloc)


if __name__ == '__main__':
    main()
