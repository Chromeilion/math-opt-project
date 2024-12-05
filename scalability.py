import random
import itertools
import unittest
import os
import time
import networkx as nx
import numpy as np
import tqdm
import gurobipy
import matplotlib.pyplot as plt

from osap import calc_osap, calc_rand
from utils import generate_data, plot_results

WEAK_SCALE_RATIO = 10
MAX_THREADS = os.cpu_count()-1
STRONG_WORKLOAD = 50
STRONG_MIN_TEAMSIZE = 3
STRONG_MAX_TEAMSIZE = 6
STRONG_N_TEAMS =  11
N_TESTS = 5


def main():
    times_weak, threads_weak = test_weak()
    times_strong, threads_strong = test_strong()
    times_strong_inv = 1 / np.array(times_strong)
    times, workload = test_reg()
    plot_results(threads_weak, times_weak, 'Weak Scaling', "Time", "Threads", "./weak.png")
    plot_results(threads_strong, times_strong, 'Strong Scaling', "Time", "Threads", "./strong.png")
    plot_results(threads_strong, times_strong_inv, 'Strong Scaling Inverted', "Time", "Threads", "./strong_inv.png")
    plot_results(workload, times, "Time VS Workload", "Time", "No. Students", "./scaling.png")


def test_reg():
    times = []
    workloads = []
    peak = 6
    for step in tqdm.tqdm(range(1, peak), total=peak, desc="Testing linear scaling"):
        workload = WEAK_SCALE_RATIO*step
        nodes = list(range(workload))
        edges = generate_data(workload)
        n_teams = workload // 5
        func = lambda: calc_osap(node_set=nodes,
                                 edge_set=edges,
                                 n_teams=n_teams,
                                 min_team_size=3,
                                 max_team_size=8)
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
