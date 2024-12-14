import random
import itertools

import networkx as nx
import numpy as np
import tqdm
import gurobipy

from osap import calc_osap, calc_rand
from utils import generate_data, plot_results


SEED = 100

np.random.seed(SEED)
random.seed(SEED)

# A bunch of test cases to ensure the model functions correctly
test_cases = [
    {"input": {"edge_set":[(0, 1), (1, 0), (2, 3), (3, 2)],
     "min_team_size": 2,
     "max_team_size": 2,
     "n_teams": 2,
     "node_set": list(range(4))},
     "best_result": 0,
     "raises": None},
    {"input": {"edge_set":[(0, 1), (1, 0), (2, 1), (1, 2), (3, 1),  (1, 3)],
     "min_team_size": 2,
     "max_team_size": 2,
     "n_teams": 2,
     "node_set": list(range(4))},
     "best_result": 2,
     "raises": None},
    {"input": {"edge_set":[(0, 1), (1, 0), (2, 1), (1, 2)],
     "min_team_size": 2,
     "max_team_size": 3,
     "n_teams": 2,
     "node_set": list(range(5))},
     "best_result": 0,
     "raises": None},
    {"input": {"edge_set": [(0, 1), (1, 0), (2, 3), (3, 2)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "student_assignments": [(0, 0), (1, 0)]},
     "best_result": 4,
     "raises": None},
    {"input": {"edge_set": [(0, 1), (1, 0)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "student_assignments": [(2, 0), (3, 0)]},
     "best_result": 2,
     "raises": None},
    {"input": {"edge_set": [(0, 1), (1, 0)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "special_student_requirement": [([0, 1], 2)]},
     "best_result": 0,
     "raises": gurobipy.GurobiError},
    {"input": {"edge_set": [(0, 1), (1, 0)],
               "min_team_size": 3,
               "max_team_size": 3,
               "n_teams": 2,
               "node_set": list(range(6)),
               "student_assignments": [(0, 0), (1, 0)],
               "special_student_requirement": [([0, 1, 2, 3], 1)]},
     "best_result": 2,
     "raises": None},
    {"input": {"edge_set": [(0, 1), (1, 0)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "student_assignments": [(0, 0), (1, 0)],
               "force_teammates": [(0, 1)]},
     "best_result": 2,
     "raises": None},
    {"input": {"edge_set": [(0, 1), (1, 0)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "student_assignments": [(0, 0), (1, 0)],
               "avert_teammates": [(0, 1)]},
     "best_result": 0,
     "raises": gurobipy.GurobiError},
    {"input": {"edge_set": [(0, 2), (2, 0), (0, 3), (3, 0)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "avert_teammates": [(0, 1)]},
     "best_result": 2,
     "raises": None},
    {"input": {"edge_set": [(0, 2), (2, 0), (0, 3), (3, 0)],
               "min_team_size": 2,
               "max_team_size": 2,
               "n_teams": 2,
               "node_set": list(range(4)),
               "maximize_inner_ties": True},
     "best_result": 4,
     "raises": None},
    {"input": {
        "edge_set":[(0, 1), (1, 0), (2, 1), (1, 2), (3, 1),  (1, 3)],
        "min_team_size": 2,
        "max_team_size": 2,
        "n_teams": 2,
        "node_set": list(range(4)),
        "dense": True},
     "best_result": 2,
     "raises": None},
    {"input": {
        "edge_set": [(0, 1), (1, 0), (2, 1), (1, 2), (3, 1), (1, 3)],
        "min_team_size": 2,
        "max_team_size": 2,
        "n_teams": 2,
        "node_set": list(range(4)),
        "dense": True,
        "dense_fancy": True},
        "best_result": None, # Getting into floating point shenanigans so I'll ignore it
        "raises": None},
]


def test_osap():
    do_plot()

    for case in test_cases:
        if case["raises"] is not None:
            try:
                teams, obj = calc_osap(**case["input"])
            except case["raises"]:
                continue
            else:
                raise AssertionError("Case should have raised an error!")
        else:
            teams, obj = calc_osap(**case["input"])
            if case["best_result"] is not None:
                assert np.isclose(obj, case["best_result"])
            assert teams.shape[1] == case["input"]["n_teams"]
            assert teams.shape[0] == len(case["input"]["node_set"])
            n_students_per_team = np.sum(teams, axis=0)
            assert np.all(case["input"]["max_team_size"] >= n_students_per_team)
            assert np.all(case["input"]["min_team_size"] <= n_students_per_team)

    n_tests = 20
    min_students_per_test = 20
    max_students_per_test = 50
    min_teams = 3
    max_teams = 5
    min_teamsize = 2
    max_teamsize = 20
    old_stats_opt = []
    new_stats_opt = []
    old_stats_rand = []
    new_stats_rand = []
    old_stats_fancy = []
    new_stats_fancy = []
    for _ in tqdm.tqdm(range(n_tests), total=n_tests):
        n_students = random.randint(min_students_per_test, max_students_per_test)
        n_teams = random.randint(min_teams, max_teams)
        edges = generate_data(n_students)
        teams, obj = calc_osap(list(range(n_students)), edges, n_teams,
                               min_teamsize, max_teamsize)
        stats = do_stats(edges, teams)
        old_stats_opt.append(stats[0])
        new_stats_opt.append(stats[1])
        teams, obj = calc_osap(list(range(n_students)), edges, n_teams,
                               min_teamsize, max_teamsize, dense=True,
                               dense_fancy=True)
        stats = do_stats(edges, teams)
        old_stats_fancy.append(stats[0])
        new_stats_fancy.append(stats[1])
        teams, obj = calc_rand(list(range(n_students)), edges, n_teams,
                               min_teamsize, max_teamsize)
        stats = do_stats(edges, teams)
        old_stats_rand.append(stats[0])
        new_stats_rand.append(stats[1])

    make_comp_plots(old_stats_opt, new_stats_opt, old_stats_rand, new_stats_rand, old_stats_fancy, new_stats_fancy)


def make_comp_plots(old_stats_opt, new_stats_opt, old_stats_rand, new_stats_rand, old_stats_fancy, new_stats_fancy):
    old_stats_opt, new_stats_opt = np.array(old_stats_opt), np.array(new_stats_opt)
    old_stats_opt_mean = old_stats_opt.mean(axis=0)
    new_stats_opt_mean = new_stats_opt.mean(axis=0)
    old_stats_rand, new_stats_rand = np.array(old_stats_rand), np.array(new_stats_rand)
    new_stats_rand_mean = new_stats_rand.mean(axis=0)
    old_stats_fancy, new_stats_fancy = np.array(old_stats_opt), np.array(new_stats_opt)
    old_stats_fancy_mean = old_stats_fancy.mean(axis=0)
    new_stats_fancy_mean = new_stats_fancy.mean(axis=0)
    diffs_opt = new_stats_opt - old_stats_opt
    diffs_rand = new_stats_rand - old_stats_rand
    diffs_opt = diffs_opt.mean(axis=0)
    diffs_rand = diffs_rand.mean(axis=0)

    print("Original values:")
    do_print(old_stats_opt_mean)
    print("Optimal values:")
    do_print(new_stats_opt_mean)
    print("Random values:")
    do_print(new_stats_rand_mean)
    print("Fancy values:")
    do_print(new_stats_fancy_mean)

    print("Optimal diffs:")
    do_print(diffs_opt)
    print("Random diffs:")
    do_print(diffs_rand)

def do_print(stats):
    print(f"Density: {stats[0]}\n"
          f"Clique: {stats[1]}\n"
          f"Degree: {stats[2]}\n"
          f"Independence: {stats[3]}\n"
          f"Diameter: {stats[4]}\n")

def do_stats(edges, teams):
    p_new_edge = 0.9
    new_edges = []
    for i in range(teams.shape[1]):
        team = teams[:, i]
        all_students = np.argwhere(team).squeeze().tolist()
        for c in itertools.combinations(all_students, 2):
            if random.random() < p_new_edge:
                if c not in edges:
                    new_edges.append(c)
    new_edges += edges
    n_students = teams.shape[0]

    stats_old = get_stats_g(n_students, edges)
    stats_new = get_stats_g(n_students, new_edges)

    return stats_old, stats_new

def get_stats_g(n_students, edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_students))
    G.add_edges_from(edges)
    student_degrees = G.degree()

    density = nx.density(G)
    clique = len(nx.algorithms.approximation.max_clique(G))
    ind = len(nx.algorithms.maximal_independent_set(G))
    avg_degree = np.array(list(dict(student_degrees).values())).mean()
    diam = nx.algorithms.diameter(G)

    return density, clique,  avg_degree, ind, diam

def do_plot():
    n_students = 16
    edges = generate_data(n_students)
    nodes = list(range(n_students))
    n_teams = 4
    min_team_size = 3
    max_team_size = 7
    teams, _ = calc_osap(
        node_set=nodes,
        edge_set=edges,
        min_team_size=min_team_size,
        max_team_size=max_team_size,
        n_teams=n_teams
    )
    plot_results(edges, teams, "./teams.png", title="Optimal Simple Team Assignments")
    teams, _ = calc_osap(
        node_set=nodes,
        edge_set=edges,
        min_team_size=min_team_size,
        max_team_size=max_team_size,
        n_teams=n_teams,
        dense=True
    )
    plot_results(edges, teams, "./dense_teams.png", title="Optimal Dense Team Assignments")
    teams, _ = calc_osap(
        node_set=nodes,
        edge_set=edges,
        min_team_size=min_team_size,
        max_team_size=max_team_size,
        n_teams=n_teams,
        dense=True,
        dense_fancy=True
    )
    plot_results(edges, teams, "./dense_fancy_teams.png", title="Optimal Fancy Dense Team Assignments")
    teams, _ = calc_osap(
        node_set=nodes,
        edge_set=edges,
        min_team_size=min_team_size,
        max_team_size=max_team_size,
        n_teams=n_teams,
        maximize_inner_ties=True
    )
    edges = generate_data(n_students)
    rand_teams, _ = calc_rand(
        node_set=nodes,
        edge_set=edges,
        n_teams=n_teams,
        min_team_size=min_team_size,
        max_team_size=max_team_size
    )
    plot_results(edges, teams, "./inverse_teams.png", title="Inverse Team Assignments")
    plot_results(edges, rand_teams, "./teams_rand.png", title="Randomly Assigned Teams")
    edges = generate_data(10)
    rand_teams, _ = calc_rand(
        node_set=list(range(10)),
        edge_set=edges,
        n_teams=1,
        min_team_size=10,
        max_team_size=10
    )
    plot_results(edges, rand_teams, "./social_net.png", no_color=True, title="Student Social Graph")

if __name__ == '__main__':
    test_osap()
