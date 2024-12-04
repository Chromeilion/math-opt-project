from osap import calc_osap, calc_rand
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
import itertools

SEED = 42
COLORS = {0: "tab:red", 1: "tab:blue", 2: "tab:green", 3: "tab:pink",
          4: "tab:orange", 5: "tab:purple", 6: "tab:brown"}
# Setting for lognorm distribution
# Is set so that most students have 1  or 2 connections
MU, SIGMA = 1, 0.3
np.random.seed(SEED)
random.seed(SEED)

# A bunch of test cases to ensure the model functions correctly
test_cases = [
    {"input": {"edge_set":[(0, 1), (1, 0), (2, 3), (3, 2)],
     "min_team_size": 2,
     "max_team_size": 2,
     "n_teams": 2,
     "node_set": list(range(4))},
     "best_result": 0}
]


def generate_data(n_students: int):
    Y = scipy.stats.lognorm(s=SIGMA, scale=np.exp(MU))
    graph = []
    all_students = list(range(n_students))
    for i in range(n_students):
        n_friends = min(np.round(Y.rvs(size=1))[0], n_students)
        friends = random.sample(all_students, random.randint(1, n_students))
        current_n_friends = len([j for j in graph if i in j])
        if current_n_friends >= n_friends:
            continue
        for j in friends:
            if i != j:
                graph.append((i, j))

    return graph

def plot_results(relationship_graph, teams):
    lab = np.where(teams)[1]
    G = nx.Graph()
    G.add_edges_from(relationship_graph)
    pos = nx.shell_layout(G)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=relationship_graph,
        width=2,
        alpha=1
    )
    node_partitions = []
    for i in range(teams.shape[1]):
        node_partitions.append(np.where(lab == i)[0])
    for p_no, partition in enumerate(node_partitions):
        p = list(partition)
        nx.draw_networkx_nodes(G, pos, nodelist=p, node_color=COLORS[p_no])
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[i for i in relationship_graph if
                      i[0] in p and i[1] in p],
            width=2,
            alpha=1,
            edge_color="tab:red"
        )

    labels = {i: i for i in range(lab.shape[0])}
    nx.draw_networkx_labels(G, pos, labels,
                            font_color="whitesmoke")
    plt.title("Student Relationship Network")
    plt.show()


def test_osap():
    for case in test_cases:
        teams, obj = calc_osap(**case["input"])
        assert np.isclose(obj, case["best_result"])
        assert teams.shape[1] == case["input"]["n_teams"]
        assert teams.shape[0] == len(case["input"]["node_set"])
        n_students_per_team = np.sum(teams, axis=0)
        assert np.all(case["input"]["max_team_size"] >= n_students_per_team)
        assert np.all(case["input"]["min_team_size"] <= n_students_per_team)

    n_tests = 4
    min_students_per_test = 20
    max_students_per_test = 30
    min_teams = 3
    max_teams = 5
    min_teamsize = 2
    max_teamsize = 10
    old_stats_opt = []
    new_stats_opt = []
    old_stats_rand = []
    new_stats_rand = []
    for _ in tqdm.tqdm(range(n_tests), total=n_tests):
        n_students = random.randint(min_students_per_test, max_students_per_test)
        n_teams = random.randint(min_teams, max_teams)
        edges = generate_data(n_students)
        teams, obj = calc_osap(list(range(n_students)), edges, n_teams,
                               min_teamsize, max_teamsize)
        stats = do_stats(edges, teams)
        old_stats_opt.append(stats[0])
        new_stats_opt.append(stats[1])
        teams, obj = calc_rand(list(range(n_students)), edges, n_teams,
                               min_teamsize, max_teamsize)
        stats = do_stats(edges, teams)
        old_stats_rand.append(stats[0])
        new_stats_rand.append(stats[1])

    make_comp_plots(old_stats_opt, new_stats_opt, old_stats_rand, new_stats_rand)
    do_plot()

def make_comp_plots(old_stats_opt, new_stats_opt, old_stats_rand, new_stats_rand):
    old_stats_opt, new_stats_opt = np.array(old_stats_opt), np.array(new_stats_opt)
    old_stats_opt_mean = old_stats_opt.mean(axis=0)
    new_stats_opt_mean = new_stats_opt.mean(axis=0)
    old_stats_rand, new_stats_rand = np.array(old_stats_rand), np.array(new_stats_rand)
    new_stats_rand_mean = new_stats_rand.mean(axis=0)
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
    cliques = nx.algorithms.clique.number_of_cliques(G)
    ind = len(nx.algorithms.maximal_independent_set(G))
    avg_clique = np.array(list(dict(cliques).values())).mean()
    avg_degree = np.array(list(dict(student_degrees).values())).mean()
    diam = nx.algorithms.diameter(G)

    return density, avg_clique,  avg_degree, ind, diam

def do_plot():
    n_students = 20
    edges = generate_data(n_students)
    nodes = list(range(n_students))
    n_teams = 6
    min_team_size = 3
    max_team_size = 4
    teams, _ = calc_osap(
        node_set=nodes,
        edge_set=edges,
        min_team_size=min_team_size,
        max_team_size=max_team_size,
        n_teams=n_teams
    )
    plot_results(edges, teams)


if __name__ == '__main__':
    test_osap()
