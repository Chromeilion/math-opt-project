import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy

COLORS = {0: "tab:red", 1: "tab:blue", 2: "tab:green", 3: "tab:pink",
          4: "tab:orange", 5: "tab:purple", 6: "tab:brown"}

# Setting for lognorm distribution
# Is set so that most students have 1  or 2 connections
MU, SIGMA = 1, 0.3

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