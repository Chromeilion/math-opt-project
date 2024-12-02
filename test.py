from osap import calc_osap


def test_osap():
    n_students = 6
    n_teams = 2
    min_team_size = 3
    max_team_size = 3
    relationship_graph = [(1, 2), (5, 1), (1, 3), (3, 2), (3, 4)]

    teams = calc_osap(
        node_set=list(range(n_students)),
        edge_set=relationship_graph,
        n_teams=n_teams,
        min_team_size=min_team_size,
        max_team_size=max_team_size,
    )
    ...


if __name__ == '__main__':
    test_osap()
