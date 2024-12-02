import gurobipy as gp
import numpy as np


def calc_osap(node_set: list[int],
              edge_set: list[tuple[int, int]],
              n_teams: int,
              min_team_size: int | list[int],
              max_team_size: int | list[int]):
    if isinstance(max_team_size, list):
        if len(max_team_size) != n_teams:
            raise AttributeError(
                "When providing the maximum team size as a list, it must "
                "have as many entries as there are teams."
            )
    else:
        max_team_size = [max_team_size] * n_teams
    if isinstance(min_team_size, list):
        if len(min_team_size) != n_teams:
            raise AttributeError(
                "When providing the maximum team size as a list, it must "
                "have as many entries as there are teams."
            )
    else:
        min_team_size = [min_team_size] * n_teams


    model = gp.Model("student_grouping")
    N = node_set
    E = edge_set
    M = list(range(n_teams))
    U = min_team_size
    O = max_team_size
    X = model.addMVar(
        shape=(len(node_set), n_teams),
        vtype=gp.GRB.BINARY,
        name="X"
    )
    Y = model.addMVar(
        shape=(len(node_set), len(node_set)),
        vtype=gp.GRB.BINARY,
        name="Y"
    )

    # Encode the objective
    model.setObjective(gp.quicksum([Y[i, j] for i, j in edge_set]), gp.GRB.MINIMIZE)

    for i in node_set:
        # Encode constraint 2: Assign every student to exactly 1 team
        model.addConstr(
            gp.quicksum(X[i, :]) == 1,
            name="2"
        )
    for k in M:
        # Encode constraint 3: team sizes must have size between U and O.
        model.addConstr(
            U[k] <= gp.quicksum(X[:, k]),
            name=f"3a{k}"
        )
        model.addConstr(
            gp.quicksum(X[:, k]) <= O[k],
            name=f"3b{k}"
        )

    for i, j in edge_set:
        # Constraint 4: conflict variable takes value 1 when two connected
        # nodes are together in a group.
        for k in M:
            model.addConstr(
                X[i, k] + X[j, k] <= Y[i, j] + 1,
                name="4"
            )

    model.optimize()
    model.fixed()
    return X.X
