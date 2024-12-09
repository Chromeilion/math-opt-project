from typing import Optional
import gurobipy as gp
import numpy as np


def calc_osap(node_set: list[int],
              edge_set: list[tuple[int, int]],
              n_teams: int,
              min_team_size: int | list[int],
              max_team_size: int | list[int],
              student_assignments: Optional[list[tuple[int, int]]] = None,
              special_student_requirement: Optional[list[tuple[list[int], int]]] = None,
              force_teammates: Optional[list[tuple[int, int]]] = None,
              avert_teammates: Optional[list[tuple[int, int]]] = None,
              maximize_inner_ties: Optional[bool] = None,
              n_threads: Optional[int] = None,
              n_concurrent: Optional[int] = None,
              dense: Optional[bool] = None,
              dense_fancy: Optional[bool] = None):
    """
    Create student optimal groups.

    Parameters
    ----------
    node_set
        A list of all nodes in the graph, probably just list(range(no_students))
    edge_set
        Connections between students
    n_teams
    min_team_size
    max_team_size
    student_assignments
        If some node/student must be assigned to a team, pass a tuple with (student, team)
    special_student_requirement
        If a certain number of students from a set are required in each team, pass this.
        Each tuple in the list contains the node set as a list and the number as an int.
    force_teammates
        List all students that must be in the same team
    avert_teammates
        List of students that cannot be together
    maximize_inner_ties
        Whether to maximize the objective function instead of minimize
    n_threads
        Number of threads to use with the Gurobi solver
    n_concurrent
        Number of concurrent solvers to use
    dense
        Whether to use a modified version of the model that's slower but better
        for dense cases.
    dense_fancy
        Whether to use the very complicated dense formula for the best overall
        result.

    Returns
    -------
    teams : np.Array
    objective_value : int
    """
    if student_assignments is None:
        student_assignments = []
    if special_student_requirement is None:
        special_student_requirement = []
    if force_teammates is None:
        force_teammates = []
    if avert_teammates is None:
        avert_teammates = []
    if maximize_inner_ties is None:
        maximize_inner_ties = False
    if dense_fancy is None:
        dense_fancy = False

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
    model.Params.LogToConsole = 0
    if n_threads is not None:
        model.Params.Threads = n_threads
    if n_concurrent is not None:
        model.Params.ConcurrentMIP = n_concurrent
    if dense is None:
        dense = False

    M = list(range(n_teams))
    U = min_team_size
    O = max_team_size
    X = model.addMVar(
        shape=(len(node_set), n_teams),
        vtype=gp.GRB.BINARY,
        name="X"
    )
    if dense:
        Y = [
            model.addMVar(
                shape=(len(node_set), len(node_set)),
                vtype=gp.GRB.BINARY,
                name=f"Y_{i}"
            ) for i in M
        ]
    else:
        Y = model.addMVar(
            shape=(len(node_set), len(node_set)),
            vtype=gp.GRB.BINARY,
            name="Y"
        )
    # Add any manually assigned students
    for student, team in student_assignments:
        model.addConstr(
            X[student, team] == 1,
            name="var-preassignment"
        )

    # Add special team constraint
    for student_set, n_students in special_student_requirement:
        model.addConstr(
            gp.quicksum([X[student, :] for student in student_set]) >= n_students
        )

    # Any students that have to be in the same team
    for student_a, student_b in force_teammates:
        for k in range(n_teams):
            model.addConstr(
                X[student_a, k] + gp.quicksum([X[student_b, k_p] for k_p in range(n_teams) if k_p != k]) <= 1
            )

    # Encode students that cant be together
    for student_a, student_b in avert_teammates:
        model.addConstr(
            X[student_a, :] + X[student_b, :] <= 1
        )

    # Encode the objective
    obj = gp.GRB.MAXIMIZE if maximize_inner_ties else gp.GRB.MINIMIZE
    if dense:
        Z = model.addVar(vtype=gp.GRB.INTEGER, name="Z", lb=0,
                         ub=len(edge_set))
        for y_k in Y:
            model.addConstr(gp.quicksum([y_k[i,j] for i,j in edge_set]) <= Z)
        if dense_fancy:
            model.setObjective(Z+1/(len(M)+1)*gp.quicksum([gp.quicksum([y_k[i,j] for i, j in edge_set]) for y_k in Y]))
        else:
            model.setObjective(Z, obj)
    else:
        model.setObjective(gp.quicksum([Y[i, j] for i, j in edge_set]), obj)

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
        # Constraint 4/8: conflict variable takes value 1 when two connected
        # nodes are together in a group.
        for k in M:
            if dense:
                model.addConstr(
                    X[i, k] + X[j, k] <= Y[k][i, j] + 1,
                    name="8"
                )
            else:
                model.addConstr(
                    X[i, k] + X[j, k] <= Y[i, j] + 1,
                    name="4"
                )

    model.optimize()
    model.fixed()
    return X.X, model.ObjVal


def calc_rand(node_set: list[int],
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
    model.Params.LogToConsole = 0
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
    return X.X, model.ObjVal

