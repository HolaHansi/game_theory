import pulp as plp
import numpy as np
import time


class Eraser:
    """
    Init will internally store an MILP representation
    of the security game provided as the only constructor argument
    """

    def __init__(self, game, attacker_type=0):
        self.attacker_uncovered = game.attacker_uncovered[:,attacker_type]
        self.attacker_covered = game.attacker_covered[:,attacker_type]
        self.defender_uncovered = game.defender_uncovered[:,attacker_type]
        self.defender_covered = game.defender_covered[:,attacker_type]

        self.num_targets = game.num_targets
        self.max_coverage = game.max_coverage

        # large constant
        self.Z = 9999

        self.prob = plp.LpProblem(name="ERASER", sense=plp.LpMaximize)
        self.d = plp.LpVariable("d", cat="Contineous")
        self.k = plp.LpVariable("k", cat="Contineous")

        self.C = np.ndarray(shape=(self.num_targets),
                            dtype=type(plp.LpVariable("dummy")))
        self.a = np.ndarray(shape=(self.num_targets),
                            dtype=type(plp.LpVariable("dummy")))

        for t in range(self.num_targets):
            self.C[t] = plp.LpVariable("c_{}".format(t),
                                       lowBound=0,
                                       upBound=1,
                                       cat="Contineous"
                                       )
            self.a[t] = plp.LpVariable("a_{}".format(t),
                                       lowBound=0,
                                       upBound=1,
                                       cat="Integer")

        # set objective function
        self.prob += self.d

        # Constraint 1
        self.prob += self.a.sum() == 1

        # Constraint 2
        self.prob += self.C.sum() <= self.max_coverage

        for t in range(self.num_targets):
            # Constraint 3
            self.prob += self.d - (self.C[t] * self.defender_covered[t] +
                            (1 - self.C[t]) * self.defender_uncovered[t]) <= \
                            (1 - self.a[t]) * self.Z


            # constain 4
            self.prob += self.k - (self.C[t] * self.attacker_covered[t] +
                            (1 - self.C[t]) * self.attacker_uncovered[t]) <=  \
                            (1 - self.a[t]) * self.Z

            self.prob += ((self.C[t] * self.attacker_covered[t]) +
                            (1 - self.C[t]) * self.attacker_uncovered[t]) - \
                            self.k <= 0


    def solve(self):
        # record start time
        start_time = time.time()

        # use GLPK solver
        self.prob.solve(plp.GLPK(keepFiles=0, msg=0))

        # save solution time (without overhead)
        self.solution_time = time.time() - start_time

        # save status
        self.status = plp.LpStatus[self.prob.status]

        # save optimal coverage
        self.opt_coverage = [plp.value(x) for x in self.C]

        # save optimal attacked target and defender payoff
        self.opt_defender_payoff = plp.value(self.prob.objective)
        self.opt_attacked_target = [t for t in range(self.num_targets)
                                    if plp.value(self.a[t]) == 1][0]

        # save solution time with overhead
        self.solution_time_with_overhead = time.time() - start_time
