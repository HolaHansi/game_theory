import pulp as plp
import numpy as np
import time


class OrigamiMILP:
    def __init__(self, game, attacker_type=0):
        self.attacker_uncovered = game.attacker_uncovered[:,attacker_type]
        self.attacker_covered = game.attacker_covered[:,attacker_type]
        self.defender_uncovered = game.defender_uncovered[:,attacker_type]
        self.defender_covered = game.defender_covered[:,attacker_type]

        self.num_targets = game.num_targets
        self.max_coverage = game.max_coverage

        # large constant
        self.Z = 9999

        self.prob = plp.LpProblem(name="ORIGAMI-MILP", sense=plp.LpMinimize)
        self.k = plp.LpVariable("k", cat="Contineous")

        self.C = np.ndarray(shape=(self.num_targets),
                            dtype=type(plp.LpVariable("dummy")))
        self.y = np.ndarray(shape=(self.num_targets),
                            dtype=type(plp.LpVariable("dummy")))

        for t in range(self.num_targets):
            self.C[t] = plp.LpVariable("c_{}".format(t),
                                       lowBound=0,
                                       upBound=1,
                                       cat="Contineous")
            self.y[t] = plp.LpVariable("y_{}".format(t),
                                       lowBound=0,
                                       upBound=1,
                                       cat="Integer")
        # set objective function
        self.prob += self.k

        # Constraint 1
        self.prob += self.C.sum() <= self.max_coverage

        for t in range(self.num_targets):
            # Constraint 2
            self.prob += (self.C[t] * self.attacker_covered[t] +
                        (1 - self.C[t]) * self.attacker_uncovered[t]) <= \
                        self.k

            # Constraint 3
            self.prob += self.k - (self.C[t] * self.attacker_covered[t] +
                        (1 - self.C[t]) * self.attacker_uncovered[t]) <= \
                        (1 - self.y[t]) * self.Z

            # Constraint 4
            self.prob += self.C[t] <= self.y[t]

    def solve(self):
        # use GLPK solver
        start_time = time.time()
        self.prob.solve(plp.GLPK(keepFiles=0, msg=0))
        # save solution time (without overhead)
        self.solution_time = time.time() - start_time
        # save status
        self.status = plp.LpStatus[self.prob.status]

        # save optimal coverage
        self.opt_coverage = [plp.value(x) for x in self.C]

        # compute optimal attacked target, the attackset and defender payoff
        self.opt_defender_payoff = float("-inf")
        self.opt_attack_set = []

        for t in np.nonzero(list(map(lambda x : plp.value(x), self.y)))[0]:
            self.opt_attack_set.append(t)

            defender_payoff = plp.value(self.C[t]) * \
                            plp.value(self.defender_covered[t]) + \
                            (1 - plp.value(self.C[t])) * \
                            plp.value(self.defender_uncovered[t])
            # print("in milp: defender_Payoffs: {}: {}".format(t,
            #                                                  defender_payoff))
            # print("coverage[{}] : {}".format(t, plp.value(self.C[t])))
            # print("def_covered[{}] : {}".format(t, self.defender_covered[t]))
            # print("def_uncovered[{}] : {}".format(t, self.defender_uncovered[t]))

            if defender_payoff > self.opt_defender_payoff:
                self.opt_defender_payoff = defender_payoff
                # set the attacked target
                self.opt_attacked_target = t

        # save solution_time with overhead
        self.solution_time_with_overhead = time.time() - start_time
