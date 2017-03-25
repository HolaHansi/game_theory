import itertools
import pulp as plp
import numpy as np
import time

class Dobbs:
    """
    Init dobbs will internally store an MILP representation
    of the game provided as the only constructor argument.
    """
    def __init__(self, game):
        # init the game as an MILP
        self.prob = plp.LpProblem(name="DOBBS", sense=plp.LpMaximize)

        # get payoffs and adversary probability distribution
        self.C = game.attacker_payoffs
        self.R = game.defender_payoffs
        self.p = game.attacker_type_probability

        # get dimensions of a game payoff matrix - needed to generate LP vars
        X, Q, L = self.R.shape
        self.X, self.Q, self.L = (X, Q, L)

        # init z_ijl vars as lp variables
        self.z = np.ndarray(shape=(X, Q, L),
                            dtype=type(plp.LpVariable("dummy")))

        for i,j,l in itertools.product(range(X), range(Q), range(L)):
            self.z[i,j,l] = plp.LpVariable("z_"+str(i)+','+str(j)+','+str(l),
                                           lowBound=0,
                                           upBound=1,
                                           cat="contineous")

        # init q_jl vars as lp variables
        self.q = np.ndarray(shape=(Q, L), dtype=type(plp.LpVariable("dummy")))
        for j,l in itertools.product(range(Q),range(L)):
            self.q[j,l] = plp.LpVariable("q_"+str(j)+','+str(l),
                                         lowBound=0,
                                         upBound=1,
                                         cat="Integer")

        # init a_l vars as lp variables
        self.a = np.ndarray(shape=(L), dtype=type(plp.LpVariable("dummy")))
        for l in range(L):
            self.a[l] = plp.LpVariable("a_"+str(l), cat="Contineous")

        # set objective function of MILP
        self.prob += sum([self.p[l]*self.R[i,j,l]*self.z[i,j,l] for i,j,l in
                                                itertools.product(range(X),
                                                                  range(Q),
                                                                  range(L))])

        # Constraint. 1
        for l in range(L):
            self.prob += sum([self.z[i,j,l] for i,j in \
                                itertools.product(range(X), range(Q))]) == 1, ""

        # Constraint. 2
        for i,l in itertools.product(range(X), range(L)):
            self.prob += sum([self.z[i,j,l] for j in range(Q)]) <= 1, ""

        # Constraint 3
        for j, l in itertools.product(range(Q), range(L)):
            self.prob += sum([self.z[i,j,l] for i in range(X)]) >= \
                                                                self.q[j,l], ""
            self.prob += sum([self.z[i,j,l] for i in range(X)]) <= 1, ""

        # Constraint 4
        for l in range(L):
            self.prob += sum([self.q[j,l] for j in range(Q)]) == 1, ""

        # Constraint 5
        for j, l in itertools.product(range(Q), range(L)):
            self.prob += self.a[l] - sum([self.C[i,j,l] * sum(self.z[i,:,l])
                                          for i in range(X)]) >= 0
            self.prob += self.a[l] - sum([self.C[i,j,l] * sum(self.z[i,:,l])
                                                    for i in range(X)]) <= \
                                                    (1-self.q[j,l])*9999

        # Constraint 6
        for i, l in itertools.product(range(X), range(L)):
            self.prob += sum([self.z[i,j,l] for j in range(Q)]) == \
                            sum([self.z[i,j,0] for j in range(Q)])

    def solve(self):
        # use GLPK solver
        start_time = time.time()
        self.prob.solve(plp.GLPK(keepFiles=0, msg=0))

        # save solution time (without overhead)
        self.solution_time = time.time() - start_time

        # save status
        self.status = plp.LpStatus[self.prob.status]

        # compute optimal attacked target and defender payoff
        self.opt_defender_payoff = plp.value(self.prob.objective)

        # derive and save the optimal defender mixed strategy
        self.opt_defender_mixed_strategy = np.zeros((self.X))
        for i in range(self.X):
            for j in range(self.Q):
                if plp.value(self.z[i,j,0]) > 0:
                    self.opt_defender_mixed_strategy[i] = \
                                                        plp.value(self.z[i,j,0])
                    break
        # derive and save the optimal attacked target for each attacker type
        self.opt_attacker_pure_strategy = np.zeros((self.L), dtype=np.int8)
        f = np.vectorize(plp.value)
        qs = f(self.q)
        for l in range(self.L):
            self.opt_attacker_pure_strategy[l] = np.nonzero(qs[:,l])[0][0]
        # convert to tuple
        self.opt_attacker_pure_strategy = tuple(self.opt_attacker_pure_strategy)

        # save solution time with overhead
        self.solution_time_with_overhead = time.time() - start_time

# for testing

# from games import PatrolGame
# x = PatrolGame(5, 2, 3)
# p = Dobbs(x)
# print("dobbs initialized")
# p.solve()

# print(p.opt_defender_mixed_strategy)
# print(p.opt_attacker_pure_strategies)

# f = np.vectorize(plp.value)
# print(p.opt_x)
# print(p.opt_x.sum())
