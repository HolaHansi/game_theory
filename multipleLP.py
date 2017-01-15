from games import PatrolGame, NormalFormGame
import operator
from dobbs import Dobbs
import itertools
import pulp as plp
import numpy as np

class MultipleLP:
    def __init__(self, normal_game):
        # number of LPs is number of pure attacker strategies
        self.X, self.Q = normal_game.R.shape
        self.LPs = []

        # get payoffs
        self.C = normal_game.C
        self.R = normal_game.R

        # construct an LP for each pure strategy
        for j in range(self.Q):
            # define problem
            prob = plp.LpProblem(name="LP-{}".format(j), sense=plp.LpMaximize)
            # init the vars
            lp_q = [plp.LpVariable("q_"+str(i),
                                   lowBound=0,
                                   upBound=1,
                                   cat="Continuous") for i in range(self.Q)]

            lp_x = [plp.LpVariable("x_"+str(i),
                                   lowBound=0,
                                   upBound=1,
                                   cat="Continuous") for i in range(self.X)]

            # Write Objective
            prob += sum([lp_x[i] * self.R[i,j] for i in range(self.X)])

            # Constraint 1 - x is a probability distribution
            prob += sum(lp_x) <= 1

            # Constraint 2 - q is a probability distribution
            prob += sum(lp_q) <= 1

            # Constraint 3 - q must be a best response to policy x
            for j_prime in range(self.Q):
                prob += sum([lp_x[i]*self.C[i,j] for i in range(self.X)]) >= \
                             sum([lp_x[i]*self.C[i,j_prime] for i in \
                                  range(self.X)])

            # add problems to the LPs container
            self.LPs.append({'q': lp_q, 'x': lp_x, 'prob': prob})

    def solve(self):
        # solve each LP sequentially
        solutionTime = 0
        for lp in self.LPs:
            lp['prob'].solve(plp.GLPK(keepFiles=1, msg=0))
            solutionTime += lp['prob'].solutionTime

        # select the LP that yielded the highest objective value
        # get list of objective values
        objective_values = list(map(lambda x: plp.value(x['prob'].objective), \
                               self.LPs))
        print(objective_values)
        opt_q, opt_value = max(enumerate(objective_values), \
                               key=operator.itemgetter(1))
        # save the solution in class instance methods
        self.opt_q = opt_q
        self.opt_value = opt_value
        self.opt_x = self.LPs[opt_q]['x']
        print("pure strat: {}".format(opt_q))
        print("solutionTime: {}".format(solutionTime))
        print("opt_value: {}".format(opt_value))

# solve using both dobbs and multipleLPs
bayse_game = PatrolGame(3,2,2)
norm_game = NormalFormGame(bayse_game)
mlp = MultipleLP(norm_game)
dob = Dobbs(bayse_game)
mlp.solve()
dob.solve()
print(list(map(lambda x: plp.value(x), mlp.opt_x)))
print(mlp.opt_q)
print("print x policy from dobbs")
print("=== mlp.opt_val: {}".format(mlp.opt_value))
print("=== dob.opt_val: {}".format(dob.opt_value))

