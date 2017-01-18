# from games import PatrolGame, NormalFormGame
import operator
# from dobbs import Dobbs
import pulp as plp

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

            # the only LP vars are the xs constituting the policy.
            lp_x = [plp.LpVariable("x_{}_prob_{}".format(i,j),
                                   lowBound=0,
                                   upBound=1,
                                   cat="Continuous") for i in range(self.X)]

            # Write Objective
            prob += sum([lp_x[i] * self.R[i,j] for i in range(self.X)])

            # Constraint 1 - x is a probability distribution
            prob += sum(lp_x) <= 1, "sum of lp_x"

            # Constraint 3 - q must be a best response to policy x
            for j_prime in range(self.Q):
                prob += sum([lp_x[i]*self.C[i,j] for i in range(self.X)]) >= \
                             sum([lp_x[i]*self.C[i,j_prime] for i in \
                                  range(self.X)])

            # add problems to the LPs container
            self.LPs.append({'x': lp_x, 'prob': prob})

    def solve(self):
        # solve each LP sequentially
        self.solutionTime = 0
        for lp in self.LPs:
            lp['prob'].solve(plp.GLPK(keepFiles=0, msg=0))
            self.solutionTime += lp['prob'].solutionTime

        # select the LP that yielded the highest objective value
        # get list of objective values
        objective_values = list(map(lambda x: plp.value(x['prob'].objective), \
                               self.LPs))
        print(objective_values)
        # print(objective_values)
        opt_q, opt_value = max(enumerate(objective_values), \
                               key=operator.itemgetter(1))
        # save the solution in class instance methods
        self.opt_q = opt_q
        self.opt_value = opt_value
        self.opt_x = list(map(lambda x: plp.value(x), self.LPs[opt_q]['x']))
        # print("pure strat: {}".format(opt_q))
        # print("solutionTime: {}".format(solutionTime))
        # print("opt_value: {}".format(opt_value))

# solve using both dobbs and multipleLPs
# bayse_game = PatrolGame(3,2,2)
# norm_game = NormalFormGame(bayse_game)
# mlp = MultipleLP(norm_game)
# dob = Dobbs(bayse_game)
# mlp.solve()
# dob.solve()
# print("print x policy from dobbs")
# print(dob.opt_x)
# print("print x policy from mlp")
# print(mlp.opt_x)
# print("=== mlp.opt_val: {}".format(mlp.opt_value))
# print("=== dob.opt_val: {}".format(dob.opt_value))
# print("opt q from mlp: {}".format(mlp.opt_q))

# print("SEE IF SOLUTIONS ARE EQUIVALENT")
# sol_mlp = 0
# sol_dob = 0
# for x in range(len(mlp.opt_x)):
#     sol_mlp += mlp.R[x,mlp.opt_q]*mlp.opt_x[x]
# import itertools
# for i, j, l in itertools.product(range(dob.X), range(dob.Q), range(dob.L)):
#     sol_dob += dob.p[l]*dob.R[i,j,l]*(plp.value(dob.q[j,l])*dob.opt_x[i])

# print(sol_mlp)
# print(sol_dob)
