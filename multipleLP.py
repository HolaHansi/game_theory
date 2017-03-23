# from games import PatrolGame, NormalFormGame
import time
import operator
# from dobbs import Dobbs
import pulp as plp

class MultipleLP:
    def __init__(self, game, attacker_type=0):
        # number of LPs is number of pure attacker strategies
        self.X = game.num_defender_strategies
        self.Q = game.num_attacker_strategies
        self.LPs = []

        # get payoffs
        self.C = game.attacker_payoffs[:, :, attacker_type]
        self.R = game.defender_payoffs[:, :, attacker_type]

        # construct an LP for each pure strategy
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
            prob += sum(lp_x) == 1, "sum of lp_x"

            # Constraint 3 - q must be a best response to policy x
            for j_prime in range(self.Q):
                prob += sum([lp_x[i]*self.C[i,j] for i in range(self.X)]) >= \
                             sum([lp_x[i]*self.C[i,j_prime] for i in \
                                  range(self.X)])

            # add problems to the LPs container
            self.LPs.append({'x': lp_x, 'prob': prob})

    def solve(self):
        # solve each LP sequentially
        start_time = time.time()
        for j, lp in enumerate(self.LPs):
            lp['prob'].solve(plp.GLPK(keepFiles=0, msg=0))

        # save solution time (without overhead)
        self.solution_time = time.time() - start_time

        # select the LP that yielded the highest objective value
        optimal_LPs = filter(lambda x: x['prob'].status == plp.LpStatusOptimal,
                             self.LPs)
        objective_values = list(map(lambda x: plp.value(x['prob'].objective),
                               optimal_LPs))

        opt_q, opt_value = max(enumerate(objective_values), \
                               key=operator.itemgetter(1))

        # save the solution in instance variables
        self.opt_attacker_pure_strategy = opt_q
        self.opt_defender_payoff = opt_value
        self.opt_defender_mixed_strategy  = \
                        list(map(lambda x: plp.value(x), self.LPs[opt_q]['x']))

        # save solution time with overhead
        self.solution_time_with_overhead = time.time() - start_time
