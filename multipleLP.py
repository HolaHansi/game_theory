import time
import operator
import pulp as plp
import itertools

class Multiple_SingleLP:
    """
    Call SingleLP for every pure strategy in the given game
    """
    def __init__(self, game):
        self.game = game
        self.attacker_pure_strategies = itertools.product(
            range(game.num_attacker_strategies),
            repeat=game.num_attacker_types)

        # Create an LP for every attacker pure strategy
        self.LPs = []
        for pure_strat in self.attacker_pure_strategies:
            self.LPs.append(SingleLP(self.game, pure_strat))

    def solve(self):
        start_time = time.time()
        self.opt_defender_payoff = float('-inf')

        for lp in self.LPs:
            lp.solve()
            if lp.opt_defender_payoff > self.opt_defender_payoff:
                self.opt_defender_payoff = lp.opt_defender_payoff
                self.opt_defender_mixed_strategy = lp.opt_defender_mixed_strategy
                self.opt_attacker_pure_strategy = lp.pure_strat

        self.solution_time = time.time() - start_time

class SingleLP:
    """
    Takes a game and a bayesian attacker pure strategy, outputs
    the opt_defender_payoff and corresponding mixed strategy.
    """
    def __init__(self, game, pure_strat):
        self.R = game.defender_payoffs
        self.C = game.attacker_payoffs
        # TODO fit to non-compact form
        self.X = game.num_defender_strategies
        self.Q = game.num_attacker_strategies
        self.L = game.num_attacker_types
        self.p = game.attacker_type_probability
        self.pure_strat = pure_strat

        # define maximization problem
        self.prob = plp.LpProblem(name="Pure_strat: {}".format(pure_strat),
                             sense=plp.LpMaximize)

        # only vars are the mixed-strategy vars for defender
        self.x = [plp.LpVariable("x_{}".format(i),
                                 lowBound=0,
                                 upBound=1,
                                 cat="Continuous") for i in range(self.X)]

        # objective is expected defender payoff given pure strategy
        self.prob += sum([self.x[i] * sum([
                                    self.p[k] * self.R[i, pure_strat[k], k]
                                    for k in range(self.L)])
                          for i in range(self.X)])

        # Constraint 1 (pure strategy must be a best response)
        for k in range(self.L):
            for j_prime in range(self.Q):
                self.prob += sum([self.x[i] * self.C[i,pure_strat[k], k] for i in
                             range(self.X)]) >= \
                        sum([self.x[i] * self.C[i, j_prime, k]
                             for i in range(self.X)])


        # Constraint 2 (x is a prob. distribution)
        self.prob += sum(self.x) == 1

    def solve(self):
        start_time = time.time()
        # solve the LP
        self.prob.solve(plp.GLPK(keepFiles=0, msg=0))
        self.solution_time = time.time() - start_time

        # check if the pure strategy was feasible
        self.feasible = self.prob.status == plp.LpStatusOptimal
        if self.feasible:
            # get the optimal defender payoff and the corresponding mixed strat.
            self.opt_defender_payoff = plp.value(self.prob.objective)
            self.opt_defender_mixed_strategy = \
                list(map(lambda x: plp.value(x), self.x))
        else:
            self.opt_defender_payoff = float("-inf")

        # save solution_time_with_overhead
        self.solution_time_with_overhead = time.time() - start_time



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
