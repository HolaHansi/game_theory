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
        self.type = game.type
        self.attacker_pure_strategies = itertools.product(
            range(game.num_attacker_strategies),
            repeat=game.num_attacker_types)

        # the accumulated LP solve time
        self.solution_time = 0

        # Create an LP for every attacker pure strategy
        self.LPs = []
        for pure_strat in self.attacker_pure_strategies:
            self.LPs.append(SingleLP(self.game, pure_strat))

    def solve(self):
        start_time = time.time()
        self.opt_defender_payoff = float('-inf')

        if self.type == "normal":
            for lp in self.LPs:
                lp.solve()
                self.solution_time += lp.solution_time
                if lp.opt_defender_payoff > self.opt_defender_payoff:
                    self.opt_defender_payoff = lp.opt_defender_payoff
                    self.opt_defender_mixed_strategy = lp.opt_defender_mixed_strategy
                    self.opt_attacker_pure_strategy = lp.pure_strat
        elif self.type == "compact":
            for lp in self.LPs:
                lp.solve()
                self.solution_time += lp.solution_time
                if lp.opt_defender_payoff > self.opt_defender_payoff:
                    self.opt_defender_payoff = lp.opt_defender_payoff
                    self.opt_coverage = lp.opt_coverage
                    self.opt_defender_mixed_strategy = lp.opt_defender_mixed_strategy
                    self.opt_attacker_pure_strategy = lp.pure_strat

        self.solution_time_with_overhead = time.time() - start_time

class SingleLP:
    """
    Takes a game and a bayesian attacker pure strategy, outputs
    the opt_defender_payoff and corresponding mixed strategy.
    """
    def __init__(self, game, pure_strat):
        self.type = game.type
        L = game.num_attacker_types
        p = game.attacker_type_probability
        self.pure_strat = pure_strat

        # define maximization problem
        self.prob = plp.LpProblem(name="Pure_strat: {}".format(pure_strat),
                            sense=plp.LpMaximize)

        if self.type == "normal":
            R = game.defender_payoffs
            C = game.attacker_payoffs
            X = game.num_defender_strategies
            Q = game.num_attacker_strategies


            # only vars are the mixed-strategy vars for defender
            self.x = [plp.LpVariable("x_{}".format(i),
                                    lowBound=0,
                                    upBound=1,
                                    cat="Continuous") for i in range(X)]

            # objective is expected defender payoff given pure strategy
            self.prob += sum([self.x[i] * sum([
                                        p[k] * R[i, pure_strat[k], k]
                                        for k in range(L)])
                            for i in range(X)])

            # Constraint 1 (pure strategy must be a best response)
            for k in range(L):
                for j_prime in range(Q):
                    self.prob += sum([self.x[i] * C[i,pure_strat[k], k] for i in
                                range(X)]) >= \
                            sum([self.x[i] * C[i, j_prime, k]
                                for i in range(X)])


            # Constraint 2 (x is a prob. distribution)
            self.prob += sum(self.x) == 1

        elif self.type == "compact":
            max_coverage = game.max_coverage
            num_targets = game.num_targets
            defender_uncovered = game.defender_uncovered
            defender_covered = game.defender_covered
            attacker_uncovered = game.attacker_uncovered
            attacker_covered = game.attacker_covered

            self.cov = [plp.LpVariable("cov_{}".format(i),
                                            lowBound=0,
                                            upBound=1,
                                            cat="Continuous")
                             for i in range(num_targets)]

            # objective function is the expected defender payoff given coverage
            self.prob += sum([p[k] * (self.cov[pure_strat[k]] * \
                             defender_covered[pure_strat[k], k] + \
                             (1 - self.cov[pure_strat[k]]) * \
                             defender_uncovered[pure_strat[k], k]) for k \
                             in range(L)])

            # constraint 1 (best response condition)
            for k in range(L):
                for t_p in range(num_targets):
                    self.prob += self.cov[pure_strat[k]] * \
                        attacker_covered[pure_strat[k], k] + \
                        (1-self.cov[pure_strat[k]]) * \
                        attacker_uncovered[pure_strat[k], k] >= \
                        self.cov[t_p] * attacker_covered[t_p, k] + \
                        (1-self.cov[t_p]) * attacker_uncovered[t_p, k]

            # constraint 2 (covereage must be less than max_cov
            self.prob += sum(self.cov) <= max_coverage



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
            if self.type == "normal":
                self.opt_defender_mixed_strategy = \
                        list(map(lambda x: plp.value(x), self.x))
            elif self.type == "compact":
                self.opt_coverage = list(map(lambda x: plp.value(x), self.cov))
                self.opt_defender_mixed_strategy = self.opt_coverage
        else:
            self.opt_defender_payoff = float("-inf")
            self.opt_defender_mixed_strategy = None

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
