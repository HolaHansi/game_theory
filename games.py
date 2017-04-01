import itertools
import numpy as np

class PatrolGame:
    """
    Class implementing the patrol game as specified
    in "An Efficient Heuristic for Security Against Multiple Adversaries in Stackelberg
    Games" Paruchi et al.
    The PatrolGame is a bayesian normal-form stackelberg game and holds the
    following values:
    m: number of houses
    d: length of patrol
    num_attacker_types: number of attacker types
    v_x[l,m]: security agent's valulation of house m when facing adversary l
    v_q[l,m]: Adversary's valuation of house m when of type l
    c_x[l]: security agent's reward for catching adversary of type l
    c_q[l]: Adversaries cost of getting caught when of type l
    """
    def __init__(self, m, d, num_attacker_types):
        # save args as instance variables
        self.m = m
        self.d = d
        self.num_attacker_types = num_attacker_types
        # generate random valuations
        self.v_x = np.random.rand(num_attacker_types, m)
        self.v_q = np.random.rand(num_attacker_types, m)
        # and costs
        self.c_x = np.random.rand(num_attacker_types)
        self.c_q = np.random.rand(num_attacker_types)
        # - generate pure defender strategies
        # targets are indexed 0 to m-1
        targets = np.arange(m)
        if m == d:
            # strategies are permutations
            self.X = list(itertools.permutations(targets))
        else:
            # strategies are permutations of (m, d) all combinations
            combs = itertools.combinations(targets, d)
            _X = map(lambda x: list(itertools.permutations(x)), list(combs))
            self.X = list(itertools.chain(*_X))

        # - generate pure attacker strategies
        self.Q = np.arange(m)

        # save num of strategies
        self.num_defender_strategies = len(self.X)
        self.num_attacker_strategies = len(self.Q)

        # - generate Pl probabilitis that robber is caught for each house
        # along the d-path
        # assuming linearity
        self.Pl = np.zeros(d)
        for index in range(len(self.Pl)):
            self.Pl[index] = 1 - (float((index+1)) / (d+1))
        # - generate payoff matrices
        self.attacker_payoffs = np.ndarray(shape=(self.num_defender_strategies,
                                                 self.num_attacker_strategies,
                                                 self.num_attacker_types),
                                          dtype=float
                                          )
        self.defender_payoffs = np.ndarray(shape=(self.num_defender_strategies,
                                                 self.num_attacker_strategies,
                                                 self.num_attacker_types),
                                          dtype=float
                                          )
        for a in range(self.num_attacker_types):
            attacker_payoff = np.zeros((self.num_defender_strategies,
                                        self.num_attacker_strategies))
            defender_payoff = np.zeros((self.num_defender_strategies,
                                        self.num_attacker_strategies))
            for i in range(len(self.X)):
                for j in range(len(self.Q)):
                    if j in self.X[i]:
                        index = self.X[i].index(j)
                        p = self.Pl[index]
                        attacker_payoff[i,j] = (p * -self.c_q[a]) + \
                                                (1-p)*self.v_q[a, j]
                        defender_payoff[i,j] = (p * self.c_x[a]) + \
                                                ((1-p)*(-self.v_x[a,j]))
                    else:
                        attacker_payoff[i,j] = self.v_q[a,j]
                        defender_payoff[i,j] = -self.v_x[a,j]

            # normalize payoffs
            attacker_payoff -= np.amin(attacker_payoff)
            attacker_payoff = attacker_payoff / (np.amax(attacker_payoff) - \
                                               np.amin(attacker_payoff))

            defender_payoff -= np.amin(defender_payoff)
            defender_payoff = defender_payoff / (np.amax(defender_payoff) - \
                                               np.amin(defender_payoff))
            # add to payoffs
            self.attacker_payoffs[:,:,a] = attacker_payoff
            self.defender_payoffs[:,:,a] = defender_payoff

        # generate probability distribution over adversaries
        # assume uniform distribution.
        self.attacker_type_probability = np.zeros(self.num_attacker_types)
        self.attacker_type_probability[:] = 1.0 / self.num_attacker_types

        self.type = "normal"

class NormalFormGame:
    def __init__(self, **kwargs):
        """
        Transform given game into normalform if it's compact form,
        Conduct harsanyi transformation if requested or if given game is already
        in normalform.
        Otherwise, generate a random game given arguments.
        """
        if "partial_game_from" in kwargs.keys():
            # generate a partial game from the given game and attacker_types
            self.game = kwargs['partial_game_from']
            self.attacker_types = kwargs['attacker_types']
            self._create_partial_game()

        elif "game" in kwargs.keys():
            self.game = kwargs['game']

            if self.game.type == "compact":
                # produce the normal form
                self._compact_to_normal()
                if kwargs["harsanyi"]:
                    # treat self as given game
                    self.game = self
                    # perform harsanyi
                    self._harsanyi()

            elif self.game.type == "normal":
                # game already in normal form, so perform harsanyi
                self._harsanyi()

        # game was not provided, so generate a random game
        else:
            self._generate_new_game(kwargs)

        # record the type of this game
        self.type = "normal"

    def _generate_new_game(self, kwargs):
        """
        Generate a new game
        """
        self.num_defender_strategies = kwargs['num_defender_strategies']
        self.num_attacker_strategies = kwargs['num_attacker_strategies']
        self.num_attacker_types = kwargs['num_attacker_types']

        # uniform distribution over attacker types
        self.attacker_type_probability = np.zeros((self.num_attacker_types))
        self.attacker_type_probability += (1.0 / self.num_attacker_types)

        # init payoff matrices
        self.attacker_payoffs = np.random.rand(self.num_defender_strategies,
                                               self.num_attacker_strategies,
                                               self.num_attacker_types)

        self.defender_payoffs = np.random.rand(self.num_defender_strategies,
                                               self.num_attacker_strategies,
                                               self.num_attacker_types)

        # payoffs should be between -100 and 100
        self.attacker_payoffs = (self.attacker_payoffs * 200) - 100
        self.defender_payoffs = (self.defender_payoffs * 200) - 100



    def _compact_to_normal(self):
        """
        every possible comination of pure coverages is a defender strategy
        get the number of possible cominations
        combinations =
        """

        # save the attacker type probability
        self.attacker_type_probability = self.game.attacker_type_probability

        # compute all possible defender strategies i.e. comb. of coverage
        self.defender_coverage_tuples = list (
                                    itertools.combinations(
                                    range(self.game.num_targets),
                                    self.game.max_coverage)
                                    )

        self.num_defender_strategies = len(self.defender_coverage_tuples)
        self.num_attacker_strategies = self.game.num_targets
        self.num_attacker_types = self.game.num_attacker_types

        # init payoff matrices
        self.defender_payoffs = np.zeros((self.num_defender_strategies,
                                         self.game.num_targets,
                                         self.game.num_attacker_types ))
        self.attacker_payoffs = np.zeros((self.num_defender_strategies,
                                         self.game.num_targets,
                                         self.game.num_attacker_types ))

        # calculate the appropriate payoffs
        for t in range(self.game.num_targets):
            for i, strat in enumerate(self.defender_coverage_tuples):
                if t in strat:
                    # t is covered
                    self.defender_payoffs[i, t, :] = \
                        self.game.defender_covered[t, :]
                    self.attacker_payoffs[i, t, :] = \
                        self.game.attacker_covered[t, :]
                else:
                    # t is not covered
                    self.defender_payoffs[i, t, :] = \
                        self.game.defender_uncovered[t, :]
                    self.attacker_payoffs[i, t, :] = \
                        self.game.attacker_uncovered[t, :]

    def _harsanyi(self):
        """
        Compute the harsanyi transformed game i.e. turn payoffs into
        a single attacker_type.
        """

        # get dimensions of payoff matrices
        self.num_defender_strategies = self.game.num_defender_strategies
        self.num_attacker_strategies = self.game.num_attacker_strategies ** \
                                        self.game.num_attacker_types
        self.num_attacker_types = 1
        self.attacker_type_probability = np.array([1])

        # generate pure strategy tuples
        self.attacker_pure_strategy_tuples = \
            list(itertools.product(*[range(self.game.num_attacker_strategies)
                                for i in range(self.game.num_attacker_types)]))

        # initiate new defender and attacker payoff matrices
        self.defender_payoffs = np.zeros((self.num_defender_strategies,
                                          self.num_attacker_strategies,
                                          self.num_attacker_types))
        self.attacker_payoffs = np.zeros((self.num_defender_strategies,
                                          self.num_attacker_strategies,
                                          self.num_attacker_types))

        # compute payoffs
        for j, pure_strat in enumerate(self.attacker_pure_strategy_tuples):
            for i in range(self.num_defender_strategies):
                self.defender_payoffs[i, j, 0], self.attacker_payoffs[i, j, 0] \
                    = self._get_payoffs(i, pure_strat)

    def _get_payoffs(self, i, pure_strat):
        """
        compute the defender and attacker expected payoff for a given
        pure strategy.
        """
        payoff_defender = 0
        payoff_attacker = 0

        for l in range(self.game.num_attacker_types):
            payoff_defender += \
            self.game.defender_payoffs[i, pure_strat[l], l] * \
                self.game.attacker_type_probability[l]
            payoff_attacker += \
            self.game.attacker_payoffs[i, pure_strat[l], l] * \
                self.game.attacker_type_probability[l]

        return (payoff_defender, payoff_attacker)

    def _create_partial_game(self):
        """
        Make a partial game out of game and attacker_types.
        """
        self.defender_payoffs = self.game.defender_payoffs[:,:,
                                                    list(self.attacker_types)]
        self.attacker_payoffs = self.game.attacker_payoffs[:,:,
                                                    list(self.attacker_types)]

        self.num_defender_strategies = self.game.num_defender_strategies
        self.num_attacker_strategies = self.game.num_attacker_strategies
        self.num_attacker_types = len(self.attacker_types)
        self.attacker_type_probability = np.zeros((self.num_attacker_types))

        # normalize attacker type probabilities
        # Save the the probability of this typespace
        self.prob_typespace = float(
            self.game.attacker_type_probability[list(self.attacker_types)].sum())

        for i, t in enumerate(self.attacker_types):
            self.attacker_type_probability[i] = \
                self.game.attacker_type_probability[t] / self.prob_typespace


# TODO enable SecurityGame to deal with partial games
class SecurityGame:
    """
    A security game is a non-bayesian game in which the payoffs for targets
    are given by their coverage status.
    Covered targets yield higher utilities for the defender, and lower
    utilities for the attacker, whilst uncovered targets yield negative
    utilities for the defender and positive utilities for the attacker.
    """
    def __init__(self, **kwargs):
        if "partial_game_from" in kwargs.keys():
            self.game = kwargs['partial_game_from']
            self.attacker_types = kwargs['attacker_types']
            self._create_partial_game()
        else:
            self.num_targets = kwargs['num_targets']
            self.max_coverage = kwargs['max_coverage']
            self.num_attacker_types = kwargs['num_attacker_types']

            # for comparisons with other algos
            self.num_attacker_strategies = self.num_targets

            # uniform distribution over attacker types
            self.attacker_type_probability = np.zeros((self.num_attacker_types))
            self.attacker_type_probability += (1.0 / self.num_attacker_types)

            # generate two arrays of random floats for defender and attacker
            attacker_random = np.random.rand(2,
                                             self.num_targets,
                                             self.num_attacker_types)
            defender_randoms = np.random.rand(2,
                                              self.num_targets,
                                              self.num_attacker_types)

            # for attacker uncovered targets yield positive utilities, and covered
            # yields negative utilities.
            self.attacker_uncovered = attacker_random[0,:,:] * 100
            self.attacker_covered = attacker_random[1,:,:] * -100

            # for defender uncovered targets yield negative utilities, and covered
            # targets yield positive utilities.
            self.defender_uncovered = defender_randoms[0,:,:] * -100
            self.defender_covered = defender_randoms[1,:,:] * 100

        # store the type of this representation
        self.type = "compact"

    def _create_partial_game(self):
        """
        Make a partial game out of game and attacker_types.
        """
        # get payoffs
        self.defender_uncovered = self.game.defender_uncovered[:,
                                                    list(self.attacker_types)]
        self.defender_covered = self.game.defender_covered[:,
                                                    list(self.attacker_types)]
        self.attacker_uncovered = self.game.attacker_uncovered[:,
                                                    list(self.attacker_types)]
        self.attacker_covered = self.game.attacker_covered[:,
                                                    list(self.attacker_types)]
        # get compact values
        self.max_coverage = self.game.max_coverage
        self.num_targets = self.game.num_targets
        self.num_attacker_strategies = self.game.num_attacker_strategies

        self.num_attacker_types = len(self.attacker_types)
        self.attacker_type_probability = np.zeros((self.num_attacker_types))

        # normalize attacker type probabilities
        # Save the the probability of this typespace
        self.prob_typespace = float(
            self.game.attacker_type_probability[list(self.attacker_types)].sum())

        for i, t in enumerate(self.attacker_types):
            self.attacker_type_probability[i] = \
                self.game.attacker_type_probability[t] / self.prob_typespace
