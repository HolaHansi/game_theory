import itertools
import numpy as np

class PatrolGame:
    """
    A class implementing the patrol game as specified
    in "An Efficient Heuristic for Security Against Multiple Adversaries in Stackelberg
    Games" Paruchi et al.
    The PatrolGame is a bayesian stackelberg game and holds the following values:
    m: number of houses
    d: length of patrol
    num_adversaries: possible types of adversaries
    v_x[l,m]: security agent's valulation of house m when facing adversary l
    v_q[l,m]: Adversary's valuation of house m when of type l
    c_x[l]: security agent's reward for catching adversary of type l
    c_q[l]: Adversaries cost of getting caught when of type l
    TODO: Finish this description


    """
    def __init__(self, m, d, num_adversaries):
        # save args as instance variables
        self.m = m
        self.d = d
        self.num_adversaries = num_adversaries
        # generate random valuations
        self.v_x = np.random.rand(num_adversaries, m)
        self.v_q = np.random.rand(num_adversaries, m)
        # and costs
        self.c_x = np.random.rand(num_adversaries)
        self.c_q = np.random.rand(num_adversaries)
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

        # - generate Pl probabilitis that robber is caught for each house
        # along the d-path
        # assuming linearity
        self.Pl = np.zeros(d)
        for index in range(len(self.Pl)):
            self.Pl[index] = 1 - (float((index+1)) / (d+1))
        # - generate payoff matrices
        self.attackerPayOffs = np.ndarray(shape=(len(self.X),
                                                 len(self.Q),
                                                 self.num_adversaries),
                                          dtype=float
                                          )
        self.defenderPayOffs = np.ndarray(shape=(len(self.X),
                                                 len(self.Q),
                                                 self.num_adversaries),
                                          dtype=float
                                          )
        for a in range(self.num_adversaries):
            attackerPayOff = np.zeros((len(self.X), len(self.Q)))
            defenderPayOff = np.zeros((len(self.X), len(self.Q)))
            for i in range(len(self.X)):
                for j in range(len(self.Q)):
                    if j in self.X[i]:
                        index = self.X[i].index(j)
                        p = self.Pl[index]
                        attackerPayOff[i,j] = (p * -self.c_q[a]) + \
                                                (1-p)*self.v_q[a, j]
                        defenderPayOff[i,j] = (p * self.c_x[a]) + \
                                                ((1-p)*(-self.v_x[a,j]))
                    else:
                        attackerPayOff[i,j] = self.v_q[a,j]
                        defenderPayOff[i,j] = -self.v_x[a,j]
            # normalize payoffs
            attackerPayOff = attackerPayOff - np.amin(attackerPayOff)
            attackerPayOff = attackerPayOff / (np.amax(attackerPayOff) - \
                                               np.amin(attackerPayOff))

            defenderPayOff = defenderPayOff - np.amin(defenderPayOff)
            defenderPayOff = defenderPayOff / (np.amax(defenderPayOff) - \
                                               np.amin(defenderPayOff))
            # add to payoffs
            self.attackerPayOffs[:,:,a] = attackerPayOff
            self.defenderPayOffs[:,:,a] = defenderPayOff

        # generate probability distribution over adversaries
        # ASSUMPTION: uniform distribution
        self.adversaryProb = np.zeros(self.num_adversaries)
        self.adversaryProb[:] = 1.0 / self.num_adversaries

class NormalFormGame:
    """
    NormalFormGame class is mainly used for transforming bayesian
    games into Normal Form games using the Harsanyi transformation
    """
    def __init__(self, bayesian_game):
        # Takes a Bayesian Stackelberg Game and transforms to
        # normalform using the Harsanye Transformation
        self.C_original = bayesian_game.attackerPayOffs
        self.R_original = bayesian_game.defenderPayOffs
        self.p = bayesian_game.adversaryProb
        self.X, self.Q, self.L = self.R_original.shape
        # generate attacker strategies
        self.q =list(itertools.product(*[range(self.Q) for i in range(self.L)]))
        # the payoff matrix of the normal form has Q^R
        self.R = np.ndarray(shape=(self.X, self.Q**self.L))
        self.C = np.ndarray(shape=(self.X, self.Q**self.L))
        # construct payoff matrix
        for j, pure_strat in enumerate(self.q):
            for i in range(self.X):
                self.R[i,j], self.C[i,j] = self._get_payoffs(i, pure_strat)

    def _get_payoffs(self, i, pure_strat):
        """
        The payoff given a pure strategy e.g. (2,3,4) where l is 3
        and adversary commits to playing 2 if l=0, 3 if l=1 etc.
        is the probability of facing adversary type l times the payoff
        of that situation occuring.
        """
        payoff_defender = 0
        payoff_attacker = 0
        for l in range(self.L):
            payoff_defender += self.R_original[i,pure_strat[l],l]*self.p[l]
            payoff_attacker += self.C_original[i,pure_strat[l],l]*self.p[l]
        return (payoff_defender, payoff_attacker)

class SecurityGame:
    """
    A security game is a non-bayesian game in which the payoffs
    are described in terms of coverage: covered vs. uncovered.
    first row is uncovered, second row is covered payoff.
    """
    def __init__(self, num_targets, max_coverage):
        self.attackerPayOffs = np.random.rand(2,num_targets)
        self.defenderPayOffs = np.random.rand(2,num_targets)
        self.num_targets = num_targets
        self.max_coverage = max_coverage

        self.attackerPayOffs[0,:] = self.attackerPayOffs[0,:]*100
        self.defenderPayOffs[0,:] = self.defenderPayOffs[0,:]*-100
        self.attackerPayOffs[1,:] = self.attackerPayOffs[1,:]*-100
        self.defenderPayOffs[1,:] = self.defenderPayOffs[1,:]*100


x = SecurityGame(10, 3)
print(x.attackerPayOffs)
print(x.defenderPayOffs)

