from games import PatrolGame
import numpy as np
import itertools
x = PatrolGame(2,2,3)

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
        return (payoff_kefender, payoff_attacker)

p = NormalFormGame(x)
print(p.R)
print(x.defenderPayOffs)
