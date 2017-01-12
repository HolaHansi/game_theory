from games import PatrolGame
import itertools
import pulp as plp
import numpy as np

class MultipleLP:
    def __init__(self, normal_game):
        # number of LPs is number of pure attacker strategies
        self.X, self.Q = normal_game.R.shape
        self.LPs = []
        for q in range(self.Q):
            prob = plp.LpProblem(name="LP-{}"format(q), sense=plp.LpMaximize)
            self.LPs.append(prob)


        # get payoffs and adversary probability distribution
        self.C = game.C
        self.R = game.R
        self.p = game.p

        # get dimensions of a game payoff matrix - needed to generate LP vars
        X, Q, L = self.R.shape
        # save these as instance vars as well.
        self.X = X
        self.Q = Q
        self.L = L


