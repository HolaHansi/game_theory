import itertools
import numpy as np

# implement class for domain patrol game

class PatrolGame:
    """
    A class implementing the patrol game as specified
    in the paper
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
                        attackerPayOff[i,j] = (p * -self.c_q[a]) + (1-p)*self.v_q[a, j]
                        defenderPayOff[i,j] = (p * self.c_x[a]) + ((1-p)*(-self.v_x[a,j]))
                    else:
                        attackerPayOff[i,j] = v_q(a,j)
                        defenderPayOff[i,j] = -v_x(a,j)
            # normalize payoffs
            attackerPayOff = attackerPayOff - np.amin(attackerPayOff)
            attackerPayOff = attackerPayOff / (np.amax(attackerPayOff) - np.amin(attackerPayOff))

            defenderPayOff = defenderPayOff - np.amin(defenderPayOff)
            defenderPayOff = defenderPayOff / (np.amax(defenderPayOff) - np.amin(defenderPayOff))
            # add to payoffs
            self.attackerPayOffs[:,:,a] = attackerPayOff
            self.defenderPayOffs[:,:,a] = defenderPayOff

            # generate probability distribution over adversaries
            # ASSUMPTION: uniform distribution
            self.adversaryProb = np.zeros(self.num_adversaries)
            self.adversaryProb = 1.0 / self.num_adversaries
