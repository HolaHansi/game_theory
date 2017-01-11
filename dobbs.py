from games import PatrolGame
import itertools
import pulp as plp
import numpy as np


class Dobbs:
    def __init__(self, game):
        # init the game as an MILP
        self.prob = plp.LpProblem(name="DOBBS", sense=plp.LpMaximize)

        # get payoffs and adversary probability distribution
        self.C = game.attackerPayOffs
        self.R = game.defenderPayOffs
        self.p = game.adversaryProb

        # get dimensions of a game payoff matrix - needed to generate LP vars
        X, Q, L = self.R.shape

        # init z_ijl vars as lp variables
        self.z = np.ndarray(shape=(X,Q,L), dtype=type(plp.LpVariable("dummy")))
        for i,j,l in itertools.product(range(X),range(Q),range(L)):
            self.z[i,j,l] = plp.LpVariable("z_"+str(i)+','+str(j)+','+str(l), \
                                    lowBound=0,
                                    upBound=1,
                                    cat="Contineous")

        # init q_jl vars as lp variables
        self.q = np.ndarray(shape=(Q,L), dtype=type(plp.LpVariable("dummy")))
        for j,l in itertools.product(range(Q),range(L)):
            self.q[j,l] = plp.LpVariable("q_"+str(j)+','+str(l), \
                                    lowBound=0,
                                    upBound=1,
                                    cat="Integer")

        # init a_l vars as lp variables
        self.a = np.ndarray(shape=(L), dtype=type(plp.LpVariable("dummy")))
        for l in range(L):
            self.a[l] = plp.LpVariable("a_"+str(l), cat="Contineous")

        # set objective function of MILP
        self.prob += sum([self.p[l]*self.R[i,j,l]*self.z[i,j,l] for i,j,l in \
                                                itertools.product(range(X), \
                                                                  range(Q), \
                                                                  range(L))])

        # TODO remove this line
        print(self.prob.objective)
        # const. 1 for all l in L
        for l in range(L):
            self.prob += sum([self.z[i,j,l] for i,j in \
                                itertools.product(range(X), range(Q))]) == 1, ""

        # const. 2 for all l in L and i in X
        for i,l in itertools.product(range(X), range(L)):
            self.prob += sum([self.z[i,j,l] for j in range(Q)]) <= 1, ""

        # const 3 for all l in L
        for j, l in itertools.product(range(Q), range(L)):
            self.prob += sum([self.z[i,j,l] for i in range(X)]) >= \
                                                                self.q[j,l], ""
            self.prob += sum([self.z[i,j,l] for i in range(X)]) <= 1, ""

        # const 4
        for l in range(L):
            self.prob += sum([self.q[j,l] for j in range(Q)]) == 1, ""

        # const 5
        for j, l in itertools.product(range(Q), range(L)):
            self.prob += self.a[l] - sum([self.C[i,j,l] * sum(self.z[i,:,l]) \
                                          for i in range(X)]) >= 0
            self.prob += self.a[l] - sum([self.C[i,j,l] * sum(self.z[i,:,l]) \
                                                    for i in range(X)]) <= \
                                                    (1-self.q[j,l])*9999

        # const 6
        for l in range(L):
            self.prob += sum([self.z[i,j,l] for i,j in \
                            itertools.product(range(X), range(Q))]) == \
                            sum([self.z[i,j,0] for i,j in \
                            itertools.product(range(X), range(Q))])

    def write_problem(self, name="problem.MPS"):
        self.prob.writeMPS(name)

    def solve(self):
        self.prob.solve()
        print(self.prob.solutionTime)


x = PatrolGame(7,2,4)
print("game generated")
p = Dobbs(x)
print("dobbs initialized")
p.write_problem()
p.solve()
