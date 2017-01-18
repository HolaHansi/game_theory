import unittest
import random as r
from games import PatrolGame, NormalFormGame

class TestHarsanyiTransformation(unittest.TestCase):
    def setUp(self):
        # generate a bayesian game (PatrolGame)
        # of 5 houses, 2 patrol length and 3 adversaries
        # and contruct the equivalent normalform game
        # using the harsanyi transformation
        self.bayseGame = PatrolGame(5, 2, 3)
        self.normGame = NormalFormGame(self.bayseGame)

    def tearDown(self):
        pass

    def test_dimensions(self):
        # test to see if dimensions fit
        # note: there are Q**L number of attacker strategies in norm game
        (X, Q, L) = self.bayseGame.attackerPayOffs.shape
        (X_d, Q_d) = self.normGame.R.shape
        (X_a, Q_a) = self.normGame.C.shape
        self.assertTrue((X, Q**L)==(X_d, Q_d) and (X, Q**L)==(X_a, Q_a))

    def test_payoff_follows_harsanyi_form(self):
        # randomly generate coordinates in harsanyi payoff matrices
        # and see if these values were correctly computed
        # run the experiment 10 times
        for k in range(10):
            i = r.randrange(0, self.normGame.X)
            j = r.randrange(0, self.normGame.Q**self.normGame.L)
            harsanyi_val_R = self.normGame.R[i,j]
            harsanyi_val_C = self.normGame.C[i,j]
            comp_val_R, comp_val_C = (0,0)
            # compute the vars from original baysian game
            for l in range(self.normGame.L):
                comp_val_R += \
                    self.normGame.p[l] * \
                self.normGame.R_original[i, self.normGame.q[j][l], l]

                comp_val_C += \
                    self.normGame.p[l] * \
                self.normGame.C_original[i, self.normGame.q[j][l], l]
            # check if computed vars equal to harsanyi vars in normGame
            self.assertEqual(harsanyi_val_R, comp_val_R, "R doesn't match for \
                             i,j = ({},{}) in run {}".format(i,j,k))

            self.assertEqual(harsanyi_val_C, comp_val_C, "R doesn't match for \
                             i,j = ({},{}) in run {}".format(i,j,k))

if __name__ == '__main__':
        unittest.main()
