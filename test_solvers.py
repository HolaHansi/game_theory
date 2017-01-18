import unittest
from games import PatrolGame, NormalFormGame
from dobbs import Dobbs
from multipleLP import MultipleLP
import itertools
import pulp as plp

class TestSolvers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bayseGame = PatrolGame(4,2,3)
        cls.normGame = NormalFormGame(cls.bayseGame)
        cls.dob = Dobbs(cls.bayseGame)
        cls.mlp = MultipleLP(cls.normGame)
        # now solve probs
        cls.dob.solve()
        cls.mlp.solve()

    def test_dob_sol_add_to_one(self):
        self.assertLessEqual(abs(1.0-sum(self.dob.opt_x)), 0.0001)


    def test_mlp_sol_add_to_one(self):
        self.assertLessEqual(abs(1.0-sum(self.mlp.opt_x)), 0.0001)

    def test_dob_mlp_same_obt_val(self):
        self.assertLessEqual(abs(self.mlp.opt_value-self.dob.opt_value),0.001)

    def test_mlp_opt_x_yields_opt_val_for_dob(self):
        """
        Computes the objective value for dobbs with the policy values computed
        in mlp and checks if they are almost identical (error <= 0.01)
        """
        sol_dob = 0
        for i,j,l in itertools.product(range(self.dob.X),
                                       range(self.dob.Q),
                                       range(self.dob.L)):
            sol_dob += self.dob.p[l]*plp.value(self.dob.R[i,j,l]) \
                *(plp.value(self.dob.q[j,l])* plp.value(self.mlp.opt_x[i]))

        self.assertLessEqual(abs(sol_dob-self.dob.opt_value), 0.01)

    def test_dob_opt_x_yields_opt_val_for_dob(self):
        """
        Tests the converse of the function above.
        """
        sol_mlp = 0
        for i in range(self.dob.X):
            sol_mlp += self.dob.opt_x[i]*self.mlp.R[i,self.mlp.opt_q]

        self.assertLessEqual(abs(sol_mlp-self.mlp.opt_value), 0.01)


if __name__ == '__main__':
        unittest.main()
