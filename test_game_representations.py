import unittest
import random as r
import math
from games import NormalFormGame, SecurityGame

class TestHarsanyiTransformation(unittest.TestCase):

    @staticmethod
    def _combinations(n, r):
        """
        Compute number of n, r combinations w.o. replacement
        """
        f = math.factorial
        return f(n) // f(r) // f(n-r)

    def setUp(self):
        """
        1) Generate a bayesian security game, turn it into normal form,
        and generate the harsanyi transformed normal form.
        2) Generate a bayesian normal form game, generate harsanyi transformed
        normal form game.
        """
        self.sec_game = SecurityGame(10,3,2)
        self.sec_norm_game = NormalFormGame(game=self.sec_game,
                                            harsanyi=False)
        self.sec_norm_hars_game = NormalFormGame(game=self.sec_norm_game)

        # generate bayesian game
        self.bayse_norm_game = NormalFormGame(num_defender_strategies=5,
                                         num_attacker_strategies=10,
                                         num_attacker_types=3)

        self.bayse_norm_hars_game = NormalFormGame(game= self.bayse_norm_game)

    def tearDown(self):
        pass


    def test_dimensions(self):
        """
        Test if dimensions on transformed representations are correct
        """

        # sec_game -> sec_norm_game
        self.assertSequenceEqual(
            (
                self.sec_norm_game.num_defender_strategies,
                self.sec_norm_game.num_attacker_strategies,
                self.sec_norm_game.num_attacker_types
            ),
            (
                self._combinations(self.sec_game.num_targets,
                                    self.sec_game.max_coverage),
                self.sec_game.num_targets,
                self.sec_game.num_attacker_types
            ),
            msg="sec_norm_game dimensions are not correct")

        # sec_norm_game -> sec_norm_hars_game
        self.assertSequenceEqual(
            (
                self.sec_norm_hars_game.num_defender_strategies,
                self.sec_norm_hars_game.num_attacker_strategies,
                self.sec_norm_hars_game.num_attacker_types
            ),
            (
                self.sec_norm_game.num_defender_strategies,
                self.sec_norm_game.num_attacker_strategies ** \
                self.sec_norm_game.num_attacker_types,
                1
            ),
            msg="sec_norm_hars_game dimensions are not correct"
        )

        # bayse_norm_game -> bayse_norm_hars_game
        self.assertSequenceEqual(
            (
                self.bayse_norm_hars_game.num_defender_strategies,
                self.bayse_norm_hars_game.num_attacker_strategies,
                self.bayse_norm_hars_game.num_attacker_types,
            ),
            (
                self.bayse_norm_game.num_defender_strategies,
                self.bayse_norm_game.num_attacker_strategies ** \
                self.bayse_norm_game.num_attacker_types,
                1
            ),
            msg="bayse_norm_hars_game dimensions are not correct"
        )

    def test_payoffs_sec_norm_game(self):
        """
        Test if payoffs have been computed correctly for sec_norm_game
        """
        # repeat test 10 times
        for test_no in range(10):
            # generate random coordinates
            i = r.randrange(0, self.sec_norm_game.num_defender_strategies)
            j = r.randrange(0, self.sec_norm_game.num_attacker_strategies)
            k = r.randrange(0, self.sec_norm_game.num_attacker_types)

            # get coverage corresponding to defender strategy i
            covered_targets = self.sec_norm_game.defender_coverage_tuples[i]

            if j in covered_targets:
                correct_def_payoff = self.sec_game.defender_covered[j, k]
                correct_att_payoff = self.sec_game.attacker_covered[j, k]
            else:
                correct_def_payoff = self.sec_game.defender_uncovered[j, k]
                correct_att_payoff = self.sec_game.attacker_uncovered[j, k]

            # get payoffs for random coordinates
            sec_norm_def_payoff = self.sec_norm_game.defender_payoffs[i,j,k]
            sec_norm_att_payoff = self.sec_norm_game.attacker_payoffs[i,j,k]

            self.assertAlmostEqual(sec_norm_def_payoff,
                                correct_def_payoff,
                                msg="sec_norm_game: defender payoff is wrong")


            self.assertAlmostEqual(sec_norm_att_payoff,
                                correct_att_payoff,
                                msg="sec_norm_game: attacker payoff is wrong")

    def test_payoffs_sec_norm_hars_game(self):
        """
        Test if payoffs have been computed correctly for sec_norm_hars_game
        """
        # repeat test 10 times
        for test_no in range(10):
            # generate random coordinates
            i = r.randrange(0, self.sec_norm_hars_game.num_defender_strategies)
            j = r.randrange(0, self.sec_norm_hars_game.num_attacker_strategies)

            # get attacker pure strategy tuple corresponding to attacker
            # strategy j
            attacker_pure_strategy_tuple = \
                self.sec_norm_hars_game.attacker_pure_strategy_tuples[j]

            # compute correct payoffs
            correct_def_payoff = 0
            correct_att_payoff = 0
            for k, j_p in enumerate(attacker_pure_strategy_tuple):
                correct_def_payoff += \
                    self.sec_norm_game.defender_payoffs[i,j_p,k] * \
                    self.sec_norm_game.attacker_type_probability[k]
                correct_att_payoff += \
                    self.sec_norm_game.attacker_payoffs[i,j_p,k] * \
                    self.sec_norm_game.attacker_type_probability[k]

            # get payoffs for random coordinates
            hars_def_payoff = self.sec_norm_hars_game.defender_payoffs[i,j,0]
            hars_att_payoff = self.sec_norm_hars_game.attacker_payoffs[i,j,0]

            self.assertAlmostEqual(hars_def_payoff,
                                   correct_def_payoff,
                                   msg="sec_norm_hars_game: def payoff wrong")

            self.assertAlmostEqual(hars_att_payoff,
                                   correct_att_payoff,
                                   msg="sec_norm_hars_game: att payoff wrong")
if __name__ == '__main__':
        unittest.main()
