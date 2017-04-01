import unittest
from games import SecurityGame, NormalFormGame
from dobbs import Dobbs
from multipleLP import MultipleLP, Multiple_SingleLP
from eraser import Eraser
from origami import Origami
from origami_milp import OrigamiMILP
from hbgs import HBGS

class TestSolvers(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        """
        Create Games:
        1) Set up a security game, generate corresponding norm_form
        and harsanyi-transformed norm_form game.
        2) Set up a large non-bayesian security game
        3) Set up bayesian norm_form, generate corresponding harsanyi-
        transformed norm_form game.
        4) Set up norm_form game.

        Solve Games:

        """
        # construct games
        # part 1
        self.sec_game = SecurityGame(num_targets=5,
                                     max_coverage=3,
                                     num_attacker_types=1)
        self.sec_norm_game = NormalFormGame(game=self.sec_game,
                                            harsanyi=False)

        self.sec_norm_hars_game = NormalFormGame(game=self.sec_norm_game)


        # part 2
        self.large_sec_game = SecurityGame(num_targets=100,
                                           max_coverage=30,
                                           num_attacker_types=1)

        # part 3
        self.bayse_sec_game = SecurityGame(num_targets=5,
                                           max_coverage=3,
                                           num_attacker_types=2)
        self.bayse_sec_norm_game = NormalFormGame(game=self.bayse_sec_game,
                                            harsanyi=False)
        self.bayse_sec_norm_hars_game = NormalFormGame(
            game=self.bayse_sec_norm_game)

        # part 4
        self.bayse_norm_game = NormalFormGame(num_defender_strategies=10,
                                         num_attacker_strategies=3,
                                         num_attacker_types=3)

        self.bayse_norm_hars_game = NormalFormGame(game=self.bayse_norm_game)

        self.bayse_norm_partial_full_game = NormalFormGame(
            partial_game_from=self.bayse_norm_game,
            attacker_types=(0,1,2))

        self.bayse_norm_partial_game = NormalFormGame(
            partial_game_from=self.bayse_norm_game,
            attacker_types=(1,2))


        # part 5
        self.norm_game = NormalFormGame(num_defender_strategies=20,
                                        num_attacker_strategies=10,
                                        num_attacker_types=1)

        # solve games:
        # part 1 (non-bayesian security games)
        print("solving part 1")
        self.p1_eraser = Eraser(self.sec_game)
        self.p1_origami = Origami(self.sec_game)
        self.p1_origami_milp = OrigamiMILP(self.sec_game)
        self.p1_dobbs = Dobbs(self.sec_norm_game)
        self.p1_multLP = MultipleLP(self.sec_norm_hars_game)
        self.p1_multSingLP_sec_game = Multiple_SingleLP(self.sec_game)
        self.p1_multSingLP_sec_norm_game = Multiple_SingleLP(self.sec_norm_game)
        self.p1_multSingLP_sec_norm_hars_game = Multiple_SingleLP(
            self.sec_norm_hars_game)

        self.p1_eraser.solve()
        self.p1_origami.solve()
        self.p1_origami_milp.solve()
        self.p1_dobbs.solve()
        self.p1_multLP.solve()
        self.p1_multSingLP_sec_game.solve()
        self.p1_multSingLP_sec_norm_game.solve()
        self.p1_multSingLP_sec_norm_hars_game.solve()

        # part 2 (large security game)
        print("solving part 2")
        self.p2_large_origami = Origami(self.large_sec_game)
        self.p2_large_origami_milp = OrigamiMILP(self.large_sec_game)
        self.p2_large_eraser = Eraser(self.large_sec_game)

        self.p2_large_origami.solve()
        self.p2_large_origami_milp.solve()
        self.p2_large_eraser.solve()

        # part 3 (bayseian security games)
        print("solving part 3")
        self.p3_dobbs = Dobbs(self.bayse_sec_norm_game)
        self.p3_multLP = MultipleLP(self.bayse_sec_norm_hars_game)
        self.p3_multSingLP = Multiple_SingleLP(self.bayse_sec_game)
        self.p3_hbgs = HBGS(self.bayse_sec_game)
        self.p3_hbgs_origami = HBGS(self.bayse_sec_game, True)
        self.p3_hbgs_norm = HBGS(self.bayse_sec_norm_game)

        self.p3_dobbs.solve()
        self.p3_multLP.solve()
        self.p3_multSingLP.solve()
        self.p3_hbgs.solve()
        self.p3_hbgs_origami.solve()
        self.p3_hbgs_norm.solve()

        # part 4 (bayesian norm_form game)
        print("solving part 4")
        self.p4_dobbs = Dobbs(self.bayse_norm_game)
        self.p4_multLP = MultipleLP(self.bayse_norm_hars_game)
        self.p4_multSingLP = Multiple_SingleLP(self.bayse_norm_game)

        self.p4_dobbs_partial_full = Dobbs(self.bayse_norm_partial_full_game)
        self.p4_multSingLP_partial_full = \
            Multiple_SingleLP(self.bayse_norm_partial_full_game)

        self.p4_dobbs_partial = Dobbs(self.bayse_norm_partial_game)
        self.p4_multSingLP_partial = \
            Multiple_SingleLP(self.bayse_norm_partial_game)

        self.p4_hbgs = HBGS(self.bayse_norm_game)

        self.p4_dobbs.solve()
        self.p4_multLP.solve()
        self.p4_multSingLP.solve()
        self.p4_dobbs_partial.solve()
        self.p4_dobbs_partial_full.solve()
        self.p4_multSingLP_partial_full.solve()
        self.p4_dobbs_partial.solve()
        self.p4_multSingLP_partial.solve()
        self.p4_hbgs.solve()

        # part 5
        print("solving part 5")
        self.p5_dobbs = Dobbs(self.norm_game)
        self.p5_multLP = MultipleLP(self.norm_game)
        self.p5_multSingLP = Multiple_SingleLP(self.norm_game)

        self.p5_dobbs.solve()
        self.p5_multLP.solve()
        self.p5_multSingLP.solve()




    def test_p1(self):
        """
        Test if opt_defender_payoff is the same across all solutions
        """


        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_eraser.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_multLP.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_origami.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_origami_milp.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_multSingLP_sec_game.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_multSingLP_sec_norm_game.\
                               opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p1_dobbs.opt_defender_payoff,
                               self.p1_multSingLP_sec_norm_hars_game.\
                               opt_defender_payoff,
                               places=1)

    def test_p2(self):
        """
        Test if solvers taking compact representations of large games
        are in agreement.
        """
        # test that every opt_target is the same for all solvers
        ori_target = self.p2_large_origami.opt_attacked_target
        ori_milp_target = self.p2_large_origami_milp.opt_attacked_target
        ers_target = self.p2_large_eraser.opt_attacked_target

        print("ORIGAMI SOL: {}".format(self.p2_large_origami.solution_time))
        print("ORIGAMI OH SOL: {}".format(self.p2_large_origami.solution_time_with_overhead))
        print("ORI_MILP SOL: {}".format(self.p2_large_origami_milp.solution_time))
        print("ORI_MILP OH SOL: {}".format(self.p2_large_origami_milp.solution_time_with_overhead
                                           ))


        self.assertEqual(ori_target,
                         ori_milp_target,
                         msg="opt target disagree: origami vs. origami_milp")
        self.assertEqual(ori_milp_target,
                         ers_target,
                         msg="opt target disagree: origami_milp vs. eraser")

        # test that coverage is the same for attacked target
        self.assertAlmostEqual(self.p2_large_origami.opt_coverage[ori_target],
                               self.p2_large_origami_milp.opt_coverage[ori_milp_target],
                               places=1,
                msg="cov for attacked target disagree: origami vs. origami_milp")

        self.assertAlmostEqual(self.p2_large_origami.opt_coverage[ori_target],
                               self.p2_large_eraser.opt_coverage[ers_target],
                               places=1,
                msg="cov for attacked target disagree: origami vs. eraser")
        # test that payoff is the same
        self.assertAlmostEqual(self.p2_large_origami.opt_defender_payoff,
                               self.p2_large_origami_milp.opt_defender_payoff,
                               places=1,
                               msg="payoff disagreement: orig vs. orig-milp")
        self.assertAlmostEqual(self.p2_large_origami.opt_defender_payoff,
                               self.p2_large_eraser.opt_defender_payoff,
                               places=1,
                               msg="payoff disagreement: ori vs. eraser")

        # test that attackset are the same for origami and origami-milp
        self.assertEqual(len(self.p2_large_origami.opt_attack_set),
                         len(self.p2_large_origami_milp.opt_attack_set),
                         msg="attackset diff. length: origami vs. origami-milp")


    def test_p3(self):
        """
        Test if opt_defender_payoff is the same across all solutions
        """
        self.assertAlmostEqual(self.p3_dobbs.opt_defender_payoff,
                               self.p3_multLP.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p3_dobbs.opt_defender_payoff,
                               self.p3_multSingLP.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p3_hbgs.opt_defender_payoff,
                               self.p3_dobbs.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p3_hbgs_origami.opt_defender_payoff,
                               self.p3_dobbs.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p3_hbgs_norm.opt_defender_payoff,
                               self.p3_dobbs.opt_defender_payoff,
                               places=1)

    def test_p4(self):
        """
        Test if bayesian normal form game solvers are in agreement.
        """
        # print("sol tim, dob, mltLP vs. Singl")
        # print(self.p4_dobbs.solution_time)
        # print(self.p4_multLP.solution_time)
        # print(self.p4_multSingLP.solution_time)



        # test that the expected opt defender payoff is the same
        self.assertAlmostEqual(self.p4_dobbs.opt_defender_payoff,
                               self.p4_multLP.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p4_dobbs.opt_defender_payoff,
                               self.p4_multSingLP.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p4_dobbs_partial_full.opt_defender_payoff,
                               self.p4_multSingLP_partial_full.\
                               opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p4_dobbs_partial.opt_defender_payoff,
                               self.p4_multSingLP_partial.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p4_hbgs.opt_defender_payoff,
                               self.p4_dobbs.opt_defender_payoff,
                               places=1)
        # test that opt defender strat. yields the same attacker pure strategy
        self.assertSequenceEqual(self.p4_dobbs.opt_attacker_pure_strategy,
                                 self.p4_multSingLP.opt_attacker_pure_strategy)

        # self.assertSequenceEqual(self.p4_dobbs.opt_attacker_pure_strategy,
        #                          self.bayse_norm_hars_game.\
        #                          attacker_pure_strategy_tuples[self.p4_multLP.\
        #                        opt_attacker_pure_strategy])

    def test_p5(self):
        """
        Test agreement between all solvers of non-bayesian normal form games.
        """
        # test that the expected opt defender payoff is the same
        self.assertAlmostEqual(self.p5_dobbs.opt_defender_payoff,
                               self.p5_multLP.opt_defender_payoff,
                               places=1)
        self.assertAlmostEqual(self.p5_dobbs.opt_defender_payoff,
                               self.p5_multSingLP.opt_defender_payoff,
                               places=1)
        # test that opt defender strat. yields the same attacker pure strategy
        self.assertSequenceEqual(self.p5_dobbs.opt_attacker_pure_strategy,
                                 self.p5_multSingLP.opt_attacker_pure_strategy)
        self.assertEqual(self.p5_dobbs.opt_attacker_pure_strategy[0],
                         self.p5_multLP.opt_attacker_pure_strategy)


if __name__ == '__main__':
        unittest.main()
