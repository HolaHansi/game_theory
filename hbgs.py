import numpy as np
from games import NormalFormGame, SecurityGame
from multipleLP import SingleLP
from origami import Origami
import time


class HBGS:
    """
    Works on normal form general bayesian games
    """

    def __init__(self, game, origami_for_leaves=False, approx=1.0):
        self.game = game
        self.num_attacker_strategies = game.num_attacker_strategies
        self.num_attacker_types = game.num_attacker_types
        self.attacker_type_probability = game.attacker_type_probability

        # bounds is implemented as a dictionary
        self.bounds = {}

        # feasible strategies
        self.feasible_strategies = {}

        # accumulated LP solver time
        self.solution_time = 0

        # use origami for computing the leaves
        self.origami_for_leaves = origami_for_leaves

        # approximation algorithm
        self.approx = approx

    def _solve(self, attacker_types=None):
        """
        Solve the tree recursively given attacker_types
        """
        # if no attacker_type, we're at the root
        if attacker_types is None:
            attacker_types = tuple(range(self.num_attacker_types))
            self.num_attacker_types = len(attacker_types)

        # print("ATTACKER TYPES: {}".format(attacker_types))
        num_attacker_types = len(attacker_types)
        if num_attacker_types > 1:
            # first solve hierarchially lower games to obtain bounds
            self._solve(attacker_types=attacker_types[:(num_attacker_types//2)])
            self._solve(attacker_types=attacker_types[(num_attacker_types//2):])


            # Optain feasible strategies
            pure_strategies = self._get_feasible_strategies(attacker_types)
            # number of feasible strategies:
            self.num_feasible_strategies = len(pure_strategies)

            print("type: {}, num_pure_strat: {}".format(attacker_types, len(pure_strategies)))

            # print("attackertype: {}".format(attacker_types))
            # print("num feasible strats: {}".format(len(pure_strategies)))

            # Sort the pure strategies in descending order of upper-bounds
            pure_strategies = sorted(pure_strategies,
                                     key = lambda x : self._get_bound(
                                         attacker_types,
                                         x),
                                     reverse=True)
            # Generate partial game
            if self.game.type == "normal":
                partial_game = NormalFormGame(partial_game_from=self.game,
                                            attacker_types=attacker_types)

            else:
                partial_game = SecurityGame(partial_game_from=self.game,
                                            attacker_types=attacker_types)

            # Solve the
            self._solve_pure_strategies(attacker_types,
                                        pure_strategies,
                                        partial_game)

        else:
            # --- BASECASE ---
            if self.origami_for_leaves:
                self._origami_for_leaves(attacker_types[0])
            else:
                # obtain feasible stragies
                pure_strategies = self._get_feasible_strategies(attacker_types)

                # Generate partial game
                if self.game.type == "normal":
                    partial_game = NormalFormGame(partial_game_from=self.game,
                                                attacker_types=attacker_types)

                else:
                    partial_game = SecurityGame(partial_game_from=self.game,
                                                attacker_types=attacker_types)

                # print("num attacker types: {}".format(partial_game.num_attacker_types))
                self._solve_pure_strategies(attacker_types,
                                            pure_strategies,
                                            partial_game)


    def _origami_for_leaves(self, attacker_type):
        # solve this singleton typespace with origami
        origami = Origami(self.game, attacker_type)
        origami.solve()
        attack_set = origami.opt_attack_set

        # save solution time
        self.solution_time += origami.solution_time

        # feasible strategies are all targets in attackset
        self.feasible_strategies[tuple([attacker_type])] = \
            set([tuple([i]) for i in attack_set])

        # bounds are defender payoffs
        for t in range(len(attack_set)):
            target = attack_set[t]
            attacker_types = tuple([attacker_type])
            pure_strat = tuple([target])
            # compute payoff for this pure_strategy
            pay_off = (origami.defender_covered[target] * \
                            origami.opt_coverage[target]) + \
                        ((1 - origami.opt_coverage[target]) * \
                        origami.defender_uncovered[target])
            probability = self.attacker_type_probability[attacker_type]

            self._update_bound(attacker_types,
                               pure_strat,
                               pay_off,
                               probability)

            # update opt_defender payoff for class
            self.opt_defender_payoff = origami.opt_defender_payoff
            self.opt_defender_mixed_strategy = origami.opt_coverage

        return (origami.opt_defender_payoff, origami.opt_coverage)


    def solve(self):
        """
        Visible API for this class
        """
        start_time_overhead = time.time()
        self._solve()
        self.solution_time_with_overhead = time.time() - start_time_overhead

    def _solve_pure_strategy(self, game, pure_strat):
        """
        Will solve the game given the pure strategy and output
        opt payoff and corresponding mixed strategy for defender.
        """
        solver = SingleLP(game, pure_strat)
        start_time = time.time()
        solver.solve()
        self.solution_time += time.time() - start_time
        return (solver.opt_defender_payoff, solver.opt_defender_mixed_strategy)


    def _solve_pure_strategies(self,
                               attacker_types,
                               pure_strategies,
                               partial_game):
        """
        Will solve every pure strategy in pure strategies and update bounds
        and feasible strategies. Will stop if the opt_payoff for a given pure st
        strategy is higher than the upper bound of the next strategy pure stra-
        tegy.
        """

        # solve the partial game for each pure strategy
        max_payoff = float("-inf")
        opt_mixed_strat = None

        for i, pure_strat in enumerate(pure_strategies):
            (opt_defender_payoff, opt_defender_mixed_strategy) = \
                    self._solve_pure_strategy(partial_game, pure_strat)

            # update max_payoff
            if opt_defender_payoff > max_payoff:
                max_payoff = opt_defender_payoff
                opt_mixed_strat = opt_defender_mixed_strategy

            # remove pure strat from feasible_strategies if necessary
            if opt_defender_payoff == float('-inf'):
                self.feasible_strategies[attacker_types].remove(pure_strat)
                continue

            self._update_bound(attacker_types,
                                pure_strat,
                                opt_defender_payoff,
                                partial_game.prob_typespace)

            # if the prob_typespace * opt_payoff is higher than the
            # upper-bound of the next pure strategy, then terminate.
            if len(attacker_types) % 2 == 0:#== self.num_attacker_types:
                    if i < len(pure_strategies) - 1:
                        # print("=======")
                        # # print("this: {}: {}".format(pure_strat, self._get_bound(attacker_types, pure_strat)))
                        # print("this:{}:{}".format(pure_strat, self._get_bound(attacker_types, pure_strat)))
                        # print("next:{}: {}".format(pure_strategies[i+1], self._get_bound(attacker_types, pure_strategies[i+1])))

                        # print("part 1")
                        # # print("PART: {}".format(pure_strategies[i+1][:2]))
                        # print(self._get_bound(attacker_types[:2], pure_strategies[i+1][:2]))


                        # # print("PART: {}".format(pure_strategies[i+1][:2][0]))
                        # print(self._get_bound(tuple([attacker_types[0]]), tuple([pure_strategies[i+1][:2][0]])))

                        # # print("Part: {}".format(pure_strategies[i+1][:2][1]))

                        # print(self._get_bound(tuple([attacker_types[1]]), tuple([pure_strategies[i+1][:2][1]])))


                        # # print("PART: {}".format(pure_strategies[i+1][2:]))
                        # print("part 2")

                        # print(self._get_bound(attacker_types[2:], pure_strategies[i+1][2:]))


                        # # print("PART: {}".format(pure_strategies[i+1][2:][0]))

                        # print(self._get_bound(tuple([attacker_types[2]]), tuple([pure_strategies[i+1][2:][0]])))


                        # # print("PART: {}".format(pure_strategies[i+1][2:][1]))


                        # print(self._get_bound(tuple([attacker_types[3]]), tuple([pure_strategies[i+1][2:][1]])))


                        # print("=======")
                        if self._get_bound(attacker_types, pure_strategies[i+1])\
                                * self.approx \
                        <= self._get_bound(attacker_types, pure_strat):
                            print("BRANCH BOUND")
                            break

        # save opt defender payoff and corresponding mixed strategy
        self.opt_defender_payoff = max_payoff
        self.opt_defender_mixed_strategy = opt_mixed_strat

        return (max_payoff, opt_mixed_strat)


    def _update_bound(self,
                      attacker_types,
                      pure_strat,
                      opt_defender_payoff,
                      prob_typespace):
        """
        Update the bound for a given pure strategy for some attacker_types
        """
        # generate a key for the pure strategy
        pure_strat_key = np.array([-1] * self.num_attacker_types)
        pure_strat_key[list(attacker_types)] = list(pure_strat)
        pure_strat_key = tuple(pure_strat_key)
        # print("UB: {}".format(pure_strat_key))
        # value of bound is the prob of typespace times opt_defender_payoff
        # print("prob_typespace: {}".format(prob_typespace))
        # print("pure_strat_key: {}".format(pure_strat_key))
        # print("opt_defender_payoff: {}".format(opt_defender_payoff))
        new_bound = prob_typespace * opt_defender_payoff
        # update bound
        self.bounds[pure_strat_key] = new_bound

    def _get_bound(self, attacker_types, pure_strat):
        """
        Computes and returns the bound for a given pure strategy.
        As a side effect, will save the new bound to self.bounds
        """
        num_attacker_types = len(attacker_types)

        # generate the key for the pure strategy
        pure_strat_key = np.array([-1] * self.num_attacker_types)
        pure_strat_key[list(attacker_types)] = list(pure_strat)
        pure_strat_key = tuple(pure_strat_key)
        # print("GB: {}".format(pure_strat_key))
        try:
            # basecase
            return self.bounds[pure_strat_key]
        except KeyError:
            # recursively compute, save and return bound
            index = num_attacker_types // 2

            # new bound is sum of bounds
            new_bound = self._get_bound(attacker_types[:index],
                                                 pure_strat[:index]) + \
                        self._get_bound(attacker_types[index:],
                                                 pure_strat[index:])
            # save this bound
            self.bounds[pure_strat_key] = new_bound

            # return new bound
            return new_bound

    def _get_feasible_strategies(self, attacker_types):
        """
        Takes an attacker_types tuple, compute the set of feasible strategies
        and outputs these as a list.
        """
        num_attacker_types = len(attacker_types)
        if num_attacker_types == 1:
            # generate feasible strategies
            strategies = [tuple([i]) for i in range(self.num_attacker_strategies)]
        else:
            f1 = self.feasible_strategies[
                attacker_types[:(num_attacker_types//2)]]
            f2 = self.feasible_strategies[
                attacker_types[(num_attacker_types//2):]]
            strategies = [x+y for x in f1 for y in f2]

        # update feasible strategies
        self.feasible_strategies[attacker_types] = set(strategies)

        # return the feasible strategies as a list
        return strategies

# from games import SecurityGame
# sec_game = SecurityGame(num_targets=10,
#                         max_coverage=3,
#                         num_attacker_types=4)

# game = NormalFormGame(game=sec_game, harsanyi=False)
game = NormalFormGame(num_defender_strategies=5,
                      num_attacker_strategies=5,
                      num_attacker_types=10)


# print("====================")
# solver = HBGS(sec_game, origami_for_leaves=True)
# solver.solve()
# print("++++++++++++++++++++++++++")
# print("payoff: {}".format(solver.opt_defender_payoff))
# print("mixed: {}".format(solver.opt_defender_mixed_strategy))
# print("sol time for HBGS: {}".format(solver.solution_time))
# print("sol OH: {}".format(solver.solution_time_with_overhead))
print("====================")
solver = HBGS(game, False, approx=0.8)
solver.solve()
print("payoff: {}".format(solver.opt_defender_payoff))
print("mixed: {}".format(solver.opt_defender_mixed_strategy))
print("sol time for HBGS: {}".format(solver.solution_time))
print("sol OH: {}".format(solver.solution_time_with_overhead))


# print("====================")

from dobbs import Dobbs
dob = Dobbs(game)
dob.solve()
print("DOBBS")
print("payoff: {}".format(dob.opt_defender_payoff))
print("mixed: {}".format(dob.opt_defender_mixed_strategy))
print("sol time for dobs: {}".format(dob.solution_time))

# print("====================")
# from multipleLP import Multiple_SingleLP
# start_time = time.time()
# mul = Multiple_SingleLP(sec_game)
# mul.solve()
# print("MUL")
# print("REAAAL TIME: {}".format(time.time() - start_time))
# print("payoff: {}".format(mul.opt_defender_payoff))
# print("mixed: {}".format(mul.opt_defender_mixed_strategy))
# print("sol time: {}".format(mul.solution_time))


