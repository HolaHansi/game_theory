# from games import SecurityGame
import numpy as np
import time

class Origami:
    """
    Origami is the procedure for computing the maximal attack-set
    it's given a security game, and will output the optimal coverage
    """
    def __init__(self, game, attacker_type=0):
        # copy game vars
        self.attacker_uncovered = game.attacker_uncovered[:, attacker_type]
        self.attacker_covered = game.attacker_covered[:, attacker_type]
        self.defender_uncovered = game.defender_uncovered[:, attacker_type]
        self.defender_covered = game.defender_covered[:, attacker_type]
        self.num_targets = game.num_targets
        self.max_coverage = game.max_coverage
        # init the coverage vector
        self.coverage = np.zeros((self.num_targets))

    def solve(self):
        """
        This is the function that will actually run the ORIGAMI algorithm
        it will increase the attack-set with each round of the for-loop
        and terminate if one of two things happen:
            1) Run out of coverage to allocate
            2) a target is assigned coverage = 1
        This implementation follows roughly the pseudo-code by Kiekintveld.
        """
        # record start time
        start_time = time.time()

        # get the targets sorted in descending order by attacker_uncovered payoff
        # and obtain the covered and uncovered payoff of these targets.
        sorted_targets = np.argsort(self.attacker_uncovered)[::-1]
        uncovered_payoff = np.take(self.attacker_uncovered, sorted_targets)
        covered_payoff = np.take(self.attacker_covered, sorted_targets)
        # the amount of coverage left after each round of the while-loop
        left = self.max_coverage
        # the coverage assigned to each target
        coverage = np.zeros((self.num_targets, 1))
        # keeps a record of how much coverage must be added to each target
        # to include next_target in the attack_set.
        added_coverage = np.zeros((self.num_targets,1))
        # these are the ratios of each target in the attackset
        # we compute this value in the while-loop for optimization purposes.
        ratio = np.zeros((self.num_targets, 1))
        # coverage_bound is updated only when a coverage > 1 is assigned
        coverage_bound = float('-inf')
        # the next target to be appended to the attack_set.
        next_target = 1
        while next_target < self.num_targets:
            # my own optimization - compute ratio ad hoc
            ratio[next_target - 1] = 1 / \
             (uncovered_payoff[next_target-1] - covered_payoff[next_target-1])
            # update coverage of every target in the current attack-set
            for t in range(next_target):
                # compute how much pay-off must be added to t
                # so to make attacker indifferent between t and next_target.
                added_coverage[t] = ((uncovered_payoff[next_target] - \
                                    uncovered_payoff[t]) / \
                                    (covered_payoff[t] - uncovered_payoff[t])) \
                                    - coverage[t]
                if added_coverage[t] + coverage[t] >= 1:
                    # we cannot assign coverage > 1 to a target, so terminate.
                    coverage_bound = covered_payoff[t]
                    break
            # get booleans for the two termination conditions
            added_coverage_sum = added_coverage.sum()
            cov_depleted_terminate = added_coverage_sum >= left
            cov_bound_terminate = coverage_bound > float('-inf')
            # if any of the two termination conditions occured, then terminate.
            if cov_depleted_terminate or cov_bound_terminate:
                break
            # no termination yet: update coverage, left and next_target.
            coverage += added_coverage
            left -= added_coverage.sum()
            next_target += 1

        # either one of the two termination conditions occured, or
        # every target has been added to the attackset.
        # attempt to add the remaining coverage using the ratios of each
        # target in a way that doesn't change the attackset.
        ratio_sum = ratio.sum()
        for t in range(next_target):
            coverage[t] += (ratio[t]*left) / ratio_sum
            if coverage[t] >= 1:
                coverage_bound = max(coverage_bound, covered_payoff[t])
        # if a target was assigned coverage > 1, we allocate coverage
        # 1 to this target, and allocate to every target in the attackset
        # enough coverage to yield the same payoff as this target.
        if coverage_bound > float('-inf'):
            for t in range(next_target):
                coverage[t] = (coverage_bound - uncovered_payoff[t]) \
                    / (covered_payoff[t] - uncovered_payoff[t])

        # save the coverage vector for the original target indices.
        self.coverage[sorted_targets] = coverage

        # save solution time without the overhead
        self.solution_time = time.time() - start_time

        # Obtain the attack_set of this coverage
        # if while loop was terminated by one of the two term. conditions
        #, we add the next_target to attack_set only if its uncovered_pay
        # equals the expected payoff given the final coverage for any
        # target in the attackset e.g. target 0.
        if next_target < self.num_targets:
            if uncovered_payoff[next_target] < coverage[0]*covered_payoff[0] + \
                    (1-coverage[0]) * uncovered_payoff[0]:
                # do not include next_target
                last_index = next_target
            else:
                # do include next_target
                last_index = next_target + 1
        else:
            # None of the two while-loop term. condition ever occured,
            # so add all targets to the attackset.
            last_index = self.num_targets + 1

        # now define the attack_set
        self.attack_set = sorted_targets[:last_index]


        # compute defender payoffs
        payoffs = np.zeros((self.attack_set.size,1))
        for i in range(self.attack_set.size):
            payoffs[i] = \
                self.defender_covered[self.attack_set[i]] *  \
                self.coverage[self.attack_set[i]] + \
                (1-self.coverage[self.attack_set[i]]) * \
                self.defender_uncovered[self.attack_set[i]]
        # the expected defender payoff is the max payoff
        self.opt_defender_payoff = payoffs.max()
        # the attacked target is the target that yield highest defender payoff
        self.opt_attacked_target = self.attack_set[np.argmax(payoffs)]
        # save opt coverage
        self.opt_coverage = list(self.coverage)

        # save opt attack set
        self.opt_attack_set = sorted(self.attack_set)

        # save solution time with overhead
        self.solution_time_with_overhead = time.time() - start_time


# x = SecurityGame(10, 4)
# y = Origami(x)
# y.solve()
# y.get_expected_payoff()
# # print(" coverage")
# # print(y.coverage)
# # print("coverage sum")
# # print(y.coverage.sum())
# # print("attacked target")
# # print(y.attacked_target)
# # print('payoff')
# print(y.opt_value)
