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
        # coverage_bound is updated only when a coverage > 1 is assigned
        coverage_bound = float('-inf')
        # the next target to be appended to the attack_set.
        next_target = 1
        while next_target < self.num_targets:
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
                    coverage_bound = max(coverage_bound, covered_payoff[t])
                    break

            # test if any of the two termination conditions are satisfied
            added_coverage_sum = added_coverage.sum()
            cov_depleted_terminate = added_coverage_sum > left
            cov_bound_terminate = coverage_bound > float('-inf')
            if cov_depleted_terminate or cov_bound_terminate:
                break

            # no termination yet: update coverage, left and next_target.
            coverage += added_coverage
            left -= added_coverage.sum()
            next_target += 1

        # save the attackset
        self.attack_set = sorted_targets[:next_target]

        ratio = np.array([1 / (uncovered_payoff[t] - covered_payoff[t]) for
                        t in range(next_target)])
        ratio_sum = ratio.sum()

        for t in range(next_target):
            coverage[t] += (ratio[t]*left) / float(ratio_sum)
            if coverage[t] >= 1:
                coverage_bound = max(coverage_bound, covered_payoff[t])

        # if a target was assigned coverage > 1, we allocate coverage
        # 1 to this target, and allocate to every target in the attackset
        # enough coverage to yield the same payoff as this target.
        if coverage_bound > float('-inf'):
            for t in range(next_target):
                coverage[t] = (coverage_bound - uncovered_payoff[t]) \
                    / (covered_payoff[t] - uncovered_payoff[t])

        # save the optimal coverage vector for the original target indices.
        self.opt_coverage = np.zeros((self.num_targets))
        self.opt_coverage[sorted_targets] = coverage

        # save solution time without the overhead
        self.solution_time = time.time() - start_time

        # compute defender payoffs
        payoffs = np.zeros((self.attack_set.size,1))
        for i in range(self.attack_set.size):
            payoffs[i] = \
                self.defender_covered[self.attack_set[i]] *  \
                self.opt_coverage[self.attack_set[i]] + \
                (1-self.opt_coverage[self.attack_set[i]]) * \
                self.defender_uncovered[self.attack_set[i]]

        # the expected defender payoff is the max payoff
        self.opt_defender_payoff = payoffs.max()

        # save the payoffs for branch-and-bound
        self.opt_defender_payoffs = payoffs

        # the attacked target is the target that yield highest defender payoff
        self.opt_attacked_target = self.attack_set[np.argmax(payoffs)]

        # save opt attack set
        self.opt_attack_set = sorted(self.attack_set)

        # save solution time with overhead
        self.solution_time_with_overhead = time.time() - start_time
