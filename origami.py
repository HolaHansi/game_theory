from games import SecurityGame
import numpy as np
class Origami:
    """
    Origami is the procedure for computing the maximal attack-set
    it's given a security game, and will output the optimal coverage
    """
    def __init__(self, game):
        self.attacker_uncovered = game.attackerPayOffs[0,:]
        self.attacker_covered = game.attackerPayOffs[1,:]
        self.defender_uncovered = game.defenderPayOffs[0,:]
        self.defender_covered = game.defenderPayOffs[1,:]

        # now init the coverage vector
        self.num_targets = game.num_targets
        self.coverage = np.zeros((self.num_targets,1))
        # save the max_coverage var
        self.max_coverage = game.max_coverage

    # TODO implement timer so we can time execution
    def solve(self):
        """
        This is the function that will actually run the ORIGAMI algorithm
        it will increase the attack-set with each round of the for-loop
        and terminate if one of two things happen:
            1) Run out of coverage to allocate
            2) a target is assigned coverage = 1
        This implementation follows roughly the pseudo-code by Kiekintveld.
        """

        # get the targets sorted in descending order by attacker_uncovered payoff
        sorted_targets = np.argsort(self.attacker_uncovered)[::-1]
        print("sorted targets\n")
        print(sorted_targets)
        pay_off = np.take(self.attacker_uncovered, sorted_targets)
        uncovered_payoff = np.take(self.attacker_uncovered, sorted_targets)
        covered_payoff = np.take(self.attacker_covered, sorted_targets)
        left = self.max_coverage
        next_target = 1
        coverage = np.zeros((self.num_targets, 1))
        added_coverage = np.zeros((self.num_targets,1))
        coverage_bound = float("-inf")
        ratio = np.zeros((self.num_targets, 1))

        while next_target < self.num_targets:
            for t in range(next_target):
                # add coverage to t such that payoff becomes the
                # same as pay_off[next_target]
                added_coverage[t] = ((pay_off[next_target] - uncovered_payoff[t]) \
                / (covered_payoff[t] - uncovered_payoff[t])) - coverage[t]
                if added_coverage[t] + coverage[t] >= 1:
                    # don't break as we might encounter a covBound later
                    coverage_bound = max(coverage_bound, covered_payoff[t])
            # my own optimization - compute ratio ad hoc
            ratio[next_target - 1] = 1 / (uncovered_payoff[t] - covered_payoff[t])
            # get sum of added_coverage
            add_cov_sum = added_coverage[0:next_target].sum()
            # we save reason for break in this bool vars
            cov_bound_break = coverage_bound > float("-inf")
            cov_sum_break = add_cov_sum >= left
            if cov_bound_break or cov_sum_break:
                break
            # no termination yet, so add coverage to coverage vector
            coverage += added_coverage
            left -= add_cov_sum
            # compute ratio
            next_target += 1

        # while-loop has terminated, check to see why
        # there are four cases to consider
        # CASE 1: We obtained af coverage_bound, but we still have
        # enough coverage
        if cov_bound_break and not cov_sum_break:
            for t in range(next_target):
                coverage[t] = (coverage_bound - uncovered_payoff[t]) / \
                              (covered_payoff[t] - uncovered_payoff[t])
        # CASE 2: We did not obtain a coverage_bound, but we ran
        # out of coverage to allocate.
        elif not cov_bound_break and cov_sum_break:
            for t in range(next_target):
                coverage[t] += ((ratio[t] * left) / ratio.sum())

        # CASE 3: We did obtain a coverage_bound, and we ran out of coverage
        # to allocate.
        elif cov_bound_break and cov_sum_break:
            old_coverage = coverage
            # CASE 3.1: we have enough coverage to assign coverage 1 to
            # the target yielding the coverage_bound, and sufficient coverage
            # to the remaining targets in the attackset.
            for t in range(next_target):
                coverage[t] = (coverage_bound - uncovered_payoff[t]) / \
                              (covered_payoff[t] - uncovered_payoff[t])
            # CASE 3.2: we don't have enough coverage to assign coverage 1
            # to the target yielding the coverage bound, so instead we
            # allocate the residual coverage according to the ratios.
            if coverage.sum() - old_coverage.sum() > left:
                for t in range(next_target):
                    coverage[t] = old_coverage[t] + ((ratio[t] * left) / ratio.sum())
        # now return the correct coverage vector i.e. the coverage in the
        # unsorted coverage vector.
        print("sorted coverage")
        print(coverage)
        self.coverage[sorted_targets] = coverage

    def get_expected_payoff(self):
        # by SSE assumption, we know which target is going to be attacked
        # we iterate over the members of the attack-set and select the target
        # yielding the highest defender payoff

        # first we obtain the attack-set
        self.attack_set = np.nonzero(self.coverage)[0]
        # TODO: take into account the pathological case where cov=0, for some
        # target is part of attackset
        pay_offs = np.zeros((len(self.attack_set),1))
        for i in range(len(self.attack_set)):
            pay_offs[i] = \
                self.defender_covered[self.attack_set[i]] *  \
                self.coverage[self.attack_set[i]] + \
                (1-self.coverage[self.attack_set[i]]) * \
                self.defender_uncovered[self.attack_set[i]]
        self.expected_payoff = pay_offs.max()
        self.attacked_target = self.attack_set[np.argmax(pay_offs)]


x = SecurityGame(10, 3)
y = Origami(x)
y.solve()
y.get_expected_payoff()
print("unsorted coverage")
print(y.coverage)
print(y.coverage.sum())
print(y.attacked_target)
print(y.expected_payoff)
