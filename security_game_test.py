from games import SecurityGame, NormalFormGame
from dobbs import Dobbs
from multipleLP import MultipleLP
from origami import Origami
from eraser import Eraser
from origami_milp import OrigamiMILP

import time

start = time.time()
comp_game = SecurityGame(12, 10, 2)
# norm_game = NormalFormGame(game=comp_game, harsanyi=False)
# hars_game = NormalFormGame(game=norm_game)


ori = Origami(comp_game)
ori.solve()
# dob = Dobbs(norm_game)
# dob.solve()
# mlp = MultipleLP(hars_game)
# mlp.solve()
# ers = Eraser(comp_game)
# ers.solve()

oriMILP = OrigamiMILP(comp_game)
oriMILP.solve()
# print("dob attacked target: {}".format(dob.opt_attacked_targets))
# print("mlp attacked target: {}".format(mlp.opt_attacker_pure_strategy))

# print("MLP opt-value: {}".format(mlp.opt_defender_payoff))
# print("DOB opt-value: {}".format(dob.opt_defender_payoff))
print("ORI opt_defender_payoff: {}".format(ori.opt_defender_payoff))
# print("ERS opt_defender_payoff: {}".format(ers.opt_defender_payoff))
print("ORI_MILP opt_defender_payoff: {}".format(oriMILP.opt_defender_payoff))

print("ORI cov: {}".format(ori.opt_coverage))
# print("ERS cov: {}".format(ers.opt_coverage))
print("ORI_MILP cov: {}".format(oriMILP.opt_coverage))

# # print("ORI at: {}".format(ori.opt_attacked_target))
# # print("ERS at: {}".format(ers.opt_attacked_target))
# # print("ORI_MILP at: {}".format(oriMILP.opt_attacked_target))

# # print("ori attackset: {}".format(ori.opt_attack_set))
# # print("orimilp attackset: {}".format(oriMILP.opt_attack_set))

# print("MLP solution time: {}".format(mlp.solution_time))
# print("DOB solution time: {}".format(dob.solution_time))
# print("origami solution time: {}".format(ori.solution_time))
# print("origami milp solution time: {}".format(oriMILP.solution_time))
# print("eraser solution time: {}".format(ers.solution_time))
# print("======")
# print("origami oh solution time: {}".format(ori.solution_time_with_overhead))
# print("origami milp solution time: {}".format(oriMILP.solution_time_with_overhead))
# print("eraser solution time: {}".format(ers.solution_time_with_overhead))

# end = time.time()
# print(end-start)
