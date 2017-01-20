import numpy as np
from games import PatrolGame, NormalFormGame
from dobbs import Dobbs
from multipleLP import MultipleLP

# TODO fit these vars to paper values
MAX_M = 3
D = 2
MAX_A = 3
print("MAX_M: {}".format(MAX_M))
print("MAX_A: {}".format(MAX_A))


# the i, j entry is solution time when m=i, a=j
dob_solution_times = np.zeros((MAX_M-1, MAX_A))
mlp_solution_times = np.zeros((MAX_M-1, MAX_A))
for m in range(2,MAX_M+1):
    # TODO: increase to range(1, 15) run test for 1 to 14 adversary types
    for a in range(1,MAX_A+1):
        # sol_time should be an average over 20
        dob_sols = []
        mlp_sols = []
        for i in range(20):
            # randomly games
            b_game = PatrolGame(m, 2, a)
            n_game = NormalFormGame(b_game)
            dob = Dobbs(b_game)
            mlp = MultipleLP(n_game)
            dob.solve()
            mlp.solve()
            dob_sols.append(dob.solution_time)
            mlp_sols.append(mlp.solution_time)
        # add solution times to matrices
        print("THE INDICES: {}, {}".format(m-2, a-1))
        dob_solution_times[m-2,a-1] = sum(dob_sols) / float(20)
        mlp_solution_times[m-2,a-1] = sum(mlp_sols) / float(20)

# TODO produce graphs depicting sol times
# a graph for m=2,3,4,5
# x = num_adversaries, y=solution_time
# use a logscale for sol time - mlp is exponentially worse than dob!


print(dob_solution_times)
print(mlp_solution_times)
