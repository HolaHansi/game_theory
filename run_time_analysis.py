import numpy as np
from games import PatrolGame, NormalFormGame
from dobbs import Dobbs
from multipleLP import MultipleLP

# mlp can't feasibly handle more than 4 houses.
# m = 4, a=14
MAX_M = 3
D = 2
MAX_A = 5
print("MAX_M: {}".format(MAX_M))
print("MAX_A: {}".format(MAX_A))

# the i, j entry is solution time when m=i, a=j
# note we only need MAX_M minus 1 rows as m>=2 must hold
dob_solution_times = np.zeros((MAX_M-1, MAX_A))
mlp_solution_times = np.zeros((MAX_M-1, MAX_A))
for m in range(2,MAX_M+1):
    for a in range(1,MAX_A+1):
        # TODO: sol_time should be an average over 20
        dob_sols = []
        mlp_sols = []
        for i in range(2):
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

np.savetxt("dob_solution_times.txt", dob_solution_times)
np.savetxt("mlp_solution_times.txt", mlp_solution_times)

print(dob_solution_times)
print(mlp_solution_times)
