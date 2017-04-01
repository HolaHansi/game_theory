import numpy as np
from games import PatrolGame, NormalFormGame
from multipleLP import Multiple_SingleLP, MultipleLP
from dobbs import Dobbs
import multiprocessing as mp


class Run_time_experiments:
    cutoff_time = 60 * 15
    MAX_NUM_TYPES = 7
    PATROL_SIZE = 2
    NUM_REPETITIONS = 5

    run_times = np.zeros((3, MAX_NUM_TYPES, NUM_REPETITIONS))
    run_times_overheads = np.zeros((3, MAX_NUM_TYPES, NUM_REPETITIONS))

    temp_solve_time = 0
    temp_solve_time_overhead = 0

    def solve(self, solver, sol_time, sol_time_oh):
        """
        The solver process
        """
        solver.solve()
        sol_time.value = solver.solution_time
        sol_time_oh.value = solver.solution_time_with_overhead

    def handler(self, solver):
        """
        Handles the solver process.
        Will terminate solver process if it doesn't terminate
        before the cutoff time has elapsed
        """
        sol_time = mp.Value('d', -1.0)
        sol_time_oh = mp.Value('d', -1.0)
        p = mp.Process(target=self.solve, args=(solver,
                                                    sol_time,
                                                    sol_time_oh))
        p.start()
        p.join(self.cutoff_time)
        if p.is_alive():
            # terminate the process
            print("cut off!")
            p.terminate()
            p.join()
            return None
        else:
            return (sol_time.value, sol_time_oh.value)

    def get_run_times(self, solver, sol_num, num_types, run_num):
        times = self.handler(solver)
        if times:
            # if experiment terminated in time, record solutions times
            self.run_times[sol_num, num_types-1, run_num] = times[0]
            self.run_times_overheads[sol_num, num_types-1, run_num] = times[1]
        else:
            # experiment didn't finish before cutoff time
            self.run_times[sol_num, num_types-1, run_num] = -1
            self.run_times_overheads[sol_num, num_types-1, run_num] = -1

    def run_experiment(self, num_houses):
        self.run_times = np.zeros((3, self.MAX_NUM_TYPES, self.NUM_REPETITIONS))
        self.run_times_overheads = np.zeros((3, self.MAX_NUM_TYPES, self.NUM_REPETITIONS))
        for run_num in range(self.NUM_REPETITIONS):
            print("run: {}".format(run_num))
            for num_types in range(1, self.MAX_NUM_TYPES+1):
                print("num types: {}".format(num_types))
                # create numpy array to hold runtimes
                # create game
                game = PatrolGame(num_houses, self.PATROL_SIZE, num_types)
                game_harsanyi = NormalFormGame(game=game, harsanyi=True)

                # init solvers
                dob = Dobbs(game)
                bMlp = Multiple_SingleLP(game)
                mlp = MultipleLP(game_harsanyi)

                # get run times
                print("dob")
                self.get_run_times(dob, 0, num_types, run_num)
                print("bmlp")
                self.get_run_times(bMlp, 1, num_types, run_num)
                print("mlp")
                self.get_run_times(mlp, 2, num_types, run_num)
                print(self.run_times[:,:,run_num])

        self.average_run_times = np.average(self.run_times, axis=2)
        self.average_run_times_overhead = np.average(self.run_times_overheads, axis=2)

        print("==== average ===== ")
        print(self.average_run_times)
        # save to text files
        np.savetxt("solution_times_houses_{}".format(num_houses),
                self.average_run_times,
                   fmt='%1.4f')
        np.savetxt("solution_times_overhead_houses_{}".format(num_houses),
                self.average_run_times_overhead,
                   fmt='%1.4f')

# 2 houses
c = Run_time_experiments()
c.run_experiment(2)

# 3 houses
c = Run_time_experiments()
c.run_experiment(3)

# 4 houses
c = Run_time_experiments()
c.run_experiment(4)

