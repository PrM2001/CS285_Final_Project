import numpy as np

from rhddp.problem import Problem
from rhddp.cost import SymbolicCost
from rhddp.dynamics import SymbolicDynamics
from rhddp.rhddp import RHDDP
from rhddp.vis import plot_compass_gait_traj
from scipy import io as scio
import matplotlib.pyplot as plt

import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, '../../data/compass_gait_dt0_003_start_reset.mat')


warm_start_data = scio.loadmat(file_path)

warm_start_u = warm_start_data.get('uRec').transpose()
warm_start_x = warm_start_data.get('xRec').transpose()
print(warm_start_x.shape)

dynamics = SymbolicDynamics(name="compass_gait_dynamics",
                                              num_states=8,
                                              num_inputs=1,
                                              num_disturbances=1)

cost = SymbolicCost(name="compass_gait_cost",
                    num_states=8,
                    num_inputs=1)

d_set = [-0.1, 0.15]
settings = {"initial_state": np.array([0.12938, 0.991595, -0.129744, 0.260098, 0.90201476, -0.117692, -0.90966, -0.8491542]),
            "horizon": 249,
            "warm_start_u": warm_start_u,
            "d_nom": 0.0,
            "d_set": d_set,
            "reset_prop": 0.004
            }

hyperparams = {"max_iters": 3,
               "alpha": 1,
               "alpha_update_period": 1,
               "alpha_increment": 100,
               "conv_criterion": np.inf,
               "regularization": 0.075,
               "regularization_update_period": 2,
               "regularization_decrease_factor": 0.95} #Use the defaults

problem = Problem(name="compass_gait",
                  dyn=dynamics,
                  cost=cost,
                  settings=settings,
                  hyperparams=hyperparams)

rhddp_controller = RHDDP(problem, verbose=False)

plt.figure()
solution = rhddp_controller.solve()
ddp_traj = solution.get("x_traj")
plot_compass_gait_traj(solution.get("x_traj"), solution.get("u_traj"), cost)
plt.savefig('plots/Compass_Gait_vis.png')
plt.close()


plt.figure()
plt.plot(ddp_traj[2, :], ddp_traj[3, :], label='result')
plt.plot(warm_start_x[2, :], warm_start_x[3, :], label='warmstart')
plt.legend()
plt.xlabel("theta1")
plt.ylabel("theta2")
plt.show()
plt.savefig('plots/Theta_Phase_Plot.png')
plt.close()

# plt.plot(np.arange(251), 2 * ddp_traj[6, :] + ddp_traj[7, :])
# plt.show()

print(f'Problem solved in {solution.get("solve_time")} seconds!')

rhddp_controller._prob.d_set = [np.array([d]) if np.isscalar(d) else d for d in d_set]
robusts_dists, _, robust_costs = rhddp_controller.evalDiscrete(x_traj = solution.get("x_traj"),
                                                                u_traj= solution.get("u_traj"),
                                                                K_traj = solution.get("K_traj"),
                                                                num_test_points = 20)
print(robust_costs)

settings["d_set"] = [0]

vanilla_problem = Problem(name="compass_gait_vanilla",
                                    dyn=dynamics,
                                    cost=cost,
                                    settings=settings,
                                    hyperparams=hyperparams)

vanilla_controller = RHDDP(vanilla_problem, verbose=False)

vanilla_solution = vanilla_controller.solve()

print(f'Vanilla Problem solved in {vanilla_solution.get("solve_time")} seconds!')

rhddp_controller._prob.d_set = [np.array([d]) if np.isscalar(d) else d for d in d_set]
vanilla_dists, _, vanilla_costs = rhddp_controller.evalDiscrete(x_traj = vanilla_solution.get("x_traj"),
                                                                u_traj= vanilla_solution.get("u_traj"),
                                                                K_traj = vanilla_solution.get("K_traj"),
                                                                num_test_points = 20)
print(vanilla_costs)




plt.figure()
plt.plot(robusts_dists, robust_costs, label="RHDDP")
plt.plot(vanilla_dists, vanilla_costs, label="Vanilla")
plt.xlabel("Disturbance Value")
plt.ylabel("Cost")
plt.legend()
plt.show()
plt.savefig('plots/CostVDisturbance.png')
plt.close()