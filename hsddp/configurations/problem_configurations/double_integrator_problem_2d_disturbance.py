import numpy as np

from rhddp.problem import Problem
from rhddp.cost import SymbolicCost
from rhddp.dynamics import SymbolicDynamics
from rhddp.rhddp import RHDDP

import os
import sys

import matplotlib.pyplot as plt

sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

dynamics = SymbolicDynamics(name="double_integrator_dynamics_2d_disturbance",
                                              num_states=2,
                                              num_inputs=1,
                                              num_disturbances=2)

cost = SymbolicCost(name="double_integrator_cost",
                                      num_states=2,
                                      num_inputs=1)

# settings = {"initial_state": np.array([0, 0]),
#             "horizon": 1000,
#             "d_nom": 0.5,
#             "d_set": [-0.5, 0.5],
#             "reset_prop": 0.8
#             }

robust_settings = {"initial_state": np.array([0, 0]),
            "horizon": 1000,
            "d_nom": np.array([0, 0.5]),
            "d_set": [np.array([-0.5, -0.5]), np.array([-0.5, 0.5]), np.array([0.5, -0.5]), np.array([0.5, 0.5])],
            "reset_prop": 0.8
            }

robust_hyperparams = {"alpha": 1,
               "alpha_update_period": 3,
               "alpha_increment": 1.5,
               "conv_criterion": 25} #Use the defaults

robust_problem = Problem(name="double_integrator",
                                    dyn=dynamics,
                                    cost=cost,
                                    settings=robust_settings,
                                    hyperparams=robust_hyperparams)

robust_controller = RHDDP(robust_problem, verbose=False)

robust_solution = robust_controller.solve()

print(f'Robust Problem solved in {robust_solution.get("solve_time")} seconds!')

robust_dists, _, robust_costs, D0, D1 = robust_controller.evalDiscrete2d(x_traj = robust_solution.get("x_traj"),
                                                                u_traj= robust_solution.get("u_traj"),
                                                                K_traj = robust_solution.get("K_traj"),
                                                                num_test_points = 10)

#Vanilla

vanilla_settings = robust_settings
vanilla_settings["d_set"] = None #same settings, but vanilla version

vanilla_hyperparams = robust_hyperparams #same hyperparameters

vanilla_problem = Problem(name="double_integrator",
                                    dyn=dynamics,
                                    cost=cost,
                                    settings=vanilla_settings,
                                    hyperparams=vanilla_hyperparams)

vanilla_controller = RHDDP(vanilla_problem, verbose=False)

vanilla_solution = vanilla_controller.solve()


vanilla_dists, _, vanilla_costs, _, _ = robust_controller.evalDiscrete2d(x_traj = vanilla_solution.get("x_traj"),
                                                                u_traj= vanilla_solution.get("u_traj"),
                                                                K_traj = vanilla_solution.get("K_traj"),
                                                                num_test_points = 10)

# print(robust_costs)
# print(vanilla_costs)
# plt.plot(robust_dists, robust_costs, label="RHDDP")
# plt.plot(vanilla_dists, vanilla_costs, label="simple")

# plt.plot(np.arange(0, robust_dists.shape[0]), robust_costs, label="RHDDP")
# plt.plot(np.arange(0, robust_dists.shape[0]), vanilla_costs, label="simple")
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(D0, D1, robust_costs)
ax.plot_surface(D0, D1, vanilla_costs)
# plt.xlabel("Disturbance Value")
# plt.ylabel("Cost")
# plt.legend()
plt.show()
plt.savefig('plots/double_integrator_robust_vs_vanilla_surface.png')