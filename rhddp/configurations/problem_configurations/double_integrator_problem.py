import numpy as np

from rhddp.problem import Problem
from rhddp.cost import SymbolicCost
from rhddp.dynamics import SymbolicDynamics
from rhddp.rhddp import RHDDP

import os
import sys

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee']) #comment out this line if you do not have LaTeX installed!

sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

dynamics = SymbolicDynamics(name="double_integrator_dynamics",
                                              num_states=2,
                                              num_inputs=1,
                                              num_disturbances=1)

cost = SymbolicCost(name="double_integrator_cost",
                                      num_states=2,
                                      num_inputs=1)

d_nom = 0.0
robust_settings = {"initial_state": np.array([-5, 0]),
            "horizon": 500,
            "d_nom": d_nom,
            "d_set": [-0.5, 0.5],
            "reset_prop": 0.8,
            "conv_criterion": 12
            }

robust_hyperparams = {"max_iters": 20, 
                      "alpha": 1,
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


# num_test_points = 500


# robust_dists, _, robust_costs= robust_controller.evalDiscreteSolved(num_test_points=num_test_points)


# #Vanilla

# vanilla_settings = robust_settings
# vanilla_settings["d_set"] = None #same settings, but vanilla version

# vanilla_hyperparams = robust_hyperparams #same hyperparameters

# vanilla_problem = Problem(name="double_integrator",
#                                     dyn=dynamics,
#                                     cost=cost,
#                                     settings=vanilla_settings,
#                                     hyperparams=vanilla_hyperparams)

# vanilla_controller = RHDDP(vanilla_problem, verbose=False)

# vanilla_solution = vanilla_controller.solve()


# vanilla_dists, _, vanilla_costs = robust_controller.evalDiscrete(x_traj = vanilla_solution.get("x_traj"),
#                                                                 u_traj = vanilla_solution.get("u_traj"),
#                                                                 K_traj = vanilla_solution.get("K_traj"),
#                                                                 num_test_points = num_test_points)

# # print(robust_costs)
# # print(vanilla_costs)
# d_worst = 0.5
# plt.figure()
# #plt.ylim(150,460)
# plt.plot(robust_dists, robust_costs, label="RH-DDP")
# plt.plot(vanilla_dists, vanilla_costs, color='blue', label="HS-DDP")
# # plt.axvline(x=d_nom,  linestyle='dotted', color='green', label='Nominal\nDisturbance')
# # plt.axvline(x=d_worst,  linestyle='dashdot', color='green', label='Worst\nDisturbance')
# plt.xlabel("Disturbance Value ($\\frac{m}{s}$)")
# plt.ylabel("Cost")
# plt.legend()
# plt.legend(loc='upper left', fontsize=6)
# plt.show()
# #plt.title("Simulated Cost vs Disturbance for Double Integrator")
# plt.savefig('plots/double_integrator/double_integrator_robust_vs_vanilla.png')

# #region worst case plotting

# worst_robust_x_traj, worst_robust_u_traj, worst_robust_cost = robust_controller.get_solved_trajectory(np.array([d_worst]))
# worst_vanilla_x_traj, worst_vanilla_u_traj, worst_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([d_worst]))

# plt.figure()
# plt.plot(np.arange(worst_robust_x_traj.shape[1])*0.01,  worst_robust_x_traj[0, :], label="RH-DDP")
# plt.plot(np.arange(worst_vanilla_x_traj.shape[1])*0.01,  worst_vanilla_x_traj[0, :], color='blue', label="HS-DDP")
# plt.xlabel("Time")
# plt.ylabel("Position")
# plt.legend(loc='lower right', fontsize=6)

# plt.show()
# #plt.title("Worst Case Double integrator Positions vs. Timestep")
# plt.savefig('plots/double_integrator/wc/wc_double_integrator_position_plots.png')

# plt.figure()
# plt.plot(np.arange(worst_robust_x_traj.shape[1])*0.01,  worst_robust_x_traj[1, :], label="RH-DDP")
# plt.plot(np.arange(worst_vanilla_x_traj.shape[1])*0.01,  worst_vanilla_x_traj[1, :], color='blue', label="HS-DDP")
# plt.xlabel("Time")
# plt.ylabel("Velocity")
# plt.legend(loc='lower left', fontsize=6)
# plt.show()
# #plt.title("Worst Case Double integrator Velocities vs. Timestep")
# plt.savefig('plots/double_integrator/wc/wc_double_integrator_velocity_plots.png')

# plt.figure()
# plt.plot(worst_robust_x_traj[0, :],  worst_robust_x_traj[1, :], label="RH-DDP",  zorder=1)
# plt.plot(worst_vanilla_x_traj[0, :],  worst_vanilla_x_traj[1, :], color='blue', label="HS-DDP", zorder=2)
# plt.scatter(5,0, marker="x", color="green", label="Goal State", s=5, zorder=3)
# plt.scatter(0,0, marker="o", color="red", label="Initial State", s=4, zorder=4)
# plt.xlabel("Position")
# plt.ylabel("Velocity")
# plt.legend(loc='upper left', fontsize=6)
# plt.ylim(-0.1, 1.4)
# plt.show()
# #plt.title("Worst Case Double integrator Position vs. Velocity Phase Plot")
# plt.savefig('plots/double_integrator/wc/wc_double_integrator_phase_plots.png')

# plt.figure()
# plt.plot(np.arange(worst_robust_u_traj.shape[1])*0.01,  worst_robust_u_traj[0, :], label="RH-DDP")
# plt.plot(np.arange(worst_vanilla_u_traj.shape[1])*0.01,  worst_vanilla_u_traj[0, :], color='blue', label="HS-DDP")
# plt.xlabel("Time (s)")
# plt.ylabel("Control Force (N)")
# plt.legend(loc='upper right', fontsize=6)
# plt.show()
# #plt.title("Worst Case Double integrator Controls vs. Timestep")
# plt.savefig('plots/double_integrator/wc/wc_double_integrator_control_plots.png')
# #endregion worst case plotting

# #region nominal plotting
# nominal_robust_x_traj, nominal_robust_u_traj, nominal_robust_cost = robust_controller.get_solved_trajectory(np.array([d_nom]))
# nominal_vanilla_x_traj, nominal_vanilla_u_traj, nominal_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([d_nom]))

# plt.figure()
# plt.plot(np.arange(nominal_robust_x_traj.shape[1])*0.01,  nominal_robust_x_traj[0, :], label="RH-DDP")
# plt.plot(np.arange(nominal_vanilla_x_traj.shape[1])*0.01,  nominal_vanilla_x_traj[0, :], color='blue', label="HS-DDP")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.legend(loc='lower right', fontsize=6)
# plt.show()
# #plt.title("Nominal Double integrator Positions vs. Timestep")
# plt.savefig('plots/double_integrator/nom/nom_double_integrator_position_plots.png')

# plt.figure()
# plt.plot(np.arange(nominal_robust_x_traj.shape[1])*0.01,  nominal_robust_x_traj[1, :], label="RH-DDP")
# plt.plot(np.arange(nominal_vanilla_x_traj.shape[1])*0.01,  nominal_vanilla_x_traj[1, :], color='blue', label="HS-DDP")
# plt.xlabel("Time")
# plt.ylabel("Velocity")
# plt.legend(loc='lower left', fontsize=6)
# plt.show()
# #plt.title("Nominal Double integrator Velocities vs. Timestep")
# plt.savefig('plots/double_integrator/nom/nom_double_integrator_velocity_plots.png')

# plt.figure()
# plt.plot(nominal_robust_x_traj[0, :],  nominal_robust_x_traj[1, :], label="RH-DDP",  zorder=1)
# plt.plot(nominal_vanilla_x_traj[0, :],  nominal_vanilla_x_traj[1, :], color='blue', label="HS-DDP", zorder=2)
# plt.scatter(5,0, marker="x", color="green", label="Goal State", s=5, zorder=3)
# plt.scatter(0,0, marker="o", color="red", label="Initial State", s=4, zorder=4)
# plt.xlabel("Position")
# plt.ylabel("Velocity")
# plt.legend(loc='upper left', fontsize=6)
# plt.ylim(-0.1, 1.4)
# plt.show()
# #plt.title("Nominal Double integrator Position vs. Velocity Phase Plot")
# plt.savefig('plots/double_integrator/nom/nom_double_integrator_phase_plots.png')

# plt.figure()
# plt.plot(np.arange(nominal_robust_u_traj.shape[1])*0.01,  nominal_robust_u_traj[0, :], label="RH-DDP")
# plt.plot(np.arange(nominal_vanilla_u_traj.shape[1])*0.01,  nominal_vanilla_u_traj[0, :], color='blue', label="HS-DDP")
# plt.xlabel("Time (s)")
# plt.ylabel("Control Force (N)")
# plt.legend(loc='upper right', fontsize=6)
# plt.show()
# #plt.title("Nominal Double integrator Controls vs. Timestep")
# plt.savefig('plots/double_integrator/nom/nom_double_integrator_control_plots.png')


# #endregion nominal plotting



# print(f"wc robust control: {np.max(worst_robust_u_traj[0, :])}")
# print(f"wc vanilla control: {np.max(worst_vanilla_u_traj[0, :])}")

# print(f"nom robust control: {np.max(nominal_robust_u_traj[0, :])}")
# print(f"nom vanilla control: {np.max(nominal_vanilla_u_traj[0, :])}")

# print(f"wc robust cost: {worst_robust_cost}")
# print(f"wc vanilla cost: {worst_vanilla_cost}")

# print(f"nom robust cost: {nominal_robust_cost}")
# print(f"nom vanilla cost: {nominal_vanilla_cost}")

# pass