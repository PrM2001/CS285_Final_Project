import numpy as np

from rhddp.problem import Problem
from rhddp.cost import SymbolicCost
from rhddp.dynamics import SymbolicDynamics
from rhddp.rhddp import RHDDP
from rhddp.vis import plot_compass_gait_traj
from scipy import io as scio
import matplotlib.pyplot as plt
import scienceplots

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
# from matplotlib import rc
# from matplotlib.ticker import MultipleLocator
plt.style.use(['science','ieee']) #comment out this line if you do not have LaTeX installed!

import os
import sys
sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

#region hide

script_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_directory, '../../data/compass_gait_dt0_003_start_reset.mat')
warm_start_data = scio.loadmat(file_path)

warm_start_u = warm_start_data.get('uRec').transpose()
warm_start_x = warm_start_data.get('xRec').transpose()
print(warm_start_x.shape)

dynamics = SymbolicDynamics(name="compass_gait_dynamics_2d_disturbance",
                                              num_states=8,
                                              num_inputs=1,
                                              num_disturbances=2)

cost = SymbolicCost(name="compass_gait_cost",
                    num_states=8,
                    num_inputs=1)


d_nom = np.array([0.0, 0.0])
#d_set = [np.array([-0.1, -0.1]), np.array([-0.1, 0.15]), np.array([0.15, -0.1]), np.array([0.15, 0.15])]
k=1
d_set = [k * np.array([-0.2, -0.2]), 
         k * np.array([-0.2, 0.2]), 
         k * np.array([0.2, -0.2]), 
         k * np.array([0.2, 0.2])]
robust_settings = {"initial_state": np.array([0.12938, 0.991595, -0.129744, 0.260098, 0.90201476, -0.117692, -0.90966, -0.8491542]),
            "horizon": 249,
            "warm_start_u": None, # warm_start_u
            "d_nom": d_nom,
            "d_set": d_set,
            "reset_prop": 0.004
            }

robust_hyperparams = {"max_iters": 10,
               "alpha": 1,
               "alpha_update_period": 1,
               "alpha_increment": 4,
               "conv_criterion": 12,
               "regularization": 0.00,
               "regularization_update_period": 1,
               "regularization_decrease_factor": 1,
               "wolfe_b": 0.001} 



robust_problem = Problem(name="compass_gait",
                  dyn=dynamics,
                  cost=cost,
                  settings=robust_settings,
                  hyperparams=robust_hyperparams)

robust_controller = RHDDP(robust_problem, verbose=False)

robust_solution = robust_controller.solve()

# plt.figure()
# ddp_traj = solution.get("x_traj")
# plot_compass_gait_traj(solution.get("x_traj"), solution.get("u_traj"), cost)
# plt.savefig('plots/Compass_Gait_vis.png')
# plt.close()


# plt.figure()
# plt.plot(ddp_traj[2, :], ddp_traj[3, :], label='result')
# plt.plot(warm_start_x[2, :], warm_start_x[3, :], label='warmstart')
# plt.legend()
# plt.xlabel("theta1")
# plt.ylabel("theta2")
# plt.show()
# plt.savefig('plots/Theta_Phase_Plot.png')
# plt.close()

# plt.plot(np.arange(251), 2 * ddp_traj[6, :] + ddp_traj[7, :])
# plt.show()

print(f'Problem solved in {robust_solution.get("solve_time")} seconds!')
num_test_points = 25


#robust_controller._prob.d_set = [np.array([d]) if np.isscalar(d) else d for d in d_set]
robusts_dists, _, robust_costs, D0, D1 = robust_controller.evalDiscrete2d(x_traj = robust_solution.get("x_traj"),
                                                                u_traj= robust_solution.get("u_traj"),
                                                                K_traj = robust_solution.get("K_traj"),
                                                                num_test_points = num_test_points)
#print(robust_costs)

vanilla_settings = robust_settings
vanilla_settings["d_set"] = None
vanilla_hyperparams = robust_hyperparams
vanilla_hyperparams["max_iters"] = 20


vanilla_problem = Problem(name="compass_gait_vanilla",
                                    dyn=dynamics,
                                    cost=cost,
                                    settings=vanilla_settings,
                                    hyperparams=vanilla_hyperparams)

vanilla_controller = RHDDP(vanilla_problem, verbose=False)

vanilla_solution = vanilla_controller.solve()

print(f'Vanilla Problem solved in {vanilla_solution.get("solve_time")} seconds!')

robust_controller._prob.d_set = [np.array([d]) if np.isscalar(d) else d for d in d_set]
vanilla_dists, _, vanilla_costs, _, _ = robust_controller.evalDiscrete2d(x_traj = vanilla_solution.get("x_traj"),
                                                                u_traj= vanilla_solution.get("u_traj"),
                                                                K_traj = vanilla_solution.get("K_traj"),
                                                                num_test_points = num_test_points)
# print(vanilla_costs)




# plt.figure()
# plt.plot(robusts_dists, robust_costs, label="RHDDP")
# plt.plot(vanilla_dists, vanilla_costs, label="Vanilla")
# plt.xlabel("Disturbance Value")
# plt.ylabel("Cost")
# plt.legend()
# plt.show()
# plt.savefig('plots/CostVDisturbance.png')
# plt.close()
#endregion hide


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

#region settings
# # try to use https://dawes.wordpress.com/2014/06/27/publication-ready-3d-figures-from-matplotlib/
# rc('font',size=28)
# rc('font',family='serif')
# rc('axes',labelsize=32)

# [t.set_va('center') for t in ax.get_yticklabels()]
# [t.set_ha('left') for t in ax.get_yticklabels()]
# [t.set_va('center') for t in ax.get_xticklabels()]
# [t.set_ha('right') for t in ax.get_xticklabels()]
# [t.set_va('center') for t in ax.get_zticklabels()]
# [t.set_ha('left') for t in ax.get_zticklabels()]

# ax.grid(False)
# ax.xaxis.pane.set_edgecolor('black')
# ax.yaxis.pane.set_edgecolor('black')
# ax.xaxis.pane.fill = False
# ax.yaxis.pane.fill = False
# ax.zaxis.pane.fill = False

# ax.xaxis._axinfo['tick']['inward_factor'] = 0
# ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
# ax.yaxis._axinfo['tick']['inward_factor'] = 0
# ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
# ax.zaxis._axinfo['tick']['inward_factor'] = 0
# ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
# ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

# ax.xaxis.set_major_locator(MultipleLocator(5))
# ax.yaxis.set_major_locator(MultipleLocator(5))
# ax.zaxis.set_major_locator(MultipleLocator(0.01))
#endregion settings


surf2 = ax.plot_surface(D0, D1, vanilla_costs, color='blue', label="Vanilla", alpha=0.6)
surf1 = ax.plot_surface(D0, D1, robust_costs, color='black', label="Robust", alpha=0.8)
#ax.grid(False)
#ax.set_facecolor('white')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

surf1._edgecolors2d = surf1._edgecolor3d
surf1._facecolors2d = surf1._facecolor3d

surf2._edgecolors2d = surf2._edgecolor3d
surf2._facecolors2d = surf2._facecolor3d

# ax.set_xlabel("d0")
# ax.set_ylabel("d1")
# ax.set_zlabel("Cost")

# ax.legend()
elev = 25
azim = 325
roll=0
ax.view_init(elev=elev, azim=azim, roll=roll)

plt.show()
plt.savefig(f'plots/compass_gait/2d_compass_gait_surface_e{elev}_r{roll}_a{azim}.png')

print(f"Max Vanilla Cost: {np.max(vanilla_costs)}")
print(f"Max Robust Cost: {np.max(robust_costs)}")

temp = np.divide(robust_costs, vanilla_costs)
argmax_indices = np.unravel_index(np.argmax(temp), temp.shape)

print("Indices of the maximum value: ", argmax_indices)
print("Disturbance of the maximum value: ", D1[argmax_indices], D0[argmax_indices])
print("Worst Ratio: ", temp[argmax_indices])



#region worst case plotting
# plt.figure()
worst_robust_x_traj, worst_robust_u_traj, worst_robust_cost = robust_controller.get_solved_trajectory(np.array([0.15, 0.15]))
# plot_compass_gait_traj(worst_robust_x_traj, worst_robust_u_traj, worst_robust_cost)
# plt.savefig('plots/Robust_Worst_Traj')

# plt.figure()
worst_vanilla_x_traj, worst_vanilla_u_traj, worst_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([0.15, 0.15]))
# plot_compass_gait_traj(worst_vanilla_x_traj, worst_vanilla_u_traj, worst_vanilla_cost)
# plt.savefig('plots/Vanilla_Worst_Traj')


# plt.figure()
# plt.plot(np.arange(worst_robust_u_traj.shape[1]),  worst_robust_u_traj[0, :], label="RH-DDP")
# plt.plot(np.arange(worst_vanilla_u_traj.shape[1]),  worst_vanilla_u_traj[0, :], label="HS-DDP")
# plt.legend(loc='upper right', fontsize=6)
# plt.show()
# #plt.title("Worst Case Compass Gait Controls vs. Timestep")
# plt.savefig('plots/compass_gait/wc/wc_control_plots.png')

plt.figure()
plt.plot(worst_robust_x_traj[2, :],  worst_robust_x_traj[3, :], label="RH-DDP")
plt.plot(worst_vanilla_x_traj[2, :],  worst_vanilla_x_traj[3, :], color='blue', label="HS-DDP")
plt.legend(loc='upper right', fontsize=6)
plt.show()
plt.xlabel('$\\theta_{1}$ (rad)')
plt.ylabel('$\\theta_{2}$ (rad)')
#plt.title("Worst Case Angle Phase Plot")
plt.savefig('plots/compass_gait/wc/wc_angle_phase_plots.png')

# plt.figure()
# plt.plot(worst_robust_x_traj[6, :],  worst_robust_x_traj[7, :], label="RH-DDP")
# plt.plot(worst_vanilla_x_traj[6, :],  worst_vanilla_x_traj[7, :], label="HS-DDP")
# plt.legend(loc='upper right', fontsize=6)
# plt.show()
# #plt.title("Worst Case Angular Velocity Phase Plot")
# plt.savefig('plots/compass_gait/wc/wc_angular_velocity_phase_plots.png')

plt.figure()
plt.plot(np.arange(worst_robust_x_traj.shape[1])*0.01,  worst_robust_x_traj[4, :], label="RH-DDP")
plt.plot(np.arange(worst_vanilla_x_traj.shape[1])*0.01,  worst_vanilla_x_traj[4, :], color='blue', label="HS-DDP")
plt.axhline(y = 0.0, color = 'g', linestyle = 'dashdot', label='Zero Velocity') 
plt.legend(loc='upper right', fontsize=6)
plt.xlabel('Time (s)')
plt.ylabel('Torso $x$ Velocity ($\\frac{m}{s}$)')
plt.show()
#plt.title("Worst Case Torso Velocity vs Time")
plt.savefig('plots/compass_gait/wc/wc_torso_velocity.png')
#endregion worst case plotting

#region nominal plotting
# plt.figure()
nom_robust_x_traj, nominal_robust_u_traj, nominal_robust_cost = robust_controller.get_solved_trajectory(d_nom)
# plot_compass_gait_traj(nom_robust_x_traj, nominal_robust_u_traj, nominal_robust_cost)
# plt.savefig('plots/Nominal_Worst_Traj')

# plt.figure()
nominal_vanilla_x_traj, nominal_vanilla_u_traj, nominal_vanilla_cost = vanilla_controller.get_solved_trajectory(d_nom)
# plot_compass_gait_traj(nominal_vanilla_x_traj, nominal_vanilla_u_traj, nominal_vanilla_cost)
# plt.savefig('plots/Nominal_Worst_Traj')


# plt.figure()
# plt.plot(np.arange(nominal_robust_u_traj.shape[1]),  nominal_robust_u_traj[0, :], label="RH-DDP")
# plt.plot(np.arange(nominal_vanilla_u_traj.shape[1]),  nominal_vanilla_u_traj[0, :], label="HS-DDP")
# plt.legend(loc='upper right', fontsize=6)
# plt.show()
# #plt.title("Nominal Compass Gait Controls vs. Timestep")
# plt.savefig('plots/compass_gait/nom/nom_control_plots.png')

plt.figure()
plt.plot(nom_robust_x_traj[2, :],  nom_robust_x_traj[3, :], label="RH-DDP")
plt.plot(nominal_vanilla_x_traj[2, :],  nominal_vanilla_x_traj[3, :], color='blue', label="HS-DDP")
plt.legend(loc='upper right', fontsize=6)
plt.xlabel('$\\theta_{1}$ (rad)')
plt.ylabel('$\\theta_{2}$  (rad)')
plt.show()
#plt.title("Nominal Angle Phase Plot")
plt.savefig('plots/compass_gait/nom/nom_angle_phase_plots.png')

# plt.figure()
# plt.plot(nom_robust_x_traj[6, :],  nom_robust_x_traj[7, :], label="RH-DDP")
# plt.plot(nominal_vanilla_x_traj[6, :],  nominal_vanilla_x_traj[7, :], label="HS-DDP")
# plt.legend(loc='upper right', fontsize=6)
# plt.show()
# #plt.title("Nominal Angular Velocity Phase Plot")
# plt.savefig('plots/compass_gait/nom/nom_angular_velocity_phase_plots.png')

plt.figure()
plt.plot(np.arange(nom_robust_x_traj.shape[1])*0.01,  nom_robust_x_traj[4, :], label="RH-DDP")
plt.plot(np.arange(nominal_vanilla_x_traj.shape[1])*0.01,  nominal_vanilla_x_traj[4, :], color='blue', label="HS-DDP")
plt.legend(loc='upper right', fontsize=6)
plt.xlabel('Time (s)')
plt.ylabel('Torso $x$ Velocity ($\\frac{m}{s}$)')
plt.show()
#plt.title("Nominal Torso Velocity vs Time")
plt.savefig('plots/compass_gait/nom/nom_torso_velocity.png')
#endregion nominal plotting

print(f"Nominal Vanilla Cost: {nominal_vanilla_cost}")
print(f"Nominal Robust Cost: {nominal_robust_cost}")

pass