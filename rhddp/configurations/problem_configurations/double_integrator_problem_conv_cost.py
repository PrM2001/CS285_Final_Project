import numpy as np

from rhddp.problem import Problem
from rhddp.cost import SymbolicCost, Cost
from rhddp.dynamics import SymbolicDynamics, Dynamics
from rhddp.rhddp import RHDDP

import os
import sys

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','ieee']) #comment out this line if you do not have LaTeX installed!

sys.path.append( os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) ))

class DIDynamics(Dynamics):
    def __init__(self):
        self.dt = 0.01
        self.A = np.array([[1, self.dt], [0, 1]])
        self.B = np.array([[0], [self.dt]])
    
    def f(self, x,u):
        return self.A @ x + self.B @ u.reshape((1,))
    
    def fx(self, x, u):
        return self.A
    
    def fu(self, x, u):
        return self.B
    
    def p(self, x, d):
        x[1] += 0.5
        x[1] += d
        return x
        #return x + np.array([0, 0.5+d]) issues with broadcasting since d is already an np array
    
    def px(self, x, d):
        return np.eye(2)
    
    def pd(self, x, d):
        return np.array([[0], [1]])
    
    @property
    def num_states(self):
        return 2

    @property
    def num_inputs(self):
        return 1

    @property
    def num_disturbances(self):
        return 1

    @property
    def name_str(self):
        pass

dynamics = DIDynamics()
# dynamics = SymbolicDynamics(name="double_integrator_dynamics",
#                                               num_states=2,
#                                               num_inputs=1,
#                                               num_disturbances=1)

class NonQCost(Cost):
    def __init__(self):
        self.goal_state = np.array([5.0, 0.0])
        self.Q = np.array([[0.01,0], [0,0.01]])
        self.Qn = np.array([[500,0], [0,1000]])
        self.quad_cost = True

        self.R = np.diag([2.5])

        if not self.quad_cost:
            self.Q = np.array([[0.001,0], [0,0.1]])
        self.k_u = 5
        self.xshift_u = np.log(self.k_u - 1) / self.k_u
        self.yshift_u = -np.log(self.k_u - 1) / self.k_u - np.log(self.k_u / (self.k_u - 1))
        self.scale_u = 2
    
    def l_input(self, u):
        if self.quad_cost:
            return u.T @ self.R @ u
        else:
            return self.scale_u * np.logaddexp((u[0] + self.xshift_u), -(self.k_u-1)*(u[0] + self.xshift_u)) + self.yshift_u
            
    def l(self, x, u):
        if self.quad_cost:
            return (x - self.goal_state).T @ self.Q @ (x - self.goal_state) + u.T @ self.R @ u
        else:
            cost_u = self.scale_u * np.logaddexp((u[0] + self.xshift_u), -(self.k_u-1)*(u[0] + self.xshift_u)) + self.yshift_u
            return (x - self.goal_state).T @ self.Q @ (x - self.goal_state) + cost_u

    def lx(self, x, u):
        return 2 * self.Q @ (x - self.goal_state)
    
    def lu(self, x, u):
        if self.quad_cost:
            return 2 * self.R @ u
        else:
            return self.scale_u * np.array([1 - self.k_u + self.k_u / (1 + np.exp(-self.k_u * (u[0] + self.xshift_u)))])

    def lxx(self, x, u):
        return 2 * self.Q
    
    def lux(self, x, u):
        return np.zeros((1, 2))
    
    def luu(self, x, u):
        if self.quad_cost:
            return 2 * self.R
        else:
            temp = np.exp(-self.k_u * (u[0] + self.xshift_u))
            return self.scale_u * self.k_u ** 2 * temp / (1 + temp) ** 2

    def phi(self, x_t):
        return (x_t - self.goal_state).T @ self.Qn @ (x_t - self.goal_state)
    
    def phix(self, x_t):
        return 2 * self.Qn @ (x_t - self.goal_state)
    
    def phixx(self, x_t):
        return 2 * self.Qn
    
    @property
    def num_inputs(self):
        return 1

    @property
    def num_states(self):
        return 2
    
    @property
    def name_str(self):
        pass

cost = NonQCost()
# cost = SymbolicCost(name="double_integrator_cost",
#                                       num_states=2,
#                                       num_inputs=1)

d_nom = 0.0
robust_settings = {"initial_state": np.array([0, 0]),
            "horizon": 1000,
            "d_nom": d_nom,
            "d_set": [-0.5, 0.5],
            "reset_prop": 0.8,
            "conv_criterion": 12
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


num_test_points = 50


robust_dists, robust_trajs, robust_costs, robust_control_costs = robust_controller.evalDiscreteSolved(num_test_points=num_test_points)


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


vanilla_dists, vanilla_trajs, vanilla_costs, vanilla_control_costs = robust_controller.evalDiscrete(x_traj = vanilla_solution.get("x_traj"),
                                                                u_traj = vanilla_solution.get("u_traj"),
                                                                K_traj = vanilla_solution.get("K_traj"),
                                                                num_test_points = num_test_points)

# robust_traj_at_worst_vanilla = robust_trajs[np.argmax(vanilla_costs)]
# vanilla_traj_at_worst_vanilla = vanilla_trajs[np.argmax(vanilla_costs)]
# robust_traj_at_worst_robust = robust_trajs[np.argmax(robust_costs)]
# vanilla_traj_at_worst_robust = vanilla_trajs[np.argmax(robust_costs)]


#THIS DOES NOT ACTUALLY FIND THE WORST ROBUST TRAJ, BUT THE TRAJ AT WORST VANILLA
worst_robust_traj = robust_trajs[np.argmax(vanilla_costs)]
worst_vanilla_traj = vanilla_trajs[np.argmax(vanilla_costs)]
#worst_robust_x_traj, worst_robust_u_traj, worst_robust_cost = robust_controller.get_solved_trajectory(np.array([d_worst]))

worst_robust_x_traj, worst_robust_u_traj, worst_robust_cost = worst_robust_traj.x_traj, worst_robust_traj.u_traj, worst_robust_traj.cost 

#worst_vanilla_x_traj, worst_vanilla_u_traj, worst_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([d_worst]))
worst_vanilla_x_traj, worst_vanilla_u_traj, worst_vanilla_cost = worst_vanilla_traj.x_traj, worst_vanilla_traj.u_traj, worst_vanilla_traj.cost 


nominal_robust_x_traj, nominal_robust_u_traj, nominal_robust_cost = robust_controller.get_solved_trajectory(np.array([d_nom]))
nominal_vanilla_x_traj, nominal_vanilla_u_traj, nominal_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([d_nom]))

# print(robust_costs)
# print(vanilla_costs)
plt.figure()
#plt.ylim(150,460)
plt.plot(robust_dists, robust_costs, label="RHDDP")
plt.plot(vanilla_dists, vanilla_costs, color='blue', label="Vanilla")
#plt.axvline(x=d_nom,  linestyle='dotted', color='green', label='Nominal\nDisturbance')
#plt.axvline(x=d_worst,  linestyle='dashdot', color='green', label='Worst\nDisturbance')
plt.xlabel("Disturbance Value")
plt.ylabel("Cost")
plt.legend()
plt.legend(loc='upper left', fontsize=6)
plt.show()
#plt.title("Simulated Cost vs Disturbance for Double Integrator")
plt.savefig('plots/double_integrator_cc/costsVdists.png')

plt.figure()
#plt.ylim(150,460)
plt.plot(robust_dists, robust_control_costs, color='blue', label="RHDDP")
plt.plot(vanilla_dists, vanilla_control_costs, linestyle='dotted', color='blue', label="Vanilla")
plt.plot(robust_dists, robust_costs-robust_control_costs, color='red', label="RHDDP")
plt.plot(vanilla_dists, vanilla_costs-vanilla_control_costs, linestyle='dotted', color='red', label="Vanilla")
plt.axvline(x=d_nom,  linestyle='dotted', color='green', label='Nominal\nDisturbance')
#plt.axvline(x=d_worst,  linestyle='dashdot', color='green', label='Worst\nDisturbance')
plt.xlabel("Disturbance Value)")
plt.ylabel("Input Cost")
plt.legend()
plt.legend(loc='upper left', fontsize=6)
plt.show()
plt.savefig('plots/double_integrator_cc/subtractedcosts.png')

#region worst case plotting
if False:
    worst_robust_x_traj, worst_robust_u_traj, worst_robust_cost = robust_controller.get_solved_trajectory(np.array([d_worst]))
    worst_vanilla_x_traj, worst_vanilla_u_traj, worst_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([d_worst]))

    plt.figure()
    plt.plot(np.arange(worst_robust_x_traj.shape[1])*0.01,  worst_robust_x_traj[0, :], label="RH-DDP")
    plt.plot(np.arange(worst_vanilla_x_traj.shape[1])*0.01,  worst_vanilla_x_traj[0, :], color='blue', label="HS-DDP")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(loc='lower right', fontsize=6)

    plt.show()
    #plt.title("Worst Case Double integrator Positions vs. Timestep")
    # plt.savefig('plots/double_integrator/wc/wc_double_integrator_position_plots.png')

    plt.figure()
    plt.plot(np.arange(worst_robust_x_traj.shape[1])*0.01,  worst_robust_x_traj[1, :], label="RH-DDP")
    plt.plot(np.arange(worst_vanilla_x_traj.shape[1])*0.01,  worst_vanilla_x_traj[1, :], color='blue', label="HS-DDP")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity ($\\frac{m}{s}$)")
    plt.legend(loc='lower left', fontsize=6)
    plt.show()
    #plt.title("Worst Case Double integrator Velocities vs. Timestep")
    # plt.savefig('plots/double_integrator/wc/wc_double_integrator_velocity_plots.png')

    plt.figure()
    plt.plot(worst_robust_x_traj[0, :],  worst_robust_x_traj[1, :], label="RH-DDP",  zorder=1)
    plt.plot(worst_vanilla_x_traj[0, :],  worst_vanilla_x_traj[1, :], color='blue', label="HS-DDP", zorder=2)
    plt.scatter(5,0, marker="x", color="green", label="Goal State", s=5, zorder=3)
    plt.scatter(0,0, marker="o", color="red", label="Initial State", s=4, zorder=4)
    plt.xlabel("Position (m)")
    plt.ylabel("Velocity ($\\frac{m}{s}$)")
    plt.legend(loc='upper left', fontsize=6)
    plt.ylim(-0.1, 1.4)
    plt.show()
    #plt.title("Worst Case Double integrator Position vs. Velocity Phase Plot")
    # plt.savefig('plots/double_integrator/wc/wc_double_integrator_phase_plots.png')

    plt.figure()
    plt.plot(np.arange(worst_robust_u_traj.shape[1])*0.01,  worst_robust_u_traj[0, :], label="RH-DDP")
    plt.plot(np.arange(worst_vanilla_u_traj.shape[1])*0.01,  worst_vanilla_u_traj[0, :], color='blue', label="HS-DDP")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Force (N)")
    plt.legend(loc='upper right', fontsize=6)
    plt.show()
    #plt.title("Worst Case Double integrator Controls vs. Timestep")
    # plt.savefig('plots/double_integrator/wc/wc_double_integrator_control_plots.png')
    #endregion worst case plotting

    #region nominal plotting
    nominal_robust_x_traj, nominal_robust_u_traj, nominal_robust_cost = robust_controller.get_solved_trajectory(np.array([d_nom]))
    nominal_vanilla_x_traj, nominal_vanilla_u_traj, nominal_vanilla_cost = vanilla_controller.get_solved_trajectory(np.array([d_nom]))

    plt.figure()
    plt.plot(np.arange(nominal_robust_x_traj.shape[1])*0.01,  nominal_robust_x_traj[0, :], label="RH-DDP")
    plt.plot(np.arange(nominal_vanilla_x_traj.shape[1])*0.01,  nominal_vanilla_x_traj[0, :], color='blue', label="HS-DDP")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend(loc='lower right', fontsize=6)
    plt.show()
    #plt.title("Nominal Double integrator Positions vs. Timestep")
    # plt.savefig('plots/double_integrator/nom/nom_double_integrator_position_plots.png')

    plt.figure()
    plt.plot(np.arange(nominal_robust_x_traj.shape[1])*0.01,  nominal_robust_x_traj[1, :], label="RH-DDP")
    plt.plot(np.arange(nominal_vanilla_x_traj.shape[1])*0.01,  nominal_vanilla_x_traj[1, :], color='blue', label="HS-DDP")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity ($\\frac{m}{s}$)")
    plt.legend(loc='lower left', fontsize=6)
    plt.show()
    #plt.title("Nominal Double integrator Velocities vs. Timestep")
    # plt.savefig('plots/double_integrator/nom/nom_double_integrator_velocity_plots.png')

    plt.figure()
    plt.plot(nominal_robust_x_traj[0, :],  nominal_robust_x_traj[1, :], label="RH-DDP",  zorder=1)
    plt.plot(nominal_vanilla_x_traj[0, :],  nominal_vanilla_x_traj[1, :], color='blue', label="HS-DDP", zorder=2)
    plt.scatter(5,0, marker="x", color="green", label="Goal State", s=5, zorder=3)
    plt.scatter(0,0, marker="o", color="red", label="Initial State", s=4, zorder=4)
    plt.xlabel("Position (m)")
    plt.ylabel("Velocity ($\\frac{m}{s}$)")
    plt.legend(loc='upper left', fontsize=6)
    plt.ylim(-0.1, 1.4)
    plt.show()
    #plt.title("Nominal Double integrator Position vs. Velocity Phase Plot")
    # plt.savefig('plots/double_integrator/nom/nom_double_integrator_phase_plots.png')

    plt.figure()
    plt.plot(np.arange(nominal_robust_u_traj.shape[1])*0.01,  nominal_robust_u_traj[0, :], label="RH-DDP")
    plt.plot(np.arange(nominal_vanilla_u_traj.shape[1])*0.01,  nominal_vanilla_u_traj[0, :], color='blue', label="HS-DDP")
    plt.xlabel("Time (s)")
    plt.ylabel("Control Force (N)")
    plt.legend(loc='upper right', fontsize=6)
    plt.show()
    #plt.title("Nominal Double integrator Controls vs. Timestep")
    # plt.savefig('plots/double_integrator/nom/nom_double_integrator_control_plots.png')


    #endregion nominal plotting




print(f"wc robust control: {np.max(worst_robust_u_traj[0, :])}")
print(f"wc vanilla control: {np.max(worst_vanilla_u_traj[0, :])}")

print(f"nom robust control: {np.max(nominal_robust_u_traj[0, :])}")
print(f"nom vanilla control: {np.max(nominal_vanilla_u_traj[0, :])}")

print(f"wc robust cost: {worst_robust_cost}")
print(f"wc vanilla cost: {worst_vanilla_cost}")

print(f"nom robust cost: {nominal_robust_cost}")
print(f"nom vanilla cost: {nominal_vanilla_cost}")

pass