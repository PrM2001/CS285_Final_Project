import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import symforce.symbolic as sf
import numpy as np
from rhddp.infrastructure.symbolic_utils import codegen_goal_state_cost

c_g = sf.Symbol("gait_cost_scalar") #gait tracking cost scalar
c_v = sf.Symbol("vel_cost_scalar") #velocity reference cost scalar
c_u = sf.Symbol("control_cost_scalar") #control cost scalar
c_t = sf.Symbol("terminal_cost_scalar") #terminal cost scalar
c_t2 = sf.Symbol("terminal_cost_scalar2")
q = sf.V4.symbolic("q")
dq = sf.V4.symbolic("dq")
u = sf.V1.symbolic("u")

terminal_state = sf.V8.symbolic("terminal_state")

# region control cost
control_cost = c_u * u[0] ** 2
# endregion control cost

# region velocity cost
vel = dq[0]
vel_ref = sf.Symbol("vel_ref")

velocity_cost = c_v * (vel - vel_ref) ** 2
# endregion velocity cost 

# region gait cost 
gait_beta = sf.V4.symbolic("gait_beta")
th1d = sf.Symbol("gait_th1d")

beta0 = gait_beta[0]
beta1 = gait_beta[1]
beta2 = gait_beta[2]
beta3 = gait_beta[3]

# for now, theta1 = absolute angle of leg1 (stance leg)
# and leg2 angle relative to leg1 (the actuated variable)
theta1 = q[2]
theta2 = q[3]
ya = theta2
yd = (theta1 + th1d) * (theta1 - th1d) * (beta3 * theta1 ** 3 + beta2 * theta1 ** 2 + beta1 * theta1 + beta0) - theta1 * 2.0

gait_cost = c_g * (ya - yd) ** 2
# endregion gait cost

running_cost = sf.M11.symbolic("running_cost") 
running_cost[0] = gait_cost + control_cost + velocity_cost #+ sf.log(1 + sf.exp(-dq[2]))

# region terminal cost
theta1_t = terminal_state[2]
theta2_t = terminal_state[3]
vx_t = terminal_state[4]

theta1_t_d = sf.Symbol("theta1_t_d")
theta2_t_d = sf.Symbol("theta2_t_d")
vx_t_d = sf.Symbol("vx_t_d")

terminal_cost = sf.M11.symbolic("terminal_cost") 
# terminal_cost[0] = c_t * ((theta1_t - theta1_t_d) ** 2 + (theta2_t - theta2_t_d) ** 2 + (vx_t - vx_t_d) ** 2)
# terminal_cost[0] = c_t * ((theta1_t - theta1_t_d) ** 2 + (theta2_t - theta2_t_d) ** 2)
terminal_cost[0] = c_t * (3 * (2 * theta1_t + theta2_t) ** 2 + (vx_t - vx_t_d) ** 2)

# endregion terminal cost

value_nonnegative = -(2 * terminal_state[6] + terminal_state[7])
terminal_cost[0] = terminal_cost[0] + c_t2 * sf.log(1 + sf.exp(-2 - value_nonnegative))
value_nonnegative = - (theta1_t + 0.05)
terminal_cost[0] = terminal_cost[0] + 17500 * sf.log(1 + sf.exp(-2 - 3 * value_nonnegative))

params_dict = {c_g: 0.0,
               c_v: 0.5,
               c_u: 3,
               c_t: 30000,
               c_t2: 1000,
               theta1_t_d: -0.13,        #initial state here, currently not the best implementation
               theta2_t_d: 0.26,
               vx_t_d: 0.0,
               beta0: -17.9892859139367,
               beta1: -49.9978114563501,
               beta2: 12.9808057982654,
               beta3: -7.79807852527223,
               th1d: -0.13,
               vel_ref: 0.325}

running_cost = 0.01 * running_cost.subs(params_dict)
terminal_cost = 0.01 * terminal_cost.subs(params_dict)

state = q.col_join(dq)

codegen_goal_state_cost(name="compass_gait_cost",
              state=state, 
              control=u,
              terminal_state=terminal_state,
              running_cost=running_cost, 
              terminal_cost=terminal_cost)