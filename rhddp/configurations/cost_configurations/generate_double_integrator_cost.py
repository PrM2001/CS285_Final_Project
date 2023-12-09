import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import symforce.symbolic as sf
import numpy as np
from rhddp.infrastructure.symbolic_utils import codegen_goal_state_cost

# Q = sf.Matrix(np.array([[0.00,0], [0,0.00]]))#sf.Matrix(np.diag([1,1]))
# #"best" results with 0.0 instead of 0.1
# Qn = sf.Matrix(np.array([[500,0], [0,500]]))#Qn = sf.Matrix(np.diag([100,100]))
# R = sf.Matrix(np.diag([2.5]))


Q = sf.Matrix(np.array([[0.01,0], [0,0.01]]))#sf.Matrix(np.diag([1,1]))
#"best" results with 0.0 instead of 0.1
Qn = sf.Matrix(np.array([[500,0], [0,1000]]))#Qn = sf.Matrix(np.diag([100,100]))
R = sf.Matrix(np.diag([2.5]))


state = sf.V2.symbolic("state")
terminal_state = sf.V2.symbolic("terminal_state")
goal_state = sf.V2.symbolic("goal_state")
control = sf.V1.symbolic("control")

running_cost = (state - goal_state).T * Q * (state - goal_state) + control.T * R * control
terminal_cost = (terminal_state - goal_state).T * Qn * (terminal_state - goal_state)


# params_dict = {goal_state: np.array([5, 0])}
params_dict = {goal_state[0]: 0, goal_state[1]: 0}
running_cost = running_cost.subs(params_dict)
terminal_cost = terminal_cost.subs(params_dict)

codegen_goal_state_cost(name="double_integrator_cost",
              state=state, 
              control=control,
              terminal_state=terminal_state,
              running_cost=running_cost, 
              terminal_cost=terminal_cost)