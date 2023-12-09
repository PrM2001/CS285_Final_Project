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
hyperparams = {"max_iters": 5, 
               "conv_criterion": 8} 

settings = {"initial_state": np.array([0, 0]),
            "horizon": 1000,
            "d_nom": d_nom,
            "reset_prop": 0.8,
            }

problem = Problem(name="double_integrator_baseline",
                  dyn=dynamics,
                  cost=cost,
                  settings=settings,
                  hyperparams=hyperparams)

#TODO:
K = 


#everythign above happens once:

#this is inside the training loop:

#TODO: draw initial_state, horizon, reset from distributions (horizon uniformly between 100 and 400 maybe), 
# reset prop uniformly from 0.2 to 0.8, initial state uniformly around 0,0 (maybe 5 in each direction)
initial_state = ...
horizon = ...
reset_prop = ...

problem.update(initial_state=initial_state, horizon=horizon, reset_prop=reset_prop)

#TODO: draw action from rl agent, given state
state = ...
action = ...

vanilla_controller = RHDDP(problem, action=None)
vanilla_solution = vanilla_controller.solve()

rl_controller = RHDDP(problem, action=action)
rl_solution = rl_controller.solve()

for _ in range(K):
    #TODO: Draw disturbance:
    disturbance = ...
    
    #TODO: create evaluation function for each, on the disturbance





# need to roll out and compute cost over disturbances.

