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
settings = {"initial_state": np.array([0, 0]),
            "horizon": 1000,
            "d_nom": d_nom,
            "reset_prop": 0.8,
            }
hyperparams = {"max_iters": 5, 
               "conv_criterion": 8} 

problem = Problem(name="double_integrator_baseline",
                  dyn=dynamics,
                  cost=cost,
                  settings=settings,
                  hyperparams=hyperparams)

controller = RHDDP(problem, action=None)
solution = controller.solve()

