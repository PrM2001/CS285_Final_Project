import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import symforce.symbolic as sf
import numpy as np
from rhddp.infrastructure.symbolic_utils import codegen_dynamics

# q: position
# dq: velocity 
q = sf.V1.symbolic("q")
dq = sf.V1.symbolic("dq")
m = sf.Symbol("mass")
f_r = sf.Symbol("friction")
u = sf.V1.symbolic("u")
#d = sf.V1.symbolic("disturbance")
d = sf.V1.symbolic("disturbance") #2d

# values that you will substitute later.
params_dict = {m: 1.0, f_r: 1}   #set f_r = 1 to match original testing     
state = q.col_join(dq)

fvec = sf.Matrix(np.zeros((2,1)))
gvec = sf.Matrix(np.zeros((2,1)))

fvec[0] = dq[0]

gvec[1] = 1/m


#reset_map = q.col_join(f_r * dq + d)

q_reset = q
reset_map = q.col_join(dq + 0.5 + d[0]) 

dt = 0.01
fvec = fvec.subs(params_dict)
gvec = gvec.subs(params_dict)
reset_map = reset_map.subs(params_dict)

codegen_dynamics("double_integrator_dynamics", state, u, d, dt, fvec, gvec, reset_map)
