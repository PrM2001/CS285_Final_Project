import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import symforce.symbolic as sf
import numpy as np
from rhddp.infrastructure.symbolic_utils import get_lagrangian_dynamics_matrices, codegen_dynamics

# q[0]: torso x position
# q[1]: torso y position
# q[2]: absolute angle of leg1 (stance leg)
# q[3]: leg2 angle relative to leg1 (the actuated variable)
q = sf.V4.symbolic("q")
dq = sf.V4.symbolic("dq")
mL = sf.Symbol("mass_leg")
lL = sf.Symbol("length_leg")
JL = sf.Symbol("interia_leg")
g = sf.Symbol("gravity")
mH = sf.Symbol("mass_hip")
u = sf.V1.symbolic("u")
d = sf.V2.symbolic("d")
# values that you will substitute later.
params_dict = {lL: 1.0, mL: 1.0, JL: 0, g: 9.81, mH: 1.0}        

# position of torso (hip).
pos_torso = sf.V2([q[0], q[1]])
# position of centor of mass (com) of leg 1 and leg 2.
pos_leg1 = pos_torso + 0.5 * lL * sf.V2([sf.sin(q[2]), -sf.cos(q[2])])
pos_leg2 = pos_torso + 0.5 * lL * sf.V2([sf.sin(q[2] + q[3]),
    -sf.cos(q[2] + q[3])])
pos_stance_foot = pos_torso + lL * sf.V2([sf.sin(q[2]), -sf.cos(q[2])])
pos_swing_foot = pos_torso + lL * sf.V2([sf.sin(q[2] + q[3]),
    -sf.cos(q[2] + q[3])])

dpos_leg1 = pos_leg1.jacobian(q) * dq
dpos_leg2 = pos_leg2.jacobian(q) * dq

# kinetic energy
ke_hip = 0.5 * mH * (dq[0] ** 2 + dq[1] ** 2)
ke_leg1 = 0.5 * mL * (dpos_leg1[0] ** 2 + dpos_leg1[1] ** 2) + 0.5 * JL * dq[2] ** 2
ke_leg2 = 0.5 * mL * (dpos_leg2[0] ** 2 + dpos_leg2[1] ** 2) + 0.5 * JL * (dq[2] + dq[3]) ** 2
ke = sf.V1(ke_hip + ke_leg1 + ke_leg2).simplify()

# potential energy
pe_leg1 = mL * g * pos_leg1[1]
pe_leg2 = mL * g * pos_leg2[1]
pe_hip = mH * g * pos_torso[1]
pe = sf.V1(pe_hip + pe_leg1 + pe_leg2).simplify()

[D, C, G, B] = get_lagrangian_dynamics_matrices(ke, pe, q, dq, sf.V1([q[3]]))

# jacobian of the stance foot constriant (used for constrained lagrangian dynamics)
jacobian_stance_foot = pos_stance_foot.jacobian(q)
# time derivative of the jacobian.
dj_stance_foot = sf.Matrix(jacobian_stance_foot.shape[0], jacobian_stance_foot.shape[1])
for i in range(jacobian_stance_foot.shape[0]):
    for j in range(jacobian_stance_foot.shape[1]):
        dj_stance_foot[i, j] = (sf.V1([jacobian_stance_foot[i, j]]).jacobian(q) * dq).simplify()

# jacobian of the swing foot (used for impact model)
jacobian_swing_foot = pos_swing_foot.jacobian(q)

# constrained Lagrangian dynamics
# schur compliment of kkt matrix w.r.t. D
MD_stance = (jacobian_stance_foot * D.inv() * jacobian_stance_foot.T)
matrix_temp1 = (jacobian_stance_foot.T * MD_stance.inv())
matrix_temp2 = (sf.Matrix(np.eye(4)) - matrix_temp1 * jacobian_stance_foot * D.inv())

# schur compliment of kkt matrix w.r.t. D
MD_swing = (jacobian_swing_foot * D.inv() * jacobian_swing_foot.T)
matrix_temp3 = (jacobian_swing_foot.T * MD_swing.inv())
matrix_temp4 = (sf.Matrix(np.eye(4)) - matrix_temp3 * jacobian_swing_foot * D.inv())
# dq here is dq at pre-impact.
# TODO: think about what disturbance we introduce here.
# change of coordinate between swing foot and stance foot.
R = sf.Matrix(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, -1]]))
q_post_impact = R * q
dq_post_impact = R * (D.inv() * matrix_temp4 * D * dq)

#dq[0] = dq[0] + d 
# q stays same during impact.
reset_map = q_post_impact.col_join(dq_post_impact)

#pranit's set up that may be working?
# reset_map[4] = reset_map[4] + d[0] 
# reset_map[5] = reset_map[5] - 2 * d[0]

#Jasons new set up
reset_map[4] = reset_map[4] * (0.8 + d[0])
reset_map[6] = - reset_map[4] / sf.cos(reset_map[2])
reset_map[5] = - sf.sin(reset_map[2]) * reset_map[6]
# This can be changed to another disturbance dimension.
reset_map[7] = reset_map[7] * (0.8 + d[1])


# autonomous part of d2q
d2q_fvec = (D.inv() * (matrix_temp2 * (- C * dq - G) - matrix_temp1 * dj_stance_foot * dq))
# control vector filed part of d2q
d2q_gvec = (D.inv() * matrix_temp2 * B)
fvec = sf.Matrix(np.zeros(8))
fvec[0:4] = dq
fvec[4:] = d2q_fvec
gvec = sf.Matrix(np.zeros((8, 1)))
gvec[4:, :] = d2q_gvec

# continuous vector field.
fvec = fvec.subs(params_dict)
gvec = gvec.subs(params_dict)
reset_map = reset_map.subs(params_dict)

state = q.col_join(dq)
dt = 0.003


codegen_dynamics("compass_gait_dynamics_2d_disturbance", state, u, d, dt, fvec, gvec, reset_map)
