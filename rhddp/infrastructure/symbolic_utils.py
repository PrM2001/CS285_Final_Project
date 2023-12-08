import symforce.symbolic as sf
from symforce import codegen
from symforce.values import Values
from symforce.codegen import codegen_util
import importlib

# TODO(pranit): make this a class that inherits the Model class in DDP_Simple
class SymbolicControlAffineModel:
    def __init__(self, name: str, codegen_dir: str='gen'):
        self.name_str = name
        self._fvec_cont_gen = importlib.import_module(codegen_dir + '.' + name + '.fvec_continuous').fvec_continuous
        self._gvec_cont_gen = importlib.import_module(codegen_dir + '.' + name + '.gvec_continuous').gvec_continuous
        
        self._fx_gen = importlib.import_module(codegen_dir + '.' + name + '.fx').fx
        self._fu_gen = importlib.import_module(codegen_dir + '.' + name + '.fu').fu        

    def fx(self, x, u):
        return self._fx_gen(x, u)

    def fu(self, x, u):
        return self._fu_gen(x)

    def dx(self, x, u):
        fx = self._fvec_cont_gen(x)[..., None]
        gx = self._gvec_cont_gen(x)
        gx = gx[..., None] if gx.ndim == 1 else gx
        return fx + gx @ u[..., None]


def get_lagrangian_dynamics_matrices(kinetic_energy: sf.V1,
                                   potential_energy: sf.V1,
                                   q: sf.Matrix,
                                   dq: sf.Matrix,
                                   q_actuated: sf.Matrix):
    """
        kinetic_energy: symbolic expression for kinetic energy.
        potential_energy: symbolic expression for potential energy.
        q: configuration variables
        dq: generalized velocity
        q_actuated: actuated configuration variables
    """
    mass_matrix = (kinetic_energy.jacobian(dq).transpose().jacobian(dq)).simplify()
    dim_q = q.shape[0]
    corioli_matrix = sf.Matrix(dim_q, dim_q)
    for k in range(dim_q):
        for j in range(dim_q):
            for i in range(dim_q):
                corioli_matrix[k,j] = corioli_matrix[k,j] + 0.5 * (mass_matrix[k,j].diff(q[i]) + mass_matrix[k,i].diff(q[j]) - mass_matrix[i,j].diff(q[k])) * dq[i]
    corioli_matrix = corioli_matrix.simplify()
    gravity_matrix = potential_energy.jacobian(q).T.simplify()
    input_matrix = q_actuated.jacobian(q).T.simplify()
    return [mass_matrix, corioli_matrix, gravity_matrix, input_matrix]

def codegen_helper(config_name: str,
                   output_name: str,
                   input: Values,
                   output: Values):
    codegen_handle = codegen.Codegen(
        inputs=input,
        outputs=output,
        config=codegen.PythonConfig(use_eigen_types=False),
        name=output_name
    )
    codegen_data = codegen_handle.generate_function(
        output_dir='gen/' + config_name,
        skip_directory_nesting=True,
    )
    return codegen_data

def codegen_control_affine_dynamics(name: str,
                                    state: sf.Matrix,
                                    control: sf.Matrix,
                                    dt: float,
                                    fvec: sf.Matrix,
                                    gvec: sf.Matrix,
                                    reset_map: sf.Matrix):
    # get discretized dynamics
    # we use simple Euler method.
    fvec_discrete = state + dt * fvec
    gvec_discrete = dt * gvec
    x_next = fvec_discrete + gvec_discrete * control

    input_x = Values()
    input_x["x"] = state

    input_xu = Values()
    input_xu["x"] = state
    input_xu["u"] = control

    input_xd = Values()
    input_xd["x"] = state
    fx_output = Values(fx=x_next.jacobian(state))
    fu_output = Values(fu=gvec_discrete)

    codegen_helper(name, 'fvec_continuous', input_x,  Values(fvec=fvec))
    codegen_helper(name, 'gvec_continuous', input_x,  Values(gvec=gvec))
    codegen_helper(name, 'reset_map', input_x,  Values(state_next=reset_map))   

    fx_codegen_data = codegen_helper(name, 'fx', input_xu, fx_output)
    fu_codegen_data = codegen_helper(name, 'fu', input_x, fu_output)


    # print("Files generated in {}.".format(fx_codegen_data.output_dir))


def codegen_dynamics(name: str,
                     state: sf.Matrix,
                     control: sf.Matrix,
                     disturbance: sf.Matrix,
                     dt: float,
                     fvec: sf.Matrix,
                     gvec: sf.Matrix,
                     reset_map: sf.Matrix):
    # get discretized dynamics
    # we use simple Euler method.
    fvec_discrete = state + dt * fvec
    gvec_discrete = dt * gvec
    x_next = fvec_discrete + gvec_discrete * control

    input_xu = Values()
    input_xu["x"] = state
    input_xu["u"] = control

    input_xd = Values()
    input_xd["x"] = state
    input_xd["d"] = disturbance

    f_output = Values(f=x_next)
    fx_output = Values(fx=x_next.jacobian(state))
    fu_output = Values(fu=gvec_discrete)

    p_output = Values(p=reset_map)
    px_output = Values(px=reset_map.jacobian(state))
    pd_output = Values(pd=reset_map.jacobian(disturbance))

    # codegen_helper(name, 'fvec_continuous', input_x,  Values(fvec=fvec)) #removed for now
    # codegen_helper(name, 'gvec_continuous', input_x,  Values(gvec=gvec))
    codegen_helper(name, 'f', input_xu, f_output)
    codegen_helper(name, 'fx', input_xu, fx_output)
    codegen_helper(name, 'fu', input_xu, fu_output)

    codegen_helper(name, 'p', input_xd,  p_output)
    codegen_helper(name, 'px', input_xd,  px_output)
    codegen_helper(name, 'pd', input_xd,  pd_output)    

    
    # print("Files generated in {}.".format(fx_codegen_data.output_dir))




def codegen_goal_state_cost(name: str,
                state: sf.Matrix,
                terminal_state: sf.Matrix,
                control: sf.Matrix,
                running_cost: sf.Matrix,
                terminal_cost: sf.Matrix
                ):
    # get discretized dynamics
    # we use simple Euler method.

    input_x_term = Values()
    input_x_term["x_t"] = terminal_state

    input_xu = Values()
    input_xu["x"] = state
    input_xu["u"] = control

    l_output = Values(l=running_cost)
    lx_output = Values(lx=running_cost.jacobian(state))
    lu_output = Values(lx=running_cost.jacobian(control))

    lxx_output = Values(lx=running_cost.jacobian(state).transpose().jacobian(state))
    lux_output = Values(lx=running_cost.jacobian(control).transpose().jacobian(state))
    luu_output = Values(lx=running_cost.jacobian(control).transpose().jacobian(control))

    phi_output = Values(phi=terminal_cost)
    phix_output = Values(phix=terminal_cost.jacobian(terminal_state))
    phixx_output = Values(phixx=terminal_cost.jacobian(terminal_state).transpose().jacobian(terminal_state))
    
    codegen_helper(name, 'l', input_xu,  l_output)
    codegen_helper(name, 'lx', input_xu,  lx_output)
    codegen_helper(name, 'lu', input_xu,  lu_output)
    codegen_helper(name, 'lxx', input_xu,  lxx_output)
    codegen_helper(name, 'lux', input_xu,  lux_output)
    codegen_helper(name, 'luu', input_xu,  luu_output)
    codegen_helper(name, 'phi', input_x_term,  phi_output)
    codegen_helper(name, 'phix', input_x_term,  phix_output)
    codegen_helper(name, 'phixx', input_x_term,  phixx_output)
