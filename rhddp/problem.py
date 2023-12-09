import numpy as np
from rhddp.cost import Cost
from rhddp.dynamics import Dynamics


class Problem():
    def __init__(self, name: str, dyn: Dynamics, cost: Cost, settings: dict, hyperparams: dict):

        print(f'\nCreating Problem Class {name} with Dynamics Class {dyn.name_str} and Cost Class {cost.name_str}')

        self._prob_name = name
        self._dyn = dyn
        self._cost = cost

        assert dyn.num_states == cost.num_states, f'{dyn.name_str} (Dynamics) has {dyn.num_states} states, {cost.name_str} (Cost) assumes {cost.num_states} states.'
        assert dyn.num_inputs == cost.num_inputs, f'{dyn.name_str} (Dynamics) has {dyn.num_inputs} inputs, {cost.name_str} (Cost) assumes {cost.num_inputs} inputs.' 
        self.num_states = dyn.num_states
        self.num_inputs = dyn.num_inputs
        self.num_disturbances = dyn.num_disturbances

        # Do we think the unpacking of settings and hyperparams should happen here? 
        # I think it should, and all compatibility checks should happen here. 
        self.initial_state = settings.get("initial_state")
        self.horizon = settings.get("horizon")
        self.warm_start_u = settings.get("warm_start_u", None)
        self.warm_start_K = settings.get("warm_start_K", None)
        self.d_nom = settings.get("d_nom")        
        self.d_set = settings.get("d_set", None)  #if no dSet is given, we will assume we want to use nonrobust HDDP
        self.reset_prop = settings.get("reset_prop")

        self.max_iters = hyperparams.get("max_iters", 10)

        self.alpha_initial = hyperparams.get("alpha", 1)
        self.alpha_update_period = hyperparams.get("alpha_update_period", np.inf)
        self.alpha_increment = hyperparams.get("alpha_increment", 1)

        self.regularization_initial = hyperparams.get("regularization", 0.0)
        self.regularization_update_period = hyperparams.get("regularization_update_period", np.inf)
        self.regularization_decrease_factor = hyperparams.get("regularization_decrease_factor", 1)

        self.beta_regularization_eps = hyperparams.get("beta_regularization_eps", 1e-4)

        self.wolfe_b = hyperparams.get("wolfe_b", 0.01)
        self.wolfe_c = hyperparams.get("wolfe_c", 0.5)

        self.conv_criterion = hyperparams.get("conv_criterion", 8)
        

        #dependent settings (?)
        self._robust = self.d_set is not None #if the only disturbance is the "zero disturbance," not robust case
        
        if np.isscalar(self.d_nom):
            self.d_nom = np.array([self.d_nom])

        if not self._robust: 
            self.d_set = [np.zeros_like(self.d_nom)] 

        self.n_d = len(self.d_set)



        self.d_set = [np.array([d]) if np.isscalar(d) else d for d in self.d_set]

        self._reset_step = int(self.reset_prop * self.horizon)
        
        #Problem creation print statement
        #print(f'Done building problem {name}, with the following hyperparameters changed from the defaults: \n {hyperparams}\n') #make this print out hyperparameters later for debugging help
        
    #region Dynamics Methods
    def step(self, x, u):
        return self._dyn.f(x, u)
        
    def fx(self, x, u):
        return self._dyn.fx(x, u)

    def fu(self, x, u):
        return self._dyn.fu(x, u)
    

    def reset(self, x, d):
        return self._dyn.p(x, d)
    
    def px(self, x, d):
        return self._dyn.px(x, d)
    
    def pd(self, x, d):
        return self._dyn.pd(x, d)
    
    def resetCondition(self, x, u, k):
        return self._dyn.resetCondition(self, x, u, k)
    
    # Could add continous vector fields (maybe for simulation?) But I don't exactly see why else. 

    #endregion Dynamics Methods

    #region Cost Methods
    def l(self, x, u):          #running cost. idk if i should rename
        return self._cost.l(x,u)
    
    def lx(self, x, u):
        return self._cost.lx(x,u)
    
    def lu(self, x, u):
        return self._cost.lu(x,u)
    
    def lxx(self, x, u):
        return self._cost.lxx(x,u)
    
    def lux(self, x, u):
        return self._cost.lux(x,u)
    
    def luu(self, x, u):
        return self._cost.luu(x,u)
    

    def phi(self, x_t):
        return self._cost.phi(x_t)
    
    def phix(self, x_t):
        return self._cost.phix(x_t)
    
    def phixx(self, x_t):
        return self._cost.phixx(x_t)
    

    def calculate_cost(self, x_traj, u_traj): #TODO modify the RHDDP algorithm to return/update the reset_step?
        costValue = 0

        assert self.horizon == u_traj.shape[1], f'{self._prob_name} has horizon {self.horizon}. Given trajectory has horizon {u_traj.shape[1]}.'

        for i in range(self._reset_step):
            costValue += self._cost.l(x_traj[:, i], u_traj[:, i])  #IDK if it's good convention to do the _Cost. here
        
        for i in range(self._reset_step + 1, self.horizon + 1):
            costValue += self._cost.l(x_traj[:, i], u_traj[:, i - 1])

        costValue += self._cost.phi(x_traj[:, -1])

        if not np.isscalar(costValue):
            costValue = costValue.item()

        return costValue
    #endregion Cost Methods

    def update(self, initial_state, horizon, reset_prop):
        self.initial_state = initial_state
        self.horizon = horizon
        self._reset_step = int(reset_prop * self.horizon)