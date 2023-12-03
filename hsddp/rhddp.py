import numpy as np

import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax
from rhddp.problem import Problem
from rhddp.trajectory import Trajectory

import time

class RHDDP():
    def __init__(self, Problem: Problem, verbose=False):
        self._prob = Problem
        self._solution = None
        self._verbose = verbose
        self._alpha_curr = self._prob.alpha_initial
        self._regularization_curr = self._prob.regularization_initial
    
    def solve(self):
        if self._solution:  
            return self._solution
        else: 
            pass
            
        start_time = time.time()

        if self._prob.warm_start_u is None:
            u_traj = 0.01 * np.ones((self._prob.num_inputs, self._prob.horizon))
        else:
            u_traj = self._prob.warm_start_u

        if self._prob.warm_start_K is None:
            K_traj = np.zeros((self._prob.num_inputs, self._prob.num_states, self._prob.horizon))
        else:
            K_traj = self._prob.warm_start_K

        x_traj, u_traj, currDist, curr_cost = self.initial_rollout(u_traj, K_traj) 

        print(f"Initial cost: {curr_cost}.")

        for iter in range(self._prob.max_iters):
            
            if iter > 0 and iter % self._prob.alpha_update_period == 0:
                self._alpha_curr *= self._prob.alpha_increment
                self.v_log(f"Alpha updated to {self._alpha_curr}.")

            if iter > 0 and iter % self._prob.regularization_update_period == 0:
                self._regularization_curr *= self._prob.regularization_decrease_factor
                self.v_log(f"Regularization updated to {self._regularization_curr}.")

            k_traj, K_traj, deltaJ = self.backward_pass(x_traj, u_traj)
            x_traj, u_traj, currDist, curr_cost, converged = self.forward_pass(x_traj, u_traj, k_traj, K_traj, deltaJ, curr_cost)

            if converged:
                print(f"Converged in {iter + 1} of {self._prob.max_iters} iterations. Final cost is {curr_cost}.")
                break
            else:
                print(f"finished {iter + 1} of {self._prob.max_iters} iterations. Current cost is {curr_cost}.")
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        self._solution = {"x_traj": x_traj,
                          "u_traj": u_traj,
                          "K_traj": K_traj,
                          "final_cost": curr_cost,
                          "solve_time": elapsed_time,
                          "num_iters": iter
                          }

        return self._solution
    
    def rollout(self, x_traj, u_traj, dist, k_traj, K_traj, eps, initial=True):
        self.v_log("rollout")

        #defining to reduce computation and increase readability
        horizon = self._prob.horizon
        reset = int(self._prob.reset_prop * horizon)

        numStates = self._prob.num_states
        numInputs = self._prob.num_inputs
        

        x_traj_new = np.zeros((numStates, horizon + 2))
        u_traj_new = np.zeros((numInputs, horizon))

        x_traj_new[:, 0] = prev_state = self._prob.initial_state # set our initial state to be correct

        if initial:
            k_traj = np.zeros((numInputs, horizon)) 
            K_traj = np.zeros((numInputs, numStates, horizon))
            x_traj = np.zeros((numStates, horizon + 2))

        for i in range(reset):
            u_traj_new[:, i] = u_traj[:, i] + eps * k_traj[:, i] + K_traj[:, :, i] @ (x_traj_new[:, i] - x_traj[:, i])
            x_traj_new[:, i+1] = prev_state = self._prob.step(prev_state, u_traj_new[:,i])

        x_traj_new[:, reset + 1] = prev_state = self._prob.reset(prev_state, dist)
                
        for i in range(reset + 1, horizon + 1):
            u_traj_new[:, i - 1] = u_traj[:, i - 1] + eps * k_traj[:, i - 1] + K_traj[:, :, i - 1] @ (x_traj_new[:, i] - x_traj[:, i])
            x_traj_new[:, i+1] = prev_state = self._prob.step(prev_state, u_traj_new[:, i - 1])
        
        cost = self._prob.calculate_cost(x_traj_new, u_traj_new) 
        return Trajectory(x_traj_new, u_traj_new, dist, cost) 

    def initial_rollout(self, u_traj, K_traj):
        self.v_log("initial rollout")
        #start with initial state
        #for each input column, propagate the dynamics forward, and update the corresponding column in state traj

        dNom = self._prob.d_nom
        dSet = self._prob.d_set
        n_d = self._prob.n_d

        disturbedTrajs = [None] * n_d
        for j in range(n_d):
            disturbedTrajs[j] = self.rollout(x_traj=None, u_traj=u_traj, dist=(dNom + dSet[j]), 
                            k_traj=None, K_traj=K_traj, eps=1, initial=True)

        worst_traj = max(disturbedTrajs, key=lambda traj: traj.cost)
        worst_cost = worst_traj.cost
        worst_dist = worst_traj.dist

        traj_nom = self.rollout(x_traj=None, u_traj=u_traj, dist=worst_dist, 
                            k_traj=None, K_traj=K_traj, eps=1, initial=True)    
        x_traj_nom = traj_nom.x_traj
        u_traj_nom = traj_nom.u_traj

        return x_traj_nom, u_traj_nom, worst_dist, worst_cost
    
    def forward_pass(self, x_traj, u_traj, k_traj, K_traj, deltaJ, prev_cost):
        self.v_log("forward pass")
        converged = False
        eps = 1
        c = self._prob.wolfe_c
        b = self._prob.wolfe_b

        d_nom = self._prob.d_nom
        d_set = self._prob.d_set
        n_d = self._prob.n_d


        disturbedTrajs = [None] * n_d
        
        while True:
            for j in range(n_d):
                disturbedTrajs[j] = self.rollout(x_traj=x_traj, u_traj=u_traj, dist=(d_nom + d_set[j]), 
                                k_traj=k_traj, K_traj=K_traj, eps=eps, initial=False)
            
            worst_traj =  max(disturbedTrajs, key=lambda traj: traj.cost)
            worst_cost = worst_traj.cost
            worst_dist = worst_traj.dist

            if (worst_cost <=  prev_cost - b * eps * deltaJ) or (eps < c ** self._prob.conv_criterion):
            # if (worst_cost <=  prev_cost - b * eps * deltaJ):
                self.v_log(eps)
                if (eps < c ** self._prob.conv_criterion):
                    converged = True
                    return x_traj, u_traj, worst_dist, worst_cost, converged #do we return the previous worst disturbance and cost?
                break

            eps *= c

        traj_nom = self.rollout(x_traj=x_traj, u_traj=u_traj, dist=worst_dist, 
                        k_traj=k_traj, K_traj=K_traj, eps=eps, initial=False)  
        x_traj_nom = traj_nom.x_traj
        u_traj_nom = traj_nom.u_traj

        return x_traj_nom, u_traj_nom, worst_dist, worst_cost, converged

    def backward_pass(self, x_traj, u_traj):
        self.v_log("backward pass")
        alpha = self._alpha_curr
        
        horizon = self._prob.horizon
        reset = int(self._prob.reset_prop * horizon)

        num_states = self._prob.num_states
        num_inputs = self._prob.num_inputs
        
        d_nom = self._prob.d_nom
        d_set = self._prob.d_set
        n_d = self._prob.n_d

        val_func_grad = np.zeros((num_states, horizon + 2))         # V_x array, intialized to 0
        val_func_hess = np.zeros((num_states, num_states, horizon + 2))         # V_xx array, intialized to 0
        delta_val_func = np.zeros(horizon + 2)

        k_traj = np.zeros((num_inputs, horizon))               # array of k (lowercase) values, init to all 0
        K_traj = np.zeros((num_inputs, num_states, horizon)) # array of K (capital) values, init to all 0
        

        #TODO: remove this next line, just use delta_val_func to calculate deltaJ
        delta_val_func[-1] = next_delta_val_func = self._prob.phi(x_traj[:,-1])          #TODO This is wrong  # The terminal value function is just l_f(x_N)
        
        val_func_grad[:,-1] = next_grad = self._prob.phix(x_traj[:,-1]) 
        val_func_hess[:,:,-1] = next_hess = self._prob.phixx(x_traj[:,-1])

        deltaJ = 0 

        for i in reversed(range(reset + 1, horizon + 1)):
            x = x_traj[:,i]
            u = u_traj[:,i - 1]

            fx = self._prob.fx(x,u)
            fu = self._prob.fu(x,u).reshape(num_states, num_inputs)

            lx = self._prob.lx(x,u)
            lu = self._prob.lu(x,u)
            lxx = self._prob.lxx(x,u)
            lux = self._prob.lux(x,u).reshape(num_inputs, num_states)
            luu = self._prob.luu(x,u).reshape(num_inputs, num_inputs)
            # if not self._prob.quad_cost:
            #     luu = self._prob.L_uu

            Qx = lx + fx.T @ next_grad
            Qu = lu + fu.T @ next_grad
            Qxx = lxx + fx.T @ next_hess @ fx  #+ next_grad @ fxx
            Qux = lux + fu.T @ next_hess @ fx  #+ next_grad @ fux
            Quu = luu + fu.T @ next_hess @ fu  + self._regularization_curr * np.identity(num_inputs)#+ next_grad @ fuu this last part is some sort of tensor product idk how to deal with
            
            k_traj[:,i - 1] = k = -np.linalg.pinv(Quu) @ Qu
            # print('Quu', Quu)
            # print('unregged Quu', Quu - self._regularization_curr * np.identity(num_inputs))
            # print('Qu',Qu)
            K_traj[:,:,i - 1] = K = -np.linalg.pinv(Quu) @ Qux

            delta_val_func[i] = next_delta_val_func = next_delta_val_func - 1/2 * k.T @ Quu @ k
            val_func_grad[:,i] = next_grad = Qx - K.T @ Quu @ k
            val_func_hess[:,:,i] = next_hess = Qxx - K.T @ Quu @ K

            deltaJ -= 0.5 * Qu.T @ k


        px = self._prob.px(x_traj[:, reset], d_nom)
        pd = self._prob.pd(x_traj[:, reset], d_nom).reshape(self._prob.num_states, self._prob.num_disturbances)

        G = np.zeros((n_d,))
        DG = np.zeros((n_d, num_states))

        for i in range(n_d):
            d = d_set[i]
            G[i] = 1/2 * d.T @ pd.T @ next_hess @ pd @ d + next_grad.T @ pd @ d
            DG[i,:] = (d @ pd.T @ next_hess @ px + next_grad.T @ px).reshape(-1).T

        #TODO Vectorize:
        # G = (0.5 * (dSet @ Pd.T @ next_hess @ Pd + next_grad @ Pd.T).sum(axis=1)).T
        # DG = ((dSet @ Pd.T @ next_hess @ Px + next_grad @ Px).reshape(-1, system_model.numStates))
        
        SM = softmax(alpha * G)
        V = 1/alpha * logsumexp(alpha * G)
        


        delta_val_func[reset] = delta_val_func[reset + 1] + V #TODO fix this (need better estimate of expected cost reduction)
        val_func_grad[:,reset] = DG.T @ SM

        beta = 0
        for i in range(n_d):
            denom = 2*(V-G[i]) + self._prob.beta_regularization_eps
            beta = max(beta, (np.linalg.norm(DG[i, :]  - val_func_grad[:,reset]) ** 2)/denom)

        self.v_log(f"beta: {beta}")
        
        val_func_hess[:,:,reset] = alpha * DG.T @ (np.diag(SM) - SM @ SM.T) @ DG + px.T @ next_hess @ px + beta*np.eye(num_states, num_states)

        for i in reversed(range(reset)):
            x = x_traj[:,i]
            u = u_traj[:,i - 1]

            fx = self._prob.fx(x,u)
            fu = self._prob.fu(x,u).reshape(self._prob.num_states, self._prob.num_inputs)

            lx = self._prob.lx(x,u)
            lu = self._prob.lu(x,u)
            lxx = self._prob.lxx(x,u)
            lux = self._prob.lux(x,u).reshape(self._prob.num_inputs, self._prob.num_states)
            luu = self._prob.luu(x,u).reshape(self._prob.num_inputs, self._prob.num_inputs)

            Qx = lx + fx.T @ next_grad
            Qu = lu + fu.T @ next_grad
            Qxx = lxx + fx.T @ next_hess @ fx  #+ next_grad @ fxx
            Qux = lux + fu.T @ next_hess @ fx  #+ next_grad @ fux
            Quu = luu + fu.T @ next_hess @ fu  #+ next_grad @ fuu this last part is some sort of tensor product idk how to deal with
            
            
            k_traj[:,i] = k = -np.linalg.pinv(Quu) @ Qu
            K_traj[:,:,i] = K = -np.linalg.pinv(Quu) @ Qux

            delta_val_func[i] = next_delta_val_func = next_delta_val_func - 1/2 * k.T @ Quu @ k
            val_func_grad[:,i] = next_grad = Qx - K.T @ Quu @ k
            val_func_hess[:,:,i] = next_hess = Qxx - K.T @ Quu @ K

            deltaJ -= 0.5 * Qu.T @ k
            
        return k_traj, K_traj, deltaJ
    
    def getSolution(self):
        assert self._solution != None, "The controller has not been solved the problem"
        return self._solution
    
    def evalDiscrete(self, x_traj, u_traj, K_traj, num_test_points=15): 
        distGrid = self._prob.d_nom + np.linspace(min(self._prob.d_set), max(self._prob.d_set), num_test_points) #eventually this should be a grid over the convex hull of the dsiturbance vectors
        costs = np.zeros((num_test_points,))
        #control_costs = np.zeros((num_test_points,))
        trajs = [None] * num_test_points
        for i in range(num_test_points):
            trajs[i] = self.rollout(x_traj=x_traj, u_traj=u_traj, dist=distGrid[i], 
                    k_traj=np.zeros_like(u_traj), K_traj=K_traj, eps=1, initial=False)
            costs[i] = trajs[i].cost
            #control_costs[i] = np.sum(np.array([self._prob._cost.l_input(np.array([ui])) for ui in trajs[i].u_traj[0]]))

        return distGrid, trajs, costs
   
    def evalDiscreteSolved(self, num_test_points=15): #Uses the solution that it has already found
        assert self._solution, "The RHDDP problem has not yet been solved"
        return self.evalDiscrete(x_traj=self._solution.get("x_traj"),
                                 u_traj=self._solution.get("u_traj"),
                                 K_traj=self._solution.get("K_traj"),
                                 num_test_points=num_test_points)

    def evalDiscrete2d(self, x_traj, u_traj, K_traj, num_test_points=15):
        d_set = self._prob.d_nom + self._prob.d_set
        
        # Automatically find the minimum and maximum values of the disturbance vectors
        d_min = np.min(d_set, axis=0)
        d_max = np.max(d_set, axis=0)

        num_d0_points = num_test_points
        num_d1_points = num_test_points

        d0_values = np.linspace(d_min[0], d_max[0], num_d0_points)
        d1_values = np.linspace(d_min[1], d_max[1], num_d1_points)

        distGrid = np.array([(d0, d1) for d0 in d0_values for d1 in d1_values])

        costs = np.zeros((num_test_points ** 2,))
        #control_costs = np.zeros((num_test_points ** 2,))
        trajs = [None] * (num_test_points ** 2)

        for i in range(num_test_points ** 2):
            trajs[i] = self.rollout(x_traj=x_traj, u_traj=u_traj, dist=distGrid[i], 
                    k_traj=np.zeros_like(u_traj), K_traj=K_traj, eps=1, initial=False)
            costs[i] = trajs[i].cost
            #control_costs[i] = np.sum(np.array([self._prob._cost.l_input(np.array([ui])) for ui in trajs[i].u_traj[0]]))

        # # Reshape the data for 3D plotting
        # distGrid = distGrid.reshape((num_d0_points, num_d1_points, 2)) 
        D0, D1 = np.meshgrid(d0_values, d1_values)
        costs = costs.reshape((num_d0_points, num_d1_points))
        #control_costs = control_costs.reshape((num_d0_points, num_d1_points))

        return distGrid, trajs, costs, D0, D1
    
    def get_solved_trajectory(self, disturbance):
        assert self._solution, "Problem is not yet solved"
        x_traj = self._solution.get("x_traj")
        u_traj = self._solution.get("u_traj")
        K_traj = self._solution.get("K_traj")

        rolled_out_traj = self.rollout(x_traj=x_traj, u_traj=u_traj, k_traj=u_traj, dist=disturbance,
                                       K_traj=K_traj, eps=0, initial=False)
        return rolled_out_traj.x_traj, rolled_out_traj.u_traj, rolled_out_traj.cost
        

    def v_log(self, str):
        if self._verbose: print(str)

