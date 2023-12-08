import numpy as np
import importlib
from abc import abstractmethod

class Dynamics():
    @abstractmethod      
    def f(self, x,u):
        pass
    
    @abstractmethod
    def fx(self, x, u):
        pass
    
    @abstractmethod
    def fu(self, x, u):
        pass
    

    @abstractmethod
    def p(self, x, d):
        pass
    
    @abstractmethod
    def px(self, x, d):
        pass
    
    @abstractmethod
    def pd(self, x, d):
        pass
    
    #TODO: Need to figure out what to do with this reset condition here.
    # @abstractmethod
    # def resetCondition(self, x, u, k) -> bool:
    #     pass

    @property
    @abstractmethod
    def num_states(self):
        pass

    @property
    @abstractmethod
    def num_inputs(self):
        pass

    @property
    @abstractmethod
    def num_disturbances(self):
        pass

    @property
    @abstractmethod
    def name_str(self):
        pass
    


class SymbolicDynamics():
    def __init__(self, name: str, num_states, num_inputs, num_disturbances, codegen_dir: str='gen'):
        self._name_str = name
        # self._fvec_cont_gen = importlib.import_module(codegen_dir + '.' + name + '.fvec_continuous').fvec_continuous
        # self._gvec_cont_gen = importlib.import_module(codegen_dir + '.' + name + '.gvec_continuous').gvec_continuous

        self._f_gen = importlib.import_module(codegen_dir + '.' + name + '.f').f
        self._fx_gen = importlib.import_module(codegen_dir + '.' + name + '.fx').fx
        self._fu_gen = importlib.import_module(codegen_dir + '.' + name + '.fu').fu

    
        self._p_gen = importlib.import_module(codegen_dir + '.' + name + '.p').p
        self._px_gen = importlib.import_module(codegen_dir + '.' + name + '.px').px
        self._pd_gen = importlib.import_module(codegen_dir + '.' + name + '.pd').pd


        #TODO: Need to figure out the reset criterion. By timestep will be kind of weird 
        # self._reset_condition_gen = ...  

        self._num_inputs = num_inputs
        self._num_states = num_states
        self._num_disturbances = num_disturbances

    def f(self, x,u):
        return self._f_gen(x,u)
    
    def fx(self, x, u):
        return self._fx_gen(x,u)
    
    def fu(self, x, u):
        return self._fu_gen(x,u)
        #return np.reshape(self._fu_gen(x,u), (self.num_states,self._num_inputs))
    
    def p(self, x, d):
        return self._p_gen(x, d)
    
    def px(self, x, d):
        return self._px_gen(x, d)
    
    def pd(self, x, d):
        return self._pd_gen(x, d)
    
    #TODO: Need to figure out what to do with this reset condition here.
    # def resetCondition(self, x, u, k) -> bool:
    #     return self._reset_condition_gen(x, u, k)
    
    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_states(self):
        return self._num_states

    @property
    def num_disturbances(self):
        return self._num_disturbances
    
    @property
    def name_str(self):
        return self._name_str
    

#TODO: Sort out the Control Affine Model Dynamics

# class SymbolicControlAffineModel:
#     def __init__(self, name: str, codegen_dir: str='gen'):
#         self.name_str = name
#         self._fvec_cont_gen = importlib.import_module(codegen_dir + '.' + name + '.fvec_continuous').fvec_continuous
#         self._gvec_cont_gen = importlib.import_module(codegen_dir + '.' + name + '.gvec_continuous').gvec_continuous
        
#         self._fx_gen = importlib.import_module(codegen_dir + '.' + name + '.fx').fx
#         self._fu_gen = importlib.import_module(codegen_dir + '.' + name + '.fu').fu        

#     def fx(self, x, u):
#         return self._fx_gen(x, u)

#     def fu(self, x, u):
#         return self._fu_gen(x)

#     def dx(self, x, u):
#         fx = self._fvec_cont_gen(x)[..., None]
#         gx = self._gvec_cont_gen(x)
#         gx = gx[..., None] if gx.ndim == 1 else gx
#         return fx + gx @ u[..., None]