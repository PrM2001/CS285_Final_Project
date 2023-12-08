import importlib
from abc import abstractmethod

class Cost:
    @abstractmethod
    def l(self, x, u):
        pass
    
    @abstractmethod
    def lx(self, x, u):
        pass
    
    @abstractmethod
    def lu(self, x, u):
        pass
    
    @abstractmethod
    def lxx(self, x, u):
        pass
    
    @abstractmethod
    def lux(self, x, u):
        pass
    
    @abstractmethod
    def luu(self, x, u):
        pass

    @abstractmethod
    def phi(self, x_t):
        pass
    
    @abstractmethod
    def phix(self, x_t):
        pass
    
    @abstractmethod
    def phixx(self, x_t):
        pass

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
    def name_str(self):
        pass

class SymbolicCost(Cost):
    def __init__(self, name: str, num_states, num_inputs, codegen_dir: str='gen') -> None:
        self._name_str = name

        self._l_gen = importlib.import_module(codegen_dir + '.' + name + '.l').l
        self._lx_gen = importlib.import_module(codegen_dir + '.' + name + '.lx').lx
        self._lu_gen = importlib.import_module(codegen_dir + '.' + name + '.lu').lu
        self._lxx_gen = importlib.import_module(codegen_dir + '.' + name + '.lxx').lxx
        self._lux_gen = importlib.import_module(codegen_dir + '.' + name + '.lux').lux
        self._luu_gen = importlib.import_module(codegen_dir + '.' + name + '.luu').luu

        self._phi_gen = importlib.import_module(codegen_dir + '.' + name + '.phi').phi
        self._phix_gen = importlib.import_module(codegen_dir + '.' + name + '.phix').phix
        self._phixx_gen = importlib.import_module(codegen_dir + '.' + name + '.phixx').phixx

        
        self._num_inputs = num_inputs
        self._num_states = num_states

    def l(self, x, u):
        return self._l_gen(x, u)

    def lx(self, x, u):
        return self._lx_gen(x, u)
    
    def lu(self, x, u):
        return self._lu_gen(x, u)

    def lxx(self, x, u):
        return self._lxx_gen(x, u)
    
    def lux(self, x, u):
        return self._lux_gen(x, u)
    
    def luu(self, x, u):
        return self._luu_gen(x, u)

    def phi(self, x_t):
        return self._phi_gen(x_t)
    
    def phix(self, x_t):
        return self._phix_gen(x_t)
    
    def phixx(self, x_t):
        return self._phixx_gen(x_t)
    
    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_states(self):
        return self._num_states
    
    @property
    def name_str(self):
        return self._name_str
    
    
