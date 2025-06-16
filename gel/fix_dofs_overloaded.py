"""Credit https://fenicsproject.discourse.group/t/specific-dof-of-a-function-as-control-variable/3300/3"""
from fenics import *
from fenics_adjoint import *
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

from .fix_dofs import fix_dofs


backend_fix_dofs = fix_dofs

class FixedDofsBlock(Block):
    def __init__(self, func, fixed_indexes, fixed_values, **kwargs):
        super(FixedDofsBlock, self).__init__() 
        self.kwargs = kwargs
        self.add_dependency(func)

        # variables that do not appear in the dependencies but are still used
        # for the computation 
        self.fixed_indexes = fixed_indexes 
        self.fixed_values = fixed_values
        
    def __str__(self): 
        return 'FixedDofsBlock'

    def evaluate_adj_component(self,
            inputs,
            adj_inputs,
            block_variable,
            idx,
            prepared=None
        ):
        adj_input = adj_inputs[0] 
        x = inputs[0].vector() 

        output = adj_input
        # the derivative of the fixed dofs is null
        output[self.fixed_indexes] = 0.0 
        return output
    
    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_fix_dofs(
            inputs[0],
            self.fixed_indexes,
            self.fixed_values
        ) 

fix_dofs = overload_function(fix_dofs, FixedDofsBlock)

