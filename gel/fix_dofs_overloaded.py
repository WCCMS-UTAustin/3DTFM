"""Overriden implementation of control variable restriction function for
automatic differentiation in pyadjoint.

Overrides `gel.fix_dofs.fix_dofs` with a Block object.

Credit https://fenicsproject.discourse.group/t/specific-dof-of-a-function-as-control-variable/3300/3"""
from fenics import *
from fenics_adjoint import *
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

from .fix_dofs import fix_dofs


_backend_fix_dofs = fix_dofs

class FixedDofsBlock(Block):
    """Tape Block for `gel.fix_dofs.fix_dofs`

    Describes how to recompute and how to compute the adjoint.
    """

    def __init__(self, func, fixed_indexes, fixed_values, **kwargs):
        """Take in arguments to the function and kwargs for adjoint."""
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
        """Zero-out the components of fixed DoFs for derivative.

        * `inputs` is the list of inputs to `gel.fix_dofs.fix_dofs`, ie.
        index 0 has the function.
        * `adj_inputs` is a list with the same number of items as
        `inputs`, has the values of the adjoint input
        * Other inputs unused.

        Returns: the resulting product accounting for fixed DoFs
        """
        adj_input = adj_inputs[0] 
        x = inputs[0].vector() 

        output = adj_input
        # the derivative of the fixed dofs is null
        output[self.fixed_indexes] = 0.0 
        return output
    
    def recompute_component(self, inputs, block_variable, idx, prepared):
        """Re-fixes the DoFs

        * `inputs` is the list of inputs to `fix_dofs` originally 
        encountered
        * Rest of inputs unused

        Returns: the DoF-fixed function.
        """
        return _backend_fix_dofs(
            inputs[0],
            self.fixed_indexes,
            self.fixed_values
        ) 

fix_dofs = overload_function(fix_dofs, FixedDofsBlock)
"""Forces DoF values of a FE function to specific values.

Is equipped for automatic differentiation.

* `func` is a FEniCS function in a space with DoFs
* `fixed_indexes` is an iterable of integer local DoF indices
* `fixed_values` is an iterable of the same size as `fixed_indexes`
with the float values to set the DoFs to.
"""

