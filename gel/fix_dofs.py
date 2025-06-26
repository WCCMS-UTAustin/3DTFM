"""Base implementation of the function to restrict the control variables
of the inverse model to only those in the event horizon.

Generalized usage is also possible, this function will be overriden in
`gel.fix_dofs_overloaded` with appropriate adjoint information for use
with automatic differentiation.

Credit https://fenicsproject.discourse.group/t/specific-dof-of-a-function-as-control-variable/3300/3"""
from fenics import *
from fenics_adjoint import *


def fix_dofs(func, fixed_indexes, fixed_values):             
    """Forces DoF values of a FE function to specific values.

    * `func` is a FEniCS function in a space with DoFs
    * `fixed_indexes` is an iterable of integer local DoF indices
    * `fixed_values` is an iterable of the same size as `fixed_indexes`
    with the float values to set the DoFs to.
    """
    new_fun = Function(func.function_space(), func.vector())
    for idx, val in zip(fixed_indexes, fixed_values):
        new_fun.vector().vec().setValueLocal(idx, val)
    new_fun.vector().apply("")
    return new_fun

