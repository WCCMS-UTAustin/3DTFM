"""Credit https://fenicsproject.discourse.group/t/specific-dof-of-a-function-as-control-variable/3300/3"""
from fenics import *
from fenics_adjoint import *


def fix_dofs(func, fixed_indexes, fixed_values):             
    new_fun = Function(func.function_space(), func.vector())
    for idx, val in zip(fixed_indexes, fixed_values):
        new_fun.vector().vec().setValueLocal(idx, val)
    new_fun.vector().apply("")
    return new_fun

