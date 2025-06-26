"""Import statements common to all source files on the backend.

If an environment variable named `SUPPRESS_ADJOINT` is set in the Python
process importing this module, then dolfin will not be overridden with
dolfin_adjoint. This is useful for saving memory if one is merely
performing forward solves.

# Passing Forward Model Solutions as Guesses

For the inverse model, in order to pass previous displacement solutions
as initial guesses between L-BFGS iterations, a bit of a hack is
required. FEniCS-adjoint is designed to automatically figure out an
appropriate way to (1) resolve and (2) solve the adjoint problem after
an initial forward solve. However, this does not easily permit changing
how solves are performed across multiple resolves. So, we must
intercept default nonlinear solve block behavior to use previous
solutions as initial guesses. Implemented wrappers include:
* `new_forward_solve` 
* `new_init_guess` 

# API
"""
import os
from dolfin import *


if "SUPPRESS_ADJOINT" not in os.environ:
    from dolfin_adjoint import *
    import dolfin as df
    # Make previous guesses get passed
    from fenics_adjoint.solving import SolveBlock
    from fenics_adjoint.variational_solver import NonlinearVariationalSolveBlock

    NonlinearVariationalSolveBlock._intercepted_solve = NonlinearVariationalSolveBlock._forward_solve

    def new_forward_solve(self, *args, **kwargs):
        """Wrapper for _forward_solve to save output."""
        self.prev_soln = self._intercepted_solve(*args, **kwargs)
        df.Function.assign(self.block_var_to_save.output, self.prev_soln)
        self.block_var_to_save.save_output() # NEED TO SET LATER
        return self.prev_soln

    NonlinearVariationalSolveBlock._forward_solve = new_forward_solve

    SolveBlock._intercepted_create_initial_guess = SolveBlock._create_initial_guess

    def new_init_guess(self):
        """Wrapper for _create_initial_guess returning saved soln."""
        if not hasattr(self, "prev_soln"):
            return self._intercepted_create_initial_guess()
        else:
            return self.prev_soln

    SolveBlock._create_initial_guess = new_init_guess


import pyadjoint as pa
import numpy as np
import sys
import meshio


comm = MPI.comm_world
"""MPI comm interface"""
rank = MPI.comm_world.rank
"""MPI rank number"""
if rank != 0:
    set_log_level(50)  # Mute output


do_nothing_print = lambda *args : None # Do nothing
"""A substitute for print and logging functions that does nothing."""

