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
        self.prev_soln = self._intercepted_solve(*args, **kwargs)
        df.Function.assign(self.block_var_to_save.output, self.prev_soln)
        self.block_var_to_save.save_output() # NEED TO SET LATER
        return self.prev_soln

    NonlinearVariationalSolveBlock._forward_solve = new_forward_solve

    SolveBlock._intercepted_create_initial_guess = SolveBlock._create_initial_guess

    def new_init_guess(self):
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
rank = MPI.comm_world.rank
if rank != 0:
    set_log_level(50)  # Mute output


do_nothing_print = lambda *args : None # Do nothing

