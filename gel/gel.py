from .header import *

from inspect import signature
from .fix_dofs_overloaded import fix_dofs

from .geometry import *
from .kinematics import *
from .mechanics import *
from .helper import *
from .objective import *


_cell_name_cache = dict()
def read_cell_name(cell_data_dir):
    """Reads a cell name from a directory with caching."""
    ret = None # Init

    if rank == 0:
        # Determine full path in case of moving current directory
        fullpath = os.path.realpath(cell_data_dir)
        if fullpath in _cell_name_cache:
            ret = _cell_name_cache[fullpath]
        else:
            # We must read the file.
            with open(os.path.join(cell_data_dir, "name.txt"), "r") as fd:
                name = fd.readline().strip()
            _cell_name_cache[fullpath] = name
            ret = name

    ret = comm.bcast(ret, root=0)

    return ret


_geo_type_cache = dict()
def read_geo_type(cell_data_dir):
    """Reads a geometry type from a directory with caching."""
    ret = None # Init

    if rank == 0:
        # Determine full path in case of moving current directory
        fullpath = os.path.realpath(cell_data_dir)
        if fullpath in _geo_type_cache:
            ret = _geo_type_cache[fullpath]
        else:
            # We must read the file.
            with open(os.path.join(cell_data_dir, "geo_type.txt"), "r") as fd:
                geo_type = fd.readline().strip()
            _geo_type_cache[fullpath] = geo_type
            ret = geo_type

    ret = comm.bcast(ret, root=0)
    
    return ret


def apply_fixes(
        geo,
        mod_repr,
        fillval=0.0
    ):
    # Construct fixed indices first
    free_indices = set()
    dof_map = geo.V0.dofmap()
    my_first, my_last = dof_map.ownership_range()
    num_dof_this_proc = my_last - my_first

    # Iterate through cells in the mesh
    for region_ind, (ci, cell) in zip(
        geo.dx.subdomain_data().array(),
        enumerate(geo.mesh.cells())
    ):
        # See if the element is in a detectable region
        if region_ind == geo.DETECTABLE_U:
            # Mark each point on that cell as free
            for dof in dof_map.cell_dofs(ci):
                free_indices.add(int(dof))

    # Remainder should be fixed
    fix_indices = set(range(num_dof_this_proc)) - free_indices
    fix_indices = np.array(list(fix_indices))

    fixed_mod_repr = fix_dofs(
        mod_repr,
        fix_indices,
        fillval*np.ones(fix_indices.shape, dtype=float)
    )

    return fixed_mod_repr


def get_variational_stuff(kinematics, Pi):
    """
    Given the kinematics and expression for energy, returns forms for the
    variational problem.

    Returns:
    * dPi
    * ddPi
    """
    geo = kinematics.geo

    du = TrialFunction(geo.V)            # Incremental displacement
    v  = TestFunction(geo.V)             # Test function

    # Compute first variation of Pi 
    # (directional derivative about u in the direction of w)
    dPi = derivative(Pi, kinematics.u, v)
    ddPi = derivative(dPi, kinematics.u, du)

    return dPi, ddPi


def run_forward(
    kinematics,
    dPi,
    ddPi,
    solver_parameters,
    ffc_options,
    load_steps=1,
    post_load_step_callback=None
):
    """
    Given the kinematics, expressions for the variational problem, and solver
    settings (including how many load_steps to split into), performs the forward
    solve.

    May provide function to call with every chunk. Note it is called after each
    chunk.

    Main side effect is to modify kinematics.u with the solution.
    """
    geo = kinematics.geo

    # Solver loop (solves the forward problem in load_steps)
    for i in range(load_steps):
        sys.stdout.flush()  

        if hasattr(geo, "scalar"):
            # updates the boundary conditions surface displacements
            geo.scalar = (i+1)/load_steps
            geo.update_bcs()

        # Be careful of load stepping and adjoint annotation!
        stop_tape = False
        if i != load_steps-1:
            stop_tape = True

        if stop_tape:
            pa.pause_annotation()

        # Solve variational problem
        nvp = NonlinearVariationalProblem(
            dPi,
            kinematics.u,
            geo.bcs,
            J=ddPi,
            form_compiler_parameters=ffc_options
        )
        nvs = NonlinearVariationalSolver(nvp)
        nvs.parameters.update(solver_parameters)
        nvs.solve()

        if stop_tape:
            pa.continue_annotation()

        if post_load_step_callback is not None:
            post_load_step_callback()


# There is no reason to record what happends here for inverse model
@pa.tape.no_annotations
def output_results_paraview(
        output_folder,
        kinematics,
        mod_repr,
        bx_inds=None,
        name="simulation_output"
    ):
    """
    Given kinematics with stuff having been computed, a directory to put the
    results, and mod_repr as used, saves mesh with solution info.

    If given a list of boundary indicators, will output that as well.

    May override default name simulation_output with name

    Output files:
    * {name}.xdmf
    * {name}.h5
    """
    geo = kinematics.geo

    # Helper
    rp = ghrp(output_folder)

    # Create output file for homogeneous forward model run
    simulation_output_file = XDMFFile(rp(f"{name}.xdmf"))
    simulation_output_file.parameters["flush_output"] = True
    simulation_output_file.parameters["functions_share_mesh"] = True

    # Renaming variables
    kinematics.u.rename("u","displacement")
    mod_repr.rename("mod_repr","representation of modulus")

    # Rank indication
    rank_ind = Function(geo.V0)
    size = rank_ind.vector().local_size()
    rank_ind.vector().set_local(rank*np.ones(size))
    rank_ind.vector().apply("")
    rank_ind.rename("rank","process ownership")

    # Writes out the variables
    simulation_output_file.write(kinematics.u,0)
    simulation_output_file.write(mod_repr,0)
    simulation_output_file.write(rank_ind,0)

    # Boundary indication
    if bx_inds:
        for name, bx_ind in bx_inds.items():
            # Evaluate expression, get closest fit we can (L2) with DoFs
            indicator = project(bx_ind, geo.V0)
            # Output result
            indicator.rename(f"boundary_{name}", "High value on {name}")
            simulation_output_file.write(indicator, 0)

    # Full shape for creating synthetic exact target
    shape_fname = rp("u_full_shape.xdmf")
    save_shape(kinematics.u, shape_fname, "u")

    # Full shape for multiple regularization stages
    shape_fname = rp("mod_repr_full_shape.xdmf")
    save_shape(mod_repr, shape_fname, "mod_repr")


class ForwardSimulation:

    def __init__(self,
            geometry_type,
            d1c1=1,
            material_model="beta",
            mod_repr_init="zero",
            vprint=do_nothing_print,
            mu_ff=MU_FF_DEFAULT,
            load_steps=1,
            restrict_ctl_dofs_to_veh=False,
            pc="hypre_amg",
            tola=1e-10,
            tolr=1e-9,
            traction=None,
            formulation_kwargs=dict(),
            **kwargs
        ):
        # Validate, get geometry
        if traction is not None:
            # Update kwargs -> no Dirichlet BC allowed
            if "suppress_cell_bc" in kwargs:
                if kwargs["suppress_cell_bc"] == False:
                    raise ValueError("Must suppress cell BC for traction")

            kwargs["suppress_cell_bc"] = True
        self.geo = self._geo_from_kwargs(geometry_type, kwargs)

        self.traction = traction
        if traction is not None:
            self.traction = Function(self.geo.V)
            t_file = XDMFFile(traction)
            t_file.read_checkpoint(self.traction, "T", 0)

        self.d1c1 = d1c1
        self.material_model = material_model
        self.mod_repr_init = mod_repr_init
        self.vprint = vprint
        self.mu_ff = mu_ff
        self.load_steps = load_steps
        self.restrict_ctl_dofs_to_veh = restrict_ctl_dofs_to_veh
        self.pc = pc
        self.tola = tola
        self.tolr = tolr
        self.formulation_kwargs = formulation_kwargs

    def _geo_from_kwargs(self, geometry_type, kwargs):
        # Parse geometry
        if geometry_type == "real_cell_gel":
            geo_cls = Geometry
        elif geometry_type == "beam":
            geo_cls = BeamGeometry
        else:
            raise ValueError(f"{geometry_type} unknown")

        #
        # Extract arguments
        #
        sig = signature(geo_cls)

        # Iterate through known acceptable arguments
        cls_args_pos_only = [ ]
        cls_args_kwarg = dict()
        for key, param in sig.parameters.items():
            # Validate
            if param.default is param.empty:
                # Must be required
                if key not in kwargs:
                    raise ValueError(
                        f"{key} must be provided to {geometry_type}"
                    )

            # Incorporate if present
            if key in kwargs:
                if param.kind == param.POSITIONAL_ONLY:
                    cls_args_pos_only.append(kwargs[key])
                else:
                    cls_args_kwarg[key] = kwargs[key]

        # Create instance
        return geo_cls(*cls_args_pos_only, **cls_args_kwarg)

    def run_forward(self):
        # Get easier access
        geo = self.geo
        formulation = self.material_model
        d1c1 = self.d1c1
        mod_repr_init = self.mod_repr_init
        vprint = self.vprint

        # Do the normal things
        kinematics = Kinematics(geo)

        # Define material
        mu_ff = self.mu_ff
        vprint(f"Using D1/C1={d1c1}")
        nu = get_nu_for_target_k(d1c1)
        vprint(f"Measured D1/C1 off of nu, found {get_k_for_target_nu(nu)}")
        vprint(f"Using mu={mu_ff} Pa")
         
        B  = Constant((0.0, 0.0, 0.0))  # Body force per unit volume
        # Traction force on the boundary per area in reference
        if self.traction is None:
            T = Constant((0.0, 0.0, 0.0))
        else:
            T = self.traction

        if isinstance(mod_repr_init, str):
            if mod_repr_init in MOD_REPR_FIELDS:
                # Case lookup
                self.ctl = MOD_REPR_FIELDS[mod_repr_init](geo)
            else:
                # Case read from file
                self.ctl = Function(geo.V0)

                mod_repr_file = XDMFFile(mod_repr_init)
                mod_repr_file.read_checkpoint(self.ctl, "mod_repr", 0)

                self.ctl.set_allow_extrapolation(True)
        else:
            # Case given the function
            self.ctl = mod_repr_init

        self.ctl.rename("ctl", "ctl")
        if self.restrict_ctl_dofs_to_veh:
            mod_repr = apply_fixes(
                geo,
                self.ctl,
                fillval=(0.0 if (formulation in ZERO_FIX_F) else 1.0)
            )
        else:
            mod_repr = self.ctl

        Pi = get_energy(
            kinematics,
            mu_ff,
            nu,
            mod_repr,
            B,
            T,
            formulation=formulation,
            **self.formulation_kwargs
        )

        # Solve
        solver_parameters = {
            "newton_solver" : {
                "absolute_tolerance" : self.tola,
                "relative_tolerance" : self.tolr,
                "linear_solver" : "gmres",
                "preconditioner" : self.pc,
                "maximum_iterations" : 8,
                "error_on_nonconvergence" : False,
                #"krylov_solver" : {
                    #"absolute_tolerance" : 0.1*self.tola,
                    #"relative_tolerance" : 1e-4
                #}
            }
        }

        ffc_options = {
            "optimize": True,
            "eliminate_zeros": True,
            "precompute_basis_const": True,
            "precompute_ip_const": True
        }

        vprint(f"solver_parameters = {solver_parameters}")
        vprint(f"ffc_options = {ffc_options}")

        dPi, ddPi = get_variational_stuff(kinematics, Pi)
        solver_fn = lambda : run_forward(
            kinematics,
            dPi,
            ddPi,
            solver_parameters,
            ffc_options,
            load_steps=self.load_steps
        )
        solver_fn()

        # Save results
        self.kinematics = kinematics
        self.mod_repr = mod_repr
        self.solver_fn = solver_fn
        self.mechanics = Mechanics(
            kinematics,
            formulation,
            mu_ff,
            d1c1,
            mod_repr
        )

