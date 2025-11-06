"""Interface to forward solving functionality and file management."""
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
    """Returns str name from directory str `cell_data_dir` with caching.

    Will look for a file "name.txt" under provided directory and read
    the first line.
    """
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
    """Returns str geometry type from directory str `cell_data_dir` w/ caching.

    Will look for file "geo_type.txt" under provided directory and read
    the first line.
    """
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
        ctl,
        fillval=0.0
    ):
    """Fixes DoFs beyond the event horizon to a fixed value.

    Uses the subdomain data in `geo` (an object of type
    `gel.geometry.Geometry`) to set DoFs of FEniCS function `ctl` to a
    uniform value of `fillval` (default 0.0) in a new function.

    Returns a new, fixed version of `ctl`, the modulus representation,
    with adjoint tracing.
    """
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
        ctl,
        fix_indices,
        fillval*np.ones(fix_indices.shape, dtype=float)
    )

    return fixed_mod_repr


def get_variational_stuff(kinematics, Pi):
    r"""Returns the right-, left-hand sides of the variational problem.

    It is defined in here, as opposed to `gel.mechanics`, because the
    operators needed to solve a nonlinear problem must be invented
    according to the solve technique; they are not intrinsic to the
    problem. We use a consistent-tangent Newton-Raphson solver, so we
    need the linearized tangent operator and a right-hand-side.

    * `kinematics`: object of type `gel.kinematics.Kinematics` through
    which to differentiate with respect to displacements
    * `Pi`: expression for energy, likely an output from `gel.mechanics`

    Returns:
    * First derivative $\frac{d\Pi}{d\mathbf{u}}$ (RHS)
    * Second derivative $\frac{d^2\Pi}{d\mathbf{u}^2}$ (LHS)
    """
    geo = kinematics.geo

    du = TrialFunction(geo.V)            # Incremental displacement
    v  = TestFunction(geo.V)             # Test function

    # Compute first variation of Pi 
    # (directional derivative about u in the direction of w)
    dPi = derivative(Pi, kinematics.u, v)
    ddPi = derivative(dPi, kinematics.u, du)

    return dPi, ddPi


# There is no reason to record what happends here for inverse model
@pa.tape.no_annotations
def output_results_paraview(
        output_folder,
        kinematics,
        mod_repr,
        bx_inds=None,
        name="simulation_output"
    ):
    """Outputs simulation state information as files in multiple forms.

    * `output_folder`: str path to directory where files will be created
    * `kinematics`: `gel.kinematics.Kinematics` with displacements
    * `mod_repr`: FEniCS function with control variable DoFs
    * `bx_inds`: dict, if not None it enables writing boundary tags on
    hydrogel mesh with keys the tags and items the names
    * `name`: str the name of the nodal output file

    Output files:
    * {name}.xdmf, .h5: nodal form with displacement, control variable,
    rank ownership for DoFs, and optionally boundary conditions
    * u_full_shape.xdmf, .h5: full shape/write_checkpoint form of
    displacements for easy reloading back into FEniCS
    * mod_repr_full_shape.xdmf, .h5: full shape/write_checkpoint form of
    the control variables for easy reloading back into FEniCS
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
    """Minimal-prerequisite interface to running forward simulations.

    Typical usage, an example of obtaining displacements from
    homogeneous modulus:
    ```
    sim = ForwardSimulation("real_cell_gel", data_directory="cell_data")
    sim.run_forward()
    kinematics = sim.kinematics # Contains displacements u
    ```

    Many internal variables (below) are exposed for customization before
    calls to sim.run_forward(). Be careful to keep pointers between the
    `geo`, `kinematics`, and `mechanics` objects in sync by not
    overwriting them or their depedency injections.
    """

    def __init__(self,
            geometry_type,
            d1c1=1,
            formulation="beta",
            mod_repr_init="zero",
            vprint=do_nothing_print,
            mu_ff=MU_FF_DEFAULT,
            load_steps=1,
            restrict_ctl_dofs_to_veh=False,
            pc="hypre_amg",
            tola=1e-10,
            tolr=1e-9,
            traction=None,
            u_init=None,
            formulation_kwargs=dict(),
            **kwargs
        ):
        r"""Constructor for object containing forward simulation state.

        * `geometry_type`: str tag of what type of geometry (only
        "real_cell_gel" -> `gel.geometry.Geometry` is currently
        implemented)
        * `d1c1`: float ratio $\frac{D_1}{c_1}$
        * `formulation`: str tag of material model formulation
        implemented in `gel.mechanics`
        * `mod_repr_init`: (the initial value of) the control variable for
        modulus; options in `gel.mechanics`
        * `vprint`: callable function to perform action of print or
        logging
        * `mu_ff`: float far field rheometry measurement, ie.
        $c_1=\frac{\mu}{2}$
        * `load_steps`: int number of load steps to use in forward solve
        (1 implies no stepping, just use the BC)
        * `restrict_ctl_dofs_to_veh`: bool, enables fixing DoFs outside
        event horizon for the inverse model
        * `pc`: str name of preconditioner, ie. an option listed in
        `list_krylov_solver_preconditioners()`
        * `tola`: float absolute tolerance of Newton-Raphson
        * `tolr`: float relative tolerance of Newton-Raphson
        * `traction`: str or None. If not None, the filename of a
        full-shape .xdmf traction function on the hydrogel mesh. Must
        be in 1st order Lagrange space, will ignore DoFs outside of
        surface. "zero" will automatically contruct a 0 traction
        function
        * `u_init`: str path to full-shape .xdmf file with initial
        displacements
        * `formulation_kwargs`: dict of additional kwargs to the
        material model formulation (if applicable) implemented in
        `gel.mechanics`
        * `kwargs`: dict of kwargs to the geometry, for instance
        the arguments to `gel.geometry.Geometry`
        """
        # Validate, get geometry
        if traction is not None:
            # Update kwargs -> no Dirichlet BC allowed
            if "suppress_cell_bc" in kwargs:
                if kwargs["suppress_cell_bc"] == False:
                    raise ValueError("Must suppress cell BC for traction")

            kwargs["suppress_cell_bc"] = True
        self.geo = self._geo_from_kwargs(geometry_type, kwargs)
        """Corresponding object of type `gel.geometry.Geometry`"""

        # Initialize displacements, kinematic quantities
        self.kinematics = None
        """Object of type `gel.kinematics.Kinematics` with displacement
        information
        """
        if u_init is None:
            self.kinematics = Kinematics(self.geo)
        else:
            self.kinematics = kinematics_from_file(self.geo, u_init)

        # Prepare modulus control variable
        self.ctl = create_mod_repr(self.geo, mod_repr_init)
        """A FEniCS function with the control variables for optimization,
        *cf.* `mod_repr`
        """

        self.mod_repr = None
        r"""The representation of modulus, for instance $\beta$, of type FEniCS
        function. DoFs have already been fixed if necessary in this object.
        *cf.* `ctl`.
        """
        if restrict_ctl_dofs_to_veh:
            self.mod_repr = apply_fixes(
                self.geo,
                self.ctl,
                fillval=(
                    0.0 if (formulation in ZERO_FIX_F) else 1.0
                )
            )
        else:
            self.mod_repr = self.ctl

        # Prepare mechanics
        self.mechanics = Mechanics(
            self.kinematics,
            formulation,
            mu_ff,
            d1c1,
            self.mod_repr,
            **formulation_kwargs
        )
        """Object of type `gel.mechanics.Mechanics` with material model
        information. Has an internal `mechanics.kinematics` variable
        that is the same as the ForwardSimulation's `kinematics`.
        """

        self.B = Constant((0.0, 0.0, 0.0))
        """FEniCS body force per unit volume"""

        self.T = None
        """FEniCS traction force on boundary per area in reference"""
        if traction is None:
            self.T = Constant((0.0, 0.0, 0.0))
        elif traction == "zero":
            self.T = Constant((0.0, 0.0, 0.0))
        else:
            self.T = load_shape(self.geo.V, traction, "T")

        self.solver_parameters = {
            "newton_solver" : {
                "absolute_tolerance" : tola,
                "relative_tolerance" : tolr,
                "linear_solver" : "gmres",
                "preconditioner" : pc,
                "maximum_iterations" : 8,
                "error_on_nonconvergence" : False,
                #"krylov_solver" : {
                    #"absolute_tolerance" : 0.1*self.tola,
                    #"relative_tolerance" : 1e-4
                #}
            }
        }
        """dict of parameters set in `NonlinearVariationalSolver`"""

        self.ffc_options = {
            "optimize": True,
            "eliminate_zeros": True,
            "precompute_basis_const": True,
            "precompute_ip_const": True
        }
        """dict of form compiler parameters set in
        `NonlinearVariationalProblem`
        """

        self.load_steps = load_steps
        """Number of load steps in Dirichlet BCs during single solve"""

        self.vprint = vprint
        """Optional print/logging functionality"""

    def _geo_from_kwargs(self, geometry_type, kwargs):
        # Parse geometry
        if geometry_type == "real_cell_gel":
            geo_cls = Geometry
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
        r"""Solves the forward problem with saved settings.

        Side effects: 
        * `kinematics` will store solved displacements
        * `mechanics` will contained solved displacements through the
        same `kinematics` instance
        * boundary conditions in `geo` are changed if load stepping is
        enabled
        * Uses `vprint` for logging before solve
        """
        # Optionally print information
        self.vprint(f"Using D1/C1={self.mechanics.d1c1}")
        self.vprint(f"Using far field mu={self.mechanics.mu_ff} Pa")
        self.vprint(f"solver_parameters = {self.solver_parameters}")
        self.vprint(f"ffc_options = {self.ffc_options}")

        # Assemble variational problem with current state
        Pi = self.mechanics.get_energy(B=self.B, T=self.T)
        dPi, ddPi = get_variational_stuff(self.kinematics, Pi)

        # Solver loop (solves the forward problem in load_steps)
        for i in range(self.load_steps):
            sys.stdout.flush()  

            if hasattr(self.geo, "scalar"):
                # updates the boundary conditions surface displacements
                self.geo.scalar = (i+1)/self.load_steps
                self.geo.update_bcs()

            # Be careful of load stepping and adjoint annotation!
            stop_tape = False
            if i != self.load_steps-1:
                stop_tape = True

            if stop_tape:
                pa.pause_annotation()

            # Solve variational problem
            nvp = NonlinearVariationalProblem(
                dPi,
                self.kinematics.u,
                self.geo.bcs,
                J=ddPi,
                form_compiler_parameters=self.ffc_options
            )
            nvs = NonlinearVariationalSolver(nvp)
            nvs.parameters.update(self.solver_parameters)
            nvs.solve()

            if stop_tape:
                pa.continue_annotation()

