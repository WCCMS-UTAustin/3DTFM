"""The inverse model.

# RECOMMENDED USAGE
    nohup inverse ... | tee -a out.txt &

This will allow it to continue in the background even if the terminal
session dies but the user session remains. Also will log error messages
that may be hard-coded to stdout by FEniCS.

# Side-Effects

Will interact with local file system when run as a command. This
includes:
* Logs what it is doing in file specified by name `LOG_FILE`
* Uses a directory `results_dir` as a super-directory for all inverse
model results. This directory should be clear of subdirectories except
for those created by **specifically this program**.
* Creates a .csv file `inverse.csv` under `results_dir` to store a chart
of all options used and some result info like runtime and quantites
for L-curve analysis.
* Creates a subdirectory under `results_dir` according to the options
provided.
* In that subdirectory, stores many .xdmf files with results and
other information.
* If a subdirectory for a collection of settings already exists, will
not proceed forward and quietly skip.

# API
"""
from gel import *
import numpy as np
import os
from sanity_utils import *
import dolfin as df
import moola


INTERM_SAVE_PERIOD = 10
"""Write current guess to filesytem every (this many) L-BFGS iterations"""


#############################################################################
### FEniCS settings
#############################################################################
parameters['linear_algebra_backend'] = 'PETSc'
parameters['form_compiler']['representation'] = 'uflacs'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['quadrature_degree'] = 3


LOG_FILE = "global_log.txt"


def _expand_obj_info(exp_info):
    obj_info = exp_info[5]
    new_exp_info = (
        *exp_info[:5],
        obj_info.gamma_num,
        obj_info.objective_type,
        obj_info.regularization_type,
        obj_info.objective_domain,
        obj_info.regularization_domain,
        obj_info.detectable_u_cutoff,
        obj_info.safe_u_weight_filename(),
        obj_info.apply_u_weight_to_reg,
        *exp_info[6:]
    )
    return new_exp_info


INVERSE_EXPERIMENT_COLS = [
    "Results Dir",
    "Cell Name",
    "target_file",
    "Formulation",
    "k",
    "gamma",
    "objective_type",
    "regularization_type",
    "objective_domain",
    "regularization_domain",
    "detectable_u_cutoff",
    "u_weight_filename",
    "apply_u_weight_to_reg",
    "lower_bound",
    "upper_bound",
    "tol",
    "mod_repr_init",
    "debug_mode",
    "max_iter",
    "load_steps",
    "restrict_ctl_dofs_to_veh",
    "preconditioner",
    "optimizer_backend",
    "bci",
    "bco",
    "mu_ff",
    "u_init",
    "outcome",
    "pure_obj",
    "reg",
    "time"
]
"""Names of columns in `inverse.csv`"""


def main(args):
    """Sets up logging to relevant files and starts the inverse model.

    * `args`: `argparse.Namespace` with parsed command arguments

    Side-effects: see intro to `gel.scripts.inverse`; does all the file
    handling except the experiment's subdirectory

    Computation: calls `gel_inverse`, which handles the experiment's
    subdirectory.
    """
    logger = get_global_logger(LOG_FILE)

    # Results
    result_csv_path = os.path.join(args.r, "inverse.csv")
    csv = ResultsCSV(result_csv_path, INVERSE_EXPERIMENT_COLS)

    logger.info(f"Detected number of processes: {MPI.comm_world.Get_size()}")

    objective_info = ObjectiveInfo(args)

    exp_info = (
        args.r,
        args.c,
        args.t,
        args.f,
        args.k,
        objective_info,
        *args.b,
        args.a,
        args.i,
        args.debug,
        args.m,
        args.l,
        not args.no_restrict_ctl_dofs_to_veh,
        args.p,
        args.opt_backend,
        args.bci,
        args.bco,
        args.mu,
        args.u_init
    )

    logger.info(f"Beginning experiment with arguments {exp_info}.")

    # Be careful of it already existing
    try:
        try:
            total_time, pure_obj, reg = gel_inverse(*exp_info)

            # Update DataFrame
            exp_info = _expand_obj_info(exp_info)
            csv.add_row(list(exp_info) + [
                "Converged",
                pure_obj,
                reg,
                float(total_time)
            ])
        except RuntimeError as e:
            logger.info(
                f"FEniCS encountered an error. Recording this outcome"
            )
            logger.info(e)

            # Update DataFrame
            exp_info = _expand_obj_info(exp_info)
            csv.add_row(list(exp_info) + ["Crash", "", "", ""])

        # Save into file
        csv.save()
    except FileExistsError:
        logger.info(
            "Found that experiment directory already exists. Skipping..."
        )
    except UnsupportedGeometryError:
        logger.info(
            "This geometry type is not supported by the inverse model at "
            "this time. Skipping..."
        )


class UnsupportedGeometryError(Exception):
    """Raised if `cell_data_dir` doesn't have a supported type."""
    pass


def gel_inverse(
        results_dir,
        cell_data_dir,
        target_file,
        formulation,
        k,
        objective_info,
        lower_bound,
        upper_bound,
        tol,
        mod_repr_init,
        debug=False,
        maxiter=10000,
        load_steps=2,
        restrict_ctl_dofs_to_veh=False,
        preconditioner="hypre_amg",
        optimizer_backend="scipy",
        bci=None,
        bco=None,
        mu=108.0,
        u_init=None
    ):
    r"""Runs the inverse model, writes subdirectory and logs progress.

    * `results_dir`: str path to directory to put all results
    * `cell_data_dir`: str path to directory with geometry information
    to use, see the constructor to `gel.geometry.Geometry` for required
    files
    * `target_file`: str path to full-shape .xdmf file with
    displacements in "u"
    * `formulation`: str a valid material model formulation, see
    `gel.mechanics` for options
    * `k`: float $\frac{D_1}{c_1}$ ratio
    * `objective_info`: `gel.objective.ObjectiveInfo` with information
    on the objective to be minimized
    * `lower_bound`: float; either the lower bound for the "beta_tilde"
    formulation described in `gel.mechanics` or the lower bound for
    L-BFGS-B if the "scipy" backend if selected
    * `upper_bound`: float; either the upper bound for the "beta_tilde"
    formulation described in `gel.mechanics` or the upper bound for
    L-BFGS-B if the "scipy" backend if selected
    * `tol`: float tolerance for inverse model convergence
    * `mod_repr_init`: str a valid modulus representation for the
    initial guess, see `gel.mechanics` for options
    * `debug`: bool, if True runs a Taylor test and outputs a tree of
    the traced components of the simulation in file "graph.dot" in the
    experiment's subdirectory
    * `maxiter`: int maximum number of L-BFGS iterations
    * `restrict_ctl_dofs_to_veh`: bool, it True will only control DoFs
    in the volume within the event horizon with cutoff specified in
    `objective_info`
    * `preconditioner`: str preconditioner to use, see
    `gel.gel.ForwardSimulation` for more details
    * `optimizer_backend`: str moola for L-BFGS and that library,
    otherwise scipy for L-BFGS-B and that library
    * `bci`: str path to .vtk file with inner BC info, see
    `gel.geometry.Geometry` for details
    * `bco`: str path to .vtk file with outer BC info, see
    `gel.geometry.Geometry` for details
    * `mu`: float far-field shear modulus from rheometry
    * `u_init`: str path to full-shape .xdmf file with initial
    displacements

    Side-effects: writes many files in new subdirectory to
    `results_dir`, see intro to `gel.scripts.inverse`

    Returns:
    * float time it took to run
    * float objective functional value $O$ at end (see `gel.objective`)
    * float regularization functional value $R$ at end (see
    `gel.objective`)
    """
    # Input validation
    validate_formulation(formulation)
    validate_mod_repr_field(mod_repr_init)

    # Get cell name
    cell_name = read_cell_name(cell_data_dir)
    # Get geometry type
    geo_type = read_geo_type(cell_data_dir)
    if geo_type != "real_cell_gel":
        raise UnsupportedGeometryError(f"{geo_type} currently not supported")

    # Output directory handling
    output_dir = prepare_experiment_dir(
        results_dir,
        cell_name,
        formulation,
        restrict_ctl_dofs_to_veh,
        mod_repr_init,
        k,
        objective_info.gamma_num,
        objective_info.objective_type,
        objective_info.regularization_type,
        objective_info.objective_domain,
        objective_info.regularization_domain,
        objective_info.detectable_u_cutoff,
        objective_info.apply_u_weight_to_reg,
        objective_info.safe_u_weight_filename(),
        lower_bound,
        upper_bound,
        tol,
        preconditioner,
        bci,
        bco,
        mu
    )

    # Setup logging
    logger, logger_destructor = create_experiment_logger(output_dir)

    #
    # Initial forward run
    #
    logger.info(f"Beginning inverse model in {formulation} formulation.")
    logger.info(f"Using cell data from {cell_data_dir}, '{cell_name}'")
    logger.info(f"Using mod_repr init strat {mod_repr_init}")
    logger.info(
        f"Restricting mod_ctl to VEH? {restrict_ctl_dofs_to_veh}"
    )
    logger.info(f"Uses geometry type {geo_type}")
    logger.info(f"Target file: {target_file}")
    logger.info(f"Using D1/C1: {k}")
    logger.info(f"Bounds: [{lower_bound}, {upper_bound}]")
    logger.info(f"Using tolerance: {tol}")
    logger.info(f"Using maximum iterations: {maxiter}")
    logger.info(f"Using load steps: {load_steps}")
    logger.info(f"Optimizer backend: {optimizer_backend}")
    logger.info(f"Preconditioner: {preconditioner}")
    logger.info(f"Override inner BC: {bci}")
    logger.info(f"Override outer BC: {bco}")
    logger.info(f"Far field modulus: {mu}")
    logger.info(f"Initial displacements: {u_init}")
    objective_info.log_info(logger)

    # RESET TAPE
    set_working_tape(Tape())

    # Timer start
    timer = ExperimentTimer()
    timer.start()

    logger.info(f"Running forward model with initial guess...")
    
    # Deal with telling geo about detectable regions, BCs
    addn_args = {"bci_data" : bci, "bco_data" : bco}
    if objective_info.detectable_u_cutoff is not None:
        addn_args["u_magnitude_subdomains_file"] = target_file
        addn_args["detectable_u_cutoff"] = objective_info.detectable_u_cutoff

    formulation_kwargs = dict()
    if formulation == "beta_tilde":
        formulation_kwargs["beta_min"] = lower_bound
        formulation_kwargs["beta_max"] = upper_bound

    sim = ForwardSimulation(
        geo_type,
        k,
        formulation=formulation,
        mod_repr_init=mod_repr_init,
        load_steps=load_steps,
        vprint=logger.info,
        mu_ff=mu,
        data_directory=cell_data_dir,
        restrict_ctl_dofs_to_veh=restrict_ctl_dofs_to_veh,
        pc=preconditioner,
        u_init=u_init,
        formulation_kwargs=formulation_kwargs,
        **addn_args
    )

    sim.run_forward()
    kinematics, mod_repr = sim.kinematics, sim.mod_repr # Pointers

    #####################
    # Must deal with intercepted block
    tape = get_working_tape()
    solve_block = tape.get_blocks()[-1]

    for block_var in solve_block.get_dependencies():
        if block_var.output == kinematics.u:
            block_var.save_output()
            solve_block.block_var_to_save = block_var
            solve_block.prev_soln = Function(sim.geo.V)
    #####################

    # Save
    geo = sim.geo
    logger.info(
        f"Finished initial guess forward run. Saving results to {output_dir}"
    )
    kinematics_sim = Kinematics(geo, kinematics.u)
    output_results_paraview(
        output_dir,
        kinematics_sim,
        mod_repr,
        name="init_guess"
    )

    # Read in target displacement
    logger.info(f"Reading target displacements from {target_file}")
    kinematics_target = kinematics_from_file(geo, target_file)

    # Debug
    if debug and rank == 0:
        tape = get_working_tape()
        tape.visualise(os.path.join(output_dir, "graph.dot"))
    
    #
    # Objective function construction
    #
    logger.info(f"Assembling inverse objective.")

    pre_assembly, pure_obj_form, reg_form = objective_info.get_objective_forms(
        geo,
        kinematics_target,
        kinematics_sim,
        mod_repr,
        logger
    )

    # Finalize
    obj = assemble(pre_assembly)

    #
    # Inverse model
    #
    # Create callback
    i = 0
    def callback(this_mod_repr):
        nonlocal i, geo, kinematics, output_dir, mod_repr, solve_block
        nonlocal optimizer_backend

        if i % INTERM_SAVE_PERIOD == 0:
            logger.info(f"Saving results for iteration {i}")

            df.Function.assign(
                kinematics.u,
                solve_block.block_var_to_save.output
            )
            kinematics_mid = Kinematics(geo, kinematics.u)

            # Assign values to mod_repr
            if optimizer_backend == "scipy":
                dm = mod_repr.function_space().dofmap()
                local_range = dm.ownership_range()
                mod_repr.vector().set_local(
                    this_mod_repr[local_range[0]:local_range[1]]
                )
                mod_repr.vector().apply("")
            elif optimizer_backend == "moola":
                mod_repr.assign(this_mod_repr.data)

            # Save
            output_results_paraview(
                output_dir,
                kinematics_mid,
                mod_repr,
                name="most_recent"
            )

        i += 1

    # Adjoint stuff, marking tape
    control = Control(sim.ctl)
    obj_hat = ReducedFunctional(obj, control)

    # Check the derivative
    if debug:
        debug_deriv(
            obj_hat,
            geo,
            mod_repr,
            logger
        )

    # Minimize
    logger.info(f"Calling minimization routine...")
    #if MPI.comm_world.rank == 0:
        #breakpoint()
    if optimizer_backend == "scipy":
        mod_repr_opt = minimize(
            obj_hat,
            method = "L-BFGS-B",
            tol=tol,
            bounds = (lower_bound, upper_bound),
            options = {"maxiter":maxiter,"gtol":tol,"disp": True},
            callback = callback
        )
    elif optimizer_backend == "moola":
        problem = MoolaOptimizationProblem(obj_hat)
        mod_repr_moola = moola.DolfinPrimalVector(sim.ctl)
        solver = moola.BFGS(
            problem,
            mod_repr_moola,
            options={
                "jtol":tol,
                "gtol":100*tol,
                "Hinit":"default",
                "maxiter":maxiter,
                "mem_lim":10,
                "display":2
            },
            hooks={"after_iteration":callback}
        )
        sol = solver.solve()
        mod_repr_opt = sol["control"].data
    else:
        raise ValueError(f"Unknown optimizer backend {optimizer_backend}")

    logger.info(f"Finished minimization")

    # Assignments after optimal parameters are found
    mod_repr.assign(mod_repr_opt)

    #
    # Post-pro: run forward of optimal params and save
    #
    logger.info(f"Running forward model with optimal mod_repr field...")

    # Run forward model with modified mod_repr
    sim.run_forward()
    logger.info(f"Finished. Saving...")

    # Compute QoIs
    kinematics_post = Kinematics(geo, kinematics.u)

    # Save
    output_results_paraview(
        output_dir,
        kinematics_post,
        mod_repr,
        name="solved"
    )

    #
    # Compute L-curve relevant quantities
    #
    _, pure_obj_form, reg_form = objective_info.get_objective_forms(
        geo,
        kinematics_target,
        kinematics_sim,
        mod_repr,
        None
    )

    pure_obj = assemble(pure_obj_form)
    if reg_form is not None:
        reg = assemble(reg_form)
    else:
        reg = 0.0

    #
    # End
    #
    # Timer end
    total_time = timer.end()
    logger.info(f"Inverse modeling procedure complete")
    logger.info(f"Took {total_time} seconds to complete")

    # Tear down logging
    logger_destructor()

    return total_time, pure_obj, reg


def inverse():
    """The function invoked by the command. Parses arguments and passes
    to `main`.
    """
    parser = get_common_parser(
        description="One stage of the inverse model to determine modulus from "
        "displacements"
    )

    add_objective_arguments(parser)

    parser.add_argument(
        "-r",
        type=str,
        metavar="RESULTS_DIR",
        default="results",
        help="superdirectory in which all results for given target u are stored"
    )
    parser.add_argument(
        "-t",
        type=str,
        metavar="TARGET_U_XDMF",
        default=None,
        help="filename with full-shape representation of target displacements"
    )
    parser.add_argument(
        "-a",
        type=float,
        metavar="TOL",
        default=1e-8,
        help="tolerance for inverse model"
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="produces output of pyadjoint tree and performs Taylor test"
    )
    parser.add_argument(
        "-m",
        type=int,
        metavar="MAX_ITER",
        default=250,
        help="maximum L-BFGS iterations"
    )
    parser.add_argument(
        "-i",
        type=str,
        metavar="INIT_STRAT",
        default="zero",
        help="what to initialize the modulus representation, i.e. beta, to"
    )
    parser.add_argument(
        "--no-restrict-ctl-dofs-to-veh",
        action="store_true",
        help="solve for the modulus everywhere in the domain instead"
    )
    parser.add_argument(
        "--opt-backend",
        type=str,
        metavar="BACKEND",
        choices=["scipy", "moola"],
        default="moola",
        help="scipy: bounded, vector repr.; moola: unbounded, L2 repr."
    )
    parser.add_argument(
        "--mu",
        type=float,
        metavar="MU_FF",
        default=108.0,
        help="far-field shear modulus from rheometry"
    )

    args = parser.parse_args()

    if args.t is None:
        args.t = os.path.join(args.c, "u_experiment.xdmf")

    main(args)


if __name__=="__main__":
    inverse()

