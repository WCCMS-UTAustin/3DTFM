#!/usr/bin/env python3
"""
forward_mesh_output.py -f FORMULATION ... -i I1_VERSION ... -a ALPHA_FIELD ...
                    -k K ... -r RESULTS_DIR -c CELL_DATA ...

Given various formulations, i1 variations, mod_repr field types, values of k, etc
runs a forward model for each combination (Cartesian product style)
and outputs mesh results in a results directory.

For cell data, if given the name of a directory, will read in geometry.

Also logs what it is doing in log.txt
"""
from gel import *
import os
from sanity_utils import *


LOG_FILE = "log.txt"


FORWARD_EXP_COLS = [
    "Results Dir",
    "Cell Name",
    "Formulation",
    "Modulus Repr",
    "Modulus Ratio",
    "pc",
    "Load Steps",
    "Absolute Tolerance",
    "Relative Tolerance",
    "Traction",
    "lower_bound",
    "upper_bound",
    "Inner Mesh",
    "Outer Mesh",
    "time"
]


def run_experiments(args):
    logger = get_global_logger(LOG_FILE)

    # Results
    result_csv_path = os.path.join(args.r, "table.csv")
    csv = ResultsCSV(result_csv_path, FORWARD_EXP_COLS)

    logger.info(f"Detected number of processes: {MPI.comm_world.Get_size()}")

    exp_info = (
        args.r,
        args.c,
        args.f,
        args.a,
        args.k,
        args.p,
        args.l,
        args.tola,
        args.tolr,
        args.t,
        args.b[0],
        args.b[1],
        args.bci,
        args.bco
    )

    logger.info(f"Beginning experiment with arguments {exp_info}.")

    # Be careful of it already existing
    try:
        total_time = output_single_forward_result(*exp_info)

        # Update DataFrame
        csv.add_row(list(exp_info)+[float(total_time)])

        # Save into file
        csv.save()
    except FileExistsError:
        logger.info("Found that directory already exists. Skipping...")


def output_single_forward_result(
        results_dir,
        cell_data_dir,
        formulation,
        mod_repr,
        k,
        pc,
        load_steps,
        tola,
        tolr,
        traction,
        lower_bound,
        upper_bound,
        bci,
        bco
    ):
    """
    Runs a single experiment with:
    results_dir : directory with all results
    cell_data_dir : the cell/gel information to use
    formulation : a valid formulation
    mod_repr : a valid mod_repr field variant
    k : float - target ratio
    pc : str - preconditioner
    pc : str - traction

    Returns:
    time_taken
    """
    # Input validation
    validate_formulation(formulation)
    validate_mod_repr_field(mod_repr)

    # Get cell name
    cell_name = read_cell_name(cell_data_dir)

    # Output directory handling
    output_dir = prepare_experiment_dir(
        results_dir,
        cell_name,
        formulation,
        mod_repr,
        k,
        pc,
        load_steps,
        traction,
        lower_bound,
        upper_bound,
        bci,
        bco
    )

    # Setup logging
    logger, logger_destructor = create_experiment_logger(output_dir)

    #
    # Experiment
    #
    logger.info(f"Beginning experiment in {formulation} formulation.")
    logger.info(f"Using cell data from {cell_data_dir}, '{cell_name}'")
    logger.info(f"Using mod_repr field: {mod_repr}")
    logger.info(f"Using D1/C1: {k}")
    logger.info(f"Using preconditioner: {pc}")
    logger.info(f"Abs tolerance: {tola}")
    logger.info(f"Rel tolerance: {tolr}")
    logger.info(f"Traction filename: {traction}")
    logger.info(f"Using load steps: {load_steps}")
    logger.info(f"Override inner BC: {bci}")
    logger.info(f"Override outer BC: {bco}")

    # Timer start
    timer = ExperimentTimer()
    timer.start()

    # run
    addn_args = {"bci_data" : bci, "bco_data" : bco}
    formulation_kwargs = dict()
    if formulation == "beta_tilde":
        formulation_kwargs["beta_min"] = lower_bound
        formulation_kwargs["beta_max"] = upper_bound

    sim = ForwardSimulation(
        "real_cell_gel",
        k,
        material_model=formulation,
        mod_repr_init=mod_repr,
        vprint=logger.info,
        data_directory=cell_data_dir,
        restrict_ctl_dofs_to_veh=False,
        pc=pc,
        load_steps=load_steps,
        tola=tola,
        tolr=tolr,
        traction=traction,
        formulation_kwargs=formulation_kwargs,
        **addn_args
    )
    sim.run_forward()
    kinematics, mod_repr = sim.kinematics, sim.mod_repr

    # Timer end
    total_time = timer.end()
    logger.info(f"Took {total_time} seconds to complete")

    # Save
    geo = kinematics.geo
    logger.info(f"Saving results to {output_dir}")
    kinematics_sim = Kinematics(geo, kinematics.u)
    output_results_paraview(output_dir, kinematics_sim, mod_repr)

    # Tear down logging
    logger_destructor()

    return total_time


def forward():
    parser = get_common_parser(
        description="Run the forward model for ground-truth"
        " simulations of test problems"
    )

    parser.add_argument(
        "-r",
        metavar="RESULTS_DIR",
        default="forward",
        help="superdirectory in which all solutions are saved"
    )
    parser.add_argument(
        "--tola",
        type=float,
        metavar="ABS_TOL",
        default=1e-10,
        help="Newton-Rasphon absolute residual tolerance"
    )
    parser.add_argument(
        "--tolr",
        type=float,
        metavar="REL_TOL",
        default=1e-9,
        help="Newton-Rasphon relative residual tolerance"
    )
    parser.add_argument(
        "-a",
        metavar="MOD_REPR",
        default="zero",
        help="the modulus field to use"
    )
    parser.add_argument(
        "-t",
        type=str,
        metavar="TRACTION",
        default=None,
        help="full-shape .xdmf file with 1st order Lagrange reference tractions"
        " 'T' over the whole gel domain, note that values outside cell surface "
        "are ignored"
    )
    args = parser.parse_args()

    run_experiments(args)


if __name__ == "__main__":
    forward()

