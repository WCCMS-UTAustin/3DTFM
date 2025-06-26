"""The forward model.

For all arguments, run `forward --help`

Will interact with local file system when run as a command. This
includes:
* Logs what it is doing in file specified by name `LOG_FILE`
* Uses a directory `results_dir` as a super-directory for all forward
model results. This directory should be clear of subdirectories except
for those created by **specifically this program**.
* Creates a .csv file `table.csv` under `results_dir` to store a chart
of all options used and some result info like runtime.
* Creates a subdirectory under `results_dir` according to the options
provided.
* In that subdirectory, stores many .xdmf files.
* If a subdirectory for a collection of settings already exists, will
not proceed forward and quietly skip.
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
"""Names of columns in `table.csv`"""


def run_experiments(args):
    """Sets up logging to relevant files and starts the forward model.

    * `args`: `argparse.Namespace` with parsed command arguments

    Side-effects: see intro to `gel.scripts.forward`; does all the file
    handling except the experiment's subdirectory

    Computation: calls `output_single_forward_result`, which handles
    the experiment's subdirectory.
    """
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
    r"""Runs a single forward model experiment, writes subdirectory and
    logs progress.

    * `results_dir`: str path to directory to put all results
    * `cell_data_dir`: str path to directory with geometry information
    to use, see the constructor to `gel.geometry.Geometry` for required
    files
    * `formulation`: str a valid material model formulation, see
    `gel.mechanics` for options
    * `mod_repr`: str a valid modulus representation, see
    `gel.mechanics` for options
    * `k`: float $\frac{D_1}{c_1}$ ratio
    * `pc`: str preconditioner to use, see `gel.gel.ForwardSimulation`
    for more details
    * `tola`: float the absolute tolerance for Newton-Raphson
    * `tolr`: float the relative tolerance for Newton-Raphson
    * `traction`: str filename to traction if using, see
    `gel.gel.ForwardSimulation` for details
    * `lower_bound`: float the lower bound for the "beta_tilde"
    formulation described in `gel.mechanics`
    * `upper_bound`: float the upper bound for the "beta_tilde"
    formulation described in `gel.mechanics`
    * `bci`: str path to .vtk file with inner BC info, see
    `gel.geometry.Geometry` for details
    * `bco`: str path to .vtk file with outer BC info, see
    `gel.geometry.Geometry` for details

    Side-effects: writes many files in new subdirectory to
    `results_dir`, see intro to `gel.scripts.forward`

    Returns: float time it took to run
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
        formulation=formulation,
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
    """The function invoked by the command. Parses arguments and passes
    to `run_experiments`.
    """
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

