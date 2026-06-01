"""Creates mesh with indications of which elements are in event horizon"""
import argparse
import os
os.environ["SUPPRESS_ADJOINT"] = "1"
from gel import *


def get_veh(cell_data_dir, u_exp_file, directory, cutoff=0.38,
            bci=None, bco=None):
    """Creates .pvd (and support files) with event horizon indication.

    * `cell_data_dir`: str path to directory with files needed by
    `gel.geometry.Geometry`
    * `u_exp_file`: str path to full-shape .xdmf file with displacements
    "u" to use in determining event horizon
    * `directory`: str path to directory within which to create new
    files
    * `cutoff`: float displacement magnitude in microns to use as event
    horizon cutoff value
    * `bci`: str or None. Override path to the inner cell-surface BC .vtk;
    when None, `gel.geometry.Geometry` defaults to `cell_data_dir/bci.vtk`
    * `bco`: str or None. Override path to the outer boundary BC .vtk

    Side-effects: creates "u_regions_{cutoff}.pvd" and support files
    with element-wise data having event horizon tags from
    `gel.geometry.Geometry`
    """
    geo = Geometry(
        cell_data_dir,
        u_magnitude_subdomains_file=u_exp_file,
        detectable_u_cutoff=cutoff,
        bci_data=bci,
        bco_data=bco
    )
    filename = os.path.join(directory, f"u_regions_{cutoff}.pvd")
    File(filename) << geo.dx.subdomain_data()


def get_veh_main():
    """The function invoked by the command. Parses arguments and passes
    to `get_veh`.
    """
    parser = argparse.ArgumentParser(
        description="Obtain a mesh indicating which cells are counted"
        " as being within the event horizon for visualization."
    )
    parser.add_argument(
        "-c",
        type=str,
        metavar="CELL_DATA",
        help="directory containing gel geometry"
    )
    parser.add_argument(
        "-i",
        type=str,
        metavar="INPUT_FULL_SHAPE",
        help="full-shape 1st order Lagrange .xdmf file with "
        "displacements 'u'"
    )
    parser.add_argument(
        "-o",
        type=str,
        metavar="OUTPUT_DIR",
        help="result will be saved as "
        "OUTPUT_DIR/u_regions_{THRESHOLD}.pvd and supporting files"
    )
    parser.add_argument(
        "-t",
        type=float,
        metavar="THRESHOLD",
        default=0.38,
        help="threshold by which VEH is determined in microns"
    )
    parser.add_argument(
        "--bci",
        type=str,
        metavar="CELL_SURF_MESH",
        default=None,
        help="override path to inner cell-surface BC .vtk (default: -c dir's "
        "bci.vtk)"
    )
    parser.add_argument(
        "--bco",
        type=str,
        metavar="OUTER_SURF_MESH",
        default=None,
        help="override path to outer boundary BC .vtk (default: -c dir's "
        "bco.vtk)"
    )
    args = parser.parse_args()

    get_veh(args.c, args.i, args.o, args.t, args.bci, args.bco)


if __name__=="__main__":
    get_veh_main()

