from .header import *
import os
import argparse
from functools import cmp_to_key


# Helper to return helper function for relative path
ghrp = lambda data_directory : (
    lambda fname : os.path.join(data_directory, fname)
)


def project_tensor(geo, tensor):
    """Given a geometry and a tensor, produce an output-ready version."""
    projected_tensor = project(
        tensor,
        geo.VC,
        solver_type="cg",
        preconditioner_type="amg"
    )
    return projected_tensor


def get_common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        metavar="CELL_DATA_DIR",
        default="cell_data"
    )
    parser.add_argument(
        "-f",
        metavar="FORMULATION",
        default="beta"
    )
    parser.add_argument(
        "-k",
        type=float,
        metavar="MODULUS_RATIO",
        default=1
    )
    parser.add_argument(
        "-b",
        nargs=2,
        type=float,
        metavar="BOUNDS",
        default=[-2.0, 20.0]
    )
    parser.add_argument(
        "-l",
        type=int,
        metavar="LOAD_STEPS",
        default=1
    )
    parser.add_argument(
        "-p",
        type=str,
        metavar="PRECONDITIONER",
        default="hypre_amg"
    )
    parser.add_argument(
        "--bci",
        type=str,
        metavar="CELL_SURF_MESH",
        default=None
    )
    parser.add_argument(
        "--bco",
        type=str,
        metavar="OUTER_SURF_MESH",
        default=None
    )

    return parser


@cmp_to_key
def lexico_sort(coord0, coord1):
    for i in range(len(coord0)):
        cmp = coord0[i] - coord1[i]
        if cmp != 0:
            return cmp
    return 0


def nodal_values_to_fenics_fcn(
        geo,
        nodal_values_filename,
        field_name="mod_repr"
    ):
    # Read in nodal DOFs
    with meshio.xdmf.TimeSeriesReader(nodal_values_filename) as reader:
        points, cells = reader.read_points_cells()
        for k in range(reader.num_steps):
            _, point_data, cell_data = reader.read_data(k)
            break

    # Corresponding function space
    coordinates = geo.V0.tabulate_dof_coordinates()

    # Helper functions
    sorted_to_meshio = sorted(
        range(len(points)), key = lambda i : lexico_sort(points[i])
    )
    sorted_to_fenics = sorted(
        range(len(coordinates)), key = lambda i : lexico_sort(coordinates[i])
    )

    # Into FEniCS order
    dofs_fenics_order = np.zeros(len(coordinates))
    dofs_fenics_order[sorted_to_fenics] = \
            point_data[field_name].flatten()[sorted_to_meshio]

    # Make function
    fcn = Function(geo.V0)
    fcn.vector().set_local(dofs_fenics_order)
    fcn.vector().apply("")
    fcn.set_allow_extrapolation(True)

    return fcn


def save_fenics_fcn_nodally(fcn, filename, fcn_name, long_fcn_name=None):
    output_file = XDMFFile(filename)
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True

    if long_fcn_name is None:
        long_fcn_name = fcn_name

    fcn.rename(fcn_name, long_fcn_name)

    output_file.write(fcn, 0)


def save_shape(fcn, filename, fcn_name):
    shape_file = XDMFFile(filename)
    shape_file.parameters["flush_output"] = True
    shape_file.parameters["functions_share_mesh"] = True

    shape_file.write_checkpoint(
        fcn,
        fcn_name,
        0,
        XDMFFile.Encoding.HDF5,
        False
    )

