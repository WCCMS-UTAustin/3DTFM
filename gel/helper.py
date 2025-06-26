"""A variety of helper functions.

Notably, includes functions for generalized reading and writing from
mesh files. This is necessary due to the variety of interfaces that
may be easiest to work with, given a task.

A `meshio.Mesh` object is useful for extracting point/cell data
according to the event horizon mesh output from `gel.scripts.get_veh`.
`nodal_values_to_meshio` achieves this task.

"Full-shape" .xdmf files are created from write_checkpoint methods, and
are suitable for reading and writing directly in and out of FEniCS
without additional machinery necessary. Associated functions include
`save_shape` and `load_shape`.

"Nodal" .xdmf files are best for post-processing with Paraview as they
have a compatible embedding of connectivity information and are smaller
files. However, some floating-point precision loss has been observed.
Associated functions for working with function DoFs in this manner
include `save_fenics_fcn_nodally` and `nodal_values_to_fenics_fcn`.

# API
"""
from .header import *
import os
import argparse
from functools import cmp_to_key


ghrp = lambda data_directory : (
    lambda fname : os.path.join(data_directory, fname)
)
"""Given directory `data_directory`, return mapping filename to path."""


def get_common_parser(*args, **kwargs):
    """Returns `argparse.ArgumentParser` equipped with common args.

    * `args`: positional arguments to `argparse.ArgumentParser`
    * `kwargs`: keyword arguments to `argparse.ArgumentParser`

    Included arguments:
    ```
    -c CELL_DATA_DIR
    -f FORMULATION
    -k MODULUS_RATIO
    -b BOUNDS
    -l LOAD_STEPS
    -p PRECONDITIONER
    --bci CELL_SURF_MESH
    --bco OUTER_SURF_MESH
    ```
    """
    if "formatter_class" not in kwargs:
        kwargs["formatter_class"] = argparse.ArgumentDefaultsHelpFormatter

    parser = argparse.ArgumentParser(*args, **kwargs)

    parser.add_argument(
        "-c",
        metavar="CELL_DATA_DIR",
        default="cell_data",
        help="directory containing gel geometry"
    )
    parser.add_argument(
        "-f",
        metavar="FORMULATION",
        default="beta",
        help="form of strain energy density"
    )
    parser.add_argument(
        "-k",
        type=float,
        metavar="MODULUS_RATIO",
        default=1,
        help="D1/C1 ratio"
    )
    parser.add_argument(
        "-b",
        nargs=2,
        type=float,
        metavar="BOUNDS",
        default=[-2.0, 20.0],
        help="bounds for inverse model, or built-in for some formulations"
    )
    parser.add_argument(
        "-l",
        type=int,
        metavar="LOAD_STEPS",
        default=1,
        help="forward model BC load step count"
    )
    parser.add_argument(
        "-p",
        type=str,
        metavar="PRECONDITIONER",
        default="hypre_amg",
        help="preconditioner for forward model Newton-Raphson linear solves"
    )
    parser.add_argument(
        "--bci",
        type=str,
        metavar="CELL_SURF_MESH",
        default=None,
        help="filename of meshio-compatible mesh with cell surface nodes and u"
    )
    parser.add_argument(
        "--bco",
        type=str,
        metavar="OUTER_SURF_MESH",
        default=None,
        help="filename of meshio-compatible mesh with outer nodes and u"
    )

    return parser


@cmp_to_key
def lexico_sort(coord0, coord1):
    for i in range(len(coord0)):
        cmp = coord0[i] - coord1[i]
        if cmp != 0:
            return cmp
    return 0


def save_fenics_fcn_nodally(fcn, filename, fcn_name, long_fcn_name=None):
    """Writes a nodal .xdmf file with given function.

    * `fcn`: FEniCS FE function to write
    * `filename`: str filename ending in ".xdmf"
    * `fcn_name`: str short name of the function
    * `long_fcn_name`: str long name of the function (default same as
    `fcn_name`)

    Side-effects: writes files `filename` with .xdmf and .h5 endings
    """
    output_file = XDMFFile(filename)
    output_file.parameters["flush_output"] = True
    output_file.parameters["functions_share_mesh"] = True

    if long_fcn_name is None:
        long_fcn_name = fcn_name

    fcn.rename(fcn_name, long_fcn_name)

    output_file.write(fcn, 0)


def nodal_values_to_meshio(filename):
    """Reads an .xdmf in "nodal" form to return a `meshio.Mesh` with
    all point and cell data.

    Note that the `meshio.xdmf.TimeSeriesReader` is involved, but only
    time 0 is read in accordance with this libraries output style.

    * `filename`: str filename ending in ".xdmf", must be in "nodal"
    form
    """
    # Read in nodal DOFs
    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        _, point_data, cell_data = reader.read_data(0)

    return meshio.Mesh(
        points,
        cells,
        point_data=point_data,
        cell_data=cell_data
    )


def nodal_values_to_fenics_fcn(
        geo,
        nodal_values_filename,
        field_name="mod_repr"
    ):
    """Reads a nodal .xdmf file to create a new **scalar** function.

    * `geo`: `gel.geometry.Geometry` with matching hydrogel mesh
    * `nodal_values_filename`: str filename ending in ".xdmf", must be
    a "full_shape" form
    * `field_name`: str the name of the scalar field in the file

    Returns: FEniCS FE scalar function in 1st order Lagrange space
    """
    # Read in nodal DOFs
    mesh = nodal_values_to_meshio(nodal_values_filename)
    points, cells, point_data, cell_data = \
        mesh.points, mesh.cells, mesh.point_data, mesh.cell_data

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


def save_shape(fcn, filename, fcn_name):
    """Writes full-shape .xdmf file with given function.

    * `fcn`: FEniCS FE function to write
    * `filename`: str filename ending in ".xdmf"
    * `fcn_name`: str short name of the function

    Side-effects: writes files `filename` with .xdmf and .h5 endings
    """
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


def load_shape(fcn_space, filename, fcn_name):
    """Reads a full-shape .xdmf file to create a new FEniCS FE function.

    * `fcn_space`: FEniCS FunctionSpace that the function was originally
    created in
    * `filename`: str filename ending in ".xdmf", must be in
    "full_shape" form
    * `fcn_name`: str the name of the function in the file

    Returns: FEniCS FE function
    """
    fcn = Function(fcn_space)
    shape_file = XDMFFile(filename)
    shape_file.read_checkpoint(fcn, fcn_name, 0)

    return fcn

